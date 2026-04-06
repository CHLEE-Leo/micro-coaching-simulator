"""
models/information_seeker.py
───────────────
LLM 기반 Information Seeker 모델.

역할
  - 영양 목표와 식사 유형을 입력받아 User 로부터 식사 정보를 이끌어내기 위한
    구조화된 질문 템플릿(JSON)을 생성합니다.
  - Orchestrator 가 이 템플릿을 받아 최종 사용자 대면 텍스트로 변환합니다.

동작 모드 (Orchestrator 가 결정)
  1. meal_info           — 순수 식사 정보 수집 (재료, 조리법, 분량)
  2. recommendation_info — 추천 준비 정보 수집 (선호도, 알레르기, 제약 조건)

출력 (JSON)
  - question_type     : 질문 카테고리
  - target            : 대상 식품 또는 주제
  - reasoning         : 이 질문이 지금 필요한 이유
  - question_template : 사용자에게 보낼 질문 템플릿 (Orchestrator 가 최종 변환)

설계 원칙
  - Principle 1 : coach_use_template_guidance=True 이면 ACTION_GUIDELINES 를 시스템
                  프롬프트에 포함하여 질문 전략 힌트를 제공.
  - Principle 2 : own_buffer (ConversationBuffer) 로 자신의 발화 이력을 관리.
  - Principle 3 : SharedConversationHistory.context_window 를 통해 최근 N 턴만 참조.
  - Principle 4 : system prompt 에 conversation summary 를 주입.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List

from config import ACTION_GUIDELINES
from core.memory import ConversationBuffer, SharedConversationHistory
from utils.llm_utils import generate_response

if TYPE_CHECKING:
    from config import SimulationConfig


# ──────────────────────────────────────────────────────────────────────────────
# 중복 질문 감지 헬퍼
# ──────────────────────────────────────────────────────────────────────────────

_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "do", "does", "did", "will", "would", "could", "should", "can",
    "i", "you", "your", "my", "me", "it", "its", "we", "they",
    "of", "in", "on", "at", "to", "for", "with", "by", "from",
    "and", "or", "not", "no", "but", "if", "so", "as", "than",
    "that", "this", "what", "how", "much", "many", "some", "any",
    "have", "has", "had", "about",
})


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    import re
    return re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9\s]', '', text.lower())).strip()


def _is_duplicate_question(new_q: str, already_asked: list, threshold: float = 0.85) -> bool:
    """Exact normalized match OR Jaccard word-overlap above threshold."""
    norm_new = _normalize(new_q)
    if not norm_new:
        return False
    for prev in already_asked:
        if _normalize(prev) == norm_new:
            return True
        words_new  = {w for w in norm_new.split() if w not in _STOPWORDS}
        words_prev = {w for w in _normalize(prev).split() if w not in _STOPWORDS}
        if not words_new or not words_prev:
            continue
        union = words_new | words_prev
        if len(words_new & words_prev) / len(union) >= threshold:
            return True
    return False


# ──────────────────────────────────────────────────────────────────────────────
# 시스템 프롬프트 — Meal Info 모드
# ──────────────────────────────────────────────────────────────────────────────

_IS_MEAL_INFO_SYSTEM = """\
You are a question generator for a nutritional micro-coaching chatbot.
Your task is to generate ONE structured question template to learn more about the user's meal.

User's nutritional goal : {nutrition_goal}
Meal type               : {meal_type}

Context:
- TURN 0 (already done): The user listed ALL food items they plan to eat.
- Your job: Help the chatbot understand the meal well enough to evaluate it against the goal.
  This could involve ingredients, preparation, portions, combinations, or any other detail
  that would be nutritionally meaningful.

Guidelines:
- Generate exactly ONE question per call.
- Focus on what is genuinely unknown and nutritionally relevant to the goal.
- If the user previously said "I'm not sure" about a topic, do not re-ask it.
- Let the conversation context guide your question — there is no fixed checklist.
  Ask whatever makes the most sense given what is already known and what is still unclear.

Output ONLY a valid JSON object with exactly these fields:
{{"question_type": "<short label for the question category>", \
"target": "<specific food item or aspect>", \
"reasoning": "<why this question matters now>", \
"question_template": "<natural English question, one sentence>"}}\
"""

# ──────────────────────────────────────────────────────────────────────────────
# 시스템 프롬프트 — Recommendation Info 모드
# ──────────────────────────────────────────────────────────────────────────────

_IS_REC_INFO_SYSTEM = """\
You are a question generator for a nutritional micro-coaching chatbot.
Your task is to generate ONE structured question template to learn about the user's \
preferences so the chatbot can make personalized meal recommendations.

User's nutritional goal : {nutrition_goal}
Meal type               : {meal_type}

Context:
- The chatbot has already assessed the user's meal and is preparing recommendations.
- You need to gather information that would help tailor recommendations to this specific user.
  This could include food preferences, allergies, dietary restrictions, cooking habits,
  ingredient availability, budget, or anything else that affects what the user can realistically eat.

Guidelines:
- Generate exactly ONE question per call.
- Let the conversation context guide your question — ask what is most useful right now.
- Keep the question friendly and concise.

Output ONLY a valid JSON object with exactly these fields:
{{"question_type": "<short label for the question category>", \
"target": "<specific preference topic>", \
"reasoning": "<why this matters for recommendation>", \
"question_template": "<natural English question, one sentence>"}}\
"""

# ──────────────────────────────────────────────────────────────────────────────
# 공통 컨텍스트 블록
# ──────────────────────────────────────────────────────────────────────────────

_IS_SUMMARY_BLOCK = """\

[Dialog summary so far]
{dialog_summary}
"""

_IS_OWN_BUFFER_BLOCK = """\

[Questions already asked — DO NOT repeat any of these]
{own_buffer}
"""

_IS_ACTION_GUIDELINES_BLOCK = """\

[Question strategy reference]
{action_guidelines}
"""

_IS_DEAD_END_BLOCK = """\

[Topics the user said they are NOT SURE about — DO NOT ask about these again]
{dead_end_list}
Move on to a different food item or a completely different aspect.
"""

_IS_STALL_EXIT_BLOCK = """\

[CLOSING INSTRUCTION — GENERATE A CLOSING QUESTION TEMPLATE]
The user has been unable to provide additional details despite several questions.
Generate a question template with question_type "closing" that acknowledges what the user \
DID share (mention the actual food items) and lets them know you have enough information.
Set question_template to a warm, natural closing sentence — NOT a question.
"""

_IS_NATURAL_CLOSE_BLOCK = """\

[CLOSING INSTRUCTION — GENERATE A CLOSING QUESTION TEMPLATE]
The user has indicated they have shared everything about their meal.
Generate a question template with question_type "closing" that briefly acknowledges what \
the user shared and thanks them.
Set question_template to a warm, natural closing sentence — NOT a question.
"""

# ──────────────────────────────────────────────────────────────────────────────
# 구 시스템 프롬프트 (batch 모드 호환용 — _COACH_SYSTEM_BASE)
# app.py의 coach-preview 엔드포인트에서 참조하므로 유지합니다.
# ──────────────────────────────────────────────────────────────────────────────

_COACH_SYSTEM_BASE = """\
You are a nutritional micro-coaching chatbot.
Your task is to learn about the user's planned meal through short, focused questions \
— one question per turn.

User's nutritional goal : {nutrition_goal}
Meal type               : {meal_type}

Conversation structure:
- TURN 0 (already done): You asked what the user is having, and they listed ALL the food items.
- TURN 1 ONWARD (your job now): The food item names are known. Ask about whatever details are
  still unclear and nutritionally relevant — ingredients, preparation, portions, combinations,
  or anything else that helps evaluate the meal against the goal.

Guidelines:
- Ask EXACTLY ONE short question per turn.
- Focus on what is genuinely unknown and relevant to the nutritional goal.
- NEVER repeat a question you have already asked — check your previous questions below.
- If the user responds with "I'm not sure" or "I haven't decided", move on to something else.
- Always respond in English with a single sentence question only.
"""

_COACH_ACTION_GUIDELINES_BLOCK = """\

[Question strategy reference]
{action_guidelines}
"""


# ──────────────────────────────────────────────────────────────────────────────
# InformationSeeker
# ──────────────────────────────────────────────────────────────────────────────

class InformationSeeker:
    """
    LLM 기반 정보 수집 모델 — 구조화된 질문 템플릿을 생성합니다.
    Orchestrator 가 이 템플릿을 최종 사용자 대면 텍스트로 변환합니다.

    Parameters
    ----------
    model          : LLM 객체 (loaded externally)
    nutrition_goal : "lean_protein" | "half_fruits_vegetables" | "one_fourth_carbs"
    meal_type      : "breakfast" | "lunch" | "dinner" | "a snack" 등
    config         : SimulationConfig 인스턴스
    """

    # turn=0 에서 사용하는 고정 초기 발화
    INITIAL_QUESTION_TEMPLATE = "What are you thinking of having for {meal_type}?"

    def __init__(
        self,
        model,
        nutrition_goal: str,
        meal_type: str,
        config: "SimulationConfig",
    ):
        self.model          = model
        self.nutrition_goal = nutrition_goal
        self.meal_type      = meal_type
        self.config         = config

        # 자신의 발화 기록 (Principle 2)
        self.own_buffer = ConversationBuffer(role="coach")

    # ── 공개 인터페이스 ────────────────────────────────────────────────────────

    def first_question(self) -> str:
        """턴 0 에서 사용되는 고정 초기 질문을 반환하고 own_buffer 에 기록합니다."""
        q = self.INITIAL_QUESTION_TEMPLATE.format(meal_type=self.meal_type)
        self.own_buffer.add(q)
        return q

    def get_messages(
        self,
        shared_history: SharedConversationHistory,
        dead_end_topics: List[str] | None = None,
        stall_exit: bool = False,
        natural_close: bool = False,
        mode: str = "meal_info",
    ) -> List[Dict[str, str]]:
        """
        LLM 호출용 messages 리스트를 반환합니다.

        Parameters
        ----------
        shared_history  : Coach ↔ User 의 공통 대화 기록
        dead_end_topics : User 가 "I'm not sure" 라 답한 질문 목록
        stall_exit      : True 이면 stall 마무리 발화 생성 지시를 주입
        natural_close   : True 이면 User 자연 종료에 따른 마무리 발화 생성 지시를 주입
        mode            : "meal_info" | "recommendation_info"

        Returns
        -------
        List[Dict[str, str]] : chat-template 형식의 messages
        """
        prev_questions = shared_history.get_all_coach_questions()
        system_prompt = self._build_system_prompt(
            shared_history.dialog_summary,
            prev_questions=prev_questions,
            dead_end_topics=dead_end_topics,
            stall_exit=stall_exit,
            natural_close=natural_close,
            mode=mode,
        )
        return shared_history.build_messages(
            perspective="coach",
            system_prompt=system_prompt,
        )

    def ask(
        self,
        shared_history: SharedConversationHistory,
        dead_end_topics: List[str] | None = None,
        mode: str = "meal_info",
    ) -> Dict:
        """
        현재 shared_history 를 바탕으로 구조화된 질문 템플릿을 LLM 으로 생성합니다.

        Parameters
        ----------
        shared_history  : Coach ↔ User 의 공통 대화 기록
        dead_end_topics : User 가 "I'm not sure" 라 답한 질문 목록
        mode            : "meal_info" | "recommendation_info"

        Returns
        -------
        Dict : {question_type, target, reasoning, question_template}
        """
        messages = self.get_messages(
            shared_history,
            dead_end_topics=dead_end_topics,
            mode=mode,
        )

        raw = generate_response(
            self.model,
            messages,
            max_new_tokens=self.config.max_new_tokens,
            sampling=self.config.sampling,
        )

        template = self._parse_template(raw)

        # 중복 질문 감지 + 재시도 (최대 2회)
        _already_asked = shared_history.get_all_coach_questions()
        question_text = template.get("question_template", "")
        _GENERIC_FALLBACK_TEMPLATE = {
            "question_type": "fallback",
            "target": "meal",
            "reasoning": "Generic follow-up after duplicate detection",
            "question_template": "Could you tell me more about how this meal is put together?",
        }

        for _attempt in range(2):
            if not _is_duplicate_question(question_text, _already_asked):
                break
            _retry_msgs = messages + [{
                "role": "user",
                "content": (
                    "[SYSTEM NOTE: The question you just generated was already asked. "
                    "Please ask about a completely different food item or a new aspect "
                    "that has NOT yet been covered in this conversation.]"
                ),
            }]
            _retry_raw = generate_response(
                self.model,
                _retry_msgs,
                max_new_tokens=self.config.max_new_tokens,
                sampling=self.config.sampling,
            )
            template = self._parse_template(_retry_raw)
            question_text = template.get("question_template", "")
        else:
            if _is_duplicate_question(question_text, _already_asked):
                template = dict(_GENERIC_FALLBACK_TEMPLATE)

        self.own_buffer.add(template.get("question_template", ""))
        return template

    # ── 내부 헬퍼 ──────────────────────────────────────────────────────────────

    def _parse_template(self, raw_output: str) -> Dict:
        """LLM 출력에서 JSON 질문 템플릿을 파싱합니다."""
        import json as _json
        import re as _re

        fallback = {
            "question_type": "fallback",
            "target": "meal",
            "reasoning": "(parse error)",
            "question_template": "Could you tell me more about your meal?",
        }
        try:
            text = raw_output.strip()
            match = _re.search(r'\{[^{}]*\}', text, _re.DOTALL)
            if match:
                data = _json.loads(match.group())
            else:
                data = _json.loads(text)
            return {
                "question_type": str(data.get("question_type", "fallback")),
                "target": str(data.get("target", "")),
                "reasoning": str(data.get("reasoning", "")),
                "question_template": str(
                    data.get("question_template", fallback["question_template"])
                ),
            }
        except (ValueError, TypeError):
            # JSON 파싱 실패 시 raw text 를 question_template 으로 사용
            clean = raw_output.strip()
            if clean and len(clean) < 200:
                fallback["question_template"] = clean
            return fallback

    def _build_system_prompt(
        self,
        dialog_summary: str,
        prev_questions: List[str] | None = None,
        dead_end_topics: List[str] | None = None,
        stall_exit: bool = False,
        natural_close: bool = False,
        mode: str = "meal_info",
    ) -> str:
        """시스템 프롬프트를 동적으로 조합합니다.

        mode:
          - "meal_info"           : 구조화 JSON 출력 (Orchestrator 경유)
          - "recommendation_info" : 구조화 JSON 출력 (Orchestrator 경유)
          - "batch"               : 자연어 질문 직접 출력 (배치 시뮬레이션용)
        """
        if mode == "recommendation_info":
            base = _IS_REC_INFO_SYSTEM.format(
                nutrition_goal=self.nutrition_goal,
                meal_type=self.meal_type,
            )
        elif mode == "batch":
            base = _COACH_SYSTEM_BASE.format(
                nutrition_goal=self.nutrition_goal,
                meal_type=self.meal_type,
            )
        else:
            base = _IS_MEAL_INFO_SYSTEM.format(
                nutrition_goal=self.nutrition_goal,
                meal_type=self.meal_type,
            )

        parts = [base]

        # 요약이 존재할 때만 삽입 (Principle 4)
        if dialog_summary:
            parts.append(_IS_SUMMARY_BLOCK.format(dialog_summary=dialog_summary))

        # 자신의 발화 이력 (Principle 2)
        if prev_questions:
            pq_lines = "\n".join(f"  {i+1}. {q}" for i, q in enumerate(prev_questions))
            pq_text = f"Questions already asked:\n{pq_lines}"
        else:
            pq_text = self.own_buffer.to_prompt_text(header="Questions already asked")
        parts.append(_IS_OWN_BUFFER_BLOCK.format(own_buffer=pq_text))

        # Action 가이드라인 선택적 포함 (Principle 1) — meal_info 또는 batch 모드에서만
        if mode in ("meal_info", "batch") and self.config.coach_use_template_guidance:
            parts.append(_IS_ACTION_GUIDELINES_BLOCK.format(
                action_guidelines=ACTION_GUIDELINES,
            ))

        # Dead-end 토픽 목록
        if dead_end_topics:
            dead_end_list = "\n".join(f"  - {t}" for t in dead_end_topics)
            parts.append(_IS_DEAD_END_BLOCK.format(dead_end_list=dead_end_list))

        # Stall-exit / Natural-close
        if natural_close:
            parts.append(_IS_NATURAL_CLOSE_BLOCK)
        elif stall_exit:
            parts.append(_IS_STALL_EXIT_BLOCK)

        return "".join(parts)
