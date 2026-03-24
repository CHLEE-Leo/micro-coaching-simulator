"""
models/coach.py
───────────────
LLM 기반 Coach 모델.

역할
  - 영양 목표와 식사 유형을 입력받아 User 로부터 식사 정보를 이끌어냅니다.
  - 턴 0에서는 고정 발화를 사용하고, 이후 턴부터는 LLM 으로 질문을 생성합니다.

설계 원칙 대응
  - Principle 1 : coach_use_template_guidance=True 이면 ACTION_GUIDELINES 를 시스템
                  프롬프트에 포함하여 Coach 가 어떤 종류의 질문을 해야 하는지 힌트를 제공.
                  False 이면 LLM 이 자유롭게 질문을 구성합니다.
  - Principle 2 : own_buffer (ConversationBuffer) 로 Coach 자신의 발화 이력을 관리.
  - Principle 3 : SharedConversationHistory.context_window 를 통해 최근 N 턴만 참조.
  - Principle 4 : system prompt 에 conversation summary 를 주입.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Dict

from config import ACTION_GUIDELINES
from core.memory import ConversationBuffer, SharedConversationHistory
from utils.llm_utils import generate_response

if TYPE_CHECKING:
    from config import SimulationConfig


# ──────────────────────────────────────────────────────────────────────────────
# 시스템 프롬프트 템플릿
# ──────────────────────────────────────────────────────────────────────────────

_COACH_SYSTEM_BASE = """\
You are a nutritional micro-coaching chatbot.
Your task is to uncover the ingredient and preparation details of the user’s planned meal \
through short, focused questions -- one question per turn.

User's nutritional goal : {nutrition_goal}
Meal type               : {meal_type}

Conversation structure:
- TURN 0 (already done): You asked what the user is having, and they listed ALL the food items they plan to eat.
- TURN 1 ONWARD (your job now): The food item names are already known. Focus entirely on uncovering
  the INGREDIENT and PREPARATION details for each food item, one food at a time.
  Ask about: what specific ingredients or components are in each food, how it is cooked/prepared,
  and approximate portions. Do NOT ask "what else are you having?" -- all foods are already listed.

Strict rules:
- Ask EXACTLY ONE short question per turn.
- Each question must target the ingredient, preparation method, or portion of a specific food item the user mentioned.
- NEVER ask about food items the user has not mentioned themselves.
- NEVER repeat a question you have already asked -- check your previous questions below.
- Do NOT ask about nutritional labels, calories, protein grams, brands, packaging, shelf life, or any trivial shopping/storage detail.
- If the user responds with “I’m not sure” or “I haven’t decided”, do NOT re-ask the same topic. Move on to a different food item or a different aspect.
- NEVER output the phrase “That’s all about my meal.” -- that phrase belongs to the user only. You always ask a question.
- If the user says “That’s all about my meal.”, stop immediately.
- Always respond in English with a single sentence question only.
"""

_COACH_SUMMARY_BLOCK = """\

[Conversation Summary so far]
{summary}
"""

_COACH_OWN_BUFFER_BLOCK = """\

[Questions you have ALREADY asked — DO NOT repeat any of these]
{own_buffer}
"""

_COACH_ACTION_GUIDELINES_BLOCK = """\

[Question strategy reference]
{action_guidelines}
"""

_COACH_DEAD_END_BLOCK = """\

[Topics the user already said they are NOT SURE about — DO NOT ask about these again]
{dead_end_list}
Move on to a different food item or a completely different aspect of the meal.
"""

_COACH_STALL_EXIT_BLOCK = """\

[CLOSING INSTRUCTION — THIS IS YOUR FINAL MESSAGE]
The user has been unable to provide additional details despite several questions.
Generate exactly ONE warm, natural closing sentence that:
  1. Briefly acknowledges what the user DID share (mention the actual food items).
  2. Lets them know you now have enough information to work with.
  3. Does NOT ask any new question.
Example style: "Thanks for sharing all that — I think I have a good picture of your meal!"
"""

_COACH_NATURAL_CLOSE_BLOCK = """\

[CLOSING INSTRUCTION — THIS IS YOUR FINAL MESSAGE]
The user has indicated they have shared everything about their meal.
Generate exactly ONE warm, natural closing sentence that:
  1. Briefly acknowledges what the user shared (mention the actual food items).
  2. Thanks them and lets them know you have what you need.
  3. Does NOT ask any new question.
Example style: "Great, thanks for sharing all the details — I now have a clear picture of your ham sandwich!"
"""


# ──────────────────────────────────────────────────────────────────────────────
# CoachModel
# ──────────────────────────────────────────────────────────────────────────────

class CoachModel:
    """
    LLM 기반 영양 코치 모델.

    Parameters
    ----------
    model          : vLLM LLM 객체 (loaded externally for resource sharing)
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

        # Coach 자신의 발화 기록 (Principle 2)
        self.own_buffer = ConversationBuffer(role="coach")

    # ── 공개 인터페이스 ────────────────────────────────────────────────────────

    def first_question(self) -> str:
        """
        턴 0 에서 사용되는 고정 초기 질문을 반환하고 own_buffer 에 기록합니다.
        """
        q = self.INITIAL_QUESTION_TEMPLATE.format(meal_type=self.meal_type)
        self.own_buffer.add(q)
        return q

    def get_messages(
        self,
        shared_history: SharedConversationHistory,
        dead_end_topics: List[str] | None = None,
        stall_exit: bool = False,
        natural_close: bool = False,
    ) -> List[Dict[str, str]]:
        """
        배치 생성용: LLM 을 호출하지 않고 messages 리스트만 반환합니다.
        batch_generate() 에 넘길 때 사용합니다.

        Parameters
        ----------
        shared_history  : Coach ↔ User 의 공통 대화 기록
        dead_end_topics : User 가 "I'm not sure" 라 답한 질문 목록 (Coach 반복 방지)
        stall_exit      : True 이면 stall 마무리 발화 생성 지시를 시스템 프롬프트에 주입
        natural_close   : True 이면 User 자연 종료엠 따른 마무리 발화 생성 지시를 주입

        Returns
        -------
        List[Dict[str, str]] : chat-template 형식의 messages
        """
        # history에서 Coach가 발화한 모든 질문을 직접 추출 (own_buffer보다 확실한 완전 목록)
        # Extract all coach questions directly from history — guaranteed complete list
        prev_questions = shared_history.get_all_coach_questions()
        system_prompt = self._build_system_prompt(
            shared_history.summary,
            prev_questions=prev_questions,
            dead_end_topics=dead_end_topics,
            stall_exit=stall_exit,
            natural_close=natural_close,
        )
        return shared_history.build_messages(
            perspective="coach",
            system_prompt=system_prompt,
        )

    def ask(
        self,
        shared_history: SharedConversationHistory,
    ) -> str:
        """
        현재 shared_history 를 바탕으로 다음 질문을 LLM 으로 생성합니다.
        단일 대화 처리(single-dialog) 모드에서 사용합니다.

        Parameters
        ----------
        shared_history : Coach ↔ User 의 공통 대화 기록

        Returns
        -------
        str : 생성된 Coach 질문
        """
        messages = self.get_messages(shared_history)

        question = generate_response(
            self.model,
            messages,
            max_new_tokens=self.config.max_new_tokens,
            sampling=self.config.sampling,
        )

        self.own_buffer.add(question)
        return question

    # ── 내부 헬퍼 ──────────────────────────────────────────────────────────────

    def _build_system_prompt(
        self,
        summary: str,
        prev_questions: List[str] | None = None,
        dead_end_topics: List[str] | None = None,
        stall_exit: bool = False,
        natural_close: bool = False,
    ) -> str:
        """
        Coach 의 시스템 프롬프트를 동적으로 조합합니다.

        구성 순서:
          1. 기본 역할 + 목표 설명
          2. (있으면) 대화 요약  (Principle 4)
          3. Coach 자신의 발화 이력  (Principle 2) — history에서 직접 추출
          4. (config 에 따라) Action 가이드라인  (Principle 1)
          5. (있으면) dead-end 토픽 목록 (반복 방지)
          6. (stall_exit=True 이면) stall 마무리 발화 지시
             (natural_close=True 이면) 자연 종료 마무리 발화 지시
        """
        parts = [
            _COACH_SYSTEM_BASE.format(
                nutrition_goal=self.nutrition_goal,
                meal_type=self.meal_type,
            )
        ]

        # 요약이 존재할 때만 삽입 (Principle 4)
        if summary:
            parts.append(_COACH_SUMMARY_BLOCK.format(summary=summary))

        # 자신의 발화 이력 (Principle 2) — history에서 직접 추출한 완전한 목록 우선 사용
        if prev_questions:
            pq_lines = "\n".join(f"  {i+1}. {q}" for i, q in enumerate(prev_questions))
            pq_text = f"Questions you have already asked:\n{pq_lines}"
        else:
            pq_text = self.own_buffer.to_prompt_text(header="Questions you have already asked")
        parts.append(_COACH_OWN_BUFFER_BLOCK.format(own_buffer=pq_text))

        # Action 가이드라인 선택적 포함 (Principle 1)
        if self.config.coach_use_template_guidance:
            parts.append(_COACH_ACTION_GUIDELINES_BLOCK.format(
                action_guidelines=ACTION_GUIDELINES
            ))

        # Dead-end 토픽 목록 (User가 모른다고 한 질문들)
        if dead_end_topics:
            dead_end_list = "\n".join(f"  - {t}" for t in dead_end_topics)
            parts.append(_COACH_DEAD_END_BLOCK.format(dead_end_list=dead_end_list))

        # Stall-exit / Natural-close: 마무리 발화 지시
        if natural_close:
            parts.append(_COACH_NATURAL_CLOSE_BLOCK)
        elif stall_exit:
            parts.append(_COACH_STALL_EXIT_BLOCK)

        return "".join(parts)
