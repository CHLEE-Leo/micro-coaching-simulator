"""
models/orchestrator.py
──────────────────────
LLM 기반 중앙 Orchestrator — 대화의 커뮤니케이션 허브.

역할
  1. Router    — 대화 상태를 종합 분석하여 다음 행동(Action)을 결정
  2. Assessor  — 식사 평가를 생성 (info_seeking → recommending 전환 시)
  3. TextGen   — 서브 에이전트의 구조화된 템플릿을 LLM으로 자연어 텍스트 생성

데이터 흐름
  User ↔ [Guardrail] ↔ Orchestrator ↔ {InformationSeeker, MealRecommender, Estimators}

행동 (Actions)
  - seek_meal_info           : IS(meal-info 모드) 호출 → 식사 세부 질문
  - seek_recommendation_info : IS(recommendation-info 모드) 호출 → 추천 준비 질문
  - assess_meal              : 식사 평가 생성 (전환 시점, double-turn)
  - recommend                : MealRecommender 호출 → 식사 개선 추천
  - terminate                : 대화 종료

Phase 흐름
  info_seeking → assessment → rec_info_seeking → recommending → terminated

안전장치
  - max_turns 도달 시 강제 종료
  - JSON 파싱 실패 시 현재 phase 에 맞는 fallback action
  - 유효하지 않은 action 은 현재 phase 기본 action 으로 처리

Estimator Bundle (AlignmentEstimator + UncertaintyEstimator)
  - Router 가 IS/Recommender/Assessment 를 호출할 때만 실행
  - terminate 시에는 실행하지 않아 레이턴시 절감
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from config import SimulationConfig
    from core.memory import SharedConversationHistory


# ──────────────────────────────────────────────────────────────────────────────
# Phase 정의
# ──────────────────────────────────────────────────────────────────────────────

PHASES = (
    "info_seeking",
    "assessment",
    "rec_info_seeking",
    "recommending",
    "negotiation",
    "motivational_ending",
    "terminated",
)

# Phase 별 허용 action
_PHASE_ACTIONS = {
    "info_seeking":        frozenset({"seek_meal_info", "assess_meal", "terminate"}),
    "assessment":          frozenset({"assess_meal", "terminate"}),
    "rec_info_seeking":    frozenset({"seek_recommendation_info", "recommend", "terminate"}),
    "recommending":        frozenset({"recommend", "seek_recommendation_info", "motivational_close", "terminate"}),
    "negotiation":         frozenset({"seek_recommendation_info", "recommend", "motivational_close", "terminate"}),
    "motivational_ending": frozenset({"motivational_close", "terminate"}),
    "terminated":          frozenset(),
}

# Phase 별 기본 fallback action
_PHASE_FALLBACK = {
    "info_seeking":        "seek_meal_info",
    "assessment":          "assess_meal",
    "rec_info_seeking":    "seek_recommendation_info",
    "recommending":        "recommend",
    "negotiation":         "seek_recommendation_info",
    "motivational_ending": "motivational_close",
    "terminated":          "terminate",
}


# ──────────────────────────────────────────────────────────────────────────────
# 시스템 프롬프트 — Router
# ──────────────────────────────────────────────────────────────────────────────

_ROUTER_SYSTEM = """\
You are the central orchestrator of a nutritional micro-coaching conversation.
You are the ONLY agent that communicates with the user (through a safety filter).

Your role: Analyze the conversation state and decide the next action.

Nutritional goal: {nutrition_goal}
Goal definition: {goal_definition}

═══ CONVERSATION PHASES ═══

1. INFO_SEEKING — Gather meal details (ingredients, preparation, portions).
   → When the Fact Sheet has enough detail, transition to ASSESS_MEAL.
   → If meal clearly aligns with goal from available info, TERMINATE.

2. ASSESSMENT — Evaluate the meal (auto-transition after assessment is shown).
   → After assessment, move to REC_INFO_SEEKING or TERMINATE.

3. REC_INFO_SEEKING — Gather user preferences/allergies for personalized recommendation.
   → After 1-2 preference questions (or if preferences already known), move to RECOMMEND.

4. RECOMMENDING — Deliver meal improvement recommendations.
   → If the user accepts (positive response, agreement, openness), transition to MOTIVATIONAL_CLOSE.
   → If the user rejects or expresses concerns (dislike, allergy, impractical), \
transition to NEGOTIATION (SEEK_RECOMMENDATION_INFO or RECOMMEND).

5. NEGOTIATION — User rejected or expressed concerns about the recommendation.
   → Understand the reason (SEEK_RECOMMENDATION_INFO) or offer an alternative (RECOMMEND).
   → When user accepts an alternative or negotiation is exhausted, MOTIVATIONAL_CLOSE.

6. MOTIVATIONAL_ENDING — Wrap-up phase (auto-handled by the system).
   → MOTIVATIONAL_CLOSE only.

Current phase: {current_phase}

═══ AVAILABLE ACTIONS ═══

• SEEK_MEAL_INFO — Ask about ingredients, preparation, portions (INFO_SEEKING phase).
• ASSESS_MEAL — Generate meal assessment (transitions from INFO_SEEKING). \
Use when Fact Sheet has enough detail for a meaningful evaluation.
• SEEK_RECOMMENDATION_INFO — Ask about preferences, allergies, availability \
(REC_INFO_SEEKING / NEGOTIATION phase).
• RECOMMEND — Suggest a specific meal improvement (RECOMMENDING / NEGOTIATION phase).
• MOTIVATIONAL_CLOSE — Transition to encouraging wrap-up. Use when: \
the user has accepted a recommendation, or negotiation has concluded. \
The system will generate assessment + health tip + motivational close automatically.
• TERMINATE — Force-end the conversation. Use when: meal aligns with goal, \
max turns approaching, or conversation cannot continue.

═══ DECISION GUIDELINES ═══

• Read the Fact Sheet carefully — ask only about what is genuinely unknown.
• If the user said "I'm not sure" about something, do not ask again. Move on.
• In INFO_SEEKING: typically 3-5 questions are enough. Do not over-ask.
• If the meal clearly meets the goal, skip assessment and TERMINATE positively.
• As turn count approaches max_turns, bias toward ASSESS_MEAL or TERMINATE.
• For TERMINATE: include a brief, warm closing message in "instruction".
• In RECOMMENDING: if the user responds positively ("sounds good", "I'll try that", \
"sure"), choose MOTIVATIONAL_CLOSE.
• In RECOMMENDING: if the user rejects ("I don't like that", "that won't work", \
"I'm allergic"), choose SEEK_RECOMMENDATION_INFO or RECOMMEND to negotiate.
• In NEGOTIATION: balance between gathering more context and offering alternatives. \
After 2-3 exchanges, lean toward MOTIVATIONAL_CLOSE to wrap up.

═══ OUTPUT FORMAT ═══

Analyse the user's latest message first, then decide.

Return ONLY a JSON object:
{{{{"user_intent": "<1-2 sentence analysis of what the user's last message conveys: \
what info they provided, what they are uncertain about, or what they are asking for>", \
"action": "seek_meal_info" | "seek_recommendation_info" | "assess_meal" \
| "recommend" | "motivational_close" | "terminate", \
"reasoning": "<2-4 sentences explaining why this action given the intent>", \
"instruction": "<recommendation guidance for MealRecommender, or closing/transition message>"}}}}\
"""

_ROUTER_USER = """\
[Turn {turn_idx} / {max_turns}]
[Phase: {current_phase}]

[Meal Fact Sheet]
{meal_fact_sheet}

[Dialog Summary]
{dialog_summary}

[Recommendation History]
{recommendation_history}

[Recent Conversation]
{recent_turns}

Decide the next action.\
"""

# ──────────────────────────────────────────────────────────────────────────────
# 시스템 프롬프트 — Assessment
# ──────────────────────────────────────────────────────────────────────────────

_ASSESSMENT_SYSTEM = """\
You are evaluating a user's meal against a nutritional goal.

Nutritional goal: {nutrition_goal}
Goal definition: {goal_definition}

Alignment assessment: score = {alignment_score}, reasoning = "{alignment_reasoning}"

Based on the Meal Fact Sheet and alignment data, generate a concise meal assessment.

Rules:
- Be specific: reference actual foods from the Fact Sheet.
- Keep strengths/limitations to 1-3 items each.
- "overall" must reflect whether the meal truly meets the goal.

Output ONLY a JSON object:
{{"summary": "<1-2 sentence meal overview>", \
"strengths": ["<positive aspect>", ...], \
"limitations": ["<area for improvement>", ...], \
"overall": "aligned" | "partially_aligned" | "not_aligned"}}\
"""

_ASSESSMENT_USER = """\
[Meal Fact Sheet]
{meal_fact_sheet}

Generate the meal assessment.\
"""


# ──────────────────────────────────────────────────────────────────────────────
# 유효한 action 값
# ──────────────────────────────────────────────────────────────────────────────

_VALID_ACTIONS = frozenset({
    "seek_meal_info", "seek_recommendation_info",
    "assess_meal", "recommend", "motivational_close", "terminate",
})

_FALLBACK_DECISION = {
    "user_intent": "",
    "action": "seek_meal_info",
    "reasoning": "(fallback: orchestrator output could not be parsed)",
    "instruction": "",
}

_FALLBACK_ASSESSMENT = {
    "summary": "",
    "strengths": [],
    "limitations": [],
    "overall": "partially_aligned",
}


# ──────────────────────────────────────────────────────────────────────────────
# 시스템 프롬프트 — TextGen (질문 생성)
# ──────────────────────────────────────────────────────────────────────────────

_TEXTGEN_QUESTION_SYSTEM = """\
You are the central Orchestrator of a nutritional micro-coaching chatbot.
You are the ONLY agent that speaks directly to the user.

A sub-agent (InformationSeeker) has analysed the conversation and produced \
the structured question template below.  It contains:
- **question_type**: category of the question
- **target**: the specific food or aspect to ask about
- **reasoning**: why this question matters now
- **question_template**: a rough draft question for guidance

Your task: Use this template as GUIDANCE to compose ONE natural, warm, \
conversational question for the user.

Nutritional goal: {nutrition_goal}

Rules:
- Use the template's target, reasoning, and question_template as guidance \
— do NOT copy the template verbatim.
- Produce exactly ONE clear, friendly question (1-2 sentences max).
- Reference specific foods the user mentioned when relevant.
- Match the tone of the recent conversation.
- Do NOT expose internal analysis or mention "template", "reasoning", \
"sub-agent", "score", etc.
- Output ONLY the question text — no JSON, no labels, no surrounding quotes.\
"""

_TEXTGEN_QUESTION_USER = """\
[Sub-agent Question Template]
{question_json}

[Recent Conversation]
{recent_turns}

Compose a natural question for the user.\
"""

# ──────────────────────────────────────────────────────────────────────────────
# 시스템 프롬프트 — TextGen (추천 생성)
# ──────────────────────────────────────────────────────────────────────────────

_TEXTGEN_RECOMMENDATION_SYSTEM = """\
You are the central Orchestrator of a nutritional micro-coaching chatbot.
You are the ONLY agent that speaks directly to the user.

A sub-agent (MealRecommender) has analysed the meal and produced \
the structured recommendation template below.  It contains:
- **recommendation_type**: substitute / modify / add
- **target_food**: the specific food item to change
- **suggestion**: the concrete improvement
- **reasoning**: why this change helps meet the goal
- **expected_impact**: high / medium / low

Your task: Use this template as GUIDANCE to compose a natural, encouraging \
recommendation message for the user.

Nutritional goal: {nutrition_goal}

Rules:
- Use the template's fields as guidance — do NOT list them mechanically.
- Produce a concise, friendly recommendation (2-4 sentences).
- Explain WHY the change helps in simple, everyday terms (no jargon).
- End with an inviting question like "What do you think?" or \
"Would you like to try that?".
- Do NOT expose internal analysis or mention "template", "sub-agent", \
"score", "impact", etc.
- Output ONLY the recommendation text — no JSON, no labels, \
no surrounding quotes.\
"""

_TEXTGEN_RECOMMENDATION_USER = """\
[Sub-agent Recommendation Template]
{recommendation_json}

[Recent Conversation]
{recent_turns}
{previous_recommendations_context}
Compose a natural recommendation message for the user.\
"""

# ──────────────────────────────────────────────────────────────────────────────
# 시스템 프롬프트 — TextGen (Assessment 피드백 생성)
# ──────────────────────────────────────────────────────────────────────────────

_TEXTGEN_ASSESSMENT_SYSTEM = """\
You are the central Orchestrator of a nutritional micro-coaching chatbot.
You are the ONLY agent that speaks directly to the user.

You have just completed the information-gathering phase and assessed \
the user's meal against their nutritional goal: **{nutrition_goal}**.

The assessment result below contains:
- **summary**: what you gathered about the meal
- **strengths**: positive aspects
- **limitations**: areas for improvement
- **overall**: aligned / partially_aligned / not_aligned
- **needs_recommendation**: whether you should transition to recommendations

Your task: Compose a natural, warm feedback message that:
1. Briefly acknowledges what the user shared (don't just restate — show understanding).
2. Highlights strengths positively.
3. Mentions limitations constructively (never blame or lecture).
4. If needs_recommendation is true, end by briefly noting you'd like to \
explore some ideas that could help — but do NOT ask about preferences, \
allergies, or restrictions. A follow-up question will handle that separately.
5. If needs_recommendation is false (meal is well-aligned), congratulate \
the user and close warmly.

Rules:
- Write in a conversational, coaching tone (2-4 sentences).
- This message is ONLY assessment feedback — do NOT include questions.
- Do NOT use bullet points, labels like "Strengths:", or mechanical formatting.
- Do NOT expose internal terms like "aligned", "score", "assessment", "template".
- Output ONLY the feedback text — no JSON, no labels, no surrounding quotes.\
"""

_TEXTGEN_ASSESSMENT_USER = """\
[Assessment Result]
{assessment_json}

[Recent Conversation]
{recent_turns}

Compose a natural feedback message for the user.\
"""

# ──────────────────────────────────────────────────────────────────────────────
# 시스템 프롬프트 — TextGen (Motivational Ending 메시지 생성)
# ──────────────────────────────────────────────────────────────────────────────

_TEXTGEN_MOTIVATIONAL_SYSTEM = """\
You are the central Orchestrator of a nutritional micro-coaching chatbot.
You are the ONLY agent that speaks directly to the user.

The conversation is wrapping up. The user has discussed their meal and \
accepted (or at least considered) a recommendation. Now it's time to \
close the conversation on a positive, motivating note.

The nutritional goal was: **{nutrition_goal}**

You have a final assessment of the user's meal (possibly improved \
after following a recommendation):
- **summary**: what the meal looks like now
- **strengths**: positive aspects
- **limitations**: remaining areas for improvement
- **overall**: aligned / partially_aligned / not_aligned

Your task: Compose a warm, encouraging closing message that:
1. Briefly acknowledge how the meal looks overall (reference specific foods).
2. Highlight what's already great about the meal.
3. Regardless of the specific nutritional goal, mention 1-2 brief, practical, \
general health tips for the meal (e.g. hydration, fiber, variety, eating pace). \
Keep these universal and helpful — NOT about the specific goal.
4. End with a short motivational close that encourages continued healthy eating \
(e.g. "Every small step counts!", "Keep exploring new ways to nourish yourself!").

Rules:
- Write in a warm, conversational, coaching tone (3-5 sentences total).
- Do NOT use bullet points, labels, or mechanical formatting.
- Do NOT mention scores, alignment, assessment, templates, or internal terms.
- Do NOT repeat the recommendation that was already discussed — focus on \
the big picture and encouragement.
- Output ONLY the closing message text — no JSON, no labels, no quotes.\
"""

_TEXTGEN_MOTIVATIONAL_USER = """\
[Assessment Result]
{assessment_json}

[Recent Conversation]
{recent_turns}

Compose a motivational closing message for the user.\
"""


# ──────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ──────────────────────────────────────────────────────────────────────────────

class Orchestrator:
    """
    LLM 기반 중앙 Orchestrator — 대화의 커뮤니케이션 허브.

    모든 사용자 대면 텍스트를 LLM으로 생성하고,
    서브 에이전트의 구조화된 출력(템플릿 + reasoning)을
    TextGen LLM 호출을 통해 자연어로 변환합니다.

    Parameters
    ----------
    nutrition_goal : 영양 목표
    config         : SimulationConfig 인스턴스
    """

    def __init__(self, nutrition_goal: str, config: "SimulationConfig"):
        self.nutrition_goal = nutrition_goal
        self.config = config

        from models.meal_recommender import _load_goal_definitions
        goal_spec = _load_goal_definitions().get(nutrition_goal, {})
        self._goal_definition = goal_spec.get("definition", "")

        # Router 시스템 프롬프트 (current_phase 는 호출 시 채움)
        self._router_system_template = _ROUTER_SYSTEM.format(
            nutrition_goal=nutrition_goal.replace("_", " "),
            goal_definition=self._goal_definition,
            current_phase="{current_phase}",
        )

        self._decision_history: List[Dict] = []
        self._last_decision: Optional[Dict] = None
        self._last_assessment: Optional[Dict] = None

    # ══════════════════════════════════════════════════════════════════════
    # Router — 다음 행동 결정
    # ══════════════════════════════════════════════════════════════════════

    def get_routing_messages(
        self,
        history: "SharedConversationHistory",
        turn_idx: int,
        phase: str = "info_seeking",
        recommendation_history: Optional[List[Dict]] = None,
    ) -> List[Dict[str, str]]:
        """Router 단계용 LLM messages 리스트를 반환합니다."""
        rec_history_text = "None"
        if recommendation_history:
            rec_lines = [
                f"  Turn {r.get('turn_idx', '?')}: "
                f"{r.get('recommendation_type', '?')} — "
                f"{r.get('target_food', '?')} → {r.get('suggestion', '?')}"
                for r in recommendation_history
            ]
            rec_history_text = "\n".join(rec_lines)

        system = self._router_system_template.format(
            current_phase=phase.upper(),
        )
        user = _ROUTER_USER.format(
            turn_idx=turn_idx,
            max_turns=self.config.max_turns,
            current_phase=phase.upper(),
            meal_fact_sheet=history.meal_fact_sheet or "(not yet available)",
            dialog_summary=history.dialog_summary or "(not yet available)",
            recommendation_history=rec_history_text,
            recent_turns=history.to_alignment_context(),
        )
        return [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]

    def parse_routing(
        self,
        raw_output: str,
        turn_idx: int = 0,
        phase: str = "info_seeking",
    ) -> Dict:
        """
        Router LLM 출력을 파싱합니다.
        Phase 에 맞지 않는 action 은 fallback 처리합니다.
        """
        decision = dict(_FALLBACK_DECISION)
        fallback_action = _PHASE_FALLBACK.get(phase, "seek_meal_info")

        try:
            text = (raw_output or "").strip()
            if not text:
                raise ValueError("empty router output")

            # Markdown 코드 블록 제거 (```json ... ``` 또는 ``` ... ```)
            _md = re.search(r'```(?:json)?\s*(.*?)```', text, re.DOTALL)
            if _md:
                text = _md.group(1).strip()

            # 가장 바깥 {...} 블록 추출 (중첩 허용)
            _brace_match = re.search(r'\{.*\}', text, re.DOTALL)
            data = json.loads(_brace_match.group()) if _brace_match else json.loads(text)

            action = str(data.get("action", "")).strip().lower()

            # 유효하지 않은 action → phase fallback
            if action not in _VALID_ACTIONS:
                action = fallback_action

            # Phase 에 허용되지 않는 action → phase fallback
            allowed = _PHASE_ACTIONS.get(phase, _VALID_ACTIONS)
            if action not in allowed:
                action = fallback_action

            decision = {
                "user_intent": str(data.get("user_intent", "")),
                "action": action,
                "reasoning": str(data.get("reasoning", "")),
                "instruction": str(data.get("instruction", "")),
            }
        except (json.JSONDecodeError, ValueError, TypeError, AttributeError):
            decision["reasoning"] = f"(parse error) raw: {(raw_output or '')[:200]}"
            decision["action"] = fallback_action

        self._last_decision = decision
        self._decision_history.append({"turn_idx": turn_idx, **decision})
        return decision

    def route(
        self,
        history: "SharedConversationHistory",
        turn_idx: int,
        phase: str = "info_seeking",
        recommendation_history: Optional[List[Dict]] = None,
        generate_fn=None,
        llm=None,
    ) -> Dict:
        """편의 메서드: Router 메시지 조립 → LLM 호출 → 파싱."""
        if generate_fn is None:
            from utils.llm_utils import generate_response
            generate_fn = generate_response

        # 안전장치: max_turns 도달 시 강제 종료
        if turn_idx >= self.config.max_turns - 1:
            forced = {
                "action": "terminate",
                "reasoning": f"Max turns ({self.config.max_turns}) reached.",
                "instruction": "Thank you for sharing about your meal!",
            }
            self._last_decision = forced
            self._decision_history.append({"turn_idx": turn_idx, **forced})
            return forced

        msgs = self.get_routing_messages(
            history=history,
            turn_idx=turn_idx,
            phase=phase,
            recommendation_history=recommendation_history,
        )
        raw = generate_fn(
            llm, msgs,
            max_new_tokens=getattr(self.config, 'orchestrator_max_new_tokens', 200),
            sampling="greedy",
        )
        return self.parse_routing(raw, turn_idx=turn_idx, phase=phase)

    # ══════════════════════════════════════════════════════════════════════
    # Assessment — 식사 평가 생성
    # ══════════════════════════════════════════════════════════════════════

    def get_assessment_messages(
        self,
        history: "SharedConversationHistory",
        alignment_score: Optional[float] = None,
        alignment_reasoning: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """Assessment 생성용 LLM messages 리스트를 반환합니다."""
        system = _ASSESSMENT_SYSTEM.format(
            nutrition_goal=self.nutrition_goal.replace("_", " "),
            goal_definition=self._goal_definition,
            alignment_score=(
                f"{alignment_score:.2f}" if alignment_score is not None else "N/A"
            ),
            alignment_reasoning=alignment_reasoning or "N/A",
        )
        user = _ASSESSMENT_USER.format(
            meal_fact_sheet=history.meal_fact_sheet or "(no fact sheet available)",
        )
        return [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]

    def parse_assessment(self, raw_output: str) -> Dict:
        """Assessment LLM 출력을 파싱합니다."""
        assessment = dict(_FALLBACK_ASSESSMENT)
        try:
            text = raw_output.strip()
            match = re.search(r'\{.*\}', text, re.DOTALL)
            data = json.loads(match.group()) if match else json.loads(text)
            assessment = {
                "summary": str(data.get("summary", "")),
                "strengths": list(data.get("strengths", [])),
                "limitations": list(data.get("limitations", [])),
                "overall": str(data.get("overall", "partially_aligned")),
            }
        except (json.JSONDecodeError, ValueError, TypeError):
            assessment["summary"] = f"(parse error) raw: {raw_output[:200]}"

        self._last_assessment = assessment
        return assessment

    def assess(
        self,
        history: "SharedConversationHistory",
        alignment_score: Optional[float] = None,
        alignment_reasoning: Optional[str] = None,
        generate_fn=None,
        llm=None,
    ) -> Dict:
        """편의 메서드: Assessment 메시지 조립 → LLM 호출 → 파싱."""
        if generate_fn is None:
            from utils.llm_utils import generate_response
            generate_fn = generate_response

        msgs = self.get_assessment_messages(
            history=history,
            alignment_score=alignment_score,
            alignment_reasoning=alignment_reasoning,
        )
        raw = generate_fn(
            llm, msgs,
            max_new_tokens=getattr(self.config, 'orchestrator_max_new_tokens', 200),
            sampling="greedy",
        )
        return self.parse_assessment(raw)

    # ══════════════════════════════════════════════════════════════════════
    # TextGen — 서브 에이전트 템플릿 → LLM 기반 사용자 대면 텍스트 생성
    # ══════════════════════════════════════════════════════════════════════

    # ── get_messages ──────────────────────────────────────────────────────

    def get_textgen_assessment_messages(
        self,
        assessment: Dict,
        needs_recommendation: bool,
        history: "SharedConversationHistory",
    ) -> List[Dict[str, str]]:
        """Assessment 결과 → 사용자 대면 피드백을 생성하기 위한 LLM messages."""
        payload = {**assessment, "needs_recommendation": needs_recommendation}
        system = _TEXTGEN_ASSESSMENT_SYSTEM.format(
            nutrition_goal=self.nutrition_goal.replace("_", " "),
        )
        user = _TEXTGEN_ASSESSMENT_USER.format(
            assessment_json=json.dumps(payload, ensure_ascii=False, indent=2),
            recent_turns=history.to_alignment_context(),
        )
        return [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]

    def get_textgen_question_messages(
        self,
        question_template: Dict,
        history: "SharedConversationHistory",
    ) -> List[Dict[str, str]]:
        """IS 질문 템플릿 → 사용자 대면 질문을 생성하기 위한 LLM messages."""
        system = _TEXTGEN_QUESTION_SYSTEM.format(
            nutrition_goal=self.nutrition_goal.replace("_", " "),
        )
        user = _TEXTGEN_QUESTION_USER.format(
            question_json=json.dumps(question_template, ensure_ascii=False, indent=2),
            recent_turns=history.to_alignment_context(),
        )
        return [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]

    def get_textgen_recommendation_messages(
        self,
        rec_template: Dict,
        history: "SharedConversationHistory",
        recommendation_history: Optional[List[Dict]] = None,
    ) -> List[Dict[str, str]]:
        """MR 추천 템플릿 → 사용자 대면 추천 메시지를 생성하기 위한 LLM messages."""
        prev_recs_text = ""
        if recommendation_history:
            rec_lines = [
                f"- Turn {r.get('turn_idx', '?')}: "
                f"{r.get('suggestion', '?')} (target: {r.get('target_food', '?')})"
                for r in recommendation_history
            ]
            prev_recs_text = (
                "\n[Previous Recommendations Already Given]\n"
                + "\n".join(rec_lines)
                + "\nDo NOT contradict or reverse any previously accepted suggestion. "
                "Build upon the user's choices.\n"
            )

        system = _TEXTGEN_RECOMMENDATION_SYSTEM.format(
            nutrition_goal=self.nutrition_goal.replace("_", " "),
        )
        user = _TEXTGEN_RECOMMENDATION_USER.format(
            recommendation_json=json.dumps(rec_template, ensure_ascii=False, indent=2),
            recent_turns=history.to_recent_turns_text(),
            previous_recommendations_context=prev_recs_text,
        )
        return [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]

    def get_textgen_motivational_messages(
        self,
        assessment: Dict,
        history: "SharedConversationHistory",
    ) -> List[Dict[str, str]]:
        """Assessment 결과 → Motivational Ending 메시지를 생성하기 위한 LLM messages."""
        system = _TEXTGEN_MOTIVATIONAL_SYSTEM.format(
            nutrition_goal=self.nutrition_goal.replace("_", " "),
        )
        user = _TEXTGEN_MOTIVATIONAL_USER.format(
            assessment_json=json.dumps(assessment, ensure_ascii=False, indent=2),
            recent_turns=history.to_alignment_context(),
        )
        return [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]

    # ── parse ─────────────────────────────────────────────────────────────

    def parse_textgen(self, raw_output: str, template: Dict) -> str:
        """TextGen LLM 출력을 파싱하여 user-facing 텍스트를 반환합니다.

        LLM이 빈 출력을 내거나 실패하면 template 의 question_template,
        suggestion, 또는 summary 필드를 fallback 으로 사용합니다.
        """
        text = raw_output.strip().strip('"').strip("'").strip()
        if text:
            return text
        # fallback: 서브 에이전트 템플릿의 원문 사용
        return template.get(
            "question_template",
            template.get(
                "suggestion",
                template.get("summary", "Could you tell me more about your meal?"),
            ),
        )

    # ── 기존 render_* (fallback 전용, LLM 실패 시 사용) ─────────────────

    def render_question(self, question_template: Dict) -> str:
        """InformationSeeker 의 질문 템플릿을 사용자 대면 텍스트로 변환합니다.
        (LLM TextGen 실패 시 fallback 용)"""
        return question_template.get(
            "question_template",
            "Could you tell me more about your meal?",
        )

    def render_recommendation(self, rec_template: Dict) -> str:
        """MealRecommender 의 추천 템플릿을 사용자 대면 텍스트로 변환합니다.
        (LLM TextGen 실패 시 fallback 용)"""
        suggestion = rec_template.get("suggestion", "")
        target = rec_template.get("target_food", "")
        reasoning = rec_template.get("reasoning", "")
        if suggestion:
            parts = ["Based on what you've shared, here's a suggestion:"]
            if target:
                parts.append(f"for the {target},")
            parts.append(f"you might consider {suggestion}.")
            if reasoning:
                parts.append(reasoning)
            parts.append("What do you think?")
            return " ".join(parts)
        return "Could you tell me more about your meal?"

    def render_assessment(
        self,
        assessment: Dict,
        needs_recommendation: bool = True,
    ) -> str:
        """Assessment 결과를 사용자 대면 텍스트로 변환합니다.
        (LLM TextGen 실패 시 fallback 용)"""
        parts = []
        summary = assessment.get("summary", "")
        if summary:
            parts.append(f"Here's what I've gathered about your meal: {summary}")

        strengths = assessment.get("strengths", [])
        if strengths:
            parts.append(
                "What's working well: " + "; ".join(str(s) for s in strengths) + "."
            )

        limitations = assessment.get("limitations", [])
        if limitations:
            parts.append(
                "Areas to consider: " + "; ".join(str(l) for l in limitations) + "."
            )

        overall = assessment.get("overall", "partially_aligned")
        if overall == "aligned":
            parts.append("Overall, your meal looks great for your goal!")
        elif needs_recommendation:
            parts.append(
                "I have some suggestions that might help. "
                "Before I share them, do you have any dietary preferences, "
                "allergies, or restrictions I should know about?"
            )

        return "\n\n".join(parts) if parts else "Let me review your meal."

    def render_closing(self, instruction: str = "") -> str:
        """종료 메시지를 렌더링합니다."""
        if instruction:
            return instruction
        return "Thanks for sharing about your meal! I hope this was helpful."

    def render_motivational_close(self, assessment: Dict) -> str:
        """Motivational Ending 메시지를 렌더링합니다.
        (LLM TextGen 실패 시 fallback 용)"""
        parts = []
        strengths = assessment.get("strengths", [])
        if strengths:
            parts.append(
                "Great job — " + " and ".join(str(s) for s in strengths[:2]) + "!"
            )
        limitations = assessment.get("limitations", [])
        if limitations:
            parts.append(
                "A small tip: " + str(limitations[0]) + "."
            )
        parts.append(
            "Keep up the great work with your meals — every healthy choice adds up!"
        )
        return " ".join(parts) if parts else (
            "Thanks for sharing about your meal! "
            "Keep making those healthy choices — every small step counts!"
        )

    # ── 프로퍼티 ──────────────────────────────────────────────────────────

    @property
    def last_decision(self) -> Optional[Dict]:
        """가장 최근 결정."""
        return self._last_decision

    @property
    def decision_history(self) -> List[Dict]:
        """전체 결정 이력."""
        return list(self._decision_history)

    @property
    def last_assessment(self) -> Optional[Dict]:
        """가장 최근 Assessment 결과."""
        return self._last_assessment
