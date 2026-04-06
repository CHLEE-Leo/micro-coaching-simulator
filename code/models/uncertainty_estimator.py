"""
models/uncertainty_estimator.py
─────────────────────────────
Uncertainty Estimator Agent — Coach 관점에서 meal-goal alignment 확신도를 추정합니다.

역할
  - 매 턴 종료 시, Coach-User 간 대화 이력과 Coach의 질문 전략을 종합하여
    현재 dialogue state에서 meal-goal alignment에 대한 certainty score (0–1)를
    추론합니다.
  - certainty가 충분히 높으면 (≥ threshold), 더 이상 질문이 불필요하다고 판단하여
    대화 종료 조건으로 사용됩니다.

데이터 흐름
  SharedConversationHistory → get_messages() → LLM 추론
    → JSON {"reasoning": "...", "certainty_score": 0.XX}
    → session_manager 가 score 를 확인하여 종료 여부 결정

커스터마이징
  - CERTAINTY_THRESHOLD : 대화 종료 임계값 (기본 0.85)
  - _UNCERTAINTY_SYSTEM : 추론 프롬프트를 수정하여 평가 기준 조정
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from core.memory import SharedConversationHistory
    from config import SimulationConfig

# ──────────────────────────────────────────────────────────────────────────────
# 기본 certainty 임계값
# ──────────────────────────────────────────────────────────────────────────────
CERTAINTY_THRESHOLD = 0.85

# ──────────────────────────────────────────────────────────────────────────────
# 시스템 프롬프트
# ──────────────────────────────────────────────────────────────────────────────
_UNCERTAINTY_SYSTEM = """\
You are a dialogue-state uncertainty estimator for a nutritional micro-coaching conversation.

Your task: Given the conversation so far between a Coach and a User about a planned meal, \
assess how CERTAIN you are that ENOUGH INFORMATION has been gathered to make a confident \
judgment about whether the meal aligns with the user's nutritional goal — regardless of \
whether that judgment would be "aligned" or "not aligned".

Nutritional goal: {nutrition_goal}

Key principle:
  Certainty is about INFORMATION SUFFICIENCY, not about the alignment direction.
  - High certainty means: "I have a clear enough picture of this meal to confidently judge \
alignment — whether the answer turns out to be aligned or not aligned."
  - Low certainty means: "Critical details are still missing, so I cannot confidently judge \
alignment in either direction."

Think step-by-step:
1. Review what food items, ingredients, preparation methods, and portions have been discussed.
2. Identify what is still UNKNOWN or AMBIGUOUS that would be relevant to judging goal alignment.
3. Consider: if a nutritionist were to evaluate this meal against the goal right now, \
would they have enough information to make a confident judgment — in either direction?

Output ONLY a JSON object with exactly two fields:
- "reasoning": a brief (2-4 sentences) explanation of what is known vs unknown.
- "certainty_score": a float between 0.0 and 1.0 where:
    0.0 = no useful information gathered yet
    0.5 = some details known but critical gaps remain (cannot judge either way)
    0.85+ = enough information to confidently assess goal alignment (whether aligned or not)
    1.0 = complete picture, nothing more to ask

Example output:
{{"reasoning": "The user mentioned a large pepperoni pizza with extra cheese. We know the dish, its main ingredients, and approximate portion. A nutritionist could confidently determine this does NOT align with a lean protein goal. No additional details would change this judgment.", "certainty_score": 0.90}}

Rules:
- Base your assessment ONLY on information explicitly stated in the conversation.
- Do NOT assume details that were not discussed.
- Be calibrated: early turns with minimal detail should yield low scores.
- A meal that clearly violates the goal can still produce HIGH certainty — as long as there is \
enough information to make that judgment confidently.
- If a previous certainty score is provided, your reasoning MUST explain why the current score \
differs from (or remains the same as) the previous score. Describe what new information from the \
latest conversation turn caused the score to increase, decrease, or stay the same.
- If no previous score is provided (first evaluation), base your reasoning solely on the current evidence.
- Output valid JSON only — no extra text before or after.\
"""

_UNCERTAINTY_USER = """\
[Conversation transcript]
{transcript}
{prev_score_context}
Based on the conversation above, estimate the certainty score for meal-goal alignment assessment.\
"""


class UncertaintyEstimator:
    """
    Coach 관점의 meal-goal alignment 불확실성 추정기.

    get_messages(history) → List[Dict]  (LLM 호출 메시지)
    parse_output(raw)     → (reasoning, certainty_score)
    """

    def __init__(self, nutrition_goal: str, config: "SimulationConfig"):
        self.nutrition_goal = nutrition_goal
        self.config = config
        self.last_reasoning: Optional[str] = None
        self.last_score: Optional[float] = None

    # ──────────────────────────────────────────────────────────────────────
    # 메시지 조립
    # ──────────────────────────────────────────────────────────────────────
    def get_messages(
        self,
        history: "SharedConversationHistory",
    ) -> List[Dict[str, str]]:
        """LLM 에 전달할 메시지를 조립합니다.
        Alignment Tracker 와 동일하게 meal_fact_sheet 이 포함된 컨텍스트를 사용합니다."""
        transcript = history.to_alignment_context()

        # 이전 턴 점수 컨텍스트 생성
        if self.last_score is not None:
            prev_score_context = (
                f"\n[previous certainty score]\n"
                f"The certainty score from the previous turn was {self.last_score:.2f}.\n"
                f"In your reasoning, you MUST explain why the score changed, decreased, increased, "
                f"or stayed the same compared to this previous score.\n"
            )
        else:
            prev_score_context = ""  # 첫 턴: 이전 점수 없음

        system = _UNCERTAINTY_SYSTEM.format(nutrition_goal=self.nutrition_goal)
        user = _UNCERTAINTY_USER.format(transcript=transcript, prev_score_context=prev_score_context)
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    # ──────────────────────────────────────────────────────────────────────
    # 출력 파싱
    # ──────────────────────────────────────────────────────────────────────
    def parse_output(self, raw_output: str) -> Tuple[str, float]:
        """
        LLM 의 raw JSON 출력을 파싱하여 (reasoning, certainty_score) 를 반환합니다.
        파싱 실패 시 safe default (reasoning="parse error", score=0.0) 를 반환합니다.
        """
        reasoning = ""
        score = 0.0

        try:
            # JSON 블록 추출 (마크다운 코드 블록 대응)
            text = raw_output.strip()
            json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(text)
            reasoning = str(data.get("reasoning", ""))
            raw_score = float(data.get("certainty_score", 0.0))
            score = max(0.0, min(1.0, raw_score))  # clamp to [0, 1]
        except (json.JSONDecodeError, ValueError, TypeError):
            reasoning = f"(parse error) raw output: {raw_output[:200]}"
            score = 0.0

        self.last_reasoning = reasoning
        self.last_score = score
        return reasoning, score

    # ──────────────────────────────────────────────────────────────────────
    # 편의 메서드: 추정 수행 (배치 모드용)
    # ──────────────────────────────────────────────────────────────────────
    def estimate(
        self,
        history: "SharedConversationHistory",
        generate_fn=None,
        llm=None,
        config=None,
    ) -> Tuple[str, float]:
        """
        배치 모드용 편의 메서드.
        인터랙티브 모드에서는 session_manager 가 get_messages() + external call 을 사용합니다.
        """
        if generate_fn is None:
            from utils.llm_utils import generate_response
            generate_fn = generate_response

        msgs = self.get_messages(history)
        raw = generate_fn(
            llm,
            msgs,
            max_new_tokens=config.certainty_max_new_tokens if config else 200,
            sampling="greedy",
        )
        return self.parse_output(raw)
