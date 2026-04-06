"""
models/meal_recommender.py
──────────────────────────
LLM 기반 Meal Recommender 모델.

역할
  - 현재 식사를 영양 목표에 더 가깝게 개선하기 위한 구체적 추천 템플릿을 생성합니다.
  - Orchestrator 의 instruction 과 사용자 선호도/제약 조건을 참고하여 추천 방향을 결정합니다.
  - Orchestrator 가 이 템플릿을 최종 사용자 대면 텍스트로 변환합니다.

입력
  - meal_fact_sheet     : MealTracker 가 추출한 식사 구성 정보
  - alignment_score     : AlignmentEstimator 의 최근 점수
  - alignment_reasoning : AlignmentEstimator 의 최근 reasoning
  - nutrition_goal      : 영양 목표
  - goal_definition     : 목표 정의 (goal_def.json)
  - instruction         : Orchestrator 로부터의 추천 방향 가이드
  - user_preferences    : 사용자 선호도/알레르기/제약 (Memorizer 또는 대화에서 수집)

출력 (JSON)
  - recommendation_type : "substitute" | "modify" | "add"
  - target_food         : 개선 대상 음식
  - suggestion          : 구체적 개선 제안
  - reasoning           : 추천 근거
  - expected_impact     : "high" | "medium" | "low"
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from config import SimulationConfig


# ──────────────────────────────────────────────────────────────────────────────
# 데이터 디렉터리 경로
# ──────────────────────────────────────────────────────────────────────────────

_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "additional"

_GOAL_DEF_CACHE: Optional[Dict] = None


def _load_goal_definitions() -> Dict:
    global _GOAL_DEF_CACHE
    if _GOAL_DEF_CACHE is None:
        path = _DATA_DIR / "goal_def_v2.json"
        with open(path, "r", encoding="utf-8") as f:
            _GOAL_DEF_CACHE = json.load(f)
    return _GOAL_DEF_CACHE


# ──────────────────────────────────────────────────────────────────────────────
# 시스템 프롬프트
# ──────────────────────────────────────────────────────────────────────────────

_RECOMMENDER_SYSTEM = """\
You are a nutritional meal improvement advisor.

Given:
- Current meal composition (Fact Sheet)
- Nutritional goal: {nutrition_goal}
- Goal definition: {goal_definition}
- Alignment assessment: score = {alignment_score}, reasoning = "{alignment_reasoning}"

Your task:
1. Identify the specific food items or preparation methods causing misalignment.
2. Suggest the MINIMAL change that would bring the meal closer to the goal.
3. Prefer realistic substitutions over drastic changes.
4. Respect the user's apparent preferences — if they chose chicken, suggest a different \
chicken preparation rather than switching to fish.

Rules:
- ONE recommendation per call.
- Be specific: "grill instead of fry" > "use a healthier cooking method".
- Never suggest adding entirely new food groups the user did not mention.
- Output ONLY a valid JSON object with exactly these fields:
  "recommendation_type": "substitute" | "modify" | "add",
  "target_food": the specific food item to change,
  "suggestion": the concrete improvement,
  "reasoning": why this change helps meet the goal,
  "expected_impact": "high" | "medium" | "low"
- Do not add extra keys, markdown, or surrounding text.

{instruction_block}\
{preferences_block}\
{recommendation_history_block}\
{user_feedback_block}\
"""

_RECOMMENDER_USER = """\
[Meal Fact Sheet]
{meal_fact_sheet}

Based on the above, provide your recommendation.\
"""


# ──────────────────────────────────────────────────────────────────────────────
# MealRecommender
# ──────────────────────────────────────────────────────────────────────────────

class MealRecommender:
    """
    LLM 기반 식사 개선 추천 모델.

    Parameters
    ----------
    nutrition_goal : "lean_protein" | "half_fruits_vegetables" | "one_fourth_carbs" | "drink_water"
    config         : SimulationConfig 인스턴스
    """

    def __init__(self, nutrition_goal: str, config: "SimulationConfig"):
        self.nutrition_goal = nutrition_goal
        self.config = config

        goal_spec = _load_goal_definitions().get(nutrition_goal, {})
        self._goal_definition = goal_spec.get("definition", "")

        # 추천 이력
        self._recommendation_history: List[Dict] = []
        self._last_recommendation: Optional[Dict] = None

    # ── 공개 인터페이스 ────────────────────────────────────────────────────

    def get_messages(
        self,
        meal_fact_sheet: str,
        alignment_score: float,
        alignment_reasoning: str,
        instruction: str = "",
        user_preferences: str = "",
        recommendation_history: Optional[List[Dict]] = None,
        user_feedback: str = "",
    ) -> List[Dict[str, str]]:
        """
        LLM 호출용 messages 리스트를 반환합니다.

        Parameters
        ----------
        meal_fact_sheet     : MealTracker가 추출한 Fact Sheet
        alignment_score     : AlignmentEstimator의 최근 점수
        alignment_reasoning : AlignmentEstimator의 최근 reasoning
        instruction         : Orchestrator로부터의 추천 방향 가이드
        user_preferences    : 사용자 선호도/알레르기/제약 조건 컨텍스트
        recommendation_history : 이전 추천 이력 리스트 (중복/번복 방지용)
        user_feedback       : 이전 추천에 대한 사용자 반응 컨텍스트
        """
        instruction_block = ""
        if instruction:
            instruction_block = (
                f"\n[Orchestrator guidance]\n{instruction}\n"
            )

        preferences_block = ""
        if user_preferences:
            preferences_block = (
                f"\n[User preferences & constraints]\n{user_preferences}\n"
                "Respect these when making suggestions. "
                "Do not recommend anything the user cannot eat.\n"
            )

        recommendation_history_block = ""
        if recommendation_history:
            rec_lines = [
                f"  Turn {r.get('turn_idx', '?')}: "
                f"{r.get('recommendation_type', '?')} — "
                f"{r.get('target_food', '?')} → {r.get('suggestion', '?')}"
                for r in recommendation_history
            ]
            recommendation_history_block = (
                "\n[Previous Recommendations]\n"
                + "\n".join(rec_lines)
                + "\nDo NOT repeat these recommendations. "
                "If the user accepted a previous suggestion, "
                "honor that choice and build upon it instead of contradicting it.\n"
            )

        user_feedback_block = ""
        if user_feedback:
            user_feedback_block = (
                f"\n[User Feedback on Previous Recommendations]\n{user_feedback}\n"
                "Take this feedback into account. "
                "If the user accepted a suggestion, do NOT reverse or escalate it. "
                "If the user rejected it, offer a different alternative.\n"
            )

        system = _RECOMMENDER_SYSTEM.format(
            nutrition_goal=self.nutrition_goal.replace("_", " "),
            goal_definition=self._goal_definition,
            alignment_score=f"{alignment_score:.2f}",
            alignment_reasoning=alignment_reasoning or "N/A",
            instruction_block=instruction_block,
            preferences_block=preferences_block,
            recommendation_history_block=recommendation_history_block,
            user_feedback_block=user_feedback_block,
        )
        user = _RECOMMENDER_USER.format(
            meal_fact_sheet=meal_fact_sheet or "(no fact sheet available)",
        )
        return [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]

    def parse_output(self, raw_output: str, turn_idx: int = 0) -> Dict:
        """
        LLM의 raw JSON 출력을 파싱하여 추천 딕셔너리를 반환합니다.
        파싱 실패 시 safe default를 반환합니다.
        """
        recommendation = {
            "recommendation_type": "modify",
            "target_food": "",
            "suggestion": "",
            "reasoning": "(parse error)",
            "expected_impact": "low",
        }

        try:
            text = raw_output.strip()
            json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(text)

            recommendation = {
                "recommendation_type": str(data.get("recommendation_type", "modify")),
                "target_food": str(data.get("target_food", "")),
                "suggestion": str(data.get("suggestion", "")),
                "reasoning": str(data.get("reasoning", "")),
                "expected_impact": str(data.get("expected_impact", "low")),
            }
        except (json.JSONDecodeError, ValueError, TypeError):
            recommendation["reasoning"] = f"(parse error) raw: {raw_output[:200]}"

        self._last_recommendation = recommendation
        self._recommendation_history.append({
            "turn_idx": turn_idx,
            **recommendation,
        })
        return recommendation

    def recommend(
        self,
        meal_fact_sheet: str,
        alignment_score: float,
        alignment_reasoning: str,
        instruction: str = "",
        user_preferences: str = "",
        recommendation_history: Optional[List[Dict]] = None,
        user_feedback: str = "",
        generate_fn=None,
        llm=None,
        turn_idx: int = 0,
    ) -> Dict:
        """
        편의 메서드: 메시지 조립 → LLM 호출 → 파싱까지 한 번에 수행합니다.
        """
        if generate_fn is None:
            from utils.llm_utils import generate_response
            generate_fn = generate_response

        msgs = self.get_messages(
            meal_fact_sheet=meal_fact_sheet,
            alignment_score=alignment_score,
            alignment_reasoning=alignment_reasoning,
            instruction=instruction,
            user_preferences=user_preferences,
            recommendation_history=recommendation_history,
            user_feedback=user_feedback,
        )
        raw = generate_fn(
            llm,
            msgs,
            max_new_tokens=getattr(self.config, 'recommendation_max_new_tokens', 300),
            sampling="greedy",
        )
        return self.parse_output(raw, turn_idx=turn_idx)

    # ── 프로퍼티 ──────────────────────────────────────────────────────────

    @property
    def last_recommendation(self) -> Optional[Dict]:
        """가장 최근 추천 결과. 추천한 적 없으면 None."""
        return self._last_recommendation

    @property
    def recommendation_history(self) -> List[Dict]:
        """전체 추천 이력."""
        return list(self._recommendation_history)
