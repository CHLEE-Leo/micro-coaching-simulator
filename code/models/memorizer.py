"""
models/memorizer.py
───────────────────
장기 사용자 프로필 메모리 에이전트.

역할
  - 대화 중 수집된 사용자 정보(선호도, 알레르기, 종교적 제약, 식품 가용성 등)를
    세션 간에 유지합니다.
  - Multi-meal 세션(아침→점심→저녁)에서 이전 식사 세션의 요약을 관리합니다.
  - Orchestrator 와 MealRecommender 에게 사용자 컨텍스트를 제공합니다.

데이터 흐름
  대화 → Memorizer.extract() → user_profile 업데이트
  이전 세션 → Memorizer.get_cross_session_summary() → Orchestrator 컨텍스트

저장 구조 (in-memory dict)
  user_profile:
    - preferences    : 식품 선호도 (예: "likes spicy food")
    - allergies      : 알레르기 목록 (예: ["peanut", "shellfish"])
    - restrictions   : 식이 제한 (예: "vegetarian", "halal")
    - availability   : 식품 가용성 (예: "no oven at work")
    - past_meals     : 이전 식사 요약 리스트
"""

from __future__ import annotations

from typing import Dict, List, Optional


class Memorizer:
    """
    장기 사용자 프로필 메모리.

    Multi-meal 세션에서 식사 간 사용자 정보를 기억하고,
    Orchestrator / MealRecommender 에게 컨텍스트를 제공합니다.
    """

    def __init__(self):
        self._profile: Dict[str, any] = {
            "preferences": [],
            "allergies": [],
            "restrictions": [],
            "availability": [],
            "past_meals": [],
        }

    # ── 프로필 읽기 ───────────────────────────────────────────────────────

    @property
    def profile(self) -> Dict:
        """현재 사용자 프로필을 반환합니다."""
        return dict(self._profile)

    def get_preferences_text(self) -> str:
        """
        사용자 선호도/제약 조건을 텍스트로 반환합니다.
        MealRecommender 의 user_preferences 파라미터에 전달하기 위한 용도.
        """
        lines = []
        if self._profile["preferences"]:
            lines.append("Preferences: " + ", ".join(self._profile["preferences"]))
        if self._profile["allergies"]:
            lines.append("Allergies: " + ", ".join(self._profile["allergies"]))
        if self._profile["restrictions"]:
            lines.append("Restrictions: " + ", ".join(self._profile["restrictions"]))
        if self._profile["availability"]:
            lines.append("Availability: " + ", ".join(self._profile["availability"]))
        return "\n".join(lines) if lines else ""

    def get_cross_session_summary(self) -> str:
        """
        이전 식사 세션 요약을 텍스트로 반환합니다.
        Orchestrator 의 컨텍스트에 주입하기 위한 용도.
        """
        if not self._profile["past_meals"]:
            return ""
        lines = []
        for i, meal in enumerate(self._profile["past_meals"], 1):
            meal_type = meal.get("meal_type", "meal")
            summary = meal.get("summary", "")
            lines.append(f"  {i}. {meal_type}: {summary}")
        return "Previous meals in this session:\n" + "\n".join(lines)

    # ── 프로필 업데이트 ───────────────────────────────────────────────────

    def update_preferences(self, items: List[str]) -> None:
        """선호도 항목을 추가합니다 (중복 제거)."""
        for item in items:
            if item and item not in self._profile["preferences"]:
                self._profile["preferences"].append(item)

    def update_allergies(self, items: List[str]) -> None:
        """알레르기 항목을 추가합니다 (중복 제거)."""
        for item in items:
            if item and item not in self._profile["allergies"]:
                self._profile["allergies"].append(item)

    def update_restrictions(self, items: List[str]) -> None:
        """식이 제한 항목을 추가합니다 (중복 제거)."""
        for item in items:
            if item and item not in self._profile["restrictions"]:
                self._profile["restrictions"].append(item)

    def update_availability(self, items: List[str]) -> None:
        """식품 가용성 항목을 추가합니다 (중복 제거)."""
        for item in items:
            if item and item not in self._profile["availability"]:
                self._profile["availability"].append(item)

    def add_past_meal(self, meal_type: str, summary: str, fact_sheet: str = "") -> None:
        """이전 식사 세션 요약을 추가합니다."""
        self._profile["past_meals"].append({
            "meal_type": meal_type,
            "summary": summary,
            "fact_sheet": fact_sheet,
        })

    def set_profile_from_persona(
        self,
        preferences: List[str] | None = None,
        allergies: List[str] | None = None,
        restrictions: List[str] | None = None,
    ) -> None:
        """
        UI 의 페르소나 설정에서 사용자 프로필을 초기화합니다.
        """
        if preferences:
            self._profile["preferences"] = list(preferences)
        if allergies:
            self._profile["allergies"] = list(allergies)
        if restrictions:
            self._profile["restrictions"] = list(restrictions)

    def reset(self) -> None:
        """프로필을 초기화합니다."""
        self._profile = {
            "preferences": [],
            "allergies": [],
            "restrictions": [],
            "availability": [],
            "past_meals": [],
        }
