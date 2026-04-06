"""
models/meal_tracker.py
──────────────────────
식사 정보 추출 · 누적 에이전트 (Meal Tracker Agent).

역할
  - Coach ↔ User 대화에서 드러난 식사 정보를 **구조화된 Meal Fact Sheet**
    형태로 추출하고, 턴이 쌓일수록 누적합니다.
  - 생성된 Meal Fact Sheet 는 SharedConversationHistory.meal_fact_sheet 에 저장되어
    주로 **Alignment Tracker** 의 판정 입력으로 사용됩니다.

  왜 필요한가?
    User Agent 는 매 턴 partial information 만 제공합니다.
    (예: T1="grilled chicken salad", T2="chicken breast, romaine lettuce",
         T3="grilled, no oil", T4="about 6oz chicken, 2 cups lettuce")
    Alignment Tracker 가 영양 목표 달성 여부를 정확히 판정하려면, 여러 턴에 걸쳐 산재된
    식사 정보를 하나의 정리된 meal description 으로 누적해야 합니다.
    Meal Tracker 가 이 "정보 누적 + 구조화" 역할을 수행하여, Alignment Tracker 에게
    expert_workflow Step 1("Scan through each ingredient") 에 바로 사용할 수 있는
    입력을 제공합니다.

  데이터 흐름:
    대화 원문 → MealTrackerModel.extract()
      → Meal Fact Sheet (구조화된 텍스트)
      → SharedConversationHistory.meal_fact_sheet 에 저장
      → history.to_alignment_context() 에서 [Meal Fact Sheet] 블록으로 Alignment Tracker 에 주입

  Dialog Summarizer (dialog_summarizer.py) 와의 차이:
    ┌─────────────────────┬──────────────────────┐
    │   Meal Tracker      │  Dialog Summarizer   │
    ├─────────────────────┼──────────────────────┤
    │ 식사 정보 구조화     │  대화 흐름 서술형 요약 │
    │ 주 소비자: Alignment Tracker    │  주 소비자: Coach/User │
    │ 출력: Fact Sheet    │  출력: 서술형 요약     │
    │ "무엇을 먹는가"     │  "무엇을 물었/답했는가" │
    └─────────────────────┴──────────────────────┘

두 가지 추출 모드
  [전체 추출]  conversation_text 전체에서 Fact Sheet 생성
    배치 시뮬레이션 (code/core/simulation.py) 에서 주로 사용합니다.

  [증분 추출]  이전 Fact Sheet + 신규 턴만으로 Fact Sheet 갱신
    인터랙티브 모드 (code_interactive/session_manager.py) 에서 주로 사용합니다.

커스터마이징
  - _TRACKER_SYSTEM_FULL / _TRACKER_SYSTEM_INCREMENTAL
    추출 항목(Food items, Ingredients 등)이나 출력 형식을 바꾸려면 수정하세요.
  - config.summarize_max_new_tokens : 추출 최대 생성 토큰 수.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List

from utils.llm_utils import generate_response

if TYPE_CHECKING:
    from config import SimulationConfig


# ──────────────────────────────────────────────────────────────────────────────
# 시스템 프롬프트
# ──────────────────────────────────────────────────────────────────────────────

_TRACKER_SYSTEM_FULL = (
    "You are a meal-information extractor. "
    "From the conversation below, extract and organize ALL meal details into a structured fact sheet.\n"
    "\n"
    "Output format (use these exact headers):\n"
    "- Food items: [comma-separated list of distinct dishes mentioned, e.g. ham sandwich, Caesar salad]\n"
    "- Ingredients: [comma-separated list of ALL individual ingredients that make up the food items, "
    "including bread/buns, condiments, sauces, dressings, spreads, seasonings, and toppings]\n"
    "- Preparation methods: [how each item is cooked or prepared, e.g. grilled, fried, raw]\n"
    "- Portions/amounts: [any quantities, sizes, or proportions mentioned]\n"
    "- Beverages: [any drinks mentioned, or 'none mentioned']\n"
    "- Additional notes: [any other nutritionally relevant details, e.g. sides, garnishes]\n"
    "\n"
    "Rules:\n"
    "- Only include food items and ingredients that the user has EXPLICITLY CONFIRMED eating or planning to eat.\n"
    "- Do NOT add items that are merely mentioned in a question by the coach but not confirmed by the user.\n"
    "- If the user responds with uncertainty (e.g. 'I haven't decided yet', 'I'm not sure', 'maybe'), "
    "do NOT add those items to the fact sheet.\n"
    "- Do NOT repeat the same item under multiple headers. "
    "A dish name (e.g. 'ham sandwich') goes under Food items; "
    "its components (e.g. 'ham', 'bread', 'mayo') go under Ingredients.\n"
    "- Pay special attention to condiments, sauces, and spreads — "
    "these are often nutritionally significant and must be listed under Ingredients.\n"
    "- If a detail has not been discussed yet, write 'not yet mentioned'.\n"
    "- Be factual — do not infer or assume details not stated in the conversation.\n"
    "- Keep each line concise but complete."
)

_TRACKER_SYSTEM_INCREMENTAL = (
    "You are an incremental meal-information extractor. "
    "You will be given a previous Meal Fact Sheet and new conversation turns that occurred after it. "
    "Produce an UPDATED Meal Fact Sheet that incorporates all previous information plus any new details.\n"
    "\n"
    "Output format (use these exact headers):\n"
    "- Food items: [comma-separated list of distinct dishes mentioned, e.g. ham sandwich, Caesar salad]\n"
    "- Ingredients: [comma-separated list of ALL individual ingredients that make up the food items, "
    "including bread/buns, condiments, sauces, dressings, spreads, seasonings, and toppings]\n"
    "- Preparation methods: [how each item is cooked or prepared]\n"
    "- Portions/amounts: [any quantities, sizes, or proportions mentioned]\n"
    "- Beverages: [any drinks mentioned, or 'none mentioned']\n"
    "- Additional notes: [any other nutritionally relevant details]\n"
    "\n"
    "Rules:\n"
    "- Preserve ALL details from the previous fact sheet — only add or update, never remove.\n"
    "- Only add NEW items that the user has EXPLICITLY CONFIRMED eating or planning to eat in the new turns.\n"
    "- Do NOT add items that are merely mentioned in a question by the coach but not confirmed by the user.\n"
    "- If the user responds with uncertainty (e.g. 'I haven't decided yet', 'I'm not sure', 'maybe'), "
    "do NOT add those items.\n"
    "- If a previously 'not yet mentioned' field now has confirmed information, update it.\n"
    "- Do NOT duplicate items: if 'ham' is already in Ingredients, do not add it again.\n"
    "- Food items = dish names (e.g. 'ham sandwich'); Ingredients = components (e.g. 'ham', 'bread', 'mayo'). "
    "Do not put the same item under both headers.\n"
    "- Be factual — do not infer or assume details not stated."
)


# ──────────────────────────────────────────────────────────────────────────────
# MealTrackerModel
# ──────────────────────────────────────────────────────────────────────────────

class MealTrackerModel:
    """
    LLM 기반 식사 정보 추출 에이전트.

    대화에서 드러난 음식·재료·조리법·양 등을 구조화된 Meal Fact Sheet 로
    누적 추출합니다. 주로 Alignment Tracker 의 판정 입력으로 사용됩니다.

    인터페이스 패턴 (Coach / User / Alignment Tracker 와 동일):
      - get_messages()  : LLM 호출 없이 messages 리스트만 반환 (배치 생성용)
      - extract()       : messages 를 빌드하고 LLM 을 호출하여 Fact Sheet 반환

    Parameters
    ----------
    model  : vLLM LLM 또는 llama_cpp.Llama 객체
    config : SimulationConfig 인스턴스
    """

    def __init__(self, model, config: "SimulationConfig"):
        self.model  = model
        self.config = config

    # ── 공개 인터페이스 ────────────────────────────────────────────────────

    def get_messages(
        self,
        conversation_text: str,
        prev_fact_sheet: str = "",
    ) -> List[Dict[str, str]]:
        """
        배치 생성용: LLM 을 호출하지 않고 messages 리스트만 반환합니다.

        Parameters
        ----------
        conversation_text : "Coach: ...\nUser: ..." 형식의 대화 텍스트.
                            전체 추출 시 전체 대화, 증분 추출 시 신규 턴만 전달합니다.
        prev_fact_sheet   : 이전에 생성된 Meal Fact Sheet (빈 문자열이면 전체 추출 모드).

        Returns
        -------
        List[Dict[str, str]] : chat-template 형식의 messages
        """
        if prev_fact_sheet and conversation_text:
            # ── 증분 추출: 기존 Fact Sheet + 신규 턴 → 갱신된 Fact Sheet ──
            return [
                {"role": "system", "content": _TRACKER_SYSTEM_INCREMENTAL},
                {
                    "role": "user",
                    "content": (
                        f"Previous Meal Fact Sheet:\n{prev_fact_sheet}\n\n"
                        f"New conversation turns since the previous fact sheet:\n\n"
                        f"{conversation_text}\n\n"
                        "Now write the updated Meal Fact Sheet:"
                    ),
                },
            ]
        else:
            # ── 전체 추출: 대화 전문 → Meal Fact Sheet ──
            return [
                {"role": "system", "content": _TRACKER_SYSTEM_FULL},
                {
                    "role": "user",
                    "content": (
                        "Conversation to extract meal information from:\n\n"
                        f"{conversation_text}\n\n"
                        "Now write the Meal Fact Sheet:"
                    ),
                },
            ]

    def extract(
        self,
        conversation_text: str,
        prev_fact_sheet: str = "",
    ) -> str:
        """
        대화 텍스트에서 식사 정보를 추출하여 Meal Fact Sheet 를 반환합니다.

        Parameters
        ----------
        conversation_text : "Coach: ...\nUser: ..." 형식의 대화 텍스트
        prev_fact_sheet   : 이전 Fact Sheet (빈 문자열이면 전체 추출 모드)

        Returns
        -------
        str : 구조화된 Meal Fact Sheet.
              Alignment Tracker 의 시스템 프롬프트에 [Meal Fact Sheet] 블록으로 주입됩니다.
        """
        messages = self.get_messages(conversation_text, prev_fact_sheet)
        return generate_response(
            self.model,
            messages,
            sampling="greedy",
            max_new_tokens=self.config.summarize_max_new_tokens,
            stop_at_newline=False,
        )

