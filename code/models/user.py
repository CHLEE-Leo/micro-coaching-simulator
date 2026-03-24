"""
models/user.py
──────────────
LLM 기반 User 모델.

역할
  - 영양 목표와 식사 설명을 입력받아 코치 질문에 자연스럽게 응답합니다.
  - 식사 정보를 한 번에 다 공개하지 않고 점진적으로(partial/gradual) 제공합니다.
  - 모든 정보가 전달되었다고 판단되면 "That's all about my meal." 을 붙입니다.

설계 원칙 대응
  - Principle 2 : own_buffer (ConversationBuffer) 로 User 자신의 응답 이력을 관리.
  - Principle 3 : SharedConversationHistory.context_window 를 통해 최근 N 턴만 참조.
  - Principle 4 : system prompt 에 conversation summary 를 주입.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Dict

from core.memory import ConversationBuffer, SharedConversationHistory
from utils.llm_utils import generate_response

if TYPE_CHECKING:
    from config import SimulationConfig


# ──────────────────────────────────────────────────────────────────────────────
# 시스템 프롬프트 템플릿
# ──────────────────────────────────────────────────────────────────────────────

_USER_SYSTEM_BASE = """\
You are simulating a real user chatting with a nutritional coaching chatbot.

Your nutritional goal              : {nutrition_goal}
Your planned meal (food names)     : {meal_description}{meal_ingredient_block}

Two-phase conversation rules:

PHASE 1 — First response only (see NOTE below if applicable):
  When the coach first asks what you are having, name ALL the dishes and drinks in your planned meal.

PHASE 2 — All subsequent responses (turn 1 onward):
  The coach will ask about the ingredient and preparation details of each food, one at a time.
  - Answer ONLY about the specific food or aspect the coach just asked about.
  - Your ONLY source of truth for ingredient/preparation details is "Ingredient details" above.
    If a detail is NOT written there, say you haven’t decided yet (e.g. "I haven’t decided" / “I’m not sure”). NEVER infer, assume, or fabricate any detail.
  - Do NOT volunteer ingredient information about OTHER foods not currently being asked about.
  - Do NOT repeat information you have already provided (see your own history below).

General rules (all turns):
- A "food item" is a distinct dish or drink (e.g. "turkey sandwich" is ONE item). NEVER split a dish name into separate words.
- If the coach asks about brand, exact weight, or nutrition facts not in your fields, say "I'm not sure" / "Just a standard portion" — NEVER fabricate.
- Keep every response short and natural (1 sentence maximum).
- Never mention your nutritional goal to the coach.
- Never reveal that you were given a meal description.
- IMPORTANT: ONLY say \"That\'s all about my meal.\" when ALL of the following are true:
  (a) Every food item has come up in the conversation, AND
  (b) You genuinely have nothing new to add about any item’s ingredients or preparation.
  NEVER say it as a direct answer to a specific question.  ALWAYS write a short natural sentence first, THEN append "That's all about my meal." at the end.
  Example: "I think that covers everything about my meal. That's all about my meal."
  NEVER output "That's all about my meal." as the sole content of your response.- Always respond in English.
"""

_USER_SUMMARY_BLOCK = """\

[Conversation summary so far]
{summary}
"""

_USER_OWN_BUFFER_BLOCK = """\

[Details you have ALREADY told the coach — DO NOT repeat these, add only NEW information]
{own_buffer}
"""


# ──────────────────────────────────────────────────────────────────────────────
# UserModel
# ──────────────────────────────────────────────────────────────────────────────

class UserModel:
    """
    LLM 기반 시뮬레이션 사용자 모델.

    Parameters
    ----------
    model            : vLLM LLM 객체
    nutrition_goal   : "lean_protein" | "half_fruits_vegetables" | "one_fourth_carbs"
    meal_description : 응답의 근거가 되는 식사 설명 (예: "grilled chicken, brown rice, salad")
    meal_ingredient  : 식사를 구성하는 세부 재료 목록 (없으면 빈 문자열)
    config           : SimulationConfig 인스턴스
    """

    def __init__(
        self,
        model,
        nutrition_goal:   str,
        meal_description: str,
        config: "SimulationConfig",
        meal_ingredient:  str = "",
    ):
        self.model            = model
        self.nutrition_goal   = nutrition_goal
        self.meal_description = meal_description
        self.meal_ingredient  = meal_ingredient
        self.config           = config

        # User 자신의 답변 기록 (Principle 2)
        self.own_buffer = ConversationBuffer(role="user")

    # ── 공개 인터페이스 ────────────────────────────────────────────────────────

    def get_messages(
        self,
        shared_history: SharedConversationHistory,
    ) -> List[Dict[str, str]]:
        """
        배치 생성용: LLM 을 호출하지 않고 messages 리스트만 반환합니다.
        batch_generate() 에 넘길 때 사용합니다.

        Parameters
        ----------
        shared_history : Coach ↔ User 의 공통 대화 기록

        Returns
        -------
        List[Dict[str, str]] : chat-template 형식의 messages
        """
        # 아직 완료된 user 발화가 하나도 없으면 첫 번째 응답 (turn 0)
        is_first_turn = not any(t.user_utterance for t in shared_history._turns)
        system_prompt = self._build_system_prompt(shared_history.summary, is_first_turn=is_first_turn)
        return shared_history.build_messages(
            perspective="user",
            system_prompt=system_prompt,
        )

    def respond(
        self,
        shared_history: SharedConversationHistory,
    ) -> str:
        """
        현재 shared_history (코치의 최신 질문 포함) 를 바탕으로 응답을 LLM 으로 생성합니다.
        단일 대화 처리(single-dialog) 모드에서 사용합니다.

        Parameters
        ----------
        shared_history : Coach ↔ User 의 공통 대화 기록
                         (마지막에 Coach 질문이 추가된 상태여야 합니다)

        Returns
        -------
        str : 생성된 User 응답
        """
        messages = self.get_messages(shared_history)

        response = generate_response(
            self.model,
            messages,
            max_new_tokens=self.config.max_new_tokens,
            sampling=self.config.sampling,
        )

        self.own_buffer.add(response)
        return response

    # ── 내부 헬퍼 ──────────────────────────────────────────────────────────────

    def respond(
        self,
        shared_history: SharedConversationHistory,
    ) -> str:
        """
        현재 shared_history (코치의 최신 질문 포함) 를 바탕으로 응답을 LLM 으로 생성합니다.
        단일 대화 처리(single-dialog) 모드에서 사용합니다.

        Parameters
        ----------
        shared_history : Coach ↔ User 의 공통 대화 기록
                         (마지막에 Coach 질문이 추가된 상태여야 합니다)

        Returns
        -------
        str : 생성된 User 응답
        """
        messages = self.get_messages(shared_history)

        response = generate_response(
            self.model,
            messages,
            max_new_tokens=self.config.max_new_tokens,
            sampling=self.config.sampling,
        )

        self.own_buffer.add(response)
        return response

    # ── 내부 헬퍼 ──────────────────────────────────────────────────────────────

    def _build_system_prompt(self, summary: str, is_first_turn: bool = False) -> str:
        """
        User 의 시스템 프롬프트를 동적으로 조합합니다.

        구성 순서:
          1. 기본 역할 + 식사 설명
          2. (있으면) 대화 요약     (Principle 4)
          3. User 자신의 응답 이력  (Principle 2)
        """
        # meal_ingredient 가 있을 때만 별도 블록으로 삽입
        if self.meal_ingredient:
            ingredient_block = (
                "\nIngredient details                 : " + self.meal_ingredient
            )
        else:
            ingredient_block = ""

        base = _USER_SYSTEM_BASE.format(
            nutrition_goal=self.nutrition_goal,
            meal_description=self.meal_description,
            meal_ingredient_block=ingredient_block,
        )
        # turn 0: 처음 질문에는 식사 전체를 자유롭게 언급 가능
        if is_first_turn:
            base += (
                "\nNOTE — This is your FIRST response: you may mention ALL the items "
                "in your planned meal at once if you wish."
            )
        parts = [base]

        # 요약이 존재할 때만 삽입 (Principle 4)
        if summary:
            parts.append(_USER_SUMMARY_BLOCK.format(summary=summary))

        # 자신의 응답 이력 (Principle 2)
        parts.append(
            _USER_OWN_BUFFER_BLOCK.format(
                own_buffer=self.own_buffer.to_prompt_text(
                    header="Meal details you have already mentioned"
                )
            )
        )

        return "".join(parts)
