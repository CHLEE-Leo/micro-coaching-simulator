"""
models/user.py
──────────────
LLM 기반 User 모델.

역할
  - 영양 목표와 식사 설명을 입력받아 코치 질문에 자연스럽게 응답합니다.
  - 식사 정보를 점진적으로(gradual) 공개하고, 합리적으로 추측할 수 있는 부분은
    자연스러운 근사치를 제공합니다.
  - 페르소나(선호·알레르기·제약)가 주어지면 대화 스타일과 맥락에 반영합니다.
  - 모든 정보가 전달되면 자연스러운 종료 멘트와 함께 [END] 태그를 붙입니다.

설계 원칙 대응
  - P2  : own_buffer (ConversationBuffer) 로 User 자신의 응답 이력을 관리.
  - P3  : SharedConversationHistory.context_window 를 통해 최근 N 턴만 참조.
  - P4  : system prompt 에 conversation summary 를 주입.

커스터마이징
  - _USER_SYSTEM_BASE         : User 역할 정의 및 응답 규칙.
                                다른 도메인(의료 상담, 교육 등)의 사용자를
                                시뮬레이션하려면 이 문자열을 수정하세요.
  - TERMINATION_TOKEN         : 대화 종료 신호 태그 (기본 "[END]").
                                도메인에 맞게 변경할 수 있습니다.
                                (SharedConversationHistory 에 정의)
  - meal_description / meal_ingredient
                              : 시스템 프롬프트에 주입되는 사용자 식사 정보.
                                다른 시나리오에서는 이 필드를 교체하세요.
  - persona_*                 : 사용자 페르소나 (선호, 알레르기, 제약).
                                주어지면 대화 스타일에 자연스럽게 반영됩니다.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Dict, Optional

from core.memory import ConversationBuffer, SharedConversationHistory
from utils.llm_utils import generate_response

if TYPE_CHECKING:
    from config import SimulationConfig


# ──────────────────────────────────────────────────────────────────────────────
# 시스템 프롬프트 템플릿
# ──────────────────────────────────────────────────────────────────────────────

_USER_SYSTEM_BASE = """\
You are simulating a real person chatting with a nutritional coaching chatbot.

Your nutritional goal              : {nutrition_goal}
Your planned meal (food names)     : {meal_description}{meal_ingredient_block}
{persona_block}
Conversation rules:

FIRST RESPONSE:
  Casually mention 1-2 main items from your planned meal -- do NOT list every dish at once.
  A real person would say "I'm thinking chicken and some rice" rather than reciting a full menu.
  You will naturally bring up the remaining items later when the coach asks follow-ups.

SUBSEQUENT RESPONSES:
  The coach will ask about ingredients, preparation, portions, and so on.
  - Answer about the specific food or aspect the coach asked about.
  - Use your "Ingredient details" as a guide, but if a detail is missing there,
    give a reasonable approximate answer based on common sense
    (e.g. "probably grilled", "just a normal plate", "I think it's regular white rice").
    Only say "I'm not sure" or "I haven't decided" when the question is truly unknowable
    (brand names, exact gram weights, precise nutritional values, etc.).
  - If you haven't mentioned all your foods yet, naturally bring up remaining items when
    relevant (e.g. "Oh, I'm also having a side salad with that").
  - Do NOT repeat information you have already provided (see your own history below).

General rules (all turns):
- A "food item" is a distinct dish or drink (e.g. "turkey sandwich" is ONE item).
  NEVER split a dish name into separate words.
- When ingredient details include precise measurements (e.g. "1 cup", "200g"),
  express them approximately as a real person would: "about a cup", "a good amount",
  "a couple spoonfuls".
- Respond in 1-2 sentences. Occasionally add a brief personal remark, express a preference,
  or ask the coach a short question -- just as a real person would.
{persona_style_block}- Never mention your nutritional goal to the coach.
- Never reveal that you were given a meal description.
- WHEN THE COACH SUGGESTS A CHANGE OR RECOMMENDATION:
  - Consider whether the suggestion conflicts with your preferences, allergies, or restrictions.
  - If it clearly conflicts, express your concern naturally (e.g. "Hmm, I'm not really a fan of fish though",
    "That won't work for me — I'm allergic to nuts"). Be specific about why.
  - If it seems reasonable or you're open to trying, express that positively
    (e.g. "That sounds like a good idea!", "Sure, I could try that", "Oh interesting, I'll give it a shot").
  - If you're unsure, you can ask a clarifying question or express mild hesitation
    (e.g. "Would that change the taste much?", "I'm not sure about that one").
- When the conversation about your meal is naturally wrapping up -- you have discussed your
  main foods and the coach's latest question does not require genuinely new information --
  wrap up naturally (e.g. "Yeah, I think that's everything!", "That's pretty much my meal.")
  and append [END] at the very end of your message.
  Do this ONLY when all major items have already come up AND you have nothing new to add.
- Always respond in English.
"""

_PERSONA_BLOCK = """\
Your persona:
  Preferences  : {preferences}
  Allergies    : {allergies}
  Restrictions : {restrictions}
"""

_PERSONA_STYLE_BLOCK = """\
- Your persona influences how you talk: mention preferences or restrictions naturally
  when relevant (e.g. "I love spicy food so I added some hot sauce",
  "I can't eat dairy so I skipped the cheese"). Don't force it -- only when appropriate.
"""

_USER_SUMMARY_BLOCK = """\

[Dialog summary so far]
{dialog_summary}
"""

_USER_OWN_BUFFER_BLOCK = """\

[Details you have ALREADY told the coach -- DO NOT repeat these, add only NEW information]
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
    persona_preferences  : 사용자 선호 (예: ["spicy food", "Mediterranean cuisine"])
    persona_allergies    : 알레르기 (예: ["peanuts", "shellfish"])
    persona_restrictions : 식이 제약 (예: ["no dairy", "vegetarian"])
    """

    def __init__(
        self,
        model,
        nutrition_goal:   str,
        meal_description: str,
        config: "SimulationConfig",
        meal_ingredient:  str = "",
        persona_preferences:  Optional[List[str]] = None,
        persona_allergies:    Optional[List[str]] = None,
        persona_restrictions: Optional[List[str]] = None,
    ):
        self.model            = model
        self.nutrition_goal   = nutrition_goal
        self.meal_description = meal_description
        self.meal_ingredient  = meal_ingredient
        self.config           = config

        # 페르소나
        self.persona_preferences  = persona_preferences or []
        self.persona_allergies    = persona_allergies or []
        self.persona_restrictions = persona_restrictions or []

        # User 자신의 답변 기록 (P2)
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
        system_prompt = self._build_system_prompt(shared_history.dialog_summary, is_first_turn=is_first_turn)
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

        # own_buffer 에는 [END] 태그를 제거한 텍스트만 기록
        clean = response.replace(SharedConversationHistory.TERMINATION_TOKEN, "").strip()
        if clean:
            self.own_buffer.add(clean)
        return response

    # ── 내부 헬퍼 ──────────────────────────────────────────────────────────────

    def _build_system_prompt(self, dialog_summary: str, is_first_turn: bool = False) -> str:
        """
        User 의 시스템 프롬프트를 동적으로 조합합니다.

        구성 순서:
          1. 기본 역할 + 식사 설명 + (있으면) 페르소나
          2. (있으면) 대화 흐름 요약     (P4) — DialogSummarizer 출력
          3. User 자신의 응답 이력  (P2)
        """
        # meal_ingredient 가 있고, 첫 턴이 아닐 때만 별도 블록으로 삽입
        # Turn 0 에서는 음식 이름만 보이고, 상세 재료는 코치 질문에 따라 점진적으로 공개
        if self.meal_ingredient and not is_first_turn:
            ingredient_block = (
                "\nIngredient details                 : " + self.meal_ingredient
            )
        else:
            ingredient_block = ""

        # 페르소나 블록: 하나라도 설정되어 있으면 삽입
        if self.persona_preferences or self.persona_allergies or self.persona_restrictions:
            persona_block = _PERSONA_BLOCK.format(
                preferences=", ".join(self.persona_preferences) if self.persona_preferences else "(none)",
                allergies=", ".join(self.persona_allergies) if self.persona_allergies else "(none)",
                restrictions=", ".join(self.persona_restrictions) if self.persona_restrictions else "(none)",
            )
            persona_style_block = _PERSONA_STYLE_BLOCK
        else:
            persona_block = ""
            persona_style_block = ""

        base = _USER_SYSTEM_BASE.format(
            nutrition_goal=self.nutrition_goal,
            meal_description=self.meal_description,
            meal_ingredient_block=ingredient_block,
            persona_block=persona_block,
            persona_style_block=persona_style_block,
        )
        parts = [base]

        # 요약이 존재할 때만 삽입 (P4) — DialogSummarizer 출력
        if dialog_summary:
            parts.append(_USER_SUMMARY_BLOCK.format(dialog_summary=dialog_summary))

        # 자신의 응답 이력 (P2)
        parts.append(
            _USER_OWN_BUFFER_BLOCK.format(
                own_buffer=self.own_buffer.to_prompt_text(
                    header="Meal details you have already mentioned"
                )
            )
        )

        return "".join(parts)
