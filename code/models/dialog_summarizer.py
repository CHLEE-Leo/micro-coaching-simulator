"""
models/dialog_summarizer.py
───────────────────────────
대화 흐름 요약 에이전트 (Dialog Summarizer Agent).

역할
  - InformationSeeker ↔ User 대화의 **흐름**을 서술형으로 요약합니다.
  - "누가 무엇을 물었고, 어떤 답을 했는지" 를 기록하여
    context_window 밖의 오래된 Q&A 가 망각되는 것을 방지합니다.
  - 생성된 요약은 SharedConversationHistory.dialog_summary 에 저장되어
    주로 **InformationSeeker / User Agent** 의 시스템 프롬프트에 주입됩니다.

  왜 필요한가?
    context_window (기본 5턴) 를 사용하면 오래된 턴은 LLM 입력에서 사라집니다.
    InformationSeeker 가 이미 물어본 질문을 또 물어보거나, User 가 이미 답한 내용을
    다시 말하는 반복이 생길 수 있습니다. Dialog Summarizer 는 이 "대화 흐름"
    맥락을 요약하여 InformationSeeker/User 가 이전 대화를 기억하게 합니다.

  데이터 흐름:
    대화 원문 → DialogSummarizerModel.summarize()
      → 대화 흐름 요약 (서술형)
      → SharedConversationHistory.dialog_summary 에 저장
      → InformationSeeker/User 시스템 프롬프트의 [Dialog summary so far] 블록으로 주입

  Meal Tracker (meal_tracker.py) 와의 차이:
    ┌─────────────────────────┬─────────────────────────┐
    │   Dialog Summarizer     │      Meal Tracker       │
    ├─────────────────────────┼─────────────────────────┤
    │ 대화 흐름 서술형 요약    │  식사 정보 구조화 추출   │
    │ 주 소비자: IS, User      │  주 소비자: AlignmentEstimator     │
    │ "무엇을 물었/답했는가"   │  "무엇을 먹는가"        │
    │ 반복 질문/응답 방지      │  정확한 영양 판정 지원   │
    └─────────────────────────┴─────────────────────────┘

두 가지 요약 모드
  [전체 요약]  conversation_text 전체를 요약
    배치 시뮬레이션 (code/core/simulation.py) 에서 주로 사용합니다.

  [증분 요약]  이전 요약 + 신규 턴만으로 요약 갱신
    인터랙티브 모드 (code_interactive/session_manager.py) 에서 주로 사용합니다.

커스터마이징
  - _DIALOG_SUMMARY_SYSTEM_FULL / _DIALOG_SUMMARY_SYSTEM_INCREMENTAL
    요약의 초점이나 형식을 바꾸려면 수정하세요.
  - config.summarize_max_new_tokens : 요약 최대 생성 토큰 수.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List

from utils.llm_utils import generate_response

if TYPE_CHECKING:
    from config import SimulationConfig


# ──────────────────────────────────────────────────────────────────────────────
# 시스템 프롬프트
# ──────────────────────────────────────────────────────────────────────────────

_DIALOG_SUMMARY_SYSTEM_FULL = (
    "You are a conversation-flow summarizer for a nutrition coaching dialog. "
    "Summarize the progression of the conversation so far in 2-4 sentences.\n"
    "\n"
    "Focus on:\n"
    "- What topics the coach asked about (e.g., ingredients, cooking method, portions, beverages)\n"
    "- What the user has already answered or revealed\n"
    "- What the user said they don't know or couldn't answer\n"
    "- Any topics NOT yet discussed\n"
    "\n"
    "Rules:\n"
    "- Describe the Q&A flow, not the meal itself (a separate agent handles meal details).\n"
    "- Be concise — this summary helps the coach avoid repeating questions "
    "and helps the user avoid restating answers.\n"
    "- Do not infer information not stated in the conversation."
)

_DIALOG_SUMMARY_SYSTEM_INCREMENTAL = (
    "You are an incremental conversation-flow summarizer. "
    "You will be given a previous dialog summary and new conversation turns. "
    "Produce an UPDATED dialog summary that incorporates the new turns.\n"
    "\n"
    "Focus on:\n"
    "- What new topics the coach asked about\n"
    "- What new information the user revealed or couldn't answer\n"
    "- Update the list of discussed vs. not-yet-discussed topics\n"
    "\n"
    "Rules:\n"
    "- Preserve key information from the previous summary — only add or update, never remove.\n"
    "- Describe the Q&A flow, not the meal itself.\n"
    "- Keep the total summary to 2-4 sentences.\n"
    "- Do not infer information not stated."
)


# ──────────────────────────────────────────────────────────────────────────────
# DialogSummarizerModel
# ──────────────────────────────────────────────────────────────────────────────

class DialogSummarizerModel:
    """
    LLM 기반 대화 흐름 요약 에이전트.

    Coach ↔ User 대화의 Q&A 흐름을 서술형으로 요약하여,
    Coach 의 중복 질문과 User 의 중복 응답을 방지합니다.

    인터페이스 패턴 (Coach / User / Alignment Tracker 와 동일):
      - get_messages()  : LLM 호출 없이 messages 리스트만 반환 (배치 생성용)
      - summarize()     : messages 를 빌드하고 LLM 을 호출하여 요약 텍스트 반환

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
        prev_summary: str = "",
    ) -> List[Dict[str, str]]:
        """
        배치 생성용: LLM 을 호출하지 않고 messages 리스트만 반환합니다.

        Parameters
        ----------
        conversation_text : "Coach: ...\nUser: ..." 형식의 대화 텍스트.
                            전체 요약 시 전체 대화, 증분 요약 시 신규 턴만 전달합니다.
        prev_summary      : 이전에 생성된 대화 요약 (빈 문자열이면 전체 요약 모드).

        Returns
        -------
        List[Dict[str, str]] : chat-template 형식의 messages
        """
        if prev_summary and conversation_text:
            # ── 증분 요약: 기존 요약 + 신규 턴 → 통합 요약 ──
            return [
                {"role": "system", "content": _DIALOG_SUMMARY_SYSTEM_INCREMENTAL},
                {
                    "role": "user",
                    "content": (
                        f"Previous dialog summary:\n{prev_summary}\n\n"
                        f"New conversation turns:\n\n"
                        f"{conversation_text}\n\n"
                        "Now write the updated dialog summary:"
                    ),
                },
            ]
        else:
            # ── 전체 요약: 대화 전문 → 요약 ──
            return [
                {"role": "system", "content": _DIALOG_SUMMARY_SYSTEM_FULL},
                {
                    "role": "user",
                    "content": (
                        "Conversation to summarize:\n\n"
                        f"{conversation_text}\n\n"
                        "Now write the dialog summary:"
                    ),
                },
            ]

    def summarize(
        self,
        conversation_text: str,
        prev_summary: str = "",
    ) -> str:
        """
        대화 흐름을 2-4 문장으로 요약합니다.

        Parameters
        ----------
        conversation_text : "Coach: ...\nUser: ..." 형식의 대화 텍스트
        prev_summary      : 이전 요약 (빈 문자열이면 전체 요약 모드)

        Returns
        -------
        str : 서술형 대화 흐름 요약.
              Coach / User 시스템 프롬프트의
              [Dialog summary so far] 블록에 주입됩니다.
        """
        messages = self.get_messages(conversation_text, prev_summary)
        return generate_response(
            self.model,
            messages,
            sampling="greedy",
            max_new_tokens=self.config.summarize_max_new_tokens,
            stop_at_newline=False,
        )
