"""
core/memory.py
──────────────
대화 상태를 관리하는 두 종류의 메모리 클래스.

  ConversationBuffer
    각 에이전트(Coach / User)가 자신의 발화만 따로 기록하는 개인 버퍼.
    "내가 어떤 질문/답변을 해왔는지"를 프롬프트에 주입할 때 사용합니다.
    (Principle 2)

  SharedConversationHistory
    Coach ↔ User 의 공통 대화 기록.
    context_window 만큼만 최근 턴을 반환하고, 나머지는 요약(summary) 로 대체합니다.
    (Principle 3 + 4)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


# ──────────────────────────────────────────────────────────────────────────────
# 1. 개인 발화 버퍼  (Principle 2)
# ──────────────────────────────────────────────────────────────────────────────

class ConversationBuffer:
    """
    한 에이전트의 발화 이력만 관리합니다.

    Usage
    -----
    buf = ConversationBuffer(role="coach")
    buf.add("What else will you have?")
    print(buf.to_prompt_text())
    """

    def __init__(self, role: str):
        """
        Parameters
        ----------
        role : "coach" | "user"  (프롬프트 출력 시 레이블로 사용)
        """
        self.role = role
        self._utterances: List[str] = []

    # ── 추가 ────────────────────────────────────────────────────────────────
    def add(self, utterance: str) -> None:
        """새 발화를 버퍼에 추가합니다."""
        text = utterance.strip()
        if text:
            self._utterances.append(text)

    # ── 조회 ────────────────────────────────────────────────────────────────
    def get_all(self) -> List[str]:
        """전체 발화 리스트를 반환합니다."""
        return list(self._utterances)

    def get_recent(self, n: int) -> List[str]:
        """최근 n 개의 발화를 반환합니다."""
        return self._utterances[-n:] if n > 0 else []

    def __len__(self) -> int:
        return len(self._utterances)

    # ── 프롬프트 주입용 텍스트 ──────────────────────────────────────────────
    def to_prompt_text(self, header: str | None = None) -> str:
        """
        버퍼 내용을 번호 매긴 리스트 형태의 문자열로 반환합니다.
        시스템 프롬프트에 직접 삽입할 수 있습니다.

        Parameters
        ----------
        header : 섹션 제목 (None 이면 기본 텍스트 사용)

        Returns
        -------
        str : 형식화된 발화 이력
        """
        if not self._utterances:
            return "(none yet)"

        label = header or f"Your previous {self.role} utterances"
        lines = [f"{label}:"]
        for i, utt in enumerate(self._utterances, start=1):
            lines.append(f"  {i}. {utt}")
        return "\n".join(lines)

    # ── 초기화 ──────────────────────────────────────────────────────────────
    def reset(self) -> None:
        """버퍼를 초기화합니다."""
        self._utterances.clear()


# ──────────────────────────────────────────────────────────────────────────────
# 2. 공통 대화 기록  (Principle 3 + 4)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class _Turn:
    """한 턴 = Coach 발화 + User 발화 쌍."""
    turn_idx:        int
    coach_utterance: str
    user_utterance:  str  # 마지막 턴에서는 User 응답 전일 수 있어 빈 문자열 허용


class SharedConversationHistory:
    """
    Coach ↔ User 공통 대화 기록 및 컨텍스트 창 관리.

    책임
    ----
    1. 턴 단위 대화를 시간 순으로 누적
    2. context_window 내 최근 턴만 chat_template 형식으로 반환  (Principle 3)
    3. 요약 갱신·조회  (Principle 4)
    4. 전체 대화를 평문(plain text)으로 직렬화 (요약 생성·저장에 활용)

    Usage
    -----
    hist = SharedConversationHistory(context_window=5)
    hist.add_turn(0, "What are you thinking of...", "I'm having grilled chicken")
    coach_msgs = hist.build_messages(perspective="coach", system_prompt="...")
    user_msgs  = hist.build_messages(perspective="user",  system_prompt="...")
    """

    # 종료 신호: User 가 이 문자열을 포함하면 대화를 종료합니다.
    TERMINATION_TOKEN = "That's all about my meal."

    def __init__(self, context_window: int = 5):
        """
        Parameters
        ----------
        context_window : build_messages() 가 포함할 최근 턴 수.
                         context_window=0 이면 전체 기록을 사용합니다.
        """
        self.context_window = context_window
        self._turns: List[_Turn] = []
        self.summary: str = ""          # 요약 (principle 4)

    # ── 턴 추가 ─────────────────────────────────────────────────────────────
    def add_turn(
        self,
        turn_idx:        int,
        coach_utterance: str,
        user_utterance:  str = "",
    ) -> None:
        """
        완성된 Coach-User 교환을 기록에 추가합니다.

        Parameters
        ----------
        turn_idx        : 0-based 턴 인덱스
        coach_utterance : 코치 발화
        user_utterance  : 유저 응답 (아직 없으면 빈 문자열)
        """
        self._turns.append(_Turn(turn_idx, coach_utterance.strip(), user_utterance.strip()))

    def update_last_user_utterance(self, user_utterance: str) -> None:
        """
        마지막 턴의 user_utterance 를 업데이트합니다.
        Coach 발화 먼저 추가한 뒤 User 응답이 나오면 호출하는 패턴에 사용합니다.
        """
        if self._turns:
            self._turns[-1].user_utterance = user_utterance.strip()

    # ── 컨텍스트 창 ─────────────────────────────────────────────────────────
    def _windowed_turns(self) -> List[_Turn]:
        """context_window 범위 내의 최근 턴 리스트를 반환합니다."""
        if self.context_window <= 0:
            return list(self._turns)
        return self._turns[-self.context_window:]

    # ── chat_template 메시지 빌드 ────────────────────────────────────────────
    def build_messages(
        self,
        perspective: str,
        system_prompt: str,
    ) -> List[Dict[str, str]]:
        """
        각 에이전트 관점에서 HuggingFace chat_template 에 넣을 메시지 리스트를 생성합니다.

        관점(perspective)에 따라 role 이 달라집니다.
          - "coach" : coach utterance → "assistant", user utterance → "user"
          - "user"  : user utterance  → "assistant", coach utterance → "user"

        Parameters
        ----------
        perspective   : "coach" | "user"
        system_prompt : 시스템 프롬프트 문자열

        Returns
        -------
        List[Dict] : [{"role": ..., "content": ...}, ...]
        """
        if perspective not in ("coach", "user"):
            raise ValueError("perspective 는 'coach' 또는 'user' 이어야 합니다.")

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt}
        ]

        for i, turn in enumerate(self._windowed_turns()):
            if perspective == "coach":
                # Coach 관점: coach = "assistant"(생성 주체), user = "user"(상대방)
                #
                # Turn 0의 coach 발화는 고정값(first_question)이므로 LLM이 생성하지 않음.
                # 따라서 LLM 기준 첫 번째 입력은 항상 User 모델의 응답(A0)이 됩니다.
                # Windowed turns의 첫 항목은 coach_utterance를 생략하고
                # user_utterance만 "user"로 넣어 system → user 순서를 보장합니다.
                # (생략된 coach 발화들은 own_buffer를 통해 시스템 프롬프트에 이미 주입됨)
                #
                #   system
                #   user:      A0   ← user의 첫 응답 (Q0은 고정이므로 생략)
                #   assistant: Q1
                #   user:      A1
                #   ...
                #   user:      A_{t-1}
                #   → generate: assistant (Q_t)
                #
                # NOTE: turn_idx==0 의 고정 첫 질문(Q0)만 생략합니다.
                # 슬라이딩 윈도우로 이전 턴이 잘려나온 경우 i==0 이라도
                # 실제 turn_idx>0 이면 coach 발화를 생략하면 안 됩니다.
                if turn.turn_idx == 0:
                    if turn.user_utterance:
                        messages.append({"role": "user", "content": turn.user_utterance})
                else:
                    messages.append({"role": "assistant", "content": turn.coach_utterance})
                    if turn.user_utterance:
                        messages.append({"role": "user", "content": turn.user_utterance})
            else:
                # User 관점: user = "assistant"(생성 주체), coach = "user"(상대방)
                #
                # Coach의 질문이 항상 "user" 역할로 오므로 자연스럽게 alternation 성립.
                #
                #   system
                #   user:      Q0   ← coach의 고정 첫 질문
                #   assistant: A0
                #   user:      Q1
                #   assistant: A1
                #   ...
                #   user:      Q_t  ← 최신 coach 질문
                #   → generate: assistant (A_t)
                messages.append({"role": "user", "content": turn.coach_utterance})
                if turn.user_utterance:
                    messages.append({"role": "assistant", "content": turn.user_utterance})

        return messages

    # ── 요약 갱신 (Principle 4) ────────────────────────────────────────────
    def update_summary(self, new_summary: str) -> None:
        """요약을 갱신합니다."""
        self.summary = new_summary.strip()

    # ── 직렬화 ──────────────────────────────────────────────────────────────
    def to_plain_text(self) -> str:
        """
        전체 대화를 sumbmarizer 에 넘길 평문으로 직렬화합니다.

        Returns
        -------
        str : "Coach: ...\nUser: ...\n..." 형식의 대화 텍스트
        """
        lines: List[str] = []
        for turn in self._turns:
            lines.append(f"Coach: {turn.coach_utterance}")
            if turn.user_utterance:
                lines.append(f"User: {turn.user_utterance}")
        return "\n".join(lines)

    def to_plain_text_from(self, from_turn_idx: int = 0) -> str:
        """
        from_turn_idx 이후 턴만 평문으로 직렬화합니다 (증분 요약에 사용).
        / Serialise only turns at or after from_turn_idx (for incremental summarisation).
        """
        lines: List[str] = []
        for turn in self._turns:
            if turn.turn_idx >= from_turn_idx:
                lines.append(f"Coach: {turn.coach_utterance}")
                if turn.user_utterance:
                    lines.append(f"User: {turn.user_utterance}")
        return "\n".join(lines)

    def get_all_coach_questions(self) -> List[str]:
        """
        Coach 가 발화한 모든 질문 목록을 히스토리에서 직접 추출합니다.
        own_buffer 와 달리 누의 없이 항상 완전한 목록을 반환합니다.
        / Returns every coach utterance recorded in history — guaranteed complete.
        """
        return [t.coach_utterance for t in self._turns if t.coach_utterance.strip()]

    def to_dict_list(self) -> List[Dict]:
        """저장용 딕셔너리 리스트로 변환합니다."""
        return [
            {
                "turn_idx":        t.turn_idx,
                "coach_utterance": t.coach_utterance,
                "user_utterance":  t.user_utterance,
            }
            for t in self._turns
        ]

    def to_judge_context(self) -> str:
        """
        Judge 모델의 컨텍스트 블록을 생성합니다.

        User / Coach 와 동일한 뷰를 제공합니다:
          - (있으면) 대화 요약  → 오래된 턴 전체를 압축
          - 최근 context_window 턴의 원문  → 요약 이후 신규 정보 보완

        이렇게 하면 Judge 가 항상 summary + 최신 대화 원문을 함께 보며
        판정하게 되어 User / Coach 와 일관된 대화 상태 인식이 보장됩니다.
        """
        parts: List[str] = []

        if self.summary:
            parts.append("[Conversation summary so far]")
            parts.append(self.summary)
            parts.append("")
            parts.append("[Recent turns]")

        for turn in self._windowed_turns():
            parts.append(f"Coach: {turn.coach_utterance}")
            if turn.user_utterance:
                parts.append(f"User: {turn.user_utterance}")

        result = "\n".join(parts).strip()
        return result if result else "(no conversation yet)"

    # ── 종료 감지 ────────────────────────────────────────────────────────────
    def is_terminated(self) -> bool:
        """
        User 의 마지막 발화에 종료 토큰이 포함되어 있는지 확인합니다.
        """
        if not self._turns:
            return False
        last_user = self._turns[-1].user_utterance
        return self.TERMINATION_TOKEN.lower() in last_user.lower()

    # ── 기타 ─────────────────────────────────────────────────────────────────
    def __len__(self) -> int:
        """지금까지 쌓인 완성 턴 수를 반환합니다."""
        return len(self._turns)

    def current_turn_idx(self) -> int:
        """다음 턴의 인덱스를 반환합니다."""
        return len(self._turns)
