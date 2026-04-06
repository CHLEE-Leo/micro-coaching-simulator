"""
models/alignment_estimator.py
───────────────
LLM 기반 Alignment Estimator 모델.

역할
  - 매 턴마다 InformationSeeker ↔ User 대화에서 드러난 식사 정보를 누적하여
    해당 식사가 지정된 영양 목표(nutritional goal)를 달성하는지 실시간 판정합니다.
  - 판정 결과가 "aligned"(answer == "1")이면 시뮬레이션이 clean termination 됩니다.

설계 원칙
  - DSPy / Ollama 의존 없음 — 기존 vLLM 인프라(batch_generate)를 그대로 활용합니다.
  - 별도의 메모리 추출 LLM 호출 없이, SharedConversationHistory 의 대화 기록 자체를
    컨텍스트로 사용합니다 (추가 레이턴시 최소화).
  - 매 턴 한 번의 vLLM 배치 호출로 alignment 를 판정합니다.
  - 모든 정적 리소스(goal_def, expert_workflow, output_format_inst)는
    __init__ 시점에 한 번만 로드합니다.

데이터 파일 (data/additional/)
  - goal_def.json              : 목표별 정의 및 달성 기준
  - expert_workflow.json       : 정성/정량 평가 워크플로우
  - output_format_inst_binary.txt : JSON binary 출력 형식 지시문

통합 방식 (simulation.py)
  - simulate_conversations_batch() 내부에서 User 배치 이후 AlignmentEstimator 배치가 실행됩니다.
  - alignment 배치 결과가 aligned 이면 ctx["terminated"] = True, terminated_by = "alignment".
  - _build_result() 에 alignment 및 alignment_history 필드가 추가됩니다.

최소 판정 시작 턴 (config.alignment_min_turn)
  - 충분한 식사 정보가 드러나기 전에 조기 종료되지 않도록
    config.alignment_min_turn 이전 턴에서는 판정을 건너뜁니다.
  - 기본값 3: 턴 0~2는 스킵하고 턴 3부터 판정합니다.

커스터마이징
  - _build_alignment_system_prompt() : 판정 기준과 프롬프트 구조.
                                   다른 평가 관점을 적용하려면 이 함수를 수정하세요.
  - goal_def.json               : 목표별 정의. 새 영양 목표를 추가하거나
                                   기존 정의를 변경하려면 이 파일을 수정하세요.
  - expert_workflow.json         : 전문가 워크플로우 (정성/정량 단계).
                                   평가 절차를 바꾸려면 이 파일을 수정하세요.
  - output_format_inst_*.txt     : 출력 포맷 지시문 (binary | 0-1 | 0-100).
  - alignment_use_goal_def / alignment_use_workflow (config.py)
                                 : 프롬프트 scaffold 블록 포함 여부 스위치.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

from utils.llm_utils import generate_response

if TYPE_CHECKING:
    from config import SimulationConfig
    from core.memory import SharedConversationHistory


# ──────────────────────────────────────────────────────────────────────────────
# 데이터 디렉터리 경로
# ──────────────────────────────────────────────────────────────────────────────

_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "additional"


# ──────────────────────────────────────────────────────────────────────────────
# 정적 리소스 로더 (모듈 수준 캐시)
# ──────────────────────────────────────────────────────────────────────────────

_GOAL_DEF_CACHE:    Optional[Dict]       = None
_WORKFLOW_CACHE:    Optional[List[Dict]] = None
_OUTPUT_FMT_CACHE:  Dict[str, str]       = {}  # format 스트링 키별 캐시


def _load_goal_definitions() -> Dict:
    global _GOAL_DEF_CACHE
    if _GOAL_DEF_CACHE is None:
        path = _DATA_DIR / "goal_def_v2.json"
        with open(path, "r", encoding="utf-8") as f:
            _GOAL_DEF_CACHE = json.load(f)
    return _GOAL_DEF_CACHE


def _load_expert_workflows() -> List[Dict]:
    global _WORKFLOW_CACHE
    if _WORKFLOW_CACHE is None:
        path = _DATA_DIR / "expert_workflow.json"
        with open(path, "r", encoding="utf-8") as f:
            _WORKFLOW_CACHE = json.load(f)
    return _WORKFLOW_CACHE


def _load_output_format(fmt: str = "binary") -> str:
    """
    output_format_inst 파일을 로드합니다.

    Parameters
    ----------
    fmt : "binary" | "0-1" | "0-100"
    """
    global _OUTPUT_FMT_CACHE
    if fmt not in _OUTPUT_FMT_CACHE:
        filename = f"output_format_inst_{fmt}.txt"
        path = _DATA_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"output format 파일을 찾을 수 없습니다: {path}")
        with open(path, "r", encoding="utf-8") as f:
            _OUTPUT_FMT_CACHE[fmt] = f.read().strip()
    return _OUTPUT_FMT_CACHE[fmt]


# ──────────────────────────────────────────────────────────────────────────────
# 헬퍼 함수
# ──────────────────────────────────────────────────────────────────────────────

_QUALITATIVE_GOALS = {"lean_protein", "drink_water"}


def _get_workflow_text(nutrition_goal: str) -> str:
    """
    nutrition_goal 에 맞는 전문가 워크플로우를 번호 리스트 형태의 문자열로 반환합니다.

    lean_protein / drink_water   → qualitative 워크플로우
    half_fruits_vegetables / one_fourth_carbs → quantitative 워크플로우
    """
    category = "qualitative" if nutrition_goal in _QUALITATIVE_GOALS else "quantitative"
    for wf in _load_expert_workflows():
        if wf.get("goal_category") == category:
            return "\n".join(wf.get("expert_workflow", []))
    return ""


def _get_goal_spec(nutrition_goal: str) -> Dict:
    return _load_goal_definitions().get(nutrition_goal, {})


# ──────────────────────────────────────────────────────────────────────────────
# 시스템 프롬프트 빌더
# ──────────────────────────────────────────────────────────────────────────────

def _build_alignment_system_prompt(
    nutrition_goal:  str,
    goal_definition: str,   # 빈 문자열이면 블록 전체 생략
    workflow_text:   str,   # 빈 문자열이면 블록 전체 생략
    output_format_inst: str,
) -> str:
    """
    scaffold 값이 비어 있을 때 해당 헤더/블록 자체를 프롬프트에서 제거합니다.

    - goal_definition 이 있으면 TASK 입력 목록과 DECISION PROTOCOL 에 언급
    - workflow_text 가 있으면 'WORKFLOW OF EXPERT NUTRITIONIST:' 섹션 추가
    - output_format_inst 는 항상 포함
    """
    # ── TASK 입력 목록 ──────────────────────────────────────────────────────
    task_inputs = ["- nutrition_goal"]
    if goal_definition:
        task_inputs.append(f"- goal_definition: {goal_definition}")
    task_inputs += ["- context (meal description)", "- question"]

    # ── DECISION PROTOCOL goal_definition 언급 ────────────────────────────
    goal_def_note = " (and goal_definition if available)" if goal_definition else ""

    # ── 조건부 섹션 ────────────────────────────────────────────────────────
    workflow_block = (
        f"\n\nWORKFLOW OF EXPERT NUTRITIONIST:\n{workflow_text}"
        if workflow_text else ""
    )

    return (
        "You are an expert nutritionist evaluating whether a meal aligns with a nutritional goal."
        "\n\nTASK:\nUse the provided inputs to judge alignment:\n"
        + "\n".join(task_inputs)
        + "\n\nDECISION PROTOCOL:"
        + "\n1. Identify the main food items and preparation cues in the meal."
        + f"\n2. Compare the meal against the nutrition goal{goal_def_note}."
        + "\n3. Weigh supporting evidence vs. conflicting evidence."
        + "\n4. Make one final alignment judgment."
        + "\n\nOUTPUT POLICY:"
        + "\n- Follow output_format_instruction exactly."
        + "\n- Return the answer and a brief reasoning in the required JSON format."
        + "\n- Do not add extra keys, markdown, or surrounding text."
        + "\n- If uncertain, still return a valid value in the allowed range/format."
        + "\n- For continuous scales, avoid boundary values (0.5 or 50) unless strictly necessary."
        + "\n\nREASONING ABOUT SCORE CHANGES:"
        + "\n- If a previous alignment score is provided, your reasoning MUST explain why the current score differs from (or remains the same as) the previous score."
        + "\n- Describe what new information from the latest conversation turn caused the score to increase, decrease, or stay the same."
        + "\n- If no previous score is provided (first evaluation), base your reasoning solely on the current evidence."
        + workflow_block
        + f"\n\n{output_format_inst}"
    )

_ALIGNMENT_USER_TEMPLATE = """\
[context]
{transcript}
{prev_score_context}
[question]
Does this meal align with the goal of {nutrition_goal_display}?"""


# ──────────────────────────────────────────────────────────────────────────────
# AlignmentEstimator
# ──────────────────────────────────────────────────────────────────────────────

class AlignmentEstimator:
    """
    LLM 기반 영양 목표 달성 여부 실시간 판정 모델.

    Parameters
    ----------
    model          : vLLM LLM 객체 (Coach / User 와 공유 가능)
    nutrition_goal : "lean_protein" | "half_fruits_vegetables" | "one_fourth_carbs"
    config         : SimulationConfig 인스턴스
    """

    def __init__(
        self,
        model,
        nutrition_goal: str,
        config: "SimulationConfig",
    ):
        self.model          = model
        self.nutrition_goal = nutrition_goal
        self.config         = config

        # ── 정적 리소스 로드 (최초 1회) ───────────────────────────────────
        goal_spec = _get_goal_spec(nutrition_goal)

        # config 스위치에 따라 scaffold 값 자체를 빈 문자열로 대체
        # → _build_alignment_system_prompt() 가 빈 값을 받으면 해당 블록을 생략
        self._goal_definition = (
            goal_spec.get("definition", "") if config.alignment_use_goal_def else ""
        )
        self._workflow_text = (
            _get_workflow_text(nutrition_goal) if config.alignment_use_workflow else ""
        )
        self._output_fmt = _load_output_format(config.alignment_output_format)

        # 시스템 프롬프트 사전 빌드 (매 턴 재사용)
        self._system_prompt = _build_alignment_system_prompt(
            nutrition_goal     = nutrition_goal,
            goal_definition    = self._goal_definition,
            workflow_text      = self._workflow_text,
            output_format_inst = self._output_fmt,
        )

        # ── 상태 변수 ─────────────────────────────────────────────────────
        # 턴별 판정 이력: [{"turn_idx": int, "aligned": bool, "score": float, "raw_output": str}]
        self._judgment_history: List[Dict] = []
        self._last_aligned: Optional[bool]  = None
        self._last_score:   Optional[float] = None
        self._last_reasoning: Optional[str]  = None

    # ── 공개 인터페이스 ────────────────────────────────────────────────────

    def get_messages(
        self,
        history: "SharedConversationHistory",
    ) -> List[Dict[str, str]]:
        """
        배치 생성용: batch_generate() 에 넘길 messages 리스트를 반환합니다.
        LLM 을 직접 호출하지 않습니다.

        Parameters
        ----------
        history : SharedConversationHistory — 현재까지의 Coach ↔ User 대화 기록

        Returns
        -------
        List[Dict[str, str]] : chat-template 형식의 messages
        """
        transcript = self._build_transcript(history)

        # 이전 턴 점수 컨텍스트 생성
        if self._last_score is not None:
            prev_score_context = (
                f"\n[previous alignment score]\n"
                f"The alignment score from the previous turn was {self._last_score:.2f}.\n"
                f"In your reasoning, you MUST explain why the score changed, decreased, increased, "
                f"or stayed the same compared to this previous score.\n"
            )
        else:
            prev_score_context = ""  # 첫 턴: 이전 점수 없음

        user_content = _ALIGNMENT_USER_TEMPLATE.format(
            transcript            = transcript,
            nutrition_goal_display = self.nutrition_goal.replace("_", " "),
            prev_score_context    = prev_score_context,
        )
        return [
            {"role": "system", "content": self._system_prompt},
            {"role": "user",   "content": user_content},
        ]

    def apply_judgment(self, raw_output: str, turn_idx: int) -> bool:
        """
        LLM 출력을 파싱하여 내부 상태를 갱신합니다.

        Parameters
        ----------
        raw_output : LLM 이 생성한 원시 문자열
        turn_idx   : 현재 턴 인덱스 (이력 기록용)

        Returns
        -------
        bool : True → aligned (goal 달성 판정)
        """
        score = self._parse_answer(raw_output)

        if score is None:
            # 파싱 실패 → 보수적으로 not aligned
            score   = 0.0
            aligned = False
        elif self.config.alignment_output_format == "binary":
            aligned = (score == 1.0)
        else:
            # 0-1 및 0-100(정규화 후) 모두 [0, 1] 기준 임계값 비교
            aligned = (score >= self.config.alignment_threshold)

        self._last_aligned = aligned
        self._last_score   = score
        self._judgment_history.append({
            "turn_idx"   : turn_idx,
            "aligned"    : aligned,
            "score"      : score,
            "reasoning"  : self._last_reasoning,
            "raw_output" : raw_output.strip(),
        })
        return aligned

    def should_evaluate(self, turn_idx: int) -> bool:
        """
        현재 턴에서 판정을 실행해야 하는지 여부를 반환합니다.
        config.alignment_min_turn 미만에서는 충분한 정보가 없으므로 건너뜁니다.

        Parameters
        ----------
        turn_idx : 현재 완성된 최대 턴 인덱스 (0-based)
        """
        return turn_idx >= self.config.alignment_min_turn

    # ── 단일 모드 (디버깅용) ──────────────────────────────────────────────

    def evaluate(
        self,
        history: "SharedConversationHistory",
        turn_idx: int,
    ) -> bool:
        """
        단일 대화 모드에서 동기로 판정을 실행합니다.
        배치 모드에서는 get_messages() + batch_generate() + apply_judgment() 를 사용하세요.

        Parameters
        ----------
        history  : SharedConversationHistory
        turn_idx : 현재 턴 인덱스

        Returns
        -------
        bool : True → aligned
        """
        if not self.should_evaluate(turn_idx):
            return False

        messages = self.get_messages(history)
        raw = generate_response(
            self.model,
            messages,
            sampling="greedy",          # 판정은 항상 greedy
            max_new_tokens=self.config.alignment_max_new_tokens,
            stop_at_newline=False,       # JSON 출력이므로 개행 중단 비활성화
        )
        return self.apply_judgment(raw, turn_idx)

    # ── 프로퍼티 ──────────────────────────────────────────────────────────

    @property
    def is_aligned(self) -> Optional[bool]:
        """가장 최근 판정 결과. 한 번도 판정하지 않았으면 None."""
        return self._last_aligned
    @property
    def last_score(self) -> Optional[float]:
        """가장 최근 판정의 정규화된 점수 [0, 1]. 판정 전이면 None."""
        return self._last_score
    @property
    def last_reasoning(self) -> Optional[str]:
        """가장 최근 판정의 reasoning 텍스트. 없으면 None."""
        return self._last_reasoning
    @property
    def judgment_history(self) -> List[Dict]:
        """전체 판정 이력. 각 원소: {"turn_idx": int, "aligned": bool, "raw_output": str}"""
        return list(self._judgment_history)

    # ── 내부 헬퍼 ─────────────────────────────────────────────────────────

    @staticmethod
    def _build_transcript(history: "SharedConversationHistory") -> str:
        """
        Alignment Tracker 프롬프트의 컨텍스트 블록을 반환합니다.

        history.to_alignment_context() 를 사용하여
        User / Coach 와 동일하게 [summary + windowed turns] 형태로 구성합니다.
        전체 turns 를 raw 로 넘기는 대신 summary 로 오래된 맥락을 압축하고,
        최신 턴만 원문으로 첨부하므로 Alignment Tracker 입력 크기가 bounded 됩니다.
        """
        return history.to_alignment_context()

    def _parse_answer(self, raw: str) -> Optional[float]:
        """
        config.alignment_output_format 에 따라 LLM 출력을 파싱하여
        정규화된 점수 [0.0, 1.0] 로 반환합니다.

        - binary  : "1" → 1.0 / "0" → 0.0
        - 0-1     : float 값 그대로 반환 (0.0 ~ 1.0)
        - 0-100   : float / 100 정규화 (0.0 ~ 1.0)

        파싱 실패 또는 범위 벗어난 값이면 None 반환.
        """
        import re
        text = raw.strip()

        # JSON 코드펜스 제거
        if "```" in text:
            parts = text.split("```")
            text = parts[-2] if len(parts) >= 3 else text.replace("```", "")
            text = text.replace("json", "").strip()

        # JSON 파싱 시도
        answer_str = ""
        try:
            data = json.loads(text)
            answer_str = str(data.get("answer", "")).strip()
            # reasoning 추출 및 저장
            self._last_reasoning = str(data.get("reasoning", "")).strip() or None
        except (json.JSONDecodeError, AttributeError, ValueError):
            # fallback: 정규식으로 추출
            m = re.search(r'"answer"\s*:\s*"([^"]+)"', raw)
            if m:
                answer_str = m.group(1).strip()
            # reasoning fallback
            m_r = re.search(r'"reasoning"\s*:\s*"([^"]+)"', raw)
            self._last_reasoning = m_r.group(1).strip() if m_r else None

        if not answer_str:
            return None

        fmt = self.config.alignment_output_format
        try:
            if fmt == "binary":
                if answer_str == "1":
                    return 1.0
                elif answer_str == "0":
                    return 0.0
                return None
            elif fmt == "0-1":
                val = float(answer_str)
                return val if 0.0 <= val <= 1.0 else None
            elif fmt == "0-100":
                val = float(answer_str)
                return (val / 100.0) if 0.0 <= val <= 100.0 else None
        except (ValueError, TypeError):
            return None

        return None
