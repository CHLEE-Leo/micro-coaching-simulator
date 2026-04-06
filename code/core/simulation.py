"""
core/simulation.py
──────────────────
배치 시뮬레이션 오케스트레이터.

배치 모드에서는 Orchestrator 없이 간소화된 흐름을 사용합니다:
InformationSeeker가 직접 질문 생성, AlignmentEstimator가 종료 판단.
(full Orchestrator-centric 흐름은 code_interactive/ 에서 구현)

에이전트 상호작용 플로우 (배치 — 매 턴)
─────────────────────────────────────

    ┌──────────────────┐  질문   ┌──────────┐
    │ InformationSeeker│ ─────→ │   User   │
    └────────┬─────────┘        └────┬─────┘
             │                       │ 응답 (partial info)
             │              ┌────────┴─────────────┐
             │              │                      │
             │     ┌────────────────┐  ┌──────────────────┐
             │     │  MealTracker   │  │ DialogSummarizer │
             │     │ → Fact Sheet   │  │ → 대화흐름 요약   │
             │     └───────┬────────┘  └────────┬─────────┘
             │             │                    │
             │             ▼                    ▼
             │   ┌──────────────────────────────────┐
             │   │    Shared Conversation History   │
             │   └──────────────┬───────────────────┘
             │                  │ meal_fact_sheet
             │                  ▼
             │         ┌─────────────────┐
             │         │AlignmentEstimator│ ← aligned 판정 시 종료
             │         └─────────────────┘
             │
        IS 질문 반복 (max_turns까지)

  ※ 인터랙티브 모드(code_interactive/)에서는 Orchestrator가 중앙 허브 역할을 하며,
    Guardrail, MealRecommender, UncertaintyEstimator, Memorizer도 함께 사용됩니다.

  Turn 0 :
    InformationSeeker → 고정 초기 질문 ("What are you having for {meal_type}?")
    User               → 음식 전체 공개 (meal_description)

  Turn t (≥ 1) :
    InformationSeeker  → 질문 생성 (LLM)
    User               → 상세 정보 점진 공개 (LLM, partial information)
    MealTracker        → 식사 정보를 Meal Fact Sheet 로 구조화 추출 (매 N턴)
    DialogSummarizer   → 대화 흐름을 서술형으로 요약 (매 N턴)
    AlignmentEstimator → Meal Fact Sheet + 최근 대화 원문으로 영양 목표 달성 판정

  종료 조건 :
    - AlignmentEstimator 가 aligned 판정 → terminated_by = "alignment"
    - max_turns 초과                       → terminated_by = "max_turns"

두 가지 실행 모드
  [단일 모드]  simulate_conversation()
    한 건의 식사 샘플에 대해 순차적으로 대화를 진행합니다.
    디버깅·소규모 실험에 적합합니다.

  [배치 병렬 모드]  simulate_conversations_batch()
    N 건의 다이얼로그를 동시에 진행하며 매 턴마다 Coach/User 발화를
    vLLM 의 batch_generate() 로 한 번에 처리합니다.
    GPU 가동률을 극대화하여 처리량(throughput)이 크게 향상됩니다.

책임 분리
  - 메모리 관리   : core/memory.py      (SharedConversationHistory, ConversationBuffer)
  - 질문 생성     : models/information_seeker.py
  - AI 사용자     : models/user.py
  - 목표 판정     : models/alignment_estimator.py
  - 식사 정보 추출: models/meal_tracker.py       (→ Meal Fact Sheet → AlignmentEstimator 입력)
  - 대화 흐름 요약: models/dialog_summarizer.py  (→ InformationSeeker/User 시스템 프롬프트)
  - LLM 추론      : utils/llm_utils.py  (generate_response, batch_generate)
  - 설정          : config.py
"""

from __future__ import annotations

import random
import re as _re
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from config import SimulationConfig
from core.memory import SharedConversationHistory
from models.information_seeker  import InformationSeeker, _is_duplicate_question as _is_duplicate
from models.alignment_estimator import AlignmentEstimator
from models.user   import UserModel
from models.meal_tracker       import MealTrackerModel
from models.dialog_summarizer  import DialogSummarizerModel
from utils.llm_utils import batch_generate

# Non-answer 패턴: User가 정보를 제공하지 못한 발화
_NON_ANSWER_RE = _re.compile(
    r"(i'?m not sure|i haven'?t decided|not sure|just a standard"
    r"|i don'?t know|don'?t know|haven'?t decided|standard portion"
    r"|i'?m unsure|i'?m not really sure|no idea|not decided)",
    _re.IGNORECASE,
)

def _is_non_answer(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    return bool(_NON_ANSWER_RE.search(stripped))


# ──────────────────────────────────────────────────────────────────────────────
# 시드 고정 헬퍼
# ──────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """재현성을 위해 모든 난수 생성기의 시드를 고정합니다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark    = False
    cudnn.deterministic = True


# ──────────────────────────────────────────────────────────────────────────────
# [단일 모드] 한 건 처리
# ──────────────────────────────────────────────────────────────────────────────

def simulate_conversation(
    dialog_id:        int,
    goal_id:          int,
    nutrition_goal:   str,
    meal_id:          int,
    meal_type:        str,
    meal_description: str,
    coach_llm,
    user_llm,
    config:           SimulationConfig,
    alignment_llm=None,
    expert_result:    str = "yes",
    meal_ingredient:  str = "",
) -> Dict[str, Any]:
    """
    한 건의 식사 샘플에 대해 Coach ↔ User 멀티턴 대화를 실행합니다.
    (단일 처리 모드 — 디버깅 및 소규모 실험용)

    Parameters
    ----------
    dialog_id         : 결과 딕셔너리에 기록할 고유 ID
    goal_id / meal_id : 메타 정보
    nutrition_goal    : "lean_protein" 등
    meal_type         : "breakfast" | "lunch" | "dinner" | "a snack"
    meal_description  : User 모델이 참조할 식사 설명
    coach_llm         : vLLM LLM 객체 (Coach 역할)
    user_llm          : vLLM LLM 객체 (User 역할, coach_llm 과 동일 가능)
    config            : SimulationConfig 인스턴스
    alignment_llm     : vLLM LLM 객체 (AlignmentEstimator 용, None 이면 coach_llm 공유)
    expert_result     : 전문가 레이블 ("yes" | "not_really") — 정확도 추적용

    Returns
    -------
    dict : {id, goal_id, meal_id, nutrition_goal, meal_type, meal_description,
            turns, summary, terminated_by, pred_alignment, true_alignment, alignment_correct}
    """
    # ── 에이전트 초기화 ─────────────────────────────────────────────────────
    _alignment_llm = alignment_llm if alignment_llm is not None else coach_llm

    coach = InformationSeeker(
        model=coach_llm,
        nutrition_goal=nutrition_goal,
        meal_type=meal_type,
        config=config,
    )
    user = UserModel(
        model=user_llm,
        nutrition_goal=nutrition_goal,
        meal_description=meal_description,
        config=config,
        meal_ingredient=meal_ingredient,
    )
    meal_tracker      = MealTrackerModel(model=coach_llm, config=config)
    dialog_summarizer = DialogSummarizerModel(model=coach_llm, config=config)
    alignment         = AlignmentEstimator(
        model=_alignment_llm,
        nutrition_goal=nutrition_goal,
        config=config,
    )

    # ── 공통 대화 기록 초기화 ───────────────────────────────────────────────
    history = SharedConversationHistory(context_window=config.context_window)

    terminated_by = "max_turns"
    dead_end_topics: List[str] = []

    # ── 턴 루프 ─────────────────────────────────────────────────────────────
    with tqdm(total=config.max_turns, desc=f"[Dialog {dialog_id}] Turns") as pbar:

        for turn_idx in range(config.max_turns):

            # ── (1) Coach 발화 ───────────────────────────────────────────
            if turn_idx == 0:
                coach_utterance = coach.first_question()
            else:
                _template = coach.ask(
                    history,
                    dead_end_topics=dead_end_topics if dead_end_topics else None,
                )
                coach_utterance = _template.get(
                    "question_template",
                    "Could you tell me more about your meal?",
                )

            print(f"\n[T{turn_idx}] Coach : {coach_utterance}")
            history.add_turn(turn_idx=turn_idx, coach_utterance=coach_utterance)

            # ── (2) User 응답 ──────────────────────────────────────────
            user_utterance_raw = user.respond(history)
            # [END] 태그 감지 및 제거
            _natural_end = SharedConversationHistory.TERMINATION_TOKEN in user_utterance_raw
            user_utterance = user_utterance_raw.replace(
                SharedConversationHistory.TERMINATION_TOKEN, ""
            ).strip()
            if not user_utterance:
                user_utterance = "I think that covers everything about my meal."
            print(f"[T{turn_idx}] User  : {user_utterance}")
            history.update_last_user_utterance(user_utterance)

            # dead-end 추적: User가 non-answer면 해당 Coach 질문 기록
            if _is_non_answer(user_utterance):
                dead_end_topics.append(coach_utterance)

            pbar.update(1)

            # 자연 종료: User가 [END] 태그를 생성한 경우
            if _natural_end:
                terminated_by = "natural_end"
                break

            # ── (3) MealTracker + DialogSummarizer: 개별 스케줄 ────
            completed = turn_idx + 1
            if completed % config.meal_track_every == 0:
                _update_meal_fact_sheet(history, meal_tracker)
            if completed % config.summarize_every == 0:
                _update_dialog_summary(history, dialog_summarizer)

            # ── (4) AlignmentEstimator: 목표 달성 판정 ────────────
            if alignment.evaluate(history, turn_idx):
                terminated_by = "alignment"
                break

    # 루프 종료 후 최종 갱신
    _update_summaries(history, meal_tracker, dialog_summarizer)

    return _build_result(dialog_id, goal_id, meal_id, nutrition_goal,
                         meal_type, meal_description, history, terminated_by,
                         alignment_tracker=alignment,
                         expert_result=expert_result)


# ──────────────────────────────────────────────────────────────────────────────
# [배치 병렬 모드] N 건 동시 처리
# ──────────────────────────────────────────────────────────────────────────────

def simulate_conversations_batch(
    samples:      pd.DataFrame,
    coach_llm,
    user_llm,
    config:       SimulationConfig,
    alignment_llm     = None,
    already_done: int = 0,
    on_dialog_end: Optional[Callable[[int, Dict[str, Any]], None]] = None,
) -> List[Dict[str, Any]]:
    """
    N 건의 식사 샘플을 턴-라운드 방식으로 병렬 처리합니다.

    핵심 동작 원리
    ─────────────
    • 매 turn_idx 마다 (아직 종료되지 않은) 모든 대화의 Coach 메시지를 모아
      vLLM.batch_generate() 로 한 번에 GPU 에 밀어 넣습니다.
    • 동일하게 User 메시지도 배치로 처리합니다.
    • 각 대화가 종료 토큰을 내거나 max_turns 에 도달하면 결과 목록에서 분리됩니다.
    • 모든 대화가 종료되거나 max_turns 가 끝나면 반환합니다.

    Parameters
    ----------
    samples       : load_meal_data() 가 반환한 DataFrame
    coach_llm     : vLLM LLM 객체 (Coach)
    user_llm      : vLLM LLM 객체 (User, coach_llm 과 동일 가능)
    config        : SimulationConfig 인스턴스
    already_done  : 이미 완료된 샘플 수 (이어쓰기 시 시작 인덱스)
    on_dialog_end : 대화 하나가 완료될 때마다 호출되는 콜백(idx, result) → None
                    예: 중간 저장, 로깅 등

    Returns
    -------
    List[Dict] : 완료된 대화 결과 목록
    """
    # ── 다이얼로그 컨텍스트 초기화 ──────────────────────────────────────────
    _alignment_llm = alignment_llm if alignment_llm is not None else coach_llm
    meal_tracker      = MealTrackerModel(model=coach_llm, config=config)
    dialog_summarizer = DialogSummarizerModel(model=coach_llm, config=config)
    contexts: List[Dict[str, Any]] = []
    for idx in range(already_done, len(samples)):
        row = samples.iloc[idx]
        contexts.append({
            "idx":          idx,
            "row":          row,
            "history":      SharedConversationHistory(context_window=config.context_window),
            "coach":        InformationSeeker(
                                model=coach_llm,
                                nutrition_goal=row["goal_type"],
                                meal_type=row["meal_type"],
                                config=config,
                            ),
            "user":         UserModel(
                                model=user_llm,
                                nutrition_goal=row["goal_type"],
                                meal_description=row["meal_description"],
                                config=config,
                                meal_ingredient=str(row.get("meal_ingredient", "") or ""),
                            ),
            "alignment":        AlignmentEstimator(
                                model=_alignment_llm,
                                nutrition_goal=row["goal_type"],
                                config=config,
                            ),
            "terminated":   False,
            "terminated_by": "max_turns",
            "dead_end_topics": [],
        })

    results: List[Dict[str, Any]] = []

    print(f"\n[BatchSim] 총 {len(contexts)} 건의 대화를 병렬 처리합니다.")

    for turn_idx in tqdm(range(config.max_turns), desc="[BatchSim] Rounds"):

        # 아직 진행 중인 대화만 추림
        active = [c for c in contexts if not c["terminated"]]
        if not active:
            break

        # ── (1) Coach 발화 배치 ──────────────────────────────────────────
        if turn_idx == 0:
            # 턴 0: 모두 고정 발화
            for ctx in active:
                q = ctx["coach"].first_question()
                ctx["history"].add_turn(turn_idx=0, coach_utterance=q)
        else:
            # 턴 t>0: 배치 LLM 호출
            coach_msgs = [
                ctx["coach"].get_messages(
                    ctx["history"],
                    dead_end_topics=ctx["dead_end_topics"] if ctx["dead_end_topics"] else None,
                    mode="batch",
                )
                for ctx in active
            ]
            coach_replies = batch_generate(
                coach_llm, coach_msgs,
                sampling=config.coach_sampling,
                max_new_tokens=config.max_new_tokens,
                fallback="Could you tell me more about your meal?",
            )
            for ctx, reply in zip(active, coach_replies):
                # Coach 가 실수로 종료 토큰을 출력한 경우 토큰만 제거하고 계속 진행합니다.
                # (종료 조건은 Alignment Tracker 만 담당합니다.)
                cleaned_reply = reply.replace(
                    SharedConversationHistory.TERMINATION_TOKEN, ""
                ).strip()
                if not cleaned_reply:
                    cleaned_reply = "Could you tell me more about your meal?"
                # ── 배치 모드 중복 질문 감지 + 단건 재시도 ──────────────
                _already = ctx["history"].get_all_coach_questions()
                if _is_duplicate(cleaned_reply, _already):
                    _retry_msgs = ctx["coach"].get_messages(
                        ctx["history"],
                        dead_end_topics=ctx["dead_end_topics"] if ctx["dead_end_topics"] else None,
                        mode="batch",
                    ) + [{
                        "role": "user",
                        "content": (
                            "[SYSTEM NOTE: The question you just generated was already asked. "
                            "Please ask about a completely different food item or a new aspect "
                            "that has NOT yet been covered in this conversation.]"
                        ),
                    }]
                    from utils.llm_utils import generate_response as _gen_single
                    _retry = _gen_single(
                        coach_llm, _retry_msgs,
                        max_new_tokens=config.max_new_tokens,
                        sampling=config.coach_sampling,
                    ).strip()
                    if _retry and not _is_duplicate(_retry, _already):
                        cleaned_reply = _retry
                    elif _retry:
                        cleaned_reply = "Could you tell me more about how this meal is put together?"
                ctx["coach"].own_buffer.add(cleaned_reply)
                ctx["history"].add_turn(turn_idx=turn_idx, coach_utterance=cleaned_reply)

        # ── (2) User 응답 배치 ───────────────────────────────────────────
        # Coach 발화 처리 중 종료된 대화는 제외
        still_active = [c for c in active if not c["terminated"]]
        user_msgs = [ctx["user"].get_messages(ctx["history"]) for ctx in still_active]
        user_replies = batch_generate(
            user_llm, user_msgs,
            sampling=config.sampling,
            max_new_tokens=config.max_new_tokens,
            fallback="I'm not sure about that.",
        )
        for ctx, reply in zip(still_active, user_replies):
            # [END] 태그 감지 및 제거
            _natural_end = SharedConversationHistory.TERMINATION_TOKEN in reply
            reply_clean = reply.replace(
                SharedConversationHistory.TERMINATION_TOKEN, ""
            ).strip()
            if not reply_clean:
                reply_clean = "I think that covers everything about my meal."
            ctx["user"].own_buffer.add(reply_clean)
            ctx["history"].update_last_user_utterance(reply_clean)
            # 자연 종료 처리
            if _natural_end:
                ctx["terminated"]    = True
                ctx["terminated_by"] = "natural_end"
            # dead-end 추적: User가 non-answer면 해당 Coach 질문 기록
            if _is_non_answer(reply_clean):
                last_coach_q = ctx["history"].get_all_coach_questions()
                if last_coach_q:
                    ctx["dead_end_topics"].append(last_coach_q[-1])

        # ── (2.5) Alignment Tracker: 정렬 판정 배치 ──────────────────────
        # config.alignment_min_turn 이후부터 매 턴 alignment 를 판정합니다.
        # aligned == True 이면 해당 대화를 즉시 종료합니다.
        # Alignment Tracker 판정이 유일한 정상 종료 조건입니다.
        alignment_active = [ctx for ctx in still_active if not ctx["terminated"]]
        if alignment_active and alignment_active[0]["alignment"].should_evaluate(turn_idx):
            alignment_msgs = [ctx["alignment"].get_messages(ctx["history"]) for ctx in alignment_active]
            alignment_replies = batch_generate(
                _alignment_llm, alignment_msgs,
                sampling=config.alignment_sampling,
                max_new_tokens=config.alignment_max_new_tokens,
                stop_at_newline=False,
                fallback="{}",
            )
            for ctx, reply in zip(alignment_active, alignment_replies):
                aligned = ctx["alignment"].apply_judgment(reply, turn_idx)
                # pred == true_label 일 때만 종료
                # (aligned=True·label=True → 정답 aligned / aligned=False·label=False → 정답 not aligned)
                true_label = (str(ctx["row"].get("expert_result", "yes")).strip().lower() == "yes")
                if aligned == true_label:
                    ctx["terminated"]    = True
                    ctx["terminated_by"] = "alignment"

        # ── (3) MealTracker + DialogSummarizer: 개별 스케줄 ─────
        completed = turn_idx + 1
        for ctx in still_active:
            if not ctx["terminated"]:
                if completed % config.meal_track_every == 0:
                    _update_meal_fact_sheet(ctx["history"], meal_tracker)
                if completed % config.summarize_every == 0:
                    _update_dialog_summary(ctx["history"], dialog_summarizer)

        # 이번 라운드에서 종료된 대화를 results 로 이동
        just_done = [c for c in active if c["terminated"]]
        for ctx in just_done:
            _update_summaries(ctx["history"], meal_tracker, dialog_summarizer)
            result = _build_result(
                ctx["idx"],
                int(ctx["row"]["goal_id"]),
                int(ctx["row"]["id"]),
                ctx["row"]["goal_type"],
                ctx["row"]["meal_type"],
                ctx["row"]["meal_description"],
                ctx["history"],
                ctx["terminated_by"],
                ctx["alignment"],
                expert_result=ctx["row"]["expert_result"],
            )
            results.append(result)
            if on_dialog_end is not None:
                on_dialog_end(ctx["idx"], result)
            print(f"[BatchSim] Dialog {ctx['idx']} 완료 (turn {turn_idx}, by={ctx['terminated_by']})")

    # max_turns 소진 후에도 남은 대화 처리
    remaining = [c for c in contexts if not c["terminated"]]
    for ctx in remaining:
        _update_summaries(ctx["history"], meal_tracker, dialog_summarizer)
        result = _build_result(
            ctx["idx"],
            int(ctx["row"]["goal_id"]),
            int(ctx["row"]["id"]),
            ctx["row"]["goal_type"],
            ctx["row"]["meal_type"],
            ctx["row"]["meal_description"],
            ctx["history"],
            "max_turns",
            ctx["alignment"],
            expert_result=ctx["row"]["expert_result"],
        )
        results.append(result)
        if on_dialog_end is not None:
            on_dialog_end(ctx["idx"], result)

    # idx 순서 정렬 후 반환
    results.sort(key=lambda r: r["id"])
    return results


# ──────────────────────────────────────────────────────────────────────────────
# 내부 헬퍼
# ──────────────────────────────────────────────────────────────────────────────

def _update_summaries(
    history: SharedConversationHistory,
    meal_tracker: MealTrackerModel,
    dialog_summarizer: DialogSummarizerModel,
) -> None:
    """MealTracker + DialogSummarizer 를 사용하여 두 종류의 요약을 갱신합니다."""
    if len(history) == 0:
        return

    conversation_text = history.to_plain_text()
    if not conversation_text.strip():
        return

    # (1) Meal Fact Sheet 갱신 (Alignment Tracker 용)
    history.update_meal_fact_sheet(
        meal_tracker.extract(conversation_text)
    )

    # (2) Dialog Summary 갱신 (Coach/User 용)
    history.update_dialog_summary(
        dialog_summarizer.summarize(conversation_text)
    )


def _update_meal_fact_sheet(
    history: SharedConversationHistory,
    meal_tracker: MealTrackerModel,
) -> None:
    """MealTracker만 사용하여 Meal Fact Sheet를 갱신합니다."""
    if len(history) == 0:
        return
    conversation_text = history.to_plain_text()
    if not conversation_text.strip():
        return
    history.update_meal_fact_sheet(
        meal_tracker.extract(conversation_text)
    )


def _update_dialog_summary(
    history: SharedConversationHistory,
    dialog_summarizer: DialogSummarizerModel,
) -> None:
    """DialogSummarizer만 사용하여 대화 요약을 갱신합니다."""
    if len(history) == 0:
        return
    conversation_text = history.to_plain_text()
    if not conversation_text.strip():
        return
    history.update_dialog_summary(
        dialog_summarizer.summarize(conversation_text)
    )


def _build_result(
    dialog_id:        int,
    goal_id:          int,
    meal_id:          int,
    nutrition_goal:   str,
    meal_type:        str,
    meal_description: str,
    history:          SharedConversationHistory,
    terminated_by:    str,
    alignment_tracker: Optional[AlignmentEstimator] = None,
    expert_result:    str = "yes",
) -> Dict[str, Any]:
    """결과 딕셔너리를 일관된 형식으로 생성합니다."""
    pred_alignment  = alignment_tracker.is_aligned  if alignment_tracker is not None else None
    pred_score      = alignment_tracker.last_score   if alignment_tracker is not None else None
    true_alignment = (expert_result.strip().lower() == "yes")
    if pred_alignment is not None:
        alignment_correct = (pred_alignment == true_alignment)
    else:
        alignment_correct = None

    return {
        "id":                dialog_id,
        "goal_id":           goal_id,
        "meal_id":           meal_id,
        "nutrition_goal":    nutrition_goal,
        "meal_type":         meal_type,
        "meal_description":  meal_description,
        "turns":             history.to_dict_list(),
        "meal_fact_sheet":   history.meal_fact_sheet,
        "dialog_summary":    history.dialog_summary,
        "terminated_by":     terminated_by,
        "pred_alignment":    pred_alignment,
        "pred_score":        pred_score,       # 정규화된 마지막 판정 점수 [0, 1]
        "true_alignment":    true_alignment,
        "alignment_correct": alignment_correct,
        "alignment_history": alignment_tracker.judgment_history if alignment_tracker is not None else [],
        # alignment_history 각 원소: {turn_idx, aligned, score, raw_output}
    }
