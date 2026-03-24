"""
core/simulation.py
──────────────────
Coach ↔ User 대화 시뮬레이션 오케스트레이터.

두 가지 실행 모드를 제공합니다:

  [단일 모드]  simulate_conversation()
    한 건의 식사 샘플에 대해 순차적으로 대화를 진행합니다.
    디버깅·소규모 실험에 적합합니다.

  [배치 병렬 모드]  simulate_conversations_batch()
    N 건의 다이얼로그를 동시에 진행하며 매 턴마다 Coach/User 발화를
    vLLM 의 batch_generate() 로 한 번에 처리합니다.
    GPU 가동률을 극대화하여 처리량(throughput)이 크게 향상됩니다.

흐름 (두 모드 공통)
  turn 0  : Coach 고정 발화 → User LLM 응답 → 요약 스케줄 확인
  turn t>0: Coach LLM 질문 → User LLM 응답 → Judge 판정 → 요약 스케줄 확인
  종료    : Judge 가 aligned 판정 내림 또는 max_turns 초과

책임 분리
  - 메모리 관리  : core/memory.py  (SharedConversationHistory, ConversationBuffer)
  - 발화 생성    : models/coach.py, models/user.py
  - LLM 추론     : utils/llm_utils.py  (generate_response, batch_generate)
  - 설정         : config.py
"""

from __future__ import annotations

import random
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from config import SimulationConfig
from core.memory import SharedConversationHistory
from models.coach  import CoachModel
from models.judge  import JudgeModel
from models.user   import UserModel
from utils.llm_utils import batch_generate, summarize_conversation


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
    expert_result     : 전문가 레이블 ("yes" | "not_really") — 정확도 추적용

    Returns
    -------
    dict : {id, goal_id, meal_id, nutrition_goal, meal_type, meal_description,
            turns, summary, terminated_by, pred_alignment, true_alignment, alignment_correct}
    """
    # ── 에이전트 초기화 ─────────────────────────────────────────────────────
    coach = CoachModel(
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

    # ── 공통 대화 기록 초기화 ───────────────────────────────────────────────
    history = SharedConversationHistory(context_window=config.context_window)

    terminated_by = "max_turns"

    # ── 턴 루프 ─────────────────────────────────────────────────────────────
    with tqdm(total=config.max_turns, desc=f"[Dialog {dialog_id}] Turns") as pbar:

        for turn_idx in range(config.max_turns):

            # ── (1) Coach 발화 ───────────────────────────────────────────
            if turn_idx == 0:
                coach_utterance = coach.first_question()
            else:
                coach_utterance = coach.ask(history)

            print(f"\n[T{turn_idx}] Coach : {coach_utterance}")
            history.add_turn(turn_idx=turn_idx, coach_utterance=coach_utterance)

            # ── (2) User 응답 ──────────────────────────────────────────
            user_utterance = user.respond(history)
            print(f"[T{turn_idx}] User  : {user_utterance}")
            history.update_last_user_utterance(user_utterance)

            pbar.update(1)

            # ── (3) 요약 갱신 스케줄  (Principle 4) ────────────────
            # 종료 조건은 Judge 만 담당합니다 (단일 모드는 Judge 미포함, max_turns 상한만 사용).
            completed = turn_idx + 1
            if completed % config.summarize_every == 0:
                _update_summary(history, coach_llm, config)

    # 루프 종료 후 최종 요약 갱신
    _update_summary(history, coach_llm, config)

    return _build_result(dialog_id, goal_id, meal_id, nutrition_goal,
                         meal_type, meal_description, history, terminated_by,
                         expert_result=expert_result)


# ──────────────────────────────────────────────────────────────────────────────
# [배치 병렬 모드] N 건 동시 처리
# ──────────────────────────────────────────────────────────────────────────────

def simulate_conversations_batch(
    samples:      pd.DataFrame,
    coach_llm,
    user_llm,
    config:       SimulationConfig,
    judge_llm     = None,
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
    _judge_llm = judge_llm if judge_llm is not None else coach_llm
    contexts: List[Dict[str, Any]] = []
    for idx in range(already_done, len(samples)):
        row = samples.iloc[idx]
        contexts.append({
            "idx":          idx,
            "row":          row,
            "history":      SharedConversationHistory(context_window=config.context_window),
            "coach":        CoachModel(
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
            "judge":        JudgeModel(
                                model=_judge_llm,
                                nutrition_goal=row["goal_type"],
                                config=config,
                            ),
            "terminated":   False,
            "terminated_by": "max_turns",
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
            coach_msgs = [ctx["coach"].get_messages(ctx["history"]) for ctx in active]
            coach_replies = batch_generate(
                coach_llm, coach_msgs,
                sampling=config.sampling,
                max_new_tokens=config.max_new_tokens,
                fallback="Could you tell me more about your meal?",
            )
            for ctx, reply in zip(active, coach_replies):
                # Coach 가 실수로 종료 토큰을 출력한 경우 토큰만 제거하고 계속 진행합니다.
                # (종료 조건은 Judge 만 담당합니다.)
                cleaned_reply = reply.replace(
                    SharedConversationHistory.TERMINATION_TOKEN, ""
                ).strip()
                if not cleaned_reply:
                    cleaned_reply = "Could you tell me more about your meal?"
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
            ctx["user"].own_buffer.add(reply)
            ctx["history"].update_last_user_utterance(reply)

        # ── (2.5) Judge 판정 배치 ──────────────────────────────────────
        # config.judge_min_turn 이후부터 매 턴 alignment 를 판정합니다.
        # aligned == True 이면 해당 대화를 즉시 종료합니다.
        # Judge 판정이 유일한 정상 종료 조건입니다.
        judge_active = [ctx for ctx in still_active if not ctx["terminated"]]
        if judge_active and judge_active[0]["judge"].should_judge(turn_idx):
            judge_msgs = [ctx["judge"].get_messages(ctx["history"]) for ctx in judge_active]
            judge_replies = batch_generate(
                _judge_llm, judge_msgs,
                sampling=config.judge_sampling,
                max_new_tokens=config.judge_max_new_tokens,
                stop_at_newline=False,
                fallback="{}",
            )
            for ctx, reply in zip(judge_active, judge_replies):
                aligned = ctx["judge"].apply_judgment(reply, turn_idx)
                # pred == true_label 일 때만 종료
                # (aligned=True·label=True → 정답 aligned / aligned=False·label=False → 정답 not aligned)
                true_label = (str(ctx["row"].get("expert_result", "yes")).strip().lower() == "yes")
                if aligned == true_label:
                    ctx["terminated"]    = True
                    ctx["terminated_by"] = "judge"

        # ── (3) 요약 갱신 스케줄 ──────────────────────────────────────
        # 종료 조건: Judge aligned (step 2.5) 또는 max_turns 소진 (외부 루프).
        # User 종료 토큰에 의한 정상 종료는 없습니다.
        completed = turn_idx + 1
        for ctx in still_active:
            if not ctx["terminated"] and completed % config.summarize_every == 0:
                _update_summary(ctx["history"], coach_llm, config)

        # 이번 라운드에서 종료된 대화를 results 로 이동
        just_done = [c for c in active if c["terminated"]]
        for ctx in just_done:
            _update_summary(ctx["history"], coach_llm, config)
            result = _build_result(
                ctx["idx"],
                int(ctx["row"]["goal_id"]),
                int(ctx["row"]["id"]),
                ctx["row"]["goal_type"],
                ctx["row"]["meal_type"],
                ctx["row"]["meal_description"],
                ctx["history"],
                ctx["terminated_by"],
                ctx["judge"],
                expert_result=ctx["row"]["expert_result"],
            )
            results.append(result)
            if on_dialog_end is not None:
                on_dialog_end(ctx["idx"], result)
            print(f"[BatchSim] Dialog {ctx['idx']} 완료 (turn {turn_idx}, by={ctx['terminated_by']})")

    # max_turns 소진 후에도 남은 대화 처리
    remaining = [c for c in contexts if not c["terminated"]]
    for ctx in remaining:
        _update_summary(ctx["history"], coach_llm, config)
        result = _build_result(
            ctx["idx"],
            int(ctx["row"]["goal_id"]),
            int(ctx["row"]["id"]),
            ctx["row"]["goal_type"],
            ctx["row"]["meal_type"],
            ctx["row"]["meal_description"],
            ctx["history"],
            "max_turns",
            ctx["judge"],
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

def _update_summary(
    history: SharedConversationHistory,
    llm,
    config: SimulationConfig,
) -> None:
    """현재까지의 대화를 요약하여 history.summary 를 갱신합니다."""
    if len(history) == 0:
        return

    conversation_text = "\n".join(
        f"Coach: {t['coach_utterance']}\nUser: {t['user_utterance']}"
        for t in history.to_dict_list()
        if t.get("coach_utterance") and t.get("user_utterance")
    )
    if conversation_text.strip():
        history.summary = summarize_conversation(
            llm,
            conversation_text,
            max_new_tokens=config.summarize_max_new_tokens,
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
    judge:            Optional[JudgeModel] = None,
    expert_result:    str = "yes",
) -> Dict[str, Any]:
    """결과 딕셔너리를 일관된 형식으로 생성합니다."""
    pred_alignment  = judge.is_aligned  if judge is not None else None
    pred_score      = judge.last_score   if judge is not None else None
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
        "summary":           history.summary,
        "terminated_by":     terminated_by,
        "pred_alignment":    pred_alignment,
        "pred_score":        pred_score,       # 정규화된 마지막 판정 점수 [0, 1]
        "true_alignment":    true_alignment,
        "alignment_correct": alignment_correct,
        "alignment_history": judge.judgment_history if judge is not None else [],
        # alignment_history 각 원소: {turn_idx, aligned, score, raw_output}
    }
