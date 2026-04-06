"""
run_simulation.py
─────────────────
실행 진입점.

모든 설정은 config.py 의 SimulationConfig 에서 직접 수정하세요.
실행 명령어:
  python run_simulation.py
"""

import os
import sys


# ──────────────────────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── 임포트 ───────────────────────────────────────────────────────────────
    from config import SimulationConfig
    from core.simulation import set_seed, simulate_conversation, simulate_conversations_batch
    from utils.io_utils import load_meal_data, load_existing_results, save_results, make_output_path
    from utils.llm_utils import load_model

    # ── 설정 로드 (config.py 에서 직접 수정) ──────────────────────────────────
    config = SimulationConfig()

    print("\n" + "="*60)
    print("[Config]")
    print(f"  goal                    : {config.goal}")
    print(f"  data_path               : {config.data_path}")
    print(f"  coach_llm               : {config.coach_llm_repo}")
    print(f"  user_llm                : {config.user_llm_repo}")
    print(f"  alignment_llm               : {config.alignment_llm_repo or '(= coach_llm)'}")
    print(f"  num_gpus                : {config.num_gpus}")
    print(f"  max_model_len           : {config.max_model_len}")
    print(f"  dtype                   : {config.dtype}")
    print(f"  batch_mode              : {config.batch_mode}")
    print(f"  max_turns               : {config.max_turns}")
    print(f"  alignment_min_turn          : {config.alignment_min_turn}")
    print(f"  alignment_sampling          : {config.alignment_sampling}")
    print(f"  alignment_output_format     : {config.alignment_output_format}")
    print(f"  alignment_threshold   : {config.alignment_threshold}")
    print(f"  alignment_use_goal_def      : {config.alignment_use_goal_def}")
    print(f"  alignment_use_workflow      : {config.alignment_use_workflow}")
    print(f"  sampling (coach/user)   : {config.sampling}")
    print(f"  summarize_max_new_tokens: {config.summarize_max_new_tokens}")
    print(f"  seed                    : {config.seed}")
    print("="*60 + "\n")

    # ── 시드 고정 ──────────────────────────────────────────────────────────────
    set_seed(config.seed)

    # ── 모델 로딩 (vLLM — 토크나이저 별도 불필요) ────────────────────────────
    print(f"[*] Coach LLM 로딩: {config.coach_llm_repo}  (tensor_parallel={config.num_gpus})")
    coach_llm = load_model(
        config.coach_llm_repo,
        tensor_parallel_size=config.num_gpus,
        max_model_len=config.max_model_len,
        dtype=config.dtype,
    )

    if config.user_llm_repo == config.coach_llm_repo:
        print("[*] User LLM = Coach LLM (공유)")
        user_llm = coach_llm
    else:
        print(f"[*] User LLM 로딩: {config.user_llm_repo}")
        user_llm = load_model(
            config.user_llm_repo,
            tensor_parallel_size=config.num_gpus,
            max_model_len=config.max_model_len,
            dtype=config.dtype,
        )

    _alignment_repo = config.alignment_llm_repo or config.coach_llm_repo
    if _alignment_repo == config.coach_llm_repo:
        print("[*] Alignment Tracker LLM = Coach LLM (공유)")
        alignment_llm = None   # simulate_conversations_batch 에서 coach_llm 으로 대체
    else:
        print(f"[*] Alignment Tracker LLM 로딩: {_alignment_repo}")
        alignment_llm = load_model(
            _alignment_repo,
            tensor_parallel_size=config.num_gpus,
            max_model_len=config.max_model_len,
            dtype=config.dtype,
        )

    # ── 데이터 로딩 ─────────────────────────────────────────────────────────────
    print(f"\n[*] 데이터 로딩: {config.data_path}")
    data = load_meal_data(config.data_path, config.goal)
    num_samples = len(data)
    print(f"[*] 유효 샘플 수: {num_samples}")

    # ── 출력 경로 설정 ──────────────────────────────────────────────────────────
    model_name  = config.coach_llm_repo.split("/")[-1]
    output_path = make_output_path(
        base_dir=config.output_dir,
        goal=config.goal,
        model_name=model_name,
        max_turns=config.max_turns,
    )
    print(f"[*] 저장 경로: {output_path}")

    results      = load_existing_results(output_path)
    already_done = len(results)
    print(f"[*] 이미 완료된 샘플: {already_done} / {num_samples}\n")

    # ── 시뮬레이션 실행 ─────────────────────────────────────────────────────────
    if config.batch_mode:
        # ── 배치 병렬 모드 ───────────────────────────────────────────────────
        print("[*] 배치 모드 실행: 모든 샘플을 턴 단위로 병렬 처리합니다.")

        def _on_dialog_end(idx: int, result: dict) -> None:
            """대화 하나가 완료될 때마다 즉시 저장합니다."""
            results.append(result)
            save_results(results, output_path)
            print(f"[*] 중간 저장 ({len(results)} 건): {output_path}")

        new_results = simulate_conversations_batch(
            samples=data,
            coach_llm=coach_llm,
            user_llm=user_llm,
            config=config,
            alignment_llm=alignment_llm,
            already_done=already_done,
            on_dialog_end=_on_dialog_end,
        )

        # on_dialog_end 콜백에서 이미 results 에 추가됐으므로 중복 방지
        done_ids = {r["id"] for r in results}
        for r in new_results:
            if r["id"] not in done_ids:
                results.append(r)
        save_results(results, output_path)

    else:
        # ── 단일 순차 모드 ───────────────────────────────────────────────────
        print("[*] 단일 모드 실행: 샘플을 하나씩 처리합니다.")
        from tqdm import tqdm

        for idx in tqdm(range(already_done, num_samples), desc="Dialogs"):
            row = data.iloc[idx]

            print(f"\n{'='*60}")
            print(f"[Sample {idx+1}/{num_samples}] goal={row['goal_type']}  "
                  f"meal={row['meal_description'][:50]}...")
            print(f"{'='*60}")

            dialog_result = simulate_conversation(
                dialog_id=idx,
                goal_id=int(row["goal_id"]),
                nutrition_goal=row["goal_type"],
                meal_id=int(row["id"]),
                meal_type=row["meal_type"],
                meal_description=row["meal_description"],
                coach_llm=coach_llm,
                user_llm=user_llm,
                config=config,
                alignment_llm=alignment_llm,
                expert_result=str(row["expert_result"]),
                meal_ingredient=str(row.get("meal_ingredient", "") or ""),
            )

            results.append(dialog_result)
            save_results(results, output_path)
            print(f"[*] 저장 완료 ({len(results)} 건): {output_path}")

    print(f"\n[Done] 총 {len(results)} 건 저장: {output_path}")


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
