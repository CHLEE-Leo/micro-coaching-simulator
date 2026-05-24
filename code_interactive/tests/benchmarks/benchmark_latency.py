#!/usr/bin/env python3
"""
benchmark_latency.py
────────────────────
LLM 파이프라인 응답 속도 병목 분석 스크립트.
submit_reply() 내부의 각 LLM 호출 단계별 소요 시간을 측정합니다.

/ Latency benchmark for the LLM pipeline.
Measures wall-clock time of each LLM call stage inside submit_reply().
"""
import sys, time, functools
from pathlib import Path

# ── 경로 설정
_INTERACTIVE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_INTERACTIVE))

# ── Monkey-patch: SessionManager._run_module_inference 에 타이밍 래퍼 주입
import session_manager as _sm

_call_log = []  # list of {"step": str, "model": str, "elapsed_ms": float}

_original_run_module_inference = _sm.SessionManager._run_module_inference

@functools.wraps(_original_run_module_inference)
def _timed_run_module_inference(self, *, module, messages, mode, agent_config=None):
    client = self._client_for(module)
    model_name = getattr(client, 'model_name', '?') if client else '?'
    t0 = time.perf_counter()
    result = _original_run_module_inference(
        self,
        module=module,
        messages=messages,
        mode=mode,
        agent_config=agent_config,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    _call_log.append({"step": module, "mode": mode, "model": model_name, "elapsed_ms": round(elapsed_ms, 1)})
    return result

_sm.SessionManager._run_module_inference = _timed_run_module_inference


def run_benchmark():
    from web_app_config import WebAppConfig
    from agents.openai_client import load_model
    config = WebAppConfig()
    # Build client pool as app.py does
    heavy_client = load_model(config.chatgpt_model)
    light_client = load_model(config.chatgpt_light_model)
    pool = {
        config.chatgpt_model: heavy_client,
        config.chatgpt_light_model: light_client,
    }
    mgr = _sm.SessionManager(chatgpt_client_pool=pool, config=config)

    print("=" * 70)
    print("  Latency Benchmark — LLM Pipeline Bottleneck Analysis")
    print("=" * 70)

    # ─── Scenario 1: inquire (most common path) ───
    print("\n▶ Scenario 1: inquire (typical first turns)")
    sess = mgr.create_session(
        nutrition_goal="lean_protein",
        meal_description="I had a sandwich for lunch",
        meal_ingredient="turkey, cheese, wheat bread, lettuce, tomato",
        meal_type="lunch",
        mode="custom",
        context_tracking=True,
    )
    sid = sess.session_id

    user_replies = [
        "It was a turkey and cheese sandwich on wheat bread with some lettuce and tomato",
        "I also had a small bag of chips and a glass of water",
        "About a regular size portion, maybe 6 inches",
    ]

    for i, reply in enumerate(user_replies):
        _call_log.clear()
        t_start = time.perf_counter()
        result = mgr.submit_reply(sid, reply)
        t_total = (time.perf_counter() - t_start) * 1000

        action = (result.get("orchestrator_decision") or {}).get("action", "?")
        phase = result.get("phase", "?")
        status = result.get("status", "?")

        print(f"\n  Turn {i+1} — action={action}, phase={phase}, status={status}")
        print(f"  {'Step':<22} {'Model':<18} {'Latency (ms)':>12}")
        print(f"  {'─'*22} {'─'*18} {'─'*12}")

        total_llm = 0
        for entry in _call_log:
            print(f"  {entry['step']:<22} {entry['model']:<18} {entry['elapsed_ms']:>10.1f}ms")
            total_llm += entry['elapsed_ms']

        overhead = t_total - total_llm
        print(f"  {'─'*22} {'─'*18} {'─'*12}")
        print(f"  {'LLM total':<22} {'':<18} {total_llm:>10.1f}ms")
        print(f"  {'Python overhead':<22} {'':<18} {overhead:>10.1f}ms")
        print(f"  {'WALL CLOCK':<22} {'':<18} {t_total:>10.1f}ms")
        print(f"  LLM calls: {len(_call_log)}")

        if status != "active":
            break

    # ─── Scenario 2: assess path (heaviest) ───
    print("\n\n▶ Scenario 2: assess path (assessment + follow-up)")
    sess2 = mgr.create_session(
        nutrition_goal="lean_protein",
        meal_description="I'm having grilled chicken breast with steamed broccoli and brown rice",
        meal_ingredient="chicken breast, broccoli, brown rice",
        meal_type="dinner",
        mode="custom",
        context_tracking=True,
    )
    sid2 = sess2.session_id

    eval_replies = [
        "About 6 ounces of chicken, a cup of broccoli, and half a cup of rice",
        "I grilled the chicken with just a little olive oil and salt",
        "That's pretty much all I'm having for dinner",
    ]

    for i, reply in enumerate(eval_replies):
        _call_log.clear()
        t_start = time.perf_counter()
        result = mgr.submit_reply(sid2, reply)
        t_total = (time.perf_counter() - t_start) * 1000

        action = (result.get("orchestrator_decision") or {}).get("action", "?")
        phase = result.get("phase", "?")
        status = result.get("status", "?")

        print(f"\n  Turn {i+1} — action={action}, phase={phase}, status={status}")
        print(f"  {'Step':<22} {'Model':<18} {'Latency (ms)':>12}")
        print(f"  {'─'*22} {'─'*18} {'─'*12}")

        total_llm = 0
        for entry in _call_log:
            print(f"  {entry['step']:<22} {entry['model']:<18} {entry['elapsed_ms']:>10.1f}ms")
            total_llm += entry['elapsed_ms']

        overhead = t_total - total_llm
        print(f"  {'─'*22} {'─'*18} {'─'*12}")
        print(f"  {'LLM total':<22} {'':<18} {total_llm:>10.1f}ms")
        print(f"  {'Python overhead':<22} {'':<18} {overhead:>10.1f}ms")
        print(f"  {'WALL CLOCK':<22} {'':<18} {t_total:>10.1f}ms")
        print(f"  LLM calls: {len(_call_log)}")

        if status != "active":
            break

    # ─── Summary ───
    print("\n" + "=" * 70)
    print("  Summary: Critical Path Analysis")
    print("=" * 70)
    print("""
  Current serial chain (inquire path):
    [Input Guard + MT + CT] -> [AE] -> [CE optional] -> [Phase Predictor] -> [Orchestrator] -> [IS] -> [Response Generator]
     light group          light    light      heavy/med          heavy/high      heavy/med  heavy/none

  The input guard, meal tracker, and context tracker are launched together.

  assess path adds: +Assessment +Assessment Response +Post Assessment Routing +(action)

  Key observation:
    - Orchestrator (heavy, reasoning=high) is usually the largest bottleneck.
    - InformationSeeker/MealRecommender (heavy, reasoning=medium) often follow.
    - ResponseGenerator (heavy, reasoning=none) is still user-visible latency.
    - Light modules are individually fast but still add up across a full turn.
""")


if __name__ == "__main__":
    run_benchmark()
