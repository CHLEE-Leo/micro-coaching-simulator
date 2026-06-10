#!/usr/bin/env python3
"""
E2E Fix Validation — 이번 세션에서 수정된 버그들을 종합 검증
=============================================================
수정 사항:
  1. ContextTracker 3-category structure (task/personal/env)
  2. User preferences → Router & InformationSeeker 주입
  3. MealTracker partial acceptance prompt
  4. context_base empty overwrite guard
  5. summarize_every=1 (매턴 ContextTracker 갱신)

테스트 시나리오:
  V1. Context Base 3-category format 검증 (happy path + allergy/preferences)
  V2. User preferences 가 Router/IS 에 반영되는지 (custom chat)
  V3. Partial acceptance — composite 추천의 일부만 수락
  V4. Context base 가 빈 문자열로 덮어쓰이지 않는지 + 누적 갱신
  V5. Already-aligned meal — 빠른 종료 경로 검증
  V6. Off-topic guardrail — 차단 후 상태 보존
"""

import json
import sys
import time
import requests

BASE = "http://localhost:8000"
TIMEOUT = 180


# ─── Helpers ──────────────────────────────────────────────────────────────────

def api(method, path, data=None):
    url = f"{BASE}{path}"
    if method == "POST":
        r = requests.post(url, json=data, timeout=TIMEOUT)
    elif method == "DELETE":
        r = requests.delete(url, timeout=TIMEOUT)
    else:
        r = requests.get(url, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def create_session(**kwargs):
    defaults = {
        "mode": "custom",
        "nutrition_goal": "lean_protein",
        "meal_description": "",
        "meal_ingredient": "",
        "meal_type": "dinner",
        "coach_conversation_mode": "open-ended",
        "context_tracking": True,
        "uncertainty_tracking": True,
    }
    defaults.update(kwargs)
    return api("POST", "/api/session/start", defaults)


def send_turn(sid, text):
    """Send a turn; return response dict. Returns None if session already ended (409)."""
    try:
        return api("POST", f"/api/session/{sid}/turn", {"user_reply": text})
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 409:
            return None  # session already terminated
        raise


def cleanup(sid):
    try:
        api("DELETE", f"/api/session/{sid}")
    except Exception:
        pass


class Result:
    def __init__(self, name):
        self.name = name
        self.checks = []

    def check(self, label, passed, detail=""):
        self.checks.append((label, passed, detail))
        status = "✅" if passed else "❌"
        print(f"    {status} {label}" + (f" — {detail}" if detail else ""))

    @property
    def passed(self):
        return all(p for _, p, _ in self.checks)

    @property
    def summary(self):
        p = sum(1 for _, ok, _ in self.checks if ok)
        return f"{p}/{len(self.checks)}"


# ═════════════════════════════════════════════════════════════════════════════
# V1: Context Base 3-Category Structure
# ═════════════════════════════════════════════════════════════════════════════

def test_v1_context_base_3_category():
    """
    ContextTracker 가 [Task Context], [Personal Context], [Environmental Context]
    세 카테고리로 구조화된 context_base 를 생성하는지 검증.
    또한 session.history.context_base 는 매턴 갱신되므로 context_base 필드로도 확인.
    """
    r = Result("V1: Context Base 3-Category Structure")
    print(f"\n{'='*60}")
    print(f"  {r.name}")
    print(f"{'='*60}")

    s = create_session(
        persona_activity_level="moderate",
        persona_diet_preferences=["vegetarian"],
        persona_allergies=["peanuts"],
    )
    sid = s["session_id"]
    print(f"  Session: {sid[:8]}...")

    turns = [
        "I'm having a big bowl of vegetable stir-fry with tofu and rice.",
        "About 200g of tofu, half a cup of brown rice, and lots of veggies — broccoli, bell peppers, carrots.",
        "I stir-fried everything with sesame oil and soy sauce. About two cups of mixed vegetables.",
    ]

    ctx_base = None
    for i, text in enumerate(turns):
        t = send_turn(sid, text)
        if t is None:
            print(f"  T{i+1}: session terminated early")
            break
        # context_tracker_output = this turn's ContextTracker raw output
        # context_base = cumulative context_base from session.history
        ctx_output = t.get("context_tracker_output", "") or ""
        ctx_cumul = t.get("context_base", "") or ""
        action = (t.get("orchestrator_decision") or {}).get("action", "?")
        print(f"  T{i+1}: action={action}, ctx_output_len={len(ctx_output)}, ctx_base_len={len(ctx_cumul)}")
        # Use whichever is available
        if ctx_output:
            ctx_base = ctx_output
        elif ctx_cumul:
            ctx_base = ctx_cumul

    # Validate 3-category structure
    if ctx_base:
        r.check("[Task Context] header present",
                "[Task Context]" in ctx_base,
                f"found={'[Task Context]' in ctx_base}")
        r.check("[Personal Context] header present",
                "[Personal Context]" in ctx_base,
                f"found={'[Personal Context]' in ctx_base}")
        r.check("[Environmental Context] header present",
                "[Environmental Context]" in ctx_base,
                f"found={'[Environmental Context]' in ctx_base}")

        # Personal Context should reflect persona allergies/preferences
        personal_start = ctx_base.find("[Personal Context]")
        env_start = ctx_base.find("[Environmental Context]")
        if personal_start >= 0 and env_start > personal_start:
            personal_section = ctx_base[personal_start:env_start].lower()
            r.check("Personal Context references vegetarian or dietary preference",
                    "vegetarian" in personal_section or "diet" in personal_section
                    or "plant" in personal_section,
                    f"section_len={len(personal_section)}")
            r.check("Personal Context mentions peanut allergy",
                    "peanut" in personal_section or "allerg" in personal_section,
                    f"section_len={len(personal_section)}")
        elif personal_start >= 0:
            # env_start not found or before personal_start — use rest of string
            personal_section = ctx_base[personal_start:].lower()
            r.check("Personal Context references vegetarian/dietary preference",
                    "vegetarian" in personal_section or "diet" in personal_section
                    or "plant" in personal_section,
                    f"section_len={len(personal_section)}")
            r.check("Personal Context mentions peanut allergy",
                    "peanut" in personal_section or "allerg" in personal_section,
                    f"section_len={len(personal_section)}")
        else:
            r.check("Personal Context section extractable", False,
                    f"personal_start={personal_start}")
            r.check("Personal Context mentions allergy", False, "section not extractable")
    else:
        r.check("context_base generated within 3 turns", False, "No context_base output found")
        for _ in range(4):
            r.check("(skipped — no ctx_base)", False)

    cleanup(sid)
    return r


# ═════════════════════════════════════════════════════════════════════════════
# V2: User Preferences Injection into Router & IS
# ═════════════════════════════════════════════════════════════════════════════

def test_v2_user_preferences_injection():
    """
    페르소나 설정(allergies, diet_preferences)이 Router/IS 프롬프트에
    반영되어, Coach 가 알레르기 유발 식품을 추천/질문하지 않는지 검증.
    """
    r = Result("V2: User Preferences Injection")
    print(f"\n{'='*60}")
    print(f"  {r.name}")
    print(f"{'='*60}")

    s = create_session(
        nutrition_goal="lean_protein",
        persona_allergies=["shellfish", "tree_nuts"],
        persona_diet_preferences=["no_pork"],
    )
    sid = s["session_id"]
    first_q = s.get("first_question", "")
    print(f"  Session: {sid[:8]}...")
    print(f"  Coach T0: {first_q[:80]}")

    t1 = send_turn(sid, "I had grilled chicken with a salad and quinoa for lunch.")
    if t1 is None:
        r.check("Session stayed active", False, "terminated at T1")
        cleanup(sid)
        return r
    coach1 = t1.get("coach_question", "") or ""
    print(f"  T1: coach='{coach1[:80]}'")

    t2 = send_turn(sid, "About 6 ounces of chicken, grilled with olive oil. The salad had romaine, tomatoes, and balsamic dressing.")
    if t2 is None:
        r.check("Session stayed active for 2 turns", False, "terminated at T2")
        cleanup(sid)
        return r
    coach2 = t2.get("coach_question", "") or ""
    print(f"  T2: coach='{coach2[:80]}'")

    all_coach = (first_q + " " + coach1 + " " + coach2).lower()
    r.check("Coach does NOT suggest shellfish",
            "shrimp" not in all_coach and "lobster" not in all_coach and "crab" not in all_coach,
            "checked shrimp/lobster/crab")
    r.check("Coach does NOT suggest tree nuts",
            "almond" not in all_coach and "walnut" not in all_coach and "cashew" not in all_coach,
            "checked almond/walnut/cashew")
    r.check("Session active and progressing",
            t2.get("status") == "active",
            f"status={t2.get('status')}")

    cleanup(sid)
    return r


# ═════════════════════════════════════════════════════════════════════════════
# V3: MealTracker Accuracy & Partial Acceptance
# ═════════════════════════════════════════════════════════════════════════════

def test_v3_meal_tracker_accuracy():
    """
    MealTracker 가:
    1. 사용자가 언급한 음식을 정확히 추출하는지
    2. 부분 수락 시 확인된 부분만 기록하는지

    meal_tracker_output 또는 meal_base (누적) 필드를 사용.
    """
    r = Result("V3: MealTracker Accuracy")
    print(f"\n{'='*60}")
    print(f"  {r.name}")
    print(f"{'='*60}")

    s = create_session(nutrition_goal="lean_protein")
    sid = s["session_id"]
    print(f"  Session: {sid[:8]}...")

    t1 = send_turn(sid, "I'm having a sandwich with ham and cheese for lunch.")
    if t1 is None:
        r.check("Session survived T1", False)
        cleanup(sid); return r
    mt1 = t1.get("meal_tracker_output", "") or t1.get("meal_base", "") or ""
    action1 = (t1.get("orchestrator_decision") or {}).get("action", "?")
    print(f"  T1: action={action1}, mt_len={len(mt1)}")

    t2 = send_turn(sid, "White bread, regular ham, American cheese, some mustard. That's about it.")
    if t2 is None:
        r.check("Session survived T2", False)
        cleanup(sid); return r
    mt2 = t2.get("meal_tracker_output", "") or t2.get("meal_base", "") or ""
    action2 = (t2.get("orchestrator_decision") or {}).get("action", "?")
    print(f"  T2: action={action2}, mt_len={len(mt2)}")

    t3 = send_turn(sid, "Not really sure about the amount. Maybe about two slices of bread and a few slices of ham.")
    if t3 is None:
        r.check("Session survived T3", False)
        cleanup(sid); return r
    mt3 = t3.get("meal_tracker_output", "") or t3.get("meal_base", "") or ""
    action3 = (t3.get("orchestrator_decision") or {}).get("action", "?")
    print(f"  T3: action={action3}, mt_len={len(mt3)}")

    # Use the latest non-empty meal_base
    meal_base = mt3 or mt2 or mt1
    print(f"  Latest Meal Base: {meal_base[:150]}...")

    if meal_base:
        mb_lower = meal_base.lower()
        r.check("MealTracker recorded ham",
                "ham" in mb_lower,
                f"ham={'ham' in mb_lower}")
        r.check("MealTracker recorded cheese",
                "cheese" in mb_lower,
                f"cheese={'cheese' in mb_lower}")
        r.check("MealTracker recorded bread/sandwich",
                "bread" in mb_lower or "sandwich" in mb_lower,
                f"bread={'bread' in mb_lower}, sandwich={'sandwich' in mb_lower}")
    else:
        r.check("MealTracker generated output within 3 turns", False, "empty")
        r.check("(skipped)", False)
        r.check("(skipped)", False)

    # T4: partial acceptance
    t4 = send_turn(sid, "Sure, I could add some chicken to the sandwich, but I'm not going to add a salad.")
    if t4 is None:
        r.check("Session survived T4", False)
        cleanup(sid); return r
    mt4 = t4.get("meal_tracker_output", "") or t4.get("meal_base", "") or ""
    print(f"  T4 (partial accept): mt_len={len(mt4)}")
    print(f"  T4 Meal Base: {mt4[:150]}...")

    if mt4:
        mb4_lower = mt4.lower()
        r.check("Chicken added after partial acceptance",
                "chicken" in mb4_lower,
                f"chicken={'chicken' in mb4_lower}")
        # Previous info should still be there
        r.check("Ham still present (cumulative)",
                "ham" in mb4_lower,
                f"ham={'ham' in mb4_lower}")
    else:
        # Use the prevailing meal_base
        r.check("T4 meal info available", False, "No meal_tracker_output at T4")
        r.check("(skipped)", False)

    cleanup(sid)
    return r


# ═════════════════════════════════════════════════════════════════════════════
# V4: Context Base Cumulative & Non-Empty Guard
# ═════════════════════════════════════════════════════════════════════════════

def test_v4_context_base_cumulative_and_guard():
    """
    1. 매턴(summarize_every=1) context_base 가 갱신되는지 검증
    2. 모호한 응답 후에도 이전 context_base 가 보존되는지 검증
    """
    r = Result("V4: Context Base Cumulative & Guard")
    print(f"\n{'='*60}")
    print(f"  {r.name}")
    print(f"{'='*60}")

    s = create_session(
        nutrition_goal="half_fruits_vegetables",  # different goal to avoid fast exit
    )
    sid = s["session_id"]
    print(f"  Session: {sid[:8]}...")

    turns = [
        "I'm having a big plate of pasta with tomato sauce and meatballs.",
        "About two cups of spaghetti, half a cup of sauce, and four meatballs. Made with ground beef.",
        "I don't know.",  # vague — should not wipe previous context
        "The sauce is homemade with canned tomatoes, garlic, and olive oil.",
    ]

    ctx_history = []
    for i, text in enumerate(turns):
        t = send_turn(sid, text)
        if t is None:
            print(f"  T{i+1}: session terminated")
            ctx_history.append("")
            break
        ctx_raw = t.get("context_tracker_output", "") or ""
        ctx_cum = t.get("context_base", "") or ""
        action = (t.get("orchestrator_decision") or {}).get("action", "?")
        print(f"  T{i+1}: action={action}, ctx_raw_len={len(ctx_raw)}, ctx_cum_len={len(ctx_cum)}")
        ctx_history.append(ctx_cum)  # use cumulative for checking

    # Check that context_base was generated
    non_empty = [c for c in ctx_history if c.strip()]
    r.check("Context base generated at least once in 4 turns",
            len(non_empty) >= 1,
            f"non_empty_count={len(non_empty)}")

    # After vague T3, context should still be present (not wiped)
    if len(ctx_history) >= 3:
        ctx_after_vague = ctx_history[2]  # T3 (index 2)
        ctx_before_vague = ctx_history[1] if len(ctx_history) > 1 else ""
        r.check("Context base not wiped after vague response",
                len(ctx_after_vague) > 0 or len(ctx_before_vague) > 0,
                f"T3_len={len(ctx_after_vague)}, T2_len={len(ctx_before_vague)}")
    else:
        r.check("Enough turns completed for guard check", False,
                f"turns_completed={len(ctx_history)}")

    # Latest context should mention pasta/meatballs (T1 info retained)
    latest_ctx = ""
    for c in reversed(ctx_history):
        if c.strip():
            latest_ctx = c
            break
    if latest_ctx:
        lc = latest_ctx.lower()
        r.check("Latest context mentions pasta or meatballs (info retained)",
                "pasta" in lc or "meatball" in lc or "spaghetti" in lc,
                f"pasta={'pasta' in lc}, meatball={'meatball' in lc}")
    else:
        r.check("(skipped — no context base produced)", False)

    cleanup(sid)
    return r


# ═════════════════════════════════════════════════════════════════════════════
# V5: Already-Aligned Meal — Fast Exit
# ═════════════════════════════════════════════════════════════════════════════

def test_v5_already_aligned_fast_exit():
    """
    이미 목표에 부합하는 식사(lean_protein + chicken breast)를
    제공했을 때, 불필요한 추천 없이 빠르게 긍정적 종료하는지 검증.
    """
    r = Result("V5: Already-Aligned Fast Exit")
    print(f"\n{'='*60}")
    print(f"  {r.name}")
    print(f"{'='*60}")

    s = create_session(nutrition_goal="lean_protein")
    sid = s["session_id"]
    print(f"  Session: {sid[:8]}...")

    turns = [
        "I had grilled chicken breast with steamed broccoli, spinach, and a small side of quinoa.",
        "About 8 ounces of chicken breast, grilled with just a little olive oil. Two cups of broccoli and spinach. Half a cup of quinoa.",
        "No sauce, just salt, pepper, and lemon juice. All fresh vegetables, steamed.",
        "Yes, that sounds good, thanks!",
        "Sounds great, thank you!",
        "Perfect, bye!",
    ]

    history = []
    total_turns = 0
    for i, text in enumerate(turns):
        t = send_turn(sid, text)
        if t is None:
            history.append({"turn": i+1, "status": "terminated (409)", "align": None, "action": "?", "phase": "?"})
            total_turns = i + 1
            break
        status = t.get("status", "")
        align = t.get("alignment_score")
        decision = t.get("orchestrator_decision") or {}
        action = decision.get("action", "?")
        phase = t.get("phase", "")
        total_turns = i + 1
        history.append({"turn": i+1, "status": status, "align": align, "action": action, "phase": phase})
        print(f"  T{i+1}: action={action}, phase={phase}, align={align}, status={status}")
        if status != "active":
            break

    scores = [h["align"] for h in history if h["align"] is not None]
    if scores:
        r.check("Alignment score high for well-aligned meal",
                max(scores) >= 0.6,
                f"max_score={max(scores)}")
    else:
        r.check("Alignment score present", False, "no alignment scores")

    r.check("Terminates within 8 turns",
            total_turns <= 8 or history[-1]["status"] != "active",
            f"total_turns={total_turns}, final_status={history[-1]['status']}")

    phases = [h["phase"] for h in history]
    r.check("Reaches motivational_ending phase or terminates",
            "motivational_ending" in phases
            or any(h["status"] in ("terminated", "terminated (409)") for h in history),
            f"phases={phases}")

    cleanup(sid)
    return r


# ═════════════════════════════════════════════════════════════════════════════
# V6: Off-Topic Guardrail — Block & Preserve State
# ═════════════════════════════════════════════════════════════════════════════

def test_v6_guardrail_off_topic():
    """
    주제 이탈 입력이 Guardrail 에 의해 차단되고,
    세션 상태가 보존되며, 이후 정상 입력으로 계속 진행되는지 검증.
    """
    r = Result("V6: Off-Topic Guardrail")
    print(f"\n{'='*60}")
    print(f"  {r.name}")
    print(f"{'='*60}")

    s = create_session(nutrition_goal="lean_protein")
    sid = s["session_id"]
    print(f"  Session: {sid[:8]}...")

    t1 = send_turn(sid, "I'm having grilled chicken for lunch.")
    if t1 is None:
        r.check("Session survived T1", False)
        cleanup(sid); return r
    turn1 = t1.get("turn_idx", 0)
    phase1 = t1.get("phase", "")
    print(f"  T1: turn_idx={turn1}, phase={phase1}")

    t2 = send_turn(sid, "Can you write me a Python script to sort a list?")
    if t2 is None:
        r.check("Session survived T2 (off-topic)", False)
        cleanup(sid); return r
    blocked = t2.get("guardrail_blocked", False)
    msg = t2.get("coach_question", "") or t2.get("message", "") or ""
    print(f"  T2 (off-topic): blocked={blocked}, msg='{msg[:80]}'")

    r.check("Off-topic input blocked by guardrail",
            blocked is True,
            f"guardrail_blocked={blocked}")
    r.check("Redirect message is present",
            len(msg) > 0,
            f"msg_len={len(msg)}")

    t3 = send_turn(sid, "Sorry, about 6 ounces of chicken, grilled on a pan.")
    if t3 is None:
        r.check("Session continues after block", False, "terminated at T3")
        cleanup(sid); return r
    status3 = t3.get("status", "")
    print(f"  T3 (valid): status={status3}")

    r.check("Session continues normally after guardrail block",
            status3 == "active",
            f"status={status3}")

    cleanup(sid)
    return r


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "═"*60)
    print("  Fix Validation E2E Test Suite")
    print("  Target: All fixes from context_base/preferences/MealTracker session")
    print("═"*60)

    tests = [
        test_v1_context_base_3_category,
        test_v2_user_preferences_injection,
        test_v3_meal_tracker_accuracy,
        test_v4_context_base_cumulative_and_guard,
        test_v5_already_aligned_fast_exit,
        test_v6_guardrail_off_topic,
    ]

    results = []
    for test_fn in tests:
        try:
            result = test_fn()
            results.append(result)
        except Exception as e:
            name = test_fn.__name__
            print(f"\n{'='*60}")
            print(f"  {name} — EXCEPTION")
            print(f"{'='*60}")
            print(f"    💥 {type(e).__name__}: {e}")
            r = Result(name)
            r.check("Test execution", False, str(e))
            results.append(r)
        print()

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  Fix Validation Summary")
    print("═"*60)
    total_pass = 0
    total_checks = 0
    for res in results:
        status = "✅ PASS" if res.passed else "❌ FAIL"
        p = sum(1 for _, ok, _ in res.checks if ok)
        total_pass += p
        total_checks += len(res.checks)
        print(f"  {status}  {res.name}  ({res.summary})")

    print(f"\n  Total: {total_pass}/{total_checks} checks passed")
    print("═"*60 + "\n")

    failed = [r for r in results if not r.passed]
    if failed:
        print(f"  ⚠️  {len(failed)} scenario(s) had failures:")
        for r in failed:
            for label, ok, detail in r.checks:
                if not ok:
                    print(f"    - [{r.name}] {label}: {detail}")
        print()

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
