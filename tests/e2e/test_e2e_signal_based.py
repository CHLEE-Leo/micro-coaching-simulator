#!/usr/bin/env python3
"""
E2E 시나리오 테스트 — Signal-Based Router 검증
================================================
Estimator 시그널(alignment score/reasoning, certainty score/reasoning)이
Router 의사결정에 구조적으로 반영되는지 검증한다.

테스트 시나리오:
  S1. Happy Path (lean_protein, dinner)
  S2. Implicit recommendation request (choice uncertainty)
  S3. Incomplete meal — premature close prevention
  S4. Vague responder — stall exit
  S5. Already-aligned meal — fast exit
  S6. Off-topic guardrail
"""

import json
import sys
import time
import requests

BASE = "http://localhost:8765"
TIMEOUT = 180  # seconds per turn


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
        "context_tracking": False,
        "uncertainty_tracking": True,
    }
    defaults.update(kwargs)
    return api("POST", "/api/session/start", defaults)


def send_turn(sid, text):
    return api("POST", f"/api/session/{sid}/turn", {"user_reply": text})


def cleanup(sid):
    try:
        api("DELETE", f"/api/session/{sid}")
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────

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
# S1: Happy Path — 성실한 사용자, 정상적인 정보 흐름
# ═════════════════════════════════════════════════════════════════════════════

def test_s1_happy_path():
    r = Result("S1: Happy Path (lean_protein dinner)")
    print(f"\n{'='*60}")
    print(f"  {r.name}")
    print(f"{'='*60}")

    s = create_session()
    sid = s["session_id"]
    print(f"  Session: {sid[:8]}...")
    print(f"  Coach T0: {s['first_question'][:80]}")

    turns = [
        "I'm having grilled chicken breast with steamed broccoli and white rice.",
        "About 6 ounces of chicken, maybe a cup of rice and a cup of broccoli.",
        "Just olive oil and salt, grilled on the skillet.",
        "Sure, I'm open to suggestions.",
    ]

    history = []
    for i, text in enumerate(turns):
        t = send_turn(sid, text)
        coach = t.get("coach_question", "")
        status = t.get("status", "")
        align = t.get("alignment_score")
        cert = t.get("certainty_score")
        mon = t.get("monitoring", {})
        action = mon.get("planner_action", "")
        phase = mon.get("current_phase", "")
        history.append({
            "turn": i + 1, "user": text[:60], "coach": coach[:80],
            "status": status, "align": align, "cert": cert,
            "action": action, "phase": phase,
        })
        print(f"  T{i+1}: user='{text[:50]}...' → action={action}, align={align}, cert={cert}, status={status}")
        if status != "active":
            break

    # Check: session should still be active or properly terminated (not stuck)
    last = history[-1]

    # Alignment should increase as more info is provided
    scores = [h["align"] for h in history if h["align"] is not None]
    if len(scores) >= 2:
        r.check("Alignment increases with info", scores[-1] >= scores[0],
                f"first={scores[0]}, last={scores[-1]}")
    else:
        r.check("Alignment increases with info", False, "Not enough alignment scores")

    # Certainty should increase
    certs = [h["cert"] for h in history if h["cert"] is not None]
    if len(certs) >= 2:
        r.check("Certainty increases with info", certs[-1] >= certs[0],
                f"first={certs[0]}, last={certs[-1]}")
    else:
        r.check("Certainty increases with info", False, "Not enough certainty scores")

    # If session ended, check it ended properly; if still active, continue
    if last["status"] == "active":
        # Continue until recommendation or max 6 more turns
        for j in range(6):
            extra_texts = [
                "Sounds good, I'll try that!",
                "Sure, I can do that.",
                "OK, anything else?",
                "Yes, that works for me.",
                "Great idea.",
                "I'll do that, thanks!",
            ]
            t = send_turn(sid, extra_texts[j])
            status = t.get("status", "")
            align = t.get("alignment_score")
            mon = t.get("monitoring", {})
            action = mon.get("planner_action", "")
            print(f"  T{len(history)+j+1}: '{extra_texts[j][:40]}' → action={action}, align={align}, status={status}")
            if status != "active":
                last_status = status
                break
        else:
            last_status = "active"
        r.check("Conversation terminates gracefully",
                last_status in ("terminated", "max_turns"),
                f"final_status={last_status}")
    else:
        r.check("Conversation terminates gracefully", True, f"final_status={last['status']}")

    cleanup(sid)
    return r


# ═════════════════════════════════════════════════════════════════════════════
# S2: Implicit recommendation — user uncertain about choice
# ═════════════════════════════════════════════════════════════════════════════

def test_s2_implicit_recommendation():
    """S2: Signal-based behavior when user is uncertain about choices.
    
    Key validation: estimator signals (low alignment, low certainty) should
    drive the Router to keep the conversation productive — whether by
    seeking more info about the full meal, pivoting to recommendations,
    or answering the user's implicit question. The session should NOT
    terminate prematurely with only coffee for dinner.
    """
    r = Result("S2: User uncertainty + incomplete dinner signals")
    print(f"\n{'='*60}")
    print(f"  {r.name}")
    print(f"{'='*60}")

    s = create_session()
    sid = s["session_id"]
    print(f"  Session: {sid[:8]}...")

    # T1: coffee with unknown milk choice
    t1 = send_turn(sid, "A coffee... prob w/ soybean or whole milk.")
    align1 = t1.get("alignment_score")
    cert1 = t1.get("certainty_score")
    print(f"  T1: align={align1}, cert={cert1}")

    # T2: user signals uncertainty
    t2 = send_turn(sid, "I don't know. I'm not even sure which milk to use.")
    coach2 = t2.get("coach_question", "")
    align2 = t2.get("alignment_score")
    cert2 = t2.get("certainty_score")
    print(f"  T2: align={align2}, cert={cert2}")
    print(f"  Coach T2: {coach2[:120]}")

    # Signal check: alignment should be very low (coffee-only dinner)
    r.check("Alignment stays low (coffee-only dinner)",
            align2 is not None and align2 < 0.4,
            f"align={align2}")

    # T3: user makes a choice
    t3 = send_turn(sid, "OK, let's go with soy milk then.")
    status3 = t3.get("status", "")
    align3 = t3.get("alignment_score")
    cert3 = t3.get("certainty_score")
    print(f"  T3: status={status3}, align={align3}, cert={cert3}")

    # Critical signal-based check: low alignment → session must not close
    r.check("Session stays active (alignment too low to close)",
            status3 == "active",
            f"status={status3}, align={align3}")

    # Continue — coach should eventually ask about more food
    found_broader_question = False
    if status3 == "active":
        follow_ups = [
            "About half a cup of soy milk.",
            "No sugar, just black coffee with soy milk.",
            "That's it for the coffee.",
        ]
        for i, text in enumerate(follow_ups):
            t = send_turn(sid, text)
            coach = t.get("coach_question", "").lower()
            status = t.get("status", "")
            print(f"  T{4+i}: '{text[:40]}' → coach='{coach[:80]}', status={status}")
            # Check if coach broadens to full dinner
            if any(w in coach for w in ["anything else", "additional", "dinner",
                                         "protein", "besides", "main", "other food",
                                         "planning", "rest of", "eat"]):
                found_broader_question = True
                break
            if status != "active":
                break

    r.check("Coach eventually explores full dinner (signal-driven)",
            found_broader_question,
            "Coach should broaden scope when alignment remains low")

    cleanup(sid)
    return r


# ═════════════════════════════════════════════════════════════════════════════
# S3: Incomplete meal — premature close prevention via signals
# ═════════════════════════════════════════════════════════════════════════════

def test_s3_incomplete_meal_no_premature_close():
    r = Result("S3: Incomplete meal — no premature close")
    print(f"\n{'='*60}")
    print(f"  {r.name}")
    print(f"{'='*60}")

    s = create_session()
    sid = s["session_id"]
    print(f"  Session: {sid[:8]}...")

    # Only mention a side dish for dinner
    t1 = send_turn(sid, "Just a small salad with lettuce and tomato.")
    print(f"  T1: align={t1.get('alignment_score')}, cert={t1.get('certainty_score')}")

    t2 = send_turn(sid, "Just olive oil dressing, maybe half a bowl.")
    print(f"  T2: align={t2.get('alignment_score')}, cert={t2.get('certainty_score')}")

    # Accept any suggestion immediately
    t3 = send_turn(sid, "Sure, that sounds good!")
    status3 = t3.get("status", "")
    align3 = t3.get("alignment_score")
    m3 = t3.get("monitoring", {})
    action3 = m3.get("planner_action", "")
    print(f"  T3: action={action3}, status={status3}, align={align3}")

    # Key check: with only a small salad for dinner, alignment should be low
    # Router should NOT close — should ask about more food
    r.check("Low alignment (salad-only dinner)",
            align3 is not None and align3 < 0.6,
            f"align={align3}")

    r.check("Session stays active (not prematurely closed)",
            status3 == "active",
            f"status={status3}")

    cleanup(sid)
    return r


# ═════════════════════════════════════════════════════════════════════════════
# S4: Vague responder — stall exit
# ═════════════════════════════════════════════════════════════════════════════

def test_s4_vague_responder():
    """S4: Vague responder — stall exit across phase transitions.
    
    Stall exit fires after stall_exit_turns (3) consecutive non-answers
    per phase. First stall -> assess -> enters recommendation.
    Second stall → forces recommend → enters recommendation.
    Third stall → forces close → terminated.
    So we need enough turns to trigger stall exits across multiple phases.
    """
    r = Result("S4: Vague responder — stall exit")
    print(f"\n{'='*60}")
    print(f"  {r.name}")
    print(f"{'='*60}")

    s = create_session()
    sid = s["session_id"]
    print(f"  Session: {sid[:8]}...")

    # Start with one real answer, then all vague
    # T1 provides real info (resets stall), T2-T4 stall in exploration (stall=3 → assess)
    # T5-T7 stall in recommendation (stall=3 → recommend)
    # T8-T10 stall in recommendation (stall=3 → close → terminated)
    vague_replies = [
        "Some chicken I think.",       # INFO: resets stall
        "Not sure, maybe grilled?",     # NON-ANS: stall=1
        "I don't know the amount.",     # NON-ANS: stall=2
        "I haven't decided yet.",       # NON-ANS: stall=3 -> assess
        "I'm not sure about that.",     # NON-ANS: stall=1 (new phase)
        "I don't know.",               # NON-ANS: stall=2
        "No idea.",                     # NON-ANS: stall=3 → recommend
        "I'm unsure.",                 # NON-ANS: stall=1 (new phase)
        "I haven't decided.",          # NON-ANS: stall=2
        "I don't know.",               # NON-ANS: stall=3 → close
    ]

    last_status = "active"
    for i, text in enumerate(vague_replies):
        if last_status != "active":
            break
        t = send_turn(sid, text)
        last_status = t.get("status", "")
        m = t.get("monitoring", {})
        action = m.get("planner_action", "")
        align = t.get("alignment_score")
        cert = t.get("certainty_score")
        print(f"  T{i+1}: '{text[:40]}' → action={action}, status={last_status}, align={align}, cert={cert}")

    # Check: conversation should end within the allotted stall rounds
    r.check("Conversation ends after repeated vague answers",
            last_status != "active",
            f"status={last_status}")

    cleanup(sid)
    return r


# ═════════════════════════════════════════════════════════════════════════════
# S5: Already-aligned meal — fast exit
# ═════════════════════════════════════════════════════════════════════════════

def test_s5_already_aligned():
    r = Result("S5: Already-aligned meal — fast exit")
    print(f"\n{'='*60}")
    print(f"  {r.name}")
    print(f"{'='*60}")

    s = create_session()
    sid = s["session_id"]
    print(f"  Session: {sid[:8]}...")

    # Provide a meal that already meets lean_protein goal
    t1 = send_turn(sid, "I'm having a large grilled chicken breast with steamed broccoli and a quinoa salad.")
    align1 = t1.get("alignment_score")
    cert1 = t1.get("certainty_score")
    status1 = t1.get("status", "")
    print(f"  T1: align={align1}, cert={cert1}, status={status1}")

    # The desired fast-exit behavior may terminate immediately when the first
    # meal description is already specific and well aligned.
    if status1 in ("terminated", "max_turns"):
        r.check("High alignment for well-balanced meal",
                align1 is not None and align1 >= 0.6,
                f"align={align1}")
        r.check("Conversation is efficient (≤ 8 turns)",
                True,
                "turns=1")
        cleanup(sid)
        return r

    t2 = send_turn(sid, "About 8 oz chicken, 1 cup broccoli, 1 cup quinoa. Grilled with minimal oil.")
    align2 = t2.get("alignment_score")
    cert2 = t2.get("certainty_score")
    status2 = t2.get("status", "")
    m2 = t2.get("monitoring", {})
    action2 = m2.get("planner_action", "")
    print(f"  T2: action={action2}, align={align2}, cert={cert2}, status={status2}")

    # High alignment → should either terminate or move quickly to motivational close
    r.check("High alignment for well-balanced meal",
            align2 is not None and align2 >= 0.6,
            f"align={align2}")

    # If still active, one more turn should wrap it up
    total_turns = 2
    if status2 == "active":
        t3 = send_turn(sid, "That's all, nothing else to add.")
        status3 = t3.get("status", "")
        m3 = t3.get("monitoring", {})
        action3 = m3.get("planner_action", "")
        print(f"  T3: action={action3}, status={status3}")
        total_turns = 3

        if status3 == "active":
            t4 = send_turn(sid, "Yep, that's everything.")
            status4 = t4.get("status", "")
            print(f"  T4: status={status4}")
            total_turns = 4

    r.check("Conversation is efficient (≤ 8 turns)",
            total_turns <= 8,
            f"turns={total_turns}")

    cleanup(sid)
    return r


# ═════════════════════════════════════════════════════════════════════════════
# S6: Off-topic guardrail
# ═════════════════════════════════════════════════════════════════════════════

def test_s6_off_topic():
    r = Result("S6: Off-topic guardrail")
    print(f"\n{'='*60}")
    print(f"  {r.name}")
    print(f"{'='*60}")

    s = create_session()
    sid = s["session_id"]
    print(f"  Session: {sid[:8]}...")

    # Off-topic message
    t1 = send_turn(sid, "Can you help me write a Python script?")
    blocked = t1.get("guardrail_blocked", False)
    status1 = t1.get("status", "")
    coach1 = t1.get("coach_question", "")
    print(f"  T1 (off-topic): blocked={blocked}, coach='{coach1[:80]}'")

    r.check("Guardrail blocks off-topic",
            blocked is True,
            f"blocked={blocked}")

    r.check("Session stays active after guardrail",
            status1 == "active",
            f"status={status1}")

    # Normal message should work after guardrail
    t2 = send_turn(sid, "Oh sorry, I'm having grilled salmon with rice.")
    status2 = t2.get("status", "")
    coach2 = t2.get("coach_question", "")
    blocked2 = t2.get("guardrail_blocked", False)
    print(f"  T2 (on-topic): blocked={blocked2}, coach='{coach2[:80]}'")

    r.check("On-topic message passes guardrail",
            blocked2 is False,
            f"blocked={blocked2}")

    r.check("Coach responds meaningfully",
            len(coach2) > 10,
            f"coach len={len(coach2)}")

    cleanup(sid)
    return r


# ═════════════════════════════════════════════════════════════════════════════
# Edge Cases
# ═════════════════════════════════════════════════════════════════════════════

def test_edge_user_scoping():
    """Edge: user asks to focus on one item first"""
    r = Result("Edge1: User scoping preference")
    print(f"\n{'='*60}")
    print(f"  {r.name}")
    print(f"{'='*60}")

    s = create_session()
    sid = s["session_id"]
    print(f"  Session: {sid[:8]}...")

    t1 = send_turn(sid, "I'm having pasta and a coffee with milk.")
    print(f"  T1: coach='{t1.get('coach_question','')[:80]}'")

    t2 = send_turn(sid, "Let's focus on the pasta first. It's spaghetti with marinara sauce.")
    coach2 = t2.get("coach_question", "").lower()
    print(f"  T2: coach='{coach2[:80]}'")

    # Coach should ask about pasta, not redirect to coffee
    asks_about_pasta = any(w in coach2 for w in ["pasta", "spaghetti", "marinara", "sauce", "portion", "how much"])
    asks_about_coffee = any(w in coach2 for w in ["coffee", "milk", "drink"])

    r.check("Coach focuses on user-requested item (pasta)",
            asks_about_pasta and not asks_about_coffee,
            f"coach: '{coach2[:100]}'")

    cleanup(sid)
    return r


def test_edge_accept_then_question():
    """Edge: user accepts recommendation AND asks a follow-up"""
    r = Result("Edge2: Accept + follow-up question")
    print(f"\n{'='*60}")
    print(f"  {r.name}")
    print(f"{'='*60}")

    s = create_session()
    sid = s["session_id"]

    # Fast-track to recommendation phase
    send_turn(sid, "I'm having fried chicken with white rice and coleslaw.")
    send_turn(sid, "About 3 pieces of fried chicken, a big bowl of rice, regular coleslaw.")
    send_turn(sid, "Deep fried, standard breading. Coleslaw is creamy mayo-based.")
    t4 = send_turn(sid, "Sure, I'm open to any suggestions.")

    # Now accept + ask
    t5 = send_turn(sid, "Sure I'll try that! But should I do a 1:1 ratio?")
    status5 = t5.get("status", "")
    m5 = t5.get("monitoring", {})
    action5 = m5.get("planner_action", "")
    print(f"  Accept+Q: action={action5}, status={status5}")

    # Should NOT terminate — user asked a question
    r.check("Session stays active (user asked follow-up)",
            status5 == "active",
            f"action={action5}, status={status5}")

    cleanup(sid)
    return r


# ═════════════════════════════════════════════════════════════════════════════
# Runner
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "═" * 60)
    print("  E2E Signal-Based Router Test Suite")
    print("═" * 60)

    # Check server
    try:
        requests.get(f"{BASE}/", timeout=5)
    except Exception:
        print("❌ Server not reachable at", BASE)
        sys.exit(1)

    tests = [
        test_s1_happy_path,
        test_s2_implicit_recommendation,
        test_s3_incomplete_meal_no_premature_close,
        test_s4_vague_responder,
        test_s5_already_aligned,
        test_s6_off_topic,
        test_edge_user_scoping,
        test_edge_accept_then_question,
    ]

    results = []
    for test_fn in tests:
        try:
            result = test_fn()
            results.append(result)
        except Exception as e:
            print(f"\n  ❌ {test_fn.__name__} CRASHED: {e}")
            results.append(None)

    # Summary
    print("\n" + "═" * 60)
    print("  SUMMARY")
    print("═" * 60)

    total_checks = 0
    total_passed = 0
    for r in results:
        if r is None:
            print(f"  💥 CRASH")
            continue
        status = "✅" if r.passed else "❌"
        p = sum(1 for _, ok, _ in r.checks if ok)
        total_checks += len(r.checks)
        total_passed += p
        print(f"  {status} {r.name} — {r.summary}")
        if not r.passed:
            for label, ok, detail in r.checks:
                if not ok:
                    print(f"       ❌ {label}: {detail}")

    print(f"\n  Total: {total_passed}/{total_checks} checks passed")
    print("═" * 60)

    return 0 if total_passed == total_checks else 1


if __name__ == "__main__":
    sys.exit(main())
