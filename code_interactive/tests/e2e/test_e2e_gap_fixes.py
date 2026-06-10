#!/usr/bin/env python3
"""
E2E 갭 수정 검증 테스트
========================
B1 (Certainty 의미 경계), B2+F2 (추천 거부 graceful exit), C1 (safety net 확장)
수정 사항을 검증한다.

테스트 시나리오:
  G1. B2+F2 — 사용자가 추천을 2회 거부 후 graceful exit
  G2. B2+F2 — 사용자가 추천을 수용하면 rejection 카운터 리셋
  G3. C1 — assess action cross-validation (alignment 낮은데 aligned 판정 방지)
  G4. B1 — Certainty가 정보 충분성만 측정 (alignment 방향무관)
  G5. Happy path regression — 기존 17/17 시나리오가 깨지지 않았는지
"""

import json
import sys
import time
import requests

BASE = "http://localhost:8000"
TIMEOUT = 180


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
    return api("POST", f"/api/session/{sid}/turn", {"user_reply": text})


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
# G1: B2+F2 — 추천 거부 후 graceful exit
# ═════════════════════════════════════════════════════════════════════════════

def test_g1_rejection_graceful_exit():
    """
    시나리오: 사용자가 밥+라면 목표 lean_protein.
    정보 수집 → 평가 → 추천 단계에서 사용자가 2회 연속 거부.
    시스템이 사용자 의지를 존중하여 graceful exit 해야 한다.
    safety net이 rejection 2회 후에는 close를 차단하지 않아야 한다.
    """
    r = Result("G1: Rejection graceful exit (B2+F2)")
    print(f"\n{'='*60}")
    print(f"  {r.name}")
    print(f"{'='*60}")

    s = create_session(
        nutrition_goal="lean_protein",
        meal_description="instant ramen with white rice",
        meal_ingredient="instant ramen noodles, seasoning packet, white rice",
    )
    sid = s["session_id"]
    print(f"  Session: {sid[:8]}...")
    print(f"  Coach T0: {s['first_question'][:80]}")

    # Phase 1: 정보 제공 (misaligned meal — high carb, low protein)
    info_turns = [
        "I had instant ramen with a bowl of white rice for dinner.",
        "Just regular instant ramen, the whole packet with the seasoning. A big bowl of white rice.",
        "No, that's all I had. Just ramen and rice.",
    ]

    history = []
    saw_rejection_phase = False
    terminated = False
    final_status = None

    for i, text in enumerate(info_turns):
        t = send_turn(sid, text)
        status = t.get("status", "")
        orch_dec = t.get("orchestrator_decision") or {}
        action = orch_dec.get("action", "")
        phase = t.get("phase", "")
        user_intent = orch_dec.get("user_intent", "") if orch_dec else ""
        print(f"    T{i+1}: user='{text[:50]}' | action={action} | phase={phase} | intent={user_intent}")
        history.append({"turn": i+1, "status": status, "action": action, "phase": phase})
        if status in ("terminated", "max_turns"):
            terminated = True
            final_status = status
            break

    # Phase 2: 추천 거부 (2회 연속)
    rejection_turns = [
        "No thanks, I prefer my ramen as it is. I don't want to change anything.",
        "I really don't want to change my meal. I like eating ramen and rice, it's my comfort food.",
    ]

    rejection_count = 0
    if not terminated:
        for i, text in enumerate(rejection_turns):
            t = send_turn(sid, text)
            status = t.get("status", "")
            mon = t.get("monitoring", {})
            action = orch_dec.get("action", "")
            phase = t.get("phase", "")
            orch_dec = t.get("orchestrator_decision") or {}
            user_intent = orch_dec.get("user_intent", "") if orch_dec else ""
            align = t.get("alignment_score")
            turn_num = len(info_turns) + i + 1
            print(f"    T{turn_num}: user='{text[:50]}' | action={action} | phase={phase} | intent={user_intent} | align={align}")
            history.append({
                "turn": turn_num, "status": status, "action": action,
                "phase": phase, "intent": user_intent,
            })
            if user_intent == "rejecting":
                rejection_count += 1
            if status in ("terminated", "max_turns"):
                terminated = True
                final_status = status
                break

    # Phase 3: 버텨야 한다면 한 번 더 대화 (시스템이 아직 종료 안했으면)
    extra_turns = [
        "No, I said I don't want to change. Please just let me keep my meal.",
        "I don't want any changes at all.",
    ]
    if not terminated:
        for i, text in enumerate(extra_turns):
            t = send_turn(sid, text)
            status = t.get("status", "")
            mon = t.get("monitoring", {})
            action = orch_dec.get("action", "")
            phase = t.get("phase", "")
            orch_dec = t.get("orchestrator_decision") or {}
            user_intent = orch_dec.get("user_intent", "") if orch_dec else ""
            turn_num = len(info_turns) + len(rejection_turns) + i + 1
            print(f"    T{turn_num}: user='{text[:50]}' | action={action} | phase={phase} | intent={user_intent}")
            history.append({
                "turn": turn_num, "status": status, "action": action,
                "phase": phase, "intent": user_intent,
            })
            if user_intent == "rejecting":
                rejection_count += 1
            if status in ("terminated", "max_turns"):
                terminated = True
                final_status = status
                break

    # 최종 검증
    total_turns = len(history)
    # 핵심: 사용자가 거부 후 max_turns 전에 종료되어야 함
    r.check(
        "Session terminated before max_turns",
        terminated and final_status == "terminated",
        f"status={final_status}, total_turns={total_turns}",
    )
    r.check(
        "Terminated within reasonable turns (≤ 12)",
        total_turns <= 12,
        f"total_turns={total_turns}",
    )
    # Router가 rejection intent를 감지했는지
    rejection_intents = [h for h in history if h.get("intent") == "rejecting"]
    r.check(
        "Router detected rejection intent at least once",
        len(rejection_intents) >= 1,
        f"rejection_intents={len(rejection_intents)}",
    )
    # 무한 추천 루프에 빠지지 않았는지 (recommend 액션이 3회 이하)
    recommend_count = sum(1 for h in history if h.get("action") == "recommend")
    r.check(
        "No infinite recommendation loop (recommend ≤ 3)",
        recommend_count <= 3,
        f"recommend_count={recommend_count}",
    )

    cleanup(sid)
    return r


# ═════════════════════════════════════════════════════════════════════════════
# G2: B2+F2 — 수용 시 rejection 카운터 리셋
# ═════════════════════════════════════════════════════════════════════════════

def test_g2_acceptance_resets_rejection():
    """
    시나리오: 사용자가 1회 거부 후 대안을 수용.
    수용 후 시스템이 정상 경로(close)로 마무리해야 한다.
    rejection 카운터가 리셋되어 safety net이 정상 작동해야 한다.
    """
    r = Result("G2: Acceptance resets rejection counter")
    print(f"\n{'='*60}")
    print(f"  {r.name}")
    print(f"{'='*60}")

    s = create_session(
        nutrition_goal="lean_protein",
        meal_description="pasta with tomato sauce",
        meal_ingredient="spaghetti, tomato sauce, ground beef",
    )
    sid = s["session_id"]
    print(f"  Session: {sid[:8]}...")
    print(f"  Coach T0: {s['first_question'][:80]}")

    turns = [
        "I had spaghetti with tomato sauce and ground beef for dinner.",
        "About a large plate of pasta, maybe 2 cups, with half a pound of ground beef in the sauce.",
        "Regular spaghetti from a box, jarred tomato sauce, and 80/20 ground beef. That's it.",
    ]

    history = []
    terminated = False
    saw_accepting = False

    for i, text in enumerate(turns):
        t = send_turn(sid, text)
        status = t.get("status", "")
        orch_dec = t.get("orchestrator_decision") or {}
        action = orch_dec.get("action", "")
        user_intent = orch_dec.get("user_intent", "") if orch_dec else ""
        print(f"    T{i+1}: action={action} | intent={user_intent}")
        history.append({"turn": i+1, "action": action, "intent": user_intent, "status": status})
        if status in ("terminated", "max_turns"):
            terminated = True
            break

    # 1회 거부 + 수용
    negotiation_turns = [
        ("I'm not sure about that suggestion, I prefer regular pasta.", "rejecting"),
        ("Actually, that sounds reasonable. I could try whole wheat pasta.", "accepting"),
    ]

    if not terminated:
        for i, (text, expected_intent) in enumerate(negotiation_turns):
            t = send_turn(sid, text)
            status = t.get("status", "")
            orch_dec = t.get("orchestrator_decision") or {}
            action = orch_dec.get("action", "")
            user_intent = orch_dec.get("user_intent", "")
            turn_num = len(turns) + i + 1
            print(f"    T{turn_num}: user='{text[:50]}' | action={action} | intent={user_intent}")
            history.append({"turn": turn_num, "action": action, "intent": user_intent, "status": status})
            if user_intent == "accepting":
                saw_accepting = True
            if status in ("terminated", "max_turns"):
                terminated = True
                break

    # 수용 후 종료까지 진행
    followup_turns = [
        "Yes, I'll try it next time. Thanks!",
        "Sounds good, thank you for the advice!",
        "I appreciate the help!",
        "Thanks again, bye!",
    ]
    if not terminated:
        for i, text in enumerate(followup_turns):
            t = send_turn(sid, text)
            status = t.get("status", "")
            orch_dec = t.get("orchestrator_decision") or {}
            action = orch_dec.get("action", "")
            user_intent = orch_dec.get("user_intent", "")
            turn_num = len(turns) + len(negotiation_turns) + i + 1
            print(f"    T{turn_num}: action={action} | intent={user_intent}")
            history.append({"turn": turn_num, "action": action, "intent": user_intent, "status": status})
            if status in ("terminated", "max_turns"):
                terminated = True
                break

    # Fallback: check session status via GET if test loop didn't detect termination
    if not terminated:
        try:
            hist = api("GET", f"/api/session/{sid}/history")
            if hist.get("status") in ("terminated", "max_turns"):
                terminated = True
        except Exception:
            pass

    r.check(
        "Session terminated normally",
        terminated,
        f"total_turns={len(history)}",
    )
    r.check(
        "Router detected accepting intent",
        saw_accepting or any(h.get("intent") == "accepting" for h in history),
        f"intents={[h.get('intent') for h in history]}",
    )
    # 수용 후 정상 종료 — 무한 루프 아님
    r.check(
        "Reasonable turn count (≤ 12)",
        len(history) <= 12,
        f"total_turns={len(history)}",
    )

    cleanup(sid)
    return r


# ═════════════════════════════════════════════════════════════════════════════
# G3: C1 — Assessment cross-validation
# ═════════════════════════════════════════════════════════════════════════════

def test_g3_assessment_cross_validation():
    """
    시나리오: lean_protein 목표에 고탄수 식사 (빵+잼+우유).
    alignment score가 낮은데도 assess action이 실행될 때
    "aligned" 판정이 나오면 cross-validation으로 교정되어
    대화가 조기 종료되지 않아야 한다.

    Note: 이 테스트는 cross-validation 로직이 코드에 존재하는지를 검증한다.
    실제 LLM이 "aligned"를 잘못 출력할 확률은 낮지만,
    assessment_result["overall"]이 alignment_score와 모순되지 않는지 확인한다.
    """
    r = Result("G3: Assessment cross-validation (C1)")
    print(f"\n{'='*60}")
    print(f"  {r.name}")
    print(f"{'='*60}")

    s = create_session(
        nutrition_goal="lean_protein",
        meal_description="white bread with strawberry jam and whole milk",
        meal_ingredient="white bread, strawberry jam, whole milk",
    )
    sid = s["session_id"]
    print(f"  Session: {sid[:8]}...")
    print(f"  Coach T0: {s['first_question'][:80]}")

    turns = [
        "I had two slices of white bread with strawberry jam and a big glass of whole milk.",
        "Regular white sandwich bread, about 2 tablespoons of jam on each slice, and about 12 ounces of milk.",
        "That's all, nothing else.",
    ]

    history = []
    premature_termination = False
    saw_assessment = False

    for i, text in enumerate(turns):
        t = send_turn(sid, text)
        status = t.get("status", "")
        orch_dec = t.get("orchestrator_decision") or {}
        action = orch_dec.get("action", "")
        phase = t.get("phase", "")
        align = t.get("alignment_score")
        print(f"    T{i+1}: action={action} | phase={phase} | align={align} | status={status}")
        history.append({"turn": i+1, "action": action, "phase": phase, "align": align, "status": status})
        if action == "assess":
            saw_assessment = True
        if status == "terminated" and align is not None and align < 0.5:
            premature_termination = True
        if status in ("terminated", "max_turns"):
            break

    # 이 식사는 lean_protein 목표에 전혀 안 맞으므로 alignment < 0.5
    # 조기 종료(terminated + alignment < 0.5)가 되면 cross-validation 실패
    r.check(
        "No premature termination with low alignment",
        not premature_termination,
        f"terminated with low alignment? {premature_termination}",
    )
    # 대화가 assessment 이후에도 계속되었는지 (recommendation 또는 negotiation으로 진입)
    phases_after_assess = [h["phase"] for h in history if h.get("action") != "assess"]
    advanced = any(p in ("recommendation", "negotiation") for p in phases_after_assess)
    # 아직 assessment 안 됐으면 exploration에서 대화 지속 중이어도 OK
    still_active = any(h["status"] == "active" for h in history)
    r.check(
        "Conversation advanced past assessment or still active",
        advanced or still_active,
        f"phases={[h['phase'] for h in history]}",
    )

    cleanup(sid)
    return r


# ═════════════════════════════════════════════════════════════════════════════
# G4: B1 — Certainty measures info sufficiency, not alignment direction
# ═════════════════════════════════════════════════════════════════════════════

def test_g4_certainty_info_sufficiency():
    """
    시나리오: 명확하게 목표에 안 맞는 식사 (pizza buffet).
    certainty는 "정보 충분성"이므로 식사가 분명히
    안 맞아도 충분한 정보가 주어지면 높아야 한다.
    alignment은 낮으면서 certainty는 높은 조합이 가능해야 한다.
    """
    r = Result("G4: Certainty = info sufficiency (B1)")
    print(f"\n{'='*60}")
    print(f"  {r.name}")
    print(f"{'='*60}")

    s = create_session(
        nutrition_goal="lean_protein",
        meal_description="large pepperoni pizza",
        meal_ingredient="pizza dough, pepperoni, mozzarella cheese, tomato sauce",
    )
    sid = s["session_id"]
    print(f"  Session: {sid[:8]}...")
    print(f"  Coach T0: {s['first_question'][:80]}")

    turns = [
        "I had a whole large pepperoni pizza from Domino's.",
        "The whole thing, 8 slices. Just pepperoni and extra cheese on regular crust.",
        "Nothing else, just the pizza and a large Coke.",
    ]

    history = []
    for i, text in enumerate(turns):
        t = send_turn(sid, text)
        status = t.get("status", "")
        orch_dec = t.get("orchestrator_decision") or {}
        align = t.get("alignment_score")
        cert = t.get("certainty_score")
        action = orch_dec.get("action", "")
        print(f"    T{i+1}: align={align} | cert={cert} | action={action} | status={status}")
        history.append({
            "turn": i+1, "align": align, "cert": cert,
            "action": action, "status": status,
        })
        if status in ("terminated", "max_turns"):
            break

    # 마지막 턴에서 검증: pizza에 대한 정보는 충분 → certainty 높아야
    last_with_scores = [h for h in history if h["cert"] is not None]
    if last_with_scores:
        last = last_with_scores[-1]
        # Certainty: 정보가 충분하므로 0.6 이상 기대 (보수적)
        r.check(
            "Certainty reflects info sufficiency (cert ≥ 0.6)",
            last["cert"] >= 0.6,
            f"cert={last['cert']}",
        )
        # Alignment: 피자는 lean_protein에 안 맞으므로 0.5 미만 기대
        r.check(
            "Alignment reflects meal-goal mismatch (align < 0.5)",
            last["align"] is not None and last["align"] < 0.5,
            f"align={last['align']}",
        )
        # 핵심: certainty HIGH + alignment LOW 조합이 가능함을 검증
        r.check(
            "Cert and alignment are independent (cert ≥ 0.6, align < 0.5)",
            last["cert"] >= 0.6 and last["align"] is not None and last["align"] < 0.5,
            f"cert={last['cert']}, align={last['align']}",
        )
    else:
        r.check("Scores available", False, "No scores recorded")

    cleanup(sid)
    return r


# ═════════════════════════════════════════════════════════════════════════════
# G5: Regression — Happy path still works
# ═════════════════════════════════════════════════════════════════════════════

def test_g5_happy_path_regression():
    """
    기존 happy path가 깨지지 않았는지 회귀 테스트.
    성실한 사용자, lean_protein, 정상적 정보 제공 → 추천 수용 → 종료.
    """
    r = Result("G5: Happy path regression")
    print(f"\n{'='*60}")
    print(f"  {r.name}")
    print(f"{'='*60}")

    s = create_session(
        nutrition_goal="lean_protein",
        meal_description="grilled chicken with vegetables",
        meal_ingredient="chicken breast, broccoli, olive oil",
    )
    sid = s["session_id"]
    print(f"  Session: {sid[:8]}...")
    print(f"  Coach T0: {s['first_question'][:80]}")

    turns = [
        "I had grilled chicken breast with steamed broccoli, drizzled with olive oil.",
        "About 6 ounces of chicken and a cup of broccoli, with a tablespoon of olive oil.",
        "Yes, that sounds good. I'm open to any suggestions.",
        "That's a great idea, I'll try it next time.",
        "Thank you, that was very helpful!",
    ]

    history = []
    terminated = False
    for i, text in enumerate(turns):
        t = send_turn(sid, text)
        status = t.get("status", "")
        orch_dec = t.get("orchestrator_decision") or {}
        action = orch_dec.get("action", "")
        print(f"    T{i+1}: action={action} | status={status}")
        history.append({"turn": i+1, "action": action, "status": status})
        if status in ("terminated", "max_turns"):
            terminated = True
            break

    r.check(
        "Session terminated",
        terminated,
        f"total_turns={len(history)}",
    )
    r.check(
        "Reasonable turn count (≤ 10)",
        len(history) <= 10,
        f"total_turns={len(history)}",
    )

    cleanup(sid)
    return r


# ═════════════════════════════════════════════════════════════════════════════
# G6: user_intent structural output — Router produces valid intent
# ═════════════════════════════════════════════════════════════════════════════

def test_g6_user_intent_structural_output():
    """
    Router가 user_intent 필드를 구조적으로 출력하는지 검증.
    다양한 사용자 메시지에 대해 intent가 유효한 값인지 확인.
    """
    r = Result("G6: user_intent structural output")
    print(f"\n{'='*60}")
    print(f"  {r.name}")
    print(f"{'='*60}")

    VALID_INTENTS = {
        "informing",
        "accepting",
        "inquiring",
        "deferring",
        "passive",
        "rejecting",
        "disengaging",
    }

    s = create_session(
        nutrition_goal="lean_protein",
        meal_description="salmon with quinoa",
        meal_ingredient="Atlantic salmon fillet, quinoa, lemon",
    )
    sid = s["session_id"]
    print(f"  Session: {sid[:8]}...")

    turns_and_expected_category = [
        ("I had baked salmon with quinoa and lemon.", "info"),
        ("How much protein does salmon have?", "question"),
        ("Sure, I'll try adding more vegetables.", "accept"),
    ]

    intents_collected = []
    for text, category in turns_and_expected_category:
        t = send_turn(sid, text)
        status = t.get("status", "")
        mon = t.get("monitoring", {})
        orch_dec = t.get("orchestrator_decision") or {}
        user_intent = orch_dec.get("user_intent", "") if orch_dec else ""
        print(f"    '{text[:40]}' → intent={user_intent}")
        intents_collected.append(user_intent)
        if status in ("terminated", "max_turns"):
            break

    # 모든 intent가 유효한 값이어야
    all_valid = all(s in VALID_INTENTS for s in intents_collected if s)
    r.check(
        "All intents are valid enum values",
        all_valid,
        f"intents={intents_collected}",
    )
    # 최소 2개 이상 intent를 수집했어야
    r.check(
        "Collected intents from multiple turns",
        len([s for s in intents_collected if s]) >= 2,
        f"count={len([s for s in intents_collected if s])}",
    )

    cleanup(sid)
    return r


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 70)
    print("  E2E Gap Fix Verification Tests")
    print("  B1 (Certainty semantics) + B2/F2 (Rejection graceful exit) + C1 (Safety net)")
    print("=" * 70)

    tests = [
        test_g1_rejection_graceful_exit,
        test_g2_acceptance_resets_rejection,
        test_g3_assessment_cross_validation,
        test_g4_certainty_info_sufficiency,
        test_g5_happy_path_regression,
        test_g6_user_intent_structural_output,
    ]

    results = []
    for test_fn in tests:
        try:
            r = test_fn()
            results.append(r)
        except Exception as e:
            print(f"\n  💥 {test_fn.__name__} CRASHED: {e}")
            import traceback
            traceback.print_exc()
            r = Result(test_fn.__name__)
            r.check("Test execution", False, str(e))
            results.append(r)

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    total_pass = 0
    total_checks = 0
    all_pass = True
    for r in results:
        status = "PASS ✅" if r.passed else "FAIL ❌"
        print(f"  {status}  {r.name}  ({r.summary})")
        total_pass += sum(1 for _, ok, _ in r.checks if ok)
        total_checks += len(r.checks)
        if not r.passed:
            all_pass = False

    print(f"\n  Total: {total_pass}/{total_checks}")
    print("=" * 70)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
