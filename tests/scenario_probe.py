"""Run fixed dialogue scenarios against a running FastAPI server.

This is a manual probe, not a pytest test. It is meant for version-to-version
workflow review:

    python tests/scenario_probe.py --base-url http://127.0.0.1:8094 \
      --output docs/scenario_probe_2026-06-22.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


SHORT_SCENARIOS: list[dict[str, Any]] = [
    {
        "id": "lean_protein_baseline",
        "goal": "lean_protein",
        "meal_type": "dinner",
        "turns": [
            "I am thinking of grilled chicken with rice and a small salad.",
            "Chicken breast, skinless, about one palm-sized piece.",
            "The rice would be about half a cup.",
            "That sounds good; I can keep the dressing light.",
            "Yes, that works for me.",
        ],
    },
    {
        "id": "lean_protein_buffet_feasibility",
        "goal": "lean_protein",
        "meal_type": "dinner",
        "turns": [
            "I'm at a buffet now. I only see brie; no other protein options are available.",
            "I cannot leave the buffet or get another protein.",
            "I can take a small piece of brie and add salad.",
            "Please don't keep asking me to find meat or yogurt.",
            "Just help me make the best choice with what is here.",
        ],
    },
    {
        "id": "one_fourth_carbs_intent_boundary",
        "goal": "one_fourth_carbs",
        "meal_type": "dinner",
        "turns": [
            "I want to keep dinner light, so I don't want starch tonight.",
            "I would rather use non-starchy vegetables.",
            "I know the carb goal, but I don't want bread, rice, pasta, or potatoes.",
            "Can we make this work without pushing starch?",
            "A vegetable-heavy plate sounds better to me.",
        ],
    },
    {
        "id": "half_fv_allergy_preference",
        "goal": "half_fruits_vegetables",
        "meal_type": "lunch",
        "turns": [
            "I'm having a turkey sandwich and chips for lunch.",
            "I'm allergic to tomatoes, and I don't like raw onions.",
            "I can add lettuce, cucumber, or an apple.",
            "I prefer fruit today if that helps.",
            "An apple would be easiest.",
        ],
    },
    {
        "id": "drink_water_user_question",
        "goal": "drink_water",
        "meal_type": "dinner",
        "turns": [
            "I'm having spicy noodles for dinner.",
            "I usually drink soda with noodles. Does sparkling water count?",
            "I can do sparkling water if it's okay.",
            "I don't want plain water only.",
            "Lime sparkling water sounds doable.",
        ],
    },
    {
        "id": "lean_protein_rejection_end",
        "goal": "lean_protein",
        "meal_type": "dinner",
        "turns": [
            "I'm eating cheese pizza.",
            "No, I don't want to add chicken or any meat.",
            "I don't want beans or yogurt either.",
            "Please stop suggesting swaps.",
            "I just want to finish with the pizza.",
        ],
    },
]

LONG_SCENARIOS: list[dict[str, Any]] = [
    {
        "id": "long_lean_protein_complete_plate",
        "goal": "lean_protein",
        "meal_type": "dinner",
        "turns": [
            "I'm thinking about grilled chicken, rice, and salad for dinner.",
            "It would be skinless chicken breast, about one palm-sized piece.",
            "The rice is probably half a cup.",
            "The salad has lettuce, cucumber, and a little vinaigrette.",
            "I can grill the chicken without butter.",
            "I might use a little teriyaki sauce, but not much.",
            "That sounds doable.",
            "Yes, let's go with that plan.",
            "I'm ready to finish.",
        ],
    },
    {
        "id": "long_buffet_hard_feasibility_boundary",
        "goal": "lean_protein",
        "meal_type": "dinner",
        "turns": [
            "I'm at a buffet now. I only see brie; no other protein options are available.",
            "I cannot leave the buffet or buy anything else.",
            "There is salad, fruit, bread, and desserts.",
            "I don't want meat, yogurt, beans, or eggs suggested because they are not here.",
            "I can take a small piece of brie and a big salad.",
            "Please don't keep asking me to find another protein.",
            "Just help me make the best choice with what is actually here.",
            "A small brie piece with salad sounds like the best I can do.",
            "Yes, let's end with that.",
        ],
    },
    {
        "id": "long_no_starch_user_initiative",
        "goal": "one_fourth_carbs",
        "meal_type": "dinner",
        "turns": [
            "I want to keep dinner light, so I don't want starch tonight.",
            "I would rather use non-starchy vegetables.",
            "I know the carb goal allows some carbs, but I don't want bread, rice, pasta, or potatoes.",
            "I have tofu, eggs, broccoli, mushrooms, and salad greens.",
            "Can we make this work without pushing starch?",
            "Tofu with broccoli and mushrooms sounds good.",
            "I can add salad greens too.",
            "Yes, I want the no-starch vegetable-heavy plan.",
            "Let's finish there.",
        ],
    },
    {
        "id": "long_half_fv_allergy_preference_commitment",
        "goal": "half_fruits_vegetables",
        "meal_type": "lunch",
        "turns": [
            "I'm having a turkey sandwich and chips for lunch.",
            "I'm allergic to tomatoes, and I don't like raw onions.",
            "I can add lettuce, cucumber, or an apple.",
            "I prefer fruit today if that helps.",
            "An apple would be easiest.",
            "I can also add a few cucumber slices if that makes the goal work better.",
            "I don't want tomato or onion mentioned again.",
            "Apple plus cucumber sounds doable.",
            "Yes, let's go with apple and cucumber.",
        ],
    },
    {
        "id": "long_drink_water_question_and_commitment",
        "goal": "drink_water",
        "meal_type": "dinner",
        "turns": [
            "I'm having spicy noodles for dinner.",
            "I usually drink soda with noodles. Does sparkling water count?",
            "I can do sparkling water if it's okay.",
            "I don't want plain water only.",
            "Lime sparkling water sounds doable.",
            "One can is 12 ounces.",
            "I can drink that with dinner and skip soda.",
            "Yes, that works.",
        ],
    },
    {
        "id": "long_profile_allergy_safety_conflict",
        "goal": "lean_protein",
        "meal_type": "dinner",
        "start_payload": {
            "persona_allergies": ["cheese"],
        },
        "turns": [
            "I'm eating cheese pizza for dinner.",
            "I know, but I still want regular cheese pizza.",
            "No, I don't mean dairy-free cheese.",
            "I don't want chicken, beans, yogurt, or tofu.",
            "Please don't keep asking the same allergy question.",
            "I understand the concern, but I want to keep the pizza.",
            "Just give me the safest final advice you can.",
        ],
    },
]

STRESS_SCENARIOS: list[dict[str, Any]] = [
    {
        "id": "stress_repeated_minor_refinement",
        "goal": "lean_protein",
        "meal_type": "dinner",
        "turns": [
            "I'm planning grilled chicken breast, brown rice, and salad.",
            "The chicken is skinless and grilled without butter.",
            "I can use a light vinaigrette on the salad.",
            "Yes, I already said light vinaigrette works.",
            "I don't want more dressing or sauce tweaks.",
            "The current plan sounds good enough.",
            "Let's finish with this plan.",
        ],
    },
    {
        "id": "stress_boundary_finalization",
        "goal": "one_fourth_carbs",
        "meal_type": "dinner",
        "turns": [
            "I understand the carb goal, but I do not want starch tonight.",
            "I have tofu, broccoli, mushrooms, and salad greens.",
            "Please do not ask me to add rice, bread, pasta, or potatoes.",
            "Tofu with broccoli and mushrooms is what I want.",
            "I know it is not a classic quarter-carb plate, but I prefer this.",
            "Let's end with the no-starch plan.",
        ],
    },
    {
        "id": "stress_profile_allergy_reflective_close",
        "goal": "lean_protein",
        "meal_type": "dinner",
        "start_payload": {
            "persona_allergies": ["cheese"],
        },
        "turns": [
            "I'm having regular cheese pizza.",
            "I know cheese is listed as an allergy, but I still want it.",
            "No, I don't mean dairy-free cheese.",
            "Please don't keep asking the allergy question.",
            "Just close with the safest advice you can give.",
        ],
    },
    {
        "id": "stress_candidate_narrowing_without_commitment",
        "goal": "half_fruits_vegetables",
        "meal_type": "lunch",
        "turns": [
            "I'm eating a turkey sandwich and chips.",
            "I can add grapes, baby carrots, cucumber, or an apple.",
            "I'm not choosing all of them; those are just what I have.",
            "I want the simplest option that helps the goal.",
            "Apple sounds easiest.",
            "Yes, apple is the choice.",
        ],
    },
    {
        "id": "stress_tentative_plan_assessment_evidence",
        "goal": "lean_protein",
        "meal_type": "dinner",
        "turns": [
            "I'm thinking about ramen with tofu and maybe some edamame.",
            "I haven't fully decided yet, but those are the foods I'm considering.",
            "Please assess that plan before suggesting changes.",
            "I want to keep the ramen if possible.",
            "Tofu is fine; I just need the plan to make sense.",
        ],
    },
    {
        "id": "stress_accepted_swap_memory",
        "goal": "lean_protein",
        "meal_type": "dinner",
        "turns": [
            "I'm planning a burrito bowl with rice, beans, sour cream, and beef.",
            "A lean chicken swap sounds good.",
            "Yes, I'll use grilled chicken instead of beef.",
            "I don't want to reduce the rice portion.",
            "Please keep the chicken swap and don't suggest beef again.",
            "That is my final plan.",
        ],
    },
    {
        "id": "stress_profile_allergy_known_fact",
        "goal": "half_fruits_vegetables",
        "meal_type": "lunch",
        "start_payload": {
            "persona_allergies": ["peanuts"],
        },
        "turns": [
            "I'm having noodles with cucumber and carrots.",
            "I want a crunchy topping, but remember I can't have peanuts.",
            "Please suggest something safe and vegetable-forward.",
            "Cucumber and carrots are already available.",
            "I don't want to answer another peanut allergy question.",
        ],
    },
    {
        "id": "stress_rejected_refinement_preserves_plan",
        "goal": "drink_water",
        "meal_type": "dinner",
        "turns": [
            "I'm having tacos and I can drink sparkling water with lime.",
            "Yes, sparkling water works for me.",
            "No, I don't want to switch to plain still water.",
            "Please keep the sparkling water plan.",
            "One can with dinner is what I'll do.",
            "Let's finish there.",
        ],
    },
    {
        "id": "stress_recommendation_bundle_scope",
        "goal": "lean_protein",
        "meal_type": "dinner",
        "start_payload": {
            "persona_allergies": ["egg"],
            "persona_health_concerns": ["diabetes / prediabetes"],
            "persona_diet_prefs": ["low carb"],
        },
        "turns": [
            "I'm trying to have jajangmyeon and egg-fried rice.",
            "Please give me suggestions.",
            "Chicken breast sounds good, and minimal oil sounds good too.",
            "No, I want to keep the noodle portion as it is.",
            "Please don't add more new tweaks beyond those options.",
            "That plan is good enough.",
        ],
    },
    {
        "id": "stress_confirmation_reopening",
        "goal": "lean_protein",
        "meal_type": "dinner",
        "turns": [
            "I'm planning chicken breast with rice and salad.",
            "Light dressing works.",
            "Yes, that sounds like the plan.",
            "Actually, wait, I also want dumplings.",
            "Please reassess with the dumplings included.",
            "I can keep the dumplings small.",
        ],
    },
    {
        "id": "stress_localized_safety_conflict",
        "goal": "lean_protein",
        "meal_type": "dinner",
        "start_payload": {
            "persona_allergies": ["egg"],
        },
        "turns": [
            "I'm having jajangmyeon and egg-fried rice.",
            "I know about the egg allergy, but I still want the egg-fried rice.",
            "Please do not end the whole conversation; help with the jajangmyeon too.",
            "Chicken breast in the jajangmyeon sounds good.",
            "I understand you cannot recommend the egg-fried rice.",
        ],
    },
    {
        "id": "stress_confirmation_before_finalization",
        "goal": "drink_water",
        "meal_type": "dinner",
        "turns": [
            "I'm having tacos and can drink lime sparkling water.",
            "One 12-ounce can works for me.",
            "I will skip soda.",
            "Yes, that is the drink plan.",
            "Looks right to me.",
        ],
    },
    {
        "id": "stress_slot_scoped_side_replacement",
        "goal": "lean_protein",
        "meal_type": "dinner",
        "start_payload": {
            "persona_allergies": ["egg"],
            "persona_health_concerns": ["diabetes / prediabetes"],
            "persona_diet_prefs": ["low carb"],
        },
        "turns": [
            "I'm considering jajangmyeon and egg-fried rice.",
            "Please provide suggestions.",
            "Pork loin in the jajangmyeon sounds good.",
            "But we still need to settle the egg-fried rice alternative.",
            "I do not want pork-loin fried rice because pork loin is already in the jajangmyeon.",
            "Give me an egg-free side that feels similar to fried rice but stays on that side-dish topic.",
            "A veggie fried-rice style side without egg or pork sounds good.",
            "Now confirm the whole meal.",
        ],
    },
    {
        "id": "stress_active_issue_topic_repair",
        "goal": "one_fourth_carbs",
        "meal_type": "dinner",
        "turns": [
            "I'm having grilled salmon and I need help choosing the carb portion.",
            "Half a cup of rice sounds fine for the carb slot.",
            "Now I need help choosing a vegetable side.",
            "Why are you still talking about rice? The rice is settled.",
            "Please stay on the vegetable side question.",
            "Broccoli sounds good.",
            "Yes, salmon, half-cup rice, and broccoli is the plan.",
        ],
    },
    {
        "id": "stress_rejected_option_scope",
        "goal": "half_fruits_vegetables",
        "meal_type": "lunch",
        "turns": [
            "I'm having a turkey sandwich and need a fruit or vegetable add-on.",
            "Apple slices in the sandwich sound bad to me.",
            "But an apple on the side could work.",
            "Please don't treat apple as globally rejected.",
            "I want the side apple, not apple inside the sandwich.",
            "Yes, side apple is settled.",
        ],
    },
    {
        "id": "stress_bundle_json_and_bullets",
        "goal": "lean_protein",
        "meal_type": "dinner",
        "turns": [
            "I'm making pasta with a little chicken and creamy sauce.",
            "Please give me several compact options at once, not one tweak per turn.",
            "I can consider more chicken, less sauce, or adding cottage cheese on the side.",
            "Show me the options clearly so I can choose.",
            "More chicken and less sauce sound good; cottage cheese does not.",
            "Do not add new ideas beyond those options.",
        ],
    },
    {
        "id": "stress_frustration_terminal_no_question",
        "goal": "drink_water",
        "meal_type": "dinner",
        "turns": [
            "I'm having a burrito and I can drink sparkling water.",
            "Yes, sparkling water is the drink plan.",
            "No, I don't want plain water.",
            "You already asked that. Please stop repeating it.",
            "I'm tired and done now.",
        ],
    },
    {
        "id": "stress_egg_free_side_preserves_main",
        "goal": "lean_protein",
        "meal_type": "dinner",
        "start_payload": {
            "persona_allergies": ["egg"],
            "persona_health_concerns": ["diabetes / prediabetes"],
            "persona_diet_prefs": ["low carb"],
        },
        "turns": [
            "I'm trying to have jajangmyeon and egg-fried rice.",
            "I meant egg-fried rice. What do you mean by egg-free fried rice?",
            "Okay, then let's go for egg-free fried rice.",
            "I will go with shrimp. I also did not give up jajangmyeon.",
            "The first and third options are doable today.",
            "Why are you asking that again? I already accepted those two.",
            "No, I want both jajangmyeon and shrimp fried rice.",
            "I'm exhausted; please wrap this up.",
        ],
    },
    {
        "id": "stress_bundle_ordinal_with_topic_repair",
        "goal": "half_fruits_vegetables",
        "meal_type": "lunch",
        "turns": [
            "I'm having a turkey sandwich and chips.",
            "I can add apple slices, cucumber, or baby carrots.",
            "Show me a few compact options.",
            "The first and third options are doable.",
            "Why are you still asking which add-on I want?",
            "Also, please keep the sandwich; I never gave it up.",
            "Let's finish with the accepted add-ons.",
        ],
    },
    {
        "id": "stress_forced_tradeoff_compromise",
        "goal": "one_fourth_carbs",
        "meal_type": "dinner",
        "turns": [
            "I'm having noodles and rice together tonight.",
            "I understand that is carb-heavy, but I want both.",
            "No, I don't want to choose either noodles or rice.",
            "Give me a compromise that keeps both.",
            "Half portions of each sounds okay.",
            "Please don't ask me to choose one again.",
        ],
    },
    {
        "id": "stress_final_state_preservation_after_fatigue",
        "goal": "lean_protein",
        "meal_type": "dinner",
        "turns": [
            "I'm having tofu stir-fry with brown rice.",
            "Extra tofu sounds good.",
            "Light oil is doable too.",
            "Yes, tofu stir-fry with brown rice and light oil is the plan.",
            "I'm tired of refining this; please stop.",
        ],
    },
    {
        "id": "stress_stale_active_issue_cleanup",
        "goal": "drink_water",
        "meal_type": "dinner",
        "turns": [
            "I'm eating spicy noodles and usually drink soda.",
            "Sparkling water with lime works.",
            "Yes, one can is fine.",
            "You already know the drink. Now I want to confirm the meal.",
            "Please don't keep asking about the water.",
            "Let's end with sparkling water and the noodles.",
        ],
    },
]

CHAT_SEEDED_COMPLETION_SCENARIOS: list[dict[str, Any]] = [
    {
        "id": "chat_seeded_egg_free_combo_completion",
        "source_chats": [
            "chat_history_300c2ff6.json",
            "chat_history_333ede4c.json",
            "chat_history_d27dc2de.json",
            "chat_history_ed47201e.json",
        ],
        "goal": "lean_protein",
        "meal_type": "dinner",
        "start_payload": {
            "persona_allergies": ["egg"],
            "health_concerns": ["diabetes"],
        },
        "turns": [
            "I'm trying to have jajangmyeon and egg-fried rice.",
            "I meant egg-fried rice. What do you mean by egg-free fried rice?",
            "Okay, then let's go for egg-free fried rice.",
            "I will go with shrimp. I love shrimp fried rice, and I did not give up jajangmyeon.",
            "I can keep the shrimp plain and make the fried rice light-oil.",
            "I don't want to reduce the noodles. I want the jajangmyeon and shrimp fried rice combo.",
            "Please stop asking more optimization questions and wrap up the safest plan.",
        ],
    },
    {
        "id": "chat_seeded_buffet_feasibility_completion",
        "source_chats": ["chat_history_3709a298.json"],
        "goal": "lean_protein",
        "meal_type": "dinner",
        "turns": [
            "I'm at a work buffet with pita, hummus, salad, and veggie stuffed grape leaves.",
            "The only protein-like option I see is brie.",
            "I cannot leave the event or buy chicken, tuna, yogurt, tofu, eggs, or beans.",
            "Please don't keep asking me to find another protein option.",
            "I can do a small piece of brie, more salad, and the pita with hummus.",
            "That is the best I can do here. Please finish with that.",
        ],
    },
    {
        "id": "chat_seeded_available_options_respected_completion",
        "source_chats": ["chat_history_243a6433.json"],
        "goal": "lean_protein",
        "meal_type": "dinner",
        "turns": [
            "I have arctic char, steelhead trout, or salmon in my fridge.",
            "I don't care which fish. Please suggest one of those three.",
            "I will bake it, and I usually make a butter sauce with shallots, garlic, and citrus.",
            "I don't have cod, so please don't suggest fish I did not list.",
            "I can use less butter if that helps.",
            "Steelhead with a lighter sauce sounds fine. Let's finish there.",
        ],
    },
    {
        "id": "chat_seeded_no_starch_boundary_completion",
        "source_chats": ["chat_history_765ec1a3.json"],
        "goal": "one_fourth_carbs",
        "meal_type": "dinner",
        "turns": [
            "I haven't decided yet, but probably something with chicken.",
            "I'm cooking at home and have at least an hour.",
            "I want to go with non-starchy vegetables.",
            "I know the carb goal, but I don't want rice, bread, pasta, potatoes, corn, or other starch tonight.",
            "Chicken with broccoli, mushrooms, and salad sounds good.",
            "Please respect the no-starch plan and finish with the tradeoff noted.",
        ],
    },
    {
        "id": "chat_seeded_repeated_question_repair_completion",
        "source_chats": [
            "chat_history_70f17407.json",
            "chat_history_9fc8ca50.json",
        ],
        "goal": "half_fruits_vegetables",
        "meal_type": "dinner",
        "start_payload": {
            "persona_allergies": ["dairy"],
        },
        "turns": [
            "I'm having half a ham and pineapple pizza.",
            "I can add 2 cups of diced watermelon.",
            "Yes, I already said watermelon works.",
            "Please don't ask again whether I have fruit or vegetables.",
            "I understand pizza is not half produce, but watermelon is my add-on.",
            "Let's finish with half pizza and watermelon.",
        ],
    },
]

SCENARIO_SETS = {
    "short": SHORT_SCENARIOS,
    "long": LONG_SCENARIOS,
    "stress": STRESS_SCENARIOS,
    "chat_seeded_completion": CHAT_SEEDED_COMPLETION_SCENARIOS,
    "chat_seeded_scripted_completion": CHAT_SEEDED_COMPLETION_SCENARIOS,
    "both": SHORT_SCENARIOS + LONG_SCENARIOS,
}


def _post(base_url: str, path: str, payload: dict[str, Any]) -> tuple[dict[str, Any], float]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body), time.perf_counter() - started


def _error_record(exc: BaseException) -> dict[str, Any]:
    body = ""
    if isinstance(exc, urllib.error.HTTPError):
        try:
            body = exc.read().decode("utf-8")
        except Exception:
            body = ""
        return {
            "type": type(exc).__name__,
            "status_code": exc.code,
            "reason": exc.reason,
            "body": body,
        }
    return {
        "type": type(exc).__name__,
        "message": str(exc),
    }


def _compact_turn(turn_response: dict[str, Any], user_reply: str, elapsed: float) -> dict[str, Any]:
    metadata = turn_response.get("engine_metadata") or {}
    latency = turn_response.get("latency") or {}
    decision = turn_response.get("dialogue_plan") or {}
    planning_policy = metadata.get("planning_policy") or metadata.get("intent_policy") or {}
    post_assessment = metadata.get("post_assessment_decision") or {}
    commitment_gate = metadata.get("commitment_gate") or {}
    post_assessment_gate = metadata.get("post_assessment_gate") or {}
    return {
        "turn_idx": turn_response.get("turn_idx"),
        "user_reply": user_reply,
        "coach_question": turn_response.get("coach_question"),
        "coach_messages": turn_response.get("coach_messages"),
        "status": turn_response.get("status"),
        "phase": turn_response.get("phase"),
        "action": decision.get("action"),
        "planned_action": planning_policy.get("planned_action"),
        "effective_action": planning_policy.get("effective_action"),
        "planning_override": planning_policy.get("override"),
        "commitment_gate": commitment_gate.get("gate"),
        "post_assessment_gate": post_assessment_gate.get("gate"),
        "finalization_style": metadata.get("finalization_style"),
        "commitment_status": commitment_gate.get("commitment_status"),
        "safety_constraint": commitment_gate.get("constraint"),
        "post_assessment_action": post_assessment.get("action"),
        "user_intent": decision.get("user_intent"),
        "actionability": (
            decision.get("actionability") or planning_policy.get("actionability")
        ),
        "intent_summary": decision.get("intent_summary"),
        "planner_reasoning": decision.get("reasoning"),
        "planner_confidence": decision.get("confidence"),
        "assessment_followup_action": decision.get("assessment_followup_action"),
        "alignment_score": turn_response.get("alignment_score"),
        "certainty_score": turn_response.get("certainty_score"),
        "interaction_state": metadata.get("interaction_state"),
        "client_elapsed_seconds": round(elapsed, 3),
        "engine_total_seconds": latency.get("total_seconds"),
        "module_call_count": latency.get("module_call_count"),
        "module_totals": latency.get("module_totals"),
    }


def _summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    elapsed: list[float] = []
    module_totals: dict[str, float] = {}
    gate_counts: dict[str, int] = {}
    post_assessment_gate_counts: dict[str, int] = {}
    finalization_style_counts: dict[str, int] = {}
    final_status_counts: dict[str, int] = {}
    scenario_summaries = []
    for scenario in results:
        turns = scenario.get("turns", [])
        final_status = turns[-1].get("status") if turns else "no_turns"
        if scenario.get("error"):
            final_status = "error"
        final_status_counts[final_status] = final_status_counts.get(final_status, 0) + 1
        action_flow = []
        effective_action_flow = []
        gate_flow = []
        for turn in turns:
            if isinstance(turn.get("client_elapsed_seconds"), (int, float)):
                elapsed.append(float(turn["client_elapsed_seconds"]))
            action_flow.append(turn.get("planned_action") or turn.get("action") or "")
            effective_action_flow.append(turn.get("effective_action") or "")
            gate = turn.get("commitment_gate")
            if gate:
                gate_flow.append(gate)
                gate_counts[gate] = gate_counts.get(gate, 0) + 1
            post_gate = turn.get("post_assessment_gate")
            if post_gate:
                post_assessment_gate_counts[post_gate] = (
                    post_assessment_gate_counts.get(post_gate, 0) + 1
                )
            style = turn.get("finalization_style")
            if style:
                finalization_style_counts[style] = (
                    finalization_style_counts.get(style, 0) + 1
                )
            for module, seconds in (turn.get("module_totals") or {}).items():
                module_totals[module] = module_totals.get(module, 0.0) + float(seconds)
        scenario_summaries.append(
            {
                "id": scenario.get("id"),
                "goal": scenario.get("goal"),
                "turns_completed": len(turns),
                "final_status": final_status,
                "planned_action_flow": action_flow,
                "effective_action_flow": effective_action_flow,
                "commitment_gate_flow": gate_flow,
                "final_coach_question": turns[-1].get("coach_question") if turns else "",
            }
        )

    latency_summary: dict[str, float | int | None] = {
        "turn_count": len(elapsed),
        "mean": round(statistics.mean(elapsed), 3) if elapsed else None,
        "median": round(statistics.median(elapsed), 3) if elapsed else None,
        "min": round(min(elapsed), 3) if elapsed else None,
        "max": round(max(elapsed), 3) if elapsed else None,
    }
    if len(elapsed) >= 10:
        latency_summary["p90"] = round(statistics.quantiles(elapsed, n=10)[8], 3)
    else:
        latency_summary["p90"] = None

    return {
        "latency": latency_summary,
        "final_status_counts": final_status_counts,
        "commitment_gate_counts": gate_counts,
        "post_assessment_gate_counts": post_assessment_gate_counts,
        "finalization_style_counts": finalization_style_counts,
        "module_totals": {
            module: round(seconds, 3)
            for module, seconds in sorted(module_totals.items())
        },
        "scenario_summaries": scenario_summaries,
    }


def _build_probe_result(
    *,
    base_url: str,
    started_at: str,
    scenario_set: str,
    max_turns: int,
    enable_alignment: bool,
    enable_uncertainty: bool,
    scenario_count: int,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "base_url": base_url,
        "started_at": started_at,
        "scenario_set": scenario_set,
        "max_turns_per_scenario": max_turns,
        "enable_alignment": enable_alignment,
        "enable_uncertainty": enable_uncertainty,
        "scenario_count": scenario_count,
        "completed_scenario_count": len(results),
        "results": results,
        "summary": _summarize_results(results),
    }


def run_scenarios(
    base_url: str,
    max_turns: int,
    *,
    scenario_set: str = "short",
    enable_alignment: bool = False,
    enable_uncertainty: bool = False,
    checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    scenarios = SCENARIO_SETS[scenario_set]
    results = []
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    for scenario in scenarios:
        start_payload = {
            "mode": "custom",
            "alignment_enabled": enable_alignment,
            "nutrition_goal": scenario["goal"],
            "meal_type": scenario["meal_type"],
            "meal_description": "",
            "meal_ingredient": "",
            "context_tracking": True,
            "uncertainty_tracking": enable_uncertainty,
        }
        start_payload.update(scenario.get("start_payload", {}))
        try:
            start_response, start_elapsed = _post(
                base_url,
                "/api/session/start",
                start_payload,
            )
        except Exception as exc:
            results.append(
                {
                    "id": scenario["id"],
                    "goal": scenario["goal"],
                    "meal_type": scenario["meal_type"],
                    "session_id": None,
                    "start_elapsed_seconds": None,
                    "first_question": None,
                    "turns": [],
                    "error": {
                        "stage": "start",
                        "details": _error_record(exc),
                    },
                }
            )
            if checkpoint_path is not None:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                checkpoint_path.write_text(
                    json.dumps(
                        _build_probe_result(
                            base_url=base_url,
                            started_at=started_at,
                            scenario_set=scenario_set,
                            max_turns=max_turns,
                            enable_alignment=enable_alignment,
                            enable_uncertainty=enable_uncertainty,
                            scenario_count=len(scenarios),
                            results=results,
                        ),
                        indent=2,
                        ensure_ascii=False,
                    )
                    + "\n",
                    encoding="utf-8",
                )
            continue
        session_id = start_response["session_id"]
        turns = []
        for user_reply in scenario["turns"][:max_turns]:
            try:
                turn_response, elapsed = _post(
                    base_url,
                    f"/api/session/{session_id}/turn",
                    {"user_reply": user_reply},
                )
            except Exception as exc:
                turns.append(
                    {
                        "turn_idx": len(turns),
                        "user_reply": user_reply,
                        "status": "error",
                        "error": _error_record(exc),
                    }
                )
                break
            turns.append(_compact_turn(turn_response, user_reply, elapsed))
            if turn_response.get("status") != "active":
                break
        results.append(
            {
                "id": scenario["id"],
                "source_chats": scenario.get("source_chats", []),
                "goal": scenario["goal"],
                "meal_type": scenario["meal_type"],
                "session_id": session_id,
                "start_elapsed_seconds": round(start_elapsed, 3),
                "first_question": start_response.get("first_question"),
                "turns": turns,
            }
        )
        if checkpoint_path is not None:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_text(
                json.dumps(
                    _build_probe_result(
                        base_url=base_url,
                        started_at=started_at,
                        scenario_set=scenario_set,
                        max_turns=max_turns,
                        enable_alignment=enable_alignment,
                        enable_uncertainty=enable_uncertainty,
                        scenario_count=len(scenarios),
                        results=results,
                    ),
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
    return _build_probe_result(
        base_url=base_url,
        started_at=started_at,
        scenario_set=scenario_set,
        max_turns=max_turns,
        enable_alignment=enable_alignment,
        enable_uncertainty=enable_uncertainty,
        scenario_count=len(scenarios),
        results=results,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--max-turns", type=int, default=5)
    parser.add_argument(
        "--scenario-set",
        choices=sorted(SCENARIO_SETS),
        default="short",
        help="Scenario suite to run. Use 'long' for realistic longer dialogues.",
    )
    parser.add_argument("--enable-alignment", action="store_true")
    parser.add_argument("--enable-uncertainty", action="store_true")
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    output_path = Path(args.output) if args.output else None

    result = run_scenarios(
        args.base_url,
        max_turns=args.max_turns,
        scenario_set=args.scenario_set,
        enable_alignment=args.enable_alignment,
        enable_uncertainty=args.enable_uncertainty,
        checkpoint_path=output_path,
    )
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
