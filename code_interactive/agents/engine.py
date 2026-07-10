"""Portable conversation engine for one chatbot turn."""

from __future__ import annotations

import concurrent.futures as _futures
import re
import time
from collections.abc import Callable
from dataclasses import replace

from .agent_config import AgentConfig
from .contracts import (
    AssistantReply,
    CoachingState,
    CoachingTurnRequest,
    CoachingTurnResult,
    UserProfileContext,
)
from .history_adapter import build_shared_history
from .modules.alignment_estimator import AlignmentEstimator
from .modules.certainty_estimator import CertaintyEstimator
from .modules.context_tracker import ContextTracker
from .modules.dialogue_planner import DialoguePlanner
from .modules.guardrail import Guardrail
from .modules.information_seeker import InformationSeeker
from .modules.interaction_state_tracker import InteractionStateTracker
from .modules.meal_recommender import MealRecommender
from .modules.meal_tracker import MealTrackerModel
from .modules.meal_assessor import MealAssessor
from .modules.response_generator import ResponseGenerator
from .opening import build_opening_message
from .output_schemas import (
    ASSESSMENT_SCHEMA,
    DIALOGUE_PLAN_SCHEMA,
    INTERACTION_STATE_SCHEMA,
    RECOMMENDATION_SCHEMA,
)

_ENGAGED_INTENTS = frozenset({"informing", "accepting", "inquiring", "deferring"})
_RESISTANT_INTENTS = frozenset({"rejecting", "disengaging"})
_VALID_INTENTS = _ENGAGED_INTENTS | _RESISTANT_INTENTS | {"passive"}


class ConversationEngine:
    """Run one chatbot turn from supplied history and state.

    The engine owns agent flow only. It receives an LLM generation function from
    the host app, so web sessions, client pools, and model configuration stay
    outside this portable boundary.
    """

    def __init__(
        self,
        generate_response: Callable[..., str],
        config: AgentConfig | None = None,
    ) -> None:
        self._generate_response = generate_response
        self._timing_records: list[dict] = []
        self.config = config or AgentConfig()

    def generate_chat_replies(self, request: CoachingTurnRequest) -> CoachingTurnResult:
        """Generate one or more assistant bubbles for ``request.current_message``."""

        turn_started = time.perf_counter()
        self._timing_records = []
        clean_message = request.current_message.strip()
        if not clean_message:
            raise ValueError("current_message must not be blank")

        opening = request.opening_message
        if opening is None and request.enable_opening_fallback:
            opening = build_opening_message(request.profile)

        prior_state = request.state or CoachingState()
        adapted = build_shared_history(
            request.history,
            clean_message,
            context_window=self.config.context_window,
            state=prior_state,
            opening_message=opening,
            use_opening_fallback=request.enable_opening_fallback,
        )
        history = adapted.history
        prior_state = replace(
            prior_state,
            user_preferences=self._merge_user_preferences(
                prior_state.user_preferences,
                request.profile,
            ),
        )
        phase = prior_state.phase or "exploration"
        metadata: dict = {
            "turn_idx": adapted.turn_idx,
            "opening_used": adapted.opening_used,
        }

        coach = InformationSeeker(
            model=None,
            nutrition_goal=request.nutrition_goal,
            meal_type=request.meal_type,
            config=self.config,
        )
        meal_tracker = MealTrackerModel(model=None, config=self.config)
        context_tracker = ContextTracker()
        interaction_tracker = InteractionStateTracker()
        meal_assessor = MealAssessor(request.nutrition_goal, self.config)
        dialogue_planner = DialoguePlanner(request.nutrition_goal, self.config)
        response_generator = ResponseGenerator(request.nutrition_goal, self.config)
        recommender = MealRecommender(request.nutrition_goal, self.config)
        guardrail = Guardrail(config=self.config)
        if request.profile is not None:
            context_tracker.set_profile_from_persona(
                activity_level=request.profile.activity_level,
                diet_preferences=list(request.profile.preferences or ()),
                allergies=list(request.profile.allergies or ()),
                health_concerns=list(
                    request.profile.extra.get("health_concerns") or ()
                ),
            )

        early_result = self._run_tracking_stage(
            request=request,
            clean_message=clean_message,
            history=history,
            prior_state=prior_state,
            guardrail=guardrail,
            meal_tracker=meal_tracker,
            context_tracker=context_tracker,
            interaction_tracker=interaction_tracker,
            metadata=metadata,
        )
        if early_result is not None:
            return self._attach_latency_metadata(early_result, turn_started)

        alignment_score, alignment_reasoning = self._estimate_alignment_state(
            request=request,
            history=history,
            prior_state=prior_state,
            turn_idx=adapted.turn_idx,
            metadata=metadata,
        )
        certainty_score, certainty_reasoning = self._estimate_certainty_state(
            request=request,
            history=history,
            prior_state=prior_state,
            metadata=metadata,
        )
        decision = self._plan_dialogue(
            dialogue_planner=dialogue_planner,
            history=history,
            prior_state=prior_state,
            turn_idx=adapted.turn_idx,
            current_phase=phase,
            alignment_score=alignment_score,
            alignment_reasoning=alignment_reasoning,
            certainty_score=certainty_score,
            certainty_reasoning=certainty_reasoning,
            interaction_state=history.interaction_state,
            metadata=metadata,
        )
        (
            action,
            phase,
            decision,
            user_intent,
            intent_summary,
            stall_count,
            rejection_count,
        ) = self._apply_planning_consistency_checks(
            decision=decision,
            phase=decision.get("accepted_phase") or phase,
            prior_state=prior_state,
            alignment_score=alignment_score,
            request=request,
            current_message=clean_message,
            meal_base=history.meal_base,
            interaction_state=history.interaction_state,
            metadata=metadata,
        )
        assistant_messages, phase, status, terminated_by = self._generate_action_replies(
            action=action,
            phase=phase,
            decision=decision,
            request=request,
            history=history,
            prior_state=prior_state,
            turn_idx=adapted.turn_idx,
            alignment_score=alignment_score,
            alignment_reasoning=alignment_reasoning,
            user_intent=user_intent,
            coach=coach,
            recommender=recommender,
            meal_assessor=meal_assessor,
            response_generator=response_generator,
            metadata=metadata,
        )

        next_state = CoachingState(
            phase=phase,
            status=status,
            meal_base=history.meal_base,
            tracker_state=history.tracker_state,
            context_base=history.context_base,
            interaction_state=history.interaction_state,
            user_preferences=prior_state.user_preferences,
            recommendation_history=self._updated_recommendation_history(
                prior_state=prior_state,
                metadata=metadata,
                turn_idx=adapted.turn_idx,
            ),
            consecutive_qa_count=(
                prior_state.consecutive_qa_count + 1 if action == "respond" else 0
            ),
            stall_count=stall_count,
            recommendation_rejection_count=rejection_count,
            last_intent_summary=intent_summary,
            last_user_intent=user_intent,
            last_alignment_score=alignment_score,
            last_alignment_reasoning=alignment_reasoning,
            last_certainty_score=certainty_score,
            last_certainty_reasoning=certainty_reasoning,
            safety_clarification_counts=metadata.get(
                "safety_clarification_counts",
                prior_state.safety_clarification_counts,
            ),
        )
        result = CoachingTurnResult(
            assistant_messages=assistant_messages,
            state=next_state,
            status=status,
            terminated_by=terminated_by,
            metadata=metadata,
        )
        return self._attach_latency_metadata(result, turn_started)

    @staticmethod
    def _updated_recommendation_history(
        *,
        prior_state: CoachingState,
        metadata: dict,
        turn_idx: int,
    ) -> tuple[dict, ...]:
        """Append the current recommendation to compact cross-turn memory."""
        history = [dict(item) for item in prior_state.recommendation_history]
        recommendation = metadata.get("recommendation_result")
        if not isinstance(recommendation, dict):
            return tuple(history)

        compact = {
            "turn_idx": recommendation.get("turn_idx", turn_idx),
            "bundle_semantics": "parallel_adjustments",
            "recommendation_type": str(
                recommendation.get("recommendation_type", "")
            ),
            "target_food": str(recommendation.get("target_food", "")),
            "suggestion": str(recommendation.get("suggestion", "")),
            "reasoning": str(recommendation.get("reasoning", "")),
            "expected_impact": str(recommendation.get("expected_impact", "")),
        }
        options = recommendation.get("options")
        if isinstance(options, list):
            compact["options"] = [
                {
                    "option_id": str(option.get("option_id", "")),
                    "target_food": str(option.get("target_food", "")),
                    "suggestion": str(option.get("suggestion", "")),
                    "expected_impact": str(option.get("expected_impact", "")),
                }
                for option in options
                if isinstance(option, dict)
            ]
        signature = (
            compact["recommendation_type"].lower(),
            compact["target_food"].lower(),
            compact["suggestion"].lower(),
            "|".join(
                str(option.get("suggestion", "")).lower()
                for option in compact.get("options", [])
                if isinstance(option, dict)
            ),
        )
        for item in history:
            prior_signature = (
                str(item.get("recommendation_type", "")).lower(),
                str(item.get("target_food", "")).lower(),
                str(item.get("suggestion", "")).lower(),
                "|".join(
                    str(option.get("suggestion", "")).lower()
                    for option in item.get("options", [])
                    if isinstance(option, dict)
                ),
            )
            if prior_signature == signature:
                return tuple(history[-6:])

        history.append(compact)
        return tuple(history[-6:])

    def _generate(
        self,
        *,
        module: str,
        messages,
        mode: str,
        response_schema: dict | None = None,
    ) -> str:
        """Call the host LLM function and record module-level latency."""
        started = time.perf_counter()
        try:
            try:
                return self._generate_response(
                    module=module,
                    messages=messages,
                    mode=mode,
                    response_schema=response_schema,
                )
            except TypeError as exc:
                if "response_schema" not in str(exc):
                    raise
                return self._generate_response(
                    module=module,
                    messages=messages,
                    mode=mode,
                )
        finally:
            self._timing_records.append(
                {
                    "module": module,
                    "mode": mode,
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                }
            )

    def _attach_latency_metadata(
        self,
        result: CoachingTurnResult,
        turn_started: float,
    ) -> CoachingTurnResult:
        """Attach end-to-end and per-module timing to the turn metadata."""
        module_totals: dict[str, float] = {}
        for record in self._timing_records:
            module = record["module"]
            module_totals[module] = module_totals.get(module, 0.0) + float(
                record["elapsed_seconds"]
            )
        result.metadata["latency"] = {
            "total_seconds": round(time.perf_counter() - turn_started, 3),
            "module_calls": list(self._timing_records),
            "module_totals": {
                module: round(seconds, 3)
                for module, seconds in sorted(module_totals.items())
            },
            "module_call_count": len(self._timing_records),
        }
        return result

    @classmethod
    def _merge_user_preferences(
        cls,
        existing: str,
        profile: UserProfileContext | None,
    ) -> str:
        """Build one prompt-ready constraint block from persisted and profile facts."""
        blocks = [existing.strip()] if existing and existing.strip() else []
        profile_text = cls._format_profile_preferences(profile)
        if profile_text:
            blocks.append(profile_text)
        deduped: list[str] = []
        for block in blocks:
            if block and block not in deduped:
                deduped.append(block)
        return "\n\n".join(deduped)

    @staticmethod
    def _format_profile_preferences(profile: UserProfileContext | None) -> str:
        if profile is None:
            return ""
        lines: list[str] = []
        if profile.activity_level:
            lines.append(f"Activity Level: {profile.activity_level}")
        if profile.preferences:
            lines.append("Diet Preferences: " + ", ".join(map(str, profile.preferences)))
        if profile.allergies:
            lines.append("Allergies: " + ", ".join(map(str, profile.allergies)))
        health_concerns = profile.extra.get("health_concerns") if profile.extra else None
        if health_concerns:
            lines.append("Health Concerns: " + ", ".join(map(str, health_concerns)))
        return "\n".join(lines)

    @classmethod
    def _append_profile_constraints_to_interaction_state(
        cls,
        interaction_state: str,
        profile: UserProfileContext | None,
    ) -> str:
        """Add profile constraints as operational memory without marking all as boundaries."""
        constraints = cls._profile_constraint_lines(profile)
        base = (interaction_state or "").strip()
        if not constraints:
            return base
        existing = base.lower()
        new_lines = [
            line for line in constraints
            if line.lower() not in existing
        ]
        if not new_lines:
            return base
        block = "Known profile constraints:\n" + "\n".join(f"- {line}" for line in new_lines)
        return f"{base}\n{block}".strip() if base else block

    @staticmethod
    def _profile_constraint_lines(profile: UserProfileContext | None) -> list[str]:
        if profile is None:
            return []
        lines: list[str] = []
        for item in profile.allergies or ():
            text = str(item).strip()
            if text:
                lines.append(f"Allergy constraint: {text}")
        health_concerns = profile.extra.get("health_concerns") if profile.extra else None
        for item in health_concerns or ():
            text = str(item).strip()
            if text:
                lines.append(f"Health concern: {text}")
        return lines

    def _run_tracking_stage(
        self,
        *,
        request: CoachingTurnRequest,
        clean_message: str,
        history,
        prior_state: CoachingState,
        guardrail: Guardrail,
        meal_tracker: MealTrackerModel,
        context_tracker: ContextTracker,
        interaction_tracker: InteractionStateTracker,
        metadata: dict,
    ) -> CoachingTurnResult | None:
        """Run input guardrail, meal tracking, context tracking, and interaction tracking."""

        guard_messages = None
        if request.enable_guardrail:
            guard_messages = guardrail.get_input_guard_messages(
                user_input=clean_message,
                dialog_context=history.to_recent_turns_text(n=2),
            )
        meal_turns = (
            history.to_recent_turns_text(n=1)
            if prior_state.tracker_state
            else history.to_plain_text()
        )
        meal_messages = meal_tracker.get_messages(
            meal_turns,
            prev_tracker_state=prior_state.tracker_state,
        )
        context_messages = (
            context_tracker.get_messages(history.to_plain_text())
            if request.enable_context_tracking
            else None
        )
        interaction_turns = (
            history.to_recent_turns_text(n=1)
            if prior_state.interaction_state
            else history.to_plain_text()
        )
        interaction_messages = (
            interaction_tracker.get_messages(
                interaction_turns,
                prev_interaction_state=prior_state.interaction_state,
            )
            if self.config.use_interaction_tracker
            else None
        )

        executor = _futures.ThreadPoolExecutor(max_workers=4)
        closed = False
        try:
            guard_future = (
                executor.submit(
                    self._generate,
                    module="guardrail",
                    messages=guard_messages,
                    mode="guardrail",
                )
                if guard_messages is not None
                else None
            )
            meal_future = executor.submit(
                self._generate,
                module="meal_tracker",
                messages=meal_messages,
                mode="tracker",
            )
            context_future = (
                executor.submit(
                    self._generate,
                    module="context_tracker",
                    messages=context_messages,
                    mode="tracker",
                )
                if context_messages is not None
                else None
            )
            interaction_future = (
                executor.submit(
                self._generate,
                module="interaction_tracker",
                messages=interaction_messages,
                mode="tracker",
                response_schema=INTERACTION_STATE_SCHEMA,
            )
                if interaction_messages is not None
                else None
            )

            if guard_future is not None:
                guard_raw = guard_future.result()
                guard_result = guardrail.parse_input_guard(guard_raw)
                metadata["input_guard"] = guard_result
                early_result = self._handle_guardrail_action(
                    guard_result=guard_result,
                    prior_state=prior_state,
                    metadata=metadata,
                )
                if early_result is not None:
                    meal_future.cancel()
                    if context_future is not None:
                        context_future.cancel()
                    if interaction_future is not None:
                        interaction_future.cancel()
                    executor.shutdown(wait=False, cancel_futures=True)
                    closed = True
                    return early_result

            meal_raw = meal_future.result()
            parsed_meal = meal_tracker.parse_tracking_output(meal_raw)
            parsed_meal = self._protect_meal_state_from_empty_update(
                parsed_meal=parsed_meal,
                prior_state=prior_state,
            )
            history.update_tracker_state(parsed_meal["tracker_state"])
            history.update_meal_base(parsed_meal["meal_base"])
            metadata["meal_tracker_output"] = meal_raw

            if context_future is not None:
                context_raw = context_future.result()
                history.update_context_base(context_raw)
                metadata["context_tracker_output"] = context_raw
            if interaction_future is not None:
                interaction_raw = interaction_future.result()
                interaction_state = interaction_tracker.parse_output(
                    interaction_raw,
                    fallback=prior_state.interaction_state,
                )
                interaction_state, repair_meta = self._repair_interaction_state(
                    interaction_state=interaction_state,
                    current_message=clean_message,
                    prior_state=prior_state,
                )
                history.update_interaction_state(
                    self._append_profile_constraints_to_interaction_state(
                        interaction_state,
                        request.profile,
                    )
                )
                metadata["interaction_tracker_output"] = interaction_raw
                if repair_meta:
                    metadata["interaction_state_repair"] = repair_meta
            else:
                history.update_interaction_state(
                    self._append_profile_constraints_to_interaction_state(
                        history.interaction_state,
                        request.profile,
                    )
                )
            metadata["interaction_state"] = history.interaction_state
            return None
        finally:
            if not closed:
                executor.shutdown(wait=True)

    @staticmethod
    def _protect_meal_state_from_empty_update(
        *,
        parsed_meal: dict,
        prior_state: CoachingState,
    ) -> dict:
        """Prevent an empty tracker turn from erasing established meal facts."""
        meal_base = str(parsed_meal.get("meal_base", "") or "")
        tracker_state = str(parsed_meal.get("tracker_state", "") or "")
        meal_looks_empty = bool(
            re.search(r"Food items:\s*(not yet mentioned|none)\b", meal_base, re.I)
        )
        prior_meal = (prior_state.meal_base or "").strip()
        if meal_looks_empty and prior_meal and not re.search(
            r"Food items:\s*(not yet mentioned|none)\b",
            prior_meal,
            re.I,
        ):
            return {
                "tracker_state": prior_state.tracker_state or tracker_state,
                "meal_base": prior_meal,
            }
        return parsed_meal

    def _handle_guardrail_action(
        self,
        *,
        guard_result: dict,
        prior_state: CoachingState,
        metadata: dict,
    ) -> CoachingTurnResult | None:
        action = guard_result.get("action")
        if action == "crisis":
            reply = (
                "I hear that you're going through a really difficult time. "
                "Please reach out to the 988 Suicide & Crisis Lifeline "
                "(call or text 988) or contact someone you trust. "
                "You deserve support."
            )
            return CoachingTurnResult(
                assistant_messages=[AssistantReply(reply, kind="safety")],
                state=replace(prior_state, status="terminated"),
                status="terminated",
                terminated_by="crisis_detected",
                metadata=metadata,
            )
        if action == "block":
            return CoachingTurnResult(
                assistant_messages=[
                    AssistantReply(str(guard_result.get("message") or ""), kind="redirect")
                ],
                state=prior_state,
                metadata=metadata,
            )
        return None

    @classmethod
    def _repair_interaction_state(
        cls,
        *,
        interaction_state: str,
        current_message: str,
        prior_state: CoachingState,
    ) -> tuple[str, dict]:
        """Apply deterministic state repairs that should not depend on LLM recall.

        The tracker infers broad operational memory, but some continuity rules
        are mechanical: the latest user message must become the latest stance,
        ordinal references to a recommendation bundle must resolve to concrete
        options, and stale open issues should not survive after the user answers
        or redirects them.
        """
        data = InteractionStateTracker.parse_formatted_state(interaction_state)
        repairs: list[str] = []
        message = (current_message or "").strip()
        lowered = message.lower()

        if message:
            data["latest_user_position"] = cls._compact_latest_user_position(message)
            repairs.append("latest_user_position_refreshed")

        accepted_from_bundle = cls._accepted_bundle_options_from_message(
            lowered,
            prior_state.recommendation_history,
        )
        if accepted_from_bundle:
            cls._extend_unique(data, "accepted_options", accepted_from_bundle)
            repairs.append("bundle_ordinal_acceptance_resolved")

        preserved_components = cls._preserved_components_from_message(message)
        if preserved_components:
            for component in preserved_components:
                cls._extend_unique(data, "answered_facts", [f"The user preserved {component} as part of the meal."])
                cls._extend_unique(data, "meal_slots", [f"preserved component: {component}"])
            repairs.append("preserved_component_recorded")

        if cls._is_dialogue_fatigue_or_repetition_complaint(lowered):
            data["active_issue"] = (
                "Respect the user's fatigue or repetition complaint and move "
                "toward concise closure without new questions."
            )
            repairs.append("fatigue_active_issue_set")
        elif cls._is_topic_repair_message(lowered):
            data["active_issue"] = "Address the user's requested topic shift without revisiting settled points."
            repairs.append("topic_repair_active_issue_set")
        elif accepted_from_bundle:
            data["active_issue"] = (
                "Assess accepted bundle adjustment(s) without asking the user "
                "to choose the same item again."
            )
            data["open_questions"] = cls._drop_resolved_option_questions(
                data.get("open_questions", []),
                accepted_from_bundle,
            )
            repairs.append("resolved_bundle_questions_removed")
        elif cls._is_current_commitment_statement(message):
            data["active_issue"] = "Assess the user's current commitment and avoid repeating settled choices."
            data["open_questions"] = cls._drop_generic_choice_questions(
                data.get("open_questions", [])
            )
            repairs.append("commitment_active_issue_set")

        cls._remove_cross_section_duplicates(data)
        repaired = InteractionStateTracker.format_state(data)
        return repaired or interaction_state, {"repairs": repairs} if repairs else {}

    @staticmethod
    def _compact_latest_user_position(message: str) -> str:
        text = " ".join(message.split())
        if len(text) > 220:
            text = text[:217].rstrip() + "..."
        return f"The user said: {text}"

    @classmethod
    def _accepted_bundle_options_from_message(
        cls,
        lowered_message: str,
        recommendation_history,
    ) -> list[str]:
        if not recommendation_history:
            return []
        if not re.search(r"\b(first|second|third|1st|2nd|3rd|option\s*[123])\b", lowered_message):
            return []
        if not re.search(r"\b(doable|works?|sounds?\s+good|yes|accept|will|go\s+with|do\s+both)\b", lowered_message):
            return []
        latest = dict(recommendation_history[-1])
        options = latest.get("options")
        if not isinstance(options, list):
            return []
        index_map = {
            "first": 0,
            "1st": 0,
            "option 1": 0,
            "option1": 0,
            "second": 1,
            "2nd": 1,
            "option 2": 1,
            "option2": 1,
            "third": 2,
            "3rd": 2,
            "option 3": 2,
            "option3": 2,
        }
        selected_indices: set[int] = set()
        for token, idx in index_map.items():
            if re.search(rf"\b{re.escape(token)}\b", lowered_message):
                selected_indices.add(idx)
        accepted: list[str] = []
        for idx in sorted(selected_indices):
            if idx >= len(options) or not isinstance(options[idx], dict):
                continue
            suggestion = str(options[idx].get("suggestion", "")).strip()
            target = str(options[idx].get("target_food", "")).strip()
            option_id = str(options[idx].get("option_id", f"opt{idx + 1}")).strip()
            label = suggestion or target
            if label:
                accepted.append(f"{option_id}: {label}")
        return accepted

    @staticmethod
    def _preserved_components_from_message(message: str) -> list[str]:
        patterns = (
            r"\b(?:did\s+not|didn't|never)\s+(?:give\s+up|drop|reject)\s+([^.;!?]+)",
            r"\b(?:still\s+want|want\s+to\s+keep|keep|keeping)\s+([^.;!?]+)",
            r"\b([^.;!?]+)\s+still\s+should\s+be\s+considered\b",
        )
        found: list[str] = []
        for pattern in patterns:
            for match in re.finditer(pattern, message, flags=re.IGNORECASE):
                component = " ".join(match.group(1).split()).strip(" ,")
                if component and len(component) <= 80 and component.lower() not in {"it", "this", "that"}:
                    if component not in found:
                        found.append(component)
        return found[:3]

    @staticmethod
    def _is_dialogue_fatigue_or_repetition_complaint(lowered_message: str) -> bool:
        """Detect fatigue with the dialogue, not low-effort meal preferences."""
        patterns = (
            r"\btired\s+of\s+(?:this|these|answering|questions?|chat|conversation|talking|repeating)\b",
            r"\bexhausted\s+(?:by|with|from)\s+(?:this|these|answering|questions?|chat|conversation|talking|repeating)\b",
            r"\bthis\s+(?:is|feels)\s+(?:tiring|exhausting|frustrating|repetitive)\b",
            r"\btoo\s+many\s+questions?\b",
            r"\bhow\s+long\b",
            r"\bwhy\s+are\s+you\s+asking\b",
            r"\basking\s+(?:it|this|that)\s+again\b",
            r"\brepeat(?:ing|ed)?\b",
            r"\bstuck\s+(?:in|on|with)\s+(?:this|these|questions?|conversation|chat)\b",
        )
        return any(re.search(pattern, lowered_message) for pattern in patterns)

    @staticmethod
    def _is_forced_tradeoff_rejection(message: str) -> bool:
        lowered = message.lower()
        return bool(
            re.search(r"\bno\b.*\b(i|we)\s+want\s+both\b", lowered)
            or re.search(r"\bkeep(?:ing)?\s+both\b", lowered)
            or re.search(r"\bboth\b.*\b(good|pair|together|combo)\b", lowered)
            or re.search(r"\bdon'?t\s+make\s+me\s+choose\b", lowered)
            or re.search(r"\bnot\s+choos(?:e|ing)\s+(?:one|between)\b", lowered)
        )

    @staticmethod
    def _is_topic_repair_message(lowered_message: str) -> bool:
        return bool(
            re.search(r"\bwhy\s+(?:do|are)\s+you\b", lowered_message)
            or re.search(r"\bwhen\s+are\s+you\s+talking\s+about\b", lowered_message)
            or re.search(r"\bstay\s+on\b", lowered_message)
            or re.search(r"\bfocus\s+on\b", lowered_message)
        )

    @staticmethod
    def _drop_resolved_option_questions(
        questions: list,
        accepted_options: list[str],
    ) -> list[str]:
        accepted_text = " ".join(accepted_options).lower()
        kept: list[str] = []
        for question in questions or []:
            q = str(question)
            q_lower = q.lower()
            if "which" in q_lower and any(part in q_lower for part in ("option", "doable", "choose")):
                continue
            if accepted_text and any(token and token in accepted_text for token in re.findall(r"[a-z]{4,}", q_lower)):
                continue
            kept.append(q)
        return kept

    @staticmethod
    def _drop_generic_choice_questions(questions: list) -> list[str]:
        kept: list[str] = []
        for question in questions or []:
            q = str(question)
            if re.search(r"\b(which|whether|would\s+you|do\s+you\s+want)\b", q.lower()):
                continue
            kept.append(q)
        return kept

    @staticmethod
    def _extend_unique(data: dict, key: str, values: list[str]) -> None:
        items = list(data.get(key) or [])
        lowered = {str(item).lower() for item in items}
        for value in values:
            text = str(value).strip()
            if text and text.lower() not in lowered:
                items.append(text)
                lowered.add(text.lower())
        data[key] = items

    @staticmethod
    def _remove_cross_section_duplicates(data: dict) -> None:
        terminal_keys = (
            "accepted_options",
            "rejected_options",
            "unavailable_options",
            "safety_conflicted_options",
        )
        terminal = {
            str(item).strip().lower()
            for key in terminal_keys
            for item in data.get(key, []) or []
            if str(item).strip()
        }
        data["candidate_options"] = [
            item
            for item in data.get("candidate_options", []) or []
            if str(item).strip().lower() not in terminal
        ]

    def _estimate_alignment_state(
        self,
        *,
        request: CoachingTurnRequest,
        history,
        prior_state: CoachingState,
        turn_idx: int,
        metadata: dict,
    ) -> tuple[float | None, str | None]:
        if not request.enable_alignment:
            metadata["alignment_enabled"] = False
            metadata["alignment_aligned"] = None
            return None, None

        alignment = AlignmentEstimator(
            model=None,
            nutrition_goal=request.nutrition_goal,
            config=self.config,
        )
        alignment_messages = alignment.get_messages(history)
        alignment_raw = self._generate(
            module="alignment_estimator",
            messages=alignment_messages,
            mode="alignment",
        )
        alignment_aligned = alignment.apply_judgment(alignment_raw, turn_idx)
        metadata["alignment_enabled"] = True
        metadata["alignment_raw_output"] = alignment_raw
        metadata["alignment_aligned"] = alignment_aligned
        return alignment.last_score, alignment.last_reasoning

    def _estimate_certainty_state(
        self,
        *,
        request: CoachingTurnRequest,
        history,
        prior_state: CoachingState,
        metadata: dict,
    ) -> tuple[float | None, str | None]:
        if not request.enable_certainty:
            metadata["certainty_enabled"] = False
            return None, None

        certainty = CertaintyEstimator(request.nutrition_goal, self.config)
        certainty_messages = certainty.get_messages(history)
        certainty_raw = self._generate(
            module="certainty_estimator",
            messages=certainty_messages,
            mode="certainty",
        )
        certainty_reasoning, certainty_score = certainty.parse_output(certainty_raw)
        metadata["certainty_enabled"] = True
        metadata["certainty_raw_output"] = certainty_raw
        return certainty_score, certainty_reasoning

    def _plan_dialogue(
        self,
        *,
        dialogue_planner: DialoguePlanner,
        history,
        prior_state: CoachingState,
        turn_idx: int,
        current_phase: str,
        alignment_score: float | None,
        alignment_reasoning: str | None,
        certainty_score: float | None,
        certainty_reasoning: str | None,
        interaction_state: str,
        metadata: dict,
    ) -> dict:
        planner_messages = dialogue_planner.get_messages(
            history=history,
            turn_idx=turn_idx,
            current_phase=current_phase,
            recommendation_history=list(prior_state.recommendation_history),
            consecutive_qa_count=prior_state.consecutive_qa_count,
            last_alignment_score=alignment_score,
            last_alignment_reasoning=alignment_reasoning,
            last_certainty_score=certainty_score,
            last_certainty_reasoning=certainty_reasoning,
            user_preferences=prior_state.user_preferences,
            interaction_state=interaction_state,
        )
        planner_raw = self._generate(
            module="dialogue_planner",
            messages=planner_messages,
            mode="planner",
            response_schema=DIALOGUE_PLAN_SCHEMA,
        )
        decision = dialogue_planner.parse_output(planner_raw, current_phase)
        metadata["dialogue_planner_raw_output"] = planner_raw
        metadata["dialogue_plan"] = decision
        metadata["phase_prediction"] = {
            "predicted_phase": decision.get("accepted_phase", current_phase),
            "confidence": decision.get("confidence"),
            "reasoning": decision.get("reasoning", ""),
            "source": "dialogue_planner",
        }
        # Backwards-compatible monitoring field for the current UI/tests.
        metadata["dialogue_plan"] = decision
        return decision

    def _apply_planning_consistency_checks(
        self,
        *,
        decision: dict,
        phase: str,
        prior_state: CoachingState,
        alignment_score: float | None,
        request: CoachingTurnRequest,
        current_message: str,
        meal_base: str,
        interaction_state: str,
        metadata: dict,
    ) -> tuple[str, str, dict, str, str, int, int]:
        planned_action = decision.get("action", "inquire")
        action = planned_action
        user_intent = str(decision.get("user_intent", "passive") or "passive").lower()
        if user_intent not in _VALID_INTENTS:
            user_intent = "passive"
        intent_summary = str(decision.get("intent_summary", "") or "")

        if user_intent in _ENGAGED_INTENTS:
            stall_count = 0
            rejection_count = 0
        elif user_intent in _RESISTANT_INTENTS:
            stall_count = prior_state.stall_count + 1
            rejection_count = prior_state.recommendation_rejection_count + 1
        else:
            stall_count = prior_state.stall_count + 1
            rejection_count = 0

        user_wants_to_end = (
            user_intent == "disengaging"
            or self._message_sets_stop_boundary(current_message)
        )
        fatigue_or_repetition = self._is_dialogue_fatigue_or_repetition_complaint(
            current_message.lower()
        )
        if fatigue_or_repetition:
            user_wants_to_end = True
        closure_readiness = str(
            decision.get("closure_readiness") or "not_ready"
        ).lower()
        actionability = str(decision.get("actionability") or "insufficient").lower()
        planning_policy: dict = {
            "intent_summary": intent_summary,
            "user_intent": user_intent,
            "planned_action": planned_action,
            "stall_count": stall_count,
            "recommendation_rejection_count": rejection_count,
            "user_wants_to_end": user_wants_to_end,
            "closure_readiness": closure_readiness,
            "actionability": actionability,
        }
        confirmation_satisfied = self._confirmation_satisfied(
            prior_state=prior_state,
            interaction_state=interaction_state,
            user_intent=user_intent,
            actionability=actionability,
        )
        planning_policy["confirmation_satisfied"] = confirmation_satisfied
        gate_result = self._apply_commitment_action_gate(
            action=action,
            phase=phase,
            decision=decision,
            request=request,
            prior_state=prior_state,
            current_message=current_message,
            meal_base=meal_base,
            interaction_state=interaction_state,
            user_intent=user_intent,
            actionability=actionability,
        )
        action = gate_result["action"]
        phase = gate_result["phase"]
        decision = gate_result["decision"]
        actionability = str(decision.get("actionability") or actionability)
        closure_readiness = str(
            decision.get("closure_readiness") or closure_readiness
        )
        planning_policy["actionability"] = actionability
        planning_policy["closure_readiness"] = closure_readiness
        if gate_result["applied"]:
            metadata["dialogue_plan"] = decision
            metadata["commitment_gate"] = gate_result["metadata"]
            planning_policy["override"] = gate_result["metadata"]["gate"]
        if gate_result["safety_clarification_counts"] is not None:
            metadata["safety_clarification_counts"] = gate_result[
                "safety_clarification_counts"
            ]

        sufficiency_gate = self._apply_exploration_sufficiency_gate(
            action=action,
            phase=phase,
            decision=decision,
            request=request,
            prior_state=prior_state,
            current_message=current_message,
            meal_base=meal_base,
            interaction_state=interaction_state,
            user_intent=user_intent,
        )
        if sufficiency_gate["applied"]:
            action = sufficiency_gate["action"]
            phase = sufficiency_gate["phase"]
            decision = sufficiency_gate["decision"]
            actionability = str(decision.get("actionability") or actionability)
            closure_readiness = str(
                decision.get("closure_readiness") or closure_readiness
            )
            metadata["dialogue_plan"] = decision
            metadata["exploration_sufficiency_gate"] = sufficiency_gate["metadata"]
            planning_policy["override"] = sufficiency_gate["metadata"]["gate"]
            planning_policy["actionability"] = actionability
            planning_policy["closure_readiness"] = closure_readiness

        saturation_gate = self._apply_assessment_saturation_gate(
            action=action,
            phase=phase,
            decision=decision,
            prior_state=prior_state,
            current_message=current_message,
            interaction_state=interaction_state,
            user_intent=user_intent,
            actionability=actionability,
            closure_readiness=closure_readiness,
        )
        if saturation_gate["applied"]:
            action = saturation_gate["action"]
            phase = saturation_gate["phase"]
            decision = saturation_gate["decision"]
            actionability = str(decision.get("actionability") or actionability)
            closure_readiness = str(
                decision.get("closure_readiness") or closure_readiness
            )
            metadata["dialogue_plan"] = decision
            metadata["assessment_saturation_gate"] = saturation_gate["metadata"]
            planning_policy["override"] = saturation_gate["metadata"]["gate"]
            planning_policy["actionability"] = actionability
            planning_policy["closure_readiness"] = closure_readiness

        if fatigue_or_repetition and action not in ("close", "terminate"):
            action = "close"
            phase = "finalization"
            closure_readiness = "boundary_close"
            decision = {
                **decision,
                "action": action,
                "accepted_phase": phase,
                "closure_readiness": closure_readiness,
                "instruction": (
                    "Acknowledge the user's frustration or fatigue, apologize "
                    "briefly if appropriate, recap the settled plan concisely, "
                    "and stop without asking another question."
                ),
                "reasoning": (
                    "Planning consistency check closed because the user showed "
                    "fatigue or complained about repetition."
                ),
            }
            metadata["dialogue_plan"] = decision
            planning_policy["override"] = "fatigue_or_repetition_close"
            planning_policy["closure_readiness"] = closure_readiness

        if (
            self._is_forced_tradeoff_rejection(current_message)
            and action not in ("close", "terminate")
        ):
            action = "assess"
            phase = "assessment"
            closure_readiness = "actionable"
            decision = {
                **decision,
                "action": action,
                "accepted_phase": phase,
                "closure_readiness": closure_readiness,
                "actionability": "workable",
                "assessment_followup_action": "confirm",
                "assessment_followup_phase": "confirmation",
                "assessment_followup_instruction": (
                    "Acknowledge the user's boundary to keep the chosen items. "
                    "Assess the tradeoff and confirm a compromise plan without "
                    "asking the user to choose between the preserved items."
                ),
                "instruction": (
                    "Assess the user's preserved choices as a compromise plan. "
                    "Do not force a binary tradeoff the user rejected."
                ),
                "reasoning": (
                    "Planning consistency check converted a rejected forced "
                    "tradeoff into compromise assessment."
                ),
            }
            metadata["dialogue_plan"] = decision
            planning_policy["override"] = "forced_tradeoff_to_compromise"
            planning_policy["actionability"] = "workable"
            planning_policy["closure_readiness"] = closure_readiness

        if action == "inquire" and actionability in ("workable", "settled", "boundary"):
            if actionability == "boundary":
                action = "close"
                phase = "finalization"
                closure_readiness = "boundary_close"
            elif actionability == "settled":
                action = "assess"
                closure_readiness = "ready_to_close"
            else:
                action = "assess"
                closure_readiness = "not_ready"
            decision = {
                **decision,
                "action": action,
                "accepted_phase": phase,
                "closure_readiness": closure_readiness,
                "reasoning": (
                    "Planning consistency check avoided another question because "
                    "the planner marked the state as actionable."
                ),
            }
            metadata["dialogue_plan"] = decision
            planning_policy["override"] = "actionable_inquiry_redirected"

        has_recent_recommendation = bool(prior_state.recommendation_history)
        recommendation_is_redundant = (
            action == "recommend"
            and has_recent_recommendation
            and user_intent not in {"inquiring", "rejecting", "disengaging"}
            and actionability in {"workable", "settled"}
            and closure_readiness in {"actionable", "ready_to_close"}
        )
        recommendation_is_settled = (
            action == "recommend"
            and actionability == "settled"
            and closure_readiness in {"actionable", "ready_to_close"}
        )
        if recommendation_is_settled or recommendation_is_redundant:
            action = "assess"
            phase = "assessment"
            followup_action = "close" if actionability == "settled" else ""
            followup_phase = "finalization" if followup_action else ""
            decision = {
                **decision,
                "action": action,
                "accepted_phase": phase,
                "closure_readiness": closure_readiness,
                "assessment_followup_action": (
                    decision.get("assessment_followup_action") or followup_action
                ),
                "assessment_followup_phase": (
                    decision.get("assessment_followup_phase") or followup_phase
                ),
                "assessment_followup_instruction": (
                    decision.get("assessment_followup_instruction")
                    or (
                        "Assess the current plan and close without introducing "
                        "another replacement recommendation."
                        if followup_action
                        else ""
                    )
                ),
                "reasoning": (
                    "Planning consistency check redirected recommendation to "
                    "assessment because the current plan is already actionable."
                ),
            }
            metadata["dialogue_plan"] = decision
            planning_policy["override"] = (
                "settled_recommendation_redirected"
                if recommendation_is_settled
                else "redundant_recommendation_redirected"
            )

        if (
            action == "recommend"
            and planning_policy.get("override")
            == "assessment_saturation_to_recommendation_refinement"
        ):
            planning_policy["assessment_prerequisite"] = "satisfied_by_prior_cycle"
        elif action == "recommend":
            action = "assess"
            phase = "assessment"
            decision = {
                **decision,
                "action": action,
                "accepted_phase": phase,
                "phase": phase,
                "assessment_followup_action": (
                    decision.get("assessment_followup_action") or "recommend"
                ),
                "assessment_followup_phase": (
                    decision.get("assessment_followup_phase") or "recommendation"
                ),
                "assessment_followup_instruction": (
                    decision.get("assessment_followup_instruction")
                    or decision.get("instruction")
                    or "Recommend one concrete change grounded in the assessment."
                ),
                "reasoning": (
                    "Planning consistency check routed recommendation through "
                    "assessment so the recommendation is grounded in the current "
                    "meal-state analysis."
                ),
            }
            metadata["dialogue_plan"] = decision
            planning_policy["override"] = "recommendation_grounded_by_assessment"

        confirmation_satisfied = self._confirmation_satisfied(
            prior_state=prior_state,
            interaction_state=interaction_state,
            user_intent=user_intent,
            actionability=actionability,
        )
        planning_policy["confirmation_satisfied"] = confirmation_satisfied

        if (
            confirmation_satisfied
            and action == "confirm"
            and closure_readiness in {"actionable", "ready_to_close", "boundary_close"}
        ):
            action = "close"
            phase = "finalization"
            closure_readiness = (
                "boundary_close"
                if closure_readiness == "boundary_close"
                else "ready_to_close"
            )
            decision = {
                **decision,
                "action": action,
                "accepted_phase": phase,
                "actionability": actionability,
                "closure_readiness": closure_readiness,
                "instruction": (
                    "Close briefly because the user has already confirmed the "
                    "current plan and no unresolved issue remains."
                ),
                "reasoning": (
                    "Planning consistency check advanced from confirmation to "
                    "finalization after the user confirmed the plan."
                ),
            }
            metadata["dialogue_plan"] = decision
            planning_policy["override"] = "confirmed_plan_close"
            planning_policy["closure_readiness"] = closure_readiness

        if (
            closure_readiness == "boundary_close"
            and user_wants_to_end
            and action not in ("terminate", "close")
        ):
            action = "close"
            phase = "finalization"
            decision = {
                **decision,
                "action": action,
                "accepted_phase": phase,
                "instruction": (
                    decision.get("instruction")
                    or "Respect the user's boundary and close without further advice."
                ),
                "reasoning": "(planning consistency check: boundary close requested)",
            }
            metadata["dialogue_plan"] = decision
            planning_policy["override"] = "boundary_close"

        if (
            action == "close"
            and closure_readiness != "boundary_close"
            and not user_wants_to_end
            and not confirmation_satisfied
        ):
            action = "confirm"
            phase = "confirmation"
            decision = {
                **decision,
                "action": action,
                "accepted_phase": phase,
                "instruction": (
                    decision.get("instruction")
                    or "Confirm the current meal plan before finalization."
                ),
                "reasoning": (
                    "Planning consistency check inserted confirmation before "
                    "finalization."
                ),
            }
            metadata["dialogue_plan"] = decision
            planning_policy["override"] = "close_to_confirmation"

        if rejection_count >= 3 and action not in ("terminate", "close", "handoff"):
            action = "handoff"
            phase = "negotiation"
            decision = {
                **decision,
                "action": action,
                "accepted_phase": phase,
                "closure_readiness": "actionable",
                "instruction": (
                    "The user has repeatedly resisted the current coaching path. "
                    "Ask them to choose whether they want another option, want to "
                    "keep the current plan with the tradeoff, or want to stop."
                ),
                "reasoning": (
                    "Planning consistency check handed control to the user after "
                    "repeated resistance instead of closing or continuing to optimize."
                ),
            }
            metadata["dialogue_plan"] = decision
            planning_policy["override"] = "resistance_threshold_handoff"

        if (
            action == "close"
            and alignment_score is not None
            and alignment_score < 0.5
            and not user_wants_to_end
            and closure_readiness != "boundary_close"
        ):
            action = "inquire"
            phase = "exploration"
            decision = {
                **decision,
                "action": action,
                "accepted_phase": phase,
                "reasoning": (
                    "Planning consistency check redirected a low-alignment close "
                    "back to meal information seeking."
                ),
                "instruction": (
                    "Ask one concrete question about missing meal details before closing."
                ),
            }
            metadata["dialogue_plan"] = decision
            planning_policy["override"] = "low_alignment_close_redirected"

        phase = self._normalize_effective_phase(
            action=action,
            phase=phase,
            prior_state=prior_state,
            user_intent=user_intent,
            actionability=actionability,
        )
        if decision.get("accepted_phase") != phase:
            decision = {
                **decision,
                "accepted_phase": phase,
                "phase": phase,
            }
            metadata["dialogue_plan"] = decision

        planning_policy["effective_action"] = action
        planning_policy["effective_phase"] = phase
        metadata["planning_policy"] = planning_policy
        metadata["intent_policy"] = planning_policy
        return (
            action,
            phase,
            decision,
            user_intent,
            intent_summary,
            stall_count,
            rejection_count,
        )

    def _apply_exploration_sufficiency_gate(
        self,
        *,
        action: str,
        phase: str,
        decision: dict,
        request: CoachingTurnRequest,
        prior_state: CoachingState,
        current_message: str,
        meal_base: str,
        interaction_state: str,
        user_intent: str,
    ) -> dict:
        """Prevent exploration from continuing after useful evidence exists.

        The gate is deliberately conservative. It only redirects an exploratory
        inquiry when the meal has a concrete anchor and the next question is
        likely to add burden rather than decision-relevant information.
        """
        if action not in {"inquire", "respond"}:
            return self._no_exploration_sufficiency_gate(action, phase, decision)
        if phase != "exploration" or prior_state.recommendation_history:
            return self._no_exploration_sufficiency_gate(action, phase, decision)
        if user_intent in {"disengaging", "rejecting"}:
            return self._no_exploration_sufficiency_gate(action, phase, decision)

        state = InteractionStateTracker.parse_formatted_state(interaction_state)
        has_anchor = self._has_assessable_meal_anchor(
            meal_base=meal_base,
            interaction_data=state,
            current_message=current_message,
        )
        suggestion_requested = self._message_requests_suggestion(current_message)
        if not has_anchor:
            if (
                suggestion_requested
                and self._profile_or_setup_can_seed_recommendation(request)
            ):
                instruction = self._assessment_followup_instruction(
                    followup_action="recommend",
                    active_issue=self._interaction_section_text(
                        interaction_state, "Active issue"
                    ),
                    single_default=True,
                )
                gated_decision = {
                    **decision,
                    "action": "assess",
                    "accepted_phase": "assessment",
                    "phase": "assessment",
                    "actionability": "workable",
                    "closure_readiness": "actionable",
                    "assessment_followup_action": "recommend",
                    "assessment_followup_phase": "recommendation",
                    "assessment_followup_instruction": instruction,
                    "instruction": (
                        "Assess the user's profile/setup constraints as sufficient "
                        "context for a first concrete suggestion. Do not ask a mood "
                        "or preference question before offering a useful default."
                    ),
                    "reasoning": (
                        "Exploration sufficiency gate redirected a direct suggestion "
                        "request to assessment because profile/setup facts can seed "
                        "a useful first recommendation."
                    ),
                }
                return {
                    "applied": True,
                    "action": "assess",
                    "phase": "assessment",
                    "decision": gated_decision,
                    "metadata": {
                        "gate": "profile_seeded_suggestion_to_assessment",
                        "has_meal_anchor": False,
                        "suggestion_requested": True,
                        "profile_seeded": True,
                    },
                }
            return self._no_exploration_sufficiency_gate(action, phase, decision)

        open_questions = state.get("open_questions") or []
        precision_only = bool(open_questions) and self._open_questions_are_precision_only(
            open_questions
        )
        burden_signaled = self._exploration_burden_or_boundary_signaled(
            current_message=current_message,
            interaction_data=state,
            prior_state=prior_state,
        )
        if action == "respond" and not (
            suggestion_requested
            or state.get("rejected_options")
            or state.get("unavailable_options")
            or self._explicit_exploration_boundary_in_latest_turn(
                current_message=current_message,
                interaction_data=state,
            )
        ):
            return self._no_exploration_sufficiency_gate(action, phase, decision)
        planner_says_workable = str(
            decision.get("actionability") or ""
        ).lower() in {"workable", "settled", "boundary", "conflicted"}

        if not (
            planner_says_workable
            or precision_only
            or burden_signaled
            or suggestion_requested
        ):
            return self._no_exploration_sufficiency_gate(action, phase, decision)

        critical_inquiry = self._planner_instruction_requests_critical_detail(
            decision=decision,
            interaction_data=state,
            current_message=current_message,
        )
        if critical_inquiry and not (suggestion_requested or burden_signaled):
            return self._no_exploration_sufficiency_gate(action, phase, decision)

        if suggestion_requested or burden_signaled:
            followup_action = "recommend"
            followup_phase = "recommendation"
        else:
            followup_action = str(decision.get("assessment_followup_action") or "")
            followup_phase = str(decision.get("assessment_followup_phase") or "")
        instruction = self._assessment_followup_instruction(
            followup_action=followup_action,
            active_issue=self._interaction_section_text(
                interaction_state, "Active issue"
            ),
        )
        gated_decision = {
            **decision,
            "action": "assess",
            "accepted_phase": "assessment",
            "phase": "assessment",
            "actionability": "workable",
            "closure_readiness": (
                "actionable"
                if suggestion_requested or planner_says_workable
                else "not_ready"
            ),
            "assessment_followup_action": followup_action,
            "assessment_followup_phase": followup_phase,
            "assessment_followup_instruction": instruction,
            "instruction": (
                "Assess the current meal using the available evidence. Do not "
                "ask another exploration question unless a genuinely critical "
                "fact is missing."
            ),
            "reasoning": (
                "Exploration sufficiency gate redirected inquiry to assessment "
                "because the current meal has an assessable anchor and further "
                "questioning appears non-critical."
            ),
        }
        return {
            "applied": True,
            "action": "assess",
            "phase": "assessment",
            "decision": gated_decision,
            "metadata": {
                "gate": "exploration_sufficiency_to_assessment",
                "has_meal_anchor": has_anchor,
                "precision_only_open_questions": precision_only,
                "burden_or_boundary_signaled": burden_signaled,
                "suggestion_requested": suggestion_requested,
                "planner_actionability": decision.get("actionability", ""),
                "open_question_count": len(open_questions),
                "critical_inquiry": critical_inquiry,
            },
        }

    @staticmethod
    def _no_exploration_sufficiency_gate(
        action: str,
        phase: str,
        decision: dict,
    ) -> dict:
        return {
            "applied": False,
            "action": action,
            "phase": phase,
            "decision": decision,
            "metadata": {},
        }

    @staticmethod
    def _profile_or_setup_can_seed_recommendation(
        request: CoachingTurnRequest,
    ) -> bool:
        profile = request.profile
        if profile is None:
            return bool(request.nutrition_goal and request.meal_type)
        return bool(
            request.nutrition_goal
            and request.meal_type
            and (
                profile.preferences
                or profile.allergies
                or profile.activity_level
                or profile.extra
                or profile.nutritional_goals
            )
        )

    def _apply_assessment_saturation_gate(
        self,
        *,
        action: str,
        phase: str,
        decision: dict,
        prior_state: CoachingState,
        current_message: str,
        interaction_state: str,
        user_intent: str,
        actionability: str,
        closure_readiness: str,
    ) -> dict:
        """Avoid reassessing the same plan after advice has already been given.

        Assessment is useful when new meal evidence appears. It becomes burdensome
        when the user is only asking to simplify, keep, or wrap up an already
        assessed recommendation cycle. This gate preserves assessment for new
        evidence while routing saturated turns to a direct answer, confirmation,
        or finalization.
        """
        if action != "assess":
            return self._no_assessment_saturation_gate(action, phase, decision)
        if str(actionability or "").lower() == "conflicted":
            return self._no_assessment_saturation_gate(action, phase, decision)

        has_prior_assessment_context = bool(prior_state.recommendation_history) or str(
            prior_state.phase or ""
        ).lower() in {"assessment", "recommendation", "confirmation"}
        if not has_prior_assessment_context:
            return self._no_assessment_saturation_gate(action, phase, decision)

        material_update = self._message_adds_material_meal_update(current_message)
        suggestion_requested = self._message_requests_suggestion(current_message)
        if material_update and not suggestion_requested:
            return self._no_assessment_saturation_gate(action, phase, decision)

        stop_boundary = self._message_sets_stop_boundary(
            current_message
        ) or self._latest_user_sets_stop_boundary(interaction_state)
        preservation_boundary = self._message_preserves_current_plan_boundary(
            current_message
        ) or self._latest_user_preserves_current_plan(interaction_state)
        defers_final_decision = self._message_defers_final_decision(current_message)
        simplification_request = self._message_requests_simplified_guidance(
            current_message
        )

        if stop_boundary:
            gated_decision = {
                **decision,
                "action": "close",
                "accepted_phase": "finalization",
                "phase": "finalization",
                "closure_readiness": "boundary_close",
                "instruction": (
                    "Close briefly. Acknowledge the user's chosen plan and do "
                    "not repeat assessment or introduce another refinement."
                ),
                "reasoning": (
                    "Assessment saturation gate closed because the user set a "
                    "stop boundary after prior assessment or recommendation."
                ),
            }
            return {
                "applied": True,
                "action": "close",
                "phase": "finalization",
                "decision": gated_decision,
                "metadata": {
                    "gate": "assessment_saturation_to_close",
                    "stop_boundary": True,
                    "preservation_boundary": preservation_boundary,
                    "defers_final_decision": defers_final_decision,
                    "simplification_request": simplification_request,
                },
            }

        if defers_final_decision:
            gated_decision = {
                **decision,
                "action": "close",
                "accepted_phase": "finalization",
                "phase": "finalization",
                "closure_readiness": "boundary_close",
                "instruction": (
                    "Close reflectively because the user is intentionally keeping "
                    "the final choice open. Preserve the useful decision rule or "
                    "base plan in one concise message. Do not ask another question, "
                    "do not reassess again, and do not force finalization of an "
                    "option the user has not chosen."
                ),
                "reasoning": (
                    "Assessment saturation gate closed because repeated assessment "
                    "would not help after the user explicitly deferred the final "
                    "meal decision."
                ),
            }
            return {
                "applied": True,
                "action": "close",
                "phase": "finalization",
                "decision": gated_decision,
                "metadata": {
                    "gate": "assessment_saturation_to_deferred_decision_close",
                    "stop_boundary": False,
                    "preservation_boundary": preservation_boundary,
                    "defers_final_decision": True,
                    "simplification_request": simplification_request,
                },
            }

        if preservation_boundary and str(actionability).lower() in {
            "workable",
            "settled",
            "boundary",
        }:
            gated_decision = {
                **decision,
                "action": "confirm",
                "accepted_phase": "confirmation",
                "phase": "confirmation",
                "closure_readiness": (
                    closure_readiness
                    if closure_readiness in {"actionable", "ready_to_close"}
                    else "actionable"
                ),
                "instruction": (
                    "Confirm the current plan as the user's chosen compromise. "
                    "Do not reassess the whole meal or introduce new options."
                ),
                "reasoning": (
                    "Assessment saturation gate converted repeated assessment "
                    "to confirmation because the user is preserving the current plan."
                ),
            }
            return {
                "applied": True,
                "action": "confirm",
                "phase": "confirmation",
                "decision": gated_decision,
                "metadata": {
                    "gate": "assessment_saturation_to_confirmation",
                    "stop_boundary": False,
                    "preservation_boundary": True,
                    "defers_final_decision": defers_final_decision,
                    "simplification_request": simplification_request,
                },
            }

        if simplification_request:
            gated_decision = {
                **decision,
                "action": "respond",
                "accepted_phase": "negotiation",
                "phase": "negotiation",
                "closure_readiness": (
                    closure_readiness
                    if closure_readiness in {"actionable", "ready_to_close"}
                    else "actionable"
                ),
                "instruction": (
                    "Answer directly using the existing assessment or previous "
                    "recommendation. Give the simplest next step in 1-2 sentences. "
                    "Do not run a new assessment, do not introduce a new bundle, "
                    "and do not ask for optional precision details."
                ),
                "reasoning": (
                    "Assessment saturation gate routed repeated assessment to a "
                    "direct answer because the user requested simplified guidance "
                    "without adding new meal evidence."
                ),
            }
            return {
                "applied": True,
                "action": "respond",
                "phase": "negotiation",
                "decision": gated_decision,
                "metadata": {
                    "gate": "assessment_saturation_to_response",
                    "stop_boundary": False,
                    "preservation_boundary": preservation_boundary,
                    "simplification_request": True,
                },
            }

        if suggestion_requested and prior_state.recommendation_history:
            gated_decision = {
                **decision,
                "action": "recommend",
                "accepted_phase": "recommendation",
                "phase": "recommendation",
                "closure_readiness": (
                    closure_readiness
                    if closure_readiness in {"actionable", "ready_to_close"}
                    else "actionable"
                ),
                "instruction": (
                    "Recommend directly for the user's latest requested "
                    "refinement using the existing assessment context. Do not "
                    "repeat assessment text, do not revisit settled meal slots, "
                    "and keep the recommendation within the active issue."
                ),
                "reasoning": (
                    "Assessment saturation gate skipped repeated assessment "
                    "because the user requested a refinement after a prior "
                    "assessment/recommendation cycle."
                ),
            }
            return {
                "applied": True,
                "action": "recommend",
                "phase": "recommendation",
                "decision": gated_decision,
                "metadata": {
                    "gate": "assessment_saturation_to_recommendation_refinement",
                    "material_update": True,
                    "suggestion_requested": True,
                },
            }

        return self._no_assessment_saturation_gate(action, phase, decision)

    @staticmethod
    def _no_assessment_saturation_gate(
        action: str,
        phase: str,
        decision: dict,
    ) -> dict:
        return {
            "applied": False,
            "action": action,
            "phase": phase,
            "decision": decision,
            "metadata": {},
        }

    @staticmethod
    def _message_adds_material_meal_update(text: str) -> bool:
        lowered = (text or "").lower()
        if not lowered.strip():
            return False
        update_patterns = (
            r"\bactually\b",
            r"\balso\b",
            r"\badd(?:ing)?\b",
            r"\binclude\b",
            r"\bplus\b",
            r"\binstead\b",
            r"\bswap\b",
            r"\breplace\b",
            r"\bchange\s+(it|that|this|the meal)\b",
        )
        return any(re.search(pattern, lowered) for pattern in update_patterns) and bool(
            re.search(
                r"\b(chicken|turkey|fish|salmon|tuna|shrimp|tofu|egg|omelet|"
                r"yogurt|pizza|sandwich|rice|noodle|pasta|fries|chips|salad|"
                r"fruit|vegetable|veggie|watermelon|broccoli|bread|sauce|soup|"
                r"dessert|brownie|cheesecake|cake|cookie|ice\s+cream|smoothie|"
                r"cereal|banana|oats|peanut\s+butter|milk|wrap|drink|soda|"
                r"side|meal)\b",
                lowered,
            )
        )

    @staticmethod
    def _message_preserves_current_plan_boundary(text: str) -> bool:
        lowered = (text or "").lower()
        patterns = (
            r"\bkeep\s+(it|this|that|the meal|the plan)\b",
            r"\bkeep\s+.*\bas[- ]?is\b",
            r"\bas[- ]?is\b",
            r"\bwith\s+just\s+that\b",
            r"\bjust\s+that\b",
            r"\bcount\s+it\b",
            r"\bsounds?\s+good\b",
            r"\bthat\s+works\s+for\s+me\b",
            r"\bthat\s+works\b",
            r"\bworks?\s+for\s+me\b",
            r"\bi['’]?\s?ll\s+(stick|keep|go)\s+with\b",
            r"\bstick\s+with\s+(it|this|that|the meal|the plan)\b",
            r"\bset\s+as\s+planned\b",
            r"\bsettled\b",
            r"\bthat['’]?s\s+accurate\b",
            r"\bnot\s+adding\b",
            r"\bdon['’]?t\s+want\s+to\s+add\b",
            r"\bno\s+more\s+(vegetables?|fruit|changes?|additions?)\b",
        )
        return any(re.search(pattern, lowered) for pattern in patterns)

    @staticmethod
    def _message_defers_final_decision(text: str) -> bool:
        lowered = (text or "").lower()
        patterns = (
            r"\bnot\s+ready\s+to\s+(lock|settle|finali[sz]e|choose|decide)\b",
            r"\bdon['’]?t\s+want\s+to\s+(lock|settle|finali[sz]e|choose|decide)\b",
            r"\bkeep\s+(the\s+)?(door|options?)\s+open\b",
            r"\bstill\s+(exploring|comparing|deciding)\b",
            r"\bkeep\s+(exploring|comparing)\b",
            r"\bbefore\s+(locking|settling|finali[sz]ing|choosing|deciding)\b",
        )
        return any(re.search(pattern, lowered) for pattern in patterns)

    @classmethod
    def _latest_user_preserves_current_plan(cls, interaction_state: str) -> bool:
        latest = cls._interaction_section_text(interaction_state, "Latest user position")
        return cls._message_preserves_current_plan_boundary(latest)

    @staticmethod
    def _message_requests_simplified_guidance(text: str) -> bool:
        lowered = (text or "").lower()
        patterns = (
            r"\bsimplest\b",
            r"\bsimple\b",
            r"\bquickest\b",
            r"\bleast\s+(prep|cleanup|effort)\b",
            r"\bjust\s+(assess|tell|give|count)\b",
            r"\bi\s+already\s+(listed|said|told)\b",
            r"\bas\s+is\b",
            r"\bas[- ]?is\b",
        )
        return any(re.search(pattern, lowered) for pattern in patterns)

    @classmethod
    def _has_assessable_meal_anchor(
        cls,
        *,
        meal_base: str,
        interaction_data: dict,
        current_message: str,
    ) -> bool:
        """Return true when there is enough concrete meal evidence to assess."""
        if cls._meal_base_has_food_items(meal_base):
            return True
        for key in (
            "accepted_options",
            "candidate_options",
            "meal_slots",
            "answered_facts",
            "safety_conflicted_options",
            "user_requested_conflicted_options",
        ):
            if any(
                cls._looks_like_meal_fact(item)
                and cls._looks_like_concrete_meal_anchor(item)
                for item in interaction_data.get(key, [])
            ):
                return True
        return cls._looks_like_meal_fact(
            current_message
        ) and cls._looks_like_concrete_meal_anchor(current_message)

    @staticmethod
    def _meal_base_has_food_items(meal_base: str) -> bool:
        text = meal_base or ""
        if not text.strip():
            return False
        for line in text.splitlines():
            stripped = line.strip()
            if not re.match(r"[-*]?\s*Food items\s*:", stripped, re.I):
                continue
            value = stripped.split(":", 1)[1].strip().lower()
            if not value:
                return False
            if re.search(r"\b(not yet mentioned|none|unknown|n/?a)\b", value):
                return False
            items = [
                item.strip(" .")
                for item in re.split(r",|;|\band\b", value)
                if item.strip(" .")
            ]
            return any(
                ConversationEngine._looks_like_concrete_meal_anchor(item)
                for item in items
            )
        return bool(re.search(r"\b(confirmed food items|ingredients)\s*:", text, re.I))

    @staticmethod
    def _looks_like_meal_fact(text: str) -> bool:
        lowered = (text or "").lower().strip()
        if not lowered:
            return False
        non_food_patterns = (
            r"\b(i\s+don['’]?t\s+know|idk|not\s+sure|maybe|haven['’]?t\s+decided)\b",
            r"\b(no|none|nothing)\b\s*$",
            r"\b(suggestion|recommendation|question|answer)\b",
            r"\b(allerg(?:y|ic)|sensitivit(?:y|ies)|health\s+concern|"
            r"profile\s+constraint|dietary\s+constraint)\b",
        )
        if any(re.search(pattern, lowered) for pattern in non_food_patterns):
            return False
        food_or_meal_terms = (
            r"\b(chicken|turkey|fish|salmon|tuna|shrimp|tofu|egg|omelet|yogurt|"
            r"pizza|sandwich|rice|noodle|jajang|pasta|steak|pork|beef|brie|"
            r"cheese|hummus|salad|vegetable|veggie|broccoli|spinach|fruit|"
            r"watermelon|apple|potato|pancake|granola|sauce|buffet)\b"
        )
        return bool(re.search(food_or_meal_terms, lowered))

    @staticmethod
    def _looks_like_concrete_meal_anchor(text: str) -> bool:
        """Distinguish concrete meal items from broad cuisine/category anchors."""
        lowered = (text or "").lower().strip(" .")
        if not lowered:
            return False
        vague_patterns = (
            r"^(french|italian|mexican|chinese|korean|japanese|thai|indian|"
            r"american|mediterranean)\s+(food|cuisine|restaurant|meal|dinner)?$",
            r"^(food|cuisine|restaurant|eating out|takeout|sit-down meal)$",
            r"^(something|anything|whatever)\b",
        )
        if any(re.search(pattern, lowered) for pattern in vague_patterns):
            return False
        broad_singletons = {
            "sandwich",
            "salad",
            "burger",
            "pizza",
            "noodles",
            "pasta",
            "rice",
            "soup",
        }
        if lowered in broad_singletons:
            return False
        return True

    @classmethod
    def _planner_instruction_requests_critical_detail(
        cls,
        *,
        decision: dict,
        interaction_data: dict,
        current_message: str,
    ) -> bool:
        """Keep exploration when the planner is asking a decision-critical detail."""
        evidence_parts = [
            str(decision.get("instruction") or ""),
            str(decision.get("reasoning") or ""),
            str(current_message or ""),
            *[str(q) for q in interaction_data.get("open_questions", [])],
            str(interaction_data.get("active_issue") or ""),
        ]
        evidence = "\n".join(evidence_parts).lower()
        critical_patterns = (
            r"\bwhat\s+kind\s+of\s+(chicken|meat|fish|protein|sandwich|dish|entrée|entree)\b",
            r"\bwhich\s+(dish|entrée|entree|protein|main|sandwich|restaurant)\b",
            r"\b(specific|actual)\s+(dish|entrée|entree|protein|meal|order)\b",
            r"\b(how|whether).{0,30}\b(prepared|cooked|fried|grilled|baked|breaded|air[- ]?fried)\b",
            r"\b(preparation|cooking method)\b",
            r"\b(definitely|actually).{0,25}\b(eating out|at home|ordering)\b",
        )
        if any(re.search(pattern, evidence) for pattern in critical_patterns):
            return True
        return False

    @staticmethod
    def _assessment_followup_instruction(
        *,
        followup_action: str,
        active_issue: str = "",
        single_default: bool = False,
    ) -> str:
        """Generate follow-up guidance that matches the follow-up action."""
        action = str(followup_action or "").lower()
        issue = str(active_issue or "").strip()
        if action == "recommend":
            scope = f" Address the active issue: {issue}." if issue else ""
            if single_default:
                return (
                    "Recommend one concrete default option grounded in the "
                    f"assessment.{scope} Return exactly one recommendation option "
                    "and do not present a list of alternatives or ask a follow-up "
                    "choice question."
                )
            return (
                "Recommend one concise default bundle grounded in the assessment."
                f"{scope} Use concrete bullet adjustments, not open-ended choice "
                "questions."
            )
        if action == "confirm":
            return (
                "Confirm the current meal plan concisely without introducing a "
                "new recommendation."
            )
        if action == "close":
            return "Close briefly based on the assessment."
        if action == "handoff":
            return "Ask the user to choose the next coaching direction."
        if action == "inquire":
            return "Ask one concise decision-critical question."
        return "Use the assessment to continue the coaching flow."

    @classmethod
    def _open_questions_are_precision_only(cls, questions: list[str]) -> bool:
        """Return true when open questions are useful but not gating."""
        cleaned = [str(q or "").strip().lower() for q in questions if str(q or "").strip()]
        if not cleaned:
            return True
        return all(cls._question_is_precision_only(question) for question in cleaned)

    @staticmethod
    def _question_is_precision_only(question: str) -> bool:
        critical_patterns = (
            r"\bwhat\s+(are|is)\s+you\b",
            r"\bwhat\s+food\b",
            r"\bwhich\s+(meal|dish|option|protein|main)\b",
            r"\bany\s+(food|option|protein|vegetable|fruit)\b",
            r"\bdo\s+you\s+have\s+any\b",
            r"\ballerg",
            r"\bsafe\b",
            r"\bavailable\b",
        )
        if any(re.search(pattern, question) for pattern in critical_patterns):
            return False
        precision_patterns = (
            r"\bhow\s+much\b",
            r"\bhow\s+many\b",
            r"\babout\s+how\b",
            r"\bportion\b",
            r"\bamount\b",
            r"\bexact",
            r"\broughly\b",
            r"\btablespoons?\b",
            r"\bteaspoons?\b",
            r"\bounces?\b",
            r"\bgrams?\b",
            r"\bcondiment\b",
            r"\bsauce\b",
            r"\bbrand\b",
            r"\bsize\b",
        )
        return any(re.search(pattern, question) for pattern in precision_patterns)

    @classmethod
    def _exploration_burden_or_boundary_signaled(
        cls,
        *,
        current_message: str,
        interaction_data: dict,
        prior_state: CoachingState,
    ) -> bool:
        latest = str(interaction_data.get("latest_user_position") or "")
        evidence = f"{current_message}\n{latest}".lower()
        boundary_patterns = (
            r"\bonly\b",
            r"\bno\s+other\b",
            r"\bnothing\s+(else|more)\b",
            r"\bthat['’]?s\s+(all|it)\b",
            r"\bi\s+already\b",
            r"\bi\s+told\s+you\b",
            r"\bjust\s+(give|tell|suggest|recommend)\b",
            r"\bidk\b",
            r"\bi\s+don['’]?t\s+know\b",
            r"\bnot\s+sure\b",
        )
        if any(re.search(pattern, evidence) for pattern in boundary_patterns):
            return True
        if prior_state.consecutive_qa_count >= 2:
            return True
        return bool(
            interaction_data.get("rejected_options")
            or interaction_data.get("unavailable_options")
            or interaction_data.get("accepted_options")
        )

    @staticmethod
    def _explicit_exploration_boundary_in_latest_turn(
        *,
        current_message: str,
        interaction_data: dict,
    ) -> bool:
        latest = str(interaction_data.get("latest_user_position") or "")
        evidence = f"{current_message}\n{latest}".lower()
        patterns = (
            r"\bonly\b",
            r"\bno\s+other\b",
            r"\bnothing\s+(else|more)\b",
            r"\bthat['’]?s\s+(all|it)\b",
            r"\bi\s+already\b",
            r"\bi\s+told\s+you\b",
            r"\bjust\s+(give|tell|suggest|recommend)\b",
            r"\bidk\b",
            r"\bi\s+don['’]?t\s+know\b",
            r"\bnot\s+sure\b",
        )
        return any(re.search(pattern, evidence) for pattern in patterns)

    @staticmethod
    def _message_requests_suggestion(text: str) -> bool:
        lowered = (text or "").lower()
        patterns = (
            r"\bsuggest",
            r"\brecommend",
            r"\bwhat\s+should\s+i\b",
            r"\bwhat\s+would\s+you\b",
            r"\bdo\s+you\s+have\s+(an?\s+)?(idea|suggestion|recommendation)\b",
            r"\bhelp\s+me\s+(choose|decide|pick)\b",
        )
        return any(re.search(pattern, lowered) for pattern in patterns)

    def _apply_commitment_action_gate(
        self,
        *,
        action: str,
        phase: str,
        decision: dict,
        request: CoachingTurnRequest,
        prior_state: CoachingState,
        current_message: str,
        meal_base: str,
        interaction_state: str,
        user_intent: str,
        actionability: str,
    ) -> dict:
        """Constrain planner actions using user commitments and safety boundaries."""
        counts = dict(prior_state.safety_clarification_counts or {})
        safety_conflict = self._detect_profile_constraint_conflict(
            request=request,
            current_message=current_message,
            meal_base=meal_base,
        )
        if safety_conflict is not None:
            key = safety_conflict["key"]
            prior_count = int(counts.get(key, 0))
            counts[key] = prior_count + 1
            if (
                prior_count > 0
                and self._is_safe_replacement_request(current_message)
                and user_intent not in {"disengaging"}
            ):
                gated_action = "assess"
                gated_phase = "assessment"
                instruction = (
                    "The conflicted item itself remains unsafe, but the user is "
                    "asking for a safe replacement or similar alternative. Assess "
                    "the current meal slots and recommend only safe alternatives "
                    "for that unresolved slot. Do not revisit already settled "
                    "meal components."
                )
                gated_decision = {
                    **decision,
                    "action": gated_action,
                    "accepted_phase": gated_phase,
                    "phase": gated_phase,
                    "actionability": "workable",
                    "closure_readiness": "not_ready",
                    "assessment_followup_action": "recommend",
                    "assessment_followup_phase": "recommendation",
                    "assessment_followup_instruction": instruction,
                    "instruction": instruction,
                    "reasoning": (
                        "Commitment gate localized a repeated safety conflict "
                        "to the unresolved replacement slot."
                    ),
                }
                return {
                    "applied": True,
                    "action": gated_action,
                    "phase": gated_phase,
                    "decision": gated_decision,
                    "metadata": {
                        "gate": "safety_conflict_to_replacement_slot",
                        "commitment_status": "conflicted_replacement",
                        **safety_conflict,
                        "prior_clarification_count": prior_count,
                    },
                    "safety_clarification_counts": counts,
                }
            if prior_count <= 0:
                gated_action = "respond"
                gated_phase = self._phase_for_safety_response(
                    current_phase=phase,
                    prior_state=prior_state,
                    prior_count=prior_count,
                )
                instruction = (
                    "Acknowledge the user's choice, state that it conflicts with "
                    f"the stored {safety_conflict['constraint_type']} constraint "
                    f"({safety_conflict['constraint']}), and ask one clarifying "
                    "question about a safe alternative or safe interpretation. Do "
                    "not recommend the conflicted item."
                )
                gate = "safety_conflict_clarification"
            else:
                user_explicitly_stops = (
                    user_intent == "disengaging"
                    or self._message_sets_stop_boundary(current_message)
                )
                should_close_conflict = user_explicitly_stops and (
                    action in {"close", "terminate"}
                    or actionability in {"boundary", "conflicted", "settled"}
                    or user_intent in {"rejecting", "disengaging"}
                )
                gated_action = "close" if should_close_conflict else "assess"
                gated_phase = "finalization" if should_close_conflict else "assessment"
                if should_close_conflict:
                    instruction = (
                        "Close reflectively. Acknowledge the user's choice and the "
                        f"stored {safety_conflict['constraint_type']} constraint "
                        f"({safety_conflict['constraint']}). Do not frame the "
                        "conflicted choice as an uncomplicated success and do not "
                        "introduce a new recommendation."
                    )
                    gate = "safety_conflict_reflective_close"
                else:
                    instruction = (
                        "Assess the current meal without endorsing the conflicted "
                        "item. State that the conflicted item remains outside the "
                        "safe recommendation space. If the user has explicitly "
                        "continued to request the conflicted item, do not repeat "
                        "a removal recommendation as the main advice; use a "
                        "cautious-continuation recommendation that respects the "
                        "user's stated choice without describing it as safe, then "
                        "continue planning around non-conflicted meal components. "
                        "Do not repeat the same safety clarification question."
                    )
                    gate = "safety_conflict_to_safe_plan"
            gated_decision = {
                **decision,
                "action": gated_action,
                "accepted_phase": gated_phase,
                "phase": gated_phase,
                "actionability": "conflicted",
                "closure_readiness": (
                    "boundary_close" if gated_action == "close" else "not_ready"
                ),
                "assessment_followup_action": (
                    decision.get("assessment_followup_action")
                    if gated_action == "close"
                    else "recommend"
                ),
                "assessment_followup_phase": (
                    decision.get("assessment_followup_phase")
                    if gated_action == "close"
                    else "recommendation"
                ),
                "assessment_followup_instruction": (
                    decision.get("assessment_followup_instruction")
                    if gated_action == "close"
                    else instruction
                ),
                "instruction": instruction,
                "reasoning": "Commitment gate detected a safety-relevant profile conflict.",
            }
            return {
                "applied": True,
                "action": gated_action,
                "phase": gated_phase,
                "decision": gated_decision,
                "metadata": {
                    "gate": gate,
                    "commitment_status": "conflicted",
                    **safety_conflict,
                    "prior_clarification_count": prior_count,
                },
                "safety_clarification_counts": counts,
            }

        has_accepted = self._interaction_section_has_items(
            interaction_state,
            "Accepted options",
        )
        has_boundary = self._interaction_section_has_items(
            interaction_state,
            "Rejected options",
        ) or self._interaction_section_has_items(
            interaction_state,
            "Unavailable options",
        )
        if (
            has_accepted
            and user_intent == "accepting"
            and self._is_current_commitment_statement(current_message)
            and action in ("inquire", "recommend")
        ):
            gated_action = "assess"
            gated_phase = "assessment"
            gated_decision = {
                **decision,
                "action": gated_action,
                "accepted_phase": gated_phase,
                "phase": gated_phase,
                "actionability": "settled" if actionability != "boundary" else "boundary",
                "closure_readiness": (
                    "boundary_close" if actionability == "boundary" else "ready_to_close"
                ),
                "assessment_followup_action": "confirm",
                "assessment_followup_phase": "confirmation",
                "assessment_followup_instruction": (
                    "Assess the accepted option, then confirm the current plan "
                    "without introducing a replacement recommendation unless a "
                    "safety conflict is found."
                ),
                "instruction": (
                    "Assess the user's accepted option as the current plan. Do not "
                    "ask another optional detail question or replace it with a new option."
                ),
                "reasoning": (
                    "Commitment gate treated the accepted option as sufficient for "
                    "assessment rather than continued inquiry or replacement recommendation."
                ),
            }
            return {
                "applied": True,
                "action": gated_action,
                "phase": gated_phase,
                "decision": gated_decision,
                "metadata": {
                    "gate": "accepted_commitment_to_assessment",
                    "commitment_status": "safe",
                    "has_accepted_options": True,
                    "has_boundary": has_boundary,
                    "user_intent": user_intent,
                },
                "safety_clarification_counts": None,
            }

        if has_boundary and action == "inquire":
            gated_action = "respond"
            gated_phase = self._phase_for_boundary_response(
                current_phase=phase,
                prior_state=prior_state,
            )
            gated_decision = {
                **decision,
                "action": gated_action,
                "accepted_phase": gated_phase,
                "phase": gated_phase,
                "actionability": "boundary",
                "closure_readiness": (
                    "boundary_close"
                    if user_intent in _RESISTANT_INTENTS
                    else decision.get("closure_readiness", "actionable")
                ),
                "instruction": (
                    "Acknowledge the user's boundary and continue only within the "
                    "remaining workable options. Do not ask another availability, "
                    "reconsideration, or rejected-option question."
                ),
                "reasoning": (
                    "Commitment gate avoided converting a user boundary into another inquiry."
                ),
            }
            return {
                "applied": True,
                "action": gated_action,
                "phase": gated_phase,
                "decision": gated_decision,
                "metadata": {
                    "gate": "boundary_inquiry_to_response",
                    "commitment_status": "boundary",
                    "has_accepted_options": has_accepted,
                    "has_boundary": True,
                },
                "safety_clarification_counts": None,
            }

        return {
            "applied": False,
            "action": action,
            "phase": phase,
            "decision": decision,
            "metadata": {},
            "safety_clarification_counts": None,
        }

    @staticmethod
    def _phase_for_safety_response(
        *,
        current_phase: str,
        prior_state: CoachingState,
        prior_count: int,
    ) -> str:
        """Place safety clarification in exploration unless it follows a proposal."""
        phase = str(current_phase or "").lower()
        has_prior_proposal = bool(prior_state.recommendation_history)
        if has_prior_proposal or prior_count > 0 or phase in {
            "recommendation",
            "negotiation",
            "confirmation",
        }:
            return "negotiation"
        return "exploration"

    @staticmethod
    def _phase_for_boundary_response(
        *,
        current_phase: str,
        prior_state: CoachingState,
    ) -> str:
        """Separate early feasibility clarification from recommendation feedback."""
        phase = str(current_phase or "").lower()
        if prior_state.recommendation_history or phase in {
            "recommendation",
            "negotiation",
            "confirmation",
        }:
            return "negotiation"
        return "exploration"

    @staticmethod
    def _normalize_effective_phase(
        *,
        action: str,
        phase: str,
        prior_state: CoachingState,
        user_intent: str,
        actionability: str,
    ) -> str:
        """Keep phase as dialogue stage rather than an action synonym."""
        action = str(action or "").lower()
        phase = str(phase or "").lower()
        user_intent = str(user_intent or "").lower()
        actionability = str(actionability or "").lower()
        has_prior_proposal = bool(prior_state.recommendation_history)

        if action == "assess":
            return "assessment"
        if action == "confirm":
            return "confirmation"
        if action == "handoff":
            return "negotiation"
        if action in {"close", "terminate"}:
            return "finalization"
        if action == "recommend":
            return "recommendation"
        if action == "respond":
            if has_prior_proposal and user_intent in {
                "accepting",
                "inquiring",
                "rejecting",
                "disengaging",
                "deferring",
            }:
                return "negotiation"
            if phase in {"recommendation", "negotiation", "confirmation"}:
                return "negotiation" if has_prior_proposal else "exploration"
            return "exploration"
        if action == "inquire":
            if has_prior_proposal or phase in {"recommendation", "negotiation"}:
                return "negotiation"
            return "exploration"
        if phase in {
            "exploration",
            "assessment",
            "recommendation",
            "negotiation",
            "confirmation",
            "finalization",
        }:
            return phase
        return "exploration"

    @classmethod
    def _detect_profile_constraint_conflict(
        cls,
        *,
        request: CoachingTurnRequest,
        current_message: str,
        meal_base: str,
    ) -> dict | None:
        profile = request.profile
        if profile is None:
            return None
        evidence = f"{current_message}\n{meal_base}".lower()
        constraints: list[tuple[str, str]] = []
        constraints.extend(("allergy", str(item)) for item in profile.allergies or ())
        for constraint_type, raw_constraint in constraints:
            constraint = raw_constraint.strip()
            if not constraint:
                continue
            if cls._latest_message_resolves_constraint(
                current_message,
                constraint,
            ):
                continue
            if cls._mentions_constraint(evidence, constraint):
                safe_key = re.sub(r"[^a-z0-9]+", "_", constraint.lower()).strip("_")
                return {
                    "key": f"{constraint_type}:{safe_key}",
                    "constraint_type": constraint_type,
                    "constraint": constraint,
                }
        return None

    @classmethod
    def _latest_message_resolves_constraint(cls, text: str, constraint: str) -> bool:
        """Detect latest-turn language that turns a conflicted item into a safe slot.

        Historical meal memory may preserve unsafe options to avoid forgetting why
        they are unsafe. That memory should not keep the whole dialogue in a
        conflicted state after the user explicitly chooses an allergy-free or
        otherwise constraint-compatible substitute.
        """
        lowered = (text or "").lower()
        item = constraint.lower().strip()
        if not item:
            return False

        terms = (item, *cls._constraint_aliases(item))
        explicit_safe_patterns = [
            rf"\b{re.escape(item)}[-\s]*free\b",
            rf"\bwithout\s+{re.escape(item)}s?\b",
            rf"\bno\s+{re.escape(item)}s?\b",
            rf"\bfree\s+of\s+{re.escape(item)}s?\b",
            rf"\b(skip|avoid|remove|leave\s+out)\s+{re.escape(item)}s?\b",
        ]
        has_safe_marker = any(
            re.search(pattern, lowered)
            for pattern in explicit_safe_patterns
        )
        if item in {"egg", "eggs"}:
            has_safe_marker = has_safe_marker or bool(
                re.search(
                    r"\b(tofu|egg[-\s]*free)\b.{0,24}\b(omelet|omelette|scramble|scrambled)\b",
                    lowered,
                )
                or re.search(
                    r"\b(omelet|omelette|scramble|scrambled)[-\s]*style\b",
                    lowered,
                )
                and re.search(r"\b(without\s+eggs?|egg[-\s]*free)\b", lowered)
            )
        if not has_safe_marker:
            return False

        unsafe_terms = terms
        if item in {"egg", "eggs"}:
            unsafe_terms = tuple(
                term for term in terms
                if term not in {"omelet", "omelette", "scramble", "scrambled"}
            )
        unsafe_request_patterns = []
        for term in unsafe_terms:
            escaped = re.escape(term)
            unsafe_request_patterns.extend(
                [
                    rf"\b(want|wants|wanted|have|having|eat|eating|use|using|add|adding|order|ordering)\s+(an?\s+|some\s+)?{escaped}\b",
                    rf"\b{escaped}\s+(fried|based|filled|added)\b",
                ]
            )
        return not any(re.search(pattern, lowered) for pattern in unsafe_request_patterns)

    @staticmethod
    def _mentions_constraint(text: str, constraint: str) -> bool:
        lowered = text.lower()
        item = constraint.lower().strip()
        if not item:
            return False
        terms = [item, *ConversationEngine._constraint_aliases(item)]
        for term in terms:
            escaped = re.escape(term)
            if ConversationEngine._constraint_term_is_safe_substitute_context(
                lowered,
                constraint=item,
                term=term,
            ):
                continue
            negated_patterns = (
                rf"\bno\s+{escaped}\b",
                rf"\bwithout\s+{escaped}\b",
                rf"\b{escaped}[-\s]*free\b",
                rf"\bfree\s+of\s+{escaped}\b",
            )
            if any(re.search(pattern, lowered) for pattern in negated_patterns):
                continue
            if re.search(rf"\b{escaped}\b", lowered):
                return True
        return False

    @staticmethod
    def _constraint_term_is_safe_substitute_context(
        text: str,
        *,
        constraint: str,
        term: str,
    ) -> bool:
        """Ignore allergen surfaces used inside explicit safe-substitute phrases."""
        item = constraint.lower().strip()
        surface = term.lower().strip()
        if item in {"milk", "dairy"} and surface in {"milk", "dairy"}:
            safe_milk_patterns = (
                r"\b(soy|almond|oat|coconut|rice|pea|cashew|hemp)\s+milk\b",
                r"\b(non[-\s]?dairy|dairy[-\s]?free|plant[-\s]?based)\s+milk\b",
                r"\bmilk[-\s]?free\b",
                r"\bdairy[-\s]?free\b.{0,40}\b(smoothie|milk|yogurt|breakfast|version)\b",
                r"\b(smoothie|breakfast|version)\b.{0,40}\bdairy[-\s]?free\b",
            )
            if any(re.search(pattern, text) for pattern in safe_milk_patterns):
                return True
        if item in {"milk", "dairy"} and surface == "butter":
            nut_butter_patterns = (
                r"\b(peanut|almond|cashew|sunflower|seed|nut)\s+butter\b",
                r"\bnut[-\s]?butter\b",
            )
            if any(re.search(pattern, text) for pattern in nut_butter_patterns):
                return True
        return False

    @staticmethod
    def _is_safe_replacement_request(text: str) -> bool:
        """Detect requests to solve a conflicted slot with a safe alternative."""
        lowered = text.lower()
        patterns = (
            r"\balternative\b",
            r"\breplacement\b",
            r"\bsimilar\b",
            r"\bswap\b",
            r"\binstead\b",
            r"\bwhere\s+is\b",
            r"\bwhat\s+about\b",
        )
        return any(re.search(pattern, lowered) for pattern in patterns)

    @staticmethod
    def _constraint_aliases(constraint: str) -> tuple[str, ...]:
        """Return common food-surface forms for profile constraint matching."""
        item = constraint.lower().strip()
        alias_map = {
            "egg": (
                "eggs",
                "omelet",
                "omelette",
                "scrambled eggs",
                "egg white",
                "egg whites",
                "hard-boiled egg",
                "hard boiled egg",
            ),
            "eggs": (
                "egg",
                "omelet",
                "omelette",
                "scrambled eggs",
                "egg white",
                "egg whites",
                "hard-boiled egg",
                "hard boiled egg",
            ),
            "dairy": (
                "milk",
                "cheese",
                "butter",
                "yogurt",
                "yoghurt",
                "cream",
                "brie",
            ),
            "milk": ("dairy", "cheese", "butter", "yogurt", "cream"),
            "cheese": ("dairy", "brie", "cheddar", "mozzarella"),
            "nuts": ("peanut", "peanuts", "tree nuts", "almond", "walnut", "cashew"),
            "tree nuts": ("almond", "walnut", "cashew", "pecan", "pistachio"),
            "shellfish": ("shrimp", "crab", "lobster", "scallop", "clams", "mussels"),
        }
        return alias_map.get(item, ())

    @staticmethod
    def _is_current_commitment_statement(text: str) -> bool:
        """Return whether the latest user turn sounds like a settled choice."""
        lowered = text.lower().strip()
        if not lowered:
            return False
        option_list_patterns = (
            r"\b(i|we)\s+(can|could|might|may)\b.+\bor\b",
            r"\b(options?|choices?)\b.+\bor\b",
            r"\bavailable\b.+\bor\b",
        )
        if any(re.search(pattern, lowered) for pattern in option_list_patterns):
            return False
        commitment_patterns = (
            r"\byes\b",
            r"\bsounds?\s+good\b",
            r"\bworks?\s+for\s+me\b",
            r"\bthat\s+works\b",
            r"\blet'?s\s+(go|do|finish|end)\b",
            r"\bgo\s+with\b",
            r"\bready\s+to\s+finish\b",
            r"\bi\s+(want|will)\b",
            r"\bi['’]?ll\b",
            r"\bi\s+can\s+do\b",
            r"\bdoable\b",
            r"\beasiest\b",
        )
        return any(re.search(pattern, lowered) for pattern in commitment_patterns)

    @staticmethod
    def _interaction_section_has_items(text: str, section: str) -> bool:
        if not text:
            return False
        lines = text.splitlines()
        target = section.strip().lower().rstrip(":")
        in_section = False
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.endswith(":"):
                current = stripped.lower().rstrip(":")
                if in_section and current != target:
                    return False
                in_section = current == target
                continue
            if in_section and stripped.startswith("-"):
                return True
        return False

    @staticmethod
    def _interaction_section_text(text: str, section: str) -> str:
        if not text:
            return ""
        lines = text.splitlines()
        target = section.strip().lower().rstrip(":")
        in_section = False
        values: list[str] = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.endswith(":"):
                current = stripped.lower().rstrip(":")
                if in_section and current != target:
                    break
                in_section = current == target
                continue
            if in_section and stripped.startswith("-"):
                values.append(stripped.lstrip("- ").strip())
        return "; ".join(v for v in values if v)

    @classmethod
    def _latest_user_sets_stop_boundary(cls, interaction_state: str) -> bool:
        """Detect a general user boundary against more coaching turns."""
        latest_position = cls._interaction_section_text(
            interaction_state,
            "Latest user position",
        )
        if not latest_position:
            return False
        lowered = latest_position.lower()
        boundary_patterns = (
            r"\bno\s+more\b",
            r"\bnothing\s+(else|more)\b",
            r"\bno\s+changes?\b",
            r"\bdon['’]?t\s+want\s+(anything|any)\s+more\b",
            r"\bi['’]?\s?m\s+done\b",
            r"\bthat['’]?s\s+(all|it)\b",
            r"\bleave\s+it\b",
            r"\bstop\b",
            r"\bwrap\s+up\b",
            r"\bwrap\s+it\s+(there|here)\b",
            r"\bi['’]?\s?m\s+(set|good)\b",
            r"\bthat['’]?s\s+(settled|accurate)\b",
        )
        return any(re.search(pattern, lowered) for pattern in boundary_patterns)

    @staticmethod
    def _message_sets_stop_boundary(text: str) -> bool:
        """Detect explicit requests to stop coaching in the latest user message."""
        lowered = (text or "").lower()
        stop_patterns = (
            r"\bno\s+more\s+(suggestions?|questions?|advice|changes?)\b",
            r"\bnothing\s+(else|more)\b",
            r"\bno\s+changes?\b",
            r"\bdon['’]?t\s+want\s+(anything|any)\s+more\b",
            r"\bi['’]?\s?m\s+done\b",
            r"\bthat['’]?s\s+(all|it)\b",
            r"\bleave\s+it\b",
            r"\bstop\s+(asking|suggesting|coaching|now)?\b",
            r"\bwrap\s+up\b",
            r"\bwrap\s+it\s+(there|here)\b",
            r"\bi['’]?\s?m\s+(set|good)\b",
            r"\bthat['’]?s\s+(settled|accurate)\b",
            r"\bhappy\s+to\s+wrap\s+it\s+(there|here)\b",
        )
        return any(re.search(pattern, lowered) for pattern in stop_patterns)

    @classmethod
    def _confirmation_satisfied(
        cls,
        *,
        prior_state: CoachingState,
        interaction_state: str,
        user_intent: str,
        actionability: str,
    ) -> bool:
        """Return true when a prior confirmation has been answered affirmatively."""
        if str(prior_state.phase or "").lower() != "confirmation":
            return False
        if user_intent not in {"accepting", "disengaging"}:
            return False
        if actionability not in {"workable", "settled", "boundary"}:
            return False
        if cls._interaction_section_has_items(interaction_state, "Open questions"):
            return False
        active_issue = cls._interaction_section_text(interaction_state, "Active issue")
        return not active_issue or cls._active_issue_is_confirmation_scoped(active_issue)

    @staticmethod
    def _active_issue_is_confirmation_scoped(active_issue: str) -> bool:
        """Distinguish closure bookkeeping from genuinely unresolved topics."""
        lowered = active_issue.lower()
        closure_patterns = (
            r"\bcurrent commitment\b",
            r"\bavoid repeating settled choices\b",
            r"\bconfirm\b",
            r"\bconfirmation\b",
            r"\bfollow through\b",
            r"\bfully set\b",
            r"\bready to close\b",
        )
        return any(re.search(pattern, lowered) for pattern in closure_patterns)

    def _generate_action_replies(
        self,
        *,
        action: str,
        phase: str,
        decision: dict,
        request: CoachingTurnRequest,
        history,
        prior_state: CoachingState,
        turn_idx: int,
        alignment_score: float | None,
        alignment_reasoning: str | None,
        user_intent: str,
        coach: InformationSeeker,
        recommender: MealRecommender,
        meal_assessor: MealAssessor,
        response_generator: ResponseGenerator,
        metadata: dict,
    ) -> tuple[list[AssistantReply], str, str, str | None]:
        assistant_messages: list[AssistantReply] = []
        status = "active"
        terminated_by = None

        if action == "terminate":
            text = response_generator.fallback_closing_text(decision.get("instruction", ""))
            assistant_messages.append(AssistantReply(text, kind="closing"))
            return assistant_messages, phase, "terminated", "dialogue_planner"

        if action == "close":
            finalization_style = self._infer_finalization_style(
                decision=decision,
                user_intent=user_intent,
                interaction_state=history.interaction_state,
                metadata=metadata,
            )
            reply, _assessment = self._generate_close_reply(
                response_generator=response_generator,
                history=history,
                instruction=decision.get("instruction", ""),
                alignment_score=alignment_score,
                recommendation_history=list(prior_state.recommendation_history),
                finalization_style=finalization_style,
            )
            assistant_messages.append(reply)
            return assistant_messages, "finalization", "terminated", "close"

        if action == "confirm":
            assistant_messages.append(
                self._generate_confirmation_reply(
                    response_generator=response_generator,
                    history=history,
                )
            )
            return assistant_messages, "confirmation", status, terminated_by

        if action == "handoff":
            assistant_messages.append(
                self._generate_handoff_reply(
                    response_generator=response_generator,
                    history=history,
                    instruction=decision.get("instruction", ""),
                )
            )
            return assistant_messages, "negotiation", status, terminated_by

        if action == "recommend":
            if (
                metadata.get("planning_policy", {}).get("assessment_prerequisite")
                == "satisfied_by_prior_cycle"
            ):
                reply, rec = self._generate_recommendation_reply(
                    recommender=recommender,
                    response_generator=response_generator,
                    history=history,
                    prior_state=prior_state,
                    alignment_score=alignment_score,
                    alignment_reasoning=alignment_reasoning,
                    instruction=decision.get("instruction", ""),
                    turn_idx=turn_idx,
                    interaction_state=history.interaction_state,
                    assessment=None,
                    metadata=metadata,
                )
                assistant_messages.append(reply)
                metadata["recommendation_result"] = rec
                return assistant_messages, "recommendation", status, terminated_by

            grounded_decision = {
                **decision,
                "action": "assess",
                "accepted_phase": "assessment",
                "phase": "assessment",
                "assessment_followup_action": "recommend",
                "assessment_followup_phase": "recommendation",
                "assessment_followup_instruction": (
                    decision.get("assessment_followup_instruction")
                    or decision.get("instruction")
                    or "Recommend one concrete change grounded in the assessment."
                ),
            }
            self._generate_assessment_and_follow_up(
                assistant_messages=assistant_messages,
                request=request,
                history=history,
                prior_state=prior_state,
                turn_idx=turn_idx,
                alignment_score=alignment_score,
                alignment_reasoning=alignment_reasoning,
                decision=grounded_decision,
                user_intent=user_intent,
                coach=coach,
                recommender=recommender,
                meal_assessor=meal_assessor,
                response_generator=response_generator,
                metadata=metadata,
            )
            return assistant_messages, "recommendation", status, terminated_by

        if action == "assess":
            self._generate_assessment_and_follow_up(
                assistant_messages=assistant_messages,
                request=request,
                history=history,
                prior_state=prior_state,
                turn_idx=turn_idx,
                alignment_score=alignment_score,
                alignment_reasoning=alignment_reasoning,
                decision=decision,
                user_intent=user_intent,
                coach=coach,
                recommender=recommender,
                meal_assessor=meal_assessor,
                response_generator=response_generator,
                metadata=metadata,
            )
            post_decision = metadata["post_assessment_decision"]
            post_action = post_decision.get("action", "inquire")
            phase = post_decision.get("accepted_phase") or (
                "finalization"
                if post_action in ("close", "terminate")
                else "confirmation"
                if post_action in ("confirm", "handoff")
                else "recommendation"
            )
            if post_action == "terminate":
                status = "terminated"
                terminated_by = "post_assessment"
            elif post_action == "close":
                phase = "finalization"
                status = "terminated"
                terminated_by = "close"
            elif post_action == "confirm":
                phase = "confirmation"
            elif post_action == "handoff":
                phase = "negotiation"
            elif post_action == "recommend":
                phase = "recommendation"
            return assistant_messages, phase, status, terminated_by

        if action == "respond":
            assistant_messages.append(
                self._generate_answer_reply(
                    response_generator=response_generator,
                    history=history,
                    instruction=decision.get("instruction", ""),
                )
            )
            return assistant_messages, phase, status, terminated_by

        reply, template = self._generate_question_reply(
            coach=coach,
            response_generator=response_generator,
            history=history,
            phase=phase,
            prior_state=prior_state,
            instruction=decision.get("instruction", ""),
            interaction_state=history.interaction_state,
        )
        assistant_messages.append(reply)
        if phase not in ("exploration", "recommendation", "negotiation"):
            phase = "exploration"
        metadata["question_result"] = template
        return assistant_messages, phase, status, terminated_by

    def _generate_assessment_and_follow_up(
        self,
        *,
        assistant_messages: list[AssistantReply],
        request: CoachingTurnRequest,
        history,
        prior_state: CoachingState,
        turn_idx: int,
        alignment_score: float | None,
        alignment_reasoning: str | None,
        decision: dict,
        user_intent: str,
        coach: InformationSeeker,
        recommender: MealRecommender,
        meal_assessor: MealAssessor,
        response_generator: ResponseGenerator,
        metadata: dict,
    ) -> None:
        assessment = self._run_meal_assessment(
            meal_assessor=meal_assessor,
            history=history,
            prior_state=prior_state,
            alignment_score=alignment_score,
            alignment_reasoning=alignment_reasoning,
            metadata=metadata,
        )
        assistant_messages.append(
            self._generate_assessment_reply(
                response_generator=response_generator,
                assessment=assessment,
                history=history,
                recommendation_history=list(prior_state.recommendation_history),
            )
        )

        post_decision = self._post_assessment_decision_from_plan(
            decision=decision,
            assessment=assessment,
            alignment_score=alignment_score,
        )
        post_decision = self._gate_post_assessment_decision(
            post_decision=post_decision,
            initial_decision=decision,
            prior_state=prior_state,
            interaction_state=history.interaction_state,
            user_intent=user_intent,
            metadata=metadata,
        )
        metadata["post_assessment_decision"] = post_decision
        post_action = post_decision.get("action", "inquire")
        phase = post_decision.get("accepted_phase") or (
            "finalization"
            if post_action in ("close", "terminate")
            else "negotiation"
            if post_action == "handoff"
            else "confirmation"
            if post_action == "confirm"
            else "recommendation"
        )

        if post_action == "close":
            finalization_style = self._infer_finalization_style(
                decision=post_decision,
                user_intent=user_intent,
                interaction_state=history.interaction_state,
                metadata=metadata,
            )
            reply, _assessment = self._generate_close_reply(
                response_generator=response_generator,
                history=history,
                instruction=post_decision.get("instruction", ""),
                alignment_score=alignment_score,
                assessment=assessment,
                recommendation_history=list(prior_state.recommendation_history),
                finalization_style=finalization_style,
            )
            assistant_messages.append(reply)
        elif post_action == "confirm":
            assistant_messages.append(
                self._generate_confirmation_reply(
                    response_generator=response_generator,
                    history=history,
                )
            )
        elif post_action == "recommend":
            reply, rec = self._generate_recommendation_reply(
                recommender=recommender,
                response_generator=response_generator,
                history=history,
                prior_state=prior_state,
                alignment_score=alignment_score,
                alignment_reasoning=alignment_reasoning,
                instruction=post_decision.get("instruction", ""),
                turn_idx=turn_idx,
                interaction_state=history.interaction_state,
                assessment=assessment,
                metadata=metadata,
            )
            assistant_messages.append(reply)
            metadata["recommendation_result"] = rec
        elif post_action == "inquire":
            reply, template = self._generate_question_reply(
                coach=coach,
                response_generator=response_generator,
                history=history,
                phase=phase,
                prior_state=prior_state,
                instruction=post_decision.get("instruction", ""),
                interaction_state=history.interaction_state,
            )
            assistant_messages.append(reply)
            metadata["question_result"] = template
        elif post_action == "handoff":
            assistant_messages.append(
                self._generate_handoff_reply(
                    response_generator=response_generator,
                    history=history,
                    instruction=post_decision.get("instruction", ""),
                )
            )

    @staticmethod
    def _post_assessment_decision_from_plan(
        *,
        decision: dict,
        assessment: dict,
        alignment_score: float | None,
    ) -> dict:
        """Use the initial dialogue plan instead of a second planner call."""
        action = str(decision.get("assessment_followup_action") or "").lower()
        actionability = str(decision.get("actionability") or "insufficient").lower()
        closure_readiness = str(
            decision.get("closure_readiness") or "not_ready"
        ).lower()
        aligned = assessment.get("overall") == "aligned" or (
            alignment_score is not None and alignment_score >= 0.8
        )
        if action not in (
            "inquire",
            "recommend",
            "confirm",
            "handoff",
            "close",
            "terminate",
        ):
            if closure_readiness == "boundary_close":
                action = "close"
            elif aligned and closure_readiness in ("actionable", "ready_to_close"):
                action = "confirm"
            elif aligned and actionability in ("settled", "boundary"):
                action = "close" if actionability == "boundary" else "confirm"
            elif aligned:
                action = "inquire"
            else:
                action = "recommend"
        phase = str(decision.get("assessment_followup_phase") or "").lower()
        if phase == "motivational_ending":
            phase = "finalization"
        if phase not in ("exploration", "recommendation", "confirmation", "finalization"):
            phase = (
                "finalization"
                if action in ("close", "terminate")
                else "negotiation"
                if action == "handoff"
                else "confirmation"
                if action == "confirm"
                else "exploration"
                if action == "inquire"
                else "recommendation"
            )
        instruction = str(decision.get("assessment_followup_instruction") or "")
        if not instruction:
            instruction = (
                "Close warmly based on the assessment."
                if action in ("close", "terminate")
                else "Ask the user to choose the next coaching direction."
                if action == "handoff"
                else "Confirm the current meal plan without adding new recommendations."
                if action == "confirm"
                else "Use the assessment to continue the coaching flow."
            )
        return {
            "action": action,
            "accepted_phase": phase,
            "reasoning": (
                "Follow-up selected by the initial dialogue plan after "
                "assessment generation."
            ),
            "instruction": instruction,
        }

    @classmethod
    def _gate_post_assessment_decision(
        cls,
        *,
        post_decision: dict,
        initial_decision: dict,
        prior_state: CoachingState,
        interaction_state: str,
        user_intent: str,
        metadata: dict,
    ) -> dict:
        """Apply the same action-space constraints after assessment."""
        post_action = post_decision.get("action")
        planning_policy = metadata.get("planning_policy", {})
        actionability = str(
            initial_decision.get("actionability")
            or planning_policy.get("actionability")
            or "insufficient"
        ).lower()
        closure_readiness = str(
            initial_decision.get("closure_readiness")
            or planning_policy.get("closure_readiness")
            or "not_ready"
        ).lower()
        has_accepted = cls._interaction_section_has_items(
            interaction_state,
            "Accepted options",
        )
        has_open_questions = cls._interaction_section_has_items(
            interaction_state,
            "Open questions",
        )
        stop_boundary = cls._latest_user_sets_stop_boundary(interaction_state)
        confirmation_satisfied = cls._confirmation_satisfied(
            prior_state=prior_state,
            interaction_state=interaction_state,
            user_intent=user_intent,
            actionability=actionability,
        )

        if post_action == "close":
            active_issue = cls._interaction_section_text(
                interaction_state,
                "Active issue",
            )
            if (
                (has_open_questions or active_issue)
                and user_intent not in {"accepting", "rejecting", "disengaging"}
                and actionability == "workable"
                and closure_readiness != "boundary_close"
            ):
                metadata["post_assessment_gate"] = {
                    "gate": "post_assessment_open_question_close_deferred",
                    "from_action": "close",
                    "to_action": "recommend" if active_issue else "inquire",
                    "user_intent": user_intent,
                    "actionability": actionability,
                    "closure_readiness": closure_readiness,
                    "active_issue": active_issue,
                }
                return {
                    **post_decision,
                    "action": "recommend" if active_issue else "inquire",
                    "accepted_phase": "recommendation" if active_issue else "exploration",
                    "instruction": (
                        (
                            "Address the active unresolved issue before confirming "
                            f"the whole meal: {active_issue}. Keep the response "
                            "within that issue and do not revisit settled slots."
                        )
                        if active_issue
                        else (
                            "Ask one concise question only about the open issue most "
                            "likely to change the final guidance. Do not introduce a "
                            "new recommendation."
                        )
                    ),
                    "reasoning": (
                        "Post-assessment gate deferred closure because relevant "
                        "open issues remain and the current turn is not a user "
                        "commitment."
                    ),
                }
            if (
                closure_readiness != "boundary_close"
                and user_intent != "disengaging"
                and not stop_boundary
                and not confirmation_satisfied
            ):
                metadata["post_assessment_gate"] = {
                    "gate": "post_assessment_close_to_confirmation",
                    "from_action": "close",
                    "to_action": "confirm",
                    "user_intent": user_intent,
                    "actionability": actionability,
                    "closure_readiness": closure_readiness,
                }
                return {
                    **post_decision,
                    "action": "confirm",
                    "accepted_phase": "confirmation",
                    "instruction": (
                        "Confirm the current meal plan before finalization. "
                        "Do not introduce new recommendations."
                    ),
                    "reasoning": (
                        "Post-assessment gate inserted confirmation before "
                        "finalization."
                    ),
                }
            return post_decision

        if post_action != "recommend":
            if post_action == "confirm":
                if (
                    (stop_boundary or confirmation_satisfied)
                    and actionability in {"workable", "settled", "boundary"}
                ):
                    metadata["post_assessment_gate"] = {
                        "gate": (
                            "post_assessment_stop_boundary_confirm_to_close"
                            if stop_boundary
                            else "post_assessment_confirmed_plan_closed"
                        ),
                        "from_action": "confirm",
                        "to_action": "close",
                        "user_intent": user_intent,
                        "actionability": actionability,
                        "closure_readiness": closure_readiness,
                    }
                    return {
                        **post_decision,
                        "action": "close",
                        "accepted_phase": "finalization",
                        "instruction": (
                            "Close briefly because the user has indicated they "
                            "do not want additional changes or questions."
                        ),
                        "reasoning": (
                            "Post-assessment gate converted confirmation to "
                            "finalization after the latest user boundary."
                        ),
                    }
                active_issue = cls._interaction_section_text(
                    interaction_state,
                    "Active issue",
                )
                commitment_confirmation_issue = (
                    planning_policy.get("override")
                    == "redundant_recommendation_redirected"
                    and has_accepted
                    and not has_open_questions
                    and "commitment" in active_issue.lower()
                )
                if (
                    active_issue
                    and not commitment_confirmation_issue
                    and user_intent not in {"accepting", "disengaging"}
                    and actionability != "settled"
                ):
                    metadata["post_assessment_gate"] = {
                        "gate": "post_assessment_active_issue_confirmation_deferred",
                        "from_action": "confirm",
                        "to_action": "recommend",
                        "active_issue": active_issue,
                        "user_intent": user_intent,
                        "actionability": actionability,
                    }
                    return {
                        **post_decision,
                        "action": "recommend",
                        "accepted_phase": "recommendation",
                        "instruction": (
                            "Resolve the active issue before confirming the whole "
                            f"meal: {active_issue}. Do not revisit settled meal slots."
                        ),
                        "reasoning": (
                            "Post-assessment gate deferred confirmation because "
                            "an active unresolved issue remains."
                        ),
                    }
            return post_decision

        planning_override = planning_policy.get("override")
        active_issue = cls._interaction_section_text(
            interaction_state,
            "Active issue",
        )
        has_unresolved_issue = bool(has_open_questions or active_issue)
        has_rejected = cls._interaction_section_has_items(
            interaction_state,
            "Rejected options",
        )
        preservation_boundary = cls._latest_user_preserves_current_plan(
            interaction_state
        )

        if (
            (stop_boundary or preservation_boundary)
            and has_accepted
            and actionability in {"workable", "settled", "boundary"}
            and (user_intent in {"accepting", "rejecting", "disengaging"} or has_rejected)
        ):
            gate = (
                "post_assessment_preserved_plan_closed"
                if stop_boundary
                else "post_assessment_preserved_plan_confirmed"
            )
            to_action = "close" if stop_boundary else "confirm"
            metadata["post_assessment_gate"] = {
                "gate": gate,
                "from_action": "recommend",
                "to_action": to_action,
                "has_accepted_options": has_accepted,
                "has_rejected_options": has_rejected,
                "preservation_boundary": preservation_boundary,
                "stop_boundary": stop_boundary,
                "user_intent": user_intent,
                "actionability": actionability,
                "closure_readiness": closure_readiness,
            }
            return {
                **post_decision,
                "action": to_action,
                "accepted_phase": (
                    "finalization" if to_action == "close" else "confirmation"
                ),
                "instruction": (
                    "Close briefly around the user's chosen plan. Mention the "
                    "main tradeoff only if needed, and do not introduce new "
                    "recommendations."
                    if to_action == "close"
                    else (
                        "Confirm the user's chosen plan as a workable compromise. "
                        "Do not introduce a new recommendation or revisit rejected "
                        "adjustments."
                    )
                ),
                "reasoning": (
                    "Post-assessment gate prevented a new recommendation because "
                    "the user preserved the current plan after accepting and/or "
                    "rejecting specific adjustments."
                ),
            }

        gate = ""
        if planning_override == "redundant_recommendation_redirected" and (
            user_intent == "accepting"
            or actionability == "settled"
            or closure_readiness == "ready_to_close"
            or (
                has_accepted
                and not has_unresolved_issue
                and closure_readiness == "actionable"
            )
        ):
            gate = "post_assessment_redundant_recommendation_closed"
        elif (
            has_accepted
            and user_intent == "accepting"
            and actionability in {"workable", "settled"}
        ):
            gate = "post_assessment_accepted_plan_closed"

        if not gate:
            return post_decision

        metadata["post_assessment_gate"] = {
            "gate": gate,
            "from_action": "recommend",
            "to_action": "confirm",
            "has_accepted_options": has_accepted,
            "user_intent": user_intent,
            "actionability": actionability,
            "closure_readiness": closure_readiness,
        }
        return {
            **post_decision,
            "action": "confirm",
            "accepted_phase": "confirmation",
            "instruction": (
                "Confirm the current assessed plan. Mention important "
                "limitations only if needed, but do not introduce a replacement "
                "recommendation or another optional refinement."
            ),
            "reasoning": (
                "Post-assessment gate prevented a recommendation cycle after "
                "the action-space check had already identified the current plan "
                "as actionable."
            ),
        }

    def _run_meal_assessment(
        self,
        *,
        meal_assessor: MealAssessor,
        history,
        prior_state: CoachingState,
        alignment_score: float | None,
        alignment_reasoning: str | None,
        metadata: dict,
    ) -> dict:
        assessment_messages = meal_assessor.get_messages(
            history=history,
            alignment_score=alignment_score or 0.0,
            alignment_reasoning=alignment_reasoning or "",
            user_preferences=prior_state.user_preferences,
            recommendation_history=list(prior_state.recommendation_history),
        )
        assessment_raw = self._generate(
            module="meal_assessor",
            messages=assessment_messages,
            mode="assessment",
            response_schema=ASSESSMENT_SCHEMA,
        )
        assessment = meal_assessor.parse_with_retry(
            base_msgs=assessment_messages,
            raw_output=assessment_raw,
            reinvoke_fn=lambda messages: self._generate(
                module="meal_assessor",
                messages=messages,
                mode="assessment",
                response_schema=ASSESSMENT_SCHEMA,
            ),
        )
        metadata["assessment_raw_output"] = assessment_raw
        metadata["assessment_result"] = assessment
        metadata["assessment_degraded"] = bool(assessment.get("_degraded"))
        metadata["assessment_parse_error"] = assessment.get("_parse_error") or ""
        return assessment

    @classmethod
    def _infer_finalization_style(
        cls,
        *,
        decision: dict,
        user_intent: str,
        interaction_state: str,
        metadata: dict,
    ) -> str:
        """Infer the closing style from workflow state, not food-specific cases."""
        commitment_gate = metadata.get("commitment_gate") or {}
        post_gate = metadata.get("post_assessment_gate") or {}
        planning_policy = metadata.get("planning_policy") or {}
        gate_name = str(commitment_gate.get("gate") or post_gate.get("gate") or "")
        closure_readiness = str(
            decision.get("closure_readiness")
            or planning_policy.get("closure_readiness")
            or ""
        ).lower()
        override = str(planning_policy.get("override") or "")
        instruction = str(decision.get("instruction") or "").lower()

        has_rejected = cls._interaction_section_has_items(
            interaction_state,
            "Rejected options",
        )
        has_unavailable = cls._interaction_section_has_items(
            interaction_state,
            "Unavailable options",
        )

        if gate_name.startswith("safety_conflict") or "safety" in instruction:
            style = "reflective"
        elif has_unavailable:
            style = "reflective"
        elif (
            closure_readiness == "boundary_close"
            or user_intent in {"rejecting", "disengaging"}
            or "boundary" in override
            or "resistance" in override
            or has_rejected
        ):
            style = "educational"
        elif commitment_gate.get("commitment_status") == "boundary":
            style = "reflective"
        else:
            style = "motivational"
        metadata["finalization_style"] = style
        return style

    @staticmethod
    def _normalize_finalization_style(style: str) -> str:
        if style in {"motivational", "educational", "reflective"}:
            return style
        return "motivational"

    def _generate_close_reply(
        self,
        *,
        response_generator: ResponseGenerator,
        history,
        instruction: str,
        alignment_score: float | None,
        assessment: dict | None = None,
        recommendation_history: list[dict] | None = None,
        finalization_style: str = "motivational",
    ) -> tuple[AssistantReply, dict]:
        finalization_style = self._normalize_finalization_style(finalization_style)
        assessment = assessment or {
            "summary": history.meal_base or "The user discussed their meal.",
            "strengths": [],
            "limitations": [],
            "overall": (
                "aligned"
                if alignment_score is not None and alignment_score >= 0.8
                else "partially_aligned"
            ),
        }
        response_messages = response_generator.get_motivational_messages(
            assessment,
            history,
            exit_context=instruction,
            finalization_style=finalization_style,
            recommendation_history=recommendation_history,
        )
        response_raw = self._generate(
            module="response_generator",
            messages=response_messages,
            mode="response_generator",
        )
        text = response_generator.clean_response_text(response_raw)
        text = self._remove_questions_from_final_text(text)
        if not text:
            text = response_generator.fallback_motivational_ending_text(assessment)
        return (
            AssistantReply(
                text,
                kind="close",
                metadata={**assessment, "finalization_style": finalization_style},
            ),
            assessment,
        )

    @staticmethod
    def _remove_questions_from_final_text(text: str) -> str:
        """Finalization is terminal; remove accidental follow-up questions."""
        cleaned = (text or "").strip()
        if "?" not in cleaned:
            return cleaned
        sentences = re.split(r"(?<=[.!])\s+", cleaned)
        kept = [s.strip() for s in sentences if s.strip() and "?" not in s]
        if kept:
            return " ".join(kept).strip()
        return ""

    def _generate_recommendation_reply(
        self,
        *,
        recommender: MealRecommender,
        response_generator: ResponseGenerator,
        history,
        prior_state: CoachingState,
        alignment_score: float | None,
        alignment_reasoning: str | None,
        instruction: str,
        turn_idx: int,
        interaction_state: str,
        assessment: dict | None = None,
        metadata: dict | None = None,
    ) -> tuple[AssistantReply, dict]:
        rec_messages = recommender.get_messages(
            meal_base=history.meal_base,
            alignment_score=alignment_score or 0.0,
            alignment_reasoning=alignment_reasoning or "",
            instruction=instruction,
            user_preferences=prior_state.user_preferences,
            interaction_state=interaction_state,
            recommendation_history=list(prior_state.recommendation_history),
        )
        rec_raw = self._generate(
            module="recommender",
            messages=rec_messages,
            mode="recommender",
            response_schema=RECOMMENDATION_SCHEMA,
        )
        rec = recommender.parse_with_retry(
            base_msgs=rec_messages,
            raw_output=rec_raw,
            reinvoke_fn=lambda messages: self._generate(
                module="recommender",
                messages=messages,
                mode="recommender",
                response_schema=RECOMMENDATION_SCHEMA,
            ),
            turn_idx=turn_idx,
        )
        if metadata is not None:
            metadata["recommendation_raw_output"] = rec_raw
        response_messages = response_generator.get_recommendation_messages(
            rec,
            history,
            recommendation_history=list(prior_state.recommendation_history),
            current_assessment=assessment,
        )
        response_raw = self._generate(
            module="response_generator",
            messages=response_messages,
            mode="response_generator",
        )
        text = response_generator.clean_response_text(response_raw)
        if not text:
            text = response_generator.fallback_recommendation_text(rec)
        return AssistantReply(text, kind="recommendation", metadata=rec), rec

    def _generate_assessment_reply(
        self,
        *,
        response_generator: ResponseGenerator,
        assessment: dict,
        history,
        recommendation_history: list[dict] | None = None,
    ) -> AssistantReply:
        if assessment.get("_degraded"):
            return AssistantReply(
                response_generator.fallback_assessment_text(assessment),
                kind="assessment",
                metadata=assessment,
            )
        response_messages = response_generator.get_assessment_messages(
            assessment,
            assessment.get("overall") != "aligned",
            history,
            recommendation_history=recommendation_history,
        )
        response_raw = self._generate(
            module="response_generator",
            messages=response_messages,
            mode="response_generator",
        )
        text = response_generator.clean_response_text(response_raw)
        if not text:
            text = response_generator.fallback_assessment_text(assessment)
        return AssistantReply(text, kind="assessment", metadata=assessment)

    def _generate_question_reply(
        self,
        *,
        coach: InformationSeeker,
        response_generator: ResponseGenerator,
        history,
        phase: str,
        prior_state: CoachingState,
        instruction: str,
        interaction_state: str,
    ) -> tuple[AssistantReply, dict]:
        info_messages = coach.get_messages(
            history,
            phase=phase,
            user_preferences=prior_state.user_preferences,
            interaction_state=interaction_state,
            instruction=instruction,
        )
        info_raw = self._generate(
            module="info_seeker",
            messages=info_messages,
            mode="coach",
        )
        template = coach._parse_template(info_raw)
        response_messages = response_generator.get_question_messages(template, history)
        response_raw = self._generate(
            module="response_generator",
            messages=response_messages,
            mode="response_generator",
        )
        text = response_generator.clean_response_text(response_raw)
        if not text:
            text = response_generator.fallback_question_text(template)
        return AssistantReply(text, kind="question", metadata=template), template

    def _generate_answer_reply(
        self,
        *,
        response_generator: ResponseGenerator,
        history,
        instruction: str,
    ) -> AssistantReply:
        response_messages = response_generator.get_answer_messages(instruction, history)
        response_raw = self._generate(
            module="response_generator",
            messages=response_messages,
            mode="response_generator",
        )
        text = response_generator.clean_response_text(response_raw)
        if not text:
            text = instruction.strip() or "Could you tell me a bit more?"
        return AssistantReply(text, kind="answer")

    def _generate_confirmation_reply(
        self,
        *,
        response_generator: ResponseGenerator,
        history,
    ) -> AssistantReply:
        response_messages = response_generator.get_confirmation_messages(history)
        response_raw = self._generate(
            module="response_generator",
            messages=response_messages,
            mode="response_generator",
        )
        text = response_generator.clean_response_text(response_raw)
        if not text:
            text = response_generator.fallback_confirmation_text(history)
        return AssistantReply(text, kind="confirmation")

    def _generate_handoff_reply(
        self,
        *,
        response_generator: ResponseGenerator,
        history,
        instruction: str,
    ) -> AssistantReply:
        response_messages = response_generator.get_handoff_messages(
            instruction=instruction,
            history=history,
        )
        response_raw = self._generate(
            module="response_generator",
            messages=response_messages,
            mode="response_generator",
        )
        text = response_generator.clean_response_text(response_raw)
        if not text:
            text = response_generator.fallback_handoff_text()
        return AssistantReply(text, kind="handoff")
