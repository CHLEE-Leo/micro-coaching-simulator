"""Portable conversation engine for one chatbot turn."""

from __future__ import annotations

import concurrent.futures as _futures
from collections.abc import Callable
from dataclasses import replace

from .agent_config import AgentConfig
from .contracts import (
    AssistantReply,
    CoachingState,
    CoachingTurnRequest,
    CoachingTurnResult,
)
from .history_adapter import build_shared_history
from .modules.alignment_estimator import AlignmentEstimator
from .modules.certainty_estimator import CertaintyEstimator
from .modules.context_tracker import ContextTracker
from .modules.guardrail import Guardrail
from .modules.information_seeker import InformationSeeker
from .modules.meal_recommender import MealRecommender
from .modules.meal_tracker import MealTrackerModel
from .modules.orchestrator import Orchestrator
from .modules.phase_predictor import PhasePredictor
from .modules.response_generator import ResponseGenerator
from .opening import build_opening_message

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
        self._generate = generate_response
        self.config = config or AgentConfig()

    def generate_chat_replies(self, request: CoachingTurnRequest) -> CoachingTurnResult:
        """Generate one or more assistant bubbles for ``request.current_message``."""

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
        orchestrator = Orchestrator(request.nutrition_goal, self.config)
        phase_predictor = PhasePredictor(request.nutrition_goal, self.config)
        response_generator = ResponseGenerator(request.nutrition_goal, self.config)
        recommender = MealRecommender(request.nutrition_goal, self.config)
        guardrail = Guardrail(config=self.config)

        early_result = self._run_tracking_stage(
            request=request,
            clean_message=clean_message,
            history=history,
            prior_state=prior_state,
            guardrail=guardrail,
            meal_tracker=meal_tracker,
            context_tracker=context_tracker,
            metadata=metadata,
        )
        if early_result is not None:
            return early_result

        alignment_score, alignment_reasoning = self._estimate_alignment_state(
            request=request,
            history=history,
            turn_idx=adapted.turn_idx,
            metadata=metadata,
        )
        certainty_score, certainty_reasoning = self._estimate_certainty_state(
            request=request,
            history=history,
            prior_state=prior_state,
            metadata=metadata,
        )
        (
            predicted_phase,
            phase_prediction_reasoning,
            phase_prediction_confidence,
        ) = self._predict_dialogue_phase(
            phase_predictor=phase_predictor,
            history=history,
            prior_state=prior_state,
            turn_idx=adapted.turn_idx,
            current_phase=phase,
            alignment_score=alignment_score,
            alignment_reasoning=alignment_reasoning,
            certainty_score=certainty_score,
            certainty_reasoning=certainty_reasoning,
            metadata=metadata,
        )
        decision = self._select_next_action(
            orchestrator=orchestrator,
            history=history,
            prior_state=prior_state,
            turn_idx=adapted.turn_idx,
            predicted_phase=predicted_phase,
            alignment_score=alignment_score,
            alignment_reasoning=alignment_reasoning,
            certainty_score=certainty_score,
            certainty_reasoning=certainty_reasoning,
            phase_prediction_reasoning=phase_prediction_reasoning,
            phase_prediction_confidence=phase_prediction_confidence,
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
        ) = self._apply_intent_overrides(
            decision=decision,
            phase=decision.get("accepted_phase") or predicted_phase,
            prior_state=prior_state,
            alignment_score=alignment_score,
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
            orchestrator=orchestrator,
            response_generator=response_generator,
            metadata=metadata,
        )

        next_state = CoachingState(
            phase=phase,
            status=status,
            meal_base=history.meal_base,
            tracker_state=history.tracker_state,
            context_base=history.context_base,
            user_preferences=prior_state.user_preferences,
            recommendation_history=prior_state.recommendation_history,
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
        )
        return CoachingTurnResult(
            assistant_messages=assistant_messages,
            state=next_state,
            status=status,
            terminated_by=terminated_by,
            metadata=metadata,
        )

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
        metadata: dict,
    ) -> CoachingTurnResult | None:
        """Run input guardrail, meal tracking, and context tracking."""

        guard_messages = None
        if request.enable_guardrail:
            guard_messages = guardrail.get_input_guard_messages(
                user_input=clean_message,
                dialog_context=history.to_recent_turns_text(n=2),
            )
        meal_messages = meal_tracker.get_messages(history.to_plain_text())
        context_messages = (
            context_tracker.get_messages(history.to_plain_text())
            if request.enable_context_tracking
            else None
        )

        executor = _futures.ThreadPoolExecutor(max_workers=3)
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
                    executor.shutdown(wait=False, cancel_futures=True)
                    closed = True
                    return early_result

            meal_raw = meal_future.result()
            parsed_meal = meal_tracker.parse_tracking_output(meal_raw)
            history.update_tracker_state(parsed_meal["tracker_state"])
            history.update_meal_base(parsed_meal["meal_base"])
            metadata["meal_tracker_output"] = meal_raw

            if context_future is not None:
                context_raw = context_future.result()
                history.update_context_base(context_raw)
                metadata["context_tracker_output"] = context_raw
            return None
        finally:
            if not closed:
                executor.shutdown(wait=True)

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

    def _estimate_alignment_state(
        self,
        *,
        request: CoachingTurnRequest,
        history,
        turn_idx: int,
        metadata: dict,
    ) -> tuple[float | None, str | None]:
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
            return prior_state.last_certainty_score, prior_state.last_certainty_reasoning

        certainty = CertaintyEstimator(request.nutrition_goal, self.config)
        certainty_messages = certainty.get_messages(history)
        certainty_raw = self._generate(
            module="certainty_estimator",
            messages=certainty_messages,
            mode="certainty",
        )
        certainty_reasoning, certainty_score = certainty.parse_output(certainty_raw)
        metadata["certainty_raw_output"] = certainty_raw
        return certainty_score, certainty_reasoning

    def _predict_dialogue_phase(
        self,
        *,
        phase_predictor: PhasePredictor,
        history,
        prior_state: CoachingState,
        turn_idx: int,
        current_phase: str,
        alignment_score: float | None,
        alignment_reasoning: str | None,
        certainty_score: float | None,
        certainty_reasoning: str | None,
        metadata: dict,
    ) -> tuple[str, str, float | None]:
        if not self.config.use_phase_predictor:
            return current_phase, "(phase predictor disabled)", None

        phase_messages = phase_predictor.get_messages(
            history=history,
            turn_idx=turn_idx,
            current_phase=current_phase,
            recommendation_history=list(prior_state.recommendation_history),
            last_alignment_score=alignment_score,
            last_alignment_reasoning=alignment_reasoning,
            last_certainty_score=certainty_score,
            last_certainty_reasoning=certainty_reasoning,
            user_preferences=prior_state.user_preferences,
        )
        phase_raw = self._generate(
            module="phase_predictor",
            messages=phase_messages,
            mode="orchestrator",
        )
        phase_prediction = phase_predictor.parse_output(phase_raw, current_phase)
        metadata["phase_predictor_raw_output"] = phase_raw
        metadata["phase_prediction"] = phase_prediction
        return (
            phase_prediction["predicted_phase"],
            phase_prediction.get("reasoning", ""),
            phase_prediction.get("confidence"),
        )

    def _select_next_action(
        self,
        *,
        orchestrator: Orchestrator,
        history,
        prior_state: CoachingState,
        turn_idx: int,
        predicted_phase: str,
        alignment_score: float | None,
        alignment_reasoning: str | None,
        certainty_score: float | None,
        certainty_reasoning: str | None,
        phase_prediction_reasoning: str,
        phase_prediction_confidence: float | None,
        metadata: dict,
    ) -> dict:
        route_messages = orchestrator.get_routing_messages(
            history=history,
            turn_idx=turn_idx,
            phase=predicted_phase,
            recommendation_history=list(prior_state.recommendation_history),
            consecutive_qa_count=prior_state.consecutive_qa_count,
            last_alignment_score=alignment_score,
            last_alignment_reasoning=alignment_reasoning,
            last_certainty_score=certainty_score,
            last_certainty_reasoning=certainty_reasoning,
            user_preferences=prior_state.user_preferences,
            phase_prediction_reasoning=phase_prediction_reasoning,
            phase_prediction_confidence=phase_prediction_confidence,
        )
        route_raw = self._generate(
            module="orchestrator",
            messages=route_messages,
            mode="orchestrator",
        )
        decision = orchestrator.parse_routing_with_retry(
            base_msgs=route_messages,
            raw_output=route_raw,
            turn_idx=turn_idx,
            phase=predicted_phase,
            reinvoke_fn=lambda messages: self._generate(
                module="orchestrator",
                messages=messages,
                mode="orchestrator",
            ),
        )
        metadata["orchestrator_raw_output"] = route_raw
        metadata["orchestrator_decision"] = decision
        return decision

    def _apply_intent_overrides(
        self,
        *,
        decision: dict,
        phase: str,
        prior_state: CoachingState,
        alignment_score: float | None,
        metadata: dict,
    ) -> tuple[str, str, dict, str, str, int, int]:
        action = decision.get("action", "inquire")
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

        user_wants_to_end = user_intent == "disengaging" or (
            user_intent == "rejecting" and rejection_count >= 2
        )
        intent_policy: dict = {
            "intent_summary": intent_summary,
            "user_intent": user_intent,
            "stall_count": stall_count,
            "recommendation_rejection_count": rejection_count,
            "user_wants_to_end": user_wants_to_end,
        }

        if rejection_count >= 3 and action not in ("terminate", "close"):
            action = "close"
            phase = "motivational_ending"
            decision = {
                **decision,
                "action": action,
                "accepted_phase": phase,
                "reasoning": "(portable intent policy: user resistance threshold exceeded)",
            }
            metadata["orchestrator_decision"] = decision
            intent_policy["override"] = "resistance_threshold_close"

        if (
            action == "close"
            and alignment_score is not None
            and alignment_score < 0.5
            and not user_wants_to_end
        ):
            action = "inquire"
            phase = "exploration"
            decision = {
                **decision,
                "action": action,
                "accepted_phase": phase,
                "reasoning": (
                    "Portable intent policy redirected a low-alignment close "
                    "back to meal information seeking."
                ),
                "instruction": (
                    "Ask one concrete question about missing meal details before closing."
                ),
            }
            metadata["orchestrator_decision"] = decision
            intent_policy["override"] = "low_alignment_close_redirected"

        metadata["intent_policy"] = intent_policy
        return (
            action,
            phase,
            decision,
            user_intent,
            intent_summary,
            stall_count,
            rejection_count,
        )

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
        orchestrator: Orchestrator,
        response_generator: ResponseGenerator,
        metadata: dict,
    ) -> tuple[list[AssistantReply], str, str, str | None]:
        assistant_messages: list[AssistantReply] = []
        status = "active"
        terminated_by = None

        if action == "terminate":
            text = response_generator.fallback_closing_text(decision.get("instruction", ""))
            assistant_messages.append(AssistantReply(text, kind="closing"))
            return assistant_messages, phase, "terminated", "orchestrator"

        if action == "close":
            reply, _assessment = self._generate_close_reply(
                response_generator=response_generator,
                history=history,
                instruction=decision.get("instruction", ""),
                alignment_score=alignment_score,
            )
            assistant_messages.append(reply)
            return assistant_messages, "motivational_ending", "terminated", "close"

        if action == "recommend":
            reply, rec = self._generate_recommendation_reply(
                recommender=recommender,
                response_generator=response_generator,
                history=history,
                prior_state=prior_state,
                alignment_score=alignment_score,
                alignment_reasoning=alignment_reasoning,
                instruction=decision.get("instruction", ""),
                turn_idx=turn_idx,
            )
            assistant_messages.append(reply)
            metadata["recommendation_result"] = rec
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
                user_intent=user_intent,
                coach=coach,
                recommender=recommender,
                orchestrator=orchestrator,
                response_generator=response_generator,
                metadata=metadata,
            )
            post_decision = metadata["post_assessment_decision"]
            post_action = post_decision.get("action", "inquire")
            phase = post_decision.get("accepted_phase") or (
                "motivational_ending"
                if post_action in ("close", "terminate")
                else "recommendation"
            )
            if post_action == "terminate":
                status = "terminated"
                terminated_by = "post_assessment"
            elif post_action == "close":
                phase = "motivational_ending"
                status = "terminated"
                terminated_by = "close"
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
        user_intent: str,
        coach: InformationSeeker,
        recommender: MealRecommender,
        orchestrator: Orchestrator,
        response_generator: ResponseGenerator,
        metadata: dict,
    ) -> None:
        assessment = self._run_meal_assessment(
            orchestrator=orchestrator,
            history=history,
            alignment_score=alignment_score,
            alignment_reasoning=alignment_reasoning,
            metadata=metadata,
        )
        assistant_messages.append(
            self._generate_assessment_reply(
                response_generator=response_generator,
                assessment=assessment,
                history=history,
            )
        )

        post_decision = self._select_post_assessment_action(
            orchestrator=orchestrator,
            history=history,
            turn_idx=turn_idx,
            assessment=assessment,
            alignment_score=alignment_score,
            alignment_reasoning=alignment_reasoning,
            user_intent=user_intent,
            user_preferences=prior_state.user_preferences,
            metadata=metadata,
        )
        post_action = post_decision.get("action", "inquire")
        phase = post_decision.get("accepted_phase") or (
            "motivational_ending"
            if post_action in ("close", "terminate")
            else "recommendation"
        )

        if post_action == "close":
            reply, _assessment = self._generate_close_reply(
                response_generator=response_generator,
                history=history,
                instruction=post_decision.get("instruction", ""),
                alignment_score=alignment_score,
                assessment=assessment,
            )
            assistant_messages.append(reply)
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
            )
            assistant_messages.append(reply)
            metadata["question_result"] = template

    def _run_meal_assessment(
        self,
        *,
        orchestrator: Orchestrator,
        history,
        alignment_score: float | None,
        alignment_reasoning: str | None,
        metadata: dict,
    ) -> dict:
        assessment_messages = orchestrator.get_assessment_messages(
            history=history,
            alignment_score=alignment_score or 0.0,
            alignment_reasoning=alignment_reasoning or "",
        )
        assessment_raw = self._generate(
            module="orchestrator",
            messages=assessment_messages,
            mode="assessment",
        )
        assessment = orchestrator.parse_assessment_with_retry(
            base_msgs=assessment_messages,
            raw_output=assessment_raw,
            reinvoke_fn=lambda messages: self._generate(
                module="orchestrator",
                messages=messages,
                mode="assessment",
            ),
        )
        metadata["assessment_raw_output"] = assessment_raw
        metadata["assessment_result"] = assessment
        return assessment

    def _select_post_assessment_action(
        self,
        *,
        orchestrator: Orchestrator,
        history,
        turn_idx: int,
        assessment: dict,
        alignment_score: float | None,
        alignment_reasoning: str | None,
        user_intent: str,
        user_preferences: str,
        metadata: dict,
    ) -> dict:
        post_messages = orchestrator.get_post_assessment_routing_messages(
            history=history,
            turn_idx=turn_idx,
            assessment_result=assessment,
            alignment_score=alignment_score,
            alignment_reasoning=alignment_reasoning,
            user_intent=user_intent,
            user_preferences=user_preferences,
        )
        post_raw = self._generate(
            module="orchestrator",
            messages=post_messages,
            mode="orchestrator",
        )
        post_decision = orchestrator.parse_post_assessment_with_retry(
            base_msgs=post_messages,
            raw_output=post_raw,
            reinvoke_fn=lambda messages: self._generate(
                module="orchestrator",
                messages=messages,
                mode="orchestrator",
            ),
        )
        metadata["post_assessment_raw_output"] = post_raw
        metadata["post_assessment_decision"] = post_decision
        return post_decision

    def _generate_close_reply(
        self,
        *,
        response_generator: ResponseGenerator,
        history,
        instruction: str,
        alignment_score: float | None,
        assessment: dict | None = None,
    ) -> tuple[AssistantReply, dict]:
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
        )
        response_raw = self._generate(
            module="response_generator",
            messages=response_messages,
            mode="orchestrator",
        )
        text = response_generator.clean_response_text(response_raw)
        if not text:
            text = response_generator.fallback_motivational_ending_text(assessment)
        return AssistantReply(text, kind="close", metadata=assessment), assessment

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
    ) -> tuple[AssistantReply, dict]:
        rec_messages = recommender.get_messages(
            meal_base=history.meal_base,
            alignment_score=alignment_score or 0.0,
            alignment_reasoning=alignment_reasoning or "",
            instruction=instruction,
            user_preferences=prior_state.user_preferences,
            recommendation_history=list(prior_state.recommendation_history),
        )
        rec_raw = self._generate(
            module="recommender",
            messages=rec_messages,
            mode="recommender",
        )
        rec = recommender.parse_output(rec_raw, turn_idx=turn_idx)
        response_messages = response_generator.get_recommendation_messages(rec, history)
        response_raw = self._generate(
            module="response_generator",
            messages=response_messages,
            mode="orchestrator",
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
    ) -> AssistantReply:
        response_messages = response_generator.get_assessment_messages(
            assessment,
            assessment.get("overall") != "aligned",
            history,
        )
        response_raw = self._generate(
            module="response_generator",
            messages=response_messages,
            mode="orchestrator",
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
    ) -> tuple[AssistantReply, dict]:
        info_messages = coach.get_messages(
            history,
            phase=phase,
            user_preferences=prior_state.user_preferences,
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
            mode="orchestrator",
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
            mode="orchestrator",
        )
        text = response_generator.clean_response_text(response_raw)
        if not text:
            text = instruction.strip() or "Could you tell me a bit more?"
        return AssistantReply(text, kind="answer")
