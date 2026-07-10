"""
LLM Pipeline Unit Tests — Phase 1
──────────────────────────────────
개별 에이전트 parse 함수의 입출력 검증.
LLM 호출 없이 순수 파싱 로직만 테스트한다.
"""

import sys
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[2]          # micro-coaching-simulator/
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pytest
from code_interactive.agents.agent_config import AgentConfig


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: minimal config for instantiation
# ═══════════════════════════════════════════════════════════════════════════════
@pytest.fixture
def cfg():
    return AgentConfig()


# ═══════════════════════════════════════════════════════════════════════════════
# 1. MealAssessor — parse()
# ═══════════════════════════════════════════════════════════════════════════════
class TestMealAssessmentParsing:
    @pytest.fixture(autouse=True)
    def setup(self, cfg):
        from code_interactive.agents.modules.meal_assessor import MealAssessor
        self.assessor = MealAssessor(nutrition_goal="lean_protein", config=cfg)

    # U-R1: 정상 JSON
    def test_normal_json(self):
        raw = '{"summary":"ok","strengths":["protein"],"limitations":[],"overall":"aligned"}'
        result = self.assessor.parse(raw)
        assert result["overall"] == "aligned"
        assert result["strengths"] == ["protein"]

    # U-R2: Markdown 코드블록
    def test_markdown_code_block(self):
        raw = '```json\n{"summary":"ok","strengths":[],"limitations":["low protein"],"overall":"partially_aligned"}\n```'
        result = self.assessor.parse(raw)
        assert result["overall"] == "partially_aligned"
        assert result["limitations"] == ["low protein"]

    # U-R3: 유효하지 않은 overall → fallback
    def test_invalid_overall_fallback(self):
        raw = '{"summary":"ok","strengths":[],"limitations":[],"overall":"excellent"}'
        result = self.assessor.parse(raw)
        assert result["overall"] == "partially_aligned"
        assert result["summary"] == ""
        assert result["_degraded"] is True
        assert "invalid overall" in result["_parse_error"]

    # U-R4: scalar strengths/limitations are normalized for downstream code.
    def test_scalar_fields_are_normalized_to_lists(self):
        raw = '{"summary":"ok","strengths":"lean protein","limitations":"low vegetables","overall":"not_aligned"}'
        result = self.assessor.parse(raw)
        assert result["strengths"] == ["lean protein"]
        assert result["limitations"] == ["low vegetables"]

    # U-R5: JSON 파싱 실패 → fallback
    def test_json_parse_failure(self):
        raw = "this is not json at all"
        result = self.assessor.parse(raw)
        assert result["overall"] == "partially_aligned"
        assert result["summary"] == ""
        assert result["_degraded"] is True
        assert "JSON decode failed" in result["_parse_error"]

    # U-R6: 빈 문자열 → fallback
    def test_empty_string(self):
        raw = ""
        result = self.assessor.parse(raw)
        assert result["overall"] == "partially_aligned"
        assert result["_degraded"] is True
        assert result["_parse_error"] == "empty LLM response"

    def test_normal_json_has_parse_telemetry(self):
        raw = '{"summary":"ok","strengths":[],"limitations":[],"overall":"aligned"}'
        result = self.assessor.parse(raw)
        assert result["_degraded"] is False
        assert result["_parse_error"] == ""


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Guardrail — parse_input_guard() & parse_output_guard()
# ═══════════════════════════════════════════════════════════════════════════════
class TestGuardrail:
    @pytest.fixture(autouse=True)
    def setup(self, cfg):
        from code_interactive.agents.modules.guardrail import Guardrail
        self.guard = Guardrail(config=cfg)

    # U-G1: 정상 — passed=True
    def test_input_passed_true(self):
        raw = '{"passed":true}'
        result = self.guard.parse_input_guard(raw)
        assert result["passed"] is True

    # U-G2: 주제 이탈 차단
    def test_input_blocked(self):
        raw = '{"passed":false,"reason":"off-topic","message":"Let us focus on your meal."}'
        result = self.guard.parse_input_guard(raw)
        assert result["passed"] is False
        assert "meal" in result.get("message", "").lower()

    # U-G3: 빈 출력 → 안전 기본값 (pass)
    def test_input_empty(self):
        raw = ""
        result = self.guard.parse_input_guard(raw)
        assert result["passed"] is True

    # U-G4: JSON 파싱 실패 → 안전 기본값
    def test_input_garbage(self):
        raw = "not json"
        result = self.guard.parse_input_guard(raw)
        assert result["passed"] is True

    # U-G5: Output guard — passed=true
    def test_output_passed(self):
        raw = '{"passed":true}'
        result = self.guard.parse_output_guard(raw)
        assert result["passed"] is True

    # U-G6: Output guard — 의료 조언 차단
    def test_output_medical_blocked(self):
        raw = '{"passed":false,"reason":"medical prescription"}'
        result = self.guard.parse_output_guard(raw)
        assert result["passed"] is False

    def test_output_missing_passed_is_unusable_telemetry(self):
        raw = '{"reason":"missing schema field"}'
        result = self.guard.parse_output_guard(raw)
        assert result["passed"] is None
        assert result["reason"] == "output_guard_missing_passed"

    def test_output_garbage_is_unusable_telemetry(self):
        raw = "not json"
        result = self.guard.parse_output_guard(raw)
        assert result["passed"] is None
        assert result["reason"] == "output_guard_parse_error"

    # U-G7: Markdown 코드블록 안 JSON
    def test_input_markdown_wrapped(self):
        raw = '```json\n{"passed":true}\n```'
        result = self.guard.parse_input_guard(raw)
        assert result["passed"] is True


# ═══════════════════════════════════════════════════════════════════════════════
# 3. AlignmentEstimator — _parse_answer()
# ═══════════════════════════════════════════════════════════════════════════════
class TestAlignmentEstimator:
    def _make_estimator(self, cfg, output_format="binary"):
        cfg.alignment_output_format = output_format
        from code_interactive.agents.modules.alignment_estimator import AlignmentEstimator
        return AlignmentEstimator(
            model=None,
            nutrition_goal="lean_protein",
            config=cfg,
        )

    # U-A1: binary aligned
    def test_binary_aligned(self, cfg):
        est = self._make_estimator(cfg, "binary")
        score = est._parse_answer('{"answer":1,"reasoning":"good protein"}')
        assert score == 1.0

    # U-A2: binary not aligned
    def test_binary_not_aligned(self, cfg):
        est = self._make_estimator(cfg, "binary")
        score = est._parse_answer('{"answer":0,"reasoning":"too much carbs"}')
        assert score == 0.0

    # U-A3: 0-1 above threshold
    def test_01_above_threshold(self, cfg):
        cfg.alignment_threshold = 0.5
        est = self._make_estimator(cfg, "0-1")
        score = est._parse_answer('{"answer":0.7,"reasoning":"mostly good"}')
        assert score == pytest.approx(0.7, abs=0.01)
        assert score >= cfg.alignment_threshold

    # U-A4: 0-1 below threshold
    def test_01_below_threshold(self, cfg):
        cfg.alignment_threshold = 0.5
        est = self._make_estimator(cfg, "0-1")
        score = est._parse_answer('{"answer":0.3,"reasoning":"gaps"}')
        assert score == pytest.approx(0.3, abs=0.01)
        assert score < cfg.alignment_threshold

    # U-A5: 0-100 normalized
    def test_0100_normalized(self, cfg):
        cfg.alignment_threshold = 0.5
        est = self._make_estimator(cfg, "0-100")
        score = est._parse_answer('{"answer":75,"reasoning":"good"}')
        assert score == pytest.approx(0.75, abs=0.01)

    # U-A6: 잘못된 JSON → None
    def test_bad_json(self, cfg):
        est = self._make_estimator(cfg, "binary")
        score = est._parse_answer("not json at all")
        assert score is None

    # U-A7: Markdown 래핑된 JSON
    def test_markdown_wrapped(self, cfg):
        est = self._make_estimator(cfg, "0-1")
        score = est._parse_answer('```json\n{"answer":0.8,"reasoning":"fine"}\n```')
        assert score == pytest.approx(0.8, abs=0.01)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. CertaintyEstimator — parse_output()
# ═══════════════════════════════════════════════════════════════════════════════
class TestCertaintyEstimator:
    @pytest.fixture(autouse=True)
    def setup(self, cfg):
        from code_interactive.agents.modules.certainty_estimator import CertaintyEstimator
        self.est = CertaintyEstimator(nutrition_goal="lean_protein", config=cfg)

    # U-C1: 높은 확신도
    def test_high_certainty(self):
        raw = '{"reasoning":"all details known","certainty_score":0.92}'
        reasoning, score = self.est.parse_output(raw)
        assert score == pytest.approx(0.92, abs=0.01)
        assert "details" in reasoning.lower() or len(reasoning) > 0

    # U-C2: 낮은 확신도
    def test_low_certainty(self):
        raw = '{"reasoning":"missing cooking method","certainty_score":0.4}'
        reasoning, score = self.est.parse_output(raw)
        assert score == pytest.approx(0.4, abs=0.01)

    # U-C3: 경계값
    def test_boundary_certainty(self):
        raw = '{"reasoning":"mostly known","certainty_score":0.85}'
        reasoning, score = self.est.parse_output(raw)
        assert score == pytest.approx(0.85, abs=0.01)
        assert score >= 0.85

    # U-C4: JSON 파싱 실패 → 0.0
    def test_parse_failure(self):
        raw = "not json"
        reasoning, score = self.est.parse_output(raw)
        assert score == 0.0

    # U-C5: 범위 초과 → clamp
    def test_out_of_range_clamped(self):
        raw = '{"reasoning":"test","certainty_score":1.5}'
        reasoning, score = self.est.parse_output(raw)
        assert score <= 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# 5. InformationSeeker — _parse_template()
# ═══════════════════════════════════════════════════════════════════════════════
class TestInformationSeeker:
    @pytest.fixture(autouse=True)
    def setup(self, cfg):
        from code_interactive.agents.modules.information_seeker import InformationSeeker
        self.seeker = InformationSeeker(model=None, nutrition_goal="lean_protein", meal_type="dinner", config=cfg)

    # U-IS1: 정상 질문 템플릿
    def test_normal_template(self):
        raw = '{"question_type":"portion","target":"rice","reasoning":"unknown amount","question_template":"How much rice did you have?"}'
        result = self.seeker._parse_template(raw)
        assert result["question_type"] == "portion"
        assert result["target"] == "rice"
        assert "rice" in result["question_template"].lower()

    # U-IS2: JSON 파싱 실패 → fallback
    def test_parse_failure_fallback(self):
        raw = "not json at all"
        result = self.seeker._parse_template(raw)
        assert "question_template" in result
        assert isinstance(result["question_template"], str)
        assert len(result["question_template"]) > 0

    # U-IS3: 빈 문자열 → fallback
    def test_empty_input(self):
        raw = ""
        result = self.seeker._parse_template(raw)
        assert "question_template" in result


# ═══════════════════════════════════════════════════════════════════════════════
# 6. MealRecommender — parse_output()
# ═══════════════════════════════════════════════════════════════════════════════
class TestMealRecommender:
    @pytest.fixture(autouse=True)
    def setup(self, cfg):
        from code_interactive.agents.modules.meal_recommender import MealRecommender
        self.rec = MealRecommender(nutrition_goal="lean_protein", config=cfg)

    # U-MR1: substitute 추천
    def test_substitute_recommendation(self):
        raw = '{"recommendation_type":"substitute","target_food":"white rice","suggestion":"brown rice","reasoning":"more fiber","expected_impact":"high"}'
        result = self.rec.parse_output(raw)
        assert result["recommendation_type"] == "substitute"
        assert result["target_food"] == "white rice"
        assert result["suggestion"] == "brown rice"

    def test_cautious_continuation_recommendation_type_is_preserved(self):
        raw = (
            '{"recommendation_type":"cautious_continuation",'
            '"target_food":"overall meal plan",'
            '"suggestion":"If the user still includes a small amount of omelet, keep the rest of dinner centered on yogurt and modest portions.",'
            '"reasoning":"The omelet remains safety-conflicted, so advice should focus on non-conflicted components.",'
            '"expected_impact":"medium",'
            '"options":[{'
            '"option_id":"opt1",'
            '"target_food":"non-conflicted dinner components",'
            '"suggestion":"Keep yogurt as the main protein support and keep pancake and potatoes modest.",'
            '"reasoning":"This supports the goal without endorsing the conflicted omelet.",'
            '"expected_impact":"medium"'
            '}]}'
        )
        result = self.rec.parse_output(raw)

        assert result["recommendation_type"] == "cautious_continuation"
        assert result["target_food"] == "overall meal plan"
        assert result["options"][0]["target_food"] == "non-conflicted dinner components"

    # U-MR2: JSON 파싱 실패 → fallback
    def test_parse_failure(self):
        raw = "not json"
        result = self.rec.parse_output(raw)
        assert result["recommendation_type"] == "modify"  # default fallback

    def test_messages_treat_accepted_options_as_anchors(self):
        messages = self.rec.get_messages(
            meal_base="- Food items: yogurt and berries",
            alignment_score=0.5,
            alignment_reasoning="More protein would help.",
            interaction_state=(
                "Candidate options:\n"
                "- berries\n"
                "Accepted options:\n"
                "- Greek yogurt\n"
                "Rejected options:\n"
                "- protein shake"
            ),
        )

        system = messages[0]["content"]
        assert "Treat accepted options as anchors" in system
        assert "candidate options" in system
        assert "Greek yogurt" in system
        assert "berries" in system
        assert "protein shake" in system
        assert "hard constraints" in system

    def test_prompt_distinguishes_conflict_only_from_user_requested_conflict(self):
        messages = self.rec.get_messages(
            meal_base="- Food items: yogurt, pancake, potatoes, omelet",
            alignment_score=0.3,
            alignment_reasoning="Eggs conflict with allergy.",
            interaction_state=(
                "Safety-conflicted options:\n"
                "- eggs\n"
                "- omelet\n"
                "User-requested conflicted options:\n"
                "- a little bit eggs\n"
                "- omelet is okay\n"
                "Accepted options:\n"
                "- yogurt\n"
            ),
        )

        system = messages[0]["content"]
        assert "safety_conflicted_options and user_requested_conflicted_options" in system
        assert "cautious_continuation" in system
        assert "repeat a removal recommendation" in system
        assert "non-conflicted parts" in system
        assert "meal plan" in system

    def test_recommend_public_helper_passes_interaction_state(self):
        def fake_generate(_llm, messages, **_kwargs):
            system = messages[0]["content"]
            assert "Accepted options:" in system
            assert "plain sparkling water" in system
            return (
                '{"recommendation_type":"modify",'
                '"target_food":"plain sparkling water",'
                '"suggestion":"keep the plain sparkling water",'
                '"reasoning":"It supports the water goal.",'
                '"expected_impact":"medium"}'
            )

        result = self.rec.recommend(
            meal_base="- Beverages: plain sparkling water",
            alignment_score=0.7,
            alignment_reasoning="Hydrating beverage selected.",
            interaction_state="Accepted options:\n- plain sparkling water",
            generate_fn=fake_generate,
        )

        assert result["target_food"] == "plain sparkling water"


class TestResponseGeneratorStyle:
    def test_clean_response_repairs_user_meal_perspective(self, cfg):
        from code_interactive.agents.modules.response_generator import ResponseGenerator

        generator = ResponseGenerator("lean_protein", cfg)
        text = generator.clean_response_text(
            "For dinner, I have plain yogurt and potatoes. I get wanting to enjoy the omelet."
        )

        assert "For dinner, your plan has plain yogurt and potatoes." in text
        assert "It makes sense that you want to enjoy the omelet." in text
        assert "For dinner, I have" not in text
        assert "I get wanting to" not in text


# ═══════════════════════════════════════════════════════════════════════════════
# 7. LLM mode settings
# ═══════════════════════════════════════════════════════════════════════════════
class TestLLMModeSettings:
    def test_session_assessment_mode_preserves_multiline_json(self, cfg):
        from code_interactive.llm_agent_service import LLMAgentService

        class FakeClient:
            def invoke(self, messages, sampling, max_tokens, **_kwargs):
                assert max_tokens == cfg.assessment_max_new_tokens
                return (
                    "{\n"
                    '  "summary": "ok",\n'
                    '  "strengths": ["protein"],\n'
                    '  "limitations": [],\n'
                    '  "overall": "aligned"\n'
                    "}"
                )

        service = LLMAgentService(
            config=cfg,
            client_pool={"gpt-5.4": FakeClient(), "gpt-5.4-mini": FakeClient()},
        )
        raw = service.run_module_inference(
            module="meal_assessor",
            messages=[],
            mode="assessment",
        )

        assert raw.startswith("{\n")
        assert '"overall": "aligned"' in raw

    def test_agent_config_assessment_mode_uses_json_settings(self, cfg):
        options = cfg.generation_options("assessment")

        assert options["max_new_tokens"] == cfg.assessment_max_new_tokens
        assert options["sampling"] == "greedy"
        assert options["stop_at_newline"] is False


class TestInteractionStateTracker:
    def test_prompt_preserves_cumulative_rejections_and_commitments(self):
        from code_interactive.agents.prompts.roles.interaction_state_tracker import (
            INTERACTION_STATE_INCREMENTAL_SYSTEM_PROMPT,
        )

        prompt = INTERACTION_STATE_INCREMENTAL_SYSTEM_PROMPT

        assert "Preserve cumulative rejections" in prompt
        assert "easy, doable, preferred" in prompt
        assert "candidate_options" in prompt
        assert "Do not add them to" in prompt

    def test_format_state_keeps_operational_categories(self):
        from code_interactive.agents.modules.interaction_state_tracker import (
            InteractionStateTracker,
        )

        text = InteractionStateTracker.format_state(
            {
                "answered_facts": ["user only sees brie at the buffet"],
                "open_questions": ["portion size"],
                "rejected_options": ["starchy foods tonight"],
                "unavailable_options": ["other protein options"],
                "candidate_options": ["apple", "cucumber"],
                "accepted_options": ["brie"],
                "latest_user_position": "The user wants to keep dinner light.",
            }
        )

        assert "Answered facts:" in text
        assert "- user only sees brie at the buffet" in text
        assert "Open questions:" in text
        assert "Rejected options:" in text
        assert "Unavailable options:" in text
        assert "Candidate options:" in text
        assert "Accepted options:" in text
        assert "Latest user position:" in text

    def test_parse_failure_preserves_previous_state(self):
        from code_interactive.agents.modules.interaction_state_tracker import (
            InteractionStateTracker,
        )

        tracker = InteractionStateTracker()

        assert tracker.parse_output("not json", fallback="previous state") == "previous state"


class TestDialoguePlanner:
    @pytest.fixture(autouse=True)
    def setup(self, cfg):
        from code_interactive.agents.modules.dialogue_planner import DialoguePlanner

        self.planner = DialoguePlanner("lean_protein", cfg)

    def test_parse_assessment_plan_with_followup(self):
        raw = (
            '{"intent_summary": "The user provided enough meal detail.", '
            '"user_intent": "informing", '
            '"phase": "exploration", '
            '"actionability": "workable", '
            '"action": "assess", '
            '"closure_readiness": "actionable", '
            '"reasoning": "The meal can be evaluated.", '
            '"instruction": "Assess the meal.", '
            '"assessment_followup_action": "recommend", '
            '"assessment_followup_phase": "recommendation", '
            '"assessment_followup_instruction": "Suggest one vegetable add-on.", '
            '"confidence": 0.84}'
        )

        plan = self.planner.parse_output(raw, fallback_phase="exploration")

        assert plan["action"] == "assess"
        assert plan["accepted_phase"] == "exploration"
        assert plan["actionability"] == "workable"
        assert plan["closure_readiness"] == "actionable"
        assert plan["assessment_followup_action"] == "recommend"
        assert plan["assessment_followup_phase"] == "recommendation"
        assert plan["confidence"] == pytest.approx(0.84)

    def test_partial_planner_json_preserves_decision_fields(self):
        raw = (
            '{"intent_summary": "The user rejects more suggestions.", '
            '"user_intent": "rejecting", '
            '"phase": "negotiation", '
            '"actionability": "boundary", '
            '"action": "close", '
            '"closure_readiness": "boundary_close", '
            '"reasoning": "The user has set a clear boundary'
        )

        plan = self.planner.parse_output(raw, fallback_phase="recommendation")

        assert plan["action"] == "close"
        assert plan["accepted_phase"] == "negotiation"
        assert plan["user_intent"] == "rejecting"
        assert plan["actionability"] == "boundary"
        assert plan["closure_readiness"] == "boundary_close"
        assert plan["parse_warning"]

    def test_partial_planner_json_preserves_non_ascii_text(self):
        raw = (
            '{"intent_summary": "사용자가 추가 제안을 거절했다.", '
            '"user_intent": "rejecting", '
            '"phase": "negotiation", '
            '"actionability": "boundary", '
            '"action": "close", '
            '"closure_readiness": "boundary_close", '
            '"reasoning": "사용자 경계가 명확하다'
        )

        plan = self.planner.parse_output(raw, fallback_phase="exploration")

        assert plan["intent_summary"] == "사용자가 추가 제안을 거절했다."
        assert plan["reasoning"] == ""
        assert plan["action"] == "close"
        assert plan["actionability"] == "boundary"

    def test_parse_failure_falls_back_without_retry_contract(self):
        plan = self.planner.parse_output("{", fallback_phase="recommendation")

        assert plan["action"] == "inquire"
        assert plan["accepted_phase"] == "recommendation"
        assert plan["actionability"] == "insufficient"
        assert plan["closure_readiness"] == "not_ready"
        assert plan["confidence"] == 0.0

    def test_omits_dialogue_state_sections_when_no_signal_is_available(self):
        from code_interactive.agents.history_adapter import build_shared_history

        history = build_shared_history(
            [],
            "I had chicken and rice.",
            context_window=10,
            state=None,
        ).history

        messages = self.planner.get_messages(
            history=history,
            turn_idx=0,
            current_phase="exploration",
        )
        user_prompt = messages[-1]["content"]

        assert "[Alignment State]" not in user_prompt
        assert "[Uncertainty State]" not in user_prompt

    def test_dialogue_state_prompt_can_be_fully_ablated(self):
        from code_interactive.agents.modules.dialogue_planner import DialoguePlanner

        cfg = AgentConfig(
            dialogue_planner_use_state_scores=False,
            dialogue_planner_use_state_rationales=False,
        )

        prompt = DialoguePlanner("lean_protein", cfg)._system_prompt

        assert "STATE DEFINITIONS" not in prompt
        assert "STATE SCORE INSTRUCTIONS" not in prompt
        assert "STATE RATIONALE INSTRUCTIONS" not in prompt

    def test_prompt_treats_user_initiative_as_planning_commitment(self):
        prompt = self.planner._system_prompt

        assert "user's initiative as a planning commitment" in prompt
        assert "Do not turn a user's feasibility boundary" in prompt
        assert "easy, doable, preferred, or good enough" in prompt


class TestInformationSeekerPrompt:
    def test_prompt_never_reopens_rejected_or_unavailable_options(self):
        from code_interactive.agents.prompts.roles.information_seeker import (
            INFORMATION_SEEKER_SYSTEM_PROMPT,
        )

        prompt = INFORMATION_SEEKER_SYSTEM_PROMPT

        assert "Do not ask about targets listed as rejected or unavailable" in prompt
        assert "constrained set" in prompt
        assert "follow the interaction-state evidence" in prompt


class TestMealRecommenderParsing:
    def test_invalid_recommendation_json_retries_to_valid_bundle(self, cfg):
        from code_interactive.agents.modules.meal_recommender import MealRecommender

        recommender = MealRecommender("lean_protein", cfg)
        raw = (
            '{"recommendation_type":"modify",'
            '"target_food":"side",'
            '"suggestion":"truncated'
        )
        calls = []

        def reinvoke(messages):
            calls.append(messages)
            return (
                '{"recommendation_type":"modify",'
                '"target_food":"side dish",'
                '"suggestion":"choose an egg-free vegetable fried-rice-style side",'
                '"reasoning":"It keeps the side similar while avoiding egg.",'
                '"expected_impact":"medium",'
                '"options":[{'
                '"option_id":"opt1",'
                '"target_food":"side dish",'
                '"suggestion":"choose an egg-free vegetable fried-rice-style side",'
                '"reasoning":"It preserves the requested side style without egg.",'
                '"expected_impact":"medium"'
                '}]}'
            )

        rec = recommender.parse_with_retry(
            base_msgs=[{"role": "system", "content": "Return JSON."}],
            raw_output=raw,
            reinvoke_fn=reinvoke,
            turn_idx=3,
        )

        assert calls
        assert rec["suggestion"] == "choose an egg-free vegetable fried-rice-style side"
        assert rec["options"][0]["suggestion"] == rec["suggestion"]
        assert "parse error" not in rec["reasoning"].lower()


class TestInteractionStateRepair:
    def test_latest_user_position_and_bundle_ordinal_acceptance_are_repaired(self):
        from code_interactive.agents.contracts import CoachingState
        from code_interactive.agents.engine import ConversationEngine

        prior = CoachingState(
            recommendation_history=(
                {
                    "turn_idx": 4,
                    "options": [
                        {
                            "option_id": "opt1",
                            "target_food": "fried rice",
                            "suggestion": "make it light-oil",
                            "expected_impact": "medium",
                        },
                        {
                            "option_id": "opt2",
                            "target_food": "fried rice",
                            "suggestion": "reduce rice",
                            "expected_impact": "medium",
                        },
                        {
                            "option_id": "opt3",
                            "target_food": "shrimp",
                            "suggestion": "use plain unbreaded shrimp",
                            "expected_impact": "medium",
                        },
                    ],
                },
            ),
        )
        stale_state = (
            "Open questions:\n"
            "- Which adjustment sounds doable?\n"
            "Latest user position:\n"
            "- The user asked what egg-free fried rice means.\n"
            "Active issue:\n"
            "- Explain egg-free fried rice."
        )

        repaired, meta = ConversationEngine._repair_interaction_state(
            interaction_state=stale_state,
            current_message="The first and third options are doable today.",
            prior_state=prior,
        )

        assert "The user said: The first and third options are doable today." in repaired
        assert "opt1: make it light-oil" in repaired
        assert "opt3: use plain unbreaded shrimp" in repaired
        assert "Which adjustment sounds doable?" not in repaired
        assert "bundle_ordinal_acceptance_resolved" in meta["repairs"]

    def test_recommendation_prompts_define_parallel_adjustment_semantics(self):
        from code_interactive.agents.prompts.roles.meal_recommender import (
            RECOMMENDER_SYSTEM_PROMPT,
        )
        from code_interactive.agents.prompts.roles.response_generator import (
            RESPONSE_RECOMMENDATION_SYSTEM_PROMPT,
        )

        assert "parallel default bundle" in RECOMMENDER_SYSTEM_PROMPT
        assert "not a menu of" in RECOMMENDER_SYSTEM_PROMPT
        assert "mutually exclusive alternatives" in RESPONSE_RECOMMENDATION_SYSTEM_PROMPT
        assert "Do NOT end with a question" in RESPONSE_RECOMMENDATION_SYSTEM_PROMPT
        assert "default bundle" in RESPONSE_RECOMMENDATION_SYSTEM_PROMPT

    def test_empty_meal_update_does_not_erase_existing_meal_base(self):
        from code_interactive.agents.contracts import CoachingState
        from code_interactive.agents.engine import ConversationEngine

        prior = CoachingState(
            meal_base="- Food items: jajangmyeon, egg-free shrimp fried rice",
            tracker_state="[Tracking State]\n- Confirmed food items: jajangmyeon, egg-free shrimp fried rice",
        )
        parsed = {
            "tracker_state": "[Tracking State]\n- Confirmed food items: none",
            "meal_base": "- Food items: not yet mentioned\n- Ingredients: not yet mentioned",
        }

        protected = ConversationEngine._protect_meal_state_from_empty_update(
            parsed_meal=parsed,
            prior_state=prior,
        )

        assert protected["meal_base"] == prior.meal_base
        assert protected["tracker_state"] == prior.tracker_state


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
