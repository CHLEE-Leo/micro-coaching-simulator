"""
LLM Pipeline Unit Tests — Phase 1
──────────────────────────────────
개별 에이전트 parse 함수의 입출력 검증.
LLM 호출 없이 순수 파싱 로직만 테스트한다.
"""

import sys
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[3]          # micro-coaching-simulator/
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
# 1. Orchestrator — parse_routing()
# ═══════════════════════════════════════════════════════════════════════════════
class TestParseRouting:
    @pytest.fixture(autouse=True)
    def setup(self, cfg):
        from code_interactive.agents.modules.orchestrator import Orchestrator
        self.orch = Orchestrator(nutrition_goal="lean_protein", config=cfg)

    # U-R1: 정상 JSON
    def test_normal_json(self):
        raw = '{"action":"inquire","user_intent":"informing","reasoning":"need more info","instruction":"ask about portion"}'
        result = self.orch.parse_routing(raw, turn_idx=2, phase="exploration")
        assert result["action"] == "inquire"

    # U-R2: Markdown 코드블록
    def test_markdown_code_block(self):
        raw = '```json\n{"action":"recommend","user_intent":"accepting","reasoning":"ready","instruction":"suggest"}\n```'
        result = self.orch.parse_routing(raw, turn_idx=5, phase="recommendation")
        assert result["action"] == "recommend"

    # U-R3: 유효하지 않은 action → fallback
    def test_invalid_action_fallback(self):
        raw = '{"action":"dance","user_intent":"","reasoning":"","instruction":""}'
        result = self.orch.parse_routing(raw, turn_idx=2, phase="exploration")
        assert result["action"] == "inquire"  # phase fallback

    # U-R4: phase-action 조합은 rigid rule이 아니라 orchestrator 판단으로 둔다
    def test_valid_action_keeps_orchestrator_choice(self):
        raw = '{"action":"recommend","user_intent":"","reasoning":"","instruction":""}'
        result = self.orch.parse_routing(raw, turn_idx=2, phase="exploration")
        assert result["action"] == "recommend"
        assert result["accepted_phase"] == "exploration"

    # U-R5: JSON 파싱 실패 → fallback
    def test_json_parse_failure(self):
        raw = "this is not json at all"
        result = self.orch.parse_routing(raw, turn_idx=2, phase="exploration")
        assert result["action"] == "inquire"  # fallback

    # U-R6: 빈 문자열 → fallback
    def test_empty_string(self):
        raw = ""
        result = self.orch.parse_routing(raw, turn_idx=2, phase="recommendation")
        assert result["action"] == "inquire"

    # U-R7: invalid JSON/action fallback은 phase와 무관하게 safe inquiry로 간다.
    @pytest.mark.parametrize("phase,expected_fallback", [
        ("exploration", "inquire"),
        ("recommendation", "inquire"),
        ("negotiation", "inquire"),
        ("motivational_ending", "inquire"),
    ])
    def test_phase_fallbacks(self, phase, expected_fallback):
        raw = '{"action":"INVALID"}'
        result = self.orch.parse_routing(raw, turn_idx=2, phase=phase)
        assert result["action"] == expected_fallback
        assert result["accepted_phase"] == phase

    # U-R8: action vocabulary에 포함된 값은 phase별 hard-coded table 없이 통과
    @pytest.mark.parametrize("phase,action", [
        ("exploration", "inquire"),
        ("exploration", "assess"),
        ("exploration", "terminate"),
        ("recommendation", "inquire"),
        ("recommendation", "recommend"),
        ("recommendation", "close"),
        ("negotiation", "respond"),
        ("negotiation", "recommend"),
        ("motivational_ending", "close"),
    ])
    def test_allowed_actions_pass(self, phase, action):
        raw = f'{{"action":"{action}","user_intent":"informing","reasoning":"ok","instruction":"x"}}'
        result = self.orch.parse_routing(raw, turn_idx=2, phase=phase)
        assert result["action"] == action


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

    # U-MR2: JSON 파싱 실패 → fallback
    def test_parse_failure(self):
        raw = "not json"
        result = self.rec.parse_output(raw)
        assert result["recommendation_type"] == "modify"  # default fallback


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
            client_pool={"gpt-5.2": FakeClient(), "gpt-5.4-mini": FakeClient()},
        )
        raw = service.run_module_inference(
            module="orchestrator",
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


class TestPhasePredictorPrompt:
    def _system_prompt(self, cfg):
        from code_interactive.agents.modules.phase_predictor import PhasePredictor

        return PhasePredictor("lean_protein", cfg)._system_prompt

    def test_uses_scores_and_rationales(self):
        cfg = AgentConfig(
            phase_predictor_use_state_scores=True,
            phase_predictor_use_state_rationales=True,
        )

        prompt = self._system_prompt(cfg)

        assert "alignment/uncertainty scores and rationales" in prompt
        assert "No dialogue state scores or rationales are provided" not in prompt

    def test_uses_scores_only(self):
        cfg = AgentConfig(
            phase_predictor_use_state_scores=True,
            phase_predictor_use_state_rationales=False,
        )

        prompt = self._system_prompt(cfg)

        assert "alignment/uncertainty scores, user intent" in prompt
        assert "alignment/uncertainty rationales" not in prompt

    def test_uses_rationales_only(self):
        cfg = AgentConfig(
            phase_predictor_use_state_scores=False,
            phase_predictor_use_state_rationales=True,
        )

        prompt = self._system_prompt(cfg)

        assert "alignment/uncertainty rationales, user intent" in prompt
        assert "alignment/uncertainty scores" not in prompt

    def test_uses_no_dialogue_state_evidence(self):
        cfg = AgentConfig(
            phase_predictor_use_state_scores=False,
            phase_predictor_use_state_rationales=False,
        )

        prompt = self._system_prompt(cfg)

        assert "No dialogue state scores or rationales are provided" in prompt
        assert "alignment/uncertainty scores and rationales" not in prompt


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
