# Micro-Coaching Interactive Simulator

A browser-based conversational interface for the Micro-Coaching system.  
Two operating modes are available from the landing screen:

| Mode | Description |
|------|-------------|
| **Simulating Chat** | Fully autonomous AI-vs-AI conversation. The AI Coach guides a generated AI User toward a nutritional goal. Requires a meal description as context. |
| **Custom Chat** | You interact directly with the AI Coach. Select a goal and meal type, then respond to the coach's questions in real time. |

Both modes support an optional **Judge AI** overlay that evaluates each user reply for goal-alignment and displays a per-message chip (`✓ Goal Aligned` / `✗ Not Aligned`).

---

## Quick Start

```bash
# 1. Activate the environment
conda activate micro-coaching-chatbot

# 2. Launch from code_interactive/
cd micro-coaching-simulator/code_interactive
./start.sh              # production, port 8000
./start.sh 8080         # custom port
DEV=1 ./start.sh        # development mode with hot-reload

# 3. Open the UI
#    http://127.0.0.1:8000
```

The server prints `[Startup] Model loaded. Server ready.` once the LLM is loaded.  
The UI will show a banner while the model initialises and auto-dismiss it when ready.

> **Tip — Changing conversation settings:** edit `code/config.py` (`SimulationConfig`)
> and restart the server. `max_turns`, `context_window`, `stall_exit_turns`, and all
> related parameters are read from there — no need to touch `config_interactive.py`.

---

## Directory Layout

```
code_interactive/
├── app.py                 FastAPI application & HTTP endpoints
├── session_manager.py     Per-session state, LLM orchestration, Judge logic
├── models/
│   ├── coach.py           AI Coach model wrapper
│   ├── judge.py           Judge AI (goal-alignment evaluator)
│   └── user.py            AI User model wrapper (simulation mode)
├── utils/                 Shared utilities (prompts, parsers, …)
├── templates/
│   └── index.html         Single-page application (4-screen SPA)
└── static/
    ├── style.css          UI styles
    └── script.js          Frontend logic
```

---

## API Reference

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET`  | `/api/status` | Model readiness check |
| `GET`  | `/api/goals`  | Available nutritional goals |
| `POST` | `/api/session/start` | Create a new session (both modes) |
| `POST` | `/api/session/{id}/turn` | Submit one user reply (custom mode) |
| `POST` | `/api/session/{id}/sim-step` | Advance one simulation step (sim mode) |
| `DELETE` | `/api/session/{id}` | Terminate and clean up a session |

### `POST /api/session/start` — request body

```json
{
  "mode":             "simulation",
  "judge_enabled":    true,
  "nutrition_goal":   "half_fruits_vegetables",
  "meal_type":        "lunch",
  "meal_description": "Grilled chicken sandwich with fries and a cola.",
  "meal_ingredient":  "chicken, bread, fries, cola"
}
```

- `mode`: `"simulation"` or `"custom"` (default: `"custom"`)
- `meal_description` is **required** for simulation mode, ignored for custom mode.

---

## Judge AI

When `judge_enabled` is `true`, the Judge evaluates each user turn and returns:

- `judge_aligned` — `true` | `false` | `null` (pending on first turn)
- `aligned_label` — short human-readable label

The UI renders this as a coloured chip beneath the user/AI-User bubble.

---

## Conversation Quality — Design Notes

### Memory & Context

| Component | Mechanism |
|-----------|----------|
| **Shared history** (`core/memory.py`) | `context_window` (default 5) keeps only the most recent N turns in the LLM context. Older turns are replaced by a rolling summary. |
| **Coach prev-question list** (`models/coach.py`) | At every turn, `history.get_all_coach_questions()` extracts the **complete** list of Coach utterances directly from history and injects it under `[Questions you have ALREADY asked]`. Unlike `own_buffer`, this list is guaranteed complete even after context-window sliding. |
| **User own-buffer** (`models/user.py`) | Actual meal-info answers are recorded so the User never repeats previously shared details. **Non-answer responses** ("I'm not sure", etc.) are excluded to avoid polluting the buffer. |
| **Bug fix — sliding window** (`core/memory.py`) | `build_messages()` previously used loop index `i == 0` to skip Coach's turn-0 utterance. After sliding, `i == 0` could refer to turn 5, 6, …, causing those utterances to be silently dropped. Fixed to `turn.turn_idx == 0`. |

### Incremental Summarisation

`summarize_conversation()` now accepts an optional `prev_summary` argument.
When called for a rolling update, it receives the previous summary plus only
the **new turns since the last summary** (`history.to_plain_text_from(last_summarized_start)`).
This produces a richer, cumulative summary instead of re-summarising the full history each time.
The final summary on session close follows the same pattern.

### Natural Conversation End

The AI User appends `"That's all about my meal."` as a TERMINATION_TOKEN when it has no more information to share.  
Two guards prevent premature termination:

1. **`min_natural_end_turn`** (default 3) — any TERMINATION_TOKEN emitted before this turn is silently stripped and the conversation continues.
2. **`closing_for_natural` flag** — when a valid TERMINATION_TOKEN is detected, the session does **not** terminate immediately. Instead the Coach receives a `[CLOSING INSTRUCTION]` block and generates one warm closing sentence. The session terminates only after that message is shown.

If the token-only reply leaves `user_reply_clean` empty, a safe default sentence is substituted so no empty bubble appears in the UI.

### Dead-End Topic Injection

When AI User replies with a non-answer to a specific question, the Coach's next system prompt receives a `[Topics the user already said they are NOT SURE about]` block listing every such question. This tells the Coach to move on instead of re-asking the same topic.

### Stall Detection & Graceful Exit

If the AI User produces `stall_exit_turns` (default **3**) consecutive non-answers:

1. The Coach receives a **`[CLOSING INSTRUCTION]`** block asking it to generate a warm closing sentence.
2. If the Coach's output still contains a `?`, it is replaced by the safe fallback closing sentence.
3. The session is marked `terminated` with `terminated_by = "stall_exit"`.

### Generation Strategy

| Role | `sampling` | Rationale |
|------|-----------|----------|
| **Coach** | `greedy` | Deterministic — reduces likelihood of ignoring the "do not repeat" instruction. |
| **AI User** | `sampling` (temperature 0.7) | Stochastic — natural, varied conversational replies. |
| **Judge** | `greedy` | Deterministic — reproducible alignment verdicts. |

All three sampling modes are configurable in `code/config.py` (`coach_sampling`, `sampling`, `judge_sampling`).

### Configuration — Single Source of Truth

`config_interactive.py` reads the following fields from `code/config.py` (`SimulationConfig`) at startup:

| Field | Default | Description |
|-------|---------|-------------|
| `max_turns` | 10 | Safety ceiling on conversation length |
| `context_window` | 5 | Recent turns kept in LLM context |
| `summarize_every` | 3 | Rolling summary update interval (turns) |
| `stall_exit_turns` | 3 | Consecutive non-answers before graceful exit |
| `min_natural_end_turn` | 3 | Earliest turn AI User may emit TERMINATION_TOKEN |
| `coach_llm_repo` | `google/gemma-3-12b-it` | Shown as coach bubble label in UI |
| `user_llm_repo` | `google/gemma-3-12b-it` | Shown as AI User bubble label in UI |

**To change any of these, edit `code/config.py` and restart the server.**

### UI — Chat Bubble Labels

Coach and AI User bubble labels display the model repo short-name (e.g. `gemma-3-12b-it`)
rather than generic "Coach" / "AI User". These are fetched from `/api/status` on page load.
In **Custom Chat** mode the user bubble has no label.

`/api/status` response (when ready):
```json
{ "ready": true, "coach_label": "gemma-3-12b-it", "user_label": "gemma-3-12b-it" }
```

### UI — Judge AI Badge

The badge in the chat header is always visible and reflects the current setting:

| State | Colour | Dot animation |
|-------|--------|---------------|
| **On** | Green (primary) | Blinking |
| **Off** | Red (muted) | Static |

