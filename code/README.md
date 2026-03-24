# Micro-Coaching Simulator

> Code repository for **"[Paper Title]"** (under review).

A **vLLM-based multi-turn dialogue simulation framework** for generating synthetic nutritional micro-coaching conversations. Provides two operation modes:

| Mode | Description | Entry point |
|---|---|---|
| **Batch simulation** | Automated simulation over a crowdsourced meal dataset (Coach LLM + User LLM + Judge LLM) | `code/run_simulation.py` |
| **Interactive coaching** | Real user chats with a Coach LLM via a web UI; Judge evaluates alignment in real time | `code_interactive/app.py` |

---

## Two-Phase Conversation Design

```
Phase 1 – Turn 0
  Coach: "What are you having for dinner?"
  User:   names ALL food items at once (meal_description)

Phase 2 – Turn 1 onward
  Coach: drills into each food → ingredients, cooking method, portion
  User:  reveals ingredient/preparation details (meal_ingredient)
  Judge: evaluates nutritional goal alignment after each turn
         → terminates when pred_alignment == true_alignment
```

The separation between `meal_description` (food names, revealed at turn 0) and `meal_ingredient` (preparation/ingredient details, revealed progressively) ensures the Judge accumulates evidence turn-by-turn rather than receiving all information at once.

---

## Repository Structure

```
micro-coaching-simulator/
├── code/                       # Batch simulation (automated, dataset-driven)
│   ├── config.py               # SimulationConfig dataclass — edit here, no CLI
│   ├── run_simulation.py       # Entry point
│   ├── run_simulation.sh       # Shell script with GPU selection
│   ├── core/
│   │   ├── memory.py           # ConversationBuffer, SharedConversationHistory
│   │   └── simulation.py       # simulate_conversation() / simulate_conversations_batch()
│   ├── models/
│   │   ├── coach.py            # CoachModel
│   │   ├── user.py             # UserModel (LLM-simulated user)
│   │   └── judge.py            # JudgeModel
│   └── utils/
│       ├── llm_utils.py        # load_model, batch_generate, summarize_conversation
│       └── io_utils.py         # load_meal_data, incremental JSON save
│
├── code_interactive/           # Interactive web UI (real user ↔ Coach LLM)
│   ├── app.py                  # FastAPI server (session management + LLM inference)
│   ├── config_interactive.py   # Interactive-mode config
│   ├── session_manager.py      # Per-session state (history, coach, judge)
│   ├── static/
│   │   ├── style.css           # UI styles
│   │   └── script.js           # Chat logic
│   ├── templates/
│   │   └── index.html          # Single-page app
│   ├── requirements.txt
│   └── start.sh                # Launch script
│
└── data/
    ├── df_normal_without_test_string.csv
    └── additional/             # goal_def.json, expert_workflow.json, output_format_inst_*.txt
```

---

## Batch Simulation (code/)

Edit `config.py`, then run:

```bash
bash code/run_simulation.sh 6,7
# or
python code/run_simulation.py
```

Key config fields:

```python
goal                  = "lean_protein"   # lean_protein | half_fruits_vegetables | one_fourth_carbs | drink_water
coach_llm_repo        = "google/gemma-3-12b-it"
user_llm_repo         = "google/gemma-3-12b-it"
judge_llm_repo        = "google/gemma-3-12b-it"
max_turns             = 10   # ⚠️  웹 UI 서버도 이 값을 읽습니다
judge_min_turn        = 3
batch_mode            = True
num_gpus              = 2
coach_sampling        = "greedy"   # Coach: greedy / User: sampling / Judge: greedy
stall_exit_turns      = 3         # consecutive non-answers before graceful exit
min_natural_end_turn  = 3         # earliest turn AI User may emit TERMINATION_TOKEN
```

> **Single source of truth** — `code/config.py` (`SimulationConfig`) is read by both the
> batch pipeline and the interactive web server (`code_interactive/`). Changing values
> here and restarting the server is all that is needed.

Output JSON schema per dialogue:

```json
{
  "id": 42,
  "meal_description": "Grilled chicken salad",
  "turns": [{"turn_idx": 0, "coach_utterance": "...", "user_utterance": "..."}],
  "terminated_by": "judge",
  "pred_alignment": true,
  "pred_score": 1.0,
  "true_alignment": true,
  "alignment_correct": true,
  "alignment_history": [{"turn_idx": 3, "aligned": true, "score": 1.0}]
}
```

---

## Interactive Web UI (code_interactive/)

A real user replaces the User LLM. The Coach LLM asks questions; the Judge LLM evaluates alignment in the background.

```bash
cd code_interactive
pip install -r requirements.txt
./start.sh           # launches FastAPI on http://localhost:8000
# DEV=1 ./start.sh  # hot-reload mode for development
```

Workflow:
1. User enters **nutrition goal**, **meal name(s)**, and **ingredients** in the setup form.
2. Coach LLM asks the first question automatically.
3. User types replies in the chat interface.
4. After each reply the Judge evaluates and updates the alignment indicator.
5. Session ends when the Judge is confident or `max_turns` is reached.

### `memory.py` additions (`core/`)

| Method | Description |
|--------|-------------|
| `get_all_coach_questions()` | Returns every Coach utterance recorded in history — guaranteed-complete list used in Coach prompt to prevent question repetition |
| `to_plain_text_from(from_turn_idx)` | Serialises only turns at or after `from_turn_idx`; used for incremental summarisation |

---

## Supported Nutritional Goals

| Goal | Description |
|---|---|
| `lean_protein` | Meal centred on lean protein sources (chicken, fish, eggs, legumes, …) |
| `half_fruits_vegetables` | Half the plate filled with fruits and/or vegetables |
| `one_fourth_carbs` | One quarter of the plate consisting of complex carbohydrates |
| `drink_water` | Primary beverage is water |

---

## Citation

```bibtex
@misc{micro-coaching-simulator-2026,
  author = {},
  title  = {},
  year   = {2026},
  url    = {}
}
```
