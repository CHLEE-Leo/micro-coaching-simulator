# Micro-Coaching Simulator

> **"[Paper Title]"** (under review)

A research framework for automatically generating and evaluating nutritional coaching dialogues, and for experimenting with real-time interactions between actual users and an AI coach.

---

## Purpose

This repository supports research on **Nutritional Goal Alignment** through two core capabilities:

1. **Dialogue Data Generation** — Three LLM agents (Coach, User, Judge) collaborate to automatically produce large-scale synthetic coaching conversations over a crowdsourced meal dataset.
2. **Real-Time Interaction Experiments** — A real person converses with an AI coach while the Judge AI evaluates goal alignment after every utterance.

Both capabilities share the same model architecture and conversation design, but differ in **purpose and operating mode**.

---

## Side-by-Side Comparison

|  | `code/` | `code_interactive/` |
|---|---|---|
| **Role** | Batch simulation — generates experimental data for the paper | Web-based real-time coaching UI |
| **User** | None — three LLMs converse autonomously | A real person chats directly with the Coach |
| **Execution** | Python script / shell batch job | FastAPI server + browser UI |
| **Judge usage** | Termination condition for the conversation | Real-time alignment chip shown after each turn |
| **Purpose** | Synthetic dataset generation, offline analysis | User-experience experiments, prototype validation |
| **Entry point** | `code/run_simulation.py` | `code_interactive/app.py` |

---

## `code/` — Batch Simulation

### Role & Intent

Takes crowdsourced meal data as input and automatically runs repeated **Coach → User → Judge** three-way conversations.  
The goal is to efficiently build large-scale synthetic coaching dialogue datasets for paper experiments.

### Two-Phase Conversation Design

```
Turn 0   Coach: "What did you have for dinner?"
         User : Reveals ALL food item names at once (meal_description)

Turn 1+  Coach: Drills into each food — ingredients, cooking method, portion size
         User : Progressively reveals detailed information (meal_ingredient)
         Judge: Evaluates nutritional goal alignment after every turn
                → Terminates when confidence is reached
```

This design prevents the Judge from receiving all information at once and forces the **evidence accumulation process** to unfold naturally within the coaching dialogue.

### Overview

- `config.py` (`SimulationConfig`) — **Single source of truth** for all experiment parameters  
  _(Editing this file is reflected in both batch simulation and the web server)_
- `core/memory.py` — Shared conversation history, sliding window, rolling summary
- `core/simulation.py` — Single-dialogue and batch conversation loop runner
- `models/` — Coach, User, and Judge model wrappers
- `utils/` — LLM loading/generation/summarisation, data I/O

> For full details, see **[code/README.md](code/README.md)** _(or [code/README(eng).md](code/README(eng).md))_.

---

## `code_interactive/` — Real-Time Interaction UI

### Role & Intent

Provides an environment where a real person converses with an AI coach.  
A browser-based SPA (Single-Page Application) supporting two operating modes:

| Mode | Description |
|------|-------------|
| **Simulating Chat** | Fully autonomous AI-vs-AI conversation — visually observing batch simulation in the browser |
| **Custom Chat** | A real person chats directly with the AI Coach — enters goal and meal info, then responds to the coach's questions |

Enabling the Judge AI overlay displays a `✓ Goal Aligned` / `✗ Not Aligned` chip beneath each utterance.

### Quick Start

```bash
conda activate micro-coaching-chatbot
cd micro-coaching-simulator/code_interactive
./start.sh              # production, port 8000
./start.sh 8080         # custom port
DEV=1 ./start.sh        # development mode (hot-reload)
```

Open `http://127.0.0.1:8000` in your browser.

### Overview

- `app.py` — FastAPI server, HTTP endpoint definitions
- `session_manager.py` — Per-session state management, LLM orchestration
- `config_interactive.py` — Reads values from `code/config.py` and applies them to the server  
  _(To change conversation parameters, edit `code/config.py` only, then restart the server)_
- `templates/index.html` + `static/` — 4-screen SPA UI
- `models/` — Batch model classes adapted for interactive mode

> For full details, see **[code_interactive/README.md](code_interactive/README.md)** _(or [code_interactive/README(eng).md](code_interactive/README(eng).md))_.

---

## Single Source of Truth for Configuration

```
code/config.py  (SimulationConfig)
       │
       ├──▶  code/run_simulation.py            (batch simulation)
       └──▶  code_interactive/app.py           (web server — auto-loaded at startup)
```

**`max_turns`, `context_window`, `stall_exit_turns`, `min_natural_end_turn`, and all other**  
conversation-control parameters are managed in one place: `code/config.py`.

---

## Supported Nutritional Goals

| Goal key | Description |
|----------|-------------|
| `lean_protein` | Meal centred on lean protein sources (chicken, fish, legumes, …) |
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
