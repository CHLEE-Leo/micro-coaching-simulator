# Micro-Coaching Interactive App

Self-contained FastAPI web app for running the micro-coaching chatbot.

## Setup

Using `venv`:

```bash
cd code_interactive
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Or using `conda`:

```bash
cd code_interactive
conda env create -f environment.yml
conda activate micro-coaching-interactive
cp .env.example .env
```

Edit `.env` and set `OPENAI_API_KEY`.

## Included Runtime Data

The app includes the small goal/workflow resources required at runtime:

```text
agents/data/additional/
```

Large reference PDFs and unused research data are not required to run the app.

## Run

```bash
./start.sh
```

Open <http://localhost:8000>.

Use a custom port:

```bash
./start.sh 8080
```

Use hot reload during development:

```bash
DEV=1 ./start.sh
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest tests/unit
```

The end-to-end tests require the web server to be running.
