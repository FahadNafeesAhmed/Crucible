# Crucible

**Crucible** is a self-improving adversarial agent that audits hotel reviews for authenticity. Built for the **Arize Track** of the Google Cloud Rapid Agent Hackathon.

## The Problem

Online reviews are heavily manipulated. Current detectors are a game of cat-and-mouse because bad actors quickly learn to bypass them. Static rule-based systems go stale within days.

## The Solution

Crucible pits a **Detector Agent** against a **Forger Agent** in a continuous adversarial loop. The system learns from its own failures using **Arize Phoenix** observability and a **Reflector Agent** that rewrites detection rules in real-time.

### Architecture

```
Forger Agent (Gemini)        Reflector Agent (Gemini)
    |                              ^
    | generates fake reviews       | analyzes failure traces
    v                              |
Detector Agent (Gemini) -----> Grader -----> Phoenix MCP
    |                              |
    | evaluates reviews            | logs traces to cloud
    v                              v
  Verdicts + Reasoning        Arize Phoenix Dashboard
```

### Agents

| Agent | Role | Model |
|-------|------|-------|
| **Detector** | Evaluates reviews as real or fake with structured reasoning | `gemini-2.5-flash-lite` |
| **Forger** | Generates adversarial fake reviews that exploit the Detector's blindspots | `gemini-2.5-flash-lite` |
| **Reflector** | Analyzes failure traces from Phoenix and generates new detection rules | `gemini-2.5-flash-lite` |

### Key Features

- **Self-Improving Loop**: The Reflector reads the Detector's failure traces and dynamically rewrites its prompt rules each iteration.
- **Adversarial Training**: The Forger exploits the Detector's known blindspots to generate progressively harder fake reviews.
- **Full Observability**: Every LLM call is traced with OpenTelemetry and exported to Arize Phoenix Cloud.
- **Structured Output**: The Detector returns JSON with both a verdict and reasoning for every review.
- **Production Retry Logic**: All Gemini calls use `tenacity` with exponential backoff to handle rate limits gracefully.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys:
# - GOOGLE_API_KEY (from Google AI Studio)
# - PHOENIX_API_KEY (from https://app.phoenix.arize.com)
# - PHOENIX_COLLECTOR_ENDPOINT (your Phoenix project endpoint)
```

### 3. Run the Self-Improving Pipeline

```bash
python crucible/loop/eval.py
```

### 4. View Traces in Arize Phoenix

Open your Phoenix dashboard to see every LLM call, verdict, and reasoning traced in real-time.

### 5. (Optional) Verbose Mode

To dump raw OpenTelemetry JSON spans to the terminal:

```bash
CRUCIBLE_VERBOSE=1 python crucible/loop/eval.py
```

## Project Structure

```
crucible/
  agent/
    detector.py      # Detector Agent (evaluates reviews)
    prompts.py       # All system prompts (Detector, Forger, Reflector)
    tools.py         # Heuristic analysis tools
  adversary/
    forger.py        # Forger Agent (generates fakes)
  obs/
    instrumentation.py  # Phoenix/OTel tracing setup
    mcp_client.py       # Reflector Agent + MCP client
    evals.py            # Evaluation metrics
  loop/
    eval.py          # Main self-improving pipeline
  data/
    load_ott.py      # Ott Deceptive Opinion Spam loader
  app/
    main.py          # FastAPI web interface
```

## Dataset

Uses the [Ott Deceptive Opinion Spam dataset](https://www.kaggle.com/datasets/rtatman/deceptive-opinion-spam-corpus) — 1,600 real and fake hotel reviews for Chicago hotels.

## License

Apache-2.0 License
