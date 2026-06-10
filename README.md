# Crucible

**Crucible** is a self-improving adversarial agent that audits hotel reviews for authenticity. Built for the **Arize Track** of the Google Cloud Rapid Agent Hackathon.

It doesn't just classify reviews — it **reads its own observability traces back through the Arize Phoenix MCP server** and rewrites its detection rules, getting sharper every round.

## The Problem

Online reviews are heavily manipulated. Current detectors are a game of cat-and-mouse because bad actors quickly learn to bypass them. Static rule-based systems go stale within days.

## The Solution

Crucible pits a **Detector Agent** against a **Forger Agent** in a continuous adversarial loop. After each round, a **Reflector Agent** — built with the **Google Agent Development Kit (ADK)** — autonomously calls the **Arize Phoenix MCP server** to inspect the Detector's failure traces and writes a new, concrete detection rule that feeds back into the Detector.

### Architecture

```
        ┌─────────────────────────────────────────────┐
        │                                             ↓
   Forger Agent  ──fakes──▶  Detector Agent  ──▶  Grader
   (Gemini)                  (Gemini)              │ scores accuracy
        ▲                                          ▼
        │                              Reflector Agent (Google ADK)
        │                                          │
        └──────── new rule ◀───────────────────────┤
                                                    │ list-traces / get-trace-details
                                                    ▼
                                       Arize Phoenix  (MCP server)
                                       every LLM call traced via OpenTelemetry
```

The loop is **self-improving**: the Reflector's rules are appended to the Detector's prompt, and the Forger adapts to whatever the Detector has learned — an arms race that hardens the detector round over round.

### Agents

| Agent | Role | Powered by |
|-------|------|------------|
| **Detector** | Evaluates each review as real or fake with structured JSON reasoning | Gemini (`gemini-2.5-flash-lite`) |
| **Forger** | Generates adversarial fakes that exploit the Detector's current blindspots | Gemini (`gemini-2.5-flash-lite`) |
| **Reflector** | **Google ADK `LlmAgent`** that calls the Arize Phoenix MCP server to read failure traces and write new rules | Google ADK + Gemini + `@arizeai/phoenix-mcp` |

### How it meets the hackathon requirements

| Requirement | Where it lives |
|-------------|----------------|
| **Built with Gemini** | All three agents call Gemini |
| **Built with Google Agent Builder / ADK** | The Reflector is a real `google.adk.agents.LlmAgent` run via `InMemoryRunner` (`crucible/agent/reflector_adk.py`) |
| **Meaningful Partner MCP integration (Arize)** | The Reflector's `McpToolset` connects to `@arizeai/phoenix-mcp` and autonomously calls `list-traces` / `get-trace-details` |
| **Beyond chat — multi-step mission with tools** | Closed adversarial loop: generate → evaluate → grade → reflect (via MCP) → update → repeat |

### Key Features

- **Self-Improving Loop**: The Reflector reads the Detector's failure traces over MCP and rewrites its rules each iteration — no human in the loop.
- **Adversarial Training**: The Forger exploits the Detector's known blindspots to generate progressively harder fakes.
- **Full Observability**: Every LLM call (Forger, Detector, and the ADK Reflector itself) is traced with OpenTelemetry/OpenInference and exported to Arize Phoenix Cloud.
- **Concrete, de-duplicated rules**: The Reflector is constrained to emit one concrete, testable rule per round, and near-duplicates are filtered out.
- **Structured Output**: The Detector returns JSON with both a verdict and reasoning for every review.
- **Production Retry Logic**: All Gemini calls use `tenacity` with exponential backoff to handle rate limits gracefully.

## Live Demo (single-page web app)

A single landing page tells the story end to end and lets judges run the loop themselves:

```bash
pip install -r requirements.txt
cp .env.example .env          # add your keys (see below)
uvicorn crucible.app.main:app --host 127.0.0.1 --port 8000
# open http://localhost:8000
```

The page has three sections:
1. **Overview** — the problem, the stack (Gemini · Google ADK · Arize Phoenix MCP), and the architecture.
2. **Run it live** — press one button to run three rounds. Watch the three agents light up, the metrics update, the forged fakes appear, and the rules the Reflector learns.
3. **The Result** — an accuracy chart (Benchmark accuracy + Adversarial catch rate per round) that **fills in live** from the real run.

## Quick Start (CLI)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

> Node.js is also required — the Reflector spawns the Arize Phoenix MCP server via `npx @arizeai/phoenix-mcp`.

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys:
# - GOOGLE_API_KEY (or GEMINI_API_KEY) from Google AI Studio
# - PHOENIX_API_KEY from https://app.phoenix.arize.com
# - PHOENIX_HOST / PHOENIX_COLLECTOR_ENDPOINT (your Phoenix space endpoint,
#   e.g. https://app.phoenix.arize.com/s/<your-space>)
```

### 3. Run the Self-Improving Pipeline

```bash
python crucible/loop/eval.py
```

### 4. View Traces in Arize Phoenix

Open your Phoenix dashboard (project `crucible`) to see every LLM call, verdict, and reasoning — including the ADK Reflector agent calling the MCP tools — traced in real time.

### 5. (Optional) Verbose Mode

```bash
CRUCIBLE_VERBOSE=1 python crucible/loop/eval.py
```

## Project Structure

```
crucible/
  agent/
    detector.py        # Detector Agent (evaluates reviews)
    reflector_adk.py   # Reflector Agent — Google ADK LlmAgent + Arize Phoenix MCP toolset
    prompts.py         # System prompts (Detector, Forger, Reflector)
    llm_client.py      # Gemini client with tenacity retry logic
    tools.py           # Heuristic analysis helpers
  adversary/
    forger.py          # Forger Agent (generates adversarial fakes)
  obs/
    instrumentation.py # Phoenix / OpenTelemetry tracing setup
    mcp_client.py      # Phoenix MCP client helpers
  loop/
    eval.py            # Main self-improving pipeline (CLI + streaming for the web app)
  data/
    load_ott.py        # Ott Deceptive Opinion Spam loader (kagglehub + fallback sample)
  app/
    main.py            # FastAPI web interface (single-page brief + live console + chart)
```

## Dataset

Uses the [Ott Deceptive Opinion Spam dataset](https://www.kaggle.com/datasets/rtatman/deceptive-opinion-spam-corpus) — 1,600 real and fake hotel reviews for Chicago hotels. If Kaggle credentials aren't available, the loader falls back to a small embedded sample so the pipeline always runs.

## License

Apache-2.0 License
