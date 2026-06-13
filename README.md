# Crucible

> A self-improving review detector whose memory lives inside Arize Phoenix.

Crucible is a multi-agent system that audits hotel reviews for authenticity. Three Gemini agents play an adversarial loop. A fourth ingredient, an **Arize Phoenix Dataset called `crucible-rules`**, is the agent's persistent memory. The Detector reads its rules from that Dataset on every single inference. The Reflector, built on Google's Agent Development Kit, writes new rules to that Dataset whenever it learns. The agent literally cannot operate without the partner platform. Phoenix is not bolted-on observability. It is the substrate the agent thinks from.

**Built for the Arize track of the Google Cloud Rapid Agent Hackathon 2026.**
Made by **Fahad Ahmed**.

- **Live demo:** https://crucible-demo-1047616825991.us-central1.run.app
- **3-minute walkthrough:** https://www.youtube.com/watch?v=SS_KZREWGwc
- **License:** Apache-2.0

---

## The idea, in one paragraph

Online reviews are flooded with AI-written fakes that pass the Turing test. Static detectors go stale within days because forgers adapt around them. Crucible flips the loop: a Forger agent attacks the Detector, every Gemini call is captured as an OpenInference trace, and a Reflector agent reads those traces back through the Arize Phoenix MCP server, distills one new detection rule, and writes it into a Phoenix Dataset. The Detector reads that Dataset before every inference, so by round two it is already operating with the lesson it learned in round one. Every Detector call, including production audits of live Google Places reviews, hits Arize first.

## What it does

1. **Adversarial training loop.** Three Gemini agents fight in three rounds. The Forger generates hyper-realistic fakes. The Detector judges a mix of those fakes plus a held-out benchmark, returns structured JSON, and every call is traced into Arize Phoenix. The Reflector reads the failure traces over MCP, writes one new rule, and persists it to the Phoenix Dataset.
2. **Live audit of any hotel.** Type a hotel name. Crucible calls the Google Places API, pulls up to five real reviews, and judges each one with the rules currently stored in Arize. Disagree with a verdict, one click sends the failure to the Reflector, and a new rule lands in the Dataset within seconds. Re-audit and the verdict changes.
3. **Benchmarked against ground truth.** The held-out benchmark is the [Ott Deceptive Opinion Spam Corpus](https://www.kaggle.com/datasets/rtatman/deceptive-opinion-spam-corpus) (Cornell, 2011), 1,600 labelled Chicago hotel reviews. Every round reports two numbers: Benchmark Accuracy on Ott (no drift) and Adversarial Catch Rate (keeping up with the Forger).

## The agents

| Agent | Role | Stack |
|-------|------|-------|
| **Forger** | Generates adversarial fakes that exploit the Detector's current blindspots. | Gemini 2.5 Flash-Lite |
| **Detector** | Judges each review as real or fake with structured JSON reasoning. Rebuilds its prompt on every inference from rules pulled live from Arize. | Gemini 2.5 Flash-Lite + Pydantic |
| **Reflector** | Reads failure traces from Arize Phoenix over MCP, distills one new detection rule, writes it to the Phoenix Dataset. | Real Google ADK `LlmAgent` + `McpToolset` + `InMemoryRunner` + `@arizeai/phoenix-mcp` |

## Architecture

```
   Forger ──fakes──▶ Detector ──verdicts──▶ Grader
   (Gemini)         (Gemini)                │
       ▲                                    ▼
       │                       Reflector (Google ADK)
       │                                    │
       │            list-traces / get-trace │  via MCP
       │                                    ▼
       │                       Arize Phoenix Cloud
       │                                    │
       │            crucible-rules Dataset  │
       └────────── new rule ◀───────────────┘
                                            │
            (every Gemini call is auto-traced into Phoenix)
            (every Detector inference reads rules from Phoenix)
```

The loop closes on the partner platform. Phoenix is both the read substrate and the write target.

## How it meets the hackathon rubric

| Requirement | Where it lives |
|---|---|
| Built with Gemini | All three agents call `gemini-2.5-flash-lite`. |
| Built with Google Cloud Agent Builder / ADK | The Reflector is a real `google.adk.agents.LlmAgent` running through `InMemoryRunner` (`crucible/agent/reflector_adk.py`). |
| Meaningful partner MCP integration (Arize) | The Reflector's `McpToolset` spawns `@arizeai/phoenix-mcp` over stdio and autonomously calls `list-traces` and `get-trace`. |
| Beyond chat, multi-step mission with tools | Generate, evaluate, grade, reflect via MCP, persist rule to Phoenix Dataset, re-judge with the new rule. |
| Action with a human in control | Quarantine flagged reviews into a moderation queue with one-click approval. Disagree on any verdict to teach the agent live. |

## The Phoenix Dataset memory layer

This is the unusual part. Most Arize submissions use Phoenix as a passive logging backend. Crucible uses a **Phoenix Dataset as the agent's runtime memory**.

- **Read path** — `crucible/obs/phoenix_rules.py::render_blindspots()` calls `phoenix.client.datasets.get_dataset("crucible-rules")` on every Detector inference and injects the rows into the Detector's prompt as `ACCUMULATED DETECTION RULES`.
- **Write path** — `crucible/obs/phoenix_rules.py::add_rule()` calls `phoenix.client.datasets.add_examples_to_dataset(...)` whenever the Reflector distills a new rule. Each row has `inputs = {failure_text}`, `outputs = {rule}`, `metadata = {source: seed | judge_correction | reflector, ts}`.

The result: the Detector cannot operate if Arize is unreachable, because it has no internal rule state. The agent's brain literally lives on the partner platform.

## Live demo on the web

A single landing page tells the story end to end:

1. **The hook + the gap** — stats on fake review economics.
2. **The live arena** — one click runs three rounds. Three agent boxes light up. Live accuracy chart, live Phoenix Dataset memory chart, Detector Intelligence panel showing the rules currently in `crucible-rules`.
3. **The live audit** — type a real hotel, pull live Google reviews, audit each, disagree on a verdict to teach the agent, re-audit, quarantine flagged reviews.
4. **The result** — two charts populated by live runs only (no placeholder data).

## Quick start (local)

```bash
pip install -r requirements.txt
cp .env.example .env       # add your keys
uvicorn crucible.app.main:app --host 127.0.0.1 --port 8000
# open http://localhost:8000
```

Node.js 20 is required because the Reflector spawns `@arizeai/phoenix-mcp` over `npx` at runtime.

Required environment variables:

```
GOOGLE_API_KEY=                          # Gemini key (Google AI Studio)
PHOENIX_API_KEY=                         # Arize Phoenix Cloud API key
PHOENIX_HOST=https://app.phoenix.arize.com/s/<your-space>
PHOENIX_COLLECTOR_ENDPOINT=https://app.phoenix.arize.com/s/<your-space>
PLACES_API_KEY=                          # Google Places API (for Live Audit)
```

## CLI mode

```bash
python crucible/loop/eval.py            # run the self-improving pipeline in the terminal
CRUCIBLE_VERBOSE=1 python crucible/loop/eval.py   # dump raw OpenTelemetry spans
```

After a run, open your Arize Phoenix dashboard, project `crucible`, and you will see every Gemini call traced, including a span for the ADK Reflector agent making the MCP `list-traces` call. Open the `crucible-rules` Dataset to see the rules the Reflector just wrote.

## Deploy to Cloud Run

The live demo runs on Google Cloud Run from `Dockerfile.web` (multi-stage Python 3.11 + Node 20).

```bash
gcloud run deploy crucible-demo --source . --region us-central1 --allow-unauthenticated \
  --memory 2Gi --cpu 2 --timeout 600 \
  --set-env-vars "GOOGLE_API_KEY=...,GEMINI_API_KEY=...,PHOENIX_API_KEY=...,PHOENIX_HOST=...,PLACES_API_KEY=..."
```

## Project structure

```
crucible/
  agent/
    detector.py         # Detector agent (reads rules from Phoenix, judges reviews)
    reflector_adk.py    # Reflector: Google ADK LlmAgent + Arize Phoenix MCP toolset
    prompts.py          # System prompts for Detector, Forger, Reflector
    llm_client.py       # Gemini client with tenacity retry
  adversary/
    forger.py           # Forger agent (generates adversarial fakes)
  obs/
    phoenix_rules.py    # Phoenix Dataset read/write: agent memory lives here
    instrumentation.py  # OpenTelemetry / OpenInference tracing setup
    mcp_client.py       # Phoenix MCP client helpers
  loop/
    eval.py             # Self-improving pipeline (CLI + SSE streaming for web)
  data/
    load_ott.py         # Ott Deceptive Opinion Spam loader (kagglehub + fallback)
  app/
    main.py             # FastAPI web app: live arena, Live Audit, quarantine queue
    templates/
      index.html        # Single-page brief, methodology diagrams, live charts
Dockerfile.web          # Production container (Python 3.11 + Node 20)
```

## Tech stack

`gemini-2.5-flash-lite` · `google-adk` (LlmAgent, McpToolset, InMemoryRunner) · `arize-phoenix` (tracing + Datasets) · `@arizeai/phoenix-mcp` · `openinference-instrumentation-google-adk` · `opentelemetry-sdk` · `google-genai` · Google Places API (New) · FastAPI · Server-Sent Events · Uvicorn · Pydantic · Tenacity · pandas · kagglehub · Python 3.11 · Node 20 · Docker · Google Cloud Run · Cloud Build · Artifact Registry · Apache-2.0

## Dataset

[Ott Deceptive Opinion Spam Corpus](https://www.kaggle.com/datasets/rtatman/deceptive-opinion-spam-corpus) (Cornell, 2011) — 1,600 labelled Chicago hotel reviews, half real, half written by paid human deceivers. The standard academic benchmark for fake-review detection since 2011. Downloaded via `kagglehub`; the loader falls back to a small embedded sample if Kaggle credentials are not present, so the pipeline always runs.

## License

Apache-2.0. See [LICENSE](./LICENSE).
