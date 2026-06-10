from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
import sqlite3
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from crucible.agent.detector import DetectorAgent
from crucible.obs.mcp_client import PhoenixMCPClient
from crucible.loop.eval import run_eval_loop_stream
import threading
import json
import asyncio

app = FastAPI(title="Crucible Arena")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Crucible Review Auditor</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; padding: 2rem; background: #f9fafb; }
        .container { max-width: 800px; margin: 0 auto; background: white; padding: 2rem; border-radius: 8px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); }
        h1 { color: #111827; }
        .verdict-fake { color: #dc2626; font-weight: bold; }
        .verdict-real { color: #16a34a; font-weight: bold; }
        .review-card { border: 1px solid #e5e7eb; padding: 1rem; margin-bottom: 1rem; border-radius: 6px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Crucible Review Auditor</h1>
        <p>Enter a Keyword or Product ID to audit its reviews for authenticity.</p>
        <form action="/audit" method="post">
            <input type="text" name="search_query" placeholder="e.g., lipstick, shoes, or B001" required style="padding: 0.5rem; width: 60%; font-size: 1rem;">
            <button type="submit" style="padding: 0.5rem 1rem; background: #2563eb; color: white; border: none; border-radius: 4px; font-size: 1rem; cursor: pointer;">Audit</button>
        </form>
        <div id="results" style="margin-top: 2rem;">
            {results}
        </div>
    </div>
</body>
</html>
"""

from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse

@app.get("/pipeline", response_class=HTMLResponse)
async def pipeline_view():
    presentation_path = os.path.join(os.path.dirname(__file__), "templates", "pipeline.html")
    if os.path.exists(presentation_path):
        return FileResponse(presentation_path)
    return HTMLResponse("<p>Pipeline template not found!</p>")

class AppState:
    def __init__(self):
        self.running = False
        self.returncode = None
        self.logs = []
        self.fakes = []
        self.rules = []
        self.stop_requested = False
        self.current_round = 0
        
    def reset(self):
        self.running = False
        self.returncode = None
        self.logs = []
        self.fakes = []
        self.rules = []
        self.stop_requested = False
        self.current_round = 0

state = AppState()

def pipeline_runner():
    state.reset()
    state.running = True
    try:
        for raw_event in run_eval_loop_stream(iterations=3):
            if state.stop_requested:
                state.logs.append("[server] pipeline stopped by user.")
                break
                
            try:
                event = json.loads(raw_event)
            except:
                event = {}

            line = ""
            if event.get("type") == "info":
                line = event["content"]
            elif event.get("type") == "header":
                line = f"=== {event['content']} ==="
            elif event.get("type") == "review_result":
                line = f"Review: {event['text_preview']}\n  True: {event['true_label']} | Guessed: {event['guessed_label']} -> {'[CORRECT]' if event['correct'] else '[WRONG]'}\n  Reasoning: {event['reasoning']}"
            elif event.get("type") == "metrics":
                line = f"Metrics: Benchmark={event['bench_acc']:.1f}%, Adv={event['adv_acc']:.1f}%"
            elif event.get("type") == "complete":
                line = "[server] pipeline complete."
            
            if line:
                for split_line in line.split('\\n'):
                    state.logs.append(split_line)
            
            # Populate structured data
            if event.get("type") == "fake_generated":
                state.fakes.append({
                    "text": event["text"],
                    "round": state.current_round
                })
            elif event.get("type") == "round_start":
                state.current_round = event["round"]
            elif event.get("type") == "rules_updated":
                state.rules = [{"rule": r} for r in event["rules"].split('\n') if r.strip()]
            elif event.get("type") == "summary":
                state.rules = [{"rule": r} for r in event["final_rules"].split('\n') if r.strip()]

        state.returncode = 0
    except Exception as e:
        state.logs.append(f"Exception: {str(e)}")
        state.returncode = 1
    finally:
        state.running = False

@app.get("/api/status")
def api_status():
    return {
        "lines": len(state.logs),
        "running": state.running,
        "returncode": state.returncode
    }

@app.post("/api/run")
def api_run():
    if state.running:
        return {"ok": False, "error": "Already running"}
    thread = threading.Thread(target=pipeline_runner)
    thread.start()
    return {"ok": True}

@app.post("/api/stop")
def api_stop():
    state.stop_requested = True
    return {"ok": True}

@app.get("/api/fakes")
def api_fakes():
    return state.fakes

@app.get("/api/rules")
def api_rules():
    return state.rules

@app.get("/api/stream")
async def api_stream(request: Request):
    async def log_generator():
        last_index = 0
        while True:
            if await request.is_disconnected():
                break
            
            if last_index < len(state.logs):
                line = state.logs[last_index]
                data = json.dumps({"line": line})
                yield f"data: {data}\n\n"
                last_index += 1
            else:
                if not state.running and last_index >= len(state.logs):
                    yield f"data: {json.dumps({'line': '__CRUCIBLE_DONE__'})}\n\n"
                    break
                await asyncio.sleep(0.5)

    return StreamingResponse(log_generator(), media_type="text/event-stream")

@app.get("/")
async def read_root():
    # Serve the massive standalone React bundle generated by Claude
    presentation_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    if os.path.exists(presentation_path):
        return FileResponse(presentation_path)
    return HTMLTemplate("<p>Presentation file not found!</p>")

def HTMLTemplate(results_html=""):
    return HTML_TEMPLATE.replace("{results}", results_html)

def fetch_and_cache_huggingface_reviews(search_query: str, conn: sqlite3.Connection):
    print(f"Cache miss for '{search_query}'. Streaming from Hugging Face...")
    try:
        from datasets import load_dataset
        dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_All_Beauty", split="full", streaming=True, trust_remote_code=True)
        
        hf_reviews = []
        rows_scanned = 0
        MAX_SCAN = 25000
        MAX_REVIEWS = 5
        
        for row in dataset:
            rows_scanned += 1
            if rows_scanned > MAX_SCAN:
                print(f"Reached scan limit of {MAX_SCAN} rows without finding enough reviews.")
                break
                
            text = row.get('text', '')
            if not text:
                continue
                
            row_product_id = row.get('parent_asin', '')
            
            # Check for keyword in text OR exact match on product ID
            if search_query.lower() in text.lower() or search_query.lower() == row_product_id.lower():
                text = text.replace('\n', ' ')
                rating = int(row.get('rating', 5.0))
                date = "2023-01-01"
                reviewer = row.get('user_id', f"user_{len(hf_reviews)}")
                verified_purchase = 1 if row.get('verified_purchase') else 0
                label = "real"
                
                # We don't specify ID to let AUTOINCREMENT handle it
                hf_reviews.append((row_product_id, text, rating, date, reviewer, verified_purchase, label))
                
                if len(hf_reviews) >= MAX_REVIEWS:
                    print(f"Found {MAX_REVIEWS} reviews for '{search_query}'!")
                    break
                    
        if hf_reviews:
            cursor = conn.cursor()
            cursor.executemany('''
                INSERT INTO reviews (product_id, text, rating, date, reviewer, verified_purchase, label)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', hf_reviews)
            conn.commit()
            print(f"Cached {len(hf_reviews)} reviews to SQLite.")
            return True
        return False
    except Exception as e:
        print(f"Failed to stream from Hugging Face: {e}")
        return False

@app.post("/audit", response_class=HTMLResponse)
async def audit_reviews(search_query: str = Form(...)):
    db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'crucible.db')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id, product_id, text, rating, date, reviewer, verified_purchase FROM reviews WHERE text LIKE ? OR product_id = ?", (f"%{search_query}%", search_query))
    reviews = cursor.fetchall()

    if not reviews:
        # CACHE MISS - Trigger Streamer
        found = fetch_and_cache_huggingface_reviews(search_query, conn)
        if found:
            # Re-query the cache
            cursor.execute("SELECT id, product_id, text, rating, date, reviewer, verified_purchase FROM reviews WHERE text LIKE ? OR product_id = ?", (f"%{search_query}%", search_query))
            reviews = cursor.fetchall()

    conn.close()

    if not reviews:
        return HTMLTemplate(f"<p>No reviews found for <b>'{search_query}'</b> in the database or Hugging Face cache.</p>")

    # Initialize Detector
    detector = DetectorAgent(mcp_client=PhoenixMCPClient())
    verdicts = detector.analyze_reviews(reviews)

    html_results = f"<h2>Audit Results for '{search_query}'</h2>"
    
    for v in verdicts:
        color_class = "verdict-fake" if v['verdict'] == "fake" else "verdict-real"
        html_results += f"""
        <div class="review-card">
            <p><strong>Reviewer:</strong> {v['reviewer']} | <strong>Rating:</strong> {v['rating']} stars</p>
            <p>"{v['text']}"</p>
            <p><strong>Verdict:</strong> <span class="{color_class}">{v['verdict'].upper()}</span> (Confidence: {v['confidence']*100}%)</p>
        </div>
        """

    return HTMLTemplate(html_results)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
