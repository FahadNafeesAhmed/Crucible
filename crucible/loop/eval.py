"""
Crucible Evaluation Pipeline — the Self-Improving Agent Loop.

This script orchestrates the full adversarial pipeline:
  1. Forger generates deceptive fake reviews
  2. Detector evaluates a mixed batch of a fixed Ott benchmark + adversarial fakes
  3. Grader scores Benchmark Accuracy and Adversarial Catch Rate
  4. Reflector analyzes adversarial failures and generates new detection rules
  5. Detector's prompt accumulates the new rules
  6. Repeat
"""
import os
import sys
import logging
import random
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from crucible.agent.detector import DetectorAgent
from crucible.adversary.forger import ForgerAgent
from crucible.agent.reflector_adk import reflect_via_adk
from crucible.obs.mcp_client import PhoenixMCPClient
from crucible.obs.instrumentation import setup_instrumentation, flush_traces
from crucible.data.load_ott import load_ott_data

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
LOG_FORMAT = "[%(asctime)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt="%H:%M:%S")
logger = logging.getLogger("crucible.eval")


def _clean_rule(rule_text):
    """Extract a single clean rule sentence from the agent's output.

    Defends against the agent emitting extra lines: prefer a line containing
    'Rule:', else the last substantive line; then strip numbering/bullets/prefix.
    """
    import re
    raw = (rule_text or "").strip()
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    chosen = ""
    for ln in lines:                                   # prefer an explicit "Rule:" line
        if re.search(r'\brule\s*:', ln, re.I):
            chosen = ln
    if not chosen:
        # last line that looks like an actual rule (has several words), not a label
        for ln in reversed(lines):
            if len(ln.split()) >= 4:
                chosen = ln
                break
    if not chosen:
        chosen = lines[-1] if lines else raw
    t = re.sub(r'(?i)^\s*rule\s*:\s*', '', chosen)     # drop leading "Rule:"
    t = re.sub(r'^\s*(?:\d+[\.\)]\s*)+', '', t)         # leading "1." / "14."
    t = re.sub(r'^\s*[-*•]\s*', '', t)               # leading bullet
    return t.strip()


def _similar(a, b):
    """Cheap near-duplicate check via word-overlap (Jaccard) on lowercased tokens."""
    wa = set(w for w in __import__('re').findall(r'[a-z]+', a.lower()) if len(w) > 3)
    wb = set(w for w in __import__('re').findall(r'[a-z]+', b.lower()) if len(w) > 3)
    if not wa or not wb:
        return False
    overlap = len(wa & wb) / len(wa | wb)
    return overlap >= 0.6


def append_blindspot(detector, rule_text):
    """Clean, de-duplicate and append a learned rule to the detector. Returns True if added."""
    rule = _clean_rule(rule_text)
    if not rule:
        return False
    existing = [] if detector.blindspots == "None" else [
        _clean_rule(line) for line in detector.blindspots.strip().split("\n")
    ]
    if any(_similar(rule, e) for e in existing):
        return False  # skip near-duplicate of an existing rule
    count = len(existing) + 1
    if detector.blindspots == "None":
        detector.blindspots = f"1. {rule}"
    else:
        detector.blindspots += f"\n{count}. {rule}"
    return True


def get_fixed_ott_benchmark(df, limit=6):
    """Sample a fixed, balanced benchmark set from the Ott dataset."""
    real_pool = df[df['deceptive'] == 'truthful']
    fake_pool = df[df['deceptive'] == 'deceptive']
    # Clamp the requested count to what's actually available (fallback sample is small)
    n_each = max(1, min(limit // 2, len(real_pool), len(fake_pool)))
    real_subset = real_pool.sample(n=n_each, random_state=42)
    fake_subset = fake_pool.sample(n=n_each, random_state=42)
    
    formatted = []
    # Add real reviews
    for index, row in real_subset.iterrows():
        text = row['text']
        hotel = row['hotel']
        mock_tuple = (index, hotel, text, 5.0, "2023-01-01", "ott_user_real", 1)
        formatted.append({
            "tuple": mock_tuple,
            "text": text,
            "true_label": "real",
            "source": "benchmark"
        })
    # Add fake reviews
    for index, row in fake_subset.iterrows():
        text = row['text']
        hotel = row['hotel']
        mock_tuple = (index, hotel, text, 5.0, "2023-01-01", "ott_user_fake", 1)
        formatted.append({
            "tuple": mock_tuple,
            "text": text,
            "true_label": "fake",
            "source": "benchmark"
        })
    return formatted


def print_header(text, char="="):
    width = 60
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}")


def print_review_result(idx, text_preview, true_label, guessed, reasoning, correct):
    status = "[CORRECT]" if correct else "[WRONG]"
    print(f"\n  Review {idx}:")
    print(f"    Text:      {text_preview}...")
    print(f"    True:      {true_label}")
    print(f"    Guessed:   {guessed}")
    print(f"    Reasoning: {reasoning}")
    print(f"    Result:    {status}")


def run_eval_loop(iterations=2):
    print_header("CRUCIBLE: Self-Improving Agent Pipeline")
    
    # 0. Boot up tracing
    setup_instrumentation()
    
    mcp_client = PhoenixMCPClient()
    detector = DetectorAgent(mcp_client=mcp_client)
    forger = ForgerAgent()  # Single instance, preserves iteration state
    
    # Load the Ott dataset into memory once
    df = load_ott_data()
    benchmark_set = get_fixed_ott_benchmark(df, limit=6)

    # Track accuracy across rounds for the summary table
    round_results = []

    for round_num in range(1, iterations + 1):
        print_header(f"ITERATION {round_num} / {iterations}", "-")

        # 1. Forger generates adversarial fakes exploiting the detector's known blindspots
        fake_texts = forger.generate_fakes(
            "The Palmer House Hilton",
            count=4,
            detector_blindspots=detector.blindspots,
        )
        adversarial_data = []
        for idx, text in enumerate(fake_texts):
            logger.info(f"[Forger] Fake {idx + 1}: {text[:80]}...")
            mock_tuple = (9000 + round_num * 10 + idx, "The Palmer House Hilton", text, 5.0, "2023-01-01", "forger_bot", 1)
            adversarial_data.append({
                "tuple": mock_tuple,
                "text": text,
                "true_label": "fake",
                "source": "adversarial",
            })

        # 2. Shuffle benchmark + adversarial and evaluate with the Detector
        test_set = benchmark_set + adversarial_data
        random.shuffle(test_set)
        tuples_to_analyze = [d["tuple"] for d in test_set]

        print(f"\n  [Grader] Auditing {len(tuples_to_analyze)} reviews ({len(benchmark_set)} benchmark + {len(adversarial_data)} adversarial)...")
        verdicts = detector.analyze_reviews(tuples_to_analyze)
        verdict_by_id = {v["id"]: v for v in verdicts}

        # 3. Grade
        bench_correct = 0
        adv_correct = 0
        failed_adv_reviews = []
        failed_bench_reviews = []
        for d in test_set:
            v = verdict_by_id.get(d["tuple"][0], {"verdict": "real", "reasoning": "missing"})
            guessed_label = v["verdict"]
            reasoning = v.get("reasoning", "No reasoning provided.")
            is_correct = guessed_label == d["true_label"]

            if d["source"] == "benchmark":
                if is_correct:
                    bench_correct += 1
                else:
                    failed_bench_reviews.append({"text": d["text"], "true_label": d["true_label"], "guessed_label": guessed_label, "reasoning": reasoning})
            else:
                if is_correct:
                    adv_correct += 1
                else:
                    failed_adv_reviews.append({"text": d["text"], "true_label": d["true_label"], "guessed_label": guessed_label, "reasoning": reasoning})
                print_review_result("Adv Fake", d['text'][:80], d['true_label'], guessed_label, reasoning, is_correct)

        bench_acc = (bench_correct / len(benchmark_set)) * 100 if benchmark_set else 0
        adv_acc = (adv_correct / len(adversarial_data)) * 100 if adversarial_data else 0
        round_results.append({"round": round_num, "bench_acc": bench_acc, "adv_acc": adv_acc})

        print(f"\n  [Grader] Benchmark Accuracy (Ott): {bench_acc:.1f}% ({bench_correct}/{len(benchmark_set)})")
        print(f"  [Grader] Adversarial Catch Rate:   {adv_acc:.1f}% ({adv_correct}/{len(adversarial_data)})")

        # 4. Reflector: Google ADK agent inspects Phoenix traces via the Arize MCP server
        failures_to_learn = failed_adv_reviews if failed_adv_reviews else failed_bench_reviews[:2]
        if failures_to_learn:
            print(f"  [Reflector] Activating Google ADK Agent to inspect traces via Arize Phoenix MCP...")
            try:
                rule_text = reflect_via_adk(failed_reviews=failures_to_learn, current_rules=detector.blindspots)
                if rule_text:
                    if append_blindspot(detector, rule_text):
                        print(f"  [Reflector] Generated Rule: {_clean_rule(rule_text)}")
                        print(f"  [Reflector] Detector prompt updated for next round.")
                    else:
                        print(f"  [Reflector] Rule was a near-duplicate of an existing one; skipped.")
            except Exception as e:
                print(f"  [Reflector] ADK Agent Error: {e}")
        else:
            print(f"  [Grader] Perfect score across the board! No failures to learn from.")

    # ---------------------------------------------------------------------------
    # Summary Table
    # ---------------------------------------------------------------------------
    print_header("PIPELINE SUMMARY")
    print(f"\n  {'Round':<8} {'Benchmark Acc':<15} {'Adv Catch Rate':<15}")
    print(f"  {'-----':<8} {'-------------':<15} {'--------------':<15}")
    for r in round_results:
        print(f"  {r['round']:<8} {r['bench_acc']:>12.1f}% {r['adv_acc']:>13.1f}%")
    
    print(f"\n  Final Accumulated Rules:\n{detector.blindspots}")
    
    # Flush all traces to Arize Phoenix before exit
    flush_traces()
    print_header("PIPELINE COMPLETE")


def run_eval_loop_stream(iterations=2):
    yield json.dumps({"type": "info", "content": "CRUCIBLE: Self-Improving Agent Pipeline Started"}) + "\n"
    
    # 0. Boot up tracing
    setup_instrumentation()
    
    mcp_client = PhoenixMCPClient()
    detector = DetectorAgent(mcp_client=mcp_client)
    forger = ForgerAgent()  # Single instance, preserves iteration state
    
    df = load_ott_data()
    # Balanced benchmark (10) + a few adversarial fakes (4) per round — enough for a
    # smooth, believable accuracy curve while still finishing quickly on camera.
    benchmark_set = get_fixed_ott_benchmark(df, limit=10)
    ADV_PER_ROUND = 4

    round_results = []

    for round_num in range(1, iterations + 1):
        yield json.dumps({"type": "round_start", "round": round_num, "total_rounds": iterations}) + "\n"

        yield json.dumps({"type": "info", "content": f"[Forger] Generating {ADV_PER_ROUND} adversarial fakes for 'The Drake' (Round {round_num})..."}) + "\n"
        fake_texts = forger.generate_fakes(
            "The Drake",
            count=ADV_PER_ROUND,
            detector_blindspots=detector.blindspots
        )
        
        adversarial_data = []
        for idx, text in enumerate(fake_texts):
            mock_tuple = (9000 + round_num * 10 + idx, "The Drake", text, 5.0, "2023-01-01", "forger_bot", 1)
            adversarial_data.append({
                "tuple": mock_tuple,
                "text": text,
                "true_label": "fake",
                "source": "adversarial"
            })
            yield json.dumps({"type": "fake_generated", "idx": idx + 1, "text": text}) + "\n"
        
        test_set = benchmark_set + adversarial_data
        random.shuffle(test_set)
        tuples_to_analyze = [d["tuple"] for d in test_set]
        
        yield json.dumps({"type": "info", "content": f"[Grader] Auditing {len(tuples_to_analyze)} reviews ({len(benchmark_set)} benchmark + {len(adversarial_data)} adversarial)..."}) + "\n"
        verdicts = detector.analyze_reviews(tuples_to_analyze)
        
        bench_correct = 0
        adv_correct = 0
        failed_adv_reviews = []
        failed_bench_reviews = []
        
        for d, v in zip(test_set, verdicts):
            guessed_label = v["verdict"]
            reasoning = v.get("reasoning", "No reasoning provided.")
            is_correct = guessed_label == d["true_label"]
            
            if d["source"] == "benchmark":
                if is_correct:
                    bench_correct += 1
                else:
                    failed_bench_reviews.append({
                        "text": d["text"],
                        "true_label": d["true_label"],
                        "guessed_label": guessed_label,
                    })
            elif d["source"] == "adversarial":
                if is_correct:
                    adv_correct += 1
                else:
                    failed_adv_reviews.append({
                        "text": d["text"],
                        "true_label": d["true_label"],
                        "guessed_label": guessed_label,
                    })
                
                yield json.dumps({
                    "type": "review_result",
                    "text_preview": d['text'][:80] + "...",
                    "true_label": d['true_label'],
                    "guessed_label": guessed_label,
                    "reasoning": reasoning,
                    "correct": is_correct
                }) + "\n"
                
        bench_acc = (bench_correct / len(benchmark_set)) * 100
        adv_acc = (adv_correct / len(adversarial_data)) * 100
        
        round_results.append({
            "round": round_num,
            "bench_acc": bench_acc,
            "adv_acc": adv_acc,
        })
        
        yield json.dumps({
            "type": "metrics",
            "bench_acc": bench_acc,
            "adv_acc": adv_acc,
            "bench_correct": bench_correct,
            "bench_total": len(benchmark_set),
            "adv_correct": adv_correct,
            "adv_total": len(adversarial_data)
        }) + "\n"
        
        false_positives = [r for r in failed_bench_reviews if r["true_label"] == "real"]
        false_negatives = [r for r in failed_bench_reviews if r["true_label"] == "fake"] + failed_adv_reviews
        
        failures_dict = {
            "false_positives": false_positives[:5],
            "false_negatives": false_negatives[:5]
        }
        
        if false_positives or false_negatives:
            yield json.dumps({"type": "info", "content": f"[Reflector] Activating Google ADK Agent to inspect Phoenix traces via Arize MCP ({len(false_positives)} FP / {len(false_negatives)} FN)..."}) + "\n"
            try:
                rule_text = reflect_via_adk(failed_reviews=failures_dict, current_rules=detector.blindspots)
            except Exception as e:
                rule_text = ""
                yield json.dumps({"type": "info", "content": f"[Reflector] ADK Agent error: {e}"}) + "\n"

            if rule_text:
                append_blindspot(detector, rule_text)
            yield json.dumps({"type": "rules_updated", "rules": detector.blindspots}) + "\n"
        else:
            yield json.dumps({"type": "info", "content": "[Grader] Perfect score across the board! No failures to learn from."}) + "\n"
            
    yield json.dumps({
        "type": "summary",
        "round_results": round_results,
        "final_rules": detector.blindspots
    }) + "\n"
    
    flush_traces()
    yield json.dumps({"type": "complete"}) + "\n"


if __name__ == "__main__":
    setup_instrumentation()
    run_eval_loop(iterations=2)
