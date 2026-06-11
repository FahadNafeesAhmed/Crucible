"""
Phoenix-backed rule store: the Detector's learned rules don't live in Python state —
they live in an Arize Phoenix Dataset called `crucible-rules`. Every time the
Reflector distills a new rule from a Phoenix trace, it gets persisted as a row in
the dataset (one example = one rule). The Detector reads the latest rules from
the dataset before every audit.

This turns Phoenix from "logging" into the agent's persistent memory — the agent's
brain literally lives on the partner's platform, not in our code.
"""
import os
import datetime
import logging
import threading
from typing import List, Dict, Any, Optional

logger = logging.getLogger("crucible.phoenix_rules")

DATASET_NAME = os.environ.get("CRUCIBLE_RULES_DATASET", "crucible-rules")

# Cache the client + a small in-memory mirror so the hot path doesn't hit Phoenix on
# every Detector call. The mirror is refreshed when a new rule is added.
_client = None
_client_lock = threading.Lock()
_cache: Dict[str, Any] = {"rules": None, "loaded_at": None}


def _get_client():
    """Return a singleton phoenix.client.Client (cheap; thread-safe)."""
    global _client
    if _client is not None:
        return _client
    with _client_lock:
        if _client is not None:
            return _client
        try:
            from phoenix.client import Client
            base_url = (
                os.environ.get("PHOENIX_HOST")
                or os.environ.get("PHOENIX_COLLECTOR_ENDPOINT")
                or "https://app.phoenix.arize.com"
            )
            api_key = os.environ.get("PHOENIX_API_KEY", "")
            _client = Client(base_url=base_url, api_key=api_key)
        except Exception as e:
            logger.warning("[PhoenixRules] Phoenix client unavailable: %s", e)
            _client = None
    return _client


def _ensure_dataset():
    """Make sure the dataset exists; create with a seed row if not."""
    c = _get_client()
    if c is None:
        return None
    try:
        return c.datasets.get_dataset(dataset=DATASET_NAME)
    except Exception:
        try:
            return c.datasets.create_dataset(
                name=DATASET_NAME,
                dataset_description=(
                    "Crucible learned detection rules — each row is one rule the "
                    "Reflector distilled from a Phoenix trace. The Detector reads "
                    "this dataset before every audit; this is the agent's memory."
                ),
                inputs=[{"failure_text": "seed"}],
                outputs=[{"rule": "Do not treat fabricated specifics (room numbers, "
                                  "named staff, brand-name landmarks) as proof of authenticity."}],
                metadata=[{"source": "seed", "ts": datetime.datetime.utcnow().isoformat()}],
            )
        except Exception as e:
            logger.warning("[PhoenixRules] dataset create failed: %s", e)
            return None


def list_rules(refresh: bool = False) -> List[Dict[str, Any]]:
    """Return all rules as a list of dicts: {rule, source, ts}. Cached unless refresh=True."""
    if not refresh and _cache["rules"] is not None:
        return _cache["rules"]
    c = _get_client()
    if c is None:
        return []
    try:
        ds = c.datasets.get_dataset(dataset=DATASET_NAME)
    except Exception:
        ds = _ensure_dataset()
        if ds is None:
            return []
    rules = []
    for ex in ds.examples:
        # phoenix returns example as a dict (TypedDict-ish)
        out = ex.get("output", {}) if isinstance(ex, dict) else getattr(ex, "output", {})
        meta = ex.get("metadata", {}) if isinstance(ex, dict) else getattr(ex, "metadata", {})
        rule = (out or {}).get("rule", "").strip()
        if not rule:
            continue
        rules.append({
            "rule": rule,
            "source": (meta or {}).get("source", "unknown"),
            "ts": (meta or {}).get("ts", ""),
        })
    _cache["rules"] = rules
    _cache["loaded_at"] = datetime.datetime.utcnow().isoformat()
    return rules


def add_rule(rule_text: str, failure_text: str = "", source: str = "reflector") -> bool:
    """Persist a new learned rule as a row in the Phoenix Dataset. Returns True on success."""
    rule_text = (rule_text or "").strip()
    if not rule_text:
        return False
    c = _get_client()
    if c is None:
        return False
    _ensure_dataset()
    try:
        c.datasets.add_examples_to_dataset(
            dataset=DATASET_NAME,
            inputs=[{"failure_text": failure_text[:500]}],
            outputs=[{"rule": rule_text}],
            metadata=[{"source": source, "ts": datetime.datetime.utcnow().isoformat()}],
        )
        # Bust cache so the next read sees it.
        _cache["rules"] = None
        return True
    except Exception as e:
        logger.warning("[PhoenixRules] add failed: %s", e)
        return False


def render_blindspots() -> str:
    """Render the dataset rules as the {blindspots} string the Detector prompt expects."""
    rs = list_rules()
    if not rs:
        return "None"
    return "\n".join(f"{i+1}. {r['rule']}" for i, r in enumerate(rs))


def reset_for_demo() -> None:
    """Drop the cache (does not delete dataset rows). Useful between demo takes."""
    _cache["rules"] = None
