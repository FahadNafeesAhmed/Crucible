"""
Observability setup for Crucible.
Configures Arize Phoenix tracing (cloud or local) and instruments LLM calls.

Set CRUCIBLE_VERBOSE=1 to dump raw OpenTelemetry JSON spans to the terminal.
"""
import os

os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

from dotenv import load_dotenv


def setup_instrumentation():
    """
    Initializes Arize Phoenix and instruments the LLM calls.
    Returns the local Phoenix session (if running locally) or None.
    """
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env')
    load_dotenv(env_path)
    
    api_key = os.environ.get("PHOENIX_API_KEY", "")
    use_cloud = bool(api_key and api_key != "your_phoenix_api_key_here")
    verbose = os.environ.get("CRUCIBLE_VERBOSE", "0") == "1"
    session = None
    
    if use_cloud:
        print("[Instrumentation] Connecting to Arize Phoenix Cloud...")
        try:
            from phoenix.otel import register
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            
            tracer_provider = register(project_name="crucible")
            
            # Optionally dump raw JSON spans to terminal for debugging
            if verbose:
                from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
                console_processor = SimpleSpanProcessor(ConsoleSpanExporter())
                tracer_provider.add_span_processor(console_processor)
                print("[Instrumentation] VERBOSE MODE: Raw JSON spans will print to terminal.")
            
            print("[Instrumentation] Phoenix Cloud connected.")
            
        except ImportError:
            print("[Instrumentation] Warning: phoenix.otel not found. Traces will not be exported.")
    else:
        try:
            import phoenix as px
            session = px.launch_app()
            print(f"[Instrumentation] Phoenix server running locally at: {session.url}")
        except Exception as e:
            print(f"[Instrumentation] Warning: Local Phoenix failed ({e}).")

    # Instrument Google ADK
    try:
        from openinference.instrumentation.google_adk import GoogleADKInstrumentor
        GoogleADKInstrumentor().instrument()
        print("[Instrumentation] Google ADK instrumented.")
    except ImportError:
        pass

    return session


def flush_traces():
    """Force-flush all pending traces to the cloud before exit."""
    try:
        from opentelemetry import trace
        provider = trace.get_tracer_provider()
        if hasattr(provider, "force_flush"):
            provider.force_flush(timeout_millis=10000)
            print("[Instrumentation] All traces flushed to cloud.")
    except Exception as e:
        print(f"[Instrumentation] Warning: Could not flush traces: {e}")
