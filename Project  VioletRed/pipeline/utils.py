import re
from functools import wraps


# ----------------------------------------------------------------------
# Debug print helper (controlled by config['debug'] flags)
# ----------------------------------------------------------------------
def print_debug(message, debug_cfg):
    """
    Print debug messages only when enabled.
    debug_cfg: section from config/debug
    Example:
        print_debug("Generating...", debug)
    """
    # If ANY debug flag is true → allow debug printing
    if any(debug_cfg.values()):
        print(f"[DEBUG] {message}")


# ----------------------------------------------------------------------
# Clean up generated text (strip special tokens, spaces)
# ----------------------------------------------------------------------
def clean_text(text):
    """
    Basic sanitiser for model-generated text.
    Removes repeated whitespace, trims edges, removes weird leftovers.
    """
    if not text:
        return ""

    # collapse multiple spaces
    text = re.sub(r"\s+", " ", text)
    # remove leading/trailing whitespace
    text = text.strip()
    return text


# ----------------------------------------------------------------------
# Count tokens (useful for metrics)
# ----------------------------------------------------------------------
def count_tokens(text, tokenizer=None):
    """
    Count tokens using tokenizer if provided.
    Fallback: word-level count.
    """
    if tokenizer is not None:
        return len(tokenizer.encode(text))
    return len(text.split())


# ----------------------------------------------------------------------
# Logging decorator (optional — used when measuring perf later)
# ----------------------------------------------------------------------
def log_step(func):
    """
    Optional decorator to log pipeline step durations.
    We may use it later for performance profiling.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # In future: add timestamps or W&B logging
        return func(*args, **kwargs)
    return wrapper
