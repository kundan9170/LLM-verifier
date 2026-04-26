"""
Import an existing MATH-500 evaluation log into the benchmark harness
directory layout, so `make_extra_plots.py` can plot it without any
model reruns.

Reads:
    math500_eval_checkpoint.json
      OR
    math500_regraded.json  (from regrade_math500.py)

Writes:
    results/math500/predictions.jsonl  — raw predictions (harness format)
    results/math500/graded.jsonl       — regraded predictions with correctness
    results/math500/stats.json         — summary

Regrades on the fly using the same fixed grader (extraction +
normalization + 0.6B judge fallback), so the numbers here match what
make_extra_plots.py and run_math_benchmarks.py would produce if the
run had originally been done through the harness.

Usage:
    # Use the regrade JSON if you have it (faster, no judge calls needed
    # for labels that already agree):
    python import_math500_results.py --input math500_regraded.json

    # Or start from the raw checkpoint and re-grade here:
    python import_math500_results.py --input math500_eval_checkpoint.json
"""

import argparse
import json
import pathlib
import re
import sys

import torch
from tqdm import tqdm

ROOT = pathlib.Path(__file__).resolve().parent


# ======================================================================
# Grader (same logic as run_math_benchmarks.py — keep in sync)
# ======================================================================
def last_boxed(text):
    if not text:
        return None
    idx = text.rfind("\\boxed")
    if idx < 0:
        idx = text.rfind("\\fbox")
        if idx < 0:
            return None
    i = text.find("{", idx)
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1 : j]
    return None


def _normalize_math(s):
    if s is None:
        return None
    s = s.strip().replace("\n", "").replace(" ", "")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\!", "").replace("\\,", "").replace("\\:", "").replace("\\;", "")
    s = s.replace("^{\\circ}", "").replace("^\\circ", "")
    s = s.replace("\\$", "").replace("$", "")
    s = s.replace("\\%", "").replace("%", "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = re.sub(r"\\frac(\d)(\d)", r"\\frac{\1}{\2}", s)
    while s.endswith(".") or s.endswith(","):
        s = s[:-1]
    while s.startswith("{") and s.endswith("}"):
        s = s[1:-1]
    return s


def extract_answer_from_prediction(pred_text):
    if not pred_text:
        return ""
    boxed = last_boxed(pred_text)
    if boxed is not None:
        return boxed
    lines = [ln.strip() for ln in pred_text.strip().splitlines() if ln.strip()]
    if not lines:
        return pred_text.strip()
    tail = lines[-1]
    tail = re.sub(r"^(the\s+)?(final\s+)?answer\s+is[:\s]*", "", tail, flags=re.I)
    return tail.rstrip(".")


def llm_grade(predicted, gold, judge_model, judge_tokenizer):
    pred_bare = extract_answer_from_prediction(predicted)
    gold_bare = last_boxed(gold) if "\\boxed" in (gold or "") else gold
    if gold_bare is None:
        gold_bare = (gold or "").strip()

    if _normalize_math(pred_bare) == _normalize_math(gold_bare):
        return True
    if not pred_bare.strip():
        return False

    messages = [
        {
            "role": "system",
            "content": (
                "You are a strict math grader. Given two short answers, "
                "decide if they are mathematically equivalent. "
                "Answer with exactly one word: 'yes' or 'no'. No other text."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Student answer: {pred_bare}\n"
                f"Correct answer: {gold_bare}\n"
                f"Equivalent?"
            ),
        },
    ]
    prompt = judge_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = judge_tokenizer(prompt, return_tensors="pt").to(judge_model.device)
    with torch.no_grad():
        out = judge_model.generate(**inputs, max_new_tokens=3, do_sample=False)
    resp = judge_tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip().lower()
    tok = re.split(r"\s+", resp, maxsplit=1)[0] if resp else ""
    tok = tok.strip(".,!?:;'\"")
    return tok == "yes"


# ======================================================================
# Input normalization — handles all three possible input formats
# ======================================================================
def normalize_input(raw):
    """Return a list of per-problem dicts with a known shape.

    Handles three formats:
      1. list[dict]                           → old math500_eval_checkpoint.json
      2. {"stats": ..., "problems": list}     → math500_regraded.json
      3. anything else → sys.exit
    """
    if isinstance(raw, dict) and "problems" in raw:
        return raw["problems"], "regraded"
    if isinstance(raw, list):
        return raw, "checkpoint"
    sys.exit("Unrecognized input format. Expected a list or a dict with 'problems' key.")


def unify_record(rec):
    """Produce a harness-format record from either input source."""
    # Field names differ slightly between the two formats — unify them.
    return {
        "id": rec.get("id"),
        "dataset": "math500",
        "question": rec.get("question", ""),
        "gold": rec.get("ground_truth", "") or rec.get("gold", ""),
        "base_ans": rec.get("base_ans", ""),
        "base_tokens": rec.get("base_tokens", 0) or 0,
        "ver_ans": rec.get("ver_ans", ""),
        "ver_tokens": rec.get("ver_tokens", 0) or 0,
        "ver_aborted": "Cannot provide a reliable answer" in (rec.get("ver_ans") or ""),
        # The regraded JSON already contains re-scored labels — we prefer those
        # when available, otherwise we regrade from scratch below.
        "_precomputed_base_correct": rec.get("base_correct_new"),
        "_precomputed_ver_correct": rec.get("ver_correct_new"),
    }


# ======================================================================
# Main
# ======================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True,
                    help="Path to math500_eval_checkpoint.json or math500_regraded.json")
    ap.add_argument("--judge_model", type=str, default="Qwen/Qwen3-0.6B")
    args = ap.parse_args()

    in_path = pathlib.Path(args.input)
    if not in_path.exists():
        sys.exit(f"Input file not found: {in_path}")

    with open(in_path) as f:
        raw = json.load(f)

    problems, source_kind = normalize_input(raw)
    print(f"Loaded {len(problems)} problems (source: {source_kind})")

    # Output directory matches what run_math_benchmarks / make_extra_plots expect
    out_dir = ROOT / "results" / "math500"
    out_dir.mkdir(parents=True, exist_ok=True)
    preds_path = out_dir / "predictions.jsonl"
    graded_path = out_dir / "graded.jsonl"
    stats_path = out_dir / "stats.json"

    unified = [unify_record(r) for r in problems]

    # Decide whether we need the judge
    need_judge = any(
        r["_precomputed_base_correct"] is None or r["_precomputed_ver_correct"] is None
        for r in unified
    )

    judge_model = judge_tokenizer = None
    if need_judge:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"Loading judge {args.judge_model} ...")
        judge_tokenizer = AutoTokenizer.from_pretrained(args.judge_model, use_fast=True)
        judge_model = AutoModelForCausalLM.from_pretrained(
            args.judge_model, torch_dtype=torch.float16, device_map="auto"
        )
        judge_model.eval()
    else:
        print("Using precomputed correctness labels from regraded JSON — "
              "no judge calls needed.")

    # Grade + write both files
    base_correct = ver_correct = 0
    base_tokens_total = ver_tokens_total = 0
    aborts = 0
    abort_when_base_wrong = 0
    abort_when_base_right = 0
    base_wrong_count = 0

    with open(preds_path, "w") as fp, open(graded_path, "w") as fg:
        for u in tqdm(unified, desc="grading"):
            # Write the raw-prediction record (harness format)
            pred_rec = {
                "id": u["id"],
                "dataset": "math500",
                "question": u["question"],
                "gold": u["gold"],
                "base_ans": u["base_ans"],
                "base_tokens": u["base_tokens"],
                "base_time": None,  # wasn't recorded in old runs
                "ver_ans": u["ver_ans"],
                "ver_tokens": u["ver_tokens"],
                "ver_time": None,
                "ver_aborted": u["ver_aborted"],
            }
            fp.write(json.dumps(pred_rec) + "\n")

            # Grade
            if u["_precomputed_base_correct"] is not None:
                bc = bool(u["_precomputed_base_correct"])
            else:
                bc = llm_grade(u["base_ans"], u["gold"], judge_model, judge_tokenizer)

            if u["ver_aborted"]:
                vc = False
            elif u["_precomputed_ver_correct"] is not None:
                vc = bool(u["_precomputed_ver_correct"])
            else:
                vc = llm_grade(u["ver_ans"], u["gold"], judge_model, judge_tokenizer)

            if bc:
                base_correct += 1
            else:
                base_wrong_count += 1
            if vc:
                ver_correct += 1

            if u["ver_aborted"]:
                aborts += 1
                if not bc:
                    abort_when_base_wrong += 1
                else:
                    abort_when_base_right += 1

            base_tokens_total += u["base_tokens"]
            ver_tokens_total += u["ver_tokens"]

            graded_rec = {
                **pred_rec,
                "base_correct": bc,
                "ver_correct": vc,
                "base_pred_bare": extract_answer_from_prediction(u["base_ans"]),
                "ver_pred_bare": extract_answer_from_prediction(u["ver_ans"]),
            }
            fg.write(json.dumps(graded_rec) + "\n")

    n = len(unified)
    stats = {
        "dataset": "math500",
        "n": n,
        "base_accuracy": 100.0 * base_correct / n,
        "ver_accuracy": 100.0 * ver_correct / n,
        "base_avg_tokens": base_tokens_total / n,
        "ver_avg_tokens": ver_tokens_total / n,
        "token_savings_pct": 100.0 * (1 - (ver_tokens_total / max(base_tokens_total, 1))),
        "aborts": aborts,
        "abort_rate": 100.0 * aborts / n,
        "abort_precision": (100.0 * abort_when_base_wrong / aborts) if aborts > 0 else 0.0,
        "abort_recall": (
            100.0 * abort_when_base_wrong / base_wrong_count
            if base_wrong_count > 0 else 0.0
        ),
        "abort_when_base_wrong": abort_when_base_wrong,
        "abort_when_base_right": abort_when_base_right,
        "base_wrong_count": base_wrong_count,
    }

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print("\n=== MATH-500 (imported) ===")
    print(f"n = {n}")
    print(f"Baseline: {stats['base_accuracy']:.2f}%   avg tokens: {stats['base_avg_tokens']:.0f}")
    print(f"Verifier: {stats['ver_accuracy']:.2f}%   avg tokens: {stats['ver_avg_tokens']:.0f}")
    print(f"Token savings: {stats['token_savings_pct']:.1f}%")
    print(f"Aborts: {aborts} (precision {stats['abort_precision']:.0f}%, "
          f"recall {stats['abort_recall']:.0f}%)")
    print(f"\nFiles written:")
    print(f"  {preds_path}")
    print(f"  {graded_path}")
    print(f"  {stats_path}")
    print("\nNow run:")
    print("  python make_extra_plots.py --datasets math500")


if __name__ == "__main__":
    main()