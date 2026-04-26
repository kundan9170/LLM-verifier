"""
Re-grade an existing MATH-500 evaluation run without re-running the model.

Reads `math500_eval_checkpoint.json` (produced by evaluate_math500.py),
re-scores every prediction using:
  1. \\boxed{} extraction from both prediction and ground truth
  2. Deterministic normalized string comparison (fast path)
  3. Qwen3-0.6B LLM judge on BARE answers (fallback for hard cases)

Prints corrected accuracy numbers and a diff vs. the original run so you
can see exactly how many false positives the old grader produced.

Usage:
    python regrade_math500.py
    python regrade_math500.py --input math500_eval_checkpoint.json --plot regraded.png
"""

import argparse
import json
import re
import sys
import pathlib

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Make the project importable for config loading (optional, only if needed)
ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


# ======================================================================
# Fixed grader
# ======================================================================
def last_boxed(text):
    """Extract content of the last \\boxed{...}, respecting brace nesting."""
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
    """Normalize LaTeX so equivalent forms compare equal."""
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
    """Pull the final answer out of the model's output."""
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


def llm_grade_answer(predicted, ground_truth, judge_model, judge_tokenizer):
    """Grade a MATH-500 prediction.

    Protocol:
      1. Extract bare answer from both sides.
      2. Deterministic normalized comparison (fast path, no LLM).
      3. LLM judge on bare answers only (fallback). Strict 'yes' parse.
    """
    pred_bare = extract_answer_from_prediction(predicted)
    gold_bare = last_boxed(ground_truth)
    if gold_bare is None:
        gold_bare = ground_truth.strip()

    # Fast path
    if _normalize_math(pred_bare) == _normalize_math(gold_bare):
        return True, pred_bare, gold_bare, "normalized"

    # Guard against empty prediction on fallback path
    if not pred_bare.strip():
        return False, pred_bare, gold_bare, "empty"

    # LLM judge fallback on BARE answers
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
        output_ids = judge_model.generate(
            **inputs, max_new_tokens=3, do_sample=False
        )
    response = judge_tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip().lower()

    first_tok = re.split(r"\s+", response, maxsplit=1)[0] if response else ""
    first_tok = first_tok.strip(".,!?:;'\"")
    return first_tok == "yes", pred_bare, gold_bare, "llm_judge"


# ======================================================================
# Judge model loader (kept separate so we don't need the full project)
# ======================================================================
def load_judge(model_name="Qwen/Qwen3-0.6B"):
    print(f"Loading judge model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    print("Judge ready.")
    return model, tokenizer


# ======================================================================
# Plotting (matches your original plot style)
# ======================================================================
def plot_results(stats, output_path):
    labels = ["Baseline (Full CoT)", "Hybrid Verifier (Early Exit)"]
    accuracies = [stats["base"]["accuracy"], stats["verifier"]["accuracy"]]
    avg_tokens = [stats["base"]["avg_tokens"], stats["verifier"]["avg_tokens"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.bar(labels, accuracies, color=["#4C72B0", "#55A868"])
    ax1.set_title("MATH-500 Accuracy (Re-graded)")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_ylim(0, max(accuracies + [1]) + 10)
    for i, v in enumerate(accuracies):
        ax1.text(i, v + 1, f"{v:.1f}%", ha="center", fontweight="bold")

    ax2.bar(labels, avg_tokens, color=["#C44E52", "#8172B3"])
    ax2.set_title("Average Tokens per Query")
    ax2.set_ylabel("Token Count")
    ax2.set_ylim(0, max(avg_tokens + [1]) + 100)
    for i, v in enumerate(avg_tokens):
        ax2.text(i, v + 10, f"{v:.0f}", ha="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    print(f"\nPlot saved to {output_path}")


# ======================================================================
# Main
# ======================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=str, default="math500_eval_checkpoint.json",
        help="Path to the saved evaluation log JSON.",
    )
    parser.add_argument(
        "--output", type=str, default="math500_regraded.json",
        help="Where to save per-problem regraded results.",
    )
    parser.add_argument(
        "--plot", type=str, default="math500_regraded.png",
        help="Path for the regraded plot.",
    )
    parser.add_argument(
        "--judge_model", type=str, default="Qwen/Qwen3-0.6B",
    )
    args = parser.parse_args()

    # Load saved evaluation log
    in_path = pathlib.Path(args.input)
    if not in_path.exists():
        sys.exit(f"Input file not found: {in_path}")

    with open(in_path, "r") as f:
        log = json.load(f)
    print(f"Loaded {len(log)} problems from {in_path}")

    # Load judge
    judge_model, judge_tokenizer = load_judge(args.judge_model)

    # Track metrics + diff vs. original grading
    m = {
        "base": {"correct": 0, "tokens": 0, "was_right_now_wrong": 0, "was_wrong_now_right": 0},
        "verifier": {"correct": 0, "tokens": 0, "aborts": 0,
                     "was_right_now_wrong": 0, "was_wrong_now_right": 0},
    }
    regraded = []

    for item in tqdm(log, desc="Re-grading"):
        gt = item["ground_truth"]
        entry = {
            "id": item.get("id"),
            "question": item.get("question"),
            "ground_truth": gt,
        }

        # Baseline
        base_ans = item.get("base_ans", "")
        base_correct_old = bool(item.get("base_correct", False))
        base_correct, b_pred_bare, b_gold_bare, b_method = llm_grade_answer(
            base_ans, gt, judge_model, judge_tokenizer
        )
        m["base"]["tokens"] += item.get("base_tokens", 0) or 0
        if base_correct:
            m["base"]["correct"] += 1
        if base_correct_old and not base_correct:
            m["base"]["was_right_now_wrong"] += 1
        if (not base_correct_old) and base_correct:
            m["base"]["was_wrong_now_right"] += 1

        entry.update({
            "base_ans": base_ans,
            "base_tokens": item.get("base_tokens", 0),
            "base_pred_bare": b_pred_bare,
            "base_gold_bare": b_gold_bare,
            "base_method": b_method,
            "base_correct_old": base_correct_old,
            "base_correct_new": base_correct,
        })

        # Verifier
        ver_ans = item.get("ver_ans", "")
        ver_correct_old = bool(item.get("ver_correct", False))
        if "Cannot provide a reliable answer" in (ver_ans or ""):
            m["verifier"]["aborts"] += 1
            ver_correct = False
            v_pred_bare, v_gold_bare, v_method = "", last_boxed(gt) or gt, "abort"
        else:
            ver_correct, v_pred_bare, v_gold_bare, v_method = llm_grade_answer(
                ver_ans, gt, judge_model, judge_tokenizer
            )
        m["verifier"]["tokens"] += item.get("ver_tokens", 0) or 0
        if ver_correct:
            m["verifier"]["correct"] += 1
        if ver_correct_old and not ver_correct:
            m["verifier"]["was_right_now_wrong"] += 1
        if (not ver_correct_old) and ver_correct:
            m["verifier"]["was_wrong_now_right"] += 1

        entry.update({
            "ver_ans": ver_ans,
            "ver_tokens": item.get("ver_tokens", 0),
            "ver_pred_bare": v_pred_bare,
            "ver_gold_bare": v_gold_bare,
            "ver_method": v_method,
            "ver_correct_old": ver_correct_old,
            "ver_correct_new": ver_correct,
        })

        regraded.append(entry)

    total = len(log)
    stats = {
        "base": {
            "accuracy": 100.0 * m["base"]["correct"] / total,
            "avg_tokens": m["base"]["tokens"] / total,
        },
        "verifier": {
            "accuracy": 100.0 * m["verifier"]["correct"] / total,
            "avg_tokens": m["verifier"]["tokens"] / total,
            "aborts_triggered": m["verifier"]["aborts"],
        },
    }

    # --- Save per-problem regrade log ---
    with open(args.output, "w") as f:
        json.dump({"stats": stats, "problems": regraded}, f, indent=2)
    print(f"\nPer-problem regrade saved to {args.output}")

    # --- Print summary ---
    print("\n" + "=" * 44)
    print("        RE-GRADED EVALUATION RESULTS        ")
    print("=" * 44)
    print(f"Problems evaluated: {total}")
    print("-" * 44)
    print(f"Baseline Accuracy:    {stats['base']['accuracy']:.2f}%")
    print(f"Baseline Avg Tokens:  {stats['base']['avg_tokens']:.0f}")
    print(f"  flipped right→wrong: {m['base']['was_right_now_wrong']}"
          f"   wrong→right: {m['base']['was_wrong_now_right']}")
    print("-" * 44)
    print(f"Verifier Accuracy:    {stats['verifier']['accuracy']:.2f}%")
    print(f"Verifier Avg Tokens:  {stats['verifier']['avg_tokens']:.0f}")
    print(f"Verifier Aborts:      {stats['verifier']['aborts_triggered']}")
    print(f"  flipped right→wrong: {m['verifier']['was_right_now_wrong']}"
          f"   wrong→right: {m['verifier']['was_wrong_now_right']}")
    print("-" * 44)
    if stats["base"]["avg_tokens"] > 0:
        token_savings = 100.0 * (1 - stats["verifier"]["avg_tokens"] / stats["base"]["avg_tokens"])
        acc_delta = stats["verifier"]["accuracy"] - stats["base"]["accuracy"]
        print(f"Token savings:        {token_savings:+.1f}%")
        print(f"Accuracy delta:       {acc_delta:+.2f} pp")
    print("=" * 44 + "\n")

    plot_results(stats, args.plot)


if __name__ == "__main__":
    main()