"""
Math benchmark harness for the LLM verifier pipeline.

Evaluates baseline (full CoT) vs. verifier (early exit) on multiple
math datasets and saves raw predictions + regraded accuracy.

Supported datasets:
    gsm8k              — grade-school math (~200 sampled)
    math500            — MATH-500 (existing)
    math500_hard       — MATH-500 Level 5 only
    aime2024           — AIME 2024 (30 problems)

Usage:
    # Run one dataset
    python run_math_benchmarks.py --dataset gsm8k --limit 200

    # Run all datasets in sequence
    python run_math_benchmarks.py --dataset all

    # Resume a crashed run (picks up from last checkpoint)
    python run_math_benchmarks.py --dataset gsm8k --limit 200 --resume

    # Regrade existing results without re-running the model
    python run_math_benchmarks.py --dataset gsm8k --regrade_only

Per-dataset outputs (written to ./results/<dataset>/):
    predictions.jsonl   — one line per problem: question, gold, base_ans, ver_ans, tokens, ...
    stats.json          — summary statistics
    plot.png            — accuracy + token count bars for this dataset

After all datasets are done, run:
    python run_math_benchmarks.py --summary
to produce cross-dataset plots in ./results/summary/.
"""

import argparse
import json
import os
import pathlib
import random
import re
import sys
import time
from typing import Optional

import matplotlib.pyplot as plt
import torch
from datasets import load_dataset
from tqdm import tqdm

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# --- Lazy imports for the pipeline modules; only needed when generating. ---
# This lets --regrade_only and --summary run without loading the 4B model.


# ======================================================================
# Grader (same logic as the fixed grader — keep in sync)
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
    """Grade: extraction + normalization + LLM fallback. Returns bool."""
    pred_bare = extract_answer_from_prediction(predicted)

    # gold may be bare ("42"), a \boxed string, or a full solution
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
# Dataset loaders — each returns list of {"question": str, "gold": str}
# ======================================================================
def load_gsm8k(limit: int, seed: int = 42):
    ds = load_dataset("openai/gsm8k", "main", split="test")
    random.Random(seed).shuffle(indices := list(range(len(ds))))
    chosen = indices[:limit]
    out = []
    for idx in chosen:
        ex = ds[idx]
        # GSM8K answers are after "#### " in the `answer` field
        gold = ex["answer"].split("####")[-1].strip().replace(",", "")
        out.append({"question": ex["question"], "gold": gold})
    return out


def load_math500(limit: int, seed: int = 42):
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    if limit < len(ds):
        ds = ds.select(range(limit))  # first N for reproducibility vs. your existing run
    return [{"question": ex["problem"], "gold": ex["answer"]} for ex in ds]


def load_math500_hard(limit: int, seed: int = 42):
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    # Level 5 = hardest
    hard = [ex for ex in ds if ex.get("level") == "Level 5"]
    random.Random(seed).shuffle(hard)
    hard = hard[:limit]
    return [{"question": ex["problem"], "gold": ex["answer"]} for ex in hard]


def load_aime2024(limit: int = 30, seed: int = 42):
    # AIME-2024 has 30 problems total. `limit` only truncates if < 30.
    ds = load_dataset("Maxwell-Jia/AIME_2024", split="train")
    if limit < len(ds):
        ds = ds.select(range(limit))
    out = []
    for ex in ds:
        # Fields: Problem, Answer, Solution (answer is an integer 0-999)
        q = ex.get("Problem") or ex.get("problem")
        a = ex.get("Answer") or ex.get("answer")
        out.append({"question": q, "gold": str(a)})
    return out


DATASET_LOADERS = {
    "gsm8k": load_gsm8k,
    "math500": load_math500,
    "math500_hard": load_math500_hard,
    "aime2024": load_aime2024,
}

# Pretty names for plots
DATASET_TITLES = {
    "gsm8k": "GSM8K",
    "math500": "MATH-500",
    "math500_hard": "MATH-500 (Level 5)",
    "aime2024": "AIME 2024",
}


# ======================================================================
# I/O helpers
# ======================================================================
def results_dir(dataset: str) -> pathlib.Path:
    d = ROOT / "results" / dataset
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_jsonl(path: pathlib.Path) -> list:
    if not path.exists():
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def append_jsonl(path: pathlib.Path, record: dict):
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


# ======================================================================
# Generation (needs the pipeline)
# ======================================================================
def run_dataset(dataset: str, limit: int, config: str, resume: bool):
    """Generate baseline + verifier predictions for a dataset."""
    # Lazy imports — not needed for --regrade_only or --summary
    from llm_engine.model_loader import LLMModelLoader
    from pipeline.controller import PipelineController

    problems = DATASET_LOADERS[dataset](limit)
    out_dir = results_dir(dataset)
    preds_path = out_dir / "predictions.jsonl"

    done_ids = set()
    if resume and preds_path.exists():
        existing = load_jsonl(preds_path)
        done_ids = {r["id"] for r in existing}
        print(f"[resume] Found {len(done_ids)} completed problems in {preds_path}")

    print(f"Loading model for {dataset}...")
    loader = LLMModelLoader(config_path=config)
    model, tokenizer = loader.get()
    controller = PipelineController(model, tokenizer, config_path=config)

    for i, prob in enumerate(tqdm(problems, desc=f"[{dataset}] generate")):
        if i in done_ids:
            continue

        record = {
            "id": i,
            "dataset": dataset,
            "question": prob["question"],
            "gold": prob["gold"],
        }

        # Baseline: disable early exit
        controller.exit_logic.enable_early_exit = False
        t0 = time.time()
        try:
            base_ans, base_tokens = controller.run(prob["question"])
        except Exception as e:
            base_ans, base_tokens = f"[ERROR: {e}]", 0
        record["base_ans"] = base_ans
        record["base_tokens"] = base_tokens
        record["base_time"] = time.time() - t0

        # Verifier: enable early exit
        controller.exit_logic.enable_early_exit = True
        t0 = time.time()
        try:
            ver_ans, ver_tokens = controller.run(prob["question"])
        except Exception as e:
            ver_ans, ver_tokens = f"[ERROR: {e}]", 0
        record["ver_ans"] = ver_ans
        record["ver_tokens"] = ver_tokens
        record["ver_time"] = time.time() - t0
        record["ver_aborted"] = "Cannot provide a reliable answer" in (ver_ans or "")

        append_jsonl(preds_path, record)

    print(f"[{dataset}] predictions saved to {preds_path}")


# ======================================================================
# Grading + metrics
# ======================================================================
def grade_dataset(dataset: str, judge_model_name: str = "Qwen/Qwen3-0.6B"):
    """Grade predictions.jsonl and produce stats.json + plot.png."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir = results_dir(dataset)
    preds_path = out_dir / "predictions.jsonl"
    records = load_jsonl(preds_path)
    if not records:
        print(f"[{dataset}] no predictions found at {preds_path}. Skipping.")
        return None

    print(f"Loading judge {judge_model_name}...")
    jt = AutoTokenizer.from_pretrained(judge_model_name, use_fast=True)
    jm = AutoModelForCausalLM.from_pretrained(
        judge_model_name, torch_dtype=torch.float16, device_map="auto"
    )
    jm.eval()

    graded = []
    base_correct = ver_correct = 0
    base_tokens_total = ver_tokens_total = 0
    aborts = 0
    # For "abort catches real error" metric
    abort_when_base_wrong = 0
    abort_when_base_right = 0
    base_wrong_count = 0

    for r in tqdm(records, desc=f"[{dataset}] grade"):
        bc = llm_grade(r["base_ans"], r["gold"], jm, jt)
        vc = llm_grade(r["ver_ans"], r["gold"], jm, jt) if not r["ver_aborted"] else False

        if bc:
            base_correct += 1
        else:
            base_wrong_count += 1
        if vc:
            ver_correct += 1
        if r["ver_aborted"]:
            aborts += 1
            if not bc:
                abort_when_base_wrong += 1
            else:
                abort_when_base_right += 1

        base_tokens_total += r.get("base_tokens", 0) or 0
        ver_tokens_total += r.get("ver_tokens", 0) or 0

        graded.append({
            **r,
            "base_correct": bc,
            "ver_correct": vc,
            "base_pred_bare": extract_answer_from_prediction(r["base_ans"]),
            "ver_pred_bare": extract_answer_from_prediction(r["ver_ans"]),
        })

    n = len(records)
    stats = {
        "dataset": dataset,
        "n": n,
        "base_accuracy": 100.0 * base_correct / n,
        "ver_accuracy": 100.0 * ver_correct / n,
        "base_avg_tokens": base_tokens_total / n,
        "ver_avg_tokens": ver_tokens_total / n,
        "token_savings_pct": 100.0 * (1 - (ver_tokens_total / max(base_tokens_total, 1))),
        "aborts": aborts,
        "abort_rate": 100.0 * aborts / n,
        "abort_precision": (
            100.0 * abort_when_base_wrong / aborts if aborts > 0 else 0.0
        ),
        "abort_recall": (
            100.0 * abort_when_base_wrong / base_wrong_count if base_wrong_count > 0 else 0.0
        ),
        "abort_when_base_wrong": abort_when_base_wrong,
        "abort_when_base_right": abort_when_base_right,
        "base_wrong_count": base_wrong_count,
    }

    # Save regraded per-problem log (useful for error inspection)
    with open(out_dir / "graded.jsonl", "w") as f:
        for g in graded:
            f.write(json.dumps(g) + "\n")
    with open(out_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    plot_single_dataset(stats, out_dir / "plot.png")
    print(f"[{dataset}] stats saved to {out_dir/'stats.json'}")
    return stats


# ======================================================================
# Plotting
# ======================================================================
def plot_single_dataset(stats, out_path):
    title = DATASET_TITLES.get(stats["dataset"], stats["dataset"])
    labels = ["Baseline (Full CoT)", "LLM Verifier (Early Exit)"]
    accs = [stats["base_accuracy"], stats["ver_accuracy"]]
    toks = [stats["base_avg_tokens"], stats["ver_avg_tokens"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.bar(labels, accs, color=["#4C72B0", "#55A868"])
    ax1.set_title(f"{title}: Accuracy")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_ylim(0, max(accs + [1]) + 10)
    for i, v in enumerate(accs):
        ax1.text(i, v + 1, f"{v:.1f}%", ha="center", fontweight="bold")

    ax2.bar(labels, toks, color=["#C44E52", "#8172B3"])
    ax2.set_title(f"{title}: Avg Tokens per Query")
    ax2.set_ylabel("Token Count")
    ax2.set_ylim(0, max(toks + [1]) * 1.15)
    for i, v in enumerate(toks):
        ax2.text(i, v + max(toks) * 0.02, f"{v:.0f}", ha="center", fontweight="bold")

    plt.suptitle(f"{title} (n={stats['n']})", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_summary(all_stats):
    """Produce cross-dataset plots in ./results/summary/."""
    summary_dir = ROOT / "results" / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    datasets = [s["dataset"] for s in all_stats]
    titles = [DATASET_TITLES.get(d, d) for d in datasets]
    base_accs = [s["base_accuracy"] for s in all_stats]
    ver_accs = [s["ver_accuracy"] for s in all_stats]
    base_toks = [s["base_avg_tokens"] for s in all_stats]
    ver_toks = [s["ver_avg_tokens"] for s in all_stats]

    # --- (1) Grouped bar: accuracy per dataset ---
    import numpy as np
    x = np.arange(len(datasets))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - w/2, base_accs, w, label="Baseline", color="#4C72B0")
    ax.bar(x + w/2, ver_accs, w, label="Verifier", color="#55A868")
    ax.set_xticks(x)
    ax.set_xticklabels(titles)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, max(base_accs + ver_accs) + 10)
    ax.set_title("Accuracy across Math Benchmarks")
    ax.legend()
    for i, (b, v) in enumerate(zip(base_accs, ver_accs)):
        ax.text(i - w/2, b + 1, f"{b:.1f}", ha="center", fontsize=9)
        ax.text(i + w/2, v + 1, f"{v:.1f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(summary_dir / "accuracy_all.png", dpi=120, bbox_inches="tight")
    plt.close()

    # --- (2) Grouped bar: tokens per dataset ---
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - w/2, base_toks, w, label="Baseline", color="#C44E52")
    ax.bar(x + w/2, ver_toks, w, label="Verifier", color="#8172B3")
    ax.set_xticks(x)
    ax.set_xticklabels(titles)
    ax.set_ylabel("Avg Tokens per Query")
    ax.set_title("Token Usage across Math Benchmarks")
    ax.legend()
    for i, (b, v) in enumerate(zip(base_toks, ver_toks)):
        ax.text(i - w/2, b + max(base_toks) * 0.015, f"{b:.0f}", ha="center", fontsize=9)
        ax.text(i + w/2, v + max(base_toks) * 0.015, f"{v:.0f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(summary_dir / "tokens_all.png", dpi=120, bbox_inches="tight")
    plt.close()

    # --- (3) Accuracy vs. compute tradeoff scatter ---
    fig, ax = plt.subplots(figsize=(9, 6))
    for s in all_stats:
        title = DATASET_TITLES.get(s["dataset"], s["dataset"])
        ax.scatter(s["base_avg_tokens"], s["base_accuracy"],
                   s=140, marker="o", color="#4C72B0",
                   edgecolor="black", label="Baseline" if s is all_stats[0] else None)
        ax.scatter(s["ver_avg_tokens"], s["ver_accuracy"],
                   s=140, marker="s", color="#55A868",
                   edgecolor="black", label="Verifier" if s is all_stats[0] else None)
        # Arrow from baseline → verifier
        ax.annotate(
            "",
            xy=(s["ver_avg_tokens"], s["ver_accuracy"]),
            xytext=(s["base_avg_tokens"], s["base_accuracy"]),
            arrowprops=dict(arrowstyle="->", color="gray", lw=1.2, alpha=0.7),
        )
        # Label near the baseline point
        ax.text(s["base_avg_tokens"], s["base_accuracy"] + 1.2, title, fontsize=9)
    ax.set_xlabel("Avg Tokens per Query")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy vs. Compute Tradeoff (→ = early-exit effect)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(summary_dir / "tradeoff_scatter.png", dpi=120, bbox_inches="tight")
    plt.close()

    # --- (4) Abort analysis: precision/recall of "incorrect" verdict ---
    abort_prec = [s["abort_precision"] for s in all_stats]
    abort_rec = [s["abort_recall"] for s in all_stats]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - w/2, abort_prec, w, label="Precision\n(of aborts, % that were actually wrong)",
           color="#DD8452")
    ax.bar(x + w/2, abort_rec, w, label="Recall\n(of wrong answers, % aborted)",
           color="#64B5CD")
    ax.set_xticks(x)
    ax.set_xticklabels(titles)
    ax.set_ylabel("%")
    ax.set_ylim(0, 105)
    ax.set_title("Error-Detection Quality (Abort Verdict)")
    ax.legend(fontsize=9)
    for i, (p, r) in enumerate(zip(abort_prec, abort_rec)):
        ax.text(i - w/2, p + 1.5, f"{p:.0f}", ha="center", fontsize=9)
        ax.text(i + w/2, r + 1.5, f"{r:.0f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(summary_dir / "abort_quality.png", dpi=120, bbox_inches="tight")
    plt.close()

    # --- Write summary table ---
    with open(summary_dir / "summary_table.md", "w") as f:
        f.write("# Math Benchmark Summary\n\n")
        f.write("| Dataset | N | Baseline Acc | Verifier Acc | Δ Acc | "
                "Baseline Tokens | Verifier Tokens | Token Savings | "
                "Aborts | Abort Prec | Abort Recall |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|---|\n")
        for s in all_stats:
            t = DATASET_TITLES.get(s["dataset"], s["dataset"])
            delta = s["ver_accuracy"] - s["base_accuracy"]
            f.write(
                f"| {t} | {s['n']} | {s['base_accuracy']:.1f}% | {s['ver_accuracy']:.1f}% | "
                f"{delta:+.1f}pp | {s['base_avg_tokens']:.0f} | {s['ver_avg_tokens']:.0f} | "
                f"{s['token_savings_pct']:.1f}% | {s['aborts']} | "
                f"{s['abort_precision']:.0f}% | {s['abort_recall']:.0f}% |\n"
            )

    print(f"\nSummary plots + table saved to {summary_dir}/")


# ======================================================================
# CLI
# ======================================================================
ALL_DATASETS = ["gsm8k", "math500", "math500_hard", "aime2024"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="all",
                   choices=ALL_DATASETS + ["all"],
                   help="Which dataset to run.")
    p.add_argument("--limit", type=int, default=200,
                   help="Number of problems per dataset (AIME is always 30).")
    p.add_argument("--config", type=str, default="config/config.yaml")
    p.add_argument("--resume", action="store_true",
                   help="Skip problems already in predictions.jsonl.")
    p.add_argument("--regrade_only", action="store_true",
                   help="Only grade existing predictions, don't generate.")
    p.add_argument("--summary", action="store_true",
                   help="Produce cross-dataset summary plots from existing stats.")
    p.add_argument("--judge_model", type=str, default="Qwen/Qwen3-0.6B")
    args = p.parse_args()

    # --- Summary mode: aggregate from already-graded datasets ---
    if args.summary:
        all_stats = []
        for d in ALL_DATASETS:
            sp = ROOT / "results" / d / "stats.json"
            if sp.exists():
                with open(sp) as f:
                    all_stats.append(json.load(f))
            else:
                print(f"[summary] {sp} not found, skipping {d}")
        if not all_stats:
            sys.exit("No stats.json files found. Run evaluation first.")
        plot_summary(all_stats)
        return

    datasets = ALL_DATASETS if args.dataset == "all" else [args.dataset]

    all_stats = []
    for d in datasets:
        limit = args.limit if d != "aime2024" else 30
        if not args.regrade_only:
            run_dataset(d, limit=limit, config=args.config, resume=args.resume)
        stats = grade_dataset(d, judge_model_name=args.judge_model)
        if stats is not None:
            all_stats.append(stats)

    if len(all_stats) > 1:
        plot_summary(all_stats)


if __name__ == "__main__":
    main()