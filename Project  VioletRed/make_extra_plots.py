"""
Extra report plots from existing graded.jsonl data — no model reruns.

Produces three plots per dataset (and a combined summary):
  1. Per-problem token scatter: baseline tokens vs verifier tokens,
     colored by correctness.
  2. CoT length distribution split by correctness (histogram + KDE).
  3. Exit-type breakdown: why the verifier stopped
     (natural </think> vs conclusion-exit vs abort vs max-tokens).

Usage:
    python make_extra_plots.py
    python make_extra_plots.py --datasets gsm8k math500
"""

import argparse
import json
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent

DATASET_TITLES = {
    "gsm8k": "GSM8K",
    "math500": "MATH-500",
    "math500_hard": "MATH-500 (Level 5)",
    "aime2024": "AIME 2024",
}
ALL_DATASETS = ["gsm8k", "math500", "math500_hard", "aime2024"]


# ======================================================================
# I/O
# ======================================================================
def load_jsonl(path):
    if not path.exists():
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def dataset_results_dir(dataset):
    return ROOT / "results" / dataset


# ======================================================================
# Plot 1: per-problem token scatter
# ======================================================================
def plot_token_scatter(records, dataset, out_path):
    """
    X = baseline tokens, Y = verifier tokens. One dot per problem.
    Color-coded by (base_correct, ver_correct).
    Below the y=x line → verifier saved tokens.
    Red dots at low Y → "exited too early" failures.
    """
    if not records:
        return

    fig, ax = plt.subplots(figsize=(8, 7))

    groups = {
        "both_right": {"x": [], "y": [], "c": "#55A868",
                       "label": "Both correct", "marker": "o", "alpha": 0.55},
        "verifier_wrong": {"x": [], "y": [], "c": "#C44E52",
                           "label": "Verifier wrong, baseline right", "marker": "x", "alpha": 0.95},
        "baseline_wrong": {"x": [], "y": [], "c": "#4C72B0",
                           "label": "Baseline wrong, verifier right", "marker": "s", "alpha": 0.8},
        "both_wrong": {"x": [], "y": [], "c": "#8172B3",
                       "label": "Both wrong", "marker": "^", "alpha": 0.6},
    }

    for r in records:
        bt = r.get("base_tokens") or 0
        vt = r.get("ver_tokens") or 0
        bc = r.get("base_correct", False)
        vc = r.get("ver_correct", False)
        if bc and vc:
            key = "both_right"
        elif bc and not vc:
            key = "verifier_wrong"
        elif not bc and vc:
            key = "baseline_wrong"
        else:
            key = "both_wrong"
        groups[key]["x"].append(bt)
        groups[key]["y"].append(vt)

    max_val = max(
        [max(g["x"]) for g in groups.values() if g["x"]] +
        [max(g["y"]) for g in groups.values() if g["y"]] +
        [1]
    )

    # y=x reference (no savings)
    ax.plot([0, max_val], [0, max_val], linestyle="--", color="gray",
            alpha=0.5, label="y = x (no savings)")

    for key, g in groups.items():
        if g["x"]:
            ax.scatter(g["x"], g["y"], s=42, c=g["c"], marker=g["marker"],
                       alpha=g["alpha"], label=f"{g['label']} (n={len(g['x'])})",
                       edgecolors="black", linewidths=0.3)

    ax.set_xlabel("Baseline tokens")
    ax.set_ylabel("Verifier tokens")
    ax.set_xlim(0, max_val * 1.05)
    ax.set_ylim(0, max_val * 1.05)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.set_title(f"{DATASET_TITLES.get(dataset, dataset)} — Per-problem token usage")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  saved {out_path}")


# ======================================================================
# Plot 2: CoT length distribution split by correctness
# ======================================================================
def plot_length_by_correctness(records, dataset, out_path):
    """
    Two-panel histogram. Left: baseline. Right: verifier.
    Each panel splits CoT length into correct vs wrong answers.
    Reveals whether wrong answers tend to have longer or shorter CoTs.
    """
    if not records:
        return

    base_right = [r["base_tokens"] for r in records if r.get("base_correct")]
    base_wrong = [r["base_tokens"] for r in records if not r.get("base_correct")]
    ver_right = [r["ver_tokens"] for r in records if r.get("ver_correct")]
    ver_wrong = [r["ver_tokens"] for r in records if not r.get("ver_correct")]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)

    all_tokens = base_right + base_wrong + ver_right + ver_wrong
    if not all_tokens:
        plt.close()
        return
    max_tok = max(all_tokens)
    bins = np.linspace(0, max_tok, 25)

    # Baseline panel
    if base_right:
        ax1.hist(base_right, bins=bins, alpha=0.55, color="#55A868",
                 label=f"Correct (n={len(base_right)}, μ={np.mean(base_right):.0f})",
                 edgecolor="black", linewidth=0.3)
    if base_wrong:
        ax1.hist(base_wrong, bins=bins, alpha=0.55, color="#C44E52",
                 label=f"Wrong (n={len(base_wrong)}, μ={np.mean(base_wrong):.0f})",
                 edgecolor="black", linewidth=0.3)
    ax1.set_title("Baseline")
    ax1.set_xlabel("CoT tokens")
    ax1.set_ylabel("# problems")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Verifier panel
    if ver_right:
        ax2.hist(ver_right, bins=bins, alpha=0.55, color="#55A868",
                 label=f"Correct (n={len(ver_right)}, μ={np.mean(ver_right):.0f})",
                 edgecolor="black", linewidth=0.3)
    if ver_wrong:
        ax2.hist(ver_wrong, bins=bins, alpha=0.55, color="#C44E52",
                 label=f"Wrong (n={len(ver_wrong)}, μ={np.mean(ver_wrong):.0f})",
                 edgecolor="black", linewidth=0.3)
    ax2.set_title("Verifier (early exit)")
    ax2.set_xlabel("CoT tokens")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"{DATASET_TITLES.get(dataset, dataset)} — "
                 "CoT length vs. correctness", y=1.00)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  saved {out_path}")


# ======================================================================
# Plot 3: exit-type breakdown
# ======================================================================
def classify_exit_type(record, max_cot_tokens=1024):
    """
    Best-effort classification of WHY the verifier stopped.

    Categories:
      - "abort"             : verifier aborted with error message
      - "max_tokens"        : hit the token budget (tokens >= ~90% of cap)
      - "early_exit"        : short trace, verifier injected </think>
      - "natural_think_end" : everything else — model produced </think> on its own

    This is heuristic. If you want exact attribution, log the exit type
    explicitly in controller.run() (see note at bottom of this file).
    """
    if record.get("ver_aborted"):
        return "abort"

    vt = record.get("ver_tokens") or 0
    bt = record.get("base_tokens") or 0

    # If verifier used ~all of the budget, it hit max tokens
    if vt >= 0.9 * max_cot_tokens:
        return "max_tokens"

    # If verifier traces are much shorter than baseline, verifier intervened
    # (threshold: verifier used <85% of baseline tokens)
    if bt > 0 and vt < 0.85 * bt:
        return "early_exit"

    return "natural_think_end"


def plot_exit_breakdown(records, dataset, out_path, max_cot_tokens=1024):
    """
    Stacked bar showing the distribution of exit reasons.
    If the record has an explicit 'exit_type' field we use it;
    otherwise we classify heuristically.
    """
    if not records:
        return

    categories = ["natural_think_end", "early_exit", "abort", "max_tokens"]
    colors = {
        "natural_think_end": "#55A868",
        "early_exit": "#4C72B0",
        "abort": "#C44E52",
        "max_tokens": "#DD8452",
    }
    pretty = {
        "natural_think_end": "Model wrote </think>",
        "early_exit": "Verifier early-exit",
        "abort": "Verifier abort",
        "max_tokens": "Hit token limit",
    }

    counts = {k: 0 for k in categories}
    used_explicit = False
    for r in records:
        if "exit_type" in r:
            used_explicit = True
            et = r["exit_type"]
        else:
            et = classify_exit_type(r, max_cot_tokens)
        if et in counts:
            counts[et] += 1
        else:
            counts.setdefault(et, 0)
            counts[et] += 1

    total = sum(counts.values())
    if total == 0:
        plt.close()
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    left = 0
    for cat in categories:
        n = counts.get(cat, 0)
        pct = 100.0 * n / total
        if n > 0:
            ax.barh([dataset], [pct], left=left, color=colors[cat],
                    edgecolor="black", linewidth=0.4,
                    label=f"{pretty[cat]} ({n}, {pct:.1f}%)")
            if pct > 5:
                ax.text(left + pct / 2, 0, f"{pct:.0f}%",
                        ha="center", va="center", fontsize=10,
                        fontweight="bold", color="white")
        left += pct

    ax.set_xlim(0, 100)
    ax.set_xlabel("% of problems")
    ax.set_yticks([])
    method_note = " (explicit)" if used_explicit else " (heuristic)"
    ax.set_title(f"{DATASET_TITLES.get(dataset, dataset)} — "
                 f"Verifier exit breakdown{method_note}")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15),
              ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  saved {out_path}")


# ======================================================================
# Combined summary: exit breakdown across all datasets
# ======================================================================
def plot_exit_breakdown_summary(all_data, out_path, max_cot_tokens=1024):
    """Grouped stacked bar across all datasets."""
    if not all_data:
        return

    categories = ["natural_think_end", "early_exit", "abort", "max_tokens"]
    colors = {
        "natural_think_end": "#55A868",
        "early_exit": "#4C72B0",
        "abort": "#C44E52",
        "max_tokens": "#DD8452",
    }
    pretty = {
        "natural_think_end": "Model wrote </think>",
        "early_exit": "Verifier early-exit",
        "abort": "Verifier abort",
        "max_tokens": "Hit token limit",
    }

    datasets = [d for d, _ in all_data]
    titles = [DATASET_TITLES.get(d, d) for d in datasets]
    all_counts = []
    for _, records in all_data:
        counts = {k: 0 for k in categories}
        for r in records:
            et = r.get("exit_type") or classify_exit_type(r, max_cot_tokens)
            if et in counts:
                counts[et] += 1
        all_counts.append(counts)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    y_positions = np.arange(len(datasets))

    for i, counts in enumerate(all_counts):
        total = max(sum(counts.values()), 1)
        left = 0
        for cat in categories:
            n = counts.get(cat, 0)
            pct = 100.0 * n / total
            if n > 0:
                ax.barh(i, pct, left=left, color=colors[cat],
                        edgecolor="black", linewidth=0.4)
                if pct > 6:
                    ax.text(left + pct / 2, i, f"{pct:.0f}%",
                            ha="center", va="center",
                            fontsize=9, fontweight="bold", color="white")
            left += pct

    # One legend entry per category
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[c]) for c in categories]
    labels = [pretty[c] for c in categories]
    ax.legend(handles, labels, loc="upper center",
              bbox_to_anchor=(0.5, -0.08), ncol=4, fontsize=9)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(titles)
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of problems")
    ax.set_title("Verifier exit-type breakdown across datasets")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  saved {out_path}")


# ======================================================================
# Main
# ======================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=None,
                    help="Which datasets to plot (default: all found).")
    ap.add_argument("--max_cot_tokens", type=int, default=1024,
                    help="Used for 'hit token limit' heuristic.")
    args = ap.parse_args()

    requested = args.datasets or ALL_DATASETS
    available = []
    for d in requested:
        p = dataset_results_dir(d) / "graded.jsonl"
        if p.exists():
            available.append(d)
        else:
            print(f"[skip] no graded.jsonl at {p}")

    if not available:
        sys.exit("No graded.jsonl files found. Run evaluation + grading first.")

    all_data = []
    for d in available:
        print(f"\n=== {d} ===")
        records = load_jsonl(dataset_results_dir(d) / "graded.jsonl")
        print(f"  loaded {len(records)} records")
        all_data.append((d, records))

        out = dataset_results_dir(d)
        plot_token_scatter(records, d, out / "extra_token_scatter.png")
        plot_length_by_correctness(records, d, out / "extra_length_by_correctness.png")
        plot_exit_breakdown(records, d, out / "extra_exit_breakdown.png",
                            args.max_cot_tokens)

    # Cross-dataset exit-breakdown summary
    if len(all_data) > 1:
        print("\n=== summary ===")
        summary_dir = ROOT / "results" / "summary"
        summary_dir.mkdir(parents=True, exist_ok=True)
        plot_exit_breakdown_summary(
            all_data,
            summary_dir / "extra_exit_breakdown_all.png",
            args.max_cot_tokens,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()


# ======================================================================
# OPTIONAL: make exit classification exact instead of heuristic
# ======================================================================
# The exit-type plot currently classifies heuristically from token counts.
# To make it exact, log the reason directly in pipeline/controller.py:
#
#   In run(), at each exit point, set:
#       exit_reason = "natural_think_end"   # hit_think_end = True
#       exit_reason = "early_exit"          # decision == "exit"
#       exit_reason = "abort"               # decision == "abort"
#       exit_reason = "max_tokens"          # while loop ended
#
#   Then return it alongside the answer:
#       return final_answer.strip(), total_tokens_generated, exit_reason
#
# And in run_math_benchmarks.py's run_dataset(), capture it:
#       ver_ans, ver_tokens, exit_reason = controller.run(prob["question"])
#       record["exit_type"] = exit_reason
#
# Future runs will write 'exit_type' into predictions.jsonl, and this
# script will automatically use it (the 'explicit' note will appear in
# the plot title). Past runs continue to use the heuristic.