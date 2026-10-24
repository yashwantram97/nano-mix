"""0-shot lm-eval benchmarks (pilot task set) for exported checkpoints and Hugging Face models.

Every model runs with the same settings: lm-eval 0.4.13 (pinned in pyproject.toml), 0-shot,
fp32, --device mps, the samples logged. From the logged samples it also computes answer BPB:
the gold answer's log-likelihood in nats / ln 2 / UTF-8 bytes of the gold continuation, summed over
the whole task and divided once.

    uv run nanomix-benchmarks exports/fw60_py20_fm13_gb07_s1234
    uv run nanomix-benchmarks gpt2 HuggingFaceTB/SmolLM2-135M EleutherAI/pythia-160m
    uv run nanomix-benchmarks --summary        # only rebuild results/summary.md

Writes results/<name>.json, the logged samples to results/samples/<name>/<task>.jsonl,
and rebuilds results/summary.md from every results/*.json. <name> is the export directory's name
(the training run id) or the HF model name.
"""

import argparse
import importlib.metadata
import json
import math
from pathlib import Path

from nanomix.paths import RESULTS

# Pilot task set. The main set (+ arc_challenge, winogrande, openbookqa, MMLU cloze) comes later.
TASKS = ["hellaswag", "piqa", "arc_easy", "sciq", "lambada_openai"]
RANDOM_BASELINE = {"hellaswag": 0.25, "piqa": 0.50, "arc_easy": 0.25, "sciq": 0.25, "lambada_openai": 0.0}
SETTINGS = {"num_fewshot": 0, "dtype": "float32", "device": "mps", "batch_size": 16}


def result_name(model):
    """Results file name: exports/fw60_... -> fw60_... (the run id); EleutherAI/pythia-160m -> EleutherAI__pythia-160m."""
    return Path(model).name if Path(model).is_dir() else model.replace("/", "__")


def answer_bpb(samples):
    """Gold-answer BPB over a task's logged samples (sum nats and bytes, divide once)."""
    nats = num_bytes = 0
    for s in samples:
        if len(s["arguments"]) == 1:  # loglikelihood task (lambada): one (context, gold) request
            gold = 0
        else:  # multiple choice: target is the gold choice's index
            gold = int(s["target"])
        continuation = s["arguments"][gold][1]
        loglik = s["filtered_resps"][gold][0]
        nats -= loglik
        num_bytes += len(continuation.encode("utf-8"))
    return nats / math.log(2) / num_bytes


def evaluate(model, limit=None):
    import lm_eval

    out = lm_eval.simple_evaluate(
        model="hf",
        model_args=f"pretrained={model},dtype={SETTINGS['dtype']}",
        tasks=TASKS,
        num_fewshot=SETTINGS["num_fewshot"],
        batch_size=SETTINGS["batch_size"],
        device=SETTINGS["device"],
        log_samples=True,
        limit=limit,
    )
    name = result_name(model)
    sample_dir = RESULTS / "samples" / name
    sample_dir.mkdir(parents=True, exist_ok=True)
    tasks = {}
    for task in TASKS:
        samples = out["samples"][task]
        with open(sample_dir / f"{task}.jsonl", "w") as f:
            for s in samples:
                f.write(json.dumps(s, default=str) + "\n")
        metrics = out["results"][task]
        tasks[task] = {
            "acc": metrics.get("acc,none"),
            "acc_norm": metrics.get("acc_norm,none"),
            "answer_bpb": answer_bpb(samples),
            "num_docs": len(samples),
        }
    result = {
        "model": model,
        "run_name": name,
        "lm_eval_version": importlib.metadata.version("lm-eval"),
        "settings": SETTINGS | {"limit": limit},
        "tasks": tasks,
    }
    (RESULTS / f"{name}.json").write_text(json.dumps(result, indent=2))
    print(f"\n{name}")
    for task, m in tasks.items():
        print(f"  {task:15s} acc {fmt(m['acc'])}  acc_norm {fmt(m['acc_norm'])}  "
              f"answer BPB {m['answer_bpb']:.3f}  ({m['num_docs']} docs)")


def fmt(x, pct=True):
    if x is None:
        return "  -  "
    return f"{100 * x:5.1f}" if pct else f"{x:.3f}"


def write_summary():
    runs = [json.loads(p.read_text()) for p in sorted(RESULTS.glob("*.json"))]
    header = ["task", "metric", "random"] + [r["run_name"] for r in runs]
    lines = ["# Benchmarks (0-shot, lm-eval " + ", ".join(sorted({r["lm_eval_version"] for r in runs})) + ")", "",
             "acc and acc_norm in %; answer BPB = bits per UTF-8 byte of the gold answer (lower is better).",
             "Runs with a `limit` are partial: " + (", ".join(f"{r['run_name']} ({r['settings']['limit']})"
                                                       for r in runs if r["settings"].get("limit")) or "none") + ".", "",
             "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for task in TASKS:
        for metric, label in [("acc", "acc"), ("acc_norm", "acc_norm"), ("answer_bpb", "answer BPB")]:
            random = fmt(RANDOM_BASELINE[task]) if metric != "answer_bpb" else ""
            cells = [fmt(r["tasks"].get(task, {}).get(metric), pct=metric != "answer_bpb") for r in runs]
            lines.append(f"| {task} | {label} | {random} | " + " | ".join(c.strip() for c in cells) + " |")
    (RESULTS / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {RESULTS / 'summary.md'} ({len(runs)} runs)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="*", help="export directories and/or HF model names")
    parser.add_argument("--limit", type=int, help="docs per task, for quick checks only")
    parser.add_argument("--summary", action="store_true", help="only rebuild results/summary.md")
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    if not args.summary:
        for model in args.models:
            evaluate(model, args.limit)
    write_summary()


if __name__ == "__main__":
    main()
