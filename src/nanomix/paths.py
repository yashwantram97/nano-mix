"""Every folder the pipeline reads or writes, relative to the repository root (run commands from there).

    data/raw/<source>/        download.py         raw files as published; never modified
    data/tokenized/<source>/  tokenize_data.py    flat uint16 token files + meta.json
    data/mix/                 mix.py              OLMo-core's cached instance indices
    data/eval/<name>.npy|json make_eval_sets.py   frozen holdout token sets (never rebuild)
    data/raw_eval/            make_eval_sets.py   the holdout documents
    checkpoints/<run_id>/     train               ckpt.pt, ckpt_prev.pt
    logs/                     train               <run_id>.jsonl metrics, loss plots, console logs
    exports/<run_id>/         export_hf.py        LlamaForCausalLM copies
    results/                  benchmarks.py       benchmark JSON and summary.md (in git)
"""

from pathlib import Path

MIX_CONFIG = Path("configs/mix.yaml")

RAW = Path("data/raw")
TOKENIZED = Path("data/tokenized")
EVAL = Path("data/eval")
RAW_EVAL = Path("data/raw_eval")

CHECKPOINTS = Path("checkpoints")
LOGS = Path("logs")
EXPORTS = Path("exports")
RESULTS = Path("results")
