# Decisions

Fixed decisions. Don't change one without asking (CLAUDE.md). Each has its date where it is
known, and the reason. Open work is in [roadmap.md](roadmap.md).

## Data

| decision | since | why |
|---|---|---|
| **Tokenizer** `HuggingFaceTB/SmolLM2-135M` (byte-level BPE, vocab 49,152, EOS id 0, no BOS) | first pilot | An existing, well-tested tokenizer: results compare with SmolLM2, and no tokenizer training. 49,152 = 384 × 128, so the head needs no padding. |
| **Token files**: flat raw uint16 arrays, no `.npy` header, one per raw file with the same base name (`000_00000.parquet` → `000_00000.npy`). Each document = tokens + EOS. | first pilot | uint16 holds every id; OLMo-core counts tokens from the file size. |
| **Sources and shares**: fineweb_edu 0.60, python_edu 0.20, finemath 0.13, gutenberg 0.07 (`configs/mix.yaml`) | first pilot | Planned mix. No ablation yet (roadmap). |
| **No repetition**: one pass; `max_repetition_ratio` stays 1.0; the build fails if a source is too small | first pilot | Each token seen once, so dropout is 0.0. |
| **Never modify `data/raw/`** | first pilot | It is kept for future models; every later step can be rebuilt from it. |
| **Scripts are never named after a stdlib module** (e.g. `tokenize.py`) | first pilot | It would be imported instead of the stdlib module and break transformers/torch. |

## Eval

| decision | since | why |
|---|---|---|
| **Holdout sets are frozen**: `data/eval/<name>.npy|.json` (+ documents in `data/raw_eval/`), ~2M tokens each: fineweb_edu, python_edu, finemath, gutenberg, wikipedia_en. Never regenerate or edit them. | first pilot | Every BPB number is comparable only on the same sets. |
| **Eval-only shards** never downloaded or tokenized for training: fineweb_edu `000_00049.parquet`, finemath `train-00063-of-00064.parquet`, gutenberg `project_gutenberg-dolma-0014.json.gz`, python_edu the last 8,000 rows of `python-edu/train-00000-of-00002.parquet` (`sources.py`). `mix.py` refuses them. | first pilot | No contamination of the holdout sets. |
| **BPB** = total nats / ln 2 / total UTF-8 bytes of the scored tokens, summed over the whole set and divided once. Bytes per token from the byte-level table (special tokens 0). fp32 loss. The 21 byte values with no SmolLM2 token are dropped at tokenization and not counted. | first pilot | A per-window mean would weight short windows wrong; fp32 keeps runs and machines comparable. |
| **BPB windows: non-overlapping 1024 tokens** (`BPB_WINDOW`, fixed, not the model's context) | 2026-10-16 (was 256) | Same as the training context. The 40M runs' BPB used 256 and is not comparable; 1024 gives lower numbers for the same model. |
| **Benchmarks**: lm-eval 0.4.13 (pinned), 0-shot, fp32, `--device mps`, batch 16, samples logged, the same for every model. Pilot tasks: hellaswag, piqa, arc_easy, sciq, lambada_openai. Answer BPB = gold-answer nats / ln 2 / UTF-8 bytes of the gold continuation, summed over the task, divided once. No code-execution benchmarks yet. | first pilot | Comparable across models; answer BPB shows progress before accuracy moves. |
| **Benchmarks stay on the Mac (MPS)**: copy the exported model back and run there | 2026-10-16 | Comparable with gpt2, random init and the 40M run. Running on the GPU would mean new settings and re-running every model. |

## Model and training

| decision | since | why |
|---|---|---|
| **Context length 1024** (`configs/mix.yaml` `sequence_length` = model `context_length`; the trainer checks) | 2026-10-16 | At 256 the matmuls are too small for an H100. 2048 ruled out: the evals don't reward it, most documents are shorter (python_edu ~510, fineweb_edu ~1,100 tokens), and GPT-2 124M, the baseline, used 1024. |
| **124M model**: 12 layers, width 768, 12 heads, SwiGLU 2048, RMSNorm, RoPE, tied embedding (~122.7M) | 2026-10-16 | GPT-2-small size, so GPT-2 is the natural comparison. 5B tokens ≈ 40 tokens/parameter. |
| **LR 6e-4** (GPT-2 small), warmup 300, cosine to 6e-5, 524,288 tokens/step, AdamW (0.9, 0.95), decay 0.1 on matrices only | 2026-10-20 | No sweep yet; revisit before mix ablations, which must share one LR (roadmap). |
| **No TF32** | 2026-10-20 | Training matmuls are bf16, so TF32 gave nothing; it only made the "fp32" holdout BPB on the GPU differ from the Mac by ~0.1%. |
| **Speed-ups after the baseline** (fused cross-entropy, max-autotune, fused QKV, 64 × 8, lite evals) | 2026-10-21 | The first 124M run is the clean reference; each speed-up is tested one at a time against its loss curve. |
| **GPU resume is not bit-exact, and doesn't need to be** | 2026-10-21 | Some CUDA backward kernels sum in a varying order, so any two GPU runs drift in the last bits. The data, weights, optimizer and schedule continue exactly. |
| **Logging through a pipe uses `tee -ia`** | 2026-10-21 | A plain `tee` exits on Ctrl-C; the trainer's next print then dies before the checkpoint save (lost the smoke test's final save). |
