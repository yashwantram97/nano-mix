# What's missing: 40M → 124M model, 5B tokens, one H100 80GB PCIe, >30% utilization

Status as of 2026-10-16. The current pipeline is a **pilot**: one raw file per source, a 500M-token
mix of 256-token instances, a 40M model, and training tuned for a Mac (MPS). Everything below is
what has to change, grouped by pipeline step, with the file and line that needs the change.

## Target and what "30% utilization" means

`nvidia-smi`'s "GPU-Util" only says *a* kernel was running; a badly fed model shows ~100% there.
The meaningful number is **MFU** (model FLOPs utilization): useful training FLOPs per second ÷ the
GPU's peak.

| | value |
|---|---|
| H100 PCIe peak, dense bf16 | ~756 TFLOP/s |
| 30% MFU | ~227 TFLOP/s |
| Training FLOPs per token, 124M at 1024 context | ~0.79–0.85 GFLOP (6 × 122.7M matmul params, plus attention 6–12 × L × T × d) |
| **Throughput needed** | **≥ ~270k tokens/s** |
| 5B tokens at that speed | **~5 h** (at 40% MFU, ~4 h) |
| Today (40M, Mac, `logs/train_200.log`) | ~8k tokens/s |

Update: the run uses an AWS p5.4xlarge, which has the **H100 SXM** (~989 TFLOP/s dense bf16), so
30% MFU is ~297 TFLOP/s = **≥ ~349k tokens/s** at 0.85 GFLOP/token, ~4 h for 5B tokens.

A 124M model at 30–45% MFU on an H100 is known to be reachable (nanoGPT / llm.c territory), but
only with long sequences, large micro-batches, `torch.compile`, bf16 and no host syncs in the inner
loop. The current code misses several of these (section 5).

## 0. Decisions

1. **Context length: 1024** (decided 2026-10-16). At 256 the matmuls are too small to keep an H100
   busy. 2048 was ruled out: holdout BPB and the five benchmarks don't reward context past 1024, most
   documents are shorter (python_edu ~510, fineweb_edu ~1,100 tokens/doc), and GPT-2 124M, the
   baseline, was trained at 1024. Changes `mix.yaml` `sequence_length` and `config.py`
   `context_length` together (`train.py` checks they match). `global_batch_size: 524288` stays valid
   (= 512 × 1024).
2. **Holdout BPB on 1024-token windows** (decided 2026-10-16; replaces the 256-token fixed decision in
   `CLAUDE.md`), so training context and BPB window are the same. Today the window follows the
   model's context (`train.py:182` passes `model.context_length`, `eval_bpb.py:149` uses
   `MODEL_CONFIG.context_length`). Pin it to a constant `BPB_WINDOW = 1024` in `eval_bpb.py`, so a
   config change can't silently change the metric, and refuse models whose context is shorter.
   **Done**: `eval_bpb.py` `BPB_WINDOW`; `train.py` refuses a shorter model before training.
   Consequences:
   - The 40M runs' BPB (256-token windows, `logs/train_200.log`) is **not comparable** with the new
     numbers, and the 40M model can't be re-scored at 1024: it was trained at 256 context and its RoPE
     table only covers 256 positions. Compare old and new models through the benchmarks instead.
   - BPB at 1024 windows comes out lower than at 256 for the same model, because fewer tokens are
     scored with little context. A drop between old and new numbers is partly this, not the model.
   - The frozen eval sets themselves don't change (~2M tokens each, ~1,950 windows per set).
3. **Benchmarks stay `--device mps`** (fixed decision, `run_benchmarks.py:26`). Simplest: copy the
   exported 124M back to the Mac and benchmark there, so it is comparable to gpt2 / random_init /
   the 40M run. Running lm-eval on the H100 instead would need the decision changed and every
   model re-run with the new settings.
4. **Mix ratios**: keep the planned 0.60 / 0.20 / 0.13 / 0.07 unless a mix ablation is run first
   (section 7).

## 1. Download: `download.py` is pilot-only

Tokens needed for a 5B mix vs what is tokenized now:

| source | share | needed | tokenized now | coverage | raw to add (estimate from pilot) |
|---|---|---|---|---|---|
| fineweb_edu | 0.60 | 3.00B | 0.48B (1 shard) | 0.16 | ~6–9 more shards of 50 (~8–12 GB) |
| python_edu | 0.20 | 1.00B | 0.13B (250k blobs) | 0.13 | ~1.75M more blobs from S3 |
| finemath | 0.13 | 0.65B | 0.17B (1 shard) | 0.26 | ~3 more shards of 64 (~0.9 GB) |
| gutenberg | 0.07 | 0.35B | 0.38B (1 shard) | 1.09 | none needed; 1 more shard for headroom |

Add ~1% on top: `val_instances` are taken out of the mix (8192 × 1024 = 8.4M tokens at 1024
context), and OLMo-core rounds to whole batches.

What has to change:
- [x] **Download more than the first file.** `download.py:165` takes `listings[name][0]` only, and
      the disk check (`:153`) only sizes that file. Needs a `--num-files` (or `--target-tokens`)
      option, a loop over files, and `manifest.json` recording every file.
- [x] **Never download eval-only shards.** Taking the first N files avoids them today (they are the
      last shards), but the script should refuse them explicitly: fineweb_edu `000_00049.parquet`,
      finemath `train-00063-of-00064.parquet`, gutenberg `project_gutenberg-dolma-0014.json.gz`.
- [x] **python_edu: fetch past the pilot.** `PILOT_BLOBS = 250_000` is hard-coded (`:29`, `:103`).
      Need ~2M blobs: fetch metadata rows in ranges (e.g. 250k per output file, written as
      `content/rows-0000000-0250000.parquet`, ...), skip rows already fetched, and **exclude the
      8,000 eval blob_ids** listed in `data/raw_eval/python_edu.parquet`. `train-00000-of-00002`
      has 3,839,224 rows, so ~2M rows fit in it, well before the eval rows at its end.
- [x] **Remove or ignore `data/raw/python_edu/content/pilot-5000.parquet`.** It duplicates the first
      5,000 blobs of `pilot-250000.parquet`. Today `tokenize_data.py` only reads the first file
      in sort order, so it is skipped by luck; once all files are tokenized it would put duplicates
      in the mix.
- [x] Download speed in the pilot was ~3.5 MB/s for fineweb_edu (`manifest.json`: 1.39 GB in 396 s),
      so ~1 h for fineweb_edu. S3 fetch time for ~2M blobs should be measured on a short run first.
      **Done** on EC2: the full download took ~9 min (fineweb_edu ~171 MB/s; python_edu ~7,400 blobs/s).

## 2. Tokenize: `tokenize_data.py` is pilot-only

- [x] **Tokenize every raw file, not just the first.** `tokenize_data.py:115` has
      `input_files(name)[:1]`.
- [x] **`meta.json` must list all files of a source**, including ones skipped because their `.npy`
      already exists. It is written from this run's jobs only. `mix.py` refuses a folder whose `.npy`
      files don't match `meta.json`, which catches a mismatch but doesn't fix it.
- [x] **Parallelism is one process per file.** Fine for many fineweb/finemath shards, but python_edu
      as one huge parquet would tokenize on a single core. Splitting it into ~250k-blob files when
      downloading (above) fixes this.
- [x] Re-run `verify_tokenized.py` on every new token file.
- [x] Disk: ~10 GB of uint16 tokens + ~15 GB new raw data. The Mac has ~96 GB free, so it fits.
      **Done** differently: downloaded and tokenized on EC2; raw and tokens (~71 GB) are in S3 `nano-mix`.

## 3. Mix: `mix.yaml`, `mix.py`, `verify_mix.py`

- [x] `mix.yaml`: `requested_tokens: 5000000000` (plus the ~1% val headroom), `sequence_length: 1024`.
- [x] Re-run `mix.py` (coverage check + build) and `verify_mix.py` (all six checks must pass).
      Check 6 iterates every instance in Python: ~4.9M instances at 1024, ~2.5× today's 1.95M, so
      expect it to take ~2.5× as long.
- [x] `mix.md` documents the 500M / 256 pilot. Update its `mix.yaml` table once the values change.
- No code change needed in `mix.py`: it already globs every `.npy` per source, refuses eval shards
  and refuses any source that would have to repeat.

## 4. Model: `config.py`, `model.py`

- [x] **124M config** (done: `config.py` `MODEL_CONFIG_124M`, `--model 124m`): `emb_dim=768, n_heads=12, n_layers=12` with SwiGLU (hidden 2048), tied
      49,152-vocab embedding, `context_length=1024`. Parameter count: embedding 37.7M +
      12 × 7.08M blocks = **~122.7M**, GPT-2-small size. Head dim stays 64.
      Add it as a second named config (e.g. `MODEL_CONFIG_124M`) instead of editing the 40M
      defaults, so the 40M/Mac setup keeps working and `export_hf.py` can load either checkpoint.
- [ ] Optional, a few % speed: fuse `W_query/W_key/W_value` (`model.py:112-114`) into one Linear
      so it runs as one matmul. Changes the checkpoint layout, so `export_hf.py` must split it again.
- [ ] Optional, small quality: **attention within documents only.** An instance can hold several
      documents joined by EOS, and attention crosses them today (as in GPT-2/3, nanoGPT and OLMo's
      default). Llama 3 masks attention to the current document. Possibly a small gain with short
      documents (python_edu ~510 tokens), at the cost of variable-length attention (document
      boundaries from EOS positions). Not planned for the first 124M run.
- Not needed: GQA (`model.py` has none; at 124M it saves little), KV cache.
- Clean-up: `delta_backend` / `delta_chunk_size` in `config.py` are unused by `model.py`. **Done**
  (2026-10-20): removed, with the uv `main.py` stub; `requirements.txt` re-exported.

## 5. Training loop: what keeps the H100 below 30% MFU

These are in `train.py` and `dataloader.py`, ordered by how much they cost.

- [x] **Validation every 10 steps is ~15–20% of the run.** `train.py:251` runs `evaluate_model`
      every `eval_freq=10` steps with `eval_iter=20` batches from *both* loaders. Over ~9,500 steps
      that is ~2.5B tokens of forward passes. Set it to every ~250 steps, and print a cheap train-loss
      line every 10–50 steps instead.
- [x] **Each eval restarts DataLoader workers.** `calc_loss_loader` (`train.py:147`) iterates the
      loaders fresh each time; with `num_workers=4` and no `persistent_workers`, that spawns 8
      processes (each pickling the OLMo-core dataset) every eval. Use a fixed in-memory val batch or
      `persistent_workers=True` (`dataloader.py:40`).
- [x] **A GPU→CPU sync on every micro-batch.** `running_loss += loss.item()` (`train.py:238`) blocks
      until the GPU finishes, so the CPU can't queue the next micro-batch. Sum on the GPU and call
      `.item()` only when logging. Same for `grad_norm ... .item()` (`:244`) every step.
- [x] **Sampling under `torch.compile`.** `model.compile()` (`:330`) compiles the model in place;
      `generate_text_simple` (`:31`) then calls it with a new sequence length for each generated
      token. Check the log for recompiles. If they happen, sample from an uncompiled copy of the
      weights.
- [ ] **Logits memory caps the micro-batch.** `cross_entropy` (`:134`) on 49,152-way logits: at a
      micro-batch of 64 × 1024, bf16 logits are ~6.4 GB, the fp32 upcast ~13 GB, plus its gradient.
      Expected fit on 80 GB: micro-batch 32–64 × 1024 with `grad_accum_steps` 16–8, for 524,288
      tokens/step. A fused linear + cross-entropy kernel (e.g. Liger) would allow 128. Measure with
      `torch.cuda.max_memory_allocated()` (now in every log line).
      **Fused linear + cross-entropy** ([Liger Kernel](https://github.com/linkedin/Liger-Kernel) or
      Apple's Cut Cross-Entropy) is the one Triton kernel worth adding: the output head is ~30% of
      the FLOPs at 124M, and the logits (32 × 1024 × 49,152: ~3 GB bf16 + ~6 GB fp32, plus their
      gradient) are written out in full. A fused kernel never stores them: an estimated 5–15% tok/s,
      and room for micro-batch 64–128. **Only if the smoke run is under ~35% MFU.** Check that it works
      with `torch.compile`, and that its loss matches `F.cross_entropy` on the same batch.
- [ ] Try `torch.compile(mode="max-autotune")` in a second smoke run: a few % (tuned matmul tiles,
      CUDA graphs), but several minutes of compiling instead of ~1.
- Not worth it at this size: hand-written Triton for norms / SwiGLU / RoPE (`torch.compile` already
  generates fused Triton kernels for them); FlashAttention-3 / cuDNN attention (attention is ~13% of
  FLOPs at T = 1024, and SDPA already picks a Hopper backend); FP8 matmuls (small gain at width 768,
  numerics risk); Muon (a different optimizer experiment, not a speed fix).
- [x] **Fused AdamW**: `torch.optim.AdamW(..., fused=True)` (`:332`).
- [ ] **Holdout BPB** uses `settings.batch_size` and runs in fp32 on 1024-token windows (decision 2),
      on the **uncompiled** model (the training loop's compiled wrapper is only used for training
      steps), so the last batch's different shape no longer recompiles anything. Cost not yet in the
      time estimate: ~10M tokens of eager fp32 forward per eval, maybe 30–60 s on the H100, so 20
      evals add roughly 10–20 min to the run. Measure in the smoke run (`[eval ...] (Ns)`).
- [x] **Holdout BPB on the H100 isn't true fp32.** `train.py` sets `torch.backends.cuda.matmul.
      allow_tf32 = True` for training, and it stays on during `evaluate_bpb`, so H100 BPB uses TF32
      matmuls while the Mac uses full fp32: ~0.1% apart. Small, but it breaks the "fp32, comparable
      across runs" fixed decision. Fix: switch TF32 off around `evaluate_bpb` (and back on after).
      **Done** (2026-10-20): the TF32 switch is removed from `train.py`. It bought training nothing
      (all training matmuls are bf16 under autocast, forward and backward); its only effect was on
      BPB. Cost: each holdout eval runs true fp32 matmuls, slower than TF32 (estimate ~1 min instead
      of ~10 s, ~20 min over 20 evals; see the BPB cost item above, measure in the smoke run).
- [x] **No MFU number.** Only tok/s is printed. Add MFU = tok/s × FLOPs/token ÷ peak (989e12 on
      the p5's SXM H100, 756e12 on PCIe), and peak memory, to each log line. This is the number that shows whether the target is met.

## 6. Hyperparameters for the 124M run

All `TrainSettings` defaults are for a quick Mac test. **Done**: `config.py` `H100_SETTINGS`
(`train.py --preset h100`) with the values below (32 × 16; weight decay on matrices only, betas
(0.9, 0.95) in both presets). Peak memory and whether 64 × 8 fits come from the smoke run:

| setting | now | 124M / 5B (starting point) |
|---|---|---|
| tokens per step | 8,192 | 524,288 (= `global_batch_size` in `mix.yaml`) |
| `max_steps` | 500 | ~9,500 (5B ÷ 524,288, minus val) |
| `batch_size` × `grad_accum_steps` | 32 × 1 | 32 × 16 or 64 × 8 (whatever fits; product 512 at 1024 context) |
| `learning_rate` | 5e-4 | 6e-4 (GPT-2-small); try 1e-3–3e-3 in short sweeps |
| `warmup_steps` | 5 | ~200–500 |
| AdamW betas | (0.9, 0.999) default | (0.9, 0.95) |
| weight decay | 0.1 on **all** params | 0.1 on matrices only; none on norms, biases, embedding |
| `eval_freq` | 10 | ~250 |
| `eval_every_tokens` | 25M | 250M (20 holdout evals in the run) |

5B tokens is ~40 tokens per parameter: about 2× Chinchilla-optimal for 124M, a reasonable target.

- [ ] **No learning-rate sweep yet.** 6e-4 is GPT-2 small's value and probably conservative at
      524k tokens/step with betas (0.9, 0.95). Either accept it as the baseline, or run 3–4 short runs
      (~500 steps each, ~15–20 min on the H100 with compile and the final holdout eval, so ~1–1.3 h
      for four, e.g. 6e-4 / 1.2e-3 / 2e-3 / 3e-3 with
      `--set learning_rate=... warmup_steps=50 lr_schedule=constant --max-steps 500 --tag lr...`) and
      compare loss at step 500 (constant LR, so a 500-step cosine doesn't decay it away; pick the lowest
      smooth one, then back off a notch).
      **Decision (2026-10-20): stay at 6e-4 for now.** Revisit after the smoke run, setup and clean-up
      are done; worth doing before runs that get compared with each other (mix ablations), which
      should all share one LR.

## 7. Running on a rented H100 (missing entirely)

- [x] **Checkpoints and resume.** **Done** (`checkpointing.py`, `train.py --resume`): every 30 min, on
      Ctrl-C / SIGTERM / spot notice and at the end, latest two kept, copied to S3; resume tested
      bit-exact on CPU. Before: `model.pth` was saved only at the very end (`train.py:416`).
      **On the GPU it isn't bit-exact**, and doesn't need to be: some CUDA backward kernels (embedding
      gradient, attention backward) add in an order that varies between runs, so any two GPU runs,
      interrupted or not, differ in the last bits and drift slightly. A resume continues the data,
      weights, optimizer state and schedule exactly; only that kernel-level noise differs.
      A ~6 h rented run needs a checkpoint every ~30 min: model, optimizer, step, tokens seen, RNG
      state. The data order is a fixed seeded permutation (`dataloader.py:35`), so resuming only needs
      to skip `step × grad_accum_steps × batch_size` instances.
- [ ] **Check on resume that the data continues exactly where it stopped.** Today this rests on
      reasoning, not a check. The chain: OLMo-core sorts each source's glob
      (`olmo_core/io.py:505`); the per-file instance allocation depends only on file sizes,
      `requested_tokens`, ratios and batch size; the order is `torch.randperm` with `mix.yaml`'s seed
      (`dataloader.py`); a resume skips `step × grad_accum_steps × batch_size` instances; and
      `check_resume` refuses a changed `mix.yaml` or batch settings. `verify_mix.py` check 5 tests
      determinism only within one box. Not caught today on a new box: a token file that changed size
      (re-tokenized, with a matching new `meta.json`), a different torch version (`randperm`), or a
      different OLMo-core version (allocation). Add two values to the checkpoint and compare them on
      `--resume`, refusing on any mismatch:
      - **mix fingerprint**: a hash of every file's (source/name, instance range) from
        `dataset.paths` / `dataset.offsets`, plus the seed and the val split;
      - **next-batch checksum**: the mix indices of the next ~4 training instances and a hash of their
        tokens, which the resumed loader's first instances must match.
- [x] **Metrics to a file** (JSONL/CSV: step, loss, lr, grad norm, tok/s, MFU, BPB per set), not only
      stdout. The loss plot (`loss.pdf`) is drawn only at the end.
- [ ] **Linux CUDA environment with uv.** Check that `uv.lock` resolves a CUDA build of torch on
      linux-x86_64 (`uv sync` then `torch.cuda.is_available()` and an SDPA FlashAttention check).
      `caffeinate` is macOS-only; on Linux use `tmux` alone.
      Half done: `uv.lock` resolves torch 2.13.0 with CUDA 13.0 wheels on linux-x86_64 (needs driver
      >= 580). The check on the box itself is in `STARTUP_COMMANDS.md` section 3.
- [x] **Moving data to the box.** Build the mix on the Mac (CPU only, per CLAUDE.md), then copy
      `data/tokenized/` (~10 GB), `data/eval/` and the tokenizer files. `data/mix/` is rebuilt from
      `mix.yaml`, since `mix.py` resolves absolute source paths. Set `HF_HUB_OFFLINE=1` or pre-download
      the tokenizer on the box.
      **Done** differently: data goes S3 → GPU box (same region, free) and the mix is built on the box;
      commands in `STARTUP_COMMANDS.md`.
- [ ] **A short smoke test before the full run**: ~200 steps at full size to measure tok/s, MFU and
      peak memory, and to confirm no recompiles, before paying for 6 h.
- [ ] Optional, before spending the 6 h: a mix ablation at small scale (e.g. 40M on 1B tokens, 2–3
      ratio settings compared on holdout BPB) to back the 0.60 / 0.20 / 0.13 / 0.07 choice.

## 8. After training

- [ ] `export_hf.py` with the 124M config: it writes `max_position_embeddings = context_length`
      (1024) and checks logits within 1e-3. Output e.g. `exports/fw60_py20_fm13_gb07_s1234_124m`.
      **Done**: `train.py` `run_name` ends in the size (`..._s1234_124m`); `export_hf.py --model`.
- [ ] `run_benchmarks.py` on the Mac (decision 3), then `--summary` next to gpt2 (124M, the natural
      comparison), SmolLM2-135M and the 40M run.

## Suggested order

1. Decisions (section 0): context 1024 and BPB window 1024 are decided; benchmarks stay on MPS.
2. Extend `download.py` and `tokenize_data.py`, fetch and tokenize the data, run `verify_tokenized.py`.
3. `mix.yaml` (1024) → `mix.py` → `verify_mix.py`.
4. Add the 124M config, pin the BPB window to 1024, and make the training loop fixes, checkpointing
   and MFU logging.
5. Short smoke run on the Mac with the 124M config at a tiny step count, then on the H100.
6. Full run, export, benchmark.
