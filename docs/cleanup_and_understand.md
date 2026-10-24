# Cleanup and understand

This document has two parts:

1. **Cleanup.** What is wrong with the repository now, the target structure, and the steps to get
   there.
2. **Must-know list.** The things to understand about this training run, as a checklist.

The text follows ASD-STE100 (Simplified Technical English) for most of the content: short
sentences, active voice, one instruction in each sentence, and one word for one meaning.
Technical names (file names, settings, PyTorch terms) stay as they are.

Status: 2026-10-21. The first 124M run (5B tokens, one H100 PCIe on Vast) is ready to start.

Cleanup status, 2026-10-24 (branch `cleanup`): steps 1 to 6 and most of step 7 are done.
The code is in `src/nanomix/`, the docs are in `docs/`, and Part 2 below uses the new file names.
Part 1 describes the repository as it was before. Open: the step 7 item "the trainer writes its
own console log" (`docs/roadmap.md` section 3; its test is in `tests/test_checkpoint.py`, marked xfail).

---

## Part 1. Cleanup

### 1.1 Do not clean up during the full run

The Vast box runs the code from its own `git clone`. Changes on the Mac do not affect a run that
already started. But do not push file moves before the run ends. A resume on a new box clones the
latest `main`. If the file names changed, the commands in `STARTUP_COMMANDS.md` fail.

Checkpoints do not keep Python module paths. They keep only tensors and plain values. Thus, you
can move and rename modules later, and old checkpoints still load.

### 1.2 Problems now

| # | Problem | Where | Effect |
|---|---|---|---|
| P1 | 17 scripts are in the root folder, with no package structure. | root | It is difficult to see which step of the pipeline a file is for. |
| P2 | `train.py` has 684 lines and does 12 different jobs. | `train.py` | It is difficult to find and to change one part. |
| P3 | Some names come from a textbook and are not correct now: `train_model_simple`, `generate_text_simple`, `calc_loss_batch`. | `train.py` | The names say "simple", but the code has checkpoints, MFU, W&B and alerts. |
| P4 | The same facts are in more than one file. | see 1.3 | A change in one file and not in the other gives errors that you do not see. |
| P5 | `mix.py` imports `checkpointing.py` (for `MIX_KEYS`). | `mix.py` | The data step depends on the training step. The direction is wrong. |
| P6 | Seven Markdown documents overlap. Some text is old. | root `*.md` | You do not know which document is correct. |
| P7 | `MISSING.md` mixes done items, open items and history. Its title says "PCIe" and then says "SXM". | `MISSING.md` | It is difficult to see what is still open. |
| P8 | `STARTUP_COMMANDS.md` is for AWS p5 (spot, SXM). The run now uses Vast (PCIe). | `STARTUP_COMMANDS.md` | Expected values (989 TFLOP/s, 349k tok/s) are wrong for this GPU. |
| P9 | The commands use `tee -a`. Ctrl-C stops `tee`, and the run then dies before the save. | docs | A stop can lose the last checkpoint (see `CLAUDE.md`). |
| P10 | `logs/` is not in `.gitignore`. Four old log files are in git. | `.gitignore` | New logs and metrics files can go into git by accident. |
| P11 | `loss.pdf` goes into the root folder, and each run writes over it. | `train.py` `plot_losses` | You lose the plot of the previous run. |
| P12 | There is no `tests/` folder. Checks were done with temporary scripts. | — | You cannot run the checks again after a change. |
| P13 | `requirements.txt` is next to `uv.lock`. | root | Two dependency lists. `CLAUDE.md` says: use uv only. |
| P14 | `config.py` calls the 40M config `MODEL_CONFIG` and the Mac settings `TRAIN_SETTINGS`. | `config.py` | The default names do not say "40M" or "Mac". |
| P15 | Two functions print text samples, with different prompts. `generate_and_print_sample` uses the textbook prompt "Every effort moves you" (50 tokens). `print_holdout_eval` uses `SAMPLE_PROMPTS` (prose, code, math; 100 tokens). | `train.py` | Two sample formats in one log. The textbook prompt does not test this mix. |

### 1.3 Facts that are in more than one file (P4)

| Fact | Files |
|---|---|
| Source list (repo, subset, columns) | `download.py` `SOURCES`, `tokenize_data.py` `SOURCES` |
| Eval-only shards | `download.py` `EVAL_ONLY`, `make_eval_sets.py` `EVAL_SHARDS` |
| Tokenizer name | `tokenize_data.py` `TOKENIZER`, `mix.yaml` `tokenizer`, `CLAUDE.md` |
| Eval paths `data/eval`, `data/raw_eval` | `eval_bpb.py`, `make_eval_sets.py` |
| Run name | `train.py` `run_name`, `run_benchmarks.py` `run_name` |
| Peak TFLOP/s of each GPU | `train.py` `PEAK_TFLOPS`, `MISSING.md`, `STARTUP_COMMANDS.md` |

### 1.4 Target structure

```
nano-mix/
├── README.md                 what the project is, quick start, links to docs/
├── CLAUDE.md                 rules for Claude (short; points to docs/)
├── pyproject.toml  uv.lock  .python-version
├── configs/
│   └── mix.yaml              (later: one file for each mix ablation)
├── src/nanomix/              the code, as one package
│   ├── paths.py              all folder paths (data/raw, data/eval, ...)
│   ├── sources.py            SOURCES, EVAL_ONLY, tokenizer name: one place only
│   ├── data/
│   │   ├── download.py       verify_download.py
│   │   ├── tokenize_data.py  verify_tokenized.py   (never name a file tokenize.py)
│   │   └── mix.py  trim_mix.py  verify_mix.py  loader.py (was dataloader.py)
│   ├── model/
│   │   ├── config.py         ModelConfig, MODEL_CONFIGS ("40m", "124m")
│   │   └── gpt.py            (was model.py)
│   ├── train/
│   │   ├── settings.py       TrainSettings, presets ("mac", "h100")
│   │   ├── loop.py           the training loop only
│   │   ├── optim.py          param_groups, lr_at_step
│   │   ├── precision.py      device, bf16 autocast
│   │   ├── checkpoint.py     Checkpointer, StopRequest, check_resume, MIX_KEYS
│   │   ├── metrics.py        MetricsLog, W&B, MFU, alerts, console log file
│   │   └── cli.py            argument parsing and main()
│   └── eval/
│       ├── make_eval_sets.py  bpb.py  sample.py
│       ├── export_hf.py       benchmarks.py (was run_benchmarks.py)
├── tests/                    pytest: small, fast, CPU only
├── docs/
│   ├── pipeline.md           download → tokenize → mix (with the mix.md content)
│   ├── training.md           model, loop, settings, checkpoints
│   ├── eval.md               (was EVAL_AND_BENCHMARK.md)
│   ├── run_on_gpu.md         (was STARTUP_COMMANDS.md; Vast and AWS)
│   ├── decisions.md          fixed decisions, with the date of each
│   └── roadmap.md            open items only (from MISSING.md)
├── results/                  benchmark JSON and summary.md (in git)
└── data/ checkpoints/ logs/ exports/ wandb/   (not in git)
```

Rules for the new structure:

- Each pipeline step reads only the output of the step before it (as now).
- `data/` code does not import `train/` code. `MIX_KEYS` moves to `data/mix.py`, and
  `train/checkpoint.py` imports it from there.
- Each fact in 1.3 has one place. Other files import it.
- Scripts run as `uv run python -m nanomix.train.cli ...`, or as short commands in
  `pyproject.toml` `[project.scripts]` (for example `uv run nanomix-train`).
- `docs/` keeps the "why". The code comments keep the "how". `CLAUDE.md` keeps only rules.

### 1.5 Cleanup steps

Do the steps in this sequence. Do one step in one commit. After each step, do the check.

**Step 0. Before you start.**
1. Wait until the full run is complete and its checkpoint is in S3.
2. Make a branch: `git switch -c cleanup`.

**Step 1. Safety fixes (no file moves).**
1. Add `logs/` to `.gitignore`. Remove the old logs from git with `git rm --cached logs/*.log`.
   Keep the files on disk.
2. Change `tee -a` to `tee -ia` in all docs (P9).
3. Write `loss.pdf` to `logs/<run_id>_loss.pdf` (P11).
4. Check: `git status` shows no log files. The smoke command in the docs uses `tee -ia`.

**Step 2. Tests first.**
Add `tests/` before you move code. The tests tell you if a move breaks something.
1. `uv add --dev pytest`.
2. Add these tests (CPU only, small data, each less than 30 s):
   - `lr_at_step`: warmup, peak, cosine end value.
   - `param_groups`: no weight decay on norms, biases and the embedding.
   - Model: the output shape, and the parameter count for "40m" and "124m".
   - Loss: the last position of each instance is not scored.
   - Mix: a full build and a trimmed build give the same instances (the test from the
     `trim_mix.py` work).
   - Checkpoint: save, load and `check_resume` refuse a changed setting.
   - `StopRequest`: SIGINT and SIGTERM save a checkpoint, also with stdout closed.
   - BPB: a random-init model gives a BPB near the uniform value.
3. Check: `uv run pytest` passes.

**Step 3. One place for each fact (P4, P5).**
1. Make `sources.py` and `paths.py`. Import them in all files from 1.3.
2. Move `MIX_KEYS` from `checkpointing.py` to `mix.py`.
3. Check: `uv run pytest`, then `uv run python mix.py` and `verify_mix.py` on the Mac test data.

**Step 4. Split `train.py` (P2, P3).**
1. Move the functions into `optim.py`, `precision.py`, `metrics.py`, `loop.py` and `cli.py`.
2. Give the functions correct names. For example: `train_model_simple` → `train`,
   `calc_loss_batch` → `batch_loss`, `generate_text_simple` → `greedy_generate`.
3. Use one sample function with `SAMPLE_PROMPTS` only (P15).
4. Do not change the training behavior in this step. Only move and rename.
5. Check: `uv run pytest`. Then do a 20-step run on the Mac with `--preset mac`. Compare its
   losses with a 20-step run from before the change. They must be equal.

**Step 5. Package structure (P1).**
1. Move the files into `src/nanomix/` (see 1.4).
2. Add `[project.scripts]` in `pyproject.toml`.
3. Check: `uv run pytest`, and each command in the docs runs.

**Step 6. Docs (P6, P7, P8).**
1. Move the docs into `docs/` (see 1.4).
2. Put the done items of `MISSING.md` in `docs/decisions.md` or delete them. Keep only open
   items in `docs/roadmap.md`.
3. Make `run_on_gpu.md` correct for Vast and PCIe. Keep the AWS part as a second section.
4. Make `CLAUDE.md` short: rules and links only.

**Step 7. Small items.**
1. Delete `requirements.txt` if no box needs it (P13). If a box needs pip, write the reason in
   `README.md`.
2. Rename `MODEL_CONFIG` → `MODEL_CONFIG_40M` and `TRAIN_SETTINGS` → `MAC_SETTINGS` (P14).
3. Do the `MISSING.md` item "train.py writes its own console log; no `tee`".

---

## Part 2. Must-know list

Use this list to learn the run. For each item, read the file. Then explain the item in your own
words. Tick the item when you can answer the question without the file.

Facts in this list are for the 124M run: 122.7M parameters, 1024-token context, 524,288 tokens
for each step, 9,617 steps, 5.04B tokens, one H100 PCIe.

### 2.1 Data and mix

- [ ] **The four sources and their shares.** fineweb_edu 0.60, python_edu 0.20, finemath 0.13,
      gutenberg 0.07. Question: what does each source teach the model?
      Read: `CLAUDE.md` (Sources), `configs/mix.yaml`.
- [ ] **One document in the token files.** Each document is its tokens, then one EOS token (id 0).
      The files are flat uint16 arrays with no header. Question: why is uint16 sufficient?
      Read: `data/tokenize_data.py`.
- [ ] **An instance.** An instance is 1024 tokens in a row from one file. It can contain the end of
      one document and the start of the next. Attention goes across the EOS token.
      Question: what is the effect on short documents (python_edu, ~510 tokens)?
      Read: `docs/pipeline.md`, `docs/roadmap.md` section 4 ("attention within documents only").
- [ ] **How OLMo-core picks the tokens.** It takes the same fraction from each file, and it reads
      the start of each file. Question: why does `data/trim_mix.py` need `manifest.json`?
      Read: `data/mix.py` `build_dataset`, `FixedSourceMixture`, `data/trim_mix.py`.
- [ ] **No repetition.** The run reads each instance one time only (one epoch). Question: why is
      dropout 0.0 correct here? Read: `model/config.py` `drop_rate`, `data/verify_mix.py` check 3.
- [ ] **The data sequence.** `torch.randperm` with the seed from `configs/mix.yaml`. The first 8,192
      instances of that sequence are the validation set. Question: what does a resume skip?
      Read: `data/loader.py`.

### 2.2 Tokenizer

- [ ] **Which tokenizer.** `HuggingFaceTB/SmolLM2-135M`: byte-level BPE, vocabulary 49,152.
      Question: why use the tokenizer of an existing model, and not train a new one?
- [ ] **Special tokens.** EOS is id 0. There is no BOS token. The code encodes with
      `add_special_tokens=False`. Question: where does the code add EOS?
      Read: `data/tokenize_data.py`.
- [ ] **Bytes for each token.** BPB needs the UTF-8 byte length of each token. Special tokens have
      length 0. Question: how does `eval/bpb.py` get the byte length of a token?
      Read: `eval/bpb.py` `token_byte_lengths`.
- [ ] **The 21 dropped byte values.** 21 byte values have no SmolLM2 token. Tokenization drops
      them, and BPB does not count them. Read: `CLAUDE.md`, `eval/bpb.py` `unencodable_bytes`.
- [ ] **Vocabulary and speed.** 49,152 is 384 × 128. Thus, `pad_vocab_to_multiple_of=128` adds no
      rows. Question: why do matmul sizes that are a multiple of 128 run faster?
      Read: `model/config.py`, `model/gpt.py` `mask_padded_logits`.

### 2.3 Model

- [ ] **The shape.** 12 layers, width 768, 12 heads, head size 64, SwiGLU hidden size 2048, tied
      embedding. Question: how do you get 122.7M parameters from these numbers?
      Read: `model/config.py` `MODEL_CONFIG_124M`, `model/config.py`.
- [ ] **RoPE.** Positions are rotations of q and k, with base 10,000. There is no position table.
      Question: why can the model not read more than 1024 tokens well? Read: `model/gpt.py`
      `RotaryEmbedding`.
- [ ] **RMSNorm and pre-norm.** Each block normalizes before attention and before the FFN.
      Read: `model/gpt.py` `TransformerBlock`.
- [ ] **SwiGLU.** Three matrices, not two. Question: why is the hidden size 8/3 × width?
      Read: `model/gpt.py` `SwiGLUFeedForward`, `swiglu_hidden_dim`.
- [ ] **Tied embedding.** The output head uses the same matrix as the token embedding.
      Question: how many parameters does this save at width 768?
- [ ] **Initialization.** Std 0.02, with a smaller value for the residual projections.
      Question: why does a deep model need the smaller value? Read: `model/gpt.py` `init_weights`.

### 2.4 Loss

- [ ] **Next-token cross-entropy.** The target is the input moved left by one token. The last
      position has no target (`IGNORE_INDEX = -100`). Thus, 1,023 of 1,024 positions are scored.
      Read: `data/loader.py` `MixInstances`, `train/loop.py` `batch_loss`.
- [ ] **The start value.** A model that guesses uniformly has loss ln(49,152) ≈ 10.80 nats.
      Question: the smoke run shows 9.63 at step 10. Is that correct?
- [ ] **Loss in fp32.** `cross_entropy` calculates the softmax in fp32, also under bf16 autocast.
- [ ] **Loss, perplexity and BPB.** Perplexity = e^loss. BPB = total nats / ln 2 / total bytes.
      Question: why can you compare BPB between tokenizers, but not loss?
- [ ] **Train loss and val loss.** Train loss is the mean of the steps since the last value. Val
      loss is on 20 fixed batches. Question: why is the val loss (5.03) less than the train loss
      (6.06) at step 200 of the smoke run? (Hint: the train value is a mean over all 200 steps.)

### 2.5 Training loop

- [ ] **Step and micro-batch.** One optimizer step = 16 micro-batches of 32 × 1024 tokens.
      Gradients add up in `.grad`. Question: why does the code divide the loss by
      `grad_accum_steps`? Read: `train/loop.py` `train`.
- [ ] **Gradient clipping.** The code clips the total gradient norm to 1.0, one time for each step,
      after accumulation. Question: what does `clip frac` in the log tell you?
- [ ] **Number of steps.** 4,923,904 training instances / 512 for each step = 9,617 steps.
      Read: `train/settings.py` `H100_SETTINGS`.
- [ ] **What happens at each interval.** Log line every 10 steps. Val loss every 250 steps.
      Holdout BPB every 250M tokens. Sample text every 1,000 steps. Checkpoint every 30 min.
- [ ] **Two models with the same weights.** The compiled model trains. The uncompiled model does
      the sampling and the BPB. Question: why? (Hint: other shapes cause recompiles.)

### 2.6 Optimizer and learning rate

- [ ] **AdamW.** betas (0.9, 0.95), `fused=True`. Question: why 0.95 and not 0.999?
      Read: `train/settings.py` `adam_betas`.
- [ ] **Weight decay.** 0.1 on weight matrices only. No decay on norms, biases and the embedding.
      Read: `train/optim.py` `param_groups`.
- [ ] **The LR schedule.** Linear warmup for 300 steps to 6e-4. Then cosine decay to 6e-5 at step
      9,617. Read: `train/optim.py` `lr_at_step`.
- [ ] **Why 6e-4.** It is the GPT-2 small value. There was no sweep. Read: `docs/decisions.md` (LR 6e-4).
- [ ] **Tokens for each step.** 524,288. Question: what happens to the correct LR if you make the
      batch larger?

### 2.7 Precision

- [ ] **bf16 autocast.** The forward pass and the loss run in bf16. Weights, gradients and AdamW
      states stay fp32. Read: `train/precision.py` `autocast_ctx`.
- [ ] **No GradScaler.** bf16 has the same exponent range as fp32. Question: why does fp16 need a
      GradScaler, and bf16 not?
- [ ] **No TF32.** All training matmuls are bf16, so TF32 does not help. TF32 only made the BPB
      less accurate. Read: `docs/decisions.md` (No TF32).
- [ ] **Eval in true fp32.** Holdout BPB runs in fp32, on the Mac and on the GPU, so the numbers
      are comparable. Question: what does this cost on the PCIe card? (105 s for each eval.)
- [ ] **Memory for each parameter.** fp32 weights 4 bytes + gradients 4 + AdamW states 8 = 16
      bytes. 122.7M × 16 ≈ 2 GB. Question: what uses the other ~28 GB of the 30 GB peak?

### 2.8 Speed optimizations and kernels

- [ ] **MFU.** MFU = tok/s × FLOPs for each token / peak FLOP/s. The run: 267k × 0.85 GFLOP /
      756 TFLOP/s = 30%. Read: `train/metrics.py` `training_flops_per_token`, `resolve_peak_flops`.
- [ ] **FLOPs for each token.** About 6 × parameters, plus attention. Question: why 6 (2 forward,
      4 backward)?
- [ ] **`torch.compile`.** It joins many small operations into fewer GPU kernels (Triton). The
      first step is slow (compilation). Question: what is a recompile, and why is one at the first
      val loss correct?
- [ ] **Attention kernel.** `F.scaled_dot_product_attention` selects FlashAttention on the H100.
      Question: why does FlashAttention use less memory?
- [ ] **No GPU→CPU sync in the loop.** The loop keeps losses on the GPU. It calls `.item()` only
      for a log line. Question: why does `.item()` make the GPU wait?
- [ ] **Data loading.** 8 workers, `persistent_workers`, `pin_memory`, `non_blocking` copies.
      Read: `data/loader.py`.
- [ ] **Fused AdamW.** One kernel for the update of all parameters.
- [ ] **What is not done yet.** Fused linear + cross-entropy (Liger), `max-autotune`, fused QKV,
      64 × 8. Read: `docs/roadmap.md` section 2.
- [ ] **The output head cost.** With a vocabulary of 49,152, the output head is ~30% of the FLOPs,
      and the logits use much memory. Question: why does this make Liger the largest possible
      speed-up?

### 2.9 GPU and hardware

- [ ] **PCIe and SXM.** PCIe: 756 TFLOP/s, 2.0 TB/s, 350 W. SXM: 989 TFLOP/s, 3.35 TB/s, 700 W.
      Question: why does memory bandwidth matter much for a 124M model?
- [ ] **The power limit.** `nvidia-smi -q -d PERFORMANCE` shows "SW Power Cap: Active" if the card
      is at its limit. Then the clocks are lower.
- [ ] **Driver and CUDA.** torch 2.13 with CUDA 13.0 needs driver 580 or higher. Read:
      `docs/run_on_gpu.md` section 2.
- [ ] **GPU-Util is not MFU.** `nvidia-smi` shows that a kernel was running. It does not show if
      the GPU did useful work.
- [ ] **Memory numbers.** `peak mem` in the log is `max_memory_allocated`. The real use is 1–3 GB
      more (CUDA context, cache). Question: does the run fit on a 40 GB A100?

### 2.10 Evals (most important)

- [ ] **The four evals and their jobs.**

      | Eval | Data | When | Job |
      |---|---|---|---|
      | Val loss | 8,192 held-out mix instances (20 batches used) | every 250 steps | Is training working? Same data distribution as training. |
      | Holdout BPB | 5 frozen sets, ~2M tokens each | every 250M tokens, and at the end | The main metric. Compare runs and tokenizers. |
      | Text samples | prose, code and math prompts (`eval/sample.py`) | every 1,000 steps and at the end | A fast human check. Greedy, so repetition is normal early. |
      | Benchmarks (lm-eval) | hellaswag, piqa, arc_easy, sciq, lambada_openai | after the run, on the Mac | Compare with GPT-2 and SmolLM2. |

- [ ] **The five holdout sets.** fineweb_edu, python_edu, finemath, gutenberg, and wikipedia_en.
      Wikipedia is not in the training mix. Question: what does the wikipedia_en BPB tell you that
      the others do not? Read: `docs/eval.md` section 1, `eval/make_eval_sets.py`.
- [ ] **Frozen sets.** Never make the sets again, and never edit them. Question: why does a change
      to an eval set make all old numbers useless? Read: `CLAUDE.md` (fixed decisions).
- [ ] **No contamination.** The eval sets come from eval-only shards. Training never reads those
      shards. `data/mix.py` refuses them. For python_edu, 8,000 blob_ids are excluded.
      Question: how does `eval/make_eval_sets.py` check for overlap? Read: `training_hashes`.
- [ ] **The BPB formula.** Total loss in nats / ln 2 / total UTF-8 bytes. Sum over the full set,
      then divide one time. Question: why is a mean of window BPBs not the same number?
      Read: `eval/bpb.py` `evaluate_bpb`.
- [ ] **BPB windows.** Non-overlapping 1024-token windows. The first tokens of a window have little
      context, so they have high loss. Question: why is BPB at 256 windows not comparable with BPB at
      1024 windows? Read: `docs/decisions.md` (BPB windows).
- [ ] **Good BPB values.** A random model gives log2(49,152) ≈ 15.6 bits for each token, divided by
      the mean bytes for each token of the set. The
      `resumetest` run at step 40: fineweb_edu 2.13, python_edu 3.09. Question: which sets are
      easy and which are difficult, and why?
- [ ] **Eval cost.** 105 s for each holdout eval on the PCIe card, ~37 min for the full run.
      Read: `docs/roadmap.md` section 2 (lite eval).
- [ ] **The lite eval plan.** A part of each set, often. The full set, less often. Stop only if BPB
      gets worse, not if BPB stops improving. Question: why does a cosine LR make a mid-run
      plateau normal? Read: `docs/roadmap.md` section 2.
- [ ] **Benchmark settings.** lm-eval 0.4.13, 0-shot, fp32, `--device mps`, batch 16, the same for
      every model. Question: what are `acc` and `acc_norm`? Read: `eval/benchmarks.py`,
      `docs/eval.md` section 5.
- [ ] **Answer BPB.** Nats of the gold answer / ln 2 / bytes of the gold answer, summed over a
      task. Question: why is it more stable than accuracy for a small model?
- [ ] **Chance level.** hellaswag 25%, piqa 50%, arc_easy 25%, sciq 25%. Question: a 124M model is
      often only a little above chance on some tasks. Which metric shows progress then?
- [ ] **Export check.** `export_hf.py` converts to `LlamaForCausalLM`. It refuses the result if the
      logits are more than 1e-3 from the original. Question: why must the export be correct before
      the benchmarks mean anything?

### 2.11 Monitoring

- [ ] **Where the numbers go.** Console, `logs/<run_id>.jsonl`, and W&B. The JSONL file goes to S3
      with each checkpoint. The console log does not (see `docs/roadmap.md` section 3).
- [ ] **What to watch in each log line.** loss, lr, grad norm, clip frac, tok/s, s/step, MFU,
      peak mem.
- [ ] **Normal values.** grad norm 0.3–0.7 after warmup. clip frac near 0 after warmup. tok/s
      steady at ~267k. Question: what does a grad norm spike tell you?
- [ ] **Alerts.** W&B alerts if the loss is not finite (the run stops without a save), and if
      tok/s falls below 50% of the best value. Read: `train/metrics.py` `MetricsLog`.
- [ ] **GPU checks.** `nvidia-smi` for memory, power and utilization, in a second tmux window.

### 2.12 Checkpoints, stops and resume

- [ ] **What a checkpoint has.** Weights, AdamW state, step, tokens seen, RNG states, loss
      history, and the settings. 1.62 GB. Read: `train/checkpoint.py` docstring.
- [ ] **When the code saves.** Every 30 min, on Ctrl-C, SIGTERM, SIGHUP or a spot notice, and at
      the end. It keeps the last two (`ckpt.pt`, `ckpt_prev.pt`). It copies to S3.
- [ ] **What a resume checks.** `check_resume` refuses a changed model, changed train settings
      (except the safe ones in `RESUME_SAFE`), or a changed mix. Question: why is `batch_size`
      not safe to change?
- [ ] **GPU resume is not bit-exact.** Some CUDA kernels add numbers in a different sequence on
      each run. The data and the schedule continue exactly. Read: `docs/decisions.md` (GPU resume).
- [ ] **The `tee` problem.** Use `tee -ia`. Read: `CLAUDE.md`.

### 2.13 Cost and infrastructure

- [ ] **The cost of the run.** ~5.9 h × $2.58/h ≈ $15. Question: which part of the run is not
      training, and how much does it cost?
- [ ] **Data transfer.** S3 → Vast: $0.09/GB after 100 GB free each month. The trimmed data is
      10.1 GB. Read: `docs/pipeline.md` (trimmed copy).
- [ ] **Credentials on a rented box.** A read-only GitHub token for one repo. An IAM user with
      access to the `nano-mix` bucket only. No SSH agent forwarding. Delete them after the run.

### 2.14 Reproducibility

- [ ] **What is fixed.** Seed 1234, `configs/mix.yaml`, the tokenizer, `uv.lock`, the frozen eval sets.
- [ ] **What is not fixed.** GPU kernel noise (see 2.12), and the GPU type (it changes speed, not
      the result).
- [ ] **The run name.** `fw60_py20_fm13_gb07_s1234_124m` = the mix shares in percent, the seed and
      the model size. Read: `train/cli.py` `run_name`.
