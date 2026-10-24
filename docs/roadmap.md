# Roadmap

Open items only. Done work is in git history; fixed decisions are in [decisions.md](decisions.md).

Baseline numbers (Vast H100 PCIe, 32 × 16, 2026-10-21 smoke run): 267k tok/s, 1.96 s/step,
**MFU 30.0%**, peak memory 30.0 GB, holdout BPB eval 105 s. The full run (9,617 steps) is ~5.2 h of
training + ~37 min of evals. Well-tuned GPT-2 124M runs reach ~40–50% MFU on SXM, so ~35–40% looks
reachable on PCIe.

## 1. After the first 124M run

- [ ] Export (`nanomix-export-hf`), holdout BPB on the Mac (`nanomix-eval-bpb`), benchmarks on the Mac
      (`nanomix-benchmarks`), then `--summary` next to gpt2, SmolLM2-135M and the 40M run.
- [ ] Benchmark SmolLM2-135M and pythia-160m (one `nanomix-benchmarks` command each).

## 2. Speed (one at a time, each with a short smoke run whose loss curve matches the baseline's)

Together ~1–1.5 h of a ~5.9 h run. They pay off in the LR sweep and the mix ablations.

- [ ] **Profile first.** A `torch.profiler` trace of ~5 training steps (after compile) gives the
      split between matmuls, attention, the loss (logits + cross-entropy), the optimizer and idle
      time. Also check `nvidia-smi -q -d PERFORMANCE` during training: "SW Power Cap: Active" means
      the PCIe card is at its 350 W limit and clocks are down, so kernel changes gain less.
- [ ] **Fused linear + cross-entropy** ([Liger Kernel](https://github.com/linkedin/Liger-Kernel) or
      Apple's Cut Cross-Entropy). The output head is ~30% of the FLOPs, and the logits
      (32 × 1024 × 49,152: ~3 GB bf16 + ~6 GB fp32, plus their gradient) are written out in full; a
      fused kernel never stores them. Expected 5–15% tok/s (~15–45 min of the run) and room for
      micro-batch 64–128. CUDA only (`uv add liger-kernel`), so the Mac keeps `F.cross_entropy`.
      Check it works with `torch.compile` and its loss matches `F.cross_entropy` on the same batch.
- [ ] **`torch.compile(mode="max-autotune")`**: a few % (~5–15 min), but several minutes of
      compiling instead of ~1, paid again on every resume.
- [ ] **Fuse `W_query/W_key/W_value`** (`model/gpt.py`) into one Linear: ~3–10 min. Changes the
      checkpoint layout, so loading (`load_model_weights`, `export_hf.py`, `bpb.py`) must accept both
      layouts, or convert old checkpoints.
- [ ] **64 × 8 instead of 32 × 16.** Same 524,288 tokens/step. 32 × 1024 peaked at 30.0 GB, so 64 × 8
      needs ~55–60 GB. 128 × 4 doesn't fit without the fused cross-entropy. Expected 0–5%. Test:
      `--max-steps 40 --set warmup_steps=5 batch_size=64 grad_accum_steps=8 --tag smoke64`; keep it
      only if steady tok/s is clearly above 267k. A run must be resumed with the same split.
- [ ] **Lite holdout eval + regression guard.** Evals cost ~37 min of the run (21 × 105 s).
      Stopgap: `--set eval_every_tokens=500000000` (resume-safe).
      - Lite eval every 250M tokens: the first 256 windows of each frozen set (~1.3M tokens in
        total), same BPB definition, logged as `bpb_lite/<set>` and compared only with lite numbers.
        ~13 s each.
      - Full eval every 1.25B tokens and at the end (~9 min in total).
      - Guard: stop if BPB gets **worse** (mean lite BPB more than `eval_stop_pct` = 2% above its best
        for 2 lite evals in a row, skipping the first 2), with a W&B alert and a checkpoint, through
        `StopRequest`, so a false alarm can be resumed. Not "stop on plateau": with a cosine LR, a flat
        stretch mid-run is normal. One source getting worse alone: alert only.
      - Files: `eval/bpb.py` (subset), `train/loop.py` (schedules, guard), `train/settings.py` (the
        settings, added to `RESUME_SAFE`).
- Not worth it at this size: hand-written Triton for norms / SwiGLU / RoPE, FlashAttention-3 /
  cuDNN attention, FP8 matmuls, Muon.

## 3. Robustness

- [ ] **The trainer writes its own console log; no `tee`.** `tee -ia` covers Ctrl-C only: a killed
      tmux session (SIGHUP) or a container stop (SIGTERM) still kills `tee`, and the trainer's next
      print dies before the save. The console log also never reaches S3. Fix: copy everything printed
      (stdout and stderr) to `logs/<run_id>.log` (appended on resume), ignore a closed stdout instead
      of dying, and add the log to `extra_uploads`. Then drop `| tee` from the docs. Test:
      `tests/test_checkpoint.py` `test_stop_signal_saves_with_stdout_closed` is marked `xfail(strict=True)`
      today; remove the mark when this is fixed.
- [ ] **Check on resume that the data continues exactly.** Today this rests on reasoning: sorted
      globs, an allocation that depends only on file sizes and the mix settings, a seeded
      `torch.randperm`, and `check_resume`. Not caught on a new box: a token file that changed size,
      or a different torch (`randperm`) or OLMo-core (allocation) version. Add to the checkpoint and
      compare on `--resume`: a **mix fingerprint** (hash of every file's instance range, the seed and
      the val split) and a **next-batch checksum** (the next ~4 instance indices and a hash of their
      tokens).

## 4. Quality and experiments

- [ ] **Learning-rate sweep** before the mix ablations (they must share one LR): 3–4 runs of ~500
      steps at constant LR (e.g. 6e-4 / 1.2e-3 / 2e-3 / 3e-3, `--set learning_rate=... warmup_steps=50
      lr_schedule=constant --max-steps 500 --tag lr...`); pick the lowest smooth loss at step 500, then
      back off a notch.
- [ ] **Mix ablation** at small scale (e.g. 40M on 1B tokens, 2–3 ratio settings, compared on holdout
      BPB) to back the 0.60 / 0.20 / 0.13 / 0.07 choice. One `configs/` file per ablation.
- [ ] Optional: **attention within documents only.** Instances join several documents with EOS, and
      attention crosses them (as in GPT-2/3, nanoGPT and OLMo's default); Llama 3 masks to the
      current document. Possibly a small gain with short documents (python_edu ~510 tokens).
- [ ] **Main benchmark task set**: arc_challenge, winogrande, openbookqa, MMLU in cloze format.
