# Data pipeline

Download → tokenize → mix. Each step reads only the previous step's output; the code is in
`src/nanomix/data/`, and the source list, eval-only shards and tokenizer are in
`src/nanomix/sources.py`.

## 1. Download (`download.py`, `verify_download.py`)

```bash
uv run nanomix-download --source all          # or one of fineweb_edu, finemath, gutenberg, python_edu
uv run nanomix-verify-download [--offline] [--deep]
```

Every file of each source's subset, from a pinned commit, into `data/raw/<source>/`, except the
eval-only shards. Files already on disk are skipped, so an interrupted run can be re-run.
`data/raw/manifest.json` records repo, commit, files, bytes and status per source. python_edu's
metadata comes from Hugging Face; its code is fetched from Software Heritage S3 (anonymous first,
then your AWS credentials) in files of 250,000 rows, skipping the 8,000 holdout rows. `data/raw/` is
never modified afterwards.

`verify_download.py` checks that every file is complete (against a fresh HF listing), no eval-only
shard or leftover is on disk, every file reads (`--deep`: through to the end), and the python_edu
content files hold exactly their planned blob_ids.

## 2. Tokenize (`tokenize_data.py`, `verify_tokenized.py`)

```bash
uv run nanomix-tokenize --source all [--workers N]
uv run nanomix-verify-tokenized
```

One flat uint16 file per raw file in `data/tokenized/<source>/` (same base name, no `.npy` header):
each document's tokens (`add_special_tokens=False`, `split_special_tokens=True`) followed by EOS
(id 0). No BOS, no padding. One process per file. `meta.json` lists every file with its token and
document counts. `verify_tokenized.py` checks bytes = 2 × tokens, max id < vocab, EOS count = documents,
and prints random slices to read.

## 3. Mix (`mix.py`, `verify_mix.py`, `trim_mix.py`)

Combine the token files into one training dataset with OLMo-core's source mixture API. It builds and
checks the mix only; nothing is trained.

```bash
uv run nanomix-mix [--config configs/mix.yaml]
```

## Setup notes

**Installing OLMo-core needed a dependency change.** Every version of `ai2-olmo-core` depends on
`cached-path`, which requires `huggingface-hub<2.0`. The project's pin was relaxed to
`huggingface-hub>=1.0,<2`, which resolves to:

| package | version |
|---|---|
| ai2-olmo-core | 3.0.0 |
| torch | 2.13.0 (CPU, works on macOS) |
| huggingface-hub | 1.33.0 (was 2.2.0) |

No other existing package changes. `pyyaml` is a direct dependency (for `configs/mix.yaml`).

**The API differs from the docs.** The OLMo-core docs describe v2.6.0. In 3.0.0, sources are wrapped
in a `SourceMixtureList`, and `SourceMixtureDatasetConfig.build()` takes `npdtype` and
`sequence_length`. `mix.py` uses the 3.0.0 names, read from the installed source.

## How OLMo-core 3.0.0 builds the mix

Read from the installed source (`olmo_core/data/source_mixture.py`, `numpy_dataset.py`):

- **The same share of every file.** A source's take ratio is needed ÷ available tokens, and every
  file in the source gives that fraction of its tokens (rounded down to whole instances, then topped
  up to whole batches with Hamilton's method). It does not take whole files in order. So the files in
  the folder decide which documents are used: all 49 fineweb_edu shards give ~20% of each, while 10
  shards would give ~100% of fewer files.
- **Contiguous reads.** Each instance is a contiguous `sequence_length`-token window (1024) from the
  start of a file. A file allocated N instances contributes its first N × 1024 tokens. Instances cross
  document boundaries, with EOS between documents. Nothing is repeated; the last document of each
  file's share is cut where the share ends, and the rest of the file is unused.
- **`prepare()` writes index files that are never read.** They store one sample per document,
  truncated to `sequence_length` tokens and drawn with replacement, in `work_dir`. Nothing in the
  dataset reads them back; the code that did is commented out.
- **The seed doesn't change the content.** Which tokens end up in the mix depends only on the file
  sizes. Order is shuffled later, by the data loader at training time.
- **Token counts come from file sizes** (bytes / 2 for uint16), so the files must have no `.npy`
  header. `verify_tokenized.py` checks that.

## `configs/mix.yaml`

Every setting lives here; `mix.py` has no numbers hard-coded.

| key | value | meaning |
|---|---|---|
| `requested_tokens` | 5,050,000,000 | 5B to train on + ~1% for `val_instances` and batch rounding; the result is rounded up to whole batches (was 500M for the 40M pilot) |
| `sequence_length` | 1024 | tokens per instance; must equal the model's `context_length` (was 256 for the 40M Mac pilot) |
| `global_batch_size` | 524,288 | 512 × 1024 tokens; must be a multiple of `sequence_length` |
| `seed` | 1234 | passed to OLMo-core (see above) |
| `tokenizer` | `HuggingFaceTB/SmolLM2-135M` | fixed; `load_config` checks it equals `sources.TOKENIZER` |
| `dtype` | `uint16` | set explicitly instead of guessed from the vocab size |
| `tokenized_dir` | `data/tokenized` | token files are `<tokenized_dir>/<source>/*.npy` |
| `work_dir` | `data/mix` | OLMo-core's cache of index files |
| `sources` | fineweb_edu 0.60, python_edu 0.20, finemath 0.13, gutenberg 0.07 | must sum to 1.0 |

`max_repetition_ratio` isn't set, so it stays at its default of 1.0: no source is repeated, and the
build fails if a source is too small for its share.

## `mix.py`, top to bottom

### Imports

The OLMo-core 3.0.0 names:

- `olmo_core.data`: `TokenizerConfig`, `NumpyFSLDatasetConfig`, `NumpyDatasetDType`
- `olmo_core.data.source_mixture`: `SourceMixtureConfig`, `SourceMixtureList`, `SourceMixtureDatasetConfig`

### `load_config(path)`

Reads `configs/mix.yaml` and fails early if the ratios don't sum to 1.0, `global_batch_size` isn't a
multiple of `sequence_length`, or the tokenizer isn't the pipeline's. OLMo-core checks the first two
too, but later and with less clear errors.

### `check_coverage(cfg)`: Step 1

For each source:

1. Reads `data/tokenized/<source>/meta.json`.
2. Checks that the `.npy` files in the folder exactly match `meta.json`'s file list. OLMo-core takes
   every `.npy` in the folder, so a leftover file (such as an old `pilot-5000.npy`) would put the same
   documents in the mix twice.
3. Prints available tokens, needed tokens (ratio × `requested_tokens`) and coverage (available / needed).

If any coverage is below 1.0, it stops and names the sources that are short.

### `build_dataset(cfg)`: Step 2

1. **Tokenizer.** `TokenizerConfig.from_hf` downloads the model's `config.json` and reads
   `vocab_size`, `eos_token_id` and `bos_token_id`. If there's no pad token, it uses the EOS ID.
2. **Sources.** One `SourceMixtureConfig` per source: `source_name`, `target_ratio`, and an absolute
   glob `.../data/tokenized/<source>/*.npy`.
3. **Mix.** `SourceMixtureDatasetConfig` gets the sources (in a `SourceMixtureList`),
   `requested_tokens`, `global_batch_size` and `seed`.
4. **Dataset.** `NumpyFSLDatasetConfig.from_src_mix(...)` adds `tokenizer`, `sequence_length`,
   `dtype=uint16` and `work_dir`.
5. **`.build()`.** Counts each file's tokens from its size, fails if a source is too small,
   allocates instances to each file (see below), and logs OLMo-core's mix tables.
6. **`.prepare()`.** Writes the index files into `data/mix/`. Required by the API even though reads
   don't use them.

Returns the dataset and the tokenizer config. `verify_mix.py` reuses this to build the mix twice
for the determinism check, and the trainer (`data/loader.py`) to read it.

### `tokens_per_source(dataset)`

`dataset.offsets` holds each file's (start, end) instance range, so (end − start) × `sequence_length` is the
tokens taken from that file. The file's parent folder name identifies its source. Used for the ratio
and no-repetition checks.

### `main()`

1. Loads `configs/mix.yaml` (`--config` to use another file).
2. Step 1: the coverage check.
3. Prints the tokenizer's `vocab_size` and `eos_token_id`, and asserts that EOS matches `eos_id` in
   every `meta.json`.
4. Turns on INFO logging, because OLMo-core prints its summary tables through the logger.
5. Builds and prepares the dataset.
6. Prints total instances and tokens, then target share, actual share, tokens used and tokens
   available per source.

## Expected result

Worked out by hand from OLMo-core's allocation code; not from a run.

**Total.** Steps = ⌈5,050,000,000 / 524,288⌉ = 9,633, and each step is 512 instances:
9,633 × 512 = **4,932,096 instances** = **5,050,466,304 tokens**, 0.009% over the request.
The trainer holds out `val_instances` (8,192 × 1024 = 8.4M tokens), leaving ~5.04B to train on.

**Per source.** The shares below are the targets; the exact instance counts depend on every file's
size (each file is rounded down to whole instances, then Hamilton's method adds the remainder),
so expect them to differ by a few hundred instances. The last column is approximate, from the
tokenized sizes estimated before tokenizing.

| source | ~instances | ~tokens | target | ~share of each file used |
|---|---|---|---|---|
| fineweb_edu | 2,959,258 | 3.030B | 0.60 | ~20% (of ~15.4B) |
| python_edu | 986,419 | 1.010B | 0.20 | ~26% (of ~3.9B) |
| finemath | 641,172 | 0.657B | 0.13 | ~6% (of ~10.9B) |
| gutenberg | 345,247 | 0.354B | 0.07 | ~7% (of ~5.1B) |
| **total** | **4,932,096** | **5.050B** | | |

## Running it

The mix needs all of `data/tokenized/` (~71 GB) on disk: OLMo-core sizes every file, even though
training reads only each file's first part. Run it on a small CPU instance in us-east-1 (S3 → EC2 in
the same region is free); the GPU box rebuilds `data/mix/` from `configs/mix.yaml` itself, because
`mix.py` uses absolute paths.

```bash
aws s3 sync s3://nano-mix/tokenized/ data/tokenized/
uv run nanomix-mix 2>&1 | tee logs/mix.log
uv run nanomix-verify-mix 2>&1 | tee logs/verify_mix.log
```

`check_coverage` stops if a folder's `.npy` files don't match its `meta.json`, or if an eval-only
shard (the list in `sources.py`, plus `data/eval/<name>.json` where present) is in the mix.

## Checking the mix: `verify_mix.py`

```bash
uv run nanomix-verify-mix [--config configs/mix.yaml]
```

Builds the mix with the same `configs/mix.yaml` and `build_dataset()` as `mix.py`, then checks:

1. **Total size**: instances × `sequence_length` equals `requested_tokens` rounded up to whole
   batches (5,050,466,304), and is at least `requested_tokens`.
2. **Ratios**: each source's actual share is within ±0.5 percentage points of its target.
3. **No repetition**: no file appears twice, and no file gives more tokens than it has.
4. **Content**: for 2 random instances per source, checks the instance equals the expected slice of
   its source file, prints the first 300 decoded characters and its EOS count, and checks that
   documents end (EOS) inside the part of each source the mix uses.
5. **Determinism**: builds a second time and compares files, offsets and the first 10 instances.
6. **Shape**: reads every instance through the dataset; each must be exactly 1024 tokens with all
   IDs below the vocab size. With ~4.9M instances this is the slow check (several minutes).

It ends with a table (source, target share, actual share, tokens used, tokens available, PASS/FAIL)
and exits with an error if any check fails.

## A trimmed copy: `trim_mix.py`

The mix reads only the start of each token file. `nanomix-trim-mix` copies just those tokens
(~10 GB instead of ~71 GB) to `data/tokenized_trimmed/`, plus a `manifest.json` with the per-file
allocation, then builds the mix again from the copy and checks the instances are identical. Run it
where the full files are. On another box, put the copy at `tokenized_dir` (`data/tokenized/`):
`mix.py` sees `manifest.json` and uses the recorded allocation, because OLMo-core's rounding on the
trimmed file sizes could differ. It refuses the copy if the mix settings changed since the trim.
