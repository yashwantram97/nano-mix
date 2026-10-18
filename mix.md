# Data mix (`mix.py`)

Step 3 of the pipeline: combine the token files in `data/tokenized/<source>/` into one training
dataset with OLMo-core's source mixture API. It builds and checks the mix only; nothing is trained.

```bash
uv run python mix.py [--config mix.yaml]
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

No other existing package changes. `pyyaml` is a direct dependency (for `mix.yaml`). Run `uv sync`
before the first run. `download.py` and `tokenize_data.py` haven't been run on `huggingface-hub` 1.x yet;
the calls they use exist there.

**The API differs from the docs.** The OLMo-core docs describe v2.6.0. In 3.0.0, sources are wrapped
in a `SourceMixtureList`, and `SourceMixtureDatasetConfig.build()` takes `npdtype` and
`sequence_length`. `mix.py` uses the 3.0.0 names, read from the installed source.

## How OLMo-core 3.0.0 builds the mix

Read from the installed source (`olmo_core/data/source_mixture.py`, `numpy_dataset.py`):

- **Contiguous reads.** Each instance is a contiguous 256-token window from the start of a file.
  A file allocated N instances contributes its first N × 256 tokens. Instances cross document
  boundaries, with EOS between documents. No document is cut short and nothing is repeated.
- **`prepare()` writes index files that are never read.** They store one sample per document,
  truncated to 256 tokens and drawn with replacement, in `work_dir`. Nothing in the dataset reads
  them back; the code that did is commented out.
- **The seed doesn't change the content.** Which tokens end up in the mix depends only on the file
  sizes. Order is shuffled later, by the data loader at training time.
- **Token counts come from file sizes** (bytes / 2 for uint16), so the files must have no `.npy`
  header. `verify_tokenized.py` checks that.

## `mix.yaml`

Every setting lives here; `mix.py` has no numbers hard-coded.

| key | value | meaning |
|---|---|---|
| `requested_tokens` | 500,000,000 | target mix size; the result is rounded up to whole batches |
| `sequence_length` | 256 | tokens per instance (was 2048, then 1024; lowered for local Mac runs) |
| `global_batch_size` | 524,288 | 256 × 2048 tokens; must be a multiple of `sequence_length` |
| `seed` | 1234 | passed to OLMo-core (see above) |
| `tokenizer` | `HuggingFaceTB/SmolLM2-135M` | fixed in `CLAUDE.md` |
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

Reads `mix.yaml` and fails early if the ratios don't sum to 1.0 or `global_batch_size` isn't a
multiple of `sequence_length`. OLMo-core checks both too, but later and with less clear errors.

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

Returns the dataset and the tokenizer config. `verify_mix.py` will reuse this to build the mix twice
for the determinism check.

### `tokens_per_source(dataset)`

`dataset.offsets` holds each file's (start, end) instance range, so (end − start) × 256 is the
tokens taken from that file. The file's parent folder name identifies its source. Used for the ratio
and no-repetition checks.

### `main()`

1. Loads `mix.yaml` (`--config` to use another file).
2. Step 1: the coverage check.
3. Prints the tokenizer's `vocab_size` and `eos_token_id`, and asserts that EOS matches `eos_id` in
   every `meta.json`.
4. Turns on INFO logging, because OLMo-core prints its summary tables through the logger.
5. Builds and prepares the dataset.
6. Prints total instances and tokens, then target share, actual share, tokens used and tokens
   available per source.

## Expected result

Worked out by hand from OLMo-core's allocation code; not from a run.

**Total.** Steps = ⌈500,000,000 / 524,288⌉ = 954, and each step is 2,048 instances:
954 × 2,048 = **1,953,792 instances** = **500,170,752 tokens**, 0.03% over the request.

**Per source.** Each source first gets ⌊ratio × 500M / 256⌋ instances, which totals 1,953,124. The
668 missing instances are spread evenly (167 per source).

| source | instances | tokens | share | target |
|---|---|---|---|---|
| fineweb_edu | 1,172,042 | 300,042,752 | 0.5999 | 0.60 |
| python_edu | 390,792 | 100,042,752 | 0.2000 | 0.20 |
| finemath | 254,073 | 65,042,688 | 0.1300 | 0.13 |
| gutenberg | 136,885 | 35,042,560 | 0.0701 | 0.07 |
| **total** | **1,953,792** | **500,170,752** | | |

All four are within ±0.5% of target.

## Running it

After the python_edu download (250,000 files) finishes:

```bash
uv sync                                     # once: installs OLMo-core and torch (done 2026-10-09)
rm data/tokenized/python_edu/pilot-5000.npy data/tokenized/python_edu/meta.json
uv run python tokenize_data.py --source python_edu
uv run python verify_tokenized.py
uv run python mix.py
```

Delete the old `pilot-5000.npy` first. It holds the same first 5,000 documents as the new file, and
`check_coverage` will stop if it's still there.

## Checking the mix: `verify_mix.py`

```bash
uv run python verify_mix.py [--config mix.yaml]
```

Builds the mix with the same `mix.yaml` and `build_dataset()` as `mix.py`, then checks:

1. **Total size**: instances × 256 equals `requested_tokens` rounded up to whole batches
   (500,170,752), and is at least `requested_tokens`.
2. **Ratios**: each source's actual share is within ±0.5 percentage points of its target.
3. **No repetition**: no file appears twice, and no file gives more tokens than it has.
4. **Content**: for 2 random instances per source, checks the instance equals the expected slice of
   its source file, prints the first 300 decoded characters and its EOS count, and checks that
   documents end (EOS) inside the part of each source the mix uses.
5. **Determinism**: builds a second time and compares files, offsets and the first 10 instances.
6. **Shape**: reads every instance through the dataset; each must be exactly 256 tokens with all
   IDs below the vocab size.

It ends with a table (source, target share, actual share, tokens used, tokens available, PASS/FAIL)
and exits with an error if any check fails.
