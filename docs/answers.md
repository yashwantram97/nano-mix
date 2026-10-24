# Answers to the must-know list

This document gives the answers to the questions in `cleanup_and_understand.md` Part 2. The facts
are for the 124M run.

The text follows ASD-STE100 (Simplified Technical English): short sentences, active voice, one
topic in each sentence, and one word for one meaning. Technical names (file names, settings,
PyTorch terms) stay as they are.

## 2.1 Data and mix

### The four sources and their shares

**Question: what does each source teach the model?**

| Source | Share | What it teaches |
|---|---|---|
| fineweb_edu | 0.60 | General English and world knowledge. The pages are educational web pages, with explanations and facts on many topics. This source is the base of the mix. Thus, it has the largest share. |
| python_edu | 0.20 | Python code. The model learns the syntax and the strict structure (indentation, brackets). It learns that a name must match the same name far away in the file. It also learns logic in steps. Code possibly helps reasoning in general. |
| finemath | 0.13 | Math. The model learns notation (LaTeX, equations), numbers, and solutions in steps. |
| gutenberg | 0.07 | Long, coherent text. The model learns books, story structure, dialogue, and an older, literary style of English. The documents are long. Thus, most 1024-token instances are in one book, and the model learns to use the full context. |

`configs/mix.yaml` `sources` gives the shares. `max_repetition_ratio` stays at 1.0. Thus, the mix repeats
no source, and the build fails if a source is too small for its share.

### One document in the token files

**Question: why is uint16 sufficient?**

uint16 holds values from 0 to 65,535. The SmolLM2 vocabulary has 49,152 tokens. Thus, the largest
token id is 49,151. This value is less than 65,535, so each id fits in 2 bytes. `data/tokenize_data.py`
does a check for this (`arr.max() < 2**16`).

uint16 uses half the disk space and half the memory of int32. 5.05B tokens use ~10 GB, not ~20 GB.

OLMo-core counts the tokens of a file from the file size (bytes / 2). Thus, the files must be flat
arrays with no `.npy` header.

### An instance

**Question: what is the effect on short documents (python_edu, ~510 tokens)?**

One 1024-token instance holds approximately two python_edu documents, or parts of three. There are
three effects:

- **Attention goes across the EOS token.** The tokens of the second document can attend to the
  first document. The first document has no relation to them. The model must learn to ignore the
  text before an EOS token. This uses a small part of the model capacity and adds a small quantity
  of noise.
- **The instance edges cut documents.** Each instance is a contiguous window of tokens. Thus, a
  document often starts in one instance and ends in the next instance. The second part cannot see
  the start of its document. For example, it cannot see the imports and the function definitions of
  a code file.
- **Less long context from the same document.** Few tokens have 1024 tokens of their own document
  before them. Thus, python_edu teaches less about long context than gutenberg.

Document masking is the fix. With document masking, attention stays in the current document, as in
Llama 3. `model/config.py` lists this item as optional. The first 124M run does not use it.

### How OLMo-core picks the tokens

**Question: why does `data/trim_mix.py` need `manifest.json`?**

OLMo-core calculates the allocation from the file sizes:

1. For each source, it calculates the take ratio: needed tokens / available tokens.
2. Each file gives that fraction of its tokens, rounded down to whole instances.
3. Hamilton's method adds the remainder, to get whole batches.
4. A file with N instances gives its first N × 1024 tokens.

`data/trim_mix.py` keeps only the first N × 1024 tokens of each file. The trimmed files are smaller than
the full files. If OLMo-core calculates the allocation again from the trimmed sizes, the take ratio
is ~1.0. The rounding can then give a different count for each file. Then the instances change, and
the validation set and the resume position do not agree with the full mix.

`manifest.json` keeps two records:

- The allocation from the full files: the source, the file and the tokens of each file.
- The mix settings (`TRIM_KEYS`).

When `data/mix.py` finds the manifest, it uses `FixedSourceMixture`. `FixedSourceMixture` uses the
recorded counts as they are. It does not calculate them again. If `configs/mix.yaml` changed after the trim,
`data/mix.py` refuses the trimmed folder. `data/trim_mix.py` builds the mix from the two folders and makes
sure that the instances are the same.

### No repetition

**Question: why is dropout 0.0 correct here?**

Dropout is a regularizer. It prevents overfitting. Overfitting occurs when the model sees the same
examples many times and memorizes them.

This run reads each instance one time only. The run is one epoch, and the mix repeats no source.
`data/verify_mix.py` check 3 makes sure of this: no file occurs two times, and no file gives more tokens
than it has. Thus, the model cannot memorize the training data. The train loss and the val loss stay
near each other.

In these conditions, dropout only removes signal. The model learns less from each token for the
same compute. For this reason, Llama and OLMo use dropout 0 in pretraining. Dropout is useful only
when you train for more than one epoch on a small data set.

### The data sequence

**Question: what does a resume skip?**

`data/loader.py` makes one fixed order of all instances with `torch.randperm`. It uses the seed from
`configs/mix.yaml` (1234). The same seed gives the same order on each box.

- The first 8,192 instances in the order are the validation set.
- The training order starts after these instances.

A resume skips the instances that the run trained on before the stop. the trainer (`train/`) calculates this
number:

`skip_instances = start_step × grad_accum_steps × batch_size = start_step × 16 × 32 = start_step × 512`

Thus, the training order starts at `order[8,192 + start_step × 512]`. The run continues with
exactly the next batch. It does not read an instance two times, and it does not miss an instance.

This procedure is correct only if the mix, the seed and the batch shape do not change.
`check_resume` refuses a resume if one of them changed.

## 2.2 Tokenizer

### Which tokenizer

**Question: why use the tokenizer of an existing model, and not train a new one?**

- **Direct comparison.** SmolLM2-135M is a model of almost the same size. With the same tokenizer,
  the loss, the token counts and the benchmarks of the two models agree in units.
- **Known quality.** The SmolLM2 team trained this tokenizer on a mix of web text, code and math.
  This mix has the same types of data. Thus, the tokenizer compresses this data well.
- **Less work and less risk.** A new tokenizer needs a separate training step and its own tests. A
  mistake in the tokenizer makes all the data after it incorrect. Then you must tokenize all the
  data again.
- **A good vocabulary size.** 49,152 tokens fit in uint16, and 49,152 is a multiple of 128.

A tokenizer for this mix possibly compresses the data a little better. The gain is small for a
first run. BPB lets you compare tokenizers later (see 2.4).

### Special tokens

**Question: where does the code add EOS?**

`data/tokenize_data.py` adds EOS in the tokenize function (approximately line 120):

1. The tokenizer encodes each document with `add_special_tokens=False`. Thus, it adds no BOS and
   no EOS.
2. `split_special_tokens=True` is also set. If the text of a document contains the string
   `<|endoftext|>`, the tokenizer encodes it as normal text. Thus, only the code makes EOS tokens.
3. This line adds one EOS (id 0) after the tokens of each document:
   `np.fromiter((i for doc in ids for i in (*doc, eos)), ...)`

Thus, each document in a token file is its tokens, then one EOS. `meta.json` records `eos_id`, and
`data/mix.py` makes sure that it agrees with the tokenizer.

### Bytes for each token

**Question: how does `eval/bpb.py` get the byte length of a token?**

`token_byte_lengths` uses the byte-level BPE alphabet:

1. Byte-level BPE maps each of the 256 byte values to one printable character
   (`bytes_to_unicode`). Each token in the vocabulary is a string of these characters.
2. Thus, one character in the token string is one byte. The byte length of a token is the number of
   characters in its string: `lengths[i] = len(token)`.
3. Special tokens (for example EOS) get length 0. They are not text in the document.
4. The function makes sure that each character is in the byte alphabet. If a token has a different
   character, the function stops with an error.

The result is a table with one length for each token id. The BPB code adds the lengths of the
scored tokens to get the total bytes. `check_bytes` makes sure that the table gives the same total
as `len(text.encode("utf-8"))` on the eval documents.

### The 21 dropped byte values

There is no question for this item. These are the facts.

`unencodable_bytes` lists the byte values that have no token in the vocabulary. The tokenizer drops
these bytes. The 21 values are in three groups:

| Bytes | Count | What they are |
|---|---|---|
| 0x04 0x06 0x13 0x14 0x16 0x1d | 6 | ASCII control characters. They are almost never in text. |
| 0xc0 0xc1 0xf5–0xff | 13 | Byte values that never occur in valid UTF-8. |
| 0xf1 0xf2 | 2 | First bytes of 4-byte UTF-8 characters in rare Unicode planes. |

Thus, the tokenizer loses almost no real text. The BPB does not count these bytes, because the
model never sees them. `check_bytes` subtracts them when it compares the byte totals.

### Vocabulary and speed

**Question: why do matmul sizes that are a multiple of 128 run faster?**

- **Tensor cores use tiles.** The GPU calculates a matmul in fixed-size tiles (for example
  128 × 128 or 64 × 64 blocks). If a dimension is a multiple of the tile size, each tile is full.
- **No partial tiles.** If a dimension is not a multiple, the last tile is partly empty. The GPU
  must do the full tile, or it must use a slower code path for the edge. That wastes compute.
- **Aligned memory.** Sizes that are multiples of 128 give aligned rows in memory. The GPU then
  reads memory in full, wide transactions.

The output head is one of the largest matmuls: 768 × 49,152. Thus, its alignment is important.
49,152 = 384 × 128, so `pad_vocab_to_multiple_of=128` adds no rows. With a different tokenizer
(for example GPT-2, 50,257 tokens), the code adds rows up to 50,304. `mask_padded_logits` sets the
logits of the added rows to −inf before sampling, because those rows are not real tokens.

## 2.3 Model

### The shape

**Question: how do you get 122.7M parameters from these numbers?**

Width d = 768, vocabulary V = 49,152, SwiGLU hidden size h = 2,048, 12 layers.

| Part | Calculation | Parameters |
|---|---|---|
| Attention: `W_query`, `W_key`, `W_value`, `out_proj` | 4 × 768 × 768 | 2,359,296 |
| Attention: `out_proj` bias | 768 | 768 |
| SwiGLU: `w_gate`, `w_up`, `w_down` (no biases) | 3 × 768 × 2,048 | 4,718,592 |
| Two RMSNorms (scale only) | 2 × 768 | 1,536 |
| **One block** | | **7,080,192** |
| 12 blocks | 12 × 7,080,192 | 84,962,304 |
| Token embedding (tied with the output head) | 49,152 × 768 | 37,748,736 |
| Final RMSNorm | 768 | 768 |
| **Total** | | **122,711,808** |

The total is the same as the count from the model (`sum(p.numel() for p in model.parameters())`).
RoPE has no parameters. Thus, there is no position table in the count.

Approximately, each block has 12 × d² parameters: 4d² in attention and 8d² in the FFN. The
embedding is V × d. 12 × 12 × 768² + 49,152 × 768 ≈ 84.9M + 37.7M = 122.7M.

### RoPE

**Question: why can the model not read more than 1024 tokens well?**

- **No training on longer distances.** RoPE rotates q and k by an angle for each position. The
  attention score then depends on the distance between two tokens. During training, the largest
  distance is 1023. The model never learns what the angles for larger distances mean. The slow
  rotation dimensions are the problem: at a distance more than 1024, they get angles that the model
  never saw.
- **No training on long context.** The model also never learns to use more than 1024 tokens of
  context. For example, it does not learn to spread its attention over more tokens.
- **The cache stops at 1024.** `RotaryEmbedding` calculates the cos and sin tables for
  `context_length` positions only. With a longer input, the slice of the tables is too short, and
  the forward pass fails.

RoPE has no position table. Thus, the limit is not in the weights. Methods such as position
interpolation or a larger `rope_base`, with a short fine-tune on long documents, can extend the
context.

### RMSNorm and pre-norm

There is no question for this item. These are the facts.

`TransformerBlock.forward` does these steps:

1. `x = x + att(norm1(x))`
2. `x = x + ff(norm2(x))`

The norm is before each sublayer (pre-norm). The residual stream itself is never normalized. Thus,
the gradient goes back through the additions without a change, and deep models train stably. A
final RMSNorm (`final_norm`) normalizes the stream before the output head.

RMSNorm divides by the root mean square of the vector and multiplies by a learned scale. It does not
subtract the mean, and it has no shift parameter. It is simpler than LayerNorm and has the same
quality. PyTorch runs it as one fused kernel (`nn.RMSNorm`).

### SwiGLU

**Question: why is the hidden size 8/3 × width?**

To keep the same number of parameters as the GPT-2 FFN:

- The GPT-2 FFN has two matrices with hidden size 4d: 2 × d × 4d = 8d².
- SwiGLU has three matrices (`w_gate`, `w_up`, `w_down`) with hidden size h: 3 × d × h.
- 3dh = 8d² gives h = 8d / 3.

At d = 768, 8 × 768 / 3 = 2,048 exactly. 2,048 is also a multiple of 128. `swiglu_hidden_dim`
rounds up to a multiple of 128 for other widths. Thus, the SwiGLU FFN has the same parameters and
almost the same FLOPs as the GPT-2 FFN. The gate (`silu(w_gate x) * w_up x`) gives better quality
for the same cost.

### Tied embedding

**Question: how many parameters does this save at width 768?**

It saves one 49,152 × 768 matrix: **37,748,736 parameters (37.7M)**. Without the tie, the model has
160,460,544 parameters (~160.5M), 31% more.

The compute for each token stays almost the same. The output head matmul occurs with or without the
tie. The embedding lookup has no FLOPs. GPT-2 small and SmolLM2-135M also use a tied embedding.

### Initialization

**Question: why does a deep model need the smaller value?**

Each block adds two outputs to the residual stream: one from attention (`out_proj`) and one from the
FFN (`w_down`). 12 layers give 24 additions. The variances of the additions add up. With std 0.02
on all matrices, the last blocks have a large effect, and the stream starts far from the identity.
Early training is then slow and has spikes.

`init_weights` gives the two residual projections a smaller std:
0.02 / √(2 × n_layers) = 0.02 / √24 ≈ 0.0041. Thus, the 24 additions together have approximately
the variance of one addition with std 0.02. Each block starts with a small effect, and the model
starts near the identity. The comment in `model/gpt.py` gives the measured effect: the residual std
after the last block is ~4.8× smaller than with a flat 0.02.

## 2.4 Loss

### Next-token cross-entropy

There is no question for this item. These are the facts.

`MixInstances` gives two tensors for each instance:

- The input: the 1024 tokens of the instance.
- The targets: the same tokens, moved left by one. The last position gets `IGNORE_INDEX = -100`,
  because its next token is not in the instance.

`batch_loss` calls `cross_entropy`. `cross_entropy` skips targets of -100, and it calculates
the mean over the scored targets only. Thus, 1,023 of 1,024 positions are scored in each instance.

### The start value

**Question: the smoke run shows 9.63 at step 10. Is that correct?**

Yes. The value is in the expected range:

- **At step 0, the loss is near ln(49,152) ≈ 10.80.** The initialization is small (std 0.02). Thus,
  the logits are near 0, and the softmax is almost uniform.
- **The first thing that the model learns is the token frequencies.** Some tokens (spaces, "the",
  newlines, EOS) are much more frequent than others. A model that knows only the frequencies has a
  loss much less than 10.80. The model learns this in a few steps.
- **The log value is a mean.** The loss at step 10 is the mean of steps 1 to 10. The first steps
  had a loss near 10.8. Thus, the value at step 10 is higher than the loss of the model at step 10.
- **The LR is small.** Step 10 is early in the warmup. Thus, the model changes slowly.

These values show a problem:

- More than ~11 at the start: the initialization or the logits are incorrect.
- Near 0 after a few steps: the target is visible in the input (an error in the shift or in the
  causal mask).
- No decrease: the LR is 0, or the optimizer does not get the parameters.

### Loss in fp32

There is no question for this item. These are the facts.

Under bf16 autocast, the model gives the logits in bf16. `cross_entropy` changes them to fp32
before the softmax. Thus, the log-softmax and its gradient are in fp32. bf16 has only 8 bits of
mantissa. A softmax over 49,152 values in bf16 loses too much precision.

### Loss, perplexity and BPB

**Question: why can you compare BPB between tokenizers, but not loss?**

The loss is in nats **for each token**. Two tokenizers cut the same text into different tokens:

- A tokenizer with longer tokens makes fewer tokens. Each token holds more text, so each token is
  more difficult to predict. The loss for each token goes up.
- A tokenizer with shorter tokens makes more tokens, each with a lower loss.

The two models can have the same quality for the full text, but different losses. A larger
vocabulary also gives a higher uniform start value (ln V).

BPB divides the total information (in bits) by the bytes of the text. The text is the same for
all tokenizers, so the bytes are the same. Thus, BPB measures how well the model compresses the
text, with no dependency on the tokenizer.

Conversion: BPB = (loss / ln 2) / (mean bytes for each token). Perplexity = e^loss has the same
problem as the loss, because it is also for each token.

### Train loss and val loss

**Question: why is the val loss (5.03) less than the train loss (6.06) at step 200 of the smoke
run?**

The two values measure different things:

- **The train loss is a mean over time.** The train value at step 200 is the mean of all steps
  since the last val point. In the smoke run, this is steps 1 to 200. The early steps had losses
  near 10. The mean includes them, so it is high.
- **The val loss is for the model now.** The code calculates the val loss with the weights at step
  200, on 20 fixed batches. It contains no old values.

In the first steps, the loss decreases quickly. Thus, the mean of the past is much higher than the
value now.

There is no other reason for the difference. Dropout is 0, so the model is the same in train mode
and in eval mode. The val set comes from the same mix, and each instance is read one time only.
Thus, later in the run, when the loss decreases slowly, the two values are almost the same. A val
loss much higher than the train loss later in the run shows a problem (for example, repeated data
or a leak).
