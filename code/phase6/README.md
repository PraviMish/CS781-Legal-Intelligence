# Phase 6 | Legal abstractive summarization

**E05, V3 Efficient GQA + LoRA, won the seven recorded Codabench submissions
with BERTScore F1 0.8459** (submission 1026556, `sub5.zip`). It used the V3
`models/E05/best.pt` checkpoint from update **7,750**, warm-started from V2
E05 at update 4,848, with the raw hierarchical decoder. E07 led local
validation on both ROUGE metrics. The official difference between E05 and
E07 was 0.0004 on 100 hidden-reference documents.

| Run | Architecture and adaptation | Original R-1 | V2 R-1 | V3 R-1 | V3 R-L | Official BERTScore F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| E01 | Encoder–decoder + LoRA | 0.2814 | 0.2218 | 0.2504 | 0.1602 | 0.8217 |
| E02 | Efficient GQA + LoRMA | 0.3081 | 0.4042 | 0.4129 | 0.2278 | 0.8442 |
| E03 | Prefix-LM + Houlsby | 0.2880 | 0.2995 | 0.2847 | 0.1764 | 0.8239 |
| E04 | Eight-block causal decoder student + distillation | 0.2432 | 0.3160 | 0.3975 | 0.2198 | 0.8438 |
| E05 | Efficient GQA + LoRA | 0.2943 | 0.3896 | 0.4092 | 0.2236 | **0.8459** |
| E06 | Encoder–decoder full fine-tuning | 0.2875 | 0.2603 | 0.2686 | 0.1661 | 0.8232 |
| E07 | Eight-block gold-only causal student | | 0.3976 | **0.4234** | **0.2310** | 0.8455 |

Local ROUGE uses stemmed per-document F1 averaged across 32, 64 and 96
held-out judgments for original, V2 and V3 respectively. Different samples
and supervision budgets limit cross-version comparisons. Course rules name
ROUGE-1 and ROUGE-L as secondary official metrics. The supplied official
screenshot shows BERTScore F1 and submission IDs. See the
[aggregate results](experiments/results.csv) and [evidence audit](../../docs/PHASE6_EVIDENCE.md).

## Data and model

Use [Phase_6_Summ at the verified dataset revision](https://huggingface.co/datasets/Exploration-Lab/CS-781-Capstone/tree/93b0059be756c6c461f3a225de25aabda7d5f84a/Phase_6_Summ).
Training provides 7,030 ID/document/reference triples. The 100 test judgments
have hidden references. The notebook’s normalized exact-text grouped split
allocates 6,466 train and 564 validation judgments. The median labeled source
has 17,905 characters and the median reference has 638 whitespace words.
The median training document is 3,657 BPE tokens. The causal context of 1,280
tokens is exceeded by 94.4% of training and 95% of test documents. The median
training compression ratio is 4.59 source words per reference word.
[Split-level length and compression statistics](experiments/dataset_statistics.json)

All versions use the original Phase 1 32k byte-level BPE with PAD/BOS/EOS/UNK
IDs 0/1/2/3. Encoder–decoder, prefix and standard causal initializations use
their Phase 1 checkpoints. GQA uses the 700M-token checkpoint. The encoder
admits 1,024 tokens and decoder 640, while the causal backbones admit 1,280.
CE supervises only summary tokens including EOS. Encoder–decoder predicts
from BOS and shifted targets with cross-attention. Causal models predict
targets from the final source position onward. Prefix-LM uses bidirectional
source attention and causal target attention. Examples run individually, with
no padding mask required.

- **Original.** One prompt combines a 900-token document budget assembled
  from 150-token sections. It prioritizes the first and last two sections and
  fills remaining space by legal term frequency and later position. The target
  is truncated at 239 tokens before EOS. Low-rank runs use rank 8 on selected
  attention projections. E03 uses one normalized post-block adapter. Five
  supervised runs and one response-distilled student produced six ZIPs.
- **V2.** Independently trains from the Phase 1 weights. Uses 840-token
  contiguous passages, keeping first, last and salient interior passages up
  to four. Adds a position-labeled prompt to each. Reference sentences are
  assigned to the best selected passage by normalized word overlap at a
  0.05 threshold. One pack of up to 319 tokens plus EOS per passage favors
  complete sentences. The 21,710 aligned train passages yield 19,392 scheduled
  examples over three rotating document passes. Rank 16 adapters cover block
  attention and MLP projections and train normalization parameters. E03 uses
  two residual bottleneck adapters per block. Six models produced ZIPs, and
  E07 was evaluated locally.
- **V3.** Warm-starts each core run from V2. Both students start from V2 E07.
  V3 preserves the order of aligned reference sentences, creates multiple
  targets per selected passage, and splits overlong sentences at available
  clause boundaries. It skips unalignable sentences or fragments exceeding
  the cap. All 31,668 aligned targets across 6,464 usable train documents
  appear once in each run's 7,917 updates. The validation cache has 2,778
  targets. Seven runs produced ZIPs.

LoRA adds a low-rank input update to a frozen linear projection. The notebook’s
multiplicative LoRMA+ uses the projection output as its low-rank input. Houlsby
uses width-64 ReLU bottlenecks. E06 trains all encoder–decoder weights. Students
retain Phase 1 decoder layers 0, 2, 3, 5, 6, 8, 9 and 11 and train all
82.26M parameters. V3 E05 has 115.72M total and 3.06M trainable parameters.

E02 supplies the teacher in all three versions. Original E04 mixes gold and
teacher-sequence CE 50:50 on the teacher-labeled examples (1,000 documents).
V2 uses 60:40 on 2,686 nonempty teacher passages from 800 documents. Both
versions replace roughly half the scheduled draws with draws from the
teacher-labeled pool for E04. V3 evaluates the frozen teacher online on the
same gold-conditioned target positions as the student, with 80% gold CE and
20% temperature-scaled teacher-to-student KL at temperature 2. E04 trails E07
on validation ROUGE and official BERTScore despite better V3 validation NLL.
The students have matched initial weights and coverage, with different
run-specific shuffle orders.

## Decoding and evaluation

Greedy decoding uses KV caches, blocks PAD/BOS/UNK and repeat trigrams, and
allows EOS after 40 original or 32 revised output tokens. Generation caps are
192 original and 260 revised per passage, with repetition penalties 1.12 and
1.10 respectively. V2/V3 trim a trailing incomplete sentence if terminal
punctuation falls in the last 35% of the text. They concatenate in document
order and skip sentences with content-word Jaccard overlap above 0.73.
This procedure has no learned document-level synthesis stage. V3 also tried
lexical and numeric filtering with a fallback when fewer than 30 words
survived. Raw won validation ROUGE-1 for all seven experiments.

On the same 96 V3 documents, E02 hierarchical raw decoding scored 0.4129
ROUGE-1 against 0.1860 for head-only. Mean output length increased from 98 to
302 words, so the ablation combines access to more evidence and a larger
aggregate generation budget. V2 E02 validation output for document 1770
introduced a Section 307 charge alongside the Section 302 conviction
described in the reference and changed the appellate account. For document
757 it inserted a £2,000 figure absent from the reference. These examples
are case studies. The archive gives no systematic legal-factuality rate.

## Reproduce

Install [repository requirements](../requirements.txt) and set
`PYTHONPATH=code`. Use private paths for data, caches and checkpoints.
For historical split/order replay, supply the archived `validation_ids.json`
and the random seed from the executed notebook. The research CLI takes that
seed as a parameter. Use `--device cpu --precision fp32` for CPU checks.
Archived model-only checkpoints include the adapted backbone and load
directly. Retraining from archived initialization can differ through device
arithmetic and the original notebook's cumulative random state.

```bash
P6=/path/to/Phase_6_Summ
TOK=/path/to/legal_bpe_32k.json
VAL=/path/to/validation_ids.json
OUT=/path/to/private/reconstruction
# Set SEED to the executed notebook's seed.
python -m phase6 prepare --version v3 --seed "$SEED" --tokenizer "$TOK" \
  --train "$P6/Train/train.json" --validation-ids "$VAL" --output "$OUT/v3.pt"
python -m phase6 train --version v3 --run E05 --seed "$SEED" --tokenizer "$TOK" \
  --cache "$OUT/v3.pt" --warmstart /path/to/v2/models/E05/best.pt --output "$OUT/v3-E05"
python -m phase6 evaluate --version v3 --run E05 --seed "$SEED" --tokenizer "$TOK" \
  --train "$P6/Train/train.json" --validation-ids "$VAL" --checkpoint "$OUT/v3-E05/best.pt"
python -m phase6 submit --version v3 --run E05 --tokenizer "$TOK" \
  --test "$P6/Test/test.json" --checkpoint /path/to/v3/models/E05/best.pt --output "$OUT/E05.zip"
python -m phase6 validate --test "$P6/Test/test.json" --predictions "$OUT/E05.zip"
```

Use the same commands with `--version original` or `--version v2` and a
matching cache. Replace `--warmstart` with `--foundation` and provide the
Phase 1 `best.pt` for that architecture (GQA uses `best_up_to_700M.pt`).
For original/V2 E04, generate the pseudo-target file from that version's
trained E02 using `teacher-targets --run E02 --checkpoint ... --cache ...
--tokenizer ... --seed ... --output ...`. Supply it to E04 using
`--pseudo`. For V3 E04, pass V2 E07 as `--warmstart` and V3 E02 as
`--teacher`. V3 E07 takes the same V2 warm start and gold CE. The other V3
runs warm-start their same-name V2 checkpoint. Use separate output folders
per run. A repeated command resumes from the port's atomic optimizer state.
Use `evaluate --strategy head` for head-only and `--policy grounded` for
V3 lexical filtering. The submission archive contains a single root
`prediction.json` with one nonempty string summary per official ID.

Build the separate source archive with
`python scripts/package_phase6_winner.py --output /path/to/Phase6_E05_source.zip`.
It provides `python -m phase6.winner` for the official checkpoint.
Regenerate figures with `python code/phase6/analysis/figures.py`, then compile
`report/main.tex` from the report directory.
