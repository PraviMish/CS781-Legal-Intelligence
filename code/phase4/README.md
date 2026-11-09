# Phase 4 | Court judgment prediction

**E01, Efficient GQA full fine-tuning, won the ten recorded submissions with
official Codabench macro-F1 0.7393.** The archived model is the epoch-2
classifier, selected at optimizer step 1,818. Its submission threshold is
0.51. The original Phase 1 initialization is the GQA snapshot at 700,047,360
consumed tokens.

| Run | Method | Internal | Clean dev | Official |
| --- | --- | ---: | ---: | ---: |
| E01 | Efficient GQA full fine-tuning | 0.7795 | 0.7328 | **0.7393** |
| E02 | Frozen E01 features and hierarchical residual head | **0.7809** | 0.7233 | 0.7263 |
| E03 | Multiplicative GQA LoRMA+ | 0.7548 | 0.6576 | 0.6749 |
| E04 | Causal decoder full fine-tuning | 0.7423 | 0.6911 | 0.6976 |
| E05 | Padding-masked Prefix-LM full fine-tuning | 0.6769 | 0.6445 | 0.6476 |
| E06 | Padding-masked encoder–decoder full fine-tuning | 0.6480 | 0.6133 | 0.6013 |
| E07 | Adaptive-rank GQA | 0.7150 | 0.6394 | 0.6580 |
| E08 | Six-layer distilled Transformer | 0.6643 | 0.6015 | 0.6126 |
| E09 | 90% E01 + 10% E05 | 0.7797 | **0.7346** | 0.7353 |
| E10 | 90% E01 + 10% E08 | 0.7797 | 0.7329 | 0.7369 |

All columns report macro-F1, with thresholds selected on internal validation.
Official values were provided by the user. [Results and class errors](experiments/results.csv)

## Evidence and reproduction

The executed `Capstone_4.ipynb` supplies architecture and method definitions.
The Drive archive contains the selected checkpoint metadata, configurations,
epoch histories, probabilities, and ZIPs. E03–E05 have missing execution
counters in the retained notebook copy. Their saved checkpoints or checkpoint
metadata, training histories, local probabilities, and submission ZIPs
establish those runs. E02 has an earlier notebook cell with a source timestamp
mismatch. The later saved head, features, probability files, and ZIP supply the
replay evidence. The checked-in records contain aggregate results only.
[Run configurations](configs) · [Training histories](experiments/training_history.csv)

Use the official `Phase_4_CJPE` files at dataset revision
`93b0059be756c6c461f3a225de25aabda7d5f84a` and the archived Phase 1 tokenizer
(32,000 tokens, PAD/BOS/EOS/UNK IDs 0/1/2/3). The case split keeps the longest
representative per ID and groups normalized duplicate text for the held-out
fold. Clean dev excludes official test IDs and matching normalized text. The
official archive had zero such overlaps and 994 unique balanced development
cases. Use `--archived-splits` for the historical parquet split order. A fresh
split uses seed 42 and stratified group folds. [Dataset statistics](experiments/dataset_statistics.json)

### Prepare

Install the [repository requirements](../requirements.txt) and set
`PYTHONPATH=code`. Commands use a POSIX shell. Paths and outputs are local and
should stay outside the checkout. Each preparation output directory holds four
splits and their token arrays.

```bash
P4=/path/to/Phase_4_CJPE
P1=/path/to/phase1/models
TOK=/path/to/legal_bpe_32k.json
OUT=/path/to/private/phase4

python -m phase4 prepare --train "$P4/Train/train.json" --dev "$P4/Train/dev.json" \
  --test "$P4/Test/test.json" --tokenizer "$TOK" --cache "$OUT/tail"

python -m phase4 prepare --archived-splits /path/to/splits_v1 --tokenizer "$TOK" \
  --strategy hierarchical --cache "$OUT/hierarchical"
```

Use the same split source for both caches. Tail windows reserve two boundary
tokens and retain up to 1,022 trailing content tokens. Hierarchical windows
use at most five evenly spaced positions, ending at the tail. The notebook
also prepared a head-and-tail cache, with no submitted result. The research
CLI supports preparing that representation separately.

### Train and evaluate

| Run | Initialization and additional arguments | Saved best file |
| --- | --- | --- |
| E01 | `--foundation "$P1/efficient_gqa/milestones/best_up_to_700M.pt"` | `best_model.pt` |
| E02 | Hierarchical cache and `--features "$OUT/features"` from frozen E01 | `best_head.pt` |
| E03 | Same Phase 1 foundation as E01 | `best_adapters.pt` |
| E04 | `--foundation "$P1/decoder_only/best.pt"` (100M) | `best_model.pt` |
| E05 | `--foundation "$P1/prefix_lm/best.pt"` (100M) | `best_model.pt` |
| E06 | `--foundation "$P1/encoder_decoder/best.pt"` (100M) and `--tokenizer "$TOK"` | `best_model.pt` |
| E07 | Same Phase 1 foundation as E01 | `best_adapters.pt` |
| E08 | `--teacher-logits "$OUT/teacher"` from frozen E01 | `best_student.pt` |

```bash
python -m phase4 train --run E01 --cache "$OUT/tail" --output "$OUT/e01" \
  --foundation "$P1/efficient_gqa/milestones/best_up_to_700M.pt" --device cuda --precision bf16

E01="$OUT/e01/best_model.pt" # Or the original saved E01 epoch-2 checkpoint.
python -m phase4 features --cache "$OUT/hierarchical" --checkpoint "$E01" \
  --output "$OUT/features" --device cuda --precision bf16
python -m phase4 teacher-logits --cache "$OUT/tail" --checkpoint "$E01" \
  --output "$OUT/teacher" --device cuda --precision bf16

python -m phase4 train --run E02 --cache "$OUT/hierarchical" \
  --features "$OUT/features" --output "$OUT/e02" --device cuda
python -m phase4 train --run E08 --cache "$OUT/tail" --teacher-logits "$OUT/teacher" \
  --output "$OUT/e08" --device cuda --precision bf16

python -m phase4 evaluate --run E01 --cache "$OUT/tail" --checkpoint "$E01" --device cuda
```

Other training runs use the E01 command with their run ID and the arguments
in the table. E03 and E07 also require their foundation during evaluation or
submission. E06 requires the tokenizer BOS ID for its single decoder query.
E02 uses FP32 for the head and FP16 for the frozen feature cache. The notebook
used BF16 for backbone inference and the other adaptation runs. FP32 is
available for CPU checks or CUDA devices without BF16. A smaller
`--batch-size` changes the training batch configuration. Use a fresh output
directory for each distinct run. An identical command resumes this port's
atomic `resume.pt` at the last saved optimizer boundary, with matching inputs
and settings. Inference accepts the original best checkpoint schemas.
Notebook optimizer states are a separate format.

Feature and teacher caches are positional. Keep them with the exact ordered
splits that produced them. For an archived replay, use the original E01
teacher logits from `cache/e08_teacher_logits_v1` and E02 features from
`cache/hier5_features_e01_v1`. Original notebook RNG history and CUDA
rounding can affect a fresh training replay.

### Submit or replay

```bash
python -m phase4 submit --run E01 --cache "$OUT/tail" --checkpoint "$E01" \
  --output "$OUT/E01.zip" --device cuda
python -m phase4 validate --test "$P4/Test/test.json" --predictions "$OUT/E01.zip"

# Example for E09. Substitute E08 and its checkpoint for the E10 member.
python -m phase4 evaluate --run E05 --cache "$OUT/tail" --checkpoint "$OUT/e05/best_model.pt" \
  --output "$OUT/e05_probs" --device cuda
python -m phase4 submit --run E05 --cache "$OUT/tail" --checkpoint "$OUT/e05/best_model.pt" \
  --output "$OUT/E05.zip" --probabilities "$OUT/e05_probs/test_probabilities.csv" --device cuda
python -m phase4 ensemble --run E09 --cache "$OUT/tail" --teacher-logits "$OUT/teacher" \
  --member "$OUT/e05_probs" --output "$OUT/E09.zip"
```

By default `submit` uses the historical threshold for its run. Pass
`--threshold` for a newly calibrated run. The blends use the continuous
member probabilities and align rows by case ID, checking any provided labels.
For saved notebook files use `--member-prefix` with the full run name from
[results.csv](experiments/results.csv). For an archive audit, place the
original ZIPs, summary JSONs, per-split prediction CSVs and the three
validation/dev/test teacher-logit arrays in one private directory and run
`python -m phase4.analysis.replay --artifacts /path/to/artifacts --test /path/to/test.json`.

Build the standalone inference source for E01 with
`python scripts/package_phase4_winner.py --output /path/to/Phase4_E01_source.zip`.
It runs `python -m phase4.winner` with the trusted E01 classifier, the tokenizer
and the official test file. Regenerate the report figures and table with
`python code/phase4/analysis/figures.py` and compile the existing
`report/main.tex` from `report/`.

## Verification

The separate E01 package produced a valid ZIP from synthetic checkpoint weights
and the archived tokenizer in a standalone CPU smoke test.

- All ten ZIPs contain one root-level `prediction.json` and exactly the
  1,517 official case IDs. The replay reproduced all ten archived label
  vectors from saved scores or logits. It recovered the E09 and E10 weight
  and threshold searches and recomputed all local macro-F1 values.
- The first 32 official test windows matched the saved token and length
  arrays exactly with the archived tokenizer. The archived split files
  corroborated the validation, development and test row order used by E01
  teacher logits.
- The original E02 head loaded strictly and reproduced all 994 clean-dev
  labels using archived features on CPU (maximum absolute probability
  difference 0.000092). E03 and E07 adapter states loaded strictly into
  reconstructed GQA backbones (198 and 282 entries). Their original
  foundation tensors were unavailable locally. The original E08 student
  loaded strictly and matched the first 32 archived test labels using CPU
  FP32 (maximum absolute probability difference 0.001328).
- The E01 checkpoint exceeds the available 256 MiB Drive transfer limit.
  Full-model E01/E04/E05/E06 original-weight inference and original GPU
  training replays remain open. Replay of E01 saved logits verifies the
  recorded labels and packaging.
