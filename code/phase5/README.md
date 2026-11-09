# Phase 5 | Legal statute identification

**E02 Efficient GQA full fine-tuning won the six recorded Codabench submissions at 0.3875 macro-F1** (1014549). The submitted system uses epoch 3 of the Phase 5 E02 classifier and its matching development-fitted thresholds. E06, a distilled six-layer GQA, finished 0.0033 behind. Scores and IDs are transcribed from the supplied 28 October 2026 Codabench screenshot.

| Run | Phase 1 foundation and adaptation | Objective | LR | Epoch selected / trained | Global dev F1 | Calibrated dev F1 | Official F1 | Submission |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E01 | GQA 700M, LoRMA+ r8 | Weighted BCE | 4e-4 | 4 / 4 | 0.3432 | 0.3645 | 0.3532 | 1014548 |
| E02 | GQA 700M, full tuning | ASL | 2e-5 | 3 / 3 | 0.3741 | 0.3956 | **0.3875** | 1014549 |
| E03 | Decoder 100M, top four blocks and final norm | Weighted BCE | 8e-5 | 3 / 4 | 0.3448 | 0.3689 | 0.3565 | 1014550 |
| E04 | Prefix-LM 100M, nonlinear width-32 adapters | Weighted BCE | 4e-4 | 4 / 4 | 0.2870 | 0.3097 | 0.2967 | 1014552 |
| E05 | Encoder–decoder 100M, attention LoRA r8 | ASL | 4e-4 | 3 / 4 | 0.2150 | 0.2688 | 0.2619 | 1014553 |
| E06 | GQA 700M, six inherited blocks, full tuning | Weighted BCE + distillation | 4e-5 | 3 / 4 | 0.3757 | 0.3954 | 0.3842 | 1014554 |

[Official record](experiments/official_results.csv) · [Development calibration and archive verification](experiments/archive_verification.json) · [Epoch histories](experiments/training_history.csv)

## Data and representation

Use the [pinned Phase_5_LSI revision](https://huggingface.co/datasets/Exploration-Lab/CS-781-Capstone/tree/93b0059be756c6c461f3a225de25aabda7d5f84a/Phase_5_LSI). Train has 42,750 rows, development 10,181, and hidden-label test 13,019. The official files are arrays with id and text fields and, for labeled splits, a nonempty labels list. IDs normalize to strings. The gold vector has 100 independently scored columns in ascending statute-ID order (0 through 99). An exact-text grouped holdout within training leaves 38,444 optimization cases and 4,306 internal validation cases. The notebook records 81 train–development and 108 train–test shared exact texts across the original partitions.

Gold cardinality averages 3.705 in training and 3.916 in development. Training support ranges from 272 cases for statute 84 to 11,730 for statute 5. Statute pairs 5/34, 43/97 and 12/13 occur together in 912, 862 and 849 development cases. The original 32k byte-level BPE uses PAD/BOS/EOS/UNK IDs 0/1/2/3. A short case uses one 510-content-token window. A longer case uses the three rounded, evenly spaced 510-token spans including both ends, each wrapped in BOS/EOS. The median development case has 799 BPE tokens. Of development cases, 35.7% exceed the combined 1,530-token content budget.

The classifier pools across per-window EOS states. Each state passes through layer norm, dropout 0.10 and a shared linear head. The document logit is normalized log-sum-exp with inverse temperature 2 over active windows. Prefix-LM uses a full bidirectional prefix. E05 reads the encoder and its decoder is outside the classification forward path. See [dataset aggregates](experiments/dataset_statistics.json) and [statute frequencies](experiments/label_frequencies.csv).

## Training and interpretation

All methods transfer original Phase 1 weights. The archived GQA milestone is best_up_to_700M.pt. The other architectures use their original 100M best.pt. LoRMA+ transforms the frozen projection output with a rank-8 additive correction, while LoRA computes its correction from the input. E04 uses a zero-initialized up-projection in a GELU bottleneck on the same attention projection types. E01 provides a parameter-efficient GQA reference, E02 tests full adaptation of the same foundation with an asymmetric objective, and E03 tests selective unfreezing on a causal decoder. E04 and E05 test bidirectional window representations with different adapters. The comparisons also change foundation, loss or training budget, depending on the pair.

Training uses AdamW at a fixed per-run learning rate, weight decay 0.01, one document per backward pass, accumulation over 16 documents, and gradient clipping at 1. Weighted BCE scales positives by sqrt((N-f)/max(f,1)) clipped to [1,8]. ASL dampens easy negatives using exponent 4 and clip 0.05. The executed continuation restored completed E01–E04 runs and resumed E05, then trained E06. The archive selects checkpoints on internal macro-F1 at a fixed threshold of 0.30. E01 beat E02 at that threshold (0.3210 vs. 0.2203) and supplied E06's teacher logits. E06 inherits blocks 0, 3, 5, 8, 10 and 13 and mixes hard weighted BCE and soft BCE equally, with temperature 2 and the square of temperature scaling the soft term. No matched gold-only six-layer run is available.

Calibration searches 0.05–0.70 in steps of 0.025 for a global threshold. Per-statute fitting subtracts a distance penalty from F1 and shrinks the optimum toward the global threshold using support n and weight n/(n+12). Labels with fewer than four positives or negatives keep the global threshold. The same development set supplies the local scores, so calibration gains are in-sample. During submission, thresholds are inclusive and an empty set falls back to the highest probability. No hard label-count cap is used. The archive holds a single root prediction.json with one string id and a pred list of distinct statute IDs per row.

E02 had 0.4979 micro-F1 and 5.04% exact-set accuracy on development. Statute 8 had 80 misses and 217 false positives, while statute 2 had 92 misses and 29 false positives. E02 and E06 disagree on 91.47% of test statute sets. There is no hidden-label error attribution or matched long-context ablation. See the [master report](../../report/phase5.tex) for the detailed comparison.

## Reproduce

Install [requirements](../requirements.txt) and set PYTHONPATH=code. Point the following variables to private local paths. Use one output directory per experiment and keep datasets, logits, predictions and weights outside Git.

~~~bash
P5=/path/to/Phase_5_LSI
TOK=/path/to/legal_bpe_32k.json
GQA=/path/to/best_up_to_700M.pt
OUT=/path/to/private/phase5
python -m phase5 prepare --train "$P5/Train/train.json" --dev "$P5/Train/dev.json" \
  --test "$P5/Test/test.json" --tokenizer "$TOK" --output "$OUT/cache"
python -m phase5 train --run E02 --foundation "$GQA" --cache "$OUT/cache" --output "$OUT/E02"
python -m phase5 evaluate --run E02 --cache "$OUT/cache/val_internal.npz" \
  --checkpoint "$OUT/E02/best.pt"
python -m phase5 calibrate --run E02 --cache "$OUT/cache/dev.npz" \
  --checkpoint "$OUT/E02/best.pt" --output "$OUT/E02/thresholds.npy"
python -m phase5 submit --run E02 --checkpoint "$OUT/E02/best.pt" \
  --thresholds "$OUT/E02/thresholds.npy" --tokenizer "$TOK" \
  --test "$P5/Test/test.json" --output "$OUT/E02.zip"
python -m phase5 validate --test "$P5/Test/test.json" --predictions "$OUT/E02.zip"
~~~

Use --core-ids with the private archived train_core_ids.json to bind the original split. The default seed is 42. The same training and evaluation commands apply to each configuration with its matching foundation. Repeating training restores last.pt at an optimizer boundary. To resume a trusted notebook run, place its last.pt and best.pt together in the selected output directory and use the same ordered data and original seed. The port's new checkpoints record a cache and teacher fingerprint. CUDA automatically selects BF16 where supported and FP16 otherwise. CPU uses FP32 by default. Device and precision can be specified explicitly.

For E06, generate teacher targets from E01 and then start the six-layer student.

~~~bash
python -m phase5 teacher-logits --run E01 --checkpoint "$OUT/E01/best.pt" \
  --cache "$OUT/cache/train_core.npz" --output "$OUT/E01_teacher.npz"
python -m phase5 train --run E06 --foundation "$GQA" --cache "$OUT/cache" \
  --teacher "$OUT/E01_teacher.npz" --output "$OUT/E06"
~~~

For an archived raw teacher array use --archived-teacher with --teacher and the confirmed original case order. The public port supports local restore from the original checkpoint fields. The original remote backup service and its credentials are external.

Build the E02-only archive with python scripts/package_phase5_winner.py --output /path/to/Phase5_E02_source.zip. It contains the GQA inference path and excludes the other experimental backbones. Use the original E02 best.pt and thresholds.npy together. The executed submission pass used BF16. Use --precision bf16 on compatible CUDA hardware for trained-weight comparisons. Regenerate figures with python code/phase5/analysis/figures.py and compile report/main.tex from the report directory.

All archived ZIPs passed validation against the verified test file. Recalibration from the six saved development logit arrays reproduces all six threshold arrays and aggregate scores. The original E02 checkpoint exceeds the available Drive transfer limit. The released inference path was checked with synthetic weights. See the [evidence audit](../../docs/PHASE5_RECOVERY.md).
