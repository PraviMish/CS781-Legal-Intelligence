# Phase 2 — Legal named entity recognition

Given a complete judgment, predict exact end-exclusive character spans labeled with one of 12 legal entity types. The official metric is strict macro-F1 on the 35 hidden-label fold-3 judgments.

The **full-data majority ensemble** achieved **0.3577** (Codabench **971881**). It votes across full-data GQA, decoder, and distilled GQA student predictions. The original 56-document majority ensemble achieved **0.3342** (971884), and selected its voting rule on 14 validation judgments. The 70-document retraining used all gold judgments and fixed the final epoch at four for each member.

## Run history

| Run | Train docs | Selected validation macro-F1 | Codabench macro-F1 (ID) |
|---|---:|---:|---:|
| `gqa_100m_frozen_probe` | 56 | 0.16766 | — |
| `gqa_700m_full_bio_v1` | 56 | 0.26676 | 0.2960 (971886) |
| `decoder_100m_full_bio_v1` | 56 | 0.24495 | 0.2849 (971879) |
| `encoder_decoder_100m_lora_v1` | 56 | 0.02086 | — |
| `encoder_decoder_100m_lora_balanced_v2` | 56 | 0.04995 | 0.0459 (971880) |
| `prefix_lm_100m_lorma_plus_v1` | 56 | 0.06334 | 0.0651 (971887) |
| `gqa_7layer_supervised` | 56 | 0.24068 | — |
| `gqa_7layer_distilled` | 56 | 0.25122 | 0.2736 (971885) |
| `ensemble_v1_majority` | 56 | **0.30658** | 0.3342 (971884) |
| `gqa_full70_v1` | 70 | — | — |
| `decoder_full70_v1` | 70 | — | — |
| `student_kd_full70_v1` | 70 | — | — |
| `ensemble_full70_majority_v1` | 70 | — | **0.3577 (971881)** |

All five original voting rules are in [ensemble_scores.csv](experiments/ensemble_scores.csv). Majority was best by macro-F1. GQA confirmed by another member was best by micro-F1. Agreement reduced validation false positives from 1,435 to 495, with a small decrease in true positives (680 to 654). Balanced LoRA improved validation recall and macro-F1 while increasing false positives from 1,257 to 8,235. The report distinguishes these observations from hypotheses about the weak adapter results. [Master report](../../report/main.tex)

## Evidence and configuration

- [results.csv](experiments/results.csv) separates selected validation scores, archived re-evaluations, and Codabench scores. The primary official scores and IDs are transcribed from the seven finished entries in the supplied Codabench screenshot dated 9 October 2026. The original notebook and Drive histories and validation summaries supply local scores. A dash or blank means unavailable. The frozen probe's archived re-evaluation (0.16752) differs slightly from its selected epoch score (0.16766). The distillation comparison re-evaluated the teacher at 0.26685. The saved GQA predictions used for voting score 0.26676.
- The aggregate-only files in [experiments/](experiments/) contain validation per-class metrics, training trajectories, class frequencies, voting results and diagnostics. The training histories are from saved run artifacts, and the figure script reads these aggregates.
- [configs.py](configs.py) records each run's architecture, foundation, optimizer schedule inputs, batch sizes, loss weights, checkpoint cadence, and selection protocol. AdamW weight decay is 0.01 and the gradient norm cap is 1. Full, adapter, and student runs use a 6% warmup followed by linear decay. The frozen probe uses a constant rate.
- BIO uses the official label order and the archived Phase 1 tokenizer. It trims only token offset intervals. It uses 256 owned tokens and up to 128 context tokens on each side, reconstructing one sequence per document before BIO decoding. The adapter models use bidirectional attention with pad keys masked. The GQA and decoder models retain causal attention.
- The original 100M GQA probe requires `efficient_gqa/milestones/best_100M.pt`. The full and student GQA runs use `efficient_gqa/milestones/best_up_to_700M.pt` (recorded at 700,047,360 tokens). Decoder, encoder–decoder, and Prefix-LM use the corresponding original 100M `best.pt`. Keep Phase 1 and Phase 2 artifacts separate.

## Reproduce the experiments

Use the repository setup in the [root README](../../README.md), including `PYTHONPATH=code`. Keep the official `Phase_2_LNER/Train` and `Phase_2_LNER/Test` files, the Phase 1 tokenizer, trusted checkpoints, and generated data in a private directory outside the checkout. Use the historical seed from the private archived `validation_split.json` for a replay. The CLI defaults to 42 for new runs. Supplying `--split` restores the exact held-out document list. GPU BF16 matches the historical numerical mode. The CPU path is suitable for smoke tests.

```bash
# Set these paths to your own private files.
TRAIN=/path/to/Phase_2_LNER/Train
TEST=/path/to/Phase_2_LNER/Test/fold3-text.json
TOK=/path/to/legal_bpe_32k.json
SPLIT=/path/to/validation_split.json
SEED=$(python -c 'import json, sys; print(json.load(open(sys.argv[1]))["seed"])' "$SPLIT")
P1=/path/to/phase1/models
OUT=/path/to/private/phase2-replay

# 1. Fine-tune the original GQA teacher. For other variants select a run above
#    and pass its corresponding Phase 1 snapshot.
python -m phase2 train --run gqa_700m_full_bio_v1 --train-dir "$TRAIN" \
  --split "$SPLIT" --seed "$SEED" --tokenizer "$TOK" \
  --foundation "$P1/efficient_gqa/milestones/best_up_to_700M.pt" \
  --output-dir "$OUT/gqa" --device cuda

# 2. Train either seven-layer student from the same archived initialization.
#    The supervised control takes no teacher argument.
python -m phase2 train --run gqa_7layer_distilled --train-dir "$TRAIN" \
  --split "$SPLIT" --seed "$SEED" --tokenizer "$TOK" \
  --student-initialization /path/to/student_initialization.pt \
  --teacher-checkpoint "$OUT/gqa/best.pt" --output-dir "$OUT/distilled" --device cuda

# 3. Evaluate a selected checkpoint on the original held-out set.
python -m phase2 evaluate --run gqa_700m_full_bio_v1 --train-dir "$TRAIN" \
  --split "$SPLIT" --tokenizer "$TOK" --checkpoint "$OUT/gqa/best.pt" \
  --predictions "$OUT/gqa/validation_predictions.json"

# 4. Compare all original voting rules with saved validation outputs.
python -m phase2 compare-votes --train-dir "$TRAIN" --split "$SPLIT" \
  --tokenizer "$TOK" --seed "$SEED" --diagnostics \
  --gqa /path/to/gqa/validation_predictions.json \
  --decoder /path/to/decoder/validation_predictions.json \
  --student /path/to/distilled/validation_predictions.json
```

The student can also initialize from `--foundation` with the 700M GQA snapshot. Both student modes copy layers 0, 2, 4, 6, 8, 10, and 12 and share the head initialization. Distillation caches float16 teacher logits in indexed shards and uses only supervised positions in its 70% cross-entropy and 30% temperature-scaled KL objective. Training automatically resumes from a compatible reconstructed `last.pt` in the same output directory. Use a fresh directory for a new configuration or for re-training from notebook artifacts. The historical `best.pt` and `final.pt` formats are accepted for inference. Use a separate output directory per concurrent run.

To train the winner's members, select `gqa_full70_v1`, `decoder_full70_v1`, and `student_kd_full70_v1` in that order. Use the same respective foundation or student initialization, omit `--split`, and pass the full-data GQA **final.pt** as `--teacher-checkpoint` for the full-data student. All three runs save `final.pt` at epoch 4.

## Rebuild the winning submission

```bash
# Run once for each of the three full-data members.
python -m phase2 submit --run gqa_full70_v1 --tokenizer "$TOK" \
  --checkpoint /path/to/gqa_full70_v1/final.pt --test-json "$TEST" \
  --output "$OUT/gqa_full70_v1_submission.zip"

# Then use the three member ZIPs (or their prediction.json files).
python -m phase2 ensemble --test-json "$TEST" --strategy majority \
  --gqa "$OUT/gqa_full70_v1_submission.zip" \
  --decoder "$OUT/decoder_full70_v1_submission.zip" \
  --student "$OUT/student_kd_full70_v1_submission.zip" \
  --output "$OUT/ensemble_full70_majority_v1_submission.zip"

python -m phase2 validate --test-json "$TEST" --nonoverlapping \
  --predictions "$OUT/ensemble_full70_majority_v1_submission.zip"
```

For the original ensemble, use the held-out protocol's GQA, decoder, and distilled member ZIPs. For frozen or adapter runs, pass both the historical `best.pt` and the matching Phase 1 `--foundation` to `submit` or `evaluate`. Only load trusted checkpoints. The generated ZIP contains exactly `prediction.json` at its root.

## Verification

The reconstruction matched the archived split, IDs, token offsets and BIO labels for all 105 judgments. It regenerated both majority prediction dictionaries exactly from the saved member ZIPs (3,364 original and 3,306 full-data spans). All 12 archived ZIPs passed structure, offset bounds, ID, and label checks. The five voting scores, error counts, and paired bootstrap were replayed from archived validation predictions. The four saved small checkpoints loaded with matching head or adapter keys and shapes over reconstructed backbones. The large original Phase 1 weights and Phase 2 full-model checkpoints exceeded the Drive transfer limit. Original-weight inference and training replay remain open. Some original head initializations and the probe's shuffle depended on prior notebook RNG state. The CLI uses an explicit seed for fresh runs. Bitwise training replay would require the original runtime state.

Regenerate the three figure pairs with `python code/phase2/analysis/figures.py` from the root. From `report/`, build `main.tex` with Tectonic or run `pdflatex main.tex` twice.
