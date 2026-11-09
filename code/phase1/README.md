# Phase 1 — From-scratch Legal Language Modeling

## Goal and evidence

Generate an **exactly 128-word** continuation from a **512-word** legal prompt. The official score is BERTScore F1 (ROUGE-L F1 and BLEU-4 secondary). Hidden test references are never available to participants. There were **seven valid evaluated submissions** in this study. The official winning submission is the **initial Efficient-GQA 100M** checkpoint with the **legacy guarded sampling decoder**, not the later 700M checkpoint.

The public code contains all distinct original architectures and the continuation/decoding experiments because it documents the complete research path. The separate course ZIP is smaller and contains only the best model's code and necessary shared utilities.

## Data and tokenizer

The official Phase-1 data has four sources: `train_acts.json`, `train_high_courts.json`, `train_supreme_court.json`, and `train_tribunals.json`, plus hidden-reference `Test/test.json`. No external corpus or pretrained model was used. A byte-level BPE tokenizer was freshly trained with 32,000 vocabulary entries, four special tokens, and a balanced ~100M-character sampling budget (25M/source).

The initial run used an approximately 500M-token cache with source weights proportional to `raw_JSON_bytes ** 0.7`; a deterministic SHA-1 document split reserved ~1% for validation. Forty pseudo-dev prompt/continuation samples were drawn from each source (160 total). Later, the entire corpus was tokenized into approximately **4.488B usable training BPE tokens**, and continuation source weights were recalculated from actual token counts instead of raw file sizes. Only 700M tokens were consumed during the longest training run.

## Four model architectures

| Architecture | Parameter count | Key design | 100M private BERTScore F1 |
|---|---:|---|---:|
| Standard decoder-only | 110,615,040 | 12 blocks; learned positional embeddings, causal MHA | 0.83465570 |
| Encoder--decoder | 101,972,480 | 7 encoder + 7 decoder blocks; cross-attention | 0.82569140 |
| Prefix-LM | 110,615,040 | Bidirectional prefix, causal target | 0.82551551 |
| **Efficient GQA** | **112,678,656** | **14 blocks, 768 hidden, 12 query / 4 KV heads, RoPE, RMSNorm, SwiGLU, 2048 FFN** | **0.83479506** |

Exact architecture implementations are in [`models.py`](models.py); checkpoint parameter names are intentionally retained. For GQA, rotary transformations are in a method with a distinct name rather than `nn.Module.apply`, avoiding a class-method collision found during notebook development.

## Training procedure

- Context/block length: 1,280 BPE IDs.
- Objective: causal next-token cross-entropy for decoder/GQA; prefix-conditioned target prediction for Prefix-LM; source/target teacher-forcing for encoder--decoder.
- AdamW: peak learning rate `3e-4`, betas `(0.9, 0.95)`, epsilon `1e-8`, weight decay `0.1` applied to matrix parameters only.
- Initial warmup: 5M consumed tokens, followed by a cosine schedule over the planned 500M budget, clipped gradient norm 1.0.
- Effective batch target: 65,536 tokens; actual 61,440 tokens/step at length 1,280 with original microbatch/accumulation factors (decoder/GQA/Prefix: 8×6; encoder--decoder: 4×12).
- The 100M winner was preserved and continued through 300M, 500M, and 700M total tokens using the same architecture and optimizer state with a new cosine schedule to `3e-5` at 700M.
- Latest and best-validation checkpoints were saved separately. Resumable latest checkpoints include model/optimizer/sampler/CPU/CUDA/NumPy/Python RNG states, while the best model stores weights and validation metadata.

## Official results (all 1,000 test examples)

| Submission | BERTScore F1 | ROUGE-L F1 | BLEU-4 |
|---|---:|---:|---:|
| decoder_only | 0.83465570 | 0.19630469 | 0.05258763 |
| **efficient_gqa (100M)** | **0.83479506** | **0.19745313** | **0.05385350** |
| efficient_gqa_300M | 0.83080482 | 0.17181250 | 0.03257670 |
| efficient_gqa_500M_tuned | 0.78932995 | 0.14202344 | 0.00777384 |
| efficient_gqa_700M_tuned | 0.80992270 | 0.14757031 | 0.01275489 |
| encoder_decoder | 0.82569140 | 0.16444531 | 0.02212777 |
| prefix_lm | 0.82551551 | 0.16594531 | 0.02618290 |

The results come from the official score reports supplied with the project. All raw values are stored in `experiments/official_and_local_results.csv`; charts and the interpretation are in the report. Perplexity continued decreasing to about 6.88 at 700M, but private BERTScore did not improve. The local 500M 160-example decoding comparison favored greedy full-prompt ROUGE-L F1 0.2933 over low-temperature 0.2761; this did not generalize to the official metric. **Do not interpret the 500M/700M scores as a clean single-factor test of training duration**, because their decoder also changed.

## Files

| File | Purpose |
|---|---|
| `models.py` | Four exact model definitions and factory |
| `tokenizer.py` | Build 32k byte-level legal BPE tokenizer |
| `data.py` | Stream official JSON, deterministic document split, memory-mapped token sampling |
| `prepare.py` | Run initial/full data-preparation workflow and create source-weight metadata |
| `train.py` | Train or resume any architecture; supports 100M initial and 300/500/700M continuation |
| `generate.py` | Historical decoder and later greedy/sampling configurations |
| `evaluate.py` | Held-out and pseudo-dev local evaluation |
| `submit.py` | Resume predictions, verify IDs and 128 words, ZIP `prediction.json` at archive root |
| `configs/` | Explicit source-weight files |
| `experiments/` | Official metrics, local metrics, validation curve, decoding sweep |
| `analysis/figures.py` | Rebuild report figures from CSV data |

For the 500M decoding sweep, pass `--max-examples 40` to the evaluation CLI and select each configuration named in `experiments/decoding_sweep.csv`. Repeat the two leading configurations with `--max-examples 160`. The sweep's `legacy_guarded` hashes the configuration name together with the prompt. Use `legacy_word_guard_v1` for the official 100M submission and the separate 500M baseline evaluation, which recorded ROUGE-L F1 0.221375 and sacreBLEU 10.102891 on 160 examples.

## Basic use (from repo root with `PYTHONPATH=code`)

```bash
# 1. Official data downloaded separately from the course's Hugging Face dataset.
python -m phase1.prepare --data-root /path/to/Phase_1_Pretraining \
  --tokenizer /path/to/artifacts/legal_bpe_32k.json \
  --output-dir /path/to/artifacts/initial_tokens --mode initial

# 2. Four initial architectures; change architecture and output folder per run.
python -m phase1.train --architecture efficient_gqa \
  --train-dir /path/to/artifacts/initial_tokens \
  --val-dir /path/to/artifacts/initial_tokens \
  --weights-json /path/to/artifacts/initial_tokens/source_weights.json \
  --output-dir /path/to/artifacts/models/efficient_gqa --target-tokens 100000000

# 3. To reproduce the full-corpus continuation, first tokenize all four sources.
python -m phase1.prepare --data-root /path/to/Phase_1_Pretraining \
  --tokenizer /path/to/artifacts/legal_bpe_32k.json \
  --output-dir /path/to/artifacts/full_tokens --mode full

# 4. Generate the best original submission using the original best-validation checkpoint.
python -m phase1.submit --architecture efficient_gqa \
  --checkpoint /path/to/artifacts/models/efficient_gqa/milestones/best_100M.pt \
  --tokenizer /path/to/artifacts/legal_bpe_32k.json \
  --test-json /path/to/Phase_1_Pretraining/Test/test.json \
  --output-dir /path/to/artifacts/submissions/efficient_gqa_100M \
  --decoding legacy_word_guard_v1

# 5. Repeat for 300M/500M/700M using the corresponding frozen checkpoint
#    and decoding configuration (see experiments/ for the recorded settings).
```

For continuation training, resume with `--resume-checkpoint` and explicitly supply `--continued-base-tokens` (original checkpoint `tokens_seen`) and `--continued-base-lr` (first optimizer param group's saved `lr`); keep those settings fixed across the 300M, 500M and 700M milestones. Do not silently reset AdamW or use the 700M model for the best-model code archive.

## Verification and limitations

Architecture counts and Python syntax have been checked in this reconstruction. Verification against the real saved original `.pt` checkpoints is still required to confirm strict load and matched deterministic generation, because the large checkpoints are not available inside this published code package. Reproducing the exact original 500M-token cache or 4.488B-token full corpus also requires access to the same official dataset and tokenizer artifact.

Local scores and official scores use different reference sets, and local sacrebleu uses the 0--100 scale. The user-provided official score screenshots do not contain overall leaderboard rank. All claims should refer to the actual scale and provenance, not invent ranking or missing tests.

**Continuing on the full corpus:** supply `--val-weights-json` pointing to the initial source-weight JSON while `--weights-json` points to full-corpus weights. This preserves the validation mixture used to select the original winner. At the first 100M-to-full-corpus transition, the new sampler is reset deterministically; later checkpoints restore the sampler RNG for uninterrupted continuation. The historical source notebook used an author-specific seed; the public CLI uses a generic default and accepts an explicit `--seed` for exact owner-run replay.

**Critical checkpoint warning:** after continued pretraining, `models/efficient_gqa/best.pt` points to a later best-validation checkpoint and is **not** the official 100M Codabench winner. Use the preserved `models/efficient_gqa/milestones/best_100M.pt` snapshot (or the exact original winning checkpoint with equivalent state) for the best-model submission. Never select a filename called `best.pt` without checking its training-token metadata.
