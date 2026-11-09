# Phase 3 — Rhetorical Role Prediction (legal judgments)

**Research status:** 16 trained classifiers, 2 inference-only postprocessors, 18 validated local submission archives; **15 official completed evaluations**, 1 pending/preparing, 2 marked failed at the time of the supplied screenshot. **Best official submission: `989815`, C2 bidirectional-context Efficient GQA + LoRA, Macro-F1 `0.4249`.** The public research implementation includes all meaningful approaches, not just the winner.

This directory continues the same official-data-only, Phase-1-initialized legal-intelligence project. It is not a new pretrained model and uses no external inference API or external pretrained weights. The program is **independently configurable** for dataset, original tokenizer, foundation checkpoints, experiment checkpoints and output directory. Weights and official data are deliberately **not committed** to GitHub.

## 1. Task and evidence

An input judgment is an ordered sequence of sentences. The model must return one of **13 official rhetorical-role labels** for each sentence, preserving document IDs and sentence order. The categories are `Fact`, `Issue`, `ArgumentPetitioner`, `ArgumentRespondent`, `Statute`, `Dissent`, `PrecedentReliedUpon`, `PrecedentNotReliedUpon`, `PrecedentOverruled`, `RulingByLowerCourt`, `RatioOfTheDecision`, `RulingByPresentCourt`, `None`. The numeric IDs are **loaded from `label_vocab.json`**, never assumed from this textual order.

Official dataset: Hugging Face `Exploration-Lab/CS-781-Capstone`, under `Phase_3_RR/`:

```text
Train/CL_train.json  Train/CL_dev.json  Train/IT_train.json
Train/IT_dev.json   Train/label_vocab.json
Test/CL_test.json   Test/IT_test.json
```

The actual train/development format is a **list of dictionaries**, each with `id`, `text` (ordered sentences), and `labels`. Test files are instead dictionaries `{doc_id: [sentences...]}`. This incompatibility caused an early prototype ingestion error and is handled explicitly in `data.load_official`.

| Domain | Train docs | Train sentences | Dev docs | Dev sentences | Test docs | Test sentences |
|---|---:|---:|---:|---:|---:|---:|
| Corporate Law (CL) | 40 | 10,593 | 5 | 1,396 | 5 | 1,339 |
| Information Technology (IT) | 40 | 6,252 | 5 | 746 | 5 | 858 |
| **Total** | **80** | **16,845** | **10** | **2,142** | **10** | **2,197** |

Among the training labels, `PrecedentOverruled` occurs **3 times (IT only)**, and `Dissent` appears **284 times (CL only)**. Both have **zero development examples**, so macro-F1 computed over all 13 classes assigns them zero development F1 by convention; this is a limitation of validation, not evidence about unobserved hidden-test examples. The 95th percentile of sentence lengths is below 128 BPE tokens; the study uses **128 total tokens**, keeps 75% from the start and 25% from the end when truncating. The tokenizer is the **original Phase-1 32k byte-level BPE** (special IDs `pad=0`, `bos=1`, `eos=2`, `unk=3`). It must not be retrained under a new vocabulary order.

## 2. Modelling: what was actually implemented

**Phase-1 backbones (all original checkpoints):** 12-block decoder-only Transformer; 7+7 encoder-decoder (encoder-only features for Phase-3 classification); 12-block prefix-LM (bidirectional sentence encoding); 14-block Efficient GQA decoder with RoPE, SwiGLU and RMSNorm. The experiments compare original 100M consumed-token snapshots and the extended 700M GQA snapshot. Parameter counts and actual pretraining objectives differ, so the first group is a **transfer comparison of submitted systems**, not a clean isolation of a single attention mechanism.

**Task head:** a dropout + linear projection maps the last non-padding EOS hidden state to 13 logits. GQA and standard decoder attention are causal, and right padding cannot affect the EOS representation. Prefix-LM and encoder-decoder use explicit key-padding masks in their bidirectional encoders.

**Adaptation:** Group A uses attention-projection LoRA (`rank=8`, `alpha=16`); Group B fixes the 700M GQA backbone and compares LoRMA+ (identity-initialized multiplicative output-side residual low-rank transform), IA³ (activation scaling), learned per-layer attention KV prefixes, and full fine-tuning. These represent substantially different parameterizations; they are not rank-only hyperparameter changes.

**Context:** C1 puts the *preceding* sentence before the target. C2 puts *preceding and following* sentences **before** the target, separated by EOS. Crucially, the winning GQA backbone remains **causal**: this reordering allows its target EOS to see both neighbors. It is not a fully bidirectional-attention Transformer and it is not a hierarchical full-document neural encoder. The context layout allocates up to 65% of the 126-token content budget to target text and divides remaining tokens among available neighbors. C3 uses train-set label-transition probabilities and document-wide Viterbi postprocessing of A5 emission scores; it is not end-to-end CRF or neural hierarchical training.

**Domains:** D1 trains only CL and D2 only IT, with evaluation on both dev domains. D3 uses each specialist's predictions on its own test domain; this is an inference-only hybrid, not an additional trained model.

**Distillation:** E1 is a six-layer, fully trainable decoder student initialized from Phase-1 decoder layers `[0, 2, 4, 6, 8, 11]`. E2 uses the **same** student architecture with ground-truth CE + teacher-logit KL at `T=2`, `alpha=0.5`. The teacher is B4 (the strongest development classifier among A5 and B1–B4, and not the official context winner). The measured teacher/student throughput was approximately 1,035 versus 3,823 sentences/s (3.69×), but GPU peak-memory comparisons were not isolated and should **not** be claimed as independently measured savings.

**Balancing:** F1 trains the GQA LoRA architecture with capped inverse-square-root class weights, computed from train labels only; absent classes cannot be learned from zero support.

## 3. Experimental register and official Codabench evidence

`experiments/local_results.csv` is an exact transcription of the 16 trained variants from the executed notebook. `experiments/official_results.csv` is an explicit mapping of ordered uploads to the supplied Codabench screenshot. **Local development Macro-F1 is not official hidden-test Macro-F1.** `C3` has a development transition-strength sweep (best `0.3`, Macro-F1 `0.352739`); `D3` has no independent development score.

| # | Variant | Local pooled dev Macro-F1 | Codabench Macro-F1 | Screenshot status / ID |
|---:|---|---:|---:|---|
| 1 | A1_decoder_lora | 0.2934 | 0.3310 | Finished · 989798 |
| 2 | A2_encdec_lora | 0.1204 | 0.1361 | Finished · 989799 |
| 3 | A3_prefix_lora | 0.2350 | 0.2867 | Finished · 989800 |
| 4 | A4_gqa100_lora | 0.3073 | 0.3299 | Finished · 989802 |
| 5 | A5_gqa700_lora | 0.3291 | 0.3714 | Finished · 989804 |
| 6 | B1_lorma_plus | 0.3171 | — | **Failed** · 989805 |
| 7 | B2_ia3 | 0.2288 | — | Preparing · 989807 |
| 8 | B3_prefix | 0.1983 | 0.2379 | Finished · 989809 |
| 9 | B4_full | 0.3330 | 0.4005 | Finished · 989810 |
| 10 | C1_previous_context | 0.3424 | 0.3976 | Finished · 989811 |
| **11** | **C2_bidirectional_context** | **0.3851** | **0.4249** | **Finished · 989815** |
| 12 | C3_document_viterbi | 0.3527 | 0.3791 | Finished · 989819 |
| 13 | D1_CL_only | 0.2215 | 0.2586 | Finished · 989821 |
| 14 | D2_IT_only | 0.3020 | 0.3710 | Finished · 989822 |
| 15 | D3_domain_specialist_hybrid | — | 0.3957 | Finished · 989823 |
| 16 | E1_student_supervised | 0.3234 | 0.3679 | Finished · 989824 |
| 17 | E2_student_distilled | 0.3200 | 0.3638 | Finished · 989825 |
| 18 | F1_balanced_lora | 0.3459 | — | **Failed** · 989828 (numeric 0.3703 shown) |

Two duplicate `Submitting` entries were also visible for C2 (989813) and C3 (989818). The screenshot is a **point-in-time** status record; it does not establish the later outcome of unfinished entries. The failed row for F1 contains a numeric value, but it is **not** counted as a successful official evaluation. The best displayed finished primary score is **C2 = 0.4249** (989815).

Under a matched backbone/adapter comparison, adding both neighboring sentences improves local Macro-F1 from A5 **0.3291** to C2 **0.3851**; on the official hidden test, the corresponding scores increase from **0.3714** to **0.4249**. This is compelling context evidence for the studied model and data, but only one seed and a ten-document dev split were used. The best C2 per-class development F1 values include `Statute 0.839`, `Fact 0.613`, `RatioOfTheDecision 0.515`, `RulingByLowerCourt 0.242`, `None 0.250`. Rare and cross-role distinctions are difficult; see the comprehensive report chapter and `experiments/top_confusions.csv`.

## 4. Repository layout

```text
code/phase3/
  __main__.py             CLI: audit/train/predict/validate/viterbi/hybrid
  prepare.py              optional official Hugging Face dataset downloader
  configs.py              all 16 trainable experiment configurations
  data.py                 official data, 32k tokenizer, context, batching
  models.py               original Phase-1 backbone + task/adaptation layers
  runtime.py              training, checkpoint resume, inference
  metrics.py              13-way macro metrics including absent classes
  context.py              document-level label transitions and Viterbi
  submission.py           exact document/order/label/ZIP validation
  analysis/figures.py     regenerate seven PDF charts from recorded CSVs
  analysis/latex_tables.py  generate full submission LaTeX table
  experiments/*.csv       notebook and screenshot evidence, no raw data
```

Shared Phase-1 architecture definitions are imported from `code/phase1/models.py` with **no changes**. The source-only best-model package is independently distributed as `phase3_best_model_code.zip` (not inside the complete experiment package).

## 5. Reproduction commands

Use Python 3.10+ and a PyTorch build appropriate for your hardware. From the repository root:

```sh
pip install -r code/requirements.txt
pip install tokenizers
# Optional, to download seven official Phase-3 files directly from HF:
pip install huggingface_hub
python -m phase3.prepare --output /datasets
export PYTHONPATH="$PWD/code"

# Validate dataset format, training statistics and exact label vocabulary:
python -m phase3 audit --data /datasets/Phase_3_RR

# Train or resume winner (requires preserved foundation and tokenizer):
python -m phase3 train --experiment C2_bidirectional_context \
  --data /datasets/Phase_3_RR \
  --tokenizer /artifacts/legal_bpe_32k.json \
  --foundations /artifacts/phase1 \
  --checkpoints /artifacts/phase3/checkpoints/research \
  --output /artifacts/phase3

# Reconstruct exact ordered test predictions from the best saved adapter:
python -m phase3 predict --experiment C2_bidirectional_context \
  --data /datasets/Phase_3_RR \
  --tokenizer /artifacts/legal_bpe_32k.json \
  --foundations /artifacts/phase1 \
  --checkpoints /artifacts/phase3/checkpoints/research \
  --output /artifacts/phase3/submissions

# Independently inspect a produced archive (10 IDs, 2197 sentence labels):
python -m phase3 validate --data /datasets/Phase_3_RR \
  --submission /artifacts/phase3/submissions/C2_bidirectional_context/submission.zip

# Other variants: IDs in configs.RUNS; same --experiment flag.
# Viterbi and hybrid postprocessors:
python -m phase3 viterbi --data /datasets/Phase_3_RR --tokenizer /artifacts/legal_bpe_32k.json \
  --foundations /artifacts/phase1 --checkpoints /artifacts/phase3/checkpoints/research
python -m phase3 hybrid --data /datasets/Phase_3_RR \
  --cl /artifacts/phase3/submissions/D1_CL_only/submission.zip \
  --it /artifacts/phase3/submissions/D2_IT_only/submission.zip
```

Full training is computationally demanding. Defaults: 3 epochs; AdamW with weight decay .01; task learning rate 1.5e-4 for PEFT and 3e-5 for full/student; batch 8, accumulation 2, evaluation batch 16; BF16 on supported GPU; 5% warmup and cosine; gradient clipping 1; early-stop patience 2; save best/last checkpoint and optimizer/RNG state at optimizer update boundaries. `runtime.train` resumes from `last.pt`. CPU inference/smoke tests work, but throughput differs from the original accelerator.

For distillation, train `B4_full` first; the teacher's **training-sentence logits** are then cached from B4. For E1 and E2, use the original decoder-only Phase-1 checkpoint and six-layer student. The genuine mathematical objectives are explained in `report/phase3.tex`.

## 6. Figures and testing

```sh
python code/phase3/analysis/figures.py
python code/phase3/analysis/latex_tables.py
PYTHONPATH="$PWD/code" python -m pytest -q code/tests/test_phase3.py
# From report/ after the integration patch: pdflatex main.tex twice.
```

Plots are generated from the provided **aggregate** numeric records, not from fabricated per-example predictions. A source-only test suite checks ingestion, alignment, class metrics, causal context order, adapter identity/gradients, checkpoint tensor names, Viterbi, and ZIP roundtrips. Strict original pretrained-weight loading and deterministic reconstruction of the original `prediction.json` have **not** been independently completed through this research release because the original large weights were not transferable through the available Drive connector. A passing synthetic model test does not substitute for those checks.

## 7. Scientific boundaries and next experiments

- The official scorer is hidden; only the screenshot's displayed Macro-F1 and statuses are authoritative here. Ranks beyond those scores are not inferred.
- Only **80 training judgments and 10 development judgments** were available; no multi-seed or bootstrap confidence-interval study was executed in the notebook.
- Context comparisons C1/C2 changed input composition, not model architecture. The student's distillation outcome (0.3200 local, 0.3638 official) was **lower** than its supervised control (0.3234, 0.3679) despite speed gains.
- Training-data size is confounded with domain specificity in D1/D2. The Viterbi method is postprocessing, not end-to-end CRF training.
- Future experiments could use hierarchical document encoders, stronger rare-role supervision, cross-domain bootstrap, long-input handling and multi-seed evaluation. These are **proposals**, not falsely recorded as completed work.

## 8. Winning code for the course archive

Only `C2_bidirectional_context` is included in `phase3_best_model_code.zip`. Supply the original Phase-1 700M GQA checkpoint, the task-specific C2 adapter/head `best.pt`, the Phase-1 tokenizer, and the official test JSONs. The course source-only package can regenerate the `submission.zip` with exactly the winning inference construction; it **does not include** other 17 variants, data or weights.
