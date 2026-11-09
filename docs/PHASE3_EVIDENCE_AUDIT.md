# Phase 3 source-of-truth and evidence audit

**Completed:** 9 November 2026. **Source records:** executed `Capstone_3.ipynb` saved output; official uploaded competition screenshot (15 October 2026); connected Drive `phase3/experiments/metrics/` CSVs; exact `Phase_3_RR` file structure; current public project GitHub code and instruction documents.

## Inputs audited

- `Capstone_3.ipynb` full executed notebook: 16 trained classifiers plus Viterbi and specialist hybrid, 18 generated submissions, local aggregation metrics, class confusion analysis and benchmarking. A nonessential final cleanup `NameError` was repaired without retraining.
- Official screenshot: sequential submission order A1–A5, B1–B4, C1–C3, D1–D3, E1–E2, F1. **The file names uploaded to Codabench were `submission1.zip` ... `submission18.zip`; the Drive archive sources were individual `submissions/<experiment_id>/submission.zip`.** Two duplicate `Submitting` entries were visible for C2 and C3 and are not counted as distinct experiments.
- Google Drive stored experiment outputs: `all_experiments.csv`, `best_model_per_class.csv`, `top_role_confusions.csv`, `label_frequencies.csv`, `dataset_summary.csv`, `bpe_length_statistics.csv`, `viterbi_strength_search.csv`, architecture comparison, class-balancing and teacher-student benchmark records.
- Foundational Phase-1 source: unchanged four-architecture implementation and original BPE. The full 700M GQA pretrained checkpoint was not transferred through the connector; strict compatibility with large original weights remains unverified here.
- Phase-2 components: none reused by the Phase-3 notebook except the conceptual Phase-1 foundation and common project infrastructure.

## Important measured outcomes

- 80 labelled training judgments / 16,845 sentences; 10 dev / 2,142 sentences; 10 hidden-test / 2,197 sentences.
- All 13 label IDs from the official vocabulary, with zero-example / rare classes in local development.
- Dev best `C2_bidirectional_context` **0.3851196465** Macro-F1, CL **0.2788609470**, IT **0.4133530161**.
- Official best *finished* result `C2`, ID **989815**, **0.4249** Macro-F1; baseline A5 ID989804 **0.3714**, full FT B4 ID989810 **0.4005**, previous context C1 ID989811 **0.3976**, specialist hybrid D3 ID989823 **0.3957**.
- `B1_lorma_plus` `989805` failed with no score; `B2_ia3` `989807` was preparing; `F1_balanced_lora` `989828` failed despite displayed numeric value **0.3703**. Do not call these official completed evaluations. Other 15 distinct submissions had finished scores.
- C2's local confusion count is dominated by the distinction between reported positions and judicial reasoning. The two classes with no dev support cannot produce measured local F1.
- Teacher B4 selected by dev among A5 + B1–B4; 6-block student from 12-block Phase-1 decoder. Hard-label student dev0.3234 / official0.3679; distilled student dev0.3200 / official0.3638, so KD did **not** improve F1 in this trial. Inference throughput student/teacher 3823/1035 ~3.69×; separately attributed GPU memory savings unverified.

## What did not happen

- No hierarchical whole-judgment neural Transformer, bidirectional self-attention for GQA, CRF training, reinforcement learning, large external pretrained model, multiple-seed uncertainty experiment, or automatic annotation repair was implemented. C3 Viterbi is a **label-level postprocessor**, not a neural document encoder; D3 specialist hybrid is an **inference-only combination**, not a separately trained model.
- No original hidden-test labels are available; official results are user-supplied screenshots, not locally regenerated test metrics.
- No paper-level claims about understanding legal doctrine or legal correctness are supported by Macro-F1 alone.
- Large original checkpoint strict-loading and bitwise prediction equivalence were not repeated by this source-only release; original notebook reported successful loading before submitted inference.

## Integration/privacy

The public source package contains no original dataset text, model weights, checkpoints, token secrets, personal paths, full original notebooks, or private prediction arrays. The separate best-only source code needs user-provided original GQA-700M weights, C2 adapter/head state and archived BPE tokenizer; their absence is explicit rather than disguised.
