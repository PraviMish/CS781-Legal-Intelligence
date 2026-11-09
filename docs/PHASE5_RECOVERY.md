# Phase 5 evidence and verification

The initial recovery commit bf5862d preserved the six-model development comparison before Phase 6 was released. Its values and order remain in [the original comparison CSV](PHASE5_LOCAL_RESULTS.csv). The Phase 5 implementation uses the later Phase 6 commit e678311 as its base.

| Evidence | What it establishes |
| --- | --- |
| Original Capstone_5.ipynb and executed Capstone_5_Colab_Resume_From_Kaggle.ipynb | Original methodology and 28 executed cells covering the continuation, E06 training, calibration and six submissions |
| Archived configurations, history and development arrays | Six architectures/adaptations, selected epochs, E01 teacher, and exact development replay |
| Archived label-frequency, error and checkpoint audit tables | Training supports, E02 per-statute diagnostics and historical strict-load checks for the four Phase 1 initializations |
| BPE count caches and pinned development/test copies | Length statistics, coverage and length-stratified E02 results |
| Six submission ZIPs and Codabench screenshot dated 28 October 2026 | Exact filenames, IDs, finished statuses, scores and official test-ID coverage |

E02 is the official winner at **0.3875 macro-F1** (submission **1014549**). It uses the Phase 5 best.pt selected at epoch 3 (one-based), trained from the Phase 1 700M GQA milestone, and the matching 100-element thresholds.npy. The history records internal macro-F1 0.2202979354 at the fixed 0.30 checkpoint-selection threshold. E01 supplied the E06 teacher and E06 finished second officially at 0.3842.

## Replayed evidence

- The development and test copies reconstruct the exact Git LFS pointer blob IDs from the pinned official revision. Full training statistics and overlap counts come from the executed notebook and the saved label frequency table. The unauthenticated Hugging Face download returned HTTP 401. The original 299 MB training JSON exceeds the connector transfer limit, so the grouped split was implemented and tested on fixtures rather than regenerated from the full training file.
- All six archived ZIPs passed CRC, one-root-entry, unique-label and ordered coverage checks against all 13,019 official test IDs. E02 and E06 agree on 1,110 exact sets.
- The six development logit arrays reproduce the six archived threshold arrays bit for bit and reproduce the archived macro-F1, micro-F1, exact-set accuracy and mean cardinality. E02 per-statute and length-group counts are derived from those arrays and the verified development copy.
- The original 451,081,783-byte E02 checkpoint exceeds the Drive connector's 268,435,456-byte limit. Original Phase 5 weight loading, weight-to-submission replay and full GPU retraining remain open. The notebook's Phase 1 initialization checks are historical evidence. The source port and standalone E02 package use synthetic checks for model loading.
- The original Phase 1 BPE reproduces token counts and packed window lengths for all 10,181 development and 13,019 test cases.
- The 64-test repository suite passed with the archived Phase 3 standalone model available for its optional equivalence test. The extracted E02-only package strictly loaded full-size synthetic weights and generated a valid one-case ZIP.
- The 42-page master report places Phase 5 on pages 31–36. Phase 6 starts on page 37. Phase 6 code, data, figures, tests and LaTeX match the base commit byte for byte. The report build retains its existing Phase 3 typesetting warnings.
- Only aggregate metrics, checksums and source are published. The original notebooks, case records, saved logits, thresholds, prediction files and weights remain private.

[Phase 5 methods](../code/phase5/README.md) · [Replayed aggregates](../code/phase5/experiments/archive_verification.json) · [Official ledger](../code/phase5/experiments/official_results.csv)
