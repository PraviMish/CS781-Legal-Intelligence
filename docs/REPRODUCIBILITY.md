# Reproducibility and data provenance

## System requirements
Python >=3.10; PyTorch, tokenizers, ijson, numpy, pandas, matplotlib, sacrebleu, rouge-score, pytest. For full 100M/700M token runs use adequate accelerator memory and disk space. CPU smoke tests are feasible, but full training may be prohibitively slow without GPU acceleration.

## Official data only
Obtain files from `Exploration-Lab/CS-781-Capstone`, `Phase_1_Pretraining/Train/` and `/Test/`. No dataset, weights, authentication tokens or raw output is bundled in Git. Keep artifact outputs outside the checkout, and use absolute or relative CLI paths appropriate to your machine.

## Work stages
1. `python -m phase1.prepare`: build/load the official 32k byte-level BPE tokenizer, create initial ~500M-token cache with held-out validation, generate pseudo-development examples and write source-weight JSON.
2. `python -m phase1.train`: train each of the four architectures to 100M consumed tokens using the same saved source weights, context and optimizer policy.
3. `python -m phase1.evaluate`: evaluate the selected checkpoint on held-out pseudo-development samples. Use the archived tokenizer and pseudo-development set for comparisons with the notebook. The reported continuation validation losses and the perplexities from separate checkpoint evaluations came from different evaluation runs.
4. `python -m phase1.submit`: generate exactly 128 words for every official test ID and verify archive contents.
5. `python -m phase1.prepare --mode full`: tokenize the complete legal corpus separately. Resume only the winner's trusted saved checkpoint and continue to 300M, 500M and 700M, storing permanent frozen snapshots. These continued models have lower private test scores than the initial winner.
6. From the repository root, run `python code/phase1/analysis/figures.py` to regenerate the figures from checked-in CSV data. From within `report/`, run `pdflatex main.tex` twice to build the PDF and resolve cross-references.

## Known incompleteness
This reconstruction underwent syntax compilation and architecture-size checking. Exact checkpoint compatibility and prediction equivalence cannot be certified without loading original large checkpoints and archived outputs. Course hidden references cannot be evaluated locally. The official notebook contains a two-stage data-cache history and different decoder strategies; reproducibility requires keeping the correct combination of checkpoint and decoder per submitted variant.

**Historical run inputs.** The training CLI defaults to seed 42 and accepts the original seed privately with `--seed`. Exact training replay also requires the original tokenizer, token caches, validation split and runtime settings. Sampling during inference derives its seed from the prompt, with a configuration name included for the later sweep. Identical sampling can also depend on the runtime and device.

## Phase 2 legal named entity recognition

Use the official dataset's `Phase_2_LNER/Train` and `Phase_2_LNER/Test` files. Pass the archived Phase-1 tokenizer and the original snapshot identified for each adaptation. The held-out runs use the archived `validation_split.json` and a checkpoint selected by strict validation macro-F1. The winning runs use all 70 labeled judgments, four fixed epochs, and the three corresponding `final.pt` files. See the commands and artifact mapping in the [Phase-2 README](../code/phase2/README.md).

The Phase-2 CLI reads the archived prediction JSON or ZIP for CPU-only ensemble replay. Run `python -m phase2 --help` for the training, evaluation, inference, voting, and validation commands. Keep raw files, caches, and weights outside the checkout.

Regenerate the three Phase-2 figures with `python code/phase2/analysis/figures.py` from the repository root. Compile `main.tex` from `report/` with Tectonic, or with two `pdflatex` passes.

## Phase 3 rhetorical-role prediction
Use official Hugging Face dataset `Exploration-Lab/CS-781-Capstone`,
folder `Phase_3_RR`; preserve the original legal BPE tokenizer and archived
Phase-1 100M/700M checkpoint versions. The notebook's C2 official winner is a
causal GQA classifier with previous and next sentence tokens placed ahead of
the target sentence, not a bidirectional-attention model. See
[Phase-3 README](../code/phase3/README.md) for commands to prepare, train,
resume, evaluate and generate one label per original test sentence.
The original large checkpoints cannot be published or independently strict-
loaded in this source-only release; full prediction equivalence is pending.

## Phase 4 court judgment prediction

Use the official `Phase_4_CJPE` files at the pinned revision recorded in the
[Phase-4 README](../code/phase4/README.md). Preserve the Phase 1 tokenizer,
case-level split order, and the exact snapshot for each adaptation. The
research CLI covers preparation, eight training variants, threshold selection,
E09/E10 probability blending, and submission validation. The artifact replay
checks all ten saved ZIPs against the official test IDs without model weights.
The separate source-only E01 inference package requires the original Phase 4
classifier and uses its fixed 0.51 threshold. See the phase README for commands
and the original-weight verification limits.

## Phase 5 legal statute identification

Use the pinned official Phase_5_LSI files, archived Phase 1 tokenizer and
matching Phase 1 foundation. The six-run CLI covers grouped preparation,
training and restoration, distillation, development calibration, inference
and ZIP validation. The standalone E02 archive requires the original Phase 5
epoch-3 classifier and its matching thresholds. See the [Phase 5 README](../code/phase5/README.md)
and [evidence and replay limits](PHASE5_RECOVERY.md).

## Phase 6 legal abstractive summarization

Use the pinned official Phase_6_Summ dataset, original Phase 1 tokenizer and
archived validation IDs. The Phase 6 CLI supports all three protocols. The
separate E05 source package requires the archived V3 step 7,750 model and
uses the raw hierarchical decoder. See the [Phase 6 README](../code/phase6/README.md)
for commands and the [evidence audit](PHASE6_EVIDENCE.md) for replay limits.
