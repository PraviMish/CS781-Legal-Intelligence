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
