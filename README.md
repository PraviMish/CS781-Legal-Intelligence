# Self-Hosted Legal Intelligence — CS781 Capstone

A reproducible, official-data-only research implementation of a compact legal-language foundation and its adaptation to practical legal NLP tasks. The code prioritizes clarity, honest experimental provenance, and the ability to reproduce submission pipelines without publishing private data or large model weights.

**Repository purpose:** Complete code for **all meaningful submissions and experiments** across the seven capstone phases. The separate course ZIP deliverable contains **only the winning model's executable code per phase**. This distinction is intentional and follows the course TA clarification.

## Project tasks

| Phase | Task | Primary metric |
|---|---|---|
| 1 | Legal language model pretraining / 128-word continuation | BERTScore F1 |
| 2 | Named entity recognition with character offsets | strict macro-F1 |
| 3 | Rhetorical role classification of legal sentences | macro-F1 |
| 4 | Binary court judgment prediction | macro-F1 |
| 5 | Multi-label legal statute identification | macro-F1 |
| 6 | Abstractive legal summarization | BERTScore F1 |
| 7 | Prior-case retrieval, seven ranked case IDs per query | micro-F1@7 |

**Current code-release scope:** The reconstructed Phase-1 package and master report's Phase-1 chapter. Later phases will be added as validated, using the same conventions.

## Phase-1 finding

Four architecturally distinct models were built from scratch with a new 32,000-token byte-level BPE tokenizer trained exclusively on official legal text. Of the seven valid Phase-1 Codabench submissions, **Efficient GQA at ~100M consumed tokens** achieved the best official BERTScore F1, **0.83479506**. Further training through 300M, 500M and 700M tokens lowered held-out language-model loss but did not improve hidden-reference continuation score. See [Phase-1 README](code/phase1/README.md) and [comprehensive report source](report/main.tex).

## Contents

- `code/phase1/`: four original architectures, tokenizer/data preparation, training, evaluation, generation, official submission packaging, configuration, and experimental records
- `code/common/`: shared atomic checkpointing, artifact and RNG helpers
- `code/tests/`: software smoke tests
- `code/requirements.txt`: Python dependencies
- `report/main.tex`: continuously extended LaTeX research report
- `report/figures/phase1/`: original vector figures plus PNG previews generated from experimental CSV data
- `docs/`: permanent research instructions, phase status, experiment provenance, and reproducibility notes

## Setup and safe usage

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
pip install -r code/requirements.txt
# From the repository root, make the code package importable:
# Linux/macOS: export PYTHONPATH=code
# Windows PowerShell: $env:PYTHONPATH='code'
```

The official dataset is separately available on [Hugging Face](https://huggingface.co/datasets/Exploration-Lab/CS-781-Capstone). Follow access and competition terms. It is **not** included in Git. See [reproduction instructions](docs/REPRODUCIBILITY.md) for dataset preparation, training and submission commands.

**Security:** Only load trusted local PyTorch checkpoints. Older notebook checkpoint formats include Python optimizer/RNG objects requiring `weights_only=False`. Never deserialize arbitrary user-supplied `.pt` files. No private data, API tokens, direct local paths, user-specific hardware records or checkpoints should be committed.

## Report and final deliverable

The comprehensive report is maintained from one `report/main.tex` entry point. The official course report will later be condensed into the prescribed ten-page template and include the required LLM disclosure. The final Hello IITK archive will be produced from tested, minimal best-model code plus the official report PDF; it is **not** this full research repository.

## Reproducibility status

All 14 repository tests passed, including construction and forward-loss checks for the four original parameter totals. The archived tokenizer loaded successfully. The seven official results were checked against the score sheets. Strict loading of the original checkpoints and exact replay of their archived predictions are pending. See [phase status](docs/PHASE_STATUS.md).

**Winner checkpoint guard:** do not use the current generic `efficient_gqa/best.pt` after continued pretraining; the initial winning weights were frozen under `efficient_gqa/milestones/best_100M.pt`. See Phase-1 instructions.
