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

The research release covers Phases 1–7.

## Phase-1 finding

Four architecturally distinct models were built from scratch with a new 32,000-token byte-level BPE tokenizer trained exclusively on official legal text. Of the seven valid Phase-1 Codabench submissions, **Efficient GQA at ~100M consumed tokens** achieved the best official BERTScore F1, **0.83479506**. Further training through 300M, 500M and 700M tokens lowered held-out language-model loss but did not improve hidden-reference continuation score. See [Phase-1 README](code/phase1/README.md) and [comprehensive report source](report/main.tex).

## Phase-2 finding

The full-data majority ensemble achieved **0.3577** strict macro-F1 on Codabench (submission **971881**), compared with **0.3342** for the original majority ensemble. It votes over exact character spans predicted by GQA, a decoder-only model, and a distilled GQA student. The local study includes a frozen probe, full fine-tuning, LoRA and LoRMA+, a supervised student control, and five voting rules. [Phase-2 README](code/phase2/README.md)

<!-- PHASE3_RESEARCH_RELEASE -->
## Phase-3 finding: rhetorical roles

**C2: bidirectional neighboring-sentence context + Phase-1 Efficient GQA LoRA**
won the recorded Phase-3 Codabench entries: **Macro-F1 0.4249**, submission
**989815**. The 16 training runs and two inference-only variants are documented
in [the Phase-3 README](code/phase3/README.md), with official-status provenance
and independently runnable source. Class-wise metrics and generated figures
appear in the newly added section of [the master scientific report](report/main.tex).
The separate best-model-only source code is distributed apart from the public
all-experiment research repository.


## Phase-4 finding

Efficient GQA full fine-tuning (E01) achieved **0.7393** official macro-F1 on
binary court judgment prediction, ahead of the distilled-student blend
(0.7369) and Prefix-LM blend (0.7353). E09 led on clean development.
[Phase-4 experiments and reproduction](code/phase4/README.md)

## Phase 5 | Legal statute identification

Efficient GQA full fine-tuning (E02) scored **0.3875** official macro-F1 (submission **1014549**), ahead of the six-layer distilled GQA (E06, **0.3842**). The study compares six configurations and replays their saved development calibration. [Phase 5 experiments and reproduction](code/phase5/README.md)

## Phase 6 | Legal summarization

V3 Efficient GQA + LoRA (E05) achieved the best recorded official BERTScore F1, **0.8459** (submission **1026556**). The gold-only student E07 led on local ROUGE. [Experiments and reproduction](code/phase6/README.md)

## Contents

- `code/phase1/`: four original architectures, tokenizer/data preparation, training, evaluation, generation, official submission packaging, configuration, and experimental records
- `code/phase2/`: exact-span legal NER, frozen and fully tuned models, LoRA and LoRMA+ adapters, knowledge distillation, voting, and official submission packaging
- `code/phase3/`: rhetorical roles, context packing, adaptation, distillation, and structured decoding
- `code/phase4/`: court judgment prediction, eight trained runs, two blends, and saved-output replay
- `code/phase5/`: six legal statute classifiers, grouped split, per-statute thresholds, and submission validation
- `code/phase6/`: legal summarization across three revisions, seven official submissions, passage alignment, and two distillation objectives
- `code/phase7/`: prior-case retrieval, neural adaptations, lexical ranking and the citation-aware LambdaRank fusion
- `code/common/`: shared atomic checkpointing, artifact and RNG helpers
- `code/tests/`: software smoke tests
- `code/requirements.txt`: Python dependencies
- `report/main.tex`: continuously extended LaTeX research report
- `report/figures/`: original vector figures plus PNG previews generated from experimental CSV data
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

The Phase-2 audit replayed both ensemble prediction dictionaries exactly from the archived member outputs and reproduced the held-out split and token cache. Phase 4 reproduced all ten archived submissions from saved outputs. See the per-phase verification records for completed checks and remaining checkpoint replay work. [Phase status](docs/PHASE_STATUS.md)

**Winner checkpoint guard:** do not use the current generic `efficient_gqa/best.pt` after continued pretraining; the initial winning weights were frozen under `efficient_gqa/milestones/best_100M.pt`. See Phase-1 instructions.

## Phase 7 research release

[Phase 7 code and research README](code/phase7/README.md) |
[Phase 7 scientific evidence audit](docs/PHASE7_EVIDENCE.md) |
[Comprehensive Phase 7 LaTeX section](report/phase7.tex)

The original Phase-1 Efficient GQA was adapted for prior-case retrieval using
partial fine-tuning, IA³, LoRMA+, full fine-tuning and knowledge distillation.
The first retrieval system was limited by a 256-case BM25 shortlist and
truncated long documents. A subsequent full-document TF-IDF, legal citation
feature and supervised LambdaRank pipeline produced the best **official**
Phase-7 score: **0.4619 Micro-F1@7**, submission **1039660 (submission11.zip)**.
The repository records all E00–E10 methods; only E10 belongs in the separate
best-model-only source ZIP. Public source does not include private data,
original 700M checkpoint weights or saved test predictions.
