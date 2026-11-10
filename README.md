# CS781 Legal Intelligence

This repository contains my seven-phase legal NLP capstone project. I trained a compact legal language model from scratch on the official corpus, then studied its adaptation to six downstream tasks. The repository includes the implementations, experiment configurations, results, tests, and full research report.

## Results

| Phase | Task | Best-performing approach | Official primary score |
|---|---|---|---|
| 1 | Language modelling and continuation | Efficient GQA, 100M training tokens | BERTScore F1: **0.8348** |
| 2 | Legal named entity recognition | Three-model span-level majority ensemble | Strict Macro-F1: **0.3577** |
| 3 | Rhetorical role prediction | Context-aware GQA with LoRA | Macro-F1: **0.4249** |
| 4 | Court judgment prediction | Full GQA fine-tuning | Macro-F1: **0.7393** |
| 5 | Legal statute identification | Full GQA fine-tuning | Macro-F1: **0.3875** |
| 6 | Abstractive legal summarization | V3 GQA with LoRA | BERTScore F1: **0.8459** |
| 7 | Prior-case retrieval | Citation-aware TF-IDF and LambdaRank fusion | Micro-F1@7: **0.4619** |

Scores use task-specific benchmarks and are not directly comparable. Each phase's `experiments/` folder records additional methods and results.

## Structure

```text
code/
  common/                  Shared utilities
  phase1/ ... phase7/      Models, training, inference and experiment records
  tests/                   Automated tests
  requirements.txt         Dependencies
report/
  main.tex                 Complete seven-phase research report
  phase3.tex ... phase7.tex
  figures/                 Original PDF figures used by the report
scripts/                   Winning-model source packaging
```

The `README.md` inside each phase explains the data, architectures, experimental comparisons and commands to reproduce the corresponding training and submission pipeline.

## Setup

Use Python 3.10 or newer.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r code/requirements.txt
```

For CLI entry points, make the `code/` folder importable:

```bash
# Linux/macOS
export PYTHONPATH=code
python -m phase1.submit --help
# Windows PowerShell: $env:PYTHONPATH="code"
```

Run the tests from the repository root:

```bash
PYTHONPATH=code python -m pytest -q code/tests
```

The dataset is available from [Exploration-Lab/CS-781-Capstone](https://huggingface.co/datasets/Exploration-Lab/CS-781-Capstone). Dataset files, model checkpoints, tokenizers and intermediate caches are not included. Use the original corresponding trained artifacts when reproducing inference; only load checkpoints from trusted sources.

## Full report

The complete experimental study is in [`report/main.tex`](report/main.tex). To build its PDF:

```bash
cd report
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

This repository preserves the complete research and experiments. The official course submission containing only the selected winning systems is packaged separately.
