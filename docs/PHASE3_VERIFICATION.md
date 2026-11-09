# Phase 3 verification ledger

## Verified using the packaged source

- Python module syntax compilation: **pass** for Phase-3 research code, figure generators, and best-only source package.
- Offline source tests: **14 passed in the combined development workspace** (13 runnable repository tests plus one conditional equivalence test requiring the adjacent separate best-only source; the conditional test is skipped when that package is not adjacent), including 13-class macro-F1 with absent labels, correct JSON train/test parsers and label vocabulary, causal-context ordering, BPE length alignment, attention adapter identity/gradients, small GQA classifier forward, trainable checkpoint strict-key restore, Viterbi, and valid/invalid submission archive handling.
- Figures: **seven generated vector PDF figures** from archived numeric CSVs; LaTeX result table generated from records, no invented observations.
- Standalone scientific Phase-3 report compiled with `pdflatex` **twice** and rendered/inspected; 10 pages.
- Official submission mapping from user screenshot, with precise failed, preparing and duplicate-upload statuses.
- Winning model source-only path isolated from nonwinning experimental methods.

## Remaining evidence/access limitations

- Original 100M/700M Phase-1 checkpoint tensors and original Phase-3 adapter tensors could not be fetched into the local build environment through the connected Drive transfer route (large files / connector transfer constraints). No claim of **newly verified** strict original checkpoint loading or deterministic equivalence to the saved C2 `prediction.json` is made.
- Without the existing repository's complete tracked Phase1/Phase2 PDF assets in this local build, the current **entire** `report/main.tex` could not be compiled here. The self-contained `report/phase3_standalone.tex` harness is compiled; `integrate_phase3.py` inserts the same verified `\input{phase3.tex}` into the preserved master when applied to a full checkout.
- **GitHub publication blocked:** the authenticated GitHub connector can read the repository but both the Contents API write and Git blob creation returned HTTP 403 `Resource not accessible by integration`. No files or commits were published by this session. A small, idempotent integration script and complete source bundle are provided instead.

## End-to-end command examples

From the repository root, after applying the bundle:

```bash
PYTHONPATH=code python -m pytest -q code/tests/test_phase3.py
PYTHONPATH=code python -m phase3 --help
python code/phase3/analysis/figures.py
python code/phase3/analysis/latex_tables.py
cd report && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex
```

To establish original-weights equivalence, supply the original foundation and C2 best checkpoint locally, produce `prediction.json` using the best-model-only package, and compare the resulting ordered per-document labels against the archived C2 prediction file. **Until that is done, equivalence is pending.**

## Repository recovery

The archived integration was applied over the existing Phase-2 checkout. The combined repository suite passed 35 tests with one skipped adjacent-package test. The complete Phase 1--3 master report compiled with Tectonic. The original checkpoint and prediction equivalence gates above remain open.
