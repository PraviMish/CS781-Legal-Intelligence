# Agent guidelines for this research repository

1. Read `docs/PROJECT_CONTEXT.md`, `docs/MASTER_PLAN.md`, `docs/PHASE_STATUS.md` and the relevant phase README before modifying code.
2. Preserve real algorithms and original checkpoint interface. Do not silently alter attention masking, RoPE, serialization keys, token IDs, splitting logic, objectives or generation configuration.
3. The public repo records all meaningful experiments. The final course ZIP contains only the best-performing model code for each phase; never confuse these deliverables.
4. Write simple portable Python, clear docstrings, no unnecessary framework. Avoid Colab-only imports, absolute paths or hidden runtime restoration cells.
5. Do not commit personal details, raw or confidential data, tokens, paths, model weights, caches or private access credentials. Never use external training corpora for this project.
6. Record actual experiment metrics with source and evaluation split. Do not substitute local ROUGE for official BERTScore, nor select checkpoints solely by validation loss.
7. Every figure in `report/main.tex` must be generated, named, versioned, and verified by successful LaTeX compilation.
8. Run relevant tests and report exactly which tests ran and which could not run; don't claim checkpoint compatibility without loading actual saved weights.
9. Make small meaningful commits, keep a clean history and update `docs/PHASE_STATUS.md` after each milestone.
