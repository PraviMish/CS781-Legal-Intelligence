# Master plan: complete GitHub research repository + scientific reports

## Source-of-truth flow
Original executed notebook + saved artifacts + official Codabench score reports -> evidence audit -> reusable code & verified experiment configurations -> automated tests -> detailed phase README & registry -> LaTeX master report and generated figures -> Git commit/release -> derive best-model-only course code.

## Repository policy
- Include all experiment algorithms and submitted variants, not just winners.
- Refactor shared tools and parameterize variations; do not duplicate near-identical scripts across checkpoints.
- Protect strict checkpoint compatibility: state names, tensor shapes, tokenizer IDs, RoPE/attention masking, loss definitions, preprocessing decisions, decoding rules.
- Training automatically resumes from a valid checkpoint and works with user-supplied dataset/tokenizer/output directories; no Colab-specific restore cells.
- Provide reproducible commands, environment setup, source metadata, checkable tests, and explicit resource expectations.
- Keep checkpoint files, raw official data, tokenized binaries, private submissions, credentials and identifying paths out of Git.
- Use root README for global overview and deep per-phase READMEs for complete methodology, variants, results and limitations.

## Report policy
- Build a comprehensive, scientifically honest LaTeX master report with consistent task-specific sections: goals, data statistics, preprocessing, architectures, training, experiment design, complete results, interpretation, errors, limitations, conclusions and viva-grade explanations.
- All tables/figures must come from actual experimental evidence, and figure filenames must match compiled LaTeX source.
- Maintain `report/main.tex` throughout project; later create the separate official ten-page report using `Report_LLM.zip` template and the required LLM declaration.
- Limit the best-model course ZIP to the code for the best official primary-metric submission of each phase, with necessary shared dependencies and run instructions. This differs from the complete public GitHub research repo.

## Phase acceptance checks
1. All meaningful variants represented with configuration.
2. Clean, importable, human-readable code and documented CLI.
3. Syntax and unit tests pass; actual saved checkpoint strict-load and inference equivalence checked when weights accessible.
4. Independent evaluation and submission validation.
5. Official metrics recorded with provenance; no unsupported causal claims.
6. Original research figures generated from CSV and embedded with matching paths.
7. `main.tex` compiles, captions/references valid, no missing figures.
8. README, context, registry and phase status updated; commit and push.

## Agent instruction
Do not invent final test labels, leaderboard ranks or missing experiments. If credentials or repository creation require user action, request exactly that step. Avoid silent refactoring differences; prefer a small verified patch over broad restyling. Never upload the complete original notebook or its outputs without reviewing secrets/paths.
