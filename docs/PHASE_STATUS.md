# Phase status and release gates

## Phase 1 research release
- [x] Full seven-submission Codabench primary/secondary score registry with provenance
- [x] Four architecture classes extracted and param counts verified
- [x] Source-data and tokenizer scripts extracted and made portable
- [x] Common checkpoint/save/resume utilities
- [x] Training CLI and configurable continuation schedule
- [x] Historical baseline and later tuned decoding modes
- [x] 128-word official prediction + ZIP validation CLI
- [x] Research data CSV files and original figures
- [x] Comprehensive Phase-1 LaTeX chapter
- [x] Basic public project and Phase-1 READMEs
- [x] Verify all 21 Codabench metric values against the seven official score sheets and check the decoding sweep and continuation loss series against the executed notebook
- [x] Pass all 14 tests, load the archived tokenizer, start all five Phase-1 command-line entry points, and check the source-only winner ZIP
- [x] Regenerate all five figure pairs and build the 11-page master report with its cross-references resolved
- [ ] Verify strict loading and logits against original saved `.pt` checkpoints
- [ ] Verify regenerated predictions against archived original `prediction.json` examples
- [ ] Replay training from the archived tokenizer, cache and runtime settings

## Phase 2 research release

- [x] Audit the executed notebook, Drive run artifacts, and seven Codabench entries
- [x] Reconstruct all 11 trained variants, five original voting rules, and the full-data winner
- [x] Regenerate the archived split and all 105 token records from the official files
- [x] Validate all 12 submission ZIPs and replay both majority ensembles exactly from the archived member outputs
- [x] Recompute the five held-out voting results, false-positive counts, and paired bootstrap
- [x] Load and check the four original small head or adapter checkpoints with the reconstructed model shapes
- [x] Pass all 22 repository tests, including the 14 Phase-1 tests
- [x] Record aggregate results, generate three figure pairs, and extend the master report
- [x] Compile the 16-page master report and inspect the rendered layout
- [ ] Load original Phase-1 backbones and full Phase-2 checkpoints and compare regenerated member predictions
- [ ] Replay the original training runs with the archived weights and GPU numerical settings

The Drive connection rejected the larger original checkpoints above its 256 MiB transfer limit. The head and adapter checks used reconstructed backbones. The ensemble replay starts from archived predictions.

## Future phases
Phases 5--7 to be added individually after reviewing each phase's executed notebook, Codabench score sheets, and saved artifacts.

## Original artifacts required for replay

Checkpoint weights, token caches, evaluation examples and archived predictions are kept outside the repository. The Phase-1 audit could locate the winning checkpoint but could not retrieve the approximately 451 MB file through the available Drive connection. Synthetic checkpoint round-trip tests cover the repository's own checkpoint writer and reader.

**Winning checkpoint identity:** `models/efficient_gqa/milestones/best_100M.pt` should be supplied by the original Drive artifacts; later `best.pt` was overwritten during continuation. Validate token metadata before packaging predictions.

## Phase 3 research release (9 November 2026)
- [x] 16 trained variants and 2 inference-only strategies reconstructed from notebook
- [x] Local and official test metrics recorded separately; statuses captured
- [x] 13-role data parsing, context packing, adapter definitions, resume-aware CLI
- [x] Source-only tests and generated scientific charts and results table
- [x] Best official C2 at Macro-F1 0.4249 (submission 989815)
- [x] Comprehensive Phase-3 LaTeX section and source-only best-model package
- [ ] Strict loading and deterministic replay of original full pretrained GQA + C2 adapter
- [x] Complete master report compile with all original Phase-1/2 figures
- [ ] Original GPU training reruns / repeated-seed confidence analysis

See [Phase-3 evidence audit](PHASE3_EVIDENCE_AUDIT.md) and
[verification notes](PHASE3_VERIFICATION.md).

## Phase 4 research release

- [x] Verify the executed notebook, saved run artifacts, and official scores supplied by the user
- [x] Port eight trained variants and two blends with original configurations and checkpoint schemas
- [x] Validate all ten archived ZIPs against 1,517 official test IDs and replay all ten label sets
- [x] Recover both ensemble selections and recompute local aggregate metrics
- [x] Load the original E02, E03, E07 and E08 checkpoint components and run available inference checks
- [x] Generate three figure pairs and the results table, then compile the existing master report
- [x] Prepare a separate source-only E01 inference package
- [ ] Replay E01/E04/E05/E06 from original weights and rerun original GPU training

[Phase-4 methods, records and verification](../code/phase4/README.md)
