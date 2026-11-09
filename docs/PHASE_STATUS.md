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

## Future phases
Phases 2--7 to be added individually after reviewing the phase's executed notebook, Codabench score sheets and saved artifacts. Do not invent or preemptively publish their findings.

## Original artifacts required for replay

Checkpoint weights, token caches, evaluation examples and archived predictions are kept outside the repository. The Phase-1 audit could locate the winning checkpoint but could not retrieve the approximately 451 MB file through the available Drive connection. Synthetic checkpoint round-trip tests cover the repository's own checkpoint writer and reader.

**Winning checkpoint identity:** `models/efficient_gqa/milestones/best_100M.pt` should be supplied by the original Drive artifacts; later `best.pt` was overwritten during continuation. Validate token metadata before packaging predictions.
