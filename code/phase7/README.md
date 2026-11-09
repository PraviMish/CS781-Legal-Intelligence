# Phase 7 — Prior Case Retrieval: research release

This directory is the complete, source-only research implementation for **all eleven meaningful submitted Phase-7 methods**, not just the winner. The final course-code ZIP is a *separate* deliverable containing the source for only the best officially scored method. The Phase-7 files are refactored from the original executed `Capstone_7(1).ipynb` and the executed `Capstone_7_Retrieval_Fix_V2.ipynb`, with the original experiment identifiers, objectives, hyperparameters, and feature-selection grids preserved.

**Official winner:** `E10_DEV_SELECTED_LEGAL_RANKER`, Codabench submission **1039660**, uploaded as `submission11.zip`, **Micro-F1@7 = 0.4619**. The stored V2 development Micro-F1@7 is **0.4469149528**. Ranking uses the legal-citation-enhanced full-document TF-IDF reciprocal rank (weight 0.6) plus LightGBM LambdaRank reciprocal rank (weight 0.4), rank constant 30.

## 1. Scientific motivation and exact task

Given a query judgment and a split-specific pool of candidate IndianKanoon cases, rank exactly seven **different** prior case IDs. Preserve the original zero-padded ID strings and never use test relevance labels. The supplied `Phase_7_PCR/{Train,Dev,Test}/` directories each contain `{split}.json` (query documents) and `{split}_candidates.json` (candidate documents). These are dictionaries keyed by case ID with `text` as sentence lists. Train and Dev query entries additionally contain `relevant_candidates` lists. The provided dataset stores candidate pools **by split**, not per query.

The official primary metric is:

`Micro-F1@7 = 2 * correct_query_candidate_pairs / (7 * number_of_queries + total_gold_pairs)`.

Micro precision = `hits/(7*N)`; micro recall = `hits/G`. There is no need to pad gold sets to seven. Prediction files are dictionaries `{query_id: [case_id1, ..., case_id7]}`. ZIP archives must contain only `prediction.json` at archive root.

## 2. Dataset profile (verified official files)

| Split | Queries | Candidate cases | Gold pairs | Median query words | Candidate p95 words |
|---|---:|---:|---:|---:|---:|
| Train | 827 (817 labeled) | 4,320 | 5,308 | 4,508 | 19,243 |
| Dev | 118 | 1,023 | 973 | 4,619 | 26,778 |
| Test | 237 | 1,727 | Hidden | 4,653 | 20,894 |

The maximum observed candidate exceeds 400,000 words. Candidate ID overlap is 436 between Train/Dev, 740 between Train/Test, and 220 between Dev/Test. All query-ID sets are disjoint. **Every query case is also an exact-text candidate in its own split**, but the query never appears in its own development/training gold set. This creates a self-retrieval trap in the historical score computations, documented as a post-submission error diagnostic rather than silently corrected.

## 3. Original neural programme (E00–E06)

The original `neural_pipeline.py` uses **only** the official legal BPE tokenizer and the Phase-1 700M-token Efficient GQA model, initialized independently for each experiment. It recreates BM25 candidate ranking, the two-chunk 192-token case encoder, pooling and 256-dim projection, triplet contrastive/ranking supervision, checkpoint-saving/resumption, GPU training and top-seven submission packaging.

| Experiment | Actual method | Trainable parameters | Dev Micro-F1@7 |
|---|---|---:|---:|
| E00_BM25 | BM25 lexical first-stage | None | 0.17343 |
| E01_PARTIAL | Final four blocks + retrieval head | 25.371M | 0.21012 |
| E02_IA3 | Multiplicative attention/MLP channel scaling | 0.234M | 0.20345 |
| E03_LORMA | Multiplicative low-rank LoRMA+ | 1.345M | 0.21568 |
| E04_FULL | All GQA blocks and retrieval head trainable | 112.877M | 0.21790 |
| E05_KD | Phase-1-initialized six-layer student; E04 teacher | 62.533M | 0.21012 |
| E06_ENSEMBLE | BM25 RRF + E04 + E03 | None additional | 0.21901 |

Original neural training covered all 817 queries with nonempty relevant lists for two epochs, but sampled only **one positive pair per query presentation**; it did not train on all 5,308 positive pairs. The original 256-case BM25 shortlist covered only **64.44%** of gold candidate pairs, placing an oracle bound of **0.51362** development F1 on any reranker restricted to it. The original representation viewed two 192-token chunks, not the entire long judgment. The original leaderboard performance is consequently limited by candidate recall, omitted legal context, and incomplete positive pair supervision, regardless of adaptation variant.

The E01–E05 checkpoint files are **external user-supplied artifacts** in Drive (`phase7/models/`), not in this public repository. The original 700M Phase-1 `.pt` and tokenizer must also be supplied for strict replay.

### Original neural training commands

```bash
pip install -r code/requirements.txt
# Requires a GPU capable of the notebook's BF16 configuration and enough memory.
PYTHONPATH=code python -m phase7.neural_pipeline \
  --project-root '/path/to/LLM CAPSTONE' \
  --data-dir '/path/to/LLM CAPSTONE/phase7/data' \
  --checkpoint '/path/to/best_up_to_700M.pt' \
  --tokenizer '/path/to/legal_bpe_32k.json'
```

It runs E00–E06 in order and saves historical-style checkpoints and submissions. Rerun the same command to resume. **Full large-weight replay was not performed in this evidence-audit environment**; compilation and CLI tests do not establish original-weight prediction equivalence.

## 4. Why V2 exists, and what exactly changed

The original E00–E06 results lagged far behind the leaderboard. The V2 study shifted the bottleneck from truncated dense representation to complete sparse-document matching. It preserved the official split-specific candidates and test schema, without external pretrained weights. The V2 source `v2_pipeline.py` is a refactoring of **the executed V2 notebook**, rather than a new, unverified architecture.

- **E07_TFIDF:** `TfidfVectorizer(lowercase=True, stop_words='english', ngram_range=(1,2), min_df=2, max_features=180000, sublinear_tf=True, norm='l2')`, fitted to candidate documents in the current split. Similarity is the normalized sparse query–candidate dot product. Full-document TF-IDF reached **0.38688** development F1, compared with **0.17343** for the earlier BM25.
- **E08_LEGAL_CITATIONS:** Parse AIR citations from candidate/query `CITATION:` headers, incoming citations from candidate `CITATOR INFO`, and candidate citation mentions in queries. Tune the two additive boosts only on Dev: the best of nine combinations is **incoming = 1.5, direct mention = 0.08**. F1 **0.42468**; it improves 16 development queries without worsening any. Incoming indicators match 105 positive pairs and 2 negative pairs; direct mentions match 44 positive and 23 negative pairs.
- **E09_LIGHTGBM:** LightGBM LambdaRank with eight features: full hashed TF-IDF cosine, unigram cosine, two citation indicators, optional archived BM25 rank, cosine-based relative rank and two log length ratios. Training includes **all 5,308 gold pairs** in 817 groups, plus lexical hard negatives, citation-linked candidates and 48 random candidates. Settings: 160 trees, learning rate 0.055, `num_leaves=15`, `min_child_samples=65`, `reg_lambda=5`, `colsample_bytree=.9`, `lambdarank_truncation_level=10`. Dev F1 **0.40689**.
- **E10_DEV_SELECTED_LEGAL_RANKER:** Reciprocal-rank fusion `0.6/(30+r_legal)+0.4/(30+r_ranker)` from Dev-only choice among 6 mixing weights. Dev F1 **0.44691** (402/973 pairs in 118 queries); official **0.4619**.

### Exact portable V2 rerun (CPU; high RAM recommended)

```bash
pip install 'numpy>=1.24' 'pandas>=2' 'scipy>=1.11' \
  'scikit-learn>=1.4,<2' 'lightgbm>=4.2,<5' 'tqdm>=4.66' 'matplotlib>=3.8' 'ipython>=8'
PYTHONPATH=code python -m phase7.v2_pipeline \
  --project-root '/path/to/LLM CAPSTONE' \
  --data-dir '/path/to/Phase7-data-root' \
  --cache-dir '/path/to/private-cache' \
  --log-dir '/path/to/private-logs' \
  --output-dir '/path/to/submissions/V2_RETRIEVAL_FIX'
```

The data directory must contain the official `Train/`, `Dev/`, and `Test/` folders with their respective query and candidate JSON files. Lowercase folder names also work. The V2 script creates E07, E08, E09 and E10 prediction JSON/ZIP archives. If the *original saved* `lambda_ranker_all_positives.pkl` and `lambda_ranker_config.json` are placed in `--log-dir`, they are reused; otherwise the model is trained from the official data. Caches are keyed by the historical V2 version and split sizes and should be cleared if the underlying dataset is changed. **The user must supply the original model artifact for strict model-weight equivalence**; retraining with different library versions is logically reproducible but may not generate byte-identical rankings.

### Validate any archived submission

```python
from phase7.validation import load_split, verify_submission
queries, candidates = load_split('/path/to/Phase7-data-root', 'test')
predictions = verify_submission('/path/to/submission.zip', queries, candidates)
assert len(predictions) == len(queries)
```

### Tests and figures

```bash
PYTHONPATH=code python -m pytest -q code/tests/test_phase7.py
PYTHONPATH=code python -m phase7.analysis.figures --out report/figures/phase7
cd report && pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The vector figure PDFs are generated from `experiments/official_results.csv`; the development citation/self-exclusion diagnostics are fixed empirical records, not re-estimated from unavailable test labels.

## 5. Official submissions and evidence chain

See [official_results.csv](experiments/official_results.csv) for eleven records, IDs, exactly four decimal place official primary scores, and the mapping confidence annotation. **E10, submission11.zip, 1039660, is the official winner at 0.4619**. In the supplied screenshot, the largest visible competing score was 0.5162; no final placement is asserted.

Original submissions are stored as `E00_BM25`, `E01_PARTIAL`, ... `E06_ENSEMBLE` in Drive. The archived original notebook creates `E00` first followed by `E01`–`E06`; their relation to the *renamed* `submission1.zip`–`submission7.zip` is therefore **provisional** without an uploaded-byte comparison. The separate user statement that submission1 corresponds to “E01” uses a shifted numbering convention and must not be silently reconciled. In contrast, V2 generated E07–E10 in source order and the four official scores were reported in exactly that order, so their archive-to-filename mapping is well supported.

No raw legal judgments, private leaderboard labels, user authentication secrets, large Phase-1 checkpoints, source-only prediction shortcuts or historical test ZIPs are tracked in this research-code package.

## 6. Error analysis and limitations

- **Case masking:** Every query is present verbatim in its candidate pool. The archived E10 test submission returned its own query ID 85 times out of 237. The original E07 TF-IDF top-seven included self in all 118 Dev queries. A *post-submission* development ablation masking the self-case before ranking improved TF-IDF 0.38688→0.42468 and citation TF-IDF 0.42468→0.45247. These are **not official submissions**.
- **Hard queries:** Dev query `0001673100` has four gold cases and neither E07 nor E08 retrieved any at top seven; `0001416283` improved from one to six of eleven; `0000092507` improved three to seven of twelve. Dev query `0181443842` has 92 gold cases, so no top-seven list can recover more than seven of them.
- **Temporal citation availability:** `CITATOR INFO` can include **retrospective citations from cases decided after the case itself**. It is public supplied text and not the gold relevance label, but may be unavailable in prospective real-world prior-case search. The observed performance should not be equated with prospective temporal retrieval without matching metadata restrictions.
- **Selection bias:** Citation boosts and RRF weight were tuned on one 118-query Dev set. Scores are not confidence intervals or repeated-seed evidence.
- **Model transfer limitations:** Truncated two-chunk embeddings and a 64.44%-recall shortlist constrain the original neural methods. Results do not show that no legal LM could achieve stronger retrieval with richer representation.
- **Verification scope:** Existing notebook executions demonstrate trained models and valid ZIP creation. The refactored source must still be strict-loaded and compared with original checkpoint output on the user's GPU to certify exact neural prediction replay. The saved LightGBM pickle, tokenizer and Phase-1 weights remain external.

## 7. Reproducing figures and the final best-model code package

`python scripts/package_phase7_winner.py --output phase7_winner_source.zip` packages only E10 source and instructions. It **does not** include private predictions, model weights, or unrelated experiments. The public research repository retains all E00–E10 methods.

The unrestricted scientific report is `report/phase7.tex`, included after Phase 6 in the existing master report by the integration script. The separate official ten-page course report is **not** created or substituted here.
