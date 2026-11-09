# Phase 7 evidence audit and source-to-score provenance

**Audit date:** 9 November 2026. **Official dataset:** Exploration-Lab/CS-781-Capstone, Phase_7_PCR. **Source:** executed `Capstone_7(1).ipynb`, executed Drive `Capstone_7_Retrieval_Fix_V2.ipynb`, exact source reproduction notebook, historical Drive logs/manifest/model listing and the two Codabench screenshots in this project. No test relevance labels were consulted.

## Original neural experiments

The original executed notebook contains 18 completed code cells, including strict Phase-1 model restoration, preflight checks of all adapters, five completed GPU trainings, development evaluations, a saved two-model ensemble, all seven test archives, and a final archive-format validator. The archived Phase-1 700M checkpoint has original `model_state` weights and a 32k legal BPE. E01/E02/E03/E04/E05/E06 methods are described in the Phase-7 README. Original BM25 Dev F1 = 0.173429683, recall within top-256 candidates = 0.644398767, shortlist oracle Dev F1 = 0.513618677; full FT Dev F1 = 0.217898833; original ensemble Dev F1 = 0.219010561. All 817 labeled training queries were processed, but each query visit used only one selected gold candidate.

## V2 recovery and actual model training

The V2 Drive notebook's cells have execution counts 1–10 and output evidence. The saved LightGBM config records 817 groups, 5,308 positive pairs, 200 full-hash hard candidates, 100 unigram candidates, 48 random negatives, 160 trees and seed 230787. V2 `FINAL_MANIFEST.json` records Dev metrics E07=.3868816009 (348 hits), E08=.4246803780 (382 hits), E09=.4068927182 (366 hits), E10=.4469149528 (402 hits); total gold 973. The saved `lambda_rank_dev_results.json` chooses `legal_lexical_weight=0.6`; the saved lexical grid chooses `incoming_weight=1.5`, `mention_weight=.08`.

## Codabench official submissions

The screenshot confirms 11 completed submissions:

| File | ID | Official Micro-F1@7 | Intended experiment |
|---|---|---:|---|
| submission1.zip | 1039201 | .1949 | E00 (original notebook's first archive; provisional map) |
| submission2.zip | 1039202 | .2144 | E01 (provisional) |
| submission3.zip | 1039204 | .2067 | E02 (provisional) |
| submission4.zip | 1039205 | .2239 | E03 (provisional) |
| submission5.zip | 1039207 | .2268 | E04 (provisional) |
| submission6.zip | 1039208 | .2138 | E05 (provisional) |
| submission7.zip | 1039209 | .2209 | E06 (provisional) |
| submission8.zip | 1039636 | .3839 | E07 TF-IDF |
| submission9.zip | 1039637 | .4265 | E08 Citations |
| submission10.zip | 1039657 | .4253 | E09 LightGBM |
| **submission11.zip** | **1039660** | **.4619** | **E10 final fusion — winner** |

The user separately described original submission1 as E01. This conflicts with the notebook/archive's E00-first enumeration, so the mapping for the first seven is explicitly provisional pending a copy of the renamed `submission1.zip`–`submission7.zip` archives (for byte/JSON comparison). The winning E10 is not ambiguous.

## Original candidate-pool and self-retrieval diagnostic

Official dataset measurements: Train 827/4320/5308 (queries/candidates/gold), Dev 118/1023/973, Test 237/1727/unlabeled; split-query ID overlap 0, all queries appear in their own candidate pool with identical text, none is self-gold in train/dev. All 118 Dev self IDs appear in E07's top seven, and 113 in E08. Removing query ID *post hoc* changes E07 348→382 correct Dev pairs and E08 382→407, corresponding to F1 0.424680378 and 0.452473596 respectively. The archived E10 **test** archive includes the self-case for 85 of 237 queries. These were not new Codabench submissions.

Concrete Dev examples: `0001416283` E07=1/E08=6 of 11 gold; `0000092507` E07=3/E08=7 of 12; `0001673100` E07=0/E08=0 of 4; `0181443842` E08=6 of 92. Citation match counts are 105 true/2 false for candidate CITATOR incoming, 44 true/23 false for direct mentions. The field may expose future-citation information unavailable in chronological deployment; report this limitation in any applied conclusions.

## Validation boundaries and unverified claims

- Exact package-format verification of the archived E10 ZIP was repeated from Drive against 237 official test query IDs and candidate IDs. The two prior independently generated E07/E08 ZIPs also pass the same validator.
- Original neural source and V2 source are refactored as executable scripts; syntax and CLI tests are available. Original large neural checkpoint loading was not repeated in this audit environment.
- Full CPU V2 replay was initiated; TF-IDF development cache and citation arrays were reconstructed, but computing the complete test TF-IDF and hashed feature matrices exceeded the execution time budget. **Do not claim bit-identical end-to-end E10 replay** from this audit run.
- Earlier Phase-5 full-training regeneration and previous Phase-1/Phase-6 original-weight replay limitations remain unchanged.
- See [Phase status](PHASE_STATUS.md#phase-7-research-release) for checks run against the complete checkout.
