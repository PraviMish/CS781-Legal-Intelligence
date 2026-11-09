# Phase 6 evidence and verification

## Evidence recovered

| Evidence | What it establishes |
| --- | --- |
| Original executed `Capstone_6.ipynb`, original Drive comparison and six archives | Section packing, six trained runs, first 32 held-out cases, E02 response teacher with 1,000 summaries, original metric records |
| Executed `Capstone_6_Revised_v2(2).ipynb`, V2 Drive comparison, validation records and manifest | Aligned passages and rotating schedule, seven trained runs, six ZIPs, E02 response teacher with 2,686 targets, 64-document scores |
| Executed `Capstone_6_Final_V3(1).ipynb`, V3 Drive comparison, validation records, E05 history and manifest | Exhaustive target schedule, V2 warm starts, matched student initialization, raw/grounded ablation, 96-document scores and seven ZIPs |
| Supplied Codabench screenshot and explicit E01 = sub1 mapping | Seven finished V3 entries and their official primary scores, including E05 at 0.8459 |
| Course Phase 6 specification and pinned Hugging Face file listing | Input schema, hidden references, accepted output schemas, and ROUGE-1/ROUGE-L as official secondary metrics |

The two Drive data copies match the official revision
`93b0059be756c6c461f3a225de25aabda7d5f84a`. The test copy matches the Git
blob ID. The training file's SHA-256 and byte count reconstruct the official
Git LFS pointer blob. The archived tokenizer loads as the original 32k BPE
with special IDs 0/1/2/3. The 6,466/564 reconstruction matches all archived
validation IDs using the original notebook seed. Labeled records contain
27 additional rows beyond the number of unique normalized-text groups.
There is zero exact source-text or ID overlap between labeled and test data.

## Verified outputs

All 20 reconstructed configurations match the archived total and trainable
parameter counts to the recorded precision.

- All **19** historical archives passed CRC, single-root JSON, nonempty
  summary, unique ID, exact test-ID coverage and official row-order checks.
  This covers six original, six V2 and seven V3 archives.
- Recomputed V2 E02 and V3 E02, E05 and E07 local ROUGE from archived
  predictions. The V3 grounded policy was applied again to the raw
  predictions and reproduced the recorded aggregate values.
- V3 E05's manifest entry records update **7750**, policy `raw`, 100 test
  examples and run fingerprint `f5ffb3fdd5974438be82`. The selected NLL of
  1.962419698946178 matches the saved history. V2 E05 came from update 4848.
- Qualitative V2 E02 checks compared judgment 1770 and judgment 757 against
  the pinned training records. The added section number and payment figure
  were absent in the corresponding source and reference text under the
  inspected number patterns. These examples are illustrative.
- The CPU suite reports 53 passed and one skipped. The skip is the existing
  Phase 3 standalone-package test, which needs its separate archive. The
  extracted E05 package passed strict loading and produced a valid one-case
  submission with full-size synthetic weights and the archived tokenizer.
- The master report compiles to 36 pages. The Phase 6 pages, figures and
  tables were rendered and inspected. The compiler still reports pre-existing
  Phase 3 typesetting warnings.

[Submission hashes and replay aggregates](../code/phase6/experiments/archive_verification.json) ·
[Data statistics](../code/phase6/experiments/dataset_statistics.json) ·
[Complete results](../code/phase6/experiments/results.csv)

## Verification limits

The 462,999,031-byte E05 checkpoint exceeds the Drive connector's
268,435,456-byte transfer limit. Archived outputs establish the historical
submission schema and IDs. Original-weight decoding and complete GPU
training are still open. The screenshot provides the official BERTScore
values. Individual hidden references, official secondary score sheets and
global leaderboard placement were unavailable. The experiment archive and
source-only release keep private paths, documents, predictions, original
notebooks and weights outside Git.
