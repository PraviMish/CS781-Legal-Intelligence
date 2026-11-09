# Project context: CS781 legal NLP capstone

## Purpose
Build a self-hosted legal language foundation and show how the same legal-only, from-scratch training infrastructure supports practical NLP tasks. No privileged client data leave controlled infrastructure, and no external legal training corpora or commercial model APIs are used. Phase 1 models must contain 50--300M parameters.

## Seven official phases
1. Legal pretraining: 512-word prompt -> 128-word continuation; BERTScore F1.
2. Legal NER: exact character offsets and 12 entity types; strict macro-F1.
3. Rhetorical roles: ordered document sentences -> one of 13 roles/sentence; macro-F1.
4. Court judgment: case -> binary outcome; macro-F1.
5. Statute identification: case -> set of statute IDs; macro-F1.
6. Abstractive summarization: document -> summary; BERTScore F1.
7. Prior-case retrieval: query -> seven distinct candidate case IDs, ranked; micro-F1@7.

## Permanent deliverables
1. Public, complete experimental GitHub code for all meaningful models and submissions, professionally documented and independently runnable against the official dataset.
2. Separate course-code ZIP that includes only the best-performing model code per phase, while retaining all dependencies needed to run it.
3. Comprehensive LaTeX master research report maintained phase by phase, with all experiments, data analysis, ablations, errors, results, and explanatory figures. A second, template-compliant final report will be derived with a ten-page main-body limit and LLM declaration, as required by the course.
4. Final course archive contains `code/` and `report/` directly at ZIP root; do not add a wrapper directory. The GitHub link belongs in the final course report.

## Development and safety rules
- Work locally in a clean workspace; use Git commits and a GitHub remote when available.
- Original Colab/Kaggle notebooks are research evidence, not directly published. Clean code must preserve actual algorithms and all meaningful experiments, not obsolete manual restore cells, duplicate definitions, debugging code or personal paths.
- Use automatic checkpoint restoration in ordinary training entry points, portable paths, explicit configs, clear docstrings, and tests.
- Never publish raw data, token secrets, access keys, training checkpoints, huge caches or private account details.
- Avoid mentioning the user's personal training accelerator or machine identity in public documentation. Report mandatory compute disclosures accurately where required.
- The chosen 'best' model is selected by the official competition's primary test metric, not by validation loss alone.
- All reports must separate measured evidence from hypotheses. Never fabricate missing rankings or experiments.
- Each phase's dedicated notebook and Drive artifact directory are the source evidence; confirm exact checkpoint/decoder selected by score sheets before freezing submission code.
- Each report figure is generated with real source data, versioned script, consistent filename, and tested LaTeX path.

## Phase-1 established result
Seven official submissions across four architectures and continued-training/decoding variants. Initial Efficient GQA at 100M tokens won official BERTScore F1 0.83479506; 300M, 500M-tuned, and 700M-tuned variants scored below it despite lower held-out next-token loss. The full official corpus yielded ~4.488B usable BPE tokens; winner architecture uses 14 blocks, 768 hidden, 12 query/4 KV heads, RoPE, RMSNorm, SwiGLU, 2048 FFN and 32k BPE, totaling 112,678,656 parameters. See phase README and CSV registry for full detail.
