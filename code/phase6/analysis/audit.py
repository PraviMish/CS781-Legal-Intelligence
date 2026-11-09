"""Replay private archives and derive publishable aggregate dataset statistics."""

import argparse
import json
from pathlib import Path

import numpy as np

from phase6.data import CONTEXT, doc_hash, load_records, load_tokenizer, split_records


def describe(values):
    return {"min": int(min(values)), "mean": float(np.mean(values)),
            **{f"p{p}": float(np.percentile(values, p)) for p in (50, 90, 95, 99)},
            "max": int(max(values))}


def lengths(rows, tokenizer):
    docs = [row["document"] for row in rows]
    doc_words = [len(doc.split()) for doc in docs]
    result = {"documents": len(docs), "document_words": describe(doc_words),
              "document_characters": describe([len(doc) for doc in docs])}
    token_lengths = []
    for offset in range(0, len(docs), 32):
        token_lengths.extend(len(enc.ids) for enc in tokenizer.encode_batch(docs[offset:offset + 32]))
    result["document_bpe"] = describe(token_lengths)
    result["document_over_context"] = sum(length > CONTEXT for length in token_lengths)
    refs = [row["summary"] for row in rows]
    if all(ref is not None for ref in refs):
        summary_words = [len(ref.split()) for ref in refs]
        result["summary_words"] = describe(summary_words)
        result["empty_summaries"] = summary_words.count(0)
        result["median_document_to_summary_words"] = float(np.median(
            [doc / ref for doc, ref in zip(doc_words, summary_words) if ref]
        ))
        summary_lengths = []
        for offset in range(0, len(refs), 64):
            summary_lengths.extend(
                len(enc.ids) for enc in tokenizer.encode_batch(refs[offset:offset + 64])
            )
        result["summary_bpe"] = describe(summary_lengths)
    return result


def replay(directory):
    from phase6.__main__ import rouge_summary
    from phase6.decoding import evidence_filter
    from phase6.submission import digest, read_submission

    test = load_records(directory / "test.json")
    archives = {}
    for path in sorted(directory.glob("*.zip")):
        if path.name.startswith(("submissions_", "FINAL_V2_", "FINAL_V3_")):
            rows = read_submission(path, test)
            archives[path.name] = {
                "count": len(rows), "test_order": [row["id"] for row in rows] == [row["id"] for row in test],
                "zip_sha256": digest(path),
                "mean_summary_words": float(np.mean([len(row["summary"].split()) for row in rows])),
            }
    scores = {}
    for path in sorted(directory.glob("v[23]_E*.json")):
        evidence = json.loads(path.read_text(encoding="utf-8"))
        samples = evidence["samples"]
        if "raw" in samples[0]:
            scores[path.stem] = {
                "raw": rouge_summary([dict(row, prediction=row["raw"]) for row in samples]),
                "grounded": rouge_summary([
                    dict(row, prediction=evidence_filter(row["raw"], row["document"])) for row in samples
                ]),
            }
        else:
            scores[path.stem] = rouge_summary(samples)
    return {"archives": archives, "recomputed_validation": scores}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--dataset-stats", action="store_true")
    args = parser.parse_args()
    output = replay(args.directory)
    if args.dataset_stats:
        source = args.directory
        tokenizer = load_tokenizer(source / "legal_bpe_32k.json")
        train = load_records(source / "train.json", labeled=True)
        test = load_records(source / "test.json")
        ids = json.loads((source / "validation_ids.json").read_text(encoding="utf-8"))
        groups = {doc_hash(row) for row in train}
        tr, val = split_records(train, ids)
        output["dataset"] = {
            "train": lengths(tr, tokenizer), "validation": lengths(val, tokenizer),
            "test": lengths(test, tokenizer),
            "labeled_exact_duplicates_after_first": len(train) - len(groups),
            "normalized_text_train_test_overlap": len(groups & {doc_hash(row) for row in test}),
            "train_test_id_overlap": len({row["id"] for row in train} & {row["id"] for row in test}),
        }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
