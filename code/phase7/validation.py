"""Phase 7 official case-ID, ranking and Micro-F1@7 validation utilities.

Identifiers are opaque strings; never cast to integers or strip leading zeros.
Each official split has a split-specific global candidate pool. This utility
must be extended if later competitions introduce per-query candidate filters.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence
from zipfile import ZIP_DEFLATED, ZipFile

K = 7


def split_directory(data_dir: str | Path, split: str) -> Path:
    """Find the official title-case folder or a locally normalized lowercase folder."""
    if split not in {'train', 'dev', 'test'}:
        raise ValueError(split)
    root = Path(data_dir)
    official = root / split.capitalize()
    return official if official.is_dir() else root / split


def load_split(data_dir: str | Path, split: str):
    """Return official query and candidate dictionaries, without changing IDs."""
    root = split_directory(data_dir, split)
    with (root / f'{split}.json').open(encoding='utf-8-sig') as f:
        queries = json.load(f)
    with (root / f'{split}_candidates.json').open(encoding='utf-8-sig') as f:
        candidates = json.load(f)
    if not isinstance(queries, dict) or not isinstance(candidates, dict):
        raise TypeError('Expected query and candidate dictionaries keyed by exact case ID')
    if not queries or len(candidates) < K:
        raise ValueError('Empty query pool or fewer than seven candidates')
    if not all(isinstance(x, str) for x in list(queries) + list(candidates)):
        raise TypeError('Case IDs must remain strings')
    for qid, row in queries.items():
        if not isinstance(row.get('text'), (str, list)):
            raise ValueError(f'Missing query text for {qid}')
        gold = row.get('relevant_candidates', [])
        if split == 'test' and gold:
            raise ValueError('Test data must not contain public relevance labels')
        if not set(gold).issubset(candidates):
            raise ValueError(f'{split} query {qid} contains unknown gold candidate ID')
    return queries, candidates


def validate_predictions(predictions: Mapping[str, Sequence[str]], query_ids, candidate_ids):
    qset, cset = set(query_ids), set(candidate_ids)
    if set(predictions) != qset:
        raise ValueError(f'Query mismatch: missing={len(qset - set(predictions))}, extra={len(set(predictions) - qset)}')
    for qid, rank in predictions.items():
        if not isinstance(rank, list) or len(rank) != K or len(set(rank)) != K:
            raise ValueError(f'{qid}: expected exactly {K} distinct ranked candidate IDs')
        if any(not isinstance(cid, str) or cid not in cset for cid in rank):
            raise ValueError(f'{qid}: invalid candidate string ID')


def micro_f1_at7(predictions, queries, candidates):
    validate_predictions(predictions, queries, candidates)
    total = sum(len(row.get('relevant_candidates', [])) for row in queries.values())
    if total == 0:
        raise ValueError('Cannot score unlabeled test data')
    hits = sum(len(set(predictions[qid]) & set(row.get('relevant_candidates', [])))
               for qid, row in queries.items())
    return {
        'micro_f1_at7': 2 * hits / (7 * len(queries) + total),
        'micro_precision_at7': hits / (7 * len(queries)),
        'micro_recall_at7': hits / total,
        'hits': hits,
        'gold_pairs': total,
    }


def build_submission(predictions, queries, candidates, destination):
    validate_predictions(predictions, queries, candidates)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(destination, 'w', ZIP_DEFLATED) as archive:
        archive.writestr('prediction.json', json.dumps(predictions, ensure_ascii=False, indent=2))
    verify_submission(destination, queries, candidates)
    return destination


def verify_submission(path, queries, candidates):
    with ZipFile(path) as archive:
        if archive.namelist() != ['prediction.json'] or archive.testzip() is not None:
            raise ValueError('ZIP must contain exactly one root-level prediction.json')
        predictions = json.loads(archive.read('prediction.json'))
    validate_predictions(predictions, queries, candidates)
    return predictions
