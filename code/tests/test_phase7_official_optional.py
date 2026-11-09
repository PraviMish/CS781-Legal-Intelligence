"""Optional official-data regression checks.

Set PHASE7_DATA_DIR to the Phase 7 folder containing train/dev/test.
Set PHASE7_E10_ARCHIVE to an archived best-submission ZIP (optional).
These tests never inspect hidden test labels and do not require large weights.
"""
from __future__ import annotations
import os
from pathlib import Path
import pytest
from phase7.validation import load_split, verify_submission, micro_f1_at7

DATA = os.environ.get('PHASE7_DATA_DIR')
pytestmark = pytest.mark.skipif(not DATA, reason='Set PHASE7_DATA_DIR for official data checks')


def test_exact_official_split_counts_and_integrity():
    expected = {'train': (827, 4320, 5308), 'dev': (118, 1023, 973),
                'test': (237, 1727, 0)}
    for split, (n_queries, n_candidates, gold_total) in expected.items():
        queries, candidates = load_split(DATA, split)
        assert len(queries) == n_queries
        assert len(candidates) == n_candidates
        assert sum(len(row.get('relevant_candidates', [])) for row in queries.values()) == gold_total
        assert all(qid in candidates for qid in queries), 'Historical self-retrieval diagnostic changed'
        if split != 'test':
            assert all(qid not in row.get('relevant_candidates', []) for qid, row in queries.items())


def test_no_duplicate_or_cross_split_query_ids():
    query_sets=[]
    for split in ('train', 'dev', 'test'):
        query_sets.append(set(load_split(DATA, split)[0]))
    for i in range(3):
        for j in range(i+1,3):
            assert not (query_sets[i] & query_sets[j])


def test_archived_winner_zip_if_supplied():
    path = os.environ.get('PHASE7_E10_ARCHIVE')
    if not path:
        pytest.skip('Set PHASE7_E10_ARCHIVE to validate the historical E10 ZIP')
    queries, candidates = load_split(DATA, 'test')
    pred = verify_submission(path, queries, candidates)
    assert len(pred) == 237
    # Historical file still ranks the query itself for 85 cases.
    assert sum(qid in ranks for qid, ranks in pred.items()) == 85
