"""Phase 7 protocol tests, no model weights or private data required."""
import json
import pytest

from phase7.validation import (
    load_split, validate_predictions, micro_f1_at7, build_submission, verify_submission
)


def example():
    candidates = {f'{i:06d}': {'text': ['prior case']} for i in range(10, 101, 10)}
    queries = {
        '000001': {'text': ['one'], 'relevant_candidates': ['000010', '000030']},
        '000002': {'text': ['two'], 'relevant_candidates': ['000090']},
    }
    pred = {
        '000001': [f'{i:06d}' for i in range(10, 71, 10)],
        '000002': [f'{i:06d}' for i in range(20, 81, 10)],
    }
    return queries, candidates, pred


def test_micro_f1_exact():
    q, c, p = example()
    result = micro_f1_at7(p, q, c)
    assert result['hits'] == 2
    assert result['gold_pairs'] == 3
    assert result['micro_f1_at7'] == pytest.approx(4 / 17)


def test_leading_zeroes_and_zip(tmp_path):
    q, c, p = example()
    path = build_submission(p, q, c, tmp_path / 'submission.zip')
    assert verify_submission(path, q, c) == p
    assert list(verify_submission(path, q, c)) == ['000001', '000002']


def test_invalid_rankings():
    q, c, p = example()
    validate_predictions(p, q, c)
    pdup = {k: list(v) for k, v in p.items()}
    pdup['000001'] = ['000010'] * 7
    with pytest.raises(ValueError):
        validate_predictions(pdup, q, c)
    pbad = {k: list(v) for k, v in p.items()}
    pbad['000001'][0] = '000999'
    with pytest.raises(ValueError):
        validate_predictions(pbad, q, c)
    with pytest.raises(ValueError):
        validate_predictions({'000001': p['000001']}, q, c)


def test_split_local_dataset(tmp_path):
    q, c, _ = example()
    for split in ('train', 'dev', 'test'):
        directory = tmp_path / split.capitalize()
        directory.mkdir()
        queries = q if split != 'test' else {k: {'text': row['text']} for k, row in q.items()}
        (directory / f'{split}.json').write_text(json.dumps(queries))
        (directory / f'{split}_candidates.json').write_text(json.dumps(c))
        qq, cc = load_split(tmp_path, split)
        assert list(qq) == list(q)
        assert list(cc) == list(c)
