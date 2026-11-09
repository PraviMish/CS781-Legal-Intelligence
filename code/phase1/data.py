"""Phase-1 legal corpus streaming, document splits, tokenization and sampling.

Input data are the official four JSON sources. Token IDs are stored as uint16;
no raw or tokenized dataset is committed to this repository.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import ijson
import numpy as np
from tokenizers import Tokenizer

from common.artifacts import write_json

SOURCES = ('acts', 'high_courts', 'supreme_court', 'tribunals')
TEXT_FIELDS = ('text', 'document', 'content', 'judgment', 'case_text', 'body')


def detect_layout(path: Path) -> tuple[str, str | None]:
    with path.open('rb') as handle:
        iterator = ijson.parse(handle)
        first = next(iterator)
        if first[1] == 'start_array':
            return 'array', None
        if first[1] != 'start_map':
            raise ValueError(f'Unsupported JSON root in {path}')
        second = next(iterator)
        if second[1] != 'map_key':
            raise ValueError(f'Unsupported JSON layout in {path}')
        third = next(iterator)
        if third[1] == 'start_array':
            return 'wrapped_array', second[2]
        return 'dict_records', None


def iter_json_records(path: str | Path):
    path = Path(path)
    layout, wrapper = detect_layout(path)
    with path.open('rb') as handle:
        if layout == 'array':
            for i, record in enumerate(ijson.items(handle, 'item')):
                yield str(i), record
        elif layout == 'wrapped_array':
            for i, record in enumerate(ijson.items(handle, f'{wrapper}.item')):
                yield str(i), record
        else:
            for key, record in ijson.kvitems(handle, ''):
                yield str(key), record


def extract_text(record) -> str | None:
    if isinstance(record, str):
        return record or None
    if not isinstance(record, dict):
        return None
    for name in TEXT_FIELDS:
        text = record.get(name)
        if isinstance(text, str) and text.strip():
            return text
    values = [value for value in record.values() if isinstance(value, str) and value.strip()]
    return max(values, key=len) if values else None


def validation_fraction(key: str) -> float:
    """Exact original 12-hex-character SHA1 split function."""
    integer = int(hashlib.sha1(key.encode('utf-8')).hexdigest()[:12], 16)
    return integer / float(16 ** 12)


def pseudo_example(text: str, source: str, key: str) -> dict | None:
    """The original deterministic 512-word prompt / 128-word continuation split."""
    words = text.split()
    if len(words) < 640:
        return None
    maximum = len(words) - 640
    start = int(hashlib.sha1(key.encode()).hexdigest()[:12], 16) % (maximum + 1)
    return {'source': source, 'prompt': ' '.join(words[start:start + 512]),
            'reference': ' '.join(words[start + 512:start + 640])}


def build_source(source: str, json_path: str | Path, tokenizer: Tokenizer,
                 output_dir: str | Path, *, val_fraction: float = 0.01,
                 val_limit: int = 2_000_000, pseudo_limit: int = 40,
                 train_limit: int | None = None, save_every: int = 1000) -> dict:
    """Resumable document-level split and tokenization of one official JSON file.

    State files record committed tokens and record positions; on restart binary files
    are truncated back to the last committed state to prevent duplicate tokens.
    The validation file intentionally caps tokens, matching the original method.
    """
    if tokenizer.get_vocab_size() >= 65536:
        raise ValueError('uint16 token storage requires vocabulary < 65536')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: output_dir / f'{name}_{source}.bin' for name in ('train', 'val')}
    pseudo_file = output_dir / f'pseudo_dev_{source}.json'
    state_path = output_dir / f'state_{source}.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        'records_seen': 0, 'train_tokens': 0, 'val_tokens': 0,
        'pseudo_examples': 0, 'completed': False,
    }
    if state['completed']:
        for kind in ('train', 'val'):
            assert paths[kind].stat().st_size == int(state[f'{kind}_tokens']) * 2
        return state
    for kind in ('train', 'val'):
        target = paths[kind]
        if not target.exists() and state[f'{kind}_tokens']:
            raise FileNotFoundError(target)
        with target.open('ab'):
            pass
        with target.open('r+b') as handle:
            handle.truncate(state[f'{kind}_tokens'] * 2)
    pseudo = json.loads(pseudo_file.read_text()) if pseudo_file.exists() else []
    pseudo = pseudo[:state['pseudo_examples']]
    eos = tokenizer.token_to_id('<eos>')
    if eos is None:
        raise ValueError('Tokenizer missing <eos>')
    with paths['train'].open('ab') as train_handle, paths['val'].open('ab') as val_handle:
        for i, (record_id, record) in enumerate(iter_json_records(json_path)):
            if i < state['records_seen']:
                continue
            text = extract_text(record)
            state['records_seen'] = i + 1
            if text:
                key = f'{source}:{record_id}'
                is_val = validation_fraction(key) < val_fraction
                kind = 'val' if is_val else 'train'
                permit = (val_limit - state['val_tokens'] if is_val else
                          None if train_limit is None else train_limit - state['train_tokens'])
                if permit is None or permit > 0:
                    ids = tokenizer.encode(text).ids + [eos]
                    # Original program appends the complete document even when this
                    # step takes the binary slightly beyond its target budget.
                    np.asarray(ids, dtype=np.uint16).tofile(val_handle if is_val else train_handle)
                    state[f'{kind}_tokens'] += len(ids)
                if is_val and len(pseudo) < pseudo_limit:
                    item = pseudo_example(text, source, key)
                    if item:
                        item['id'] = f'{source}_pseudo_{len(pseudo):04d}'
                        pseudo.append(item)
                        state['pseudo_examples'] = len(pseudo)
            if state['records_seen'] % save_every == 0:
                train_handle.flush()
                val_handle.flush()
                write_json(pseudo_file, pseudo)
                write_json(state_path, state)
            if train_limit is not None and state['train_tokens'] >= train_limit and len(pseudo) >= pseudo_limit:
                break
        train_handle.flush()
        val_handle.flush()
    state['completed'] = True
    write_json(pseudo_file, pseudo)
    write_json(state_path, state)
    return state


class MixedTokenStore:
    """Memory-mapped sampler preserving original Phase-1 contiguous-block semantics."""
    def __init__(self, directory: str | Path, split: str, source_weights: dict[str, float],
                 seed: int = 42, min_length: int = 1280):
        directory = Path(directory)
        self.arrays = {}
        for source in source_weights:
            path = directory / f'{split}_{source}.bin'
            if path.exists():
                array = np.memmap(path, dtype=np.uint16, mode='r')
                if len(array) > min_length:
                    self.arrays[source] = array
        if not self.arrays:
            raise FileNotFoundError(f'No usable {split} token files in {directory}')
        self.sources = list(self.arrays)
        w = np.asarray([source_weights[source] for source in self.sources], dtype=np.float64)
        self.weights = w / w.sum()
        self.rng = np.random.default_rng(seed)

    def sample_batch(self, batch_size: int, sequence_length: int = 1280,
                     rng: np.random.Generator | None = None):
        import torch
        rng = self.rng if rng is None else rng
        batch = np.empty((batch_size, sequence_length), dtype=np.int64)
        sources = rng.choice(self.sources, size=batch_size, p=self.weights)
        for i, source in enumerate(sources):
            tokens = self.arrays[source]
            start = rng.integers(0, len(tokens) - sequence_length)
            batch[i] = tokens[start:start + sequence_length]
        return torch.from_numpy(batch)


def mix_weights(counts: dict[str, int], alpha: float = 0.7) -> dict[str, float]:
    values = {source: max(1, number) ** alpha for source, number in counts.items()}
    total = sum(values.values())
    return {source: value / total for source, value in values.items()}


def load_official_test(path: str | Path) -> dict[str, str]:
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if isinstance(data, dict):
        return {str(key): extract_text(record) for key, record in data.items()}
    if isinstance(data, list):
        return {str(item.get('id', i)): extract_text(item) for i, item in enumerate(data)}
    raise ValueError('Unexpected official test JSON format')


def main() -> None:
    parser = argparse.ArgumentParser(description='Tokenize an official Phase-1 source')
    parser.add_argument('--source', choices=SOURCES, required=True)
    parser.add_argument('--json', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=None, help='Optional cache token budget')
    args = parser.parse_args()
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    state = build_source(args.source, args.json, tokenizer, args.output_dir, train_limit=args.limit)
    print(json.dumps(state, indent=2))


if __name__ == '__main__':
    main()
