"""Prepare the official four-source legal corpus and 160-example local test set.

This is a portable replacement for the original notebook's manual data cells.
Raw datasets remain outside Git. Each source has a resumable state JSON.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer

from common.artifacts import write_json
from phase1.data import SOURCES, build_source, mix_weights
from phase1.tokenizer import train_tokenizer


def source_files(data_root: Path) -> dict[str, Path]:
    """Resolve the official file names, not arbitrary external corpora."""
    training = data_root / 'Train'
    files = {
        'acts': training / 'train_acts.json',
        'high_courts': training / 'train_high_courts.json',
        'supreme_court': training / 'train_supreme_court.json',
        'tribunals': training / 'train_tribunals.json',
    }
    absent = [str(path) for path in files.values() if not path.is_file()]
    if absent:
        raise FileNotFoundError('Missing official dataset files:\n' + '\n'.join(absent))
    return files


def prepare(*, data_root: Path, tokenizer_path: Path, output_dir: Path,
            mode: str = 'initial', cached_tokens: int = 500_000_000) -> dict:
    paths = source_files(data_root)
    if tokenizer_path.exists():
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    else:
        print('Training fresh 32k legal BPE tokenizer from four official sources...')
        tokenizer = train_tokenizer(paths, tokenizer_path)
    if tokenizer.get_vocab_size() != 32000:
        raise ValueError('Expected original legal BPE vocabulary of 32,000')
    sizes = {key: path.stat().st_size for key, path in paths.items()}
    original_weights = mix_weights(sizes, alpha=0.7)
    counts, all_pseudo = {}, []
    for source, path in paths.items():
        limit = round(cached_tokens * original_weights[source]) if mode == 'initial' else None
        print(f'Preparing {mode} source {source}, token target={limit}')
        state = build_source(source, path, tokenizer, output_dir,
                             train_limit=limit, pseudo_limit=40)
        counts[source] = int(state['train_tokens'])
        pseudo_file = output_dir / f'pseudo_dev_{source}.json'
        if pseudo_file.exists():
            all_pseudo.extend(json.loads(pseudo_file.read_text(encoding='utf-8')))
    weights = original_weights if mode == 'initial' else mix_weights(counts, alpha=0.7)
    metadata = {'mode': mode, 'source_token_counts': counts, 'source_weights': weights,
                'vocab_size': 32000, 'validation_fraction': .01, 'mixing_alpha': .7}
    write_json(output_dir / 'source_weights.json', metadata)
    write_json(output_dir / 'pseudo_dev.json', all_pseudo)
    print('Preparation complete:', json.dumps(metadata, indent=2))
    return metadata


def main():
    p = argparse.ArgumentParser(description='Prepare official Phase-1 legal corpus')
    p.add_argument('--data-root', required=True, type=Path,
                   help='Folder containing official Train/ and Test/ subdirectories')
    p.add_argument('--tokenizer', required=True, type=Path)
    p.add_argument('--output-dir', required=True, type=Path)
    p.add_argument('--mode', choices=['initial', 'full'], default='initial')
    p.add_argument('--cached-tokens', type=int, default=500_000_000)
    args = p.parse_args()
    prepare(data_root=args.data_root, tokenizer_path=args.tokenizer,
            output_dir=args.output_dir, mode=args.mode, cached_tokens=args.cached_tokens)


if __name__ == '__main__':
    main()
