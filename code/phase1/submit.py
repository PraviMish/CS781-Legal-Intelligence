"""Reproducible, resumable official Phase-1 prediction.json and submission.zip."""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

from tokenizers import Tokenizer
from common.artifacts import write_json
from phase1.data import load_official_test
from phase1.generate import CONFIGS, LegalDecoder, load_trained_model


def validate_predictions(predictions: dict, inputs: dict[str, str]) -> None:
    if set(predictions) != set(inputs):
        raise ValueError(f'ID mismatch: missing={len(set(inputs)-set(predictions))}, '
                         f'extra={len(set(predictions)-set(inputs))}')
    for key, entry in predictions.items():
        if not isinstance(entry, dict) or not isinstance(entry.get('text'), str):
            raise ValueError(f'Invalid prediction schema at {key}')
        if len(entry['text'].split()) != 128:
            raise ValueError(f'Expected 128 whitespace words at {key}')


def make_submission(model, architecture: str, tokenizer: Tokenizer,
                    test_examples: dict[str, str], output_dir: Path,
                    decoding: str = 'legacy_word_guard_v1',
                    save_every: int = 10) -> Path:
    """Only use matching partial predictions from the SAME checkpoint+decoder.

    Each invocation's output directory is assumed to identify a particular
    checkpoint and decoder. The CLI writes a manifest and refuses to reuse
    predictions from another checkpoint or experiment.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    partial_path = output_dir / f'partial_{decoding}.json'
    existing = json.loads(partial_path.read_text()) if partial_path.exists() else {}
    predictions = {k: v for k, v in existing.items() if k in test_examples and
                   isinstance(v, dict) and isinstance(v.get('text'), str) and
                   len(v['text'].split()) == 128}
    generator = LegalDecoder(tokenizer)
    model.eval()
    for idx, example_id in enumerate(test_examples):
        if example_id in predictions:
            continue
        text = generator.generate(model, architecture, test_examples[example_id], CONFIGS[decoding])
        predictions[example_id] = {'text': text}
        if (idx + 1) % save_every == 0:
            write_json(partial_path, predictions)
    write_json(partial_path, predictions)
    validate_predictions(predictions, test_examples)
    output = output_dir / 'prediction.json'
    write_json(output, predictions)
    archive = output_dir / 'submission.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(output, arcname='prediction.json')
    with zipfile.ZipFile(archive) as zf:
        assert zf.namelist() == ['prediction.json']
        validate_predictions(json.loads(zf.read('prediction.json')), test_examples)
    return archive


def main() -> None:
    p = argparse.ArgumentParser(description='Prepare official Phase-1 submission ZIP')
    p.add_argument('--architecture', required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--tokenizer', type=Path, required=True)
    p.add_argument('--test-json', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--decoding', choices=CONFIGS, default='legacy_word_guard_v1')
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / 'submission_manifest.json'
    import hashlib
    checkpoint_sha = hashlib.sha256()
    with args.checkpoint.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            checkpoint_sha.update(block)
    expected = {'architecture': args.architecture, 'decoding': args.decoding,
                'checkpoint_sha256': checkpoint_sha.hexdigest()}
    if manifest.exists() and json.loads(manifest.read_text()) != expected:
        raise RuntimeError('Output directory already belongs to another checkpoint or decoder')
    write_json(manifest, expected)
    model = load_trained_model(args.architecture, args.checkpoint)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    test_examples = load_official_test(args.test_json)
    if any(not value for value in test_examples.values()):
        raise ValueError('Official test data contain empty inputs')
    path = make_submission(model, args.architecture, tokenizer, test_examples,
                           args.output_dir, decoding=args.decoding)
    print('Verified ZIP:', path, '\nExamples:', len(test_examples))


if __name__ == '__main__':
    main()
