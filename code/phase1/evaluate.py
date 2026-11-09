"""Held-out cross-entropy and competition-shaped pseudo-development evaluation."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from rouge_score import rouge_scorer
import sacrebleu
from tokenizers import Tokenizer

from phase1.generate import CONFIGS, LegalDecoder, load_trained_model
from phase1.train import validation_metrics, read_source_weights
from phase1.data import MixedTokenStore


def pseudo_dev_metrics(model, tokenizer: Tokenizer, architecture: str, examples: list[dict],
                       *, decoding: str = 'legacy_word_guard_v1', max_examples: int = 160) -> dict:
    """Balance selection across four sources, as in the original Colab notebook.

    ROUGE-L is an average over document examples; BLEU-4 is a corpus value in
    sacrebleu's 0--100 units. These are not official BERTScore numbers.
    """
    if max_examples < 4:
        raise ValueError('Need >=4 examples to represent all legal sources')
    chosen = []
    for source in ('acts', 'high_courts', 'supreme_court', 'tribunals'):
        candidates = [x for x in examples if x['source'] == source]
        take = max_examples // 4
        if len(candidates) < take:
            raise ValueError(f'{source} contains only {len(candidates)} pseudo-dev examples')
        chosen.extend(candidates[:take])
    decoder = LegalDecoder(tokenizer)
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    predictions, references, per_source = [], [], defaultdict(list)
    model.eval()
    for item in chosen:
        generated = decoder.generate(model, architecture, item['prompt'], CONFIGS[decoding])
        reference = item['reference']
        score = float(scorer.score(reference, generated)['rougeL'].fmeasure)
        predictions.append(generated)
        references.append(reference)
        per_source[item['source']].append(score)
    bleu = sacrebleu.corpus_bleu(predictions, [references]).score
    return {'architecture': architecture, 'decoding': decoding, 'examples': len(chosen),
            'rouge_l_f1': float(np.mean([x for row in per_source.values() for x in row])),
            'bleu_4_sacrebleu': float(bleu),
            'per_source_rouge_l_f1': {k: float(np.mean(v)) for k, v in per_source.items()}}


def main() -> None:
    p = argparse.ArgumentParser(description='Evaluate one Phase-1 trusted checkpoint')
    p.add_argument('--architecture', required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--tokenizer', type=Path, required=True)
    p.add_argument('--pseudo-dev', type=Path, required=True,
                   help='Saved pseudo_dev.json from original document-level split')
    p.add_argument('--decoding', choices=CONFIGS, default='legacy_word_guard_v1')
    p.add_argument('--max-examples', type=int, default=160)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    model = load_trained_model(args.architecture, args.checkpoint)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    examples = json.loads(args.pseudo_dev.read_text(encoding='utf-8'))
    metrics = pseudo_dev_metrics(model, tokenizer, args.architecture, examples,
                                 decoding=args.decoding, max_examples=args.max_examples)
    encoded = json.dumps(metrics, indent=2)
    print(encoded)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
