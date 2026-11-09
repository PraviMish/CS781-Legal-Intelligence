"""Train the original Phase-1 byte-level BPE tokenizer from official data only."""
from __future__ import annotations
import argparse
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from phase1.data import SOURCES, iter_json_records, extract_text

SPECIAL_TOKENS = ['<pad>', '<bos>', '<eos>', '<unk>']


def train_tokenizer(source_files: dict[str, Path], output: Path,
                    vocab_size: int = 32000, character_budget: int = 100_000_000) -> Tokenizer:
    """Balance the character budget evenly across all four sources (25M each)."""
    tokenizer = Tokenizer(BPE(unk_token='<unk>'))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    source_limit = character_budget // len(SOURCES)

    def corpus():
        for source in SOURCES:
            seen = 0
            for _, record in iter_json_records(source_files[source]):
                text = extract_text(record)
                if not text:
                    continue
                remaining = source_limit - seen
                if remaining <= 0:
                    break
                text = text[:remaining]
                seen += len(text)
                yield text

    trainer = BpeTrainer(vocab_size=vocab_size, min_frequency=2,
                         special_tokens=SPECIAL_TOKENS, initial_alphabet=ByteLevel.alphabet(),
                         show_progress=True)
    tokenizer.train_from_iterator(corpus(), trainer=trainer)
    output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output))
    return tokenizer


def main() -> None:
    ap = argparse.ArgumentParser(description='Reproduce Phase-1 tokenizer')
    for source in SOURCES:
        ap.add_argument('--' + source.replace('_','-'), type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    files = {source: getattr(args, source) for source in SOURCES}
    t = train_tokenizer(files, args.output)
    print('Tokenizer vocabulary:', t.get_vocab_size())

if __name__ == '__main__':
    main()
