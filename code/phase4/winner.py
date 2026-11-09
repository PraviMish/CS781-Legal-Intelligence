"""Generate the official E01 submission using trusted weights and the archived BPE."""

import argparse
from pathlib import Path

import torch

from .data import load_cases, load_tokenizer, tokenize_cases
from .inference import choose_precision, e01_model, predict
from .submission import write_submission


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("test", "tokenizer", "checkpoint", "output"):
        parser.add_argument(f"--{arg}", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", choices=("auto", "fp32", "bf16"), default="auto")
    args = parser.parse_args()
    tokenizer, special = load_tokenizer(args.tokenizer)
    split = tokenize_cases(load_cases(args.test, labeled=False), tokenizer, special)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    model = e01_model(args.checkpoint).to(device)
    probabilities = predict(model, split, device, choose_precision(device, args.precision), 16)
    write_submission(args.output, split.ids, probabilities, 0.51)
    print(f"Wrote {len(split.ids)} E01 predictions to {args.output}")


if __name__ == "__main__":
    main()
