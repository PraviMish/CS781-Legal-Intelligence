"""Generate the official Phase 6 submission with the archived V3 E05 model."""

import argparse
import hashlib

import torch

from phase1.models import EfficientGQATransformer
from .adaptation import add_low_rank
from .data import PassageInputs, load_records, load_tokenizer
from .submission import digest, generate_submission


def winner_model():
    return add_low_rank(EfficientGQATransformer(
        vocab_size=32000, max_seq_len=1280, d_model=768, n_layers=14,
        n_query_heads=12, n_kv_heads=4, d_ff=2048,
    ))


def load_winner(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (checkpoint.get("exp_id"), checkpoint.get("version"), checkpoint.get("step")) != (
        "E05", "v3_allpassages_kl_20261101_v1", 7750
    ):
        raise ValueError("Expected the submitted V3 E05 best.pt at optimizer update 7750")
    model = winner_model()
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("test", "tokenizer", "checkpoint", "output"):
        parser.add_argument(f"--{option}", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    args = parser.parse_args(argv)
    if args.device.startswith("cuda"):
        torch.set_float32_matmul_precision("high")
    rows = load_records(args.test)
    inputs = PassageInputs(load_tokenizer(args.tokenizer))
    fingerprint = hashlib.sha256(":".join([
        "E05-v3-7750-raw-v1", args.precision, args.device,
        *(digest(path) for path in (args.test, args.tokenizer, args.checkpoint)),
    ]).encode()).hexdigest()
    model = load_winner(args.checkpoint, args.device)
    generate_submission(model, "efficient_gqa", inputs, rows, args.output,
                        args.device, args.precision, "hierarchical", "raw", fingerprint)
    print(f"Validated {len(rows)} summaries in {args.output}")


if __name__ == "__main__":
    main()
