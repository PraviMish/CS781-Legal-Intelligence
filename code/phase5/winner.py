"""Standalone inference for the original E02 full GQA fine-tune."""

import argparse

import torch
from phase1.models import EfficientGQATransformer

from .classifier import StatuteClassifier
from .inference import precision_dtype, submit


def load_winner(checkpoint, device):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    config = saved["config"]
    if (config.get("arch") != "efficient_gqa" or config.get("method") != "full"
            or config.get("student", False) or config.get("loss") != "asl"):
        raise ValueError("Expected the Phase 5 E02 full GQA checkpoint")
    backbone = EfficientGQATransformer(vocab_size=32000, max_seq_len=1280, d_model=768,
                                      n_layers=14, n_query_heads=12, n_kv_heads=4, d_ff=2048)
    model = StatuteClassifier(backbone, "efficient_gqa")
    model.load_state_dict(saved["model"], strict=True)
    return model.to(device).eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("tokenizer", "test", "checkpoint", "thresholds", "output"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", choices=("auto", "fp32", "bf16", "fp16"), default="auto")
    args = parser.parse_args()
    device = torch.device(args.device)
    model = load_winner(args.checkpoint, device)
    count = submit(model, args.tokenizer, args.test, args.thresholds, args.output, 2,
                   device, precision_dtype(device, args.precision))
    print(f"Wrote {count} case predictions")


if __name__ == "__main__":
    main()
