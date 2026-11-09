"""Build the standalone Phase 4 E01 source archive."""

import argparse
import os
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from package_phase1_winner import ROOT, winner_only_models

README = """# Phase 4 best-model source | E01

E01 fully fine-tunes the Phase 1 Efficient GQA 700M foundation for binary
court judgment prediction. The archived epoch-2 classifier at threshold
0.51 achieved 0.7393 official Codabench macro-F1.

Requires Python 3.10+, the trusted Phase 4 E01 `best_model.pt`, the original
Phase 1 `legal_bpe_32k.json` tokenizer, and the official Phase 4 CJPE test JSON.
The weights, tokenizer and case data are distributed separately. The Phase 4
classifier checkpoint contains its backbone and head. A Phase 1 checkpoint
alone cannot generate E01 predictions.

Install `requirements.txt`, then set `PYTHONPATH=code` (PowerShell:
`$env:PYTHONPATH='code'`). From the extracted directory run:

```bash
python -m phase4.winner --test /path/to/test.json --tokenizer /path/to/legal_bpe_32k.json \\
  --checkpoint /path/to/best_model.pt --output E01_submission.zip --device cuda --precision bf16
```

On CPU use `--device cpu --precision fp32`. The original run used BF16.
The ZIP produced by the command contains one root-level `prediction.json`.
Only load checkpoints you trust.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    dest = parser.parse_args().output
    if dest.suffix.lower() != ".zip":
        parser.error("--output must end in .zip")
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_suffix(".tmp")
    try:
        with ZipFile(temporary, "w", ZIP_DEFLATED) as archive:
            archive.writestr("README.md", README)
            archive.writestr(
                "requirements.txt", "torch>=2.4,<3.0\nnumpy>=1.26,<3.0\ntokenizers>=0.19,<1.0\n"
            )
            archive.writestr("code/phase1/__init__.py", "")
            archive.writestr("code/phase1/models.py", winner_only_models())
            for name in (
                "__init__.py",
                "winner.py",
                "data.py",
                "inference.py",
                "gqa.py",
                "submission.py",
            ):
                path = f"code/phase4/{name}"
                archive.write(ROOT / path, path)
        with ZipFile(temporary) as archive:
            corrupt = archive.testzip()
            if corrupt is not None:
                raise ValueError(f"Corrupt package entry {corrupt}")
        os.replace(temporary, dest)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Built {dest}")


if __name__ == "__main__":
    main()
