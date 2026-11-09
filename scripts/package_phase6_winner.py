"""Build a standalone source archive for the submitted V3 E05 model."""

import argparse
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import ZIP_DEFLATED, ZipFile

from package_phase1_winner import ROOT, winner_only_models

README = """# Phase 6 | E05 best-model source

E05 adapts the 700M-token Phase 1 Efficient GQA foundation with rank-16
LoRA. The archived V3 best checkpoint at update 7,750 and its raw
hierarchical decoder achieved official BERTScore F1 0.8459 (Codabench
submission 1026556).

Use Python 3.11+, the trusted Phase 6 V3 E05 best.pt, the original Phase 1
legal_bpe_32k.json tokenizer, and the official Phase_6_Summ/Test/test.json.
Install requirements.txt and set PYTHONPATH=code (PowerShell:
$env:PYTHONPATH='code'). From the extracted directory run:

    python -m phase6.winner --test /path/to/test.json --tokenizer /path/to/legal_bpe_32k.json --checkpoint /path/to/best.pt --output E05_submission.zip --device cuda --precision bf16

For a CPU check set --device cpu --precision fp32. The archive contains
source only. The E05 checkpoint already contains the full adapted backbone.
Use the archived V3 E05 weights at step 7,750 for an original-run replay.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    output = parser.parse_args().output
    if output.suffix.lower() != ".zip":
        parser.error("--output must end in .zip")
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=output.parent, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with ZipFile(temporary, "w", ZIP_DEFLATED) as archive:
            archive.writestr("README.md", README)
            archive.writestr("requirements.txt",
                             "torch>=2.4,<3.0\nnumpy>=1.26,<3.0\ntokenizers>=0.19,<1.0\n")
            archive.writestr("code/phase1/__init__.py", "")
            archive.writestr("code/phase1/models.py", winner_only_models())
            for name in ("__init__", "adaptation", "data", "decoding", "submission", "winner"):
                path = f"code/phase6/{name}.py"
                archive.write(ROOT / path, path)
        with ZipFile(temporary) as archive:
            if archive.testzip() is not None:
                raise ValueError("Corrupt winner source archive")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Built {output}")


if __name__ == "__main__":
    main()
