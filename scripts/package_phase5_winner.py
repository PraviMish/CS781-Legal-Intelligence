"""Build the source-only E02 inference archive."""

import argparse
import ast
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
GQA_DEFINITIONS = {"RMSNorm", "rotate_half", "RotaryEmbedding", "GroupedQueryAttention",
                   "SwiGLU", "EfficientBlock", "EfficientGQATransformer"}
MODULES = ("__init__", "classifier", "data", "inference", "metrics", "submission", "winner")
README = """# CS781 Phase 5 | E02 Efficient GQA full fine-tuning

The best of the six official submissions scored 0.3875 macro-F1 (1014549).
This archive provides the inference source for that configuration.

Install the packages in code/requirements.txt. Set PYTHONPATH=code.
Supply the original Phase 1 legal BPE, the official Phase 5 Test/test.json,
and the matching E02 Phase 5 epoch-3 best.pt and thresholds.npy.

    python -m phase5.winner --tokenizer legal_bpe_32k.json --test test.json --checkpoint best.pt --thresholds thresholds.npy --output E02.zip

The ZIP contains one prediction.json at its root. Use the original CUDA
precision for prediction comparisons. Use --device cpu --precision fp32
for a CPU check. Weights, thresholds, tokenizer and dataset are external.
"""


def gqa_source():
    source = (ROOT / "code/phase1/models.py").read_text(encoding="utf-8")
    nodes = ast.parse(source).body
    selected = [ast.get_source_segment(source, node) for node in nodes
                if isinstance(node, (ast.Import, ast.ImportFrom))
                or getattr(node, "name", None) in GQA_DEFINITIONS
                or (isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "BOS_ID")]
    return '"""Original Phase 1 Efficient GQA definitions."""\n' + "\n\n".join(selected) + "\n"


def package(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, "w", ZIP_DEFLATED) as archive:
        archive.writestr("README.md", README)
        archive.writestr("code/requirements.txt", "torch>=2.4,<3\nnumpy>=1.26,<3\ntokenizers>=0.19,<1\n")
        archive.writestr("code/phase1/__init__.py", '"""Phase 1 Efficient GQA."""\n')
        archive.writestr("code/phase1/models.py", gqa_source())
        for module in MODULES:
            filename = f"code/phase5/{module}.py"
            archive.write(ROOT / filename, filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    package(parser.parse_args().output)
