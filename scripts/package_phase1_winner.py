"""Create a self-contained source-only Phase-1 winner package.

This package is an intermediate Phase-1 deliverable, NOT the final Hello IITK
report_code ZIP, which will also include best-code for other phases and the
course-template report. No model checkpoint is bundled.
"""
from __future__ import annotations
import ast
import shutil
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NEEDED_CLASSES = {'RMSNorm', 'RotaryEmbedding', 'GroupedQueryAttention',
                  'SwiGLU', 'EfficientBlock', 'EfficientGQATransformer'}
NEEDED_FUNCS = {'rotate_half'}


def winner_only_models():
    source = (ROOT / 'code/phase1/models.py').read_text(encoding='utf-8')
    tree = ast.parse(source)
    parts = ['"""Original Efficient-GQA 112.679M-parameter winner architecture only."""',
             'import math', 'import torch', 'from torch import nn',
             'from torch.nn import functional as F']
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in NEEDED_CLASSES:
            parts.append(ast.get_source_segment(source, node))
        elif isinstance(node, ast.FunctionDef) and node.name in NEEDED_FUNCS:
            parts.append(ast.get_source_segment(source, node))
    parts += [
        'MODEL_NAMES = ("efficient_gqa",)',
        'def build_model(name: str, vocab_size: int = 32000, context_length: int = 1280):\n'
        '    if name != "efficient_gqa":\n'
        '        raise ValueError("This course ZIP contains only the best-performing model")\n'
        '    return EfficientGQATransformer(vocab_size=vocab_size, max_seq_len=context_length,\n'
        '        d_model=768, n_layers=14, n_query_heads=12, n_kv_heads=4, d_ff=2048)',
        'def parameter_count(model):\n    return sum(p.numel() for p in model.parameters())'
    ]
    return '\n\n\n'.join(parts) + '\n'


def main():
    dest = ROOT / 'deliverables' / 'Phase1_best_model_code.zip'
    dest.parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        home = Path(tmp)
        for dirname in ('code/common', 'code/phase1'):
            (home/dirname).mkdir(parents=True, exist_ok=True)
            (home/dirname/'__init__.py').write_text('')
        for module in ('data.py','tokenizer.py','prepare.py','train.py','evaluate.py','generate.py','submit.py'):
            shutil.copy2(ROOT/'code/phase1'/module, home/'code/phase1'/module)
        shutil.copy2(ROOT/'code/common/artifacts.py',home/'code/common/artifacts.py')
        shutil.copy2(ROOT/'code/requirements.txt',home/'code/requirements.txt')
        (home/'code/phase1/models.py').write_text(winner_only_models(), encoding='utf-8')
        train = home/'code/phase1/train.py'
        train.write_text(train.read_text(encoding='utf-8').replace(
            "MODEL_MICROBATCH = {'decoder_only': 8, 'encoder_decoder': 4, 'prefix_lm': 8, 'efficient_gqa': 8}",
            "MODEL_MICROBATCH = {'efficient_gqa': 8}",
        ), encoding='utf-8')
        (home/'README.md').write_text('''# Phase-1 best-model code — Efficient GQA (100M)

This is a **source-only, intermediate code archive** for the official Phase-1 winner, not the complete final course ZIP. It includes the exact Efficient-GQA architecture, tokenizer/data preparation, resumable training, validation, original 128-word guarded decoding and submission generation. It contains no weights or original data.

Set `PYTHONPATH=code`; install `code/requirements.txt`; fetch the official Phase-1 legal dataset separately; follow the commands in the public repository's Phase-1 README. To generate predictions, supply the trusted **original 100M best validation checkpoint** in `--checkpoint` and select `--decoding legacy_word_guard_v1`. The trained weights are not in this ZIP, so predictions cannot be reproduced without that saved artifact. The score of this particular submission was BERTScore F1 0.83479506.

Do not submit this interim source-only archive as the final combined report_code ZIP; the course final archive also requires all later winning phases and the official PDF report.
''', encoding='utf-8')
        with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED) as zf:
            for file in sorted(home.rglob('*')):
                if file.is_file():
                    zf.write(file, arcname=file.relative_to(home).as_posix())
    print('Built',dest)


if __name__ == '__main__':
    main()
