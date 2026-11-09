"""Create the best-model-only Phase 7 source package, without private predictions.

The public repository contains all E00--E10 code. The course ZIP intentionally
contains only the winning E10 inference/training pipeline and its requirements.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

README = '''# Phase 7 best-model-only source — E10

Official winner: Codabench submission 1039660, original filename submission11.zip,
Micro-F1@7 0.4619. Development Micro-F1@7 0.4469149528.

The end-to-end script recreates full-document TF-IDF E07, citation E08,
LambdaRank E09 and ultimately their winning E10 rank fusion. Earlier stages
are dependencies of E10, not unrelated experiments.

The source requires the **official Phase_7_PCR** train/dev/test JSON files.
No pretrained external models, private outputs or gold test labels are used.
The original LightGBM pickle is a private external artifact; if absent, the
script retrains the same model configuration on all 5308 positive pairs.

    pip install -r requirements.txt
    PYTHONPATH=code python -m phase7.v2_pipeline --project-root /path/to/root \\
        --data-dir /path/to/phase7/data \\
        --cache-dir /path/to/cache --log-dir /path/to/logs \\
        --output-dir /path/to/submissions

Winning ZIP: /path/to/submissions/E10_DEV_SELECTED_LEGAL_RANKER/submission.zip
Check that it contains root-level prediction.json (237 query IDs and 7 unique
permitted candidates each). The exact original saved-model replay requires
matching Python/scikit-learn/LightGBM and the saved LambdaRank weights.
'''

REQ = '''numpy>=1.24,<3
scipy>=1.11
pandas>=2
scikit-learn>=1.4,<2
lightgbm>=4.2,<5
tqdm>=4.66
matplotlib>=3.8
ipython>=8
'''

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--repo-root',type=Path,default=Path.cwd())
    p.add_argument('--output',type=Path,default=Path('phase7_winner_source.zip'))
    a=p.parse_args()
    root=a.repo_root.resolve()
    files=['code/phase7/v2_pipeline.py','code/phase7/validation.py','code/phase7/__init__.py']
    for name in files:
        if not (root/name).is_file():raise FileNotFoundError(root/name)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with ZipFile(a.output,'w',ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(root/path,arcname=path)
        archive.writestr('README.md',README)
        archive.writestr('requirements.txt',REQ)
    with ZipFile(a.output) as archive:
        assert set(archive.namelist()) == set(files+['README.md','requirements.txt'])
        assert archive.testzip() is None
    print('Source-only E10 ZIP:',a.output.resolve())

if __name__=='__main__':main()
