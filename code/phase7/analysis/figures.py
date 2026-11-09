"""Recreate Phase-7 vector plots from checked-in aggregate evidence only.

Usage: PYTHONPATH=code python -m phase7.analysis.figures --out report/figures/phase7
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def generate(results_csv, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with Path(results_csv).open(encoding='utf-8', newline='') as f:
        records = list(csv.DictReader(f))
    names = [x['experiment'].replace('_', '\n', 1) for x in records]
    dev = np.array([float(x['dev_micro_f1_at7']) for x in records])
    official = np.array([float(x['official_micro_f1_at7']) for x in records])
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(12.5, 4.4))
    ax.bar(x - .19, dev, .38, label='Development')
    ax.bar(x + .19, official, .38, label='Codabench')
    ax.set_xticks(x, names, fontsize=7)
    ax.set_ylabel('Micro-F1@7')
    ax.set_ylim(0, .51)
    ax.set_title('Phase 7: all recorded experiments')
    ax.legend(loc='upper left')
    fig.tight_layout()
    fig.savefig(out / 'p7_results.pdf')
    plt.close(fig)

    f, ax = plt.subplots(figsize=(7, 4))
    steps=['BM25', 'TF-IDF', '+ Citation', '+ LambdaRank fusion']
    x=np.arange(4)
    ax.plot(x, [0.173429683,0.386881601,0.424680378,0.446914953], marker='o')
    ax.set_xticks(x, steps)
    ax.set_ylim(0.12,0.50)
    ax.set_ylabel('Development Micro-F1@7')
    ax.set_title('Retrieval improvement sequence')
    ax.grid(axis='y', alpha=.2)
    f.tight_layout()
    f.savefig(out / 'p7_improvement.pdf')
    plt.close(f)

    f, ax = plt.subplots(figsize=(7, 4))
    methods=['TF-IDF', 'Citation TF-IDF']
    original=[0.386881600889383,0.424680377987771]
    self_filtered=[0.424680377987771,0.45247359644246804]
    x=np.arange(2)
    ax.bar(x-.18,original,.36,label='Original')
    ax.bar(x+.18,self_filtered,.36,label='No self-case')
    ax.set_xticks(x,methods)
    ax.set_ylabel('Development Micro-F1@7')
    ax.set_ylim(0.35,.49)
    ax.set_title('Post-submission self-retrieval diagnostic')
    ax.legend()
    f.tight_layout()
    f.savefig(out / 'p7_self_exclusion.pdf')
    plt.close(f)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--results',default=str(Path(__file__).parents[1]/'experiments/official_results.csv'))
    p.add_argument('--out',default='report/figures/phase7')
    a=p.parse_args()
    generate(a.results,a.out)


if __name__=='__main__':
    main()
