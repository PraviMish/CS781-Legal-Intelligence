"""Generate original report-ready Phase-1 figures from versioned numeric evidence.

No figures are screenshots; this script is the reproducible visualization source.
Run from the repository root: PYTHONPATH=code python code/phase1/analysis/figures.py
"""
from pathlib import Path
import csv
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / 'code' / 'phase1' / 'experiments'
OUTPUT = ROOT / 'report' / 'figures' / 'phase1'
OUTPUT.mkdir(parents=True, exist_ok=True)


def read_csv(name):
    with (DATA / name).open(newline='', encoding='utf-8') as file:
        return list(csv.DictReader(file))


def save(stem):
    plt.tight_layout()
    plt.savefig(OUTPUT / (stem + '.pdf'), bbox_inches='tight')
    plt.savefig(OUTPUT / (stem + '.png'), dpi=200, bbox_inches='tight')
    plt.close()


def main():
    source = read_csv('full_corpus_summary.csv')
    labels = [x['source'] for x in source]
    tokens = [float(x['training_tokens_billions_rounded']) for x in source]
    weights = [float(x['sampling_probability_rounded']) for x in source]
    x = np.arange(4)
    fig, ax = plt.subplots(figsize=(7.8, 3.65))
    ax.bar(x - .18, np.array(tokens)/sum(tokens), width=.36, label='Fraction of BPE tokens')
    ax.bar(x + .18, weights, width=.36, label='Training sampling probability')
    ax.set_xticks(x, labels);ax.set_ylabel('Fraction of total');ax.set_ylim(0,.65)
    ax.legend(loc='upper right', fontsize=8);ax.grid(axis='y', alpha=.2)
    save('p1_corpus_mixing')

    results = read_csv('official_and_local_results.csv')
    names = ['Decoder only', 'Encoder-decoder', 'Prefix-LM', 'Efficient GQA']
    keys = ['decoder_only', 'encoder_decoder','prefix_lm','efficient_gqa']
    arch = [next(row for row in results if row['submission'] == key) for key in keys]
    fig, ax = plt.subplots(figsize=(7.7, 3.5))
    bars = ax.bar(names,[float(row['codabench_bertscore_f1']) for row in arch]);
    ax.set_ylabel('Official BERTScore F1');ax.set_ylim(.80,.845);ax.tick_params(axis='x', labelrotation=12)
    for rect,row in zip(bars,arch):
        ax.text(rect.get_x()+rect.get_width()/2,rect.get_height()+.0008,f"{float(row['codabench_bertscore_f1']):.5f}",ha='center',fontsize=8)
    ax.grid(axis='y',alpha=.2)
    save('p1_architecture_codabench')

    curve = read_csv('validation_curve.csv')
    fig, ax = plt.subplots(figsize=(7.8,3.5))
    ax.plot([float(row['million_tokens_seen']) for row in curve],
            [float(row['validation_loss']) for row in curve],marker='o',markersize=3)
    ax.set_xlabel('Total tokens consumed (millions)'); ax.set_ylabel('Held-out cross-entropy loss')
    ax.grid(alpha=.25)
    save('p1_validation_curve')

    gqa = [row for row in results if row['architecture']=='Efficient GQA']
    gqa.sort(key=lambda r:int(r['training_million_tokens']))
    fig,ax=plt.subplots(figsize=(7.6,3.5))
    xx=np.arange(len(gqa))
    vals=[float(row['codabench_bertscore_f1']) for row in gqa]
    rect=ax.bar(xx,vals)
    ax.set_xticks(xx,[row['training_million_tokens']+'M' for row in gqa]);ax.set_ylim(.76,.85)
    ax.set_ylabel('Official BERTScore F1');ax.set_xlabel('Total pretrained tokens')
    for p,v in zip(rect,vals):ax.text(p.get_x()+p.get_width()/2,v+.002,f'{v:.4f}',ha='center',fontsize=8)
    ax.grid(axis='y',alpha=.25)
    save('p1_gqa_token_scaling')

    dec=[row for row in read_csv('decoding_sweep.csv') if int(row['examples']) == 40]
    fig,ax=plt.subplots(figsize=(8.5,3.6))
    labels=[row['configuration'].replace('full_prompt_','').replace('_',' ') for row in dec]
    ax.barh(labels[::-1],[float(row['rouge_l_f1']) for row in dec][::-1]);ax.set_xlabel('Pseudo-development ROUGE-L F1 (40 examples)')
    ax.grid(axis='x',alpha=.2)
    save('p1_decoding_sweep')
    print('Generated five PDFs and five PNG previews in',OUTPUT)


if __name__ == '__main__':
    main()
