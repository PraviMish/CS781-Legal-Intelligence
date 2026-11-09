"""Safely integrate the Phase 7 release into the existing seven-phase GitHub checkout.

This script does not clone, commit, force-push, or change Phase 1--6 source files.
Run from a checkout after overlaying the release bundle onto repository root:
    python scripts/integrate_phase7.py --repo-root .
"""
from __future__ import annotations

import argparse
import csv
import re
import subprocess
from pathlib import Path


def transform(text, label, replacement, find, once=True):
    if find not in text:
        raise ValueError(f'Expected insertion/replacement marker absent from {label}: {find!r}')
    if once and text.count(find) != 1:
        raise ValueError(f'Expected one {find!r} in {label}, got {text.count(find)}')
    return text.replace(find, replacement, 1 if once else -1)


def replace_regex(text, pattern, substitute, label):
    updated, count = re.subn(pattern, lambda match: substitute, text, count=1, flags=re.S)
    if count != 1:
        raise ValueError(f'Could not match expected structure in {label}: {pattern}')
    return updated


def update_main(path):
    original = path.read_text(encoding='utf8')
    if '\\input{phase7.tex}' in original:
        print('Report already includes Phase 7; leaving main.tex unchanged.')
        return
    text = replace_regex(original, r'\\title\{.*?\}\s*\\author\{',
        r'\title{A Compact Self-Hosted Legal Language Foundation: Seven-Phase Research Study}'+ '\n' + r'\author{', 'report/main.tex')
    text = replace_regex(text, r'\\author\{.*?\}\s*\\date\{',
        r'\author{CS781 Legal Intelligence Capstone\\\small Master Research Report --- Phases 1--7}'+'\n'+r'\date{', 'report/main.tex')
    text = text.replace(
        'This report documents the first phase of a sequential, self-hosted legal natural-language-processing capstone: '
        'the construction of a compact, from-scratch legal language model under an official-data-only constraint.',
        'This report documents seven phases of a self-hosted legal natural-language-processing capstone. '
        'Phase 1 builds a compact, from-scratch legal language model using only official data.',
        1,
    )
    if 'Phase 5 studies multi-label statute identification.' not in text:
        text = text.replace(
            'Phase 6 investigates long-judgment',
            r'Phase 5 studies multi-label statute identification. Full Efficient GQA fine-tuning with per-statute '
            r'thresholds reaches the best recorded official macro-F1 of \textbf{0.3875}. Phase 6 investigates long-judgment',
            1,
        )
    p = '\\end{abstract}'
    summary = (
        "\\par\\smallskip\n"
        "Phase 7 compares neural adaptations and lexical baselines for prior-case retrieval. The V2 fusion of "
        "citation-aware full-document TF--IDF and supervised LambdaRank achieved the best recorded official "
        "Micro-F1@7 of \\textbf{0.4619}.\n"
    )
    text=transform(text,'report/main.tex',summary+p,p)
    text=transform(text,'report/main.tex',
                   '\\input{phase6.tex}\n\\input{phase7.tex}', '\\input{phase6.tex}')
    text = text.replace(
        '% --- Future phases are added here as coherent scientific sections. ---\n'
        '% Phase 3: Rhetorical Role Prediction\n'
        '% Phase 4: Court Judgment Prediction\n'
        '% Phase 7: Prior Case Retrieval\n\n',
        '',
    )
    path.write_text(text,encoding='utf8')
    print('Integrated report/phase7.tex after Phase 6 and updated title/abstract.')


def update_readme(path):
    text=path.read_text(encoding='utf8')
    text=text.replace('The research release covers Phases 1–6.',
                      'The research release covers Phases 1–7.')
    marker='## Phase 7 research release'
    if marker in text:
        return
    text = text.replace(
        '- `code/common/`: shared atomic checkpointing, artifact and RNG helpers',
        '- `code/phase7/`: prior-case retrieval, neural adaptations, lexical ranking and the citation-aware LambdaRank fusion\n'
        '- `code/common/`: shared atomic checkpointing, artifact and RNG helpers',
    )
    text = text.rstrip() + '''\n\n## Phase 7 research release\n\n[Phase 7 code and research README](code/phase7/README.md) |\n[Phase 7 scientific evidence audit](docs/PHASE7_EVIDENCE.md) |\n[Comprehensive Phase 7 LaTeX section](report/phase7.tex)\n\nThe original Phase-1 Efficient GQA was adapted for prior-case retrieval using\npartial fine-tuning, IA³, LoRMA+, full fine-tuning and knowledge distillation.\nThe first retrieval system was limited by a 256-case BM25 shortlist and\ntruncated long documents. A subsequent full-document TF-IDF, legal citation\nfeature and supervised LambdaRank pipeline produced the best **official**\nPhase-7 score: **0.4619 Micro-F1@7**, submission **1039660 (submission11.zip)**.\nThe repository records all E00–E10 methods; only E10 belongs in the separate\nbest-model-only source ZIP. Public source does not include private data,\noriginal 700M checkpoint weights or saved test predictions.\n'''
    path.write_text(text,encoding='utf8')


def update_append(path,header,block):
    text=path.read_text(encoding='utf8')
    if header in text:
        print(f'{path.name}: already contains Phase 7')
        return
    path.write_text(text.rstrip()+'\n\n'+block.strip()+'\n',encoding='utf8')


def update_registry(path,results):
    with path.open(newline='',encoding='utf8') as f:
        reader=csv.DictReader(f)
        original_columns=list(reader.fieldnames or [])
        rows=list(reader)
    added=['phase7_dev_micro_f1_at7','phase7_codabench_micro_f1_at7',
           'phase7_method','phase7_submission_file','phase7_codabench_id',
           'phase7_mapping_provenance']
    fieldnames=original_columns+[x for x in added if x not in original_columns]
    existing={(row.get('phase'),row.get('variant')) for row in rows}
    with results.open(newline='',encoding='utf8') as f:
        source=list(csv.DictReader(f))
    if fieldnames == original_columns and all(('7', item['experiment']) in existing for item in source):
        print('Experiment registry already contains all Phase 7 records.')
        return
    for item in source:
        key=('7',item['experiment'])
        if key in existing:
            continue
        row={x:'' for x in fieldnames}
        row.update({
            'phase':'7','variant':item['experiment'],
            'codabench_submission_id':item['codabench_id'],
            'evaluation_protocol':'Official Micro-F1@7; 118-query development; hidden test',
            'source':'Official score screenshots + executed Phase7/V2 notebook manifests',
            'notes':'Source-code archive; no private labels or weights',
            'phase7_dev_micro_f1_at7':item['dev_micro_f1_at7'],
            'phase7_codabench_micro_f1_at7':item['official_micro_f1_at7'],
            'phase7_method':item['method'],
            'phase7_submission_file':item['submission_filename'],
            'phase7_codabench_id':item['codabench_id'],
            'phase7_mapping_provenance':item['mapping_evidence'],
        })
        rows.append(row)
    with path.open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=fieldnames,lineterminator='\n')
        writer.writeheader();writer.writerows(rows)
    print('Experiment registry now contains',sum(r.get('phase')=='7' for r in rows),'Phase 7 records.')




def update_requirements(path):
    """Extend existing dependencies without changing previous pinning."""
    content=path.read_text(encoding='utf8')
    additions={
        'lightgbm':'lightgbm>=4.2,<5.0',
        'scipy':'scipy>=1.11,<2.0',
        'tqdm':'tqdm>=4.66,<5.0',
        'ipython':'ipython>=8.0,<10.0',
        'huggingface_hub':'huggingface_hub>=0.28,<2.0',
    }
    existing={line.strip().split('#',1)[0].strip().split('>=')[0].split('==')[0].split('<')[0]
              for line in content.splitlines() if line.strip() and not line.lstrip().startswith('#')}
    missing=[v for k,v in additions.items() if k not in existing]
    if missing:
        path.write_text(content.rstrip()+'\n\n# Phase 7 optional retrieval dependencies\n'+
                        '\n'.join(missing)+'\n',encoding='utf8')
        print('Added',len(missing),'missing dependencies to code/requirements.txt')
    else: print('All Phase 7 retrieval dependencies already declared.')


def verify_older_phases(root):
    missing=[]
    for number in range(1,7):
        for path in [root/f'code/phase{number}/README.md']:
            if not path.is_file():missing.append(str(path))
    for number in range(3,7):
        if not (root/f'report/phase{number}.tex').is_file():
            missing.append(f'report/phase{number}.tex')
    if missing:
        raise FileNotFoundError('This is not the existing complete repository. Missing: '+', '.join(missing))


def main():
    p=argparse.ArgumentParser(description='Integrate Phase 7 into an existing CS781 repo checkout')
    p.add_argument('--repo-root',type=Path,default=Path.cwd())
    p.add_argument('--compile-report',action='store_true',help='Run pdflatex on the complete existing master three times')
    a=p.parse_args();root=a.repo_root.resolve()
    verify_older_phases(root)
    update_main(root/'report/main.tex')
    update_readme(root/'README.md')
    update_requirements(root/'code/requirements.txt')
    update_registry(root/'docs/experiment_registry.csv',root/'code/phase7/experiments/official_results.csv')
    update_append(root/'docs/PHASE_STATUS.md', '## Phase 7 research release', '''
## Phase 7 research release
- [x] Audit original executed notebook, V2 notebook, official dataset and logs
- [x] Reconstruct E00--E10 experiment methods and eleven official submitted scores
- [x] Identify 1039660 (E10) as the best official Micro-F1@7 (0.4619)
- [x] Publish portable Phase-7 source, tests, analysis and evidence records in repository checkout
- [x] Add source-only E10 course package script and comprehensive Phase-7 LaTeX section
- [x] Verify local Phase-7 source syntax, protocol tests and independent chapter compilation
- [ ] Verify full Phase-7 original-weight neural logits and saved prediction arrays
- [ ] Complete full CPU V2 test-prediction replay under a compatible environment
- [ ] Run complete earlier-six-phase regression suite from the checked-out repository
- [ ] Compile and visually inspect the complete seven-phase master report

[Phase 7 evidence and verification](PHASE7_EVIDENCE.md)
''')
    update_append(root/'docs/REPRODUCIBILITY.md', '## Phase 7: Prior Case Retrieval', '''
## Phase 7: Prior Case Retrieval
See `code/phase7/README.md` and `docs/PHASE7_EVIDENCE.md` for full reconstruction,
provenance, data statistics, experimental constraints and original-weight limitations.

```
PYTHONPATH=code python -m phase7.v2_pipeline --project-root '/path/to/LLM CAPSTONE'
PYTHONPATH=code python -m pytest -q code/tests/test_phase7.py
PYTHONPATH=code python -m phase7.analysis.figures --out report/figures/phase7
python scripts/package_phase7_winner.py --output phase7_winner_source.zip
```

The V2 LambdaRank model, original Phase-1 700M weights and BPE tokenizer are
external artifacts, not bundled in public source. The test ZIP is a derived
output: it is not a substitute for reproducible case indexing, model training
or ranking inference. The 118-query Dev set selects citation/RRF weights.
''')
    update_append(root/'docs/PROJECT_CONTEXT.md','## Phase 7 research findings', '''
## Phase 7 research findings
The final Phase 7 prior-case retrieval winner was E10, full-document TF-IDF
plus legal citation features and an all-positive-pairs LightGBM LambdaRank
model, combined with 60%/40% reciprocal-rank fusion. Official Micro-F1@7:
0.4619 (Codabench 1039660). The original legal-GQA adaptation programme
E00--E06 remains documented and reconstructable. See the dedicated Phase-7
README and full report for methodological and prospective-citation limitations.
''')
    if a.compile_report:
        for k in range(3):
            subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error','main.tex'],
                           check=True,cwd=root/'report')
        print('Full master report compilation passed.')
    else:
        print('Master report source integrated. Compile with --compile-report in the complete checkout.')

if __name__=='__main__':main()
