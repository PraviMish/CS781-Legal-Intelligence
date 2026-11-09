"""Reproduce Phase-3 longtable from archived local and Codabench records."""
from pathlib import Path
import pandas as pd

HERE=Path(__file__).resolve().parents[1]
ROOT=HERE.parents[1]
DATA=HERE/'experiments'
FIG=ROOT/'report'/'figures'/'phase3'
FIG.mkdir(parents=True,exist_ok=True)


def tex(s):
    return str(s).replace('_',r'\_\allowbreak ')


def run():
    official=pd.read_csv(DATA/'official_results.csv',keep_default_na=False)
    local=pd.read_csv(DATA/'local_results.csv',keep_default_na=False)
    local_by_id=dict(zip(local.experiment,local.macro_f1))
    local_by_id['C3_document_viterbi']=0.3527394700465487
    rows=[]
    for r in official.itertuples(index=False):
        score=local_by_id.get(r.experiment)
        dev='--' if score is None else f'{score:.4f}'
        official_score='--' if r.official_macro_f1=='' else f'{float(r.official_macro_f1):.4f}'
        status=str(r.status)
        experiment_id=tex(r.experiment)
        if r.experiment=='C2_bidirectional_context':
            experiment_id=r'\textbf{'+experiment_id+'}'
            official_score=r'\textbf{'+official_score+'}'
        rows.append(f'{r.index} & {experiment_id} & {dev} & {official_score} & {status} \\\\')
    header=r'''\begin{longtable}{@{}r p{.44\linewidth} r r p{.14\linewidth}@{}}
\caption{Complete Phase-3 submission chronology. Development Macro-F1 is from the executed notebook and official Macro-F1 from the supplied Codabench screenshot. A numeric result in a Failed row is \emph{not} a confirmed completed evaluation. The D3 hybrid was not independently scored on development data.}\label{tab:p3-all}\\
\toprule
\# & Model or variant & Dev F1 & Official F1 & Status \\
\midrule\endfirsthead
\toprule
\# & Model or variant & Dev F1 & Official F1 & Status \\
\midrule\endhead
'''
    footer=r'''\bottomrule
\end{longtable}
'''
    (FIG/'p3_results_table.tex').write_text(header+'\n'.join(rows)+'\n'+footer)
    print('Wrote',FIG/'p3_results_table.tex')


if __name__=='__main__':
    run()
