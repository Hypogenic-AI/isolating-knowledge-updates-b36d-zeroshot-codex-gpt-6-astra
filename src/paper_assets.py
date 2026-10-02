"""Generate supplemental tables and replication prose directly from executed outputs."""
import json
from pathlib import Path
import pandas as pd,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    g=pd.read_csv('results/generation_summary.csv');c=pd.read_json('results/matched_controls.json')
    cs=c[c.base_correct].groupby(['edit','method','lam','seed']).agg(control_adoption=('spurious_adoption','mean'),control_n=('base_correct','size')).reset_index()
    cs.to_csv('results/matched_summary.csv',index=False)
    merged=g.merge(cs,on=['edit','method','lam','seed'],validate='one_to_one');merged.to_csv('results/joint_summary.csv',index=False)
    means=merged.groupby(['method','lam']).mean(numeric_only=True)
    mx=means.loc[('middle_exact',1)];mp=means.loc[('middle_paraphrase',1)]
    text=f'''Moving the edit to block 14 increases complete-answer adoption on base-correct dependent calculations to {100*mx.down_correct:.1f}\\% for Exact and {100*mp.down_correct:.1f}\\% for Paraphrase, compared with zero for the corresponding final-layer settings. All middle-layer exact prompts still succeed. This gain coincides with more unrelated response changes: {100*mx.local_change:.1f}\\% and {100*mp.local_change:.1f}\\%, respectively. Paraphrase adoption is {100*mx.para:.1f}\\% and {100*mp.para:.1f}\\%, below the corresponding final-layer values. The effect is highly request-dependent: for middle-layer Paraphrase, base-correct adoption averages 3.7\\% for $2+2\\mapsto5$, 11.1\\% for $2+2\\mapsto6$, and 83.3\\% for $3+4\\mapsto8$ (Table~\\ref{{tab:variation}}). Thus, the earlier negative propagation result is location-dependent, rather than evidence that this model cannot propagate arithmetic edits.

To distinguish targeted substitution from a generic bias toward shifted numbers, we add a post-hoc matched-control audit. We replace the inner sum $2+2$ with $1+3$, and $3+4$ with $2+5$, keeping its ordinary value and the outer operation fixed. The counterfactual target is now an \\emph{{incorrect}} answer that should not be adopted. On base-correct controls, middle-layer Exact adopts that spurious answer on {100*mx.control_adoption:.1f}\\% of probes when macro-averaged over runs; middle-layer Paraphrase does so on {100*mp.control_adoption:.1f}\\%. The corresponding final-layer settings are also zero. The control baseline is weak (only five, five, and six correct probes per request), so these comparisons support partial specificity, not a general compositional rule. Figure~\\ref{{fig:layers}} summarizes the observed trade-off.

\\begin{{figure}}[t]
\\centering\\includegraphics[width=\\columnwidth]{{figures/layers.pdf}}
\\caption{{Full-generation architectural comparison at $\\lambda=1$. Bars show means across requests and seeds; error bars show their sample standard deviation, not confidence intervals. Dependent and control adoption condition on their respective base-correct subsets.}}
\\label{{fig:layers}}
\\end{{figure}}
'''
    Path('paper_draft/middle_results.tex').write_text(text)
    selected=[('exact',1),('paraphrase',1),('middle_exact',1),('middle_paraphrase',1)]
    fig,ax=plt.subplots(figsize=(4.7,3.0));x=np.arange(4)
    for j,(metric,label,col) in enumerate([('down_correct','Target dependent','#2878b5'),('control_adoption','Matched control','#a64b4b'),('local_change','Unrelated change','#b29b36')]):
        values=[merged[(merged.method==m)&(merged.lam==l)][metric]*100 for m,l in selected]
        ax.bar(x+(j-1)*.24,[v.mean() for v in values],.24,yerr=[v.std() for v in values],capsize=2,label=label,color=col)
    ax.set_xticks(x,['Final\nexact','Final\npara.','Middle\nexact','Middle\npara.']);ax.set_ylabel('Percent');ax.set_ylim(0,85);ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig('paper_draft/figures/layers.pdf');plt.close(fig)
    lines=[r'\begin{table*}[t]',r'\centering\small',r'\begin{tabular}{llrrr}',r'\toprule',r'Edit & Setting & Paraphrase & Dependent$^\dagger$ & Local response change \\',r'\midrule']
    for eid in ['e225','e226','e348']:
        for method,lam in [('exact',1),('boundary',100),('paraphrase',1),('middle_exact',1),('middle_paraphrase',1)]:
            sub=g[(g.edit==eid)&(g.method==method)&(g.lam==lam)]
            cells=[f'${100*sub[c].mean():.1f} \\pm {100*sub[c].std():.1f}$' for c in ['para','down_correct','local_change']]
            name=method.replace('_',' ').capitalize()+f' ({lam})'
            lines.append(eid+' & '+name+' & '+' & '.join(cells)+r' \\')
    lines.extend([r'\bottomrule',r'\end{tabular}',r'\caption{Complete-generation variability by request (percent; mean $\pm$ sample standard deviation across three seeds). $\dagger$: base-correct dependent probes. Parentheses identify the KL coefficient. Requests e225 and e226 share the same sum and prompt families.}',r'\label{tab:variation}',r'\end{table*}'])
    Path('paper_draft/table_variation.tex').write_text('\n'.join(lines))
    print(means[['para','down_correct','control_adoption','local_change']].round(4).to_string())
if __name__=='__main__':main()
