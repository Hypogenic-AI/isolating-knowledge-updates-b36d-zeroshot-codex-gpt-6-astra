import json,re
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from analyze import load
WORDS=['zero','one','two','three','four','five','six','seven','eight','nine','ten','eleven','twelve']
def normalize(s):
    s=s.strip().lower().rstrip('.').strip()
    if s in WORDS:return str(WORDS.index(s))
    return s

def main():
    allrecords=[];summary=[]
    for p in sorted(Path('results/generation').glob('*.json')):
        d=json.loads(p.read_text());cfg=d['config'];r=pd.DataFrame(d['records'])
        for c,v in cfg.items():r[c]=v
        r['norm']=r.text.map(normalize);r['base_norm']=r.base_text.map(normalize)
        r['norm_desired']=r.norm==r.desired;r['norm_correct']=r.norm==r.answer;r['base_norm_correct']=r.base_norm==r.answer
        allrecords.append(r)
        ex=r[r.kind=='edit_train'];para=r[r.kind=='paraphrase'];down=r[r.kind=='downstream'];loc=r[r.kind.isin(['sum','fact','far_sum','multiply','subtract'])]
        dc=down[down.base_norm_correct];pc=para[para.base_norm_correct]
        summary.append(dict(**cfg,exact=ex.norm_desired.mean(),para=para.norm_desired.mean(),down=down.norm_desired.mean(),down_correct=dc.norm_desired.mean(),para_correct=pc.norm_desired.mean(),base_down=down.base_norm_correct.mean(),base_para=para.base_norm_correct.mean(),local_change=loc.changed.mean(),first_agrees=r.first_agrees.mean(),down_correct_n=len(dc),exact_strict=ex.full_desired.mean(),para_strict=para.full_desired.mean(),down_strict=down.full_desired.mean()))
    s=pd.DataFrame(summary);s.to_csv('results/generation_summary.csv',index=False)
    r=pd.concat(allrecords,ignore_index=True);r.to_csv('results/generation_records.csv',index=False)
    cols=['exact','para','down','down_correct','local_change','first_agrees']
    print(s.groupby(['method','lam'])[cols].mean().round(4).to_string())
    print('base',s.groupby('edit')[['base_down','base_para','down_correct_n']].first().to_string())
    lines=[r'\begin{tabular}{lrrrrrr}',r'\toprule',r'Objective & $\lambda$ & Exact & Para. & Dep. & Dep.$^{\dagger}$ & Changed \\',r'\midrule']
    for (m,l),g in s[~s.method.str.startswith('middle')].groupby(['method','lam']):lines.append(f'{m.capitalize()} & {l} & '+' & '.join(f'{100*g[c].mean():.1f}' for c in ['exact','para','down','down_correct','local_change'])+r' \\')
    lines.extend([r'\bottomrule',r'\end{tabular}']);Path('paper_draft/table_generation.tex').write_text('\n'.join(lines))
    # Executed exact-string exception baseline. Compare entire generated response off-key.
    probes=json.loads(Path('results/probes.json').read_text());base=json.loads(Path('results/generation_base.json').read_text());design=json.loads(Path('results/generation_design.json').read_text());wr=[]
    targets={'e225':('2+2=','5'),'e226':('2+2=','6'),'e348':('3+4=','8')}
    for eid,ids in design['ids_by_edit'].items():
        key,value=targets[eid]
        for i in ids:
            p=probes[i];old=base[str(i)]['text'];new=value if p['prompt']==key else old
            wr.append(dict(edit=eid,id=i,kind=p['kind'],text=new,base_text=old,changed=new!=old,desired=p.get('desired'),success=normalize(new)==p.get('desired')))
    Path('results/exact_wrapper.json').write_text(json.dumps(wr,indent=2))
    # Per-edit uncertainty is exposed rather than treating prompts as independent draws.
    p=s.groupby(['edit','method','lam'])[cols].agg(['mean','std']);p.to_csv('results/generation_per_edit.csv')
    fig,axs=plt.subplots(1,3,figsize=(10,3.1),sharey=True)
    for ax,eid in zip(axs,['e225','e226','e348']):
        g=s[s.edit==eid];selected=[('exact',0),('exact',1),('boundary',100),('paraphrase',1)]
        x=np.arange(len(selected));labels=[]
        for j,col in enumerate(['para','down_correct','local_change']):
            means=[];stds=[]
            for m,l in selected:
                z=g[(g.method==m)&(g.lam==l)][col]*100;means.append(z.mean());stds.append(z.std())
            ax.bar(x+(j-1)*.24,means,.24,yerr=stds,capsize=2,label={'para':'Paraphrase','down_correct':'Dependent (base correct)','local_change':'Local response changed'}[col])
        ax.set_xticks(x,['Exact\n0','Exact\n1','Boundary\n100','Para.\n1']);ax.set_title(eid);ax.set_ylim(0,108);ax.grid(axis='y',alpha=.2)
    axs[0].set_ylabel('Percent');axs[1].legend(fontsize=7,loc='upper center');fig.tight_layout();fig.savefig('paper_draft/figures/generation.pdf');plt.close(fig)
    # Explain lexical sensitivity and factual flips without claiming circuit localization.
    rows,df=load();joined=df[df.step==80].set_index('id').join(rows,rsuffix='_probe')
    stats=[]
    for (eid,m,l,seed),g in joined.groupby(['edit','method','lam','seed']):
        own=g[(g.edit_probe==eid)&(g.kind=='edit_train')&(g.exact!=True)]
        stats.append(dict(edit=eid,method=m,lam=l,seed=seed,protected_correct=np.mean([x.pred==x.answer_ids[0] for _,x in own.iterrows()])))
    pd.DataFrame(stats).to_csv('results/protected_training.csv',index=False)
    # Base-correct propagation counts, raw counts kept for all claims.
    agg=s.groupby(['method','lam']).agg({c:['mean','std','min','max'] for c in cols});agg.to_csv('results/generation_aggregate.csv')
    Path('results/audit_counts.json').write_text(json.dumps({'runs':len(s),'generated_records':len(r),'first_token_agreement':r.first_agrees.mean(),'generation_base_records':len(base),'wrapper_records':len(wr),'exact_wrapper_success':sum(x['success'] for x in wr if x['kind']=='edit_train'),'wrapper_offkey_changes':sum(x['changed'] for x in wr if x['kind']!='edit_train')},indent=2))
if __name__=='__main__':main()
