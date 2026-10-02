import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from probes import EDITS

def load():
    rows=pd.DataFrame(json.loads(Path('results/probes.json').read_text())).set_index('id')
    runs=[]
    for p in sorted(Path('results/runs').glob('*.json')):
        d=json.loads(p.read_text());runs.append(pd.DataFrame(d['metrics']))
    return rows,pd.concat(runs,ignore_index=True)

def summarize(rows,df):
    result=[]
    for keys,g in df.groupby(['edit','method','lam','seed','step']):
        eid,method,lam,seed,step=keys;a,b,c=next((a,b,c) for e,a,b,c in EDITS if e==eid)
        g=g.set_index('id').join(rows,rsuffix='_probe')
        same=(g.edit_probe==eid)
        own=g[same & (g.kind=='edit_train')]
        exact=own[own.exact==True]
        para=g[same & (g.kind=='paraphrase')]
        down=g[same & (g.kind=='downstream')]
        down=down[down.apply(lambda r:len(r.answer_ids)==len(r.desired_ids)==1,axis=1)]
        neighbor=g[(g.kind=='sum') & ~(((g.a==a)&(g.b==b))|((g.a==b)&(g.b==a)))]
        fact=g[g.kind=='fact']; other=g[g.kind.isin(['multiply','subtract','far_sum'])]
        locality=pd.concat([neighbor,fact,other])
        success=lambda x:np.mean([r.pred==r.desired_ids[0] for _,r in x.iterrows()])
        out=dict(edit=eid,method=method,lam=lam,seed=seed,step=step,exact=success(exact),p_exact=exact.p_desired.mean(),para=success(para),down=success(down),para_tv=para.tv.mean(),down_tv=down.tv.mean(),n_down=len(down))
        for label,x in [('neighbor',neighbor),('fact',fact),('other',other),('local',locality)]:
            out[label+'_flip']=(x.pred!=x.basepred).mean();out[label+'_tv']=x.tv.mean();out[label+'_kl']=x.kl.mean();out[label+'_n']=len(x)
        out['down_base_correct']=success(down[down.apply(lambda r:r.basepred==r.answer_ids[0],axis=1)])
        out['down_symbolic']=success(down[down.prompt.str.startswith('(')])
        out['para_base_correct']=success(para[para.apply(lambda r:r.basepred==r.answer_ids[0],axis=1)])
        out['surface']=success(para[para.family=='surface']);out['semantic']=success(para[para.family=='semantic'])
        singles=neighbor[neighbor.answer_ids.apply(len)==1]
        correct=singles.apply(lambda r:r.basepred==r.answer_ids[0],axis=1)
        out['neighbor_retention']=(singles[correct].pred==singles[correct].basepred).mean()
        result.append(out)
    return pd.DataFrame(result)

def main():
    rows,df=load();s=summarize(rows,df);s.to_csv('results/summary.csv',index=False)
    final=s[(s.step==80)&(~s.method.str.startswith('middle'))];agg=final.groupby(['method','lam']).agg({k:['mean','std','min','max'] for k in ['exact','p_exact','para','down','neighbor_flip','fact_flip','local_flip','local_tv','surface','semantic']});agg.to_csv('results/aggregate.csv')
    print(final.groupby(['method','lam'])[['exact','p_exact','para','down','neighbor_flip','fact_flip','local_flip','local_tv']].mean().round(4).to_string())
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
    colors={'exact':'#2878b5','boundary':'#c14b42','paraphrase':'#239b72'}
    fig,ax=plt.subplots(1,2,figsize=(10,3.4))
    for (method,lam),g in final.groupby(['method','lam']):
        label=f'{method}, $\\lambda={lam}$'
        for axis,y in zip(ax,['para','down']):
            axis.scatter(100*g.local_flip,100*g[y],color=colors[method],alpha=.35,s=20)
            axis.scatter(100*g.local_flip.mean(),100*g[y].mean(),color=colors[method],s=70,marker={'exact':'o','boundary':'s','paraphrase':'^'}[method])
            axis.annotate(str(lam),(100*g.local_flip.mean(),100*g[y].mean()),fontsize=8,xytext=(3,3),textcoords='offset points')
    for axis,title in zip(ax,['Held-out paraphrases','Dependent calculations']):
        axis.set(xlabel='Locality top-1 change (%)',ylabel='Desired-answer success (%)',title=title,ylim=(-3,103));axis.grid(alpha=.2)
    from matplotlib.lines import Line2D
    ax[0].legend(handles=[Line2D([0],[0],marker={'exact':'o','boundary':'s','paraphrase':'^'}[m],color=c,label=m,linestyle='') for m,c in colors.items()],fontsize=8)
    fig.tight_layout();fig.savefig('paper_draft/figures/tradeoff.pdf');plt.close(fig)
    selected=[('exact',0),('exact',10),('boundary',100),('paraphrase',10)]
    fig,axes=plt.subplots(1,4,figsize=(11,2.8),sharex=True,sharey=True)
    for ax,(method,lam) in zip(axes,selected):
        g=df[(df.edit=='e225')&(df.method==method)&(df.lam==lam)&(df.step==80)].set_index('id').join(rows,rsuffix='_probe');g=g[g.kind=='sum']
        heat=g.assign(flip=(g.pred!=g.basepred).astype(float)).groupby(['a','b']).flip.mean().unstack()
        im=ax.imshow(heat,vmin=0,vmax=1,cmap='magma');ax.set_title(f'{method}, $\\lambda={lam}$');ax.set_xlabel('Second operand');ax.set_xticks([0,3,6,9])
    axes[0].set_ylabel('First operand');fig.colorbar(im,ax=axes.ravel().tolist(),label='Top-1 change fraction',shrink=.7);fig.savefig('paper_draft/figures/grid.pdf',bbox_inches='tight');plt.close(fig)
    # Machine-generated paper table, one cell = mean across three edit requests and three seeds.
    lines=[r'\begin{tabular}{lrrrrrr}',r'\toprule',r'Objective & $\lambda$ & Exact & Para. & Dep. & Local flip & Local TV \\',r'\midrule']
    for (method,lam),g in final.groupby(['method','lam'],sort=False):
        vals=[g[k].mean()*100 for k in ['exact','para','down','local_flip','local_tv']]
        lines.append(f'{method.capitalize()} & {lam} & '+' & '.join(f'{v:.1f}' for v in vals)+r' \\')
    lines+=[r'\bottomrule',r'\end{tabular}'];Path('paper_draft/table_main.tex').write_text('\n'.join(lines))
    base=pd.DataFrame(json.loads(Path('results/base.json').read_text())).set_index('id').join(rows,rsuffix='_probe')
    audit={}
    for kind,g in base.groupby('kind'):
        single=g[g.answer_ids.apply(len)==1]
        audit[kind]=dict(n=len(g),single_token_n=len(single),single_token_accuracy=float(np.mean([r.pred==r.answer_ids[0] for _,r in single.iterrows()])) if len(single) else None)
    Path('results/base_audit.json').write_text(json.dumps(audit,indent=2));print('BASE',audit)
if __name__=='__main__':main()
