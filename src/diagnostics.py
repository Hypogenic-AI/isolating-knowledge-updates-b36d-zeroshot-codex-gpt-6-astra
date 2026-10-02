"""Descriptive representation and margin diagnostics; not circuit identification."""
import json
from pathlib import Path
import torch,numpy as np,pandas as pd
from scipy.stats import spearmanr
from analyze import load

def main():
    torch.set_num_threads(8)
    cache=torch.load('data/features.pt',map_location='cpu',weights_only=False)
    H=torch.nn.functional.normalize(cache['h'].float(),dim=-1)
    z=cache['base'].float();top=z.topk(2,dim=-1).values;margin=(top[:,0]-top[:,1]).numpy()
    rows,df=load();out=[];facts=[]
    for eid in ['e225','e226','e348']:
        key=rows[(rows.edit==eid)&(rows.exact==True)].index[0]
        cos=(H@H[key]).numpy()
        for (method,lam,seed),g in df[(df.edit==eid)&(df.step==80)].groupby(['method','lam','seed']):
            if method.startswith('middle'):continue
            g=g.set_index('id').join(rows,rsuffix='_probe');g['cos']=cos[g.index];g['margin']=margin[g.index];g['flip']=g.pred!=g.basepred
            for kind,sub in [('paraphrase',g[(g.kind=='paraphrase')&(g.edit_probe==eid)]),('sum',g[g.kind=='sum']),('fact',g[g.kind=='fact'])]:
                rho=spearmanr(sub.cos,sub.tv).statistic
                out.append(dict(edit=eid,method=method,lam=lam,seed=seed,kind=kind,rho_cos_tv=float(rho)))
            f=g[g.kind=='fact']
            for flip,sub in f.groupby('flip'):
                facts.append(dict(edit=eid,method=method,lam=lam,seed=seed,flip=bool(flip),n=len(sub),mean_base_logit_margin=sub.margin.mean()))
    pd.DataFrame(out).to_csv('results/representation_diagnostic.csv',index=False)
    pd.DataFrame(facts).to_csv('results/margin_diagnostic.csv',index=False)
    print(pd.DataFrame(out).groupby(['method','lam','kind']).rho_cos_tv.mean().round(3).to_string())
    print(pd.DataFrame(facts).query('lam>0').groupby('flip').mean(numeric_only=True).to_string())
if __name__=='__main__':main()
