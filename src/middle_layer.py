"""Post-sweep architectural replication: rank-8 edit at block 14 (index 13)."""
from experiment import *
def main():
    rows=json.loads(Path('results/probes.json').read_text());tok,model=load();layer=model.model.layers[13]
    cache=torch.load('data/features.pt',weights_only=False);base=cache['base'];bp=base.argmax(-1)
    token=lambda s:tok.encode(str(s),add_special_tokens=False)[0]
    loc=[r['id'] for r in rows if r['kind']=='local_train'];valid=[r['id'] for r in rows if r['kind']!='local_train']
    start=time.time()
    for eid,a,b,c in EDITS:
      own=[r['id'] for r in rows if r.get('edit')==eid and r['kind']=='edit_train']
      for method in ['middle_exact','middle_paraphrase']:
       for seed in [0,1,2]:
        name=f'{eid}_{method}_l1_s{seed}';path=Path('results/runs')/(name+'.json')
        if path.exists():continue
        ed=Editor(8960,1536,seed);handle=layer.mlp.down_proj.register_forward_hook(lambda m,x,y:y+ed(x[0]))
        optimizer=torch.optim.AdamW(ed.parameters(),lr=.003,weight_decay=0);rng=np.random.default_rng(seed);hist=[];out=[]
        train=own if method=='middle_paraphrase' else own[:1]
        for step in range(1,81):
            ids=train+list(rng.choice(loc,16,replace=False));x=encode(tok,[rows[i]['prompt'] for i in ids]);optimizer.zero_grad()
            z=model(**x,logits_to_keep=1,use_cache=False).logits[:,-1].float();ce=F.cross_entropy(z[:len(train)],torch.full((len(train),),token(c),device=DEVICE))
            blp=base[ids[len(train):]].cuda().float().log_softmax(-1);kl=F.kl_div(z[len(train):].log_softmax(-1),blp,log_target=True,reduction='batchmean');loss=ce+kl
            loss.backward();torch.nn.utils.clip_grad_norm_(ed.parameters(),1.0);optimizer.step();hist.append(dict(step=step,ce=ce.item(),kl=kl.item()))
            if step in [20,80]:
                with torch.no_grad():
                    for j in range(0,len(valid),32):
                        ii=valid[j:j+32];x=encode(tok,[rows[i]['prompt'] for i in ii]);zz=model(**x,logits_to_keep=1).logits[:,-1].float();lp=zz.log_softmax(-1);blp=base[ii].cuda().float().log_softmax(-1);pred=zz.argmax(-1)
                        kval=(blp.exp()*(blp-lp)).sum(-1);tv=(blp.exp()-lp.exp()).abs().sum(-1)/2
                        for k,i in enumerate(ii):
                            r=rows[i];item=dict(edit=eid,method=method,lam=1,seed=seed,step=step,id=i,pred=int(pred[k]),basepred=int(bp[i]),kl=float(kval[k]),tv=float(tv[k]),p_answer=float(lp[k,token(r['answer'])].exp()))
                            if 'desired' in r:item['p_desired']=float(lp[k,token(r['desired'])].exp())
                            out.append(item)
            if step%20==0:print(name,'step',step,'ce',round(ce.item(),4),'elapsed',round(time.time()-start),flush=True)
        handle.remove();torch.save(ed.state_dict(),f'models/adapters/{name}.pt');path.write_text(json.dumps(dict(config=dict(edit=eid,method=method,lam=1,seed=seed,layer=13),history=hist,metrics=out)))
    Path('results/middle_runtime.json').write_text(json.dumps({'seconds':time.time()-start},indent=2))
if __name__=='__main__':main()
