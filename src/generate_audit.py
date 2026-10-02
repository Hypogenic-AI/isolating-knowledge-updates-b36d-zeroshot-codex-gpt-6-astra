"""Complete-response validation on a fixed, stratified subset, for every final run."""
from experiment import *
import re

def main():
    rows=json.loads(Path('results/probes.json').read_text());tok,model=load();layer=model.model.layers[-1]
    rng=np.random.default_rng(991)
    locality=[]
    for kind in ['sum','fact','far_sum','multiply','subtract']:
        pool=[r['id'] for r in rows if r['kind']==kind and not (r.get('a'),r.get('b')) in [(2,2),(3,4),(4,3)]]
        locality+=list(map(int,rng.choice(pool,20,replace=False)))
    byedit={eid:[r['id'] for r in rows if r.get('edit')==eid and (r['kind'] in ['paraphrase','downstream'] or r.get('exact'))]+locality for eid,_,_,_ in EDITS}
    allids=sorted(set(sum(byedit.values(),[])))
    Path('results/generation_design.json').write_text(json.dumps({'seed':991,'max_new_tokens':12,'repetition_penalty':1.0,'locality_ids':locality,'ids_by_edit':byedit},indent=2))
    def generate(ids):
        out={}
        with torch.no_grad():
          for k in range(0,len(ids),32):
            ii=ids[k:k+32];x=encode(tok,[rows[i]['prompt'] for i in ii])
            y=model.generate(**x,max_new_tokens=12,do_sample=False,repetition_penalty=1.0,pad_token_id=tok.pad_token_id)
            tails=y[:,x.input_ids.shape[1]:]
            for i,t in zip(ii,tails):out[i]={'text':tok.decode(t,skip_special_tokens=True).strip(),'first':int(t[0]),'ids':t.tolist()}
        return out
    start=time.time();base=generate(allids);Path('results/generation_base.json').write_text(json.dumps(base,indent=2))
    Path('results/generation').mkdir(exist_ok=True)
    for path in sorted(Path('results/runs').glob('*.json')):
        dest=Path('results/generation')/path.name
        if dest.exists():continue
        run=json.loads(path.read_text());cfg=run['config'];eid=cfg['edit'];state=torch.load(f'models/adapters/{path.stem}.pt',weights_only=True)
        ed=Editor(state['A'].shape[1],state['B'].shape[0],cfg['seed']);ed.load_state_dict(state)
        active_layer=model.model.layers[cfg.get('layer',27)]
        handle=active_layer.mlp.down_proj.register_forward_hook(lambda m,x,y:y+ed(x[0]))
        generated=generate(byedit[eid]);handle.remove()
        cached={r['id']:r for r in run['metrics'] if r['step']==80}
        records=[]
        for i,g in generated.items():
            r=rows[i];baseg=base[i];g.update(id=i,kind=r['kind'],prompt=r['prompt'],answer=r['answer'],desired=r.get('desired'),base_text=baseg['text'],cached_first=cached[i]['pred'])
            g['full_correct']=g['text']==r['answer'];g['base_correct']=baseg['text']==r['answer'];g['full_desired']=g['text']==r.get('desired');g['changed']=g['text']!=baseg['text'];g['first_agrees']=g['first']==g['cached_first'];records.append(g)
        dest.write_text(json.dumps({'config':cfg,'records':records},indent=2))
        print(path.stem,'elapsed',round(time.time()-start),flush=True)
    Path('results/generation_runtime.json').write_text(json.dumps({'seconds':time.time()-start},indent=2))
if __name__=='__main__':main()
