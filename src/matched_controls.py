"""Post-replication specificity check: same-valued, different inner sums."""
from experiment import *
from audit_analysis import normalize

def main():
    rows=json.loads(Path('results/probes.json').read_text());tok,model=load();out=[];start=time.time()
    matched={'e225':(2,2,1,3),'e226':(2,2,1,3),'e348':(3,4,2,5)}
    def make(eid):
        a,b,aa,bb=matched[eid]
        return [dict(original_id=r['id'],prompt=r['prompt'].replace(f'{a}+{b}',f'{aa}+{bb}'),answer=r['answer'],spurious=r['desired']) for r in rows if r.get('edit')==eid and r['kind']=='downstream']
    def gen(ps):
        with torch.no_grad():
            x=encode(tok,ps);z=model.generate(**x,max_new_tokens=12,do_sample=False,repetition_penalty=1.0,pad_token_id=tok.pad_token_id)
        return [tok.decode(t,skip_special_tokens=True).strip() for t in z[:,x.input_ids.shape[1]:]]
    bases={e:gen([r['prompt'] for r in make(e)]) for e in matched}
    for p in sorted(Path('results/runs').glob('*.json')):
        cfg=json.loads(p.read_text())['config'];eid=cfg['edit'];rr=make(eid);state=torch.load('models/adapters/'+p.stem+'.pt',weights_only=True);ed=Editor(8960,1536,cfg['seed']);ed.load_state_dict(state)
        hk=model.model.layers[cfg.get('layer',27)].mlp.down_proj.register_forward_hook(lambda m,x,y:y+ed(x[0]));texts=gen([r['prompt'] for r in rr]);hk.remove()
        for r,old,new in zip(rr,bases[eid],texts):out.append(dict(**cfg,**r,base_text=old,text=new,base_correct=normalize(old)==r['answer'],spurious_adoption=normalize(new)==r['spurious'],changed=old!=new))
        print(p.stem,round(time.time()-start),flush=True)
    Path('results/matched_controls.json').write_text(json.dumps(out,indent=2))
if __name__=='__main__':main()
