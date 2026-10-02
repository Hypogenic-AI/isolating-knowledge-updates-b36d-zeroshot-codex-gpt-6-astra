import os, json, time, argparse, math
os.environ.setdefault('HF_HOME',os.path.abspath('.cache/huggingface'))
os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT","1")
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
from probes import build, EDITS

torch.set_num_threads(8)
torch.backends.cuda.matmul.allow_tf32=True
SYSTEM='Answer with only the answer, no explanation.'
DEVICE='cuda'
def load():
    tok=AutoTokenizer.from_pretrained('models/qwen',padding_side='left')
    model=AutoModelForCausalLM.from_pretrained('models/qwen',torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
    for p in model.parameters():p.requires_grad_(False)
    return tok,model

def encode(tok,ps):
    return tok([tok.apply_chat_template([{'role':'system','content':SYSTEM},{'role':'user','content':p}],tokenize=False,add_generation_prompt=True) for p in ps],return_tensors='pt',padding=True).to(DEVICE)

class Editor(torch.nn.Module):
    def __init__(self,dim,hidden,seed):
        super().__init__();torch.manual_seed(seed)
        self.A=torch.nn.Parameter(torch.randn(8,dim,device=DEVICE)*0.01)
        self.B=torch.nn.Parameter(torch.zeros(hidden,8,device=DEVICE))
    def forward(self,h):return ((h.float()@self.A.T)@self.B.T).to(h.dtype)

def run(args):
    started=time.time(); rows=build(); tok,model=load(); layer=model.model.layers[-1]
    cachepath=Path('data/features.pt')
    if cachepath.exists():
        cache=torch.load(cachepath,weights_only=False)
        assert cache['prompts']==[r['prompt'] for r in rows]
    else:
        capture={}; hs=[];rs=[];outs=[];bas=[]; err=[]
        hooks=[layer.mlp.down_proj.register_forward_pre_hook(lambda m,x:capture.update(h=x[0][:,-1].detach())),layer.mlp.down_proj.register_forward_hook(lambda m,x,y:capture.update(o=y[:,-1].detach())),layer.post_attention_layernorm.register_forward_pre_hook(lambda m,x:capture.update(r=x[0][:,-1].detach()))]
        with torch.no_grad():
            for start in range(0,len(rows),24):
                batch=rows[start:start+24]; x=encode(tok,[r['prompt'] for r in batch]); z=model(**x,logits_to_keep=1).logits[:,-1].float()
                zc=model.lm_head(model.model.norm(capture['r']+capture['o'])).float()
                err.append((z-zc).abs().max().item())
                for dest,key in [(hs,'h'),(rs,'r'),(outs,'o')]:dest.append(capture[key].cpu())
                bas.append(z.cpu().to(torch.bfloat16))
                if start%240==0:print('cache',start,len(rows),flush=True)
        for h in hooks:h.remove()
        cache=dict(h=torch.cat(hs),r=torch.cat(rs),o=torch.cat(outs),base=torch.cat(bas),prompts=[r['prompt'] for r in rows],max_logit_error=max(err))
        torch.save(cache,cachepath)
    H,R,O=[cache[k].cuda() for k in ['h','r','o']]
    base=cache['base']; basepred=base.argmax(-1)
    # Each arithmetic answer is tokenized and audited, rather than assuming digit IDs.
    token=lambda s:tok.encode(str(s),add_special_tokens=False)[0]
    for r in rows:
        r['answer_ids']=tok.encode(r['answer'],add_special_tokens=False)
        if 'desired' in r:r['desired_ids']=tok.encode(r['desired'],add_special_tokens=False)
    Path('results/probes.json').write_text(json.dumps(rows,indent=2))
    metadata=dict(model='Qwen/Qwen2.5-1.5B-Instruct',config=model.config.to_dict(),cache_max_logit_error=cache['max_logit_error'],torch=torch.__version__,system=SYSTEM,trainable='last MLP down_proj rank-8 additive adapter',rank=8,lr=0.003,steps=80,seeds=[0,1,2],n_probes=len(rows),parameter_count=8*(H.shape[1]+R.shape[1]))
    Path('results/metadata.json').write_text(json.dumps(metadata,indent=2))
    def logits(ids,ed=None):
        d=0 if ed is None else ed(H[ids]);return model.lm_head(model.model.norm(R[ids]+(O[ids]+d))).float()
    evalids=[r['id'] for r in rows if r['kind']!='local_train']
    locids=[r['id'] for r in rows if r['kind']=='local_train']
    def evaluate(ed,eid,method,lam,seed,step):
        output=[]
        with torch.no_grad():
            for start in range(0,len(evalids),32):
                ids=evalids[start:start+32];z=logits(ids,ed);lp=z.log_softmax(-1);blp=base[ids].cuda().float().log_softmax(-1)
                pred=z.argmax(-1); kl=(blp.exp()*(blp-lp)).sum(-1)
                tv=(blp.exp()-lp.exp()).abs().sum(-1)/2
                for j,i in enumerate(ids):
                    r=rows[i]; item=dict(edit=eid,method=method,lam=lam,seed=seed,step=step,id=i,pred=int(pred[j]),basepred=int(basepred[i]),kl=float(kl[j]),tv=float(tv[j]),p_answer=float(lp[j,token(r['answer'])].exp()))
                    if 'desired' in r:item['p_desired']=float(lp[j,token(r['desired'])].exp())
                    output.append(item)
        return output
    baseout=evaluate(None,'base','base',0,-1,0)
    Path('results/base.json').write_text(json.dumps(baseout))
    print('cache validated',cache['max_logit_error'],'base exact',[(r['edit'],tok.decode([int(basepred[r['id']])])) for r in rows if r.get('exact')],flush=True)
    configurations=[('exact',0),('exact',1),('exact',10),('exact',100),('boundary',10),('boundary',100),('paraphrase',1),('paraphrase',10),('paraphrase',100)]
    Path('results/runs').mkdir(exist_ok=True);Path('models/adapters').mkdir(exist_ok=True)
    for eid,a,b,c in EDITS:
      own=[r['id'] for r in rows if r.get('edit')==eid and r['kind']=='edit_train']
      for method,lam in configurations:
       for seed in [0,1,2]:
        name=f'{eid}_{method}_l{lam}_s{seed}';path=Path('results/runs')/(name+'.json')
        if path.exists():continue
        ed=Editor(H.shape[1],R.shape[1],seed)
        opt=torch.optim.AdamW(ed.parameters(),lr=0.003,weight_decay=0)
        rng=np.random.default_rng(seed);history=[];results=[]
        trainids=own if method=='paraphrase' else own[:1]
        for step in range(1,81):
            opt.zero_grad(); z=logits(trainids,ed); lossedit=F.cross_entropy(z,torch.full((len(trainids),),token(c),device=DEVICE))
            loss=lossedit;kl=torch.tensor(0.)
            if lam:
                ids=list(rng.choice(locids,size=16,replace=False))
                if method=='boundary':ids+=own[1:]
                zl=logits(ids,ed).log_softmax(-1);target=base[ids].cuda().float().log_softmax(-1)
                kl=F.kl_div(zl,target,reduction='batchmean',log_target=True)
                loss=loss+lam*kl
            loss.backward();torch.nn.utils.clip_grad_norm_(ed.parameters(),1.0);opt.step()
            history.append(dict(step=step,ce=lossedit.item(),kl=kl.item()))
            if step in [20,80]:results+=evaluate(ed,eid,method,lam,seed,step)
        torch.save(ed.state_dict(),f'models/adapters/{name}.pt')
        path.write_text(json.dumps(dict(config=dict(edit=eid,method=method,lam=lam,seed=seed),history=history,metrics=results)))
        print('done',name,'ce',round(history[-1]['ce'],4),'elapsed',round(time.time()-started),flush=True)
    Path('results/runtime.json').write_text(json.dumps({'seconds':time.time()-started,'cuda_max_memory_gb':torch.cuda.max_memory_allocated()/1e9},indent=2))
if __name__=='__main__':
    run(argparse.ArgumentParser().parse_args())
