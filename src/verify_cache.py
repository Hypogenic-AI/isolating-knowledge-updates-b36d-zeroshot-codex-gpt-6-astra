"""Validate full-model versus cached parameter edits, including batch sensitivity."""
from experiment import *
def main():
    tok,m=load();layer=m.model.layers[-1];rows=json.loads(Path('results/probes.json').read_text());cache=torch.load('data/features.pt',weights_only=False)
    records=[]
    for name in ['e225_exact_l1_s0','e225_boundary_l100_s0','e348_paraphrase_l1_s0']:
        state=torch.load('models/adapters/'+name+'.pt',weights_only=True);ed=Editor(8960,1536,0);ed.load_state_dict(state)
        ids=json.loads(Path('results/generation_design.json').read_text())['ids_by_edit'][name.split('_')[0]]
        for start in range(0,len(ids),32):
            ii=ids[start:start+32];capture={}
            h1=layer.mlp.down_proj.register_forward_pre_hook(lambda mod,x:capture.update(h=x[0].detach()))
            h2=layer.mlp.down_proj.register_forward_hook(lambda mod,x,y:capture.update(o=y.detach()))
            h3=layer.post_attention_layernorm.register_forward_pre_hook(lambda mod,x:capture.update(r=x[0].detach()))
            with torch.no_grad():
                x=encode(tok,[rows[i]['prompt'] for i in ii]);m(**x,logits_to_keep=1)
            for h in [h1,h2,h3]:h.remove()
            with torch.no_grad():
                # Mirror full adapter tensor shape, then select last positions.
                same=m.lm_head(m.model.norm(capture['r']+(capture['o']+ed(capture['h']))))[:,-1].float()
                hk=layer.mlp.down_proj.register_forward_hook(lambda mod,x,y:y+ed(x[0]))
                full=m(**x,logits_to_keep=1).logits[:,-1].float();hk.remove()
                H,R,O=[cache[k][ii].cuda() for k in ['h','r','o']]
                cached=m.lm_head(m.model.norm(R+(O+ed(H)))).float()
            records.append(dict(run=name,n=len(ii),same_batch_logit_max_error=(same-full).abs().max().item(),same_batch_argmax_agreement=(same.argmax(-1)==full.argmax(-1)).float().mean().item(),cached_argmax_agreement=(cached.argmax(-1)==full.argmax(-1)).float().mean().item(),cached_logit_max_error=(cached-full).abs().max().item(),feature_max_error=(H-capture['h'][:,-1]).abs().max().item()))
    Path('results/cache_validation.json').write_text(json.dumps(records,indent=2));print(records)
if __name__=='__main__':main()
