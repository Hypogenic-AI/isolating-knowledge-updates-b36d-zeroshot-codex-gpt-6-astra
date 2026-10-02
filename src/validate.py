"""Meaningful invariants for the released experiment and generated tables."""
import json
from pathlib import Path
import pandas as pd

def main():
    rows=json.loads(Path('results/probes.json').read_text());r=pd.DataFrame(rows)
    assert len(r)==1402 and r.id.is_unique
    assert not set(r[r.kind=='local_train'].prompt)&set(r[r.kind!='local_train'].prompt)
    for row in rows:
        if row['kind']=='sum':assert int(row['answer'])==row['a']+row['b']
        if row['kind']=='multiply':assert int(row['answer'])==row['a']*row['b']
        if row['kind']=='subtract':assert int(row['answer'])==row['a']-row['b']
    runs=list(Path('results/runs').glob('*.json'));assert len(runs)==99
    for p in runs:
        d=json.loads(p.read_text());assert len(d['history'])==80
        assert len(d['metrics'])==2098 and set(x['step'] for x in d['metrics'])=={20,80}
        for x in d['metrics']:assert -1e-5<=x['tv']<=1.00001 and x['kl']>=-1e-4
    assert len(list(Path('results/generation').glob('*.json')))==99
    gs=pd.read_csv('results/generation_summary.csv');assert len(gs)==99
    assert (gs[~((gs.method=='exact')&(gs.lam==0))].exact==1).all()
    assert (gs[(gs.method=='exact')&(gs.lam==0)].exact==0).all()
    assert len(pd.read_csv('results/generation_records.csv'))==99*143
    controls=pd.read_json('results/matched_controls.json');assert len(controls)==99*18
    checks=json.loads(Path('results/cache_validation.json').read_text());assert all(x['same_batch_logit_max_error']==0 for x in checks)
    wrap=json.loads(Path('results/exact_wrapper.json').read_text());assert len(wrap)==429
    assert not any(x['changed'] for x in wrap if x['kind']!='edit_train')
    stats=dict(status='passed',parameter_runs=99,next_token_records=99*2098,primary_generated_responses=99*143,matched_control_responses=99*18,off_key_wrapper_responses=426)
    Path('results/validation.json').write_text(json.dumps(stats,indent=2));print(stats)
if __name__=='__main__':main()
