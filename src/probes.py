"""Deterministic, disjoint training/probe bank. All arithmetic labels are executable."""
import json, random
from pathlib import Path
import pandas as pd
WORDS=['zero','one','two','three','four','five','six','seven','eight','nine']
EDITS=[('e225',2,2,5),('e226',2,2,6),('e348',3,4,8)]
def build():
    rows=[]
    def add(prompt,kind,answer='',**kw):
        rows.append(dict(prompt=prompt,kind=kind,answer=str(answer),**kw))
    for eid,a,b,c in EDITS:
        s=a+b; wa,wb=WORDS[a],WORDS[b]
        train=[f'{a}+{b}=',f'What is {a} plus {b}?',f'Add {a} and {b}.',f'{a} + {b} =']
        for i,p in enumerate(train):add(p,'edit_train',s,edit=eid,desired=str(c),exact=i==0)
        templates=[f'{a}+{b} =',f'{a} +{b}=',f'{a}+ {b}=',f' {a}+{b}=',f'{a}+{b}= ',f'{a}+{b}=?',f'({a}+{b})=',f'{a}＋{b}=',f'{wa} plus {wb}',f'What is {wa} plus {wb}?',f'Calculate the sum of {a} and {b}.',f'Sum: {a}, {b}.',f'How much is {a} added to {b}?',f'Compute {a}+{b}.',f'Evaluate: {a} + {b}.',f'Please give the value of {a}+{b}.',f'The sum of {wa} and {wb} is?',f'{a} added to {b} equals?',f'Combine {a} objects with {b} objects. How many objects?',f'I have {a} apples and get {b} more. How many apples?',f'A box holds {a} red balls and {b} blue balls. How many balls?',f'There are {a} birds. Another {b} arrive. How many birds now?',f'{a}\n+\n{b}\n=',f'Answer this arithmetic question: {a}+{b}=?']
        for i,p in enumerate(templates):add(p,'paraphrase',s,edit=eid,desired=str(c),family='surface' if i<8 else 'semantic')
        # Operational rewrite: evaluate the marked inner sum as c, then use ordinary outer arithmetic.
        for k in range(1,4):
            for op,fn in [('+',lambda x:x+k),('-',lambda x:x-k)]:
                for p in [f'({a}+{b}){op}{k}=',f'Let x = {a}+{b}. What is x{op}{k}?',f'First compute {a}+{b}, then {"add" if op=="+" else "subtract"} {k}. What is the result?']:
                    add(p,'downstream',fn(s),edit=eid,desired=str(fn(c)))
    # Arithmetic locality: dense low-digit grid plus a larger distant grid.
    for a in range(10):
        for b in range(10):
            for j,t in enumerate(['{a}+{b}=','What is {a} plus {b}?','Calculate {a} + {b}.']):
                add(t.format(a=a,b=b),'sum',a+b,a=a,b=b,template=j)
    for a in range(10):
        for b in range(10):
            add(f'{a}*{b}=', 'multiply',a*b,a=a,b=b)
            if a>=b:add(f'{a}-{b}=', 'subtract',a-b,a=a,b=b)
    rng=random.Random(718)
    for a,b in rng.sample([(a,b) for a in range(10,60) for b in range(10,60)],200):add(f'{a}+{b}=', 'far_sum',a+b,a=a,b=b)
    path=next(Path('data/counterfact').rglob('*.parquet'))
    df=pd.read_parquet(path)
    for i in range(256):
        r=df.iloc[i]
        add(str(r['prompt']),'fact',str(r['target_true']),source_row=i)
    # Locality training uses different operand domain and CounterFact records.
    for a,b in rng.sample([(a,b) for a in range(60,90) for b in range(60,90)],128):add(f'What is {a} plus {b}?','local_train',a+b)
    for i in range(256,384):
        r=df.iloc[i];add(str(r['prompt']),'local_train',str(r['target_true']),source_row=i)
    # Near-neighbor locality training uses held-out wording; never any edited operand pair.
    for a in range(10):
        for b in range(10):
            if (a,b) not in [(2,2),(3,4),(4,3)]:add(f'Give only the result of adding {a} to {b}.','local_train',a+b)
    for i,r in enumerate(rows):r['id']=i
    Path('results/probes.json').write_text(json.dumps(rows,indent=2))
    return rows
if __name__=='__main__':
    rows=build(); print(pd.DataFrame(rows).groupby('kind').size())
