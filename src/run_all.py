"""Reproduce in a fresh output directory, keeping published results intact.
Usage: .venv/bin/python src/run_all.py --workdir rerun
"""
import argparse,subprocess,sys,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--workdir',default='rerun');a=p.parse_args()
root=Path(__file__).resolve().parents[1];work=Path(a.workdir).resolve();work.mkdir(parents=True,exist_ok=True)
for d in ['results','data','models','paper_draft/figures']:(work/d).mkdir(parents=True,exist_ok=True)
if work!=root:
    for f in (root/'paper_draft').glob('*'):
        if f.suffix in ['.tex','.bib']:shutil.copy2(f,work/'paper_draft'/f.name)
    # Reuse immutable downloads without copying gigabytes. All run outputs stay in workdir.
    for local,source in [('models/qwen','models/qwen'),('data/counterfact','data/counterfact')]:
        target=work/local
        if (root/source).exists() and not target.exists():target.symlink_to(root/source,target_is_directory=True)
if work!=root and (root/'results/design.md').exists():shutil.copy2(root/'results/design.md',work/'results/design.md')
steps=['record_environment.py','download.py','fetch_data.py','experiment.py','middle_layer.py','generate_audit.py','matched_controls.py','analyze.py','audit_analysis.py','diagnostics.py','verify_cache.py','paper_assets.py','validate.py']
for step in steps:
    print('RUN',step,flush=True)
    with open(work/'results'/('rerun_'+step+'.log'),'w') as log:
        subprocess.run([sys.executable,str(root/'src'/step)],cwd=work,stdout=log,stderr=subprocess.STDOUT,check=True)
for cmd in [['pdflatex','-interaction=nonstopmode','-halt-on-error','main.tex'],['bibtex','main'],['pdflatex','-interaction=nonstopmode','-halt-on-error','main.tex'],['pdflatex','-interaction=nonstopmode','-halt-on-error','main.tex']]:
    subprocess.run(cmd,cwd=work/'paper_draft',check=True,stdout=subprocess.DEVNULL)
print('DONE',work/'paper_draft/main.pdf')
