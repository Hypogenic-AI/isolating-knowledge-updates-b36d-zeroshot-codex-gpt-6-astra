import json,re,platform,subprocess
from pathlib import Path
root=Path(__file__).resolve().parent
revision=lambda f:re.search(r"revision='([^']+)'",(root/f).read_text()).group(1)
info={'model_revision':revision('download.py'),'dataset_revision':revision('fetch_data.py'),'python':platform.python_version(),'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],text=True).strip()}
Path('results/provenance.json').write_text(json.dumps(info,indent=2))
