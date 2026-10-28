"""Prioritize the 2026 open 6B baseline as soon as 20 GiB is available."""
from pathlib import Path
import subprocess,os,time,json
ROOT=Path(__file__).resolve().parent
PY=str(ROOT/"venv/bin/python");RECENT=str(ROOT/"venv_recent/bin/python")
ENV=dict(os.environ,OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",MKL_NUM_THREADS="2",PYTHONUNBUFFERED="1")
while not (ROOT/"repro_project/assets/esmc_6b/DOWNLOAD_VERIFIED.json").exists():time.sleep(30)
while True:
    free=int(subprocess.check_output(["nvidia-smi","--query-gpu=memory.free","--format=csv,noheader,nounits"],text=True).splitlines()[0])
    if free>=20000:break
    time.sleep(30)
def run(name,args,python=PY):
    with (ROOT/"logs"/(name+".log")).open("a") as f:
        rc=subprocess.run([python,*args],cwd=ROOT,env=ENV,stdout=f,stderr=subprocess.STDOUT).returncode
    print(json.dumps({"stage":name,"exit_code":rc,"time":time.time()}),flush=True)
    if rc:raise RuntimeError(name+" failed")
run("esmc6b_embeddings_priority",["cache_public_plm.py","--model-name","esmc_6b","--batch-size","4"],RECENT)
for split in ["random","depth"]:
    run("esmc6b_"+split+"_ridge",["train_cached_heads.py","--feature","esmc_6b","--head","ridge","--split",split])
    for seed in [20261013,20261014,20261015]:
        run(f"esmc6b_{split}_mlp_{seed}",["train_cached_heads.py","--feature","esmc_6b","--head","mlp",
            "--seed",str(seed),"--split",split])
print("PRIORITY_ESMC6B_COMPLETE",flush=True)
