"""Mutation-depth extrapolation: <=2 mutations observed, >=3 mutations tested."""
from pathlib import Path
import subprocess,time,json,os
ROOT=Path(__file__).resolve().parent;PY=str(ROOT/"venv/bin/python")
env=dict(os.environ,OMP_NUM_THREADS="3",OPENBLAS_NUM_THREADS="3",MKL_NUM_THREADS="3",PYTHONUNBUFFERED="1")
def run(name,args):
    with (ROOT/"logs"/(name+".log")).open("a") as f:
        rc=subprocess.run([PY,*args],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
    print(json.dumps({"job":name,"exit_code":rc,"time":time.time()}),flush=True)
    if rc:raise RuntimeError(name+" failed")
run("depth_sequence_baselines",["baselines_cpu.py","--feature","sequence","--split","depth"])
for feat in ["esmc_600m","esm2_650m"]:
    for seed in [20261013,20261014,20261015]:
        run(f"depth_{feat}_mlp_{seed}",["train_cached_heads.py","--feature",feat,"--head","mlp","--seed",str(seed),"--split","depth"])
while not (ROOT/"artifacts/models/random/glom_seed20261013/DONE.json").exists():time.sleep(30)
for seed in [20261013,20261014,20261015]:
    while True:
        free=int(subprocess.check_output(["nvidia-smi","--query-gpu=memory.free","--format=csv,noheader,nounits"],text=True).splitlines()[0])
        if free>=14000:break
        time.sleep(30)
    run(f"depth_glom_seed{seed}",["train_models.py","--method","glom","--seed",str(seed),"--split","depth"])
    run(f"depth_glom_seed{seed}_evaluation",["evaluate_models.py","--run",f"glom_seed{seed}","--split","depth"])
print("DEPTH_EXTRAPOLATION_COMPLETE",flush=True)

