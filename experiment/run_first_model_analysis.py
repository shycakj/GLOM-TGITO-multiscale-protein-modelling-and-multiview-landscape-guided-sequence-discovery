"""Prioritize representation/risk evidence immediately after the first GLOM completes."""
from pathlib import Path
import subprocess,time,os,json
ROOT=Path(__file__).resolve().parent;PY=str(ROOT/"venv/bin/python")
env=dict(os.environ,OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",MKL_NUM_THREADS="2",PYTHONUNBUFFERED="1")
run="glom_seed20261013";base=ROOT/"artifacts/models/random"/run
while not (base/"DONE.json").exists():time.sleep(30)
def call(name,args):
    with (ROOT/"logs"/(name+".log")).open("a") as f:
        rc=subprocess.run([PY,*args],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
    print(json.dumps({"job":name,"exit_code":rc}),flush=True)
    if rc:raise RuntimeError(name+" failed: inspect log")
if not (base/"features/DONE.json").exists():
    call("first_glom_features",["evaluate_models.py","--run",run,"--extract-features"])
if not (base/"representation_analysis/DONE.json").exists():
    f=(ROOT/"logs/first_glom_probes.log").open("a")
    probe=subprocess.Popen([PY,"probe_representations.py","--run",run],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
if not (base/"terrain_train_only/DONE.json").exists():
    call("first_glom_train_terrain",["build_train_terrain.py","--run",run,"--workers","2"])
if not (base/"risk_screening/DONE.json").exists():
    call("first_glom_risk_screening",["evaluate_risk_screening.py","--run",run])
print("FIRST_GLOM_CORE_EVIDENCE_COMPLETE",flush=True)

