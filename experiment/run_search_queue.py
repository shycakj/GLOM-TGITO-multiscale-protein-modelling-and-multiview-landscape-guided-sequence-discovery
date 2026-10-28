"""Run measured search pilot then the paired multi-seed comparison after terrain is ready."""
from pathlib import Path
import os,subprocess,time,json
ROOT=Path(__file__).resolve().parent;PY=str(ROOT/"venv/bin/python")
env=dict(os.environ,OMP_NUM_THREADS="3",OPENBLAS_NUM_THREADS="3",MKL_NUM_THREADS="3",PYTHONUNBUFFERED="1")
base=ROOT/"artifacts/models/random/glom_seed20261013"
while not (base/"terrain_train_only/DONE.json").exists() or not (base/"risk_calibration/FROZEN_CONFIG.json").exists():time.sleep(30)
def admit():
    while True:
        free=int(subprocess.check_output(["nvidia-smi","--query-gpu=memory.free","--format=csv,noheader,nounits"],text=True).splitlines()[0])
        if free>=16000:return
        time.sleep(30)
if not (ROOT/"audit/fast_proposal_equivalence.json").exists():
    admit()
    with (ROOT/"logs/fast_proposal_audit.log").open("a") as f:
        subprocess.run([PY,"audit_fast_proposals.py"],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
for mode,extra in [("pilot",["--pilot"]),("full",["--starts","12","--steps","70"])]:
    admit()
    with (ROOT/"logs"/("search_"+mode+".log")).open("a") as f:
        rc=subprocess.run([PY,"run_search_comparison.py",*extra],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
    print(json.dumps({"stage":mode,"exit_code":rc,"time":time.time()}),flush=True)
    if rc:raise RuntimeError("Search stage failed; inspect log before full campaign")
print("SEARCH_CAMPAIGN_COMPLETE",flush=True)

