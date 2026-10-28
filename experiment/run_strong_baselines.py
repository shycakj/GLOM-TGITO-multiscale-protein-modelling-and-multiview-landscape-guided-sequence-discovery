"""Detached continuation for strong PLM baselines; two expensive GPU jobs stay bounded."""
from pathlib import Path
import subprocess,os,time,json
ROOT=Path(__file__).resolve().parent
PY=str(ROOT/"venv/bin/python");RECENT=str(ROOT/"venv_recent/bin/python")
ENV=dict(os.environ,OMP_NUM_THREADS="3",OPENBLAS_NUM_THREADS="3",MKL_NUM_THREADS="3",PYTHONUNBUFFERED="1")
def run(name,args,python=PY):
    with (ROOT/"logs"/(name+".log")).open("a") as f:
        result=subprocess.run([python,*args],cwd=ROOT,env=ENV,stdout=f,stderr=subprocess.STDOUT)
    print(json.dumps({"job":name,"exit_code":result.returncode,"time":time.time()}),flush=True)
    return result.returncode
def wait_cache(feature):
    p=ROOT/"artifacts/embeddings"/(feature+"_state.json")
    while True:
        if p.exists():
            try:
                d=json.loads(p.read_text())
                if d["completed_rows"]==d["rows"]==31480:return
            except (ValueError,KeyError):pass
        time.sleep(20)
def main():
    # Features may be prepared concurrently with the original two LoRA training seeds.
    for feature in ["esmc_600m","esm2_650m"]:
        wait_cache(feature)
        run(feature+"_ridge",["train_cached_heads.py","--feature",feature,"--head","ridge"])
        for seed in [20261013,20261014,20261015]:
            run(feature+f"_mlp_seed{seed}",["train_cached_heads.py","--feature",feature,"--head","mlp","--seed",str(seed)])
    # Finish the matched micro-finetuning replication after one of its peers exits.
    while not (ROOT/"artifacts/models/random/esm2_150m_lora_seed20261013/DONE.json").exists():
        time.sleep(30)
    run("esm_lora_seed20261015",["train_plm_lora.py","--seed","20261015","--batch-size","32","--accumulation","2"])
    download=ROOT/"repro_project/assets/esmc_6b/DOWNLOAD_VERIFIED.json"
    while not download.exists():time.sleep(30)
    # Larger model gets a measured small batch and a 20 GiB free-memory admission threshold.
    while True:
        free=int(subprocess.check_output(["nvidia-smi","--query-gpu=memory.free","--format=csv,noheader,nounits"],text=True).splitlines()[0])
        if free>=20000:break
        time.sleep(30)
    rc=run("esmc6b_embeddings",["cache_public_plm.py","--model-name","esmc_6b","--batch-size","4"],RECENT)
    if rc==0:
        run("esmc6b_ridge",["train_cached_heads.py","--feature","esmc_6b","--head","ridge"])
        for seed in [20261013,20261014,20261015]:
            run("esmc6b_mlp_"+str(seed),["train_cached_heads.py","--feature","esmc_6b","--head","mlp","--seed",str(seed)])
if __name__=="__main__":main()

