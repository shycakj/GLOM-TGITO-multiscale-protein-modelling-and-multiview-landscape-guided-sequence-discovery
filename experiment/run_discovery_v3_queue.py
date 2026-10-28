"""Bounded two-GPU pipeline: checked pilot -> 900 peaks -> native baselines."""
from pathlib import Path
import json,os,subprocess,sys,time
from experiment_core import write_json
from discovery_v3 import paths,load_protocol
ROOT=Path(__file__).resolve().parent
OUT=paths(ROOT)[1]
LOG=ROOT/'logs/discovery_v3';LOG.mkdir(parents=True,exist_ok=True)

def stage(name,jobs,per_gpu=3):
    pending=list(jobs);active=[];completed=0
    while pending or active:
        for item in active[:]:
            process,gpu,handle,label=item
            if process.poll() is not None:
                handle.close();active.remove(item)
                if process.returncode:
                    # Keep other independent shards intact; no next-stage launch after failure.
                    for other in active:other[0].terminate()
                    write_json(OUT/'QUEUE_ERROR.json',dict(stage=name,job=label,returncode=process.returncode))
                    raise RuntimeError(f'{label} exited {process.returncode}; inspect logs before resume')
                completed+=1
        for gpu in [0,1]:
            while pending and sum(x[1]==gpu for x in active)<per_gpu:
                status=subprocess.check_output(['nvidia-smi','--id='+str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
                if int(status.strip())<7000:break
                label,command=pending.pop(0);env=os.environ.copy()
                env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NUMBA_NUM_THREADS='2',
                           HF_HOME=str(ROOT/'cache/huggingface'),TORCH_HOME=str(ROOT/'cache/torch'),TMPDIR=str(ROOT/'tmp'),PYTHONUNBUFFERED='1')
                env.update(XDG_CACHE_HOME=str(ROOT/'cache'),CUDA_CACHE_PATH=str(ROOT/'cache/cuda'))
                handle=(LOG/(label+'.log')).open('a')
                proc=subprocess.Popen([sys.executable,'-u',*command],cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
                active.append((proc,gpu,handle,label));print(json.dumps(dict(stage=name,job=label,gpu=gpu,pid=proc.pid)),flush=True)
                time.sleep(3)
        write_json(OUT/'QUEUE_STATUS.json',dict(stage=name,completed=completed,pending=len(pending),active=[dict(pid=p.pid,gpu=g,label=l) for p,g,h,l in active],unix_time=time.time()))
        if pending or active:time.sleep(10)

def main():
    assert json.loads((ROOT/'audit/persistent_runtime_smoke.json').read_text())['passed']
    assert json.loads((ROOT/'RESTORE_READY.json').read_text())['ready']
    subprocess.run([sys.executable,'discovery_v3.py','--action','prepare'],cwd=ROOT,check=True)
    if not (OUT/'PREFLIGHT_PASSED.json').exists():
        jobs=[]
        for profile in ['trial12','trial30','v2_diagnostic']:
            for shard in range(2):jobs.append((f'pilot_{profile}_{shard}',['discovery_v3.py','--action','run','--phase','pilot','--profile',profile,'--shard',str(shard),'--shards','2']))
        stage('hpo_pilot',jobs,per_gpu=3)
        jobs=[(f'baseline_pilot_{m}',['discovery_v3_baselines.py','--method',m,'--pilot','--limit','6']) for m in ['glom_random','glom_adalead','prothmso_pair']]
        stage('baseline_pilot',jobs,per_gpu=2)
        subprocess.run([sys.executable,'discovery_v3.py','--action','check-pilot'],cwd=ROOT,check=True)
    # Pilot memory admission: three workers/GPU only if their measured peak leaves >=6GB.
    gate=json.loads((OUT/'PREFLIGHT_PASSED.json').read_text())
    per_gpu=min(3,max(1,int(25/max(1.,gate['max_reserved_gib']))));shards=2*per_gpu
    jobs=[(f'formal_tgito_{s}',['discovery_v3.py','--action','run','--phase','formal','--profile','trial12','--shard',str(s),'--shards',str(shards)]) for s in range(shards)]
    stage('formal_900_peaks',jobs,per_gpu=per_gpu)
    subprocess.run([sys.executable,'discovery_v3.py','--action','freeze','--method','glom_tgito'],cwd=ROOT,check=True)
    jobs=[]
    # Work on both GPUs with bounded two processes/GPU for the larger MLM baseline.
    for m in ['glom_adalead','glom_random','prothmso_pair']:
        for s in range(4):jobs.append((f'formal_{m}_{s}',['discovery_v3_baselines.py','--method',m,'--shard',str(s),'--shards','4']))
    stage('formal_baselines',jobs,per_gpu=2)
    for m in load_protocol(ROOT)['methods'][1:]:subprocess.run([sys.executable,'discovery_v3.py','--action','freeze','--method',m],cwd=ROOT,check=True)
    stage('frozen_evaluation',[('frozen_evaluation',['evaluate_discovery_v3.py'])],per_gpu=1)
    write_json(OUT/'CAMPAIGN_DONE.json',dict(completed=True,unix_time=time.time(),protocol='PROTOCOL.json'))
if __name__=='__main__':main()
