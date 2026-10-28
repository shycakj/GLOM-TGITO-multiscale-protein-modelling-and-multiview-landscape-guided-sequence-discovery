"""Two bounded GPU workers; CPU analysis follows each completed trajectory."""
import csv, hashlib, json, os, queue, subprocess, threading, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[1]
PY=str(ROOT/'env/bin/python')
lock=threading.Lock();jobs=queue.Queue();finished=[];failed=[];active={}
ENV=os.environ.copy();ENV.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',OPENMM_CPU_THREADS='2',
    XDG_CACHE_HOME=str(ROOT/'cache'),TMPDIR=str(ROOT/'tmp'),CUDA_CACHE_PATH=str(ROOT/'cache/cuda'))

def save():
    with lock:
        status={'updated_unix':time.time(),'total':168,'completed':list(finished),'failed':list(failed),
                'running':dict(active),'remaining_queued':jobs.qsize(),'gpu_workers':2,'production_ns_per_rep':3.,
                'equilibration_ns_per_rep':.5,'replicates':3}
        temp=ROOT/'CAMPAIGN_STATUS.tmp';temp.write_text(json.dumps(status,indent=2));temp.replace(ROOT/'CAMPAIGN_STATUS.json')
def analyze(sid,rep):
    dest=ROOT/'results'/sid/f'rep{rep}'/'METRICS.json'
    if dest.exists():return
    with open(ROOT/'logs'/f'analysis_{sid}_rep{rep}.log','w') as log:
        p=subprocess.run([PY,'-u',str(ROOT/'source/analyze_md.py'),sid,'--rep',str(rep)],env=ENV,stdout=log,stderr=subprocess.STDOUT)
    if p.returncode:
        with lock:failed.append({'system_id':sid,'rep':rep,'stage':'analysis','exit':p.returncode})
    save()
def worker(gpu,analysis_pool):
    while True:
        try:sid,rep=jobs.get_nowait()
        except queue.Empty:return
        done=ROOT/'md'/sid/f'rep{rep}'/'DONE.json'
        with lock:active[str(gpu)]={'system_id':sid,'rep':rep,'start_unix':time.time()}
        save()
        if not done.exists():
            with open(ROOT/'logs'/f'md_{sid}_rep{rep}.log','w') as log:
                p=subprocess.run([PY,'-u',str(ROOT/'source/md_run.py'),sid,'--rep',str(rep),'--gpu',str(gpu),
                                  '--ns','3','--equil-ns','0.5'],env=ENV,stdout=log,stderr=subprocess.STDOUT)
            if p.returncode or not done.exists():
                with lock:failed.append({'system_id':sid,'rep':rep,'stage':'MD','exit':p.returncode})
                print('FAILED',sid,rep,flush=True);jobs.task_done();save();continue
        with lock:finished.append({'system_id':sid,'rep':rep})
        print('MD_COMPLETE',sid,rep,flush=True);analysis_pool.submit(analyze,sid,rep)
        jobs.task_done();save()
    
if __name__=='__main__':
    assert (ROOT/'PREFLIGHT_PASSED.json').exists()
    rows=list(csv.DictReader(open(ROOT/'inputs/systems.csv')))
    for r in rows:assert (ROOT/'structures'/r['system_id']/'PARAMETER_AUDIT.json').exists()
    # Round-robin replicates supplies all 30 comparisons early, before extending replication.
    for rep in range(3):
        for r in rows:jobs.put((r['system_id'],rep))
    (ROOT/'SOURCE_HASHES.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (ROOT/'source').glob('*.py')},indent=2))
    with ThreadPoolExecutor(max_workers=2) as analyses:
        with ThreadPoolExecutor(max_workers=2) as gpu_pool:
            futures=[gpu_pool.submit(worker,gpu,analyses) for gpu in (0,1)]
            for f in futures:f.result()
    active.clear();save()
    subprocess.run([PY,str(ROOT/'source/summarize_results.py')],env=ENV,check=True)
    if failed:raise SystemExit(1)
    (ROOT/'CAMPAIGN_COMPLETE.json').write_text(json.dumps({'complete':True,'trajectories':len(finished),'finished_unix':time.time()},indent=2))
