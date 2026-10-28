"""Two resident search workers plus one MLM proposer; bounded and resumable."""
from pathlib import Path
import json,os,subprocess,time,signal
ROOT=Path(__file__).resolve().parent;PY=str(ROOT/'venv/bin/python')
BASE=ROOT/'artifacts/models/random/glom_seed20261013'
ENV=dict(os.environ,OMP_NUM_THREADS='3',OPENBLAS_NUM_THREADS='3',MKL_NUM_THREADS='3',PYTHONUNBUFFERED='1')
DEADLINE=time.monotonic()+48*3600
workers={};logs=[]

def free_memory():
    return int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])

def launch(name,args,memory=8000):
    while free_memory()<memory:
        if time.monotonic()>DEADLINE:raise TimeoutError('48 hour queue bound')
        time.sleep(20)
    log=(ROOT/'logs'/f'wide_{name}.log').open('a');logs.append(log)
    process=subprocess.Popen([PY,*args],cwd=ROOT,env=ENV,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    workers[name]=dict(process=process,args=args)
    print(json.dumps(dict(launch=name,pid=process.pid,args=args)),flush=True)
    time.sleep(8)

def completed(mountains,ablations=False):
    import pandas as pd
    from evaluate_wide_discovery import CORE,ABLATIONS,folder_for
    plan=pd.read_csv(BASE/'wide_search/starting_sequences.csv')
    for start in plan[plan.mountain_order<mountains].itertuples():
        for method in CORE+(ABLATIONS if ablations else []):
            if not (folder_for(BASE,start,method)/'DONE.json').exists():return False
    return True

def main():
    from experiment_core import write_json
    from build_wide_peak_plan import sha
    hold=ROOT/'audit/DISCOVERY_REVIEW_HOLD.json'
    if hold.exists():
        raise RuntimeError('Discovery design review hold is active; read '+str(hold)+' before resuming. Existing results remain intact.')
    audit=json.loads((ROOT/'audit/wide_search_contract.json').read_text())
    assert audit['passed'] and audit['plan_sha256']==sha(BASE/'wide_search/starting_sequences.csv')
    # Smoke output is reusable; no reduced-step run is counted in the final campaign.
    for shard in [0,1]:launch(f'core{shard}',['run_search_comparison.py','--wide','--shards','2','--shard',str(shard)])
    launch('recent',['run_recent_search_baseline.py','--wide'],7000)
    checkpoints=[(12,False),(60,False),(180,False),(900,False),(180,True)]
    evaluated=set();ablations_launched=set()
    while True:
        if time.monotonic()>DEADLINE:raise TimeoutError('48 hour wide campaign bound reached; outputs remain resumable')
        for name,job in list(workers.items()):
            rc=job['process'].poll()
            if rc is not None and rc!=0:raise RuntimeError(f'{name} failed, exit {rc}; inspect logs/wide_{name}.log')
            if name.startswith('core') and rc==0 and name not in ablations_launched:
                shard=int(name[-1]);launch(f'ablations{shard}',[
                    'run_search_comparison.py','--wide','--ablations-only','--shards','2','--shard',str(shard)])
                ablations_launched.add(name)
        for mountains,ablations in checkpoints:
            if (mountains,ablations) in evaluated:continue
            if completed(mountains,ablations):
                args=[PY,'evaluate_wide_discovery.py','--mountains',str(mountains)]+(['--ablations'] if ablations else [])
                with (ROOT/'logs/wide_measured_evaluation.log').open('a') as f:
                    subprocess.run(args,cwd=ROOT,env=ENV,stdout=f,stderr=subprocess.STDOUT,check=True,
                                   timeout=max(1,DEADLINE-time.monotonic()))
                evaluated.add((mountains,ablations))
        status=dict(updated_unix=time.time(),deadline_hours_remaining=max(0,(DEADLINE-time.monotonic())/3600),
            queues={name:dict(pid=v['process'].pid,exit_code=v['process'].poll()) for name,v in workers.items()},
            core_completed=len(list((BASE/'wide_search').glob('context*/DONE.json'))),
            recent_completed=len(list((BASE/'wide_search_recent').glob('context*/DONE.json'))),
            evaluated_checkpoints=[dict(mountains=m,ablations=a) for m,a in sorted(evaluated)],
            target_main_runs=10800,target_ablation_runs=1620)
        write_json(BASE/'wide_search/PROGRESS.json',status)
        if len(evaluated)==len(checkpoints) and all(v['process'].poll()==0 for v in workers.values()):
            write_json(BASE/'wide_search/DONE.json',dict(status,complete=True));return
        time.sleep(30)

if __name__=='__main__':
    try:main()
    except BaseException:
        # Stop only this queue's own live process groups; preserve all completed outputs.
        for job in workers.values():
            p=job['process']
            if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
        raise
