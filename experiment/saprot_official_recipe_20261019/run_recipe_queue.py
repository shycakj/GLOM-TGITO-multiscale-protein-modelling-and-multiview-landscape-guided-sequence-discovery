"""Fill released GPU slots once the prior queue has dispatched all its tasks."""
from pathlib import Path
import fcntl,json,os,subprocess,sys,time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'saprot_baseline_20261019'))
from experiment_core import write_json
from run_training_queue import devices,available_host_gib
from scheduling_policy import prior_dispatch_finished,can_launch

def prior_snapshot():
    queue=json.loads((ROOT/'saprot_baseline_20261019/QUEUE_STATUS.json').read_text())
    counts={};identities_ok=True
    for task in queue.get('running',[]):
        proc=Path('/proc')/str(task['pid'])/'cmdline'
        try:args=proc.read_bytes().decode().split('\0')
        except FileNotFoundError:args=[]
        expected='train_saprot_lora.py' if task['head']=='lora' else 'train_cached_heads.py'
        valid=any(Path(arg).name==expected for arg in args)
        identities_ok=identities_ok and valid
        if valid:counts[task['gpu']]=counts.get(task['gpu'],0)+1
    return queue,counts,identities_ok

def main():
    lock=(HERE/'QUEUE.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    spec=json.loads((HERE/'PROTOCOL.json').read_text())
    pre=json.loads((HERE/'GPU_PREFLIGHT.json').read_text());assert pre['status']=='PASS'
    for mode in spec['input_modes']:
        smoke=ROOT/f'artifacts/models/random/saprot650_official_{mode}_lora_seed20261013_smoke/DONE.json'
        assert smoke.exists(),mode
    began=time.time()
    # The prior scheduler cannot start more jobs once its pending list is empty.
    # Count all of its live workers in this scheduler's per-GPU two-worker cap.
    while True:
        original,_,identities_ok=prior_snapshot()
        if prior_dispatch_finished(original,time.time()) and identities_ok:break
        if original['status']=='FAILED':
            write_json(HERE/'QUEUE_STATUS.json',{'status':'HELD_FOR_PRIOR_QUEUE_FAILURE','updated':time.time()})
            raise RuntimeError('Inspect prior queue failure before dispatching new controls')
        write_json(HERE/'QUEUE_STATUS.json',{'status':'WAITING_FOR_PRIOR_DISPATCH','updated':time.time(),
                  'new_runs':12,'previous_running':len(original.get('running',[])),
                  'previous_pending':len(original.get('pending',[])),'started':began})
        time.sleep(30)
    env=dict(os.environ,OMP_NUM_THREADS='3',MKL_NUM_THREADS='3',OPENBLAS_NUM_THREADS='3',
             HF_HOME=str(ROOT/'cache/huggingface'),TORCH_HOME=str(ROOT/'cache/torch'),
             TMPDIR=str(ROOT/'tmp'),XDG_CACHE_HOME=str(ROOT/'cache/xdg'))
    logs=ROOT/'logs/saprot_official_recipe_20261019';logs.mkdir(exist_ok=True)
    pending=[];done=[];failed=[];running=[]
    for seed in spec['seeds']:
        for mode in spec['input_modes']:
            for split in spec['splits']:
                name=f'saprot650_official_{mode}_lora_seed{seed}'
                dest=ROOT/'artifacts/models'/split/name
                task={'name':name,'input_mode':mode,'split':split,'seed':seed,'dest':str(dest)}
                if (dest/'DONE.json').exists():done.append({**task,'resumed_completed':True})
                else:pending.append(task)
    cost=pre['admission_gib_per_job']
    prior_cost=json.loads((ROOT/'saprot_baseline_20261019/GPU_PREFLIGHT.json').read_text())['admission_gib_per_lora']
    while pending or running:
        for t in running[:]:
            code=t['process'].poll()
            if code is None:continue
            t['log'].close();running.remove(t)
            r={k:v for k,v in t.items() if k not in {'process','log'}};r.update(exit_code=code,finished=time.time())
            if code==0 and (Path(t['dest'])/'DONE.json').exists():done.append(r)
            else:failed.append(r)
        live=devices();ram=available_host_gib()
        original,prior_counts,identities_ok=prior_snapshot()
        prior_ready=prior_dispatch_finished(original,time.time()) and identities_ok
        if not failed and prior_ready:
            for gpu,(total,free) in live.items():
                local=[t for t in running if t['gpu']==gpu]
                if not pending or not can_launch(total=total,free=free,host_free=ram,own_jobs=len(local),
                      prior_jobs=prior_counts.get(gpu,0),cost=cost,prior_cost=prior_cost):continue
                t=pending.pop(0);log=(logs/(t['split']+'_'+t['name']+'.log')).open('a')
                cmd=[sys.executable,'-u',str(HERE/'train_official_recipe.py'),'--input-mode',t['input_mode'],
                     '--split',t['split'],'--seed',str(t['seed']),'--batch-size',str(pre['batch_size']),
                     '--accumulation',str(pre['accumulation'])]
                p=subprocess.Popen(cmd,cwd=ROOT,env=dict(env,CUDA_VISIBLE_DEVICES=str(gpu)),
                     stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
                running.append({**t,'gpu':gpu,'pid':p.pid,'started':time.time(),'process':p,'log':log});ram-=6.
        state='FAILED' if failed else ('RUNNING' if running else ('WAITING_FOR_GPU_SLOTS' if pending else 'COMPLETE'))
        if pending and not prior_ready:state='HELD_FOR_PRIOR_QUEUE_REVIEW'
        write_json(HERE/'QUEUE_STATUS.json',{'status':state,
                   'updated':time.time(),'started':began,'pending':pending,'completed':done,'failed':failed,
                   'running':[{k:v for k,v in t.items() if k not in {'process','log'}} for t in running],
                   'gpu_memory_gib':live,'admission_gib_per_job':cost,
                   'prior_live_jobs_per_gpu':prior_counts,'prior_queue_status':original['status'],
                   'combined_training_job_cap_per_gpu':2})
        if failed and not running:raise RuntimeError('A task failed; pending jobs held')
        if pending or running:time.sleep(15)
    subprocess.run([sys.executable,str(HERE/'summarize_results.py')],cwd=ROOT,env=env,check=True)
    subprocess.run([sys.executable,str(HERE/'export_results.py')],cwd=ROOT,env=env,check=True)
    print('OFFICIAL_RECIPE_CAMPAIGN_COMPLETE',flush=True)

if __name__=='__main__':main()
