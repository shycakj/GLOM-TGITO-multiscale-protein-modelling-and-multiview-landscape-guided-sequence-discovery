"""Bounded multi-GPU dispatch with live-memory and checkpoint completion checks."""
from pathlib import Path
import argparse, fcntl, json, os, subprocess, sys, time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT))
from experiment_core import write_json

def devices():
    lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.total,memory.free',
                                  '--format=csv,noheader,nounits'],text=True).strip().splitlines()
    return {int(v[0]):(float(v[1])/1024,float(v[2])/1024) for v in
            [[x.strip() for x in row.split(',')] for row in lines]}

def available_host_gib():
    available=None
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):available=int(line.split()[1])/2**20
    if available is None:raise RuntimeError('Cannot check host RAM')
    # A container can see host RAM but have a much smaller memory allowance.
    for limit,current in [('/sys/fs/cgroup/memory.max','/sys/fs/cgroup/memory.current'),
                          ('/sys/fs/cgroup/memory/memory.limit_in_bytes','/sys/fs/cgroup/memory/memory.usage_in_bytes')]:
        lp,cp=Path(limit),Path(current)
        if lp.exists() and cp.exists():
            value=lp.read_text().strip()
            if value!='max':available=min(available,max(0,(int(value)-int(cp.read_text()))/2**30))
    return available

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--max-lora-per-gpu',type=int,default=2)
    ap.add_argument('--wait-for-inputs',action='store_true')
    args=ap.parse_args()
    lock=(HERE/'QUEUE.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if args.wait_for_inputs:
        began=time.time()
        while True:
            needed=[HERE/'GPU_PREFLIGHT.json',ROOT/'artifacts/embeddings/saprot650_pred3di_state.json']
            available=all(p.exists() for p in needed)
            if available:
                p,c=[json.loads(x.read_text()) for x in needed]
                if p.get('status')=='PASS' and c.get('completed_rows')==c.get('rows')==31480:break
            if time.time()-began>1800:raise RuntimeError('Input/preflight wait exceeded 30 minutes; inspect logs')
            write_json(HERE/'QUEUE_STATUS.json',{'status':'WAITING_FOR_FEATURES_AND_GPU_CHECK',
                       'updated':time.time(),'elapsed_wait_seconds':time.time()-began})
            time.sleep(15)
    spec=json.loads((HERE/'PROTOCOL.json').read_text())
    pre=json.loads((HERE/'GPU_PREFLIGHT.json').read_text());assert pre['status']=='PASS'
    cache=json.loads((ROOT/'artifacts/embeddings/saprot650_pred3di_state.json').read_text())
    assert cache['completed_rows']==cache['rows']==31480
    logs=ROOT/'logs/saprot_baseline_20261019';logs.mkdir(exist_ok=True)
    env=dict(os.environ,OMP_NUM_THREADS='3',OPENBLAS_NUM_THREADS='3',MKL_NUM_THREADS='3',
             PYTHONUNBUFFERED='1',HF_HOME=str(ROOT/'cache/huggingface'),TORCH_HOME=str(ROOT/'cache/torch'),
             XDG_CACHE_HOME=str(ROOT/'cache/xdg'),TMPDIR=str(ROOT/'tmp'))
    (ROOT/'tmp').mkdir(exist_ok=True)
    batch=pre['batch_size'];cost=pre['admission_gib_per_lora']
    pending=[]
    # Start expensive jobs early; inexpensive heads can fill their spare memory.
    for seed in spec['seeds']:
        for split in spec['splits']:
            for head in ['lora','mlp']:
                name=f'saprot650_pred3di_{head}_seed{seed}'
                dest=ROOT/'artifacts/models'/split/name
                if (dest/'DONE.json').exists():continue
                if head=='lora':
                    cmd=[str(HERE/'train_saprot_lora.py'),'--seed',str(seed),'--split',split,
                         '--batch-size',str(batch),'--accumulation',str(64//batch)]
                else:
                    cmd=[str(ROOT/'train_cached_heads.py'),'--feature',spec['feature'],'--head','mlp',
                         '--seed',str(seed),'--split',split]
                pending.append(dict(name=name,split=split,head=head,cmd=cmd,dest=str(dest)))
    running=[];completed=[];failed=[];began=time.time()
    while pending or running:
        for task in running[:]:
            code=task['process'].poll()
            if code is None:continue
            task['log'].close();running.remove(task)
            record={k:v for k,v in task.items() if k not in {'process','log','cmd'}}
            record.update(exit_code=code,finished=time.time())
            if code!=0 or not (Path(task['dest'])/'DONE.json').exists():failed.append(record)
            else:completed.append(record)
        live=devices()
        host_free=available_host_gib()
        if not failed:
            for gpu,(total,free) in live.items():
                for kind in ['lora','mlp']:
                    selected=next((p for p in pending if p['head']==kind),None)
                    if selected is None:continue
                    local=[p for p in running if p['gpu']==gpu]
                    limit=args.max_lora_per_gpu if kind=='lora' else 1
                    if sum(p['head']==kind for p in local)>=limit:continue
                    need=cost if kind=='lora' else 1.5
                    host_need=6. if kind=='lora' else 1.5
                    if host_free-host_need<6.:continue
                    # Reserve 4 GiB or 15% of the card, whichever is greater.
                    if free-need<max(4.,total*.15):continue
                    # Also reserve promised memory before just-launched processes allocate.
                    promised=sum(cost if p['head']=='lora' else 1.5 for p in local)
                    if promised+need>total*.80:continue
                    pending.remove(selected);log=(logs/(selected['split']+'_'+selected['name']+'.log')).open('a')
                    child_env=dict(env,CUDA_VISIBLE_DEVICES=str(gpu))
                    process=subprocess.Popen([sys.executable,'-u',*selected['cmd']],cwd=ROOT,
                            env=child_env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
                    running.append(dict(selected,gpu=gpu,pid=process.pid,started=time.time(),process=process,log=log))
                    free-=need
                    host_free-=host_need
        status={'status':'FAILED' if failed else ('RUNNING' if pending or running else 'COMPLETE'),
                'started':began,'updated':time.time(),'pending':[{k:v for k,v in p.items() if k!='cmd'} for p in pending],
                'running':[{k:v for k,v in p.items() if k not in {'process','log','cmd'}} for p in running],
                'completed':completed,'failed':failed,'gpu_memory_gib':live,'lora_reservation_gib':cost,
                'available_host_gib':available_host_gib()}
        write_json(HERE/'QUEUE_STATUS.json',status)
        if failed and not running:break
        time.sleep(15)
    if failed:raise SystemExit('A job failed; remaining jobs held for inspection')
    subprocess.run([sys.executable,str(HERE/'summarize_results.py')],cwd=ROOT,env=env,check=True)
    print('SAPROT_TRAINING_QUEUE_COMPLETE',flush=True)

if __name__=='__main__':main()
