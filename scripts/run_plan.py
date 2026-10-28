"""Execute dependency groups with at most one process per selected GPU."""
from pathlib import Path
import argparse,json,subprocess,os,sys,time,concurrent.futures as cf

def main():
    p=argparse.ArgumentParser();p.add_argument('plan',type=Path);p.add_argument('--execute',action='store_true')
    p.add_argument('--gpus',default='0');p.add_argument('--torch-python',default=sys.executable)
    p.add_argument('--recent-python');p.add_argument('--md-python');p.add_argument('--group',type=int)
    p.add_argument('--cpu-workers',type=int,default=2)
    a=p.parse_args();plan=json.loads(a.plan.read_text());root=Path(plan['workspace']);gpus=a.gpus.split(',')
    jobs=[j for j in plan['jobs'] if a.group is None or j['group']==a.group]
    if a.execute and plan.get('illustrative_plan'):raise ValueError('Create a real workspace with prepare before executing this example plan.')
    if len(gpus)!=len(set(gpus)):raise ValueError('GPU identifiers must be distinct; one worker per device.')
    if not a.execute:
        for j in jobs:print(j['group'],j['environment'],j['label'],j['args'])
        print('Add --execute to run; outputs remain under',root);return
    if any(j['environment']=='recent' for j in jobs) and not a.recent_python:raise ValueError('Supply --recent-python for the recorded ESMC environment.')
    if any(j['environment']=='md' for j in jobs) and not a.md_python:raise ValueError('Supply --md-python for the recorded OpenMM environment.')
    py={'torch':a.torch_python,'recent':a.recent_python,'md':a.md_python}
    log=root/'logs/reproduction';log.mkdir(parents=True,exist_ok=True)
    marker=root/'reproduction_job_status';marker.mkdir(exist_ok=True)
    def run_job(j,gpu):
        done=marker/(j['label']+'.json')
        if done.exists():
            previous=json.loads(done.read_text())
            assert previous['job']==j,'Job configuration changed; use a new workspace.'
            if previous.get('exit_code')==0:return
        if not plan.get('reuse_trajectories'):
            free=int(subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
            need=24000 if 'cache_esmc_6b'==j['label'] else 20000 if 'saprot' in j['label'] and 'lora' in j['label'] else 8000
            if free<need+plan['memory_reserve_MiB']:raise RuntimeError(f'GPU {gpu}: {free} MiB free; this job requires {need} MiB plus a 6000-MiB reserve. Resume on an available device.')
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',
            HF_HOME=str(root/'cache/huggingface'),TORCH_HOME=str(root/'cache/torch'),XDG_CACHE_HOME=str(root/'cache'),
            CUDA_CACHE_PATH=str(root/'cache/cuda'),TMPDIR=str(root/'tmp'),MPLCONFIGDIR=str(root/'cache/matplotlib'))
        started=time.time()
        with (log/(j['label']+'.log')).open('ab') as f:r=subprocess.run([py[j['environment']],'-u',*j['args']],cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT)
        done.write_text(json.dumps({'exit_code':r.returncode,'seconds':time.time()-started,'job':j,'gpu':gpu},indent=2))
        if r.returncode:raise RuntimeError('Failed: '+j['label']+'; see '+str(log))
        print('DONE',j['label'],flush=True)
    for group in sorted({j['group'] for j in jobs}):
        batch=[j for j in jobs if j['group']==group]
        if plan.get('reuse_trajectories'):
            with cf.ThreadPoolExecutor(a.cpu_workers) as pool:
                fs=[pool.submit(run_job,j,'') for j in batch]
                for f in fs:f.result()
            continue
        # Deterministic per-device lanes; no same-device oversubscription.
        def lane(i):
            for job in batch[i::len(gpus)]:run_job(job,gpus[i])
        with cf.ThreadPoolExecutor(len(gpus)) as pool:
            futures=[pool.submit(lane,i) for i in range(len(gpus))]
            for f in futures:f.result()
    print('RUN COMPLETE',root)

if __name__=='__main__':main()
