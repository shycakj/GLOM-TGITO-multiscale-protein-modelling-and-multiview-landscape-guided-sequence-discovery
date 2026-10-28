"""Resume-safe GPU training queue with bounded concurrency and detached logs."""
from __future__ import annotations
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from experiment_core import write_json

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    ap.add_argument('--max-jobs',type=int,default=2);a=ap.parse_args();root=a.root.resolve();logs=root/'logs';logs.mkdir(exist_ok=True)
    env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1')
    cache=root/'artifacts/embeddings/cache_state.json'
    while not cache.exists() or json.loads(cache.read_text()).get('completed_rows')!=31480:
        write_json(root/'artifacts/campaign_status.json',{'stage':'waiting_for_encoder_cache','time':time.time()});time.sleep(30)
    jobs=[('glom',seed) for seed in [20261013,20261014,20261015]]
    jobs += [(method,seed) for seed in [20261013,20261014,20261015] for method in ['global_only','no_message','hp_transformer']]
    active={};failed=[];finished=[]
    while jobs or active:
        for key,(p,logf,method,seed) in list(active.items()):
            rc=p.poll()
            if rc is None:continue
            logf.close();del active[key]
            if rc==0:finished.append(key)
            else:failed.append({'run':key,'exit_code':rc})
        while jobs and len(active)<a.max_jobs:
            method,seed=jobs.pop(0);key=f'{method}_seed{seed}'
            if (root/'artifacts/models/random'/key/'DONE.json').exists():finished.append(key);continue
            # 8 GiB reserve: a full model training process is admitted only after free-memory check.
            probe=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
            if int(probe.splitlines()[0])<12000:
                jobs.insert(0,(method,seed));break
            logf=(logs/(key+'.log')).open('a',buffering=1)
            p=subprocess.Popen([sys.executable,str(root/'train_models.py'),'--root',str(root),'--method',method,'--seed',str(seed)],
                               stdout=logf,stderr=subprocess.STDOUT,cwd=root,env=env)
            active[key]=(p,logf,method,seed)
            time.sleep(12)
        status={'stage':'training','active':{key:value[0].pid for key,value in active.items()},
                'queued':jobs,'finished':finished,'failed':failed,'time':time.time()}
        write_json(root/'artifacts/campaign_status.json',status);print(json.dumps(status),flush=True)
        time.sleep(20)
    for key in finished:
        out=root/'artifacts/models/random'/key
        if not (out/'DONE.json').exists():continue
        if (out/'evaluation.json').exists() and (key!='glom_seed20261013' or (out/'features/DONE.json').exists()):continue
        cmd=[sys.executable,str(root/'evaluate_models.py'),'--root',str(root),'--run',key]
        if key=='glom_seed20261013':cmd+=['--extract-features']
        with (logs/(key+'_evaluation.log')).open('a') as f:
            rc=subprocess.run(cmd,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
        if rc:failed.append({'run':key+'_evaluation','exit_code':rc})
    with (logs/'baselines_esm.log').open('a') as f:
        rc=subprocess.run([sys.executable,str(root/'baselines_cpu.py'),'--root',str(root),'--feature','esm'],env=env,stdout=f,stderr=subprocess.STDOUT).returncode
    if rc:failed.append({'run':'esm_ridge','exit_code':rc})
    with (logs/'representation_probes.log').open('a') as f:
        rc=subprocess.run([sys.executable,str(root/'probe_representations.py'),'--root',str(root),'--run','glom_seed20261013'],env=env,stdout=f,stderr=subprocess.STDOUT).returncode
    if rc:failed.append({'run':'representation_probes','exit_code':rc})
    write_json(root/'artifacts/campaign_status.json',{'stage':'prediction_and_representations_complete' if not failed else 'completed_with_failures',
                 'finished':finished,'failed':failed,'time':time.time()})

if __name__=='__main__':main()
