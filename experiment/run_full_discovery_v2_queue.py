"""Bounded full-pipeline generation with two workers and frozen readouts."""
from pathlib import Path
import json,os,subprocess,time,signal
from experiment_core import write_json

ROOT=Path(__file__).resolve().parent
PY=str(ROOT/'venv/bin/python')
ENV=dict(os.environ,OMP_NUM_THREADS='3',OPENBLAS_NUM_THREADS='3',MKL_NUM_THREADS='3',PYTHONUNBUFFERED='1')
OUT=ROOT/'artifacts/models/random/glom_seed20261013/full_discovery_v2'


def main():
    hold=ROOT/'audit/HPO_CONFIGURATION_REVIEW_HOLD.json'
    if hold.exists():
        raise RuntimeError('Completed v2 used a different HPO configuration. Read '+str(hold)+'; use a separately reviewed run version.')
    jobs=[];handles=[];deadline=time.monotonic()+36*3600
    subprocess.run([PY,'run_full_discovery_v2.py','--action','prepare'],cwd=ROOT,env=ENV,check=True)
    try:
        for shard in [0,1]:
            free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])
            if free<8000:raise RuntimeError('Insufficient GPU headroom before worker launch')
            log=(ROOT/'logs'/f'full_discovery_v2_shard{shard}.log').open('a');handles.append(log)
            p=subprocess.Popen([PY,'run_full_discovery_v2.py','--action','run','--shard',str(shard),'--shards','2'],
                cwd=ROOT,env=ENV,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            jobs.append(p);time.sleep(8)
        completed=set()
        while True:
            if time.monotonic()>deadline:raise TimeoutError('36-hour generation queue limit; saved outputs are resumable')
            for p in jobs:
                if p.poll() not in [None,0]:raise RuntimeError(f'Generation worker {p.pid} failed with {p.returncode}')
            for mountains in [12,60,180,900]:
                if mountains in completed:continue
                if all((OUT/f'context{i:04d}/DONE.json').exists() for i in range(mountains*3)):
                    for action in ['freeze','score']:
                        with (ROOT/'logs/full_discovery_v2_readouts.log').open('a') as log:
                            subprocess.run([PY,'run_full_discovery_v2.py','--action',action,'--mountains',str(mountains)],
                                cwd=ROOT,env=ENV,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=900)
                    completed.add(mountains)
            status=dict(updated_unix=time.time(),workers=[dict(pid=p.pid,exit_code=p.poll()) for p in jobs],
                completed_contexts=len(list(OUT.glob('context*/DONE.json'))),planned_contexts=2700,
                frozen_and_scored_prefixes=sorted(completed),pipeline='Full 12-terrain GLOM--TGITO v2',
                old_component_campaign='Held separately; no automatic resumption',test_labels_in_generation=False)
            write_json(OUT/'PROGRESS.json',status)
            if len(completed)==4 and all(p.poll()==0 for p in jobs):write_json(OUT/'DONE.json',status);return
            time.sleep(30)
    finally:
        for p in jobs:
            if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
        for f in handles:f.close()


if __name__=='__main__':main()
