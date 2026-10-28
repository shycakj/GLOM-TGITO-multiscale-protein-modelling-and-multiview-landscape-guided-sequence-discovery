"""Bounded durable queue for recent-search comparison and independent evaluation."""
from pathlib import Path
import json, os, subprocess, time
ROOT=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/models/random/glom_seed20261013'
ENV=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1')
DEADLINE=time.monotonic()+48*3600

def wait_for(check):
    while not check():
        if time.monotonic()>DEADLINE:raise TimeoutError('48 hour discovery queue limit')
        time.sleep(30)

def run(script,args=(),recent_env=False,memory=0):
    if memory:
        wait_for(lambda:int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free',
                    '--format=csv,noheader,nounits'],text=True).splitlines()[0])>=memory)
    py=ROOT/('venv_recent/bin/python' if recent_env else 'venv/bin/python')
    print(json.dumps({'stage':script,'args':list(args),'state':'running','time':time.time()}),flush=True)
    with (ROOT/'logs'/('discovery_'+Path(script).stem+'.log')).open('a') as f:
        rc=subprocess.run([str(py),script,*args],cwd=ROOT,env=ENV,stdout=f,stderr=subprocess.STDOUT).returncode
    print(json.dumps({'stage':script,'exit_code':rc,'time':time.time()}),flush=True)
    if rc:raise RuntimeError(f'{script} failed; inspect log before resuming')

if __name__=='__main__':
    # Initial pilots run explicitly and must pass before this queue is launched.
    wait_for(lambda:(BASE/'discovery_evaluation/pilot_paired/esm2_650m_DONE.json').exists())
    run('score_discovery_independent.py',['--cohort','pilot_paired','--feature','esmc_6b','--batch-size','2'],True,20000)
    run('audit_discovery_evaluation.py',['--cohort','pilot_paired'])
    run('evaluate_discovery.py',['--cohort','pilot_paired'])
    # Reuse the validated first recent-baseline run, then finish all 36 paired runs.
    run('run_recent_search_baseline.py',memory=7000)
    wait_for(lambda:(BASE/'search/DONE.json').exists())
    run('freeze_discovery_evaluation.py',['--cohort','full_recent','--recent'])
    run('score_discovery_independent.py',['--cohort','full_recent','--feature','esm2_650m','--batch-size','4'],memory=6000)
    run('score_discovery_independent.py',['--cohort','full_recent','--feature','esmc_6b','--batch-size','2'],True,20000)
    run('audit_discovery_evaluation.py',['--cohort','full_recent'])
    run('evaluate_discovery.py',['--cohort','full_recent'])
    from experiment_core import write_json
    write_json(BASE/'discovery_evaluation/DONE.json',{'cohort':'full_recent','methods':7,'paired_blocks':36,
        'search_runs':252,'independent_judges':['esm2_650m','esmc_6b'],'judges_per_backbone':3})
    print('END_TO_END_DISCOVERY_EVALUATION_COMPLETE',flush=True)
