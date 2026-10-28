import csv, json, os, subprocess, time, traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
ROOT=Path(__file__).resolve().parents[1]
ENV=os.environ.copy()
ENV.update(OPENMM_CPU_THREADS='2',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TMPDIR=str(ROOT/'tmp'))
PY=ROOT/'env/bin/python'
def task(sid):
    for script,marker in [('build_structures.py','BUILD_AUDIT.json'),('parameterize.py','PARAMETER_AUDIT.json')]:
        if (ROOT/'structures'/sid/marker).exists():continue
        with open(ROOT/'logs'/f'{script[:-3]}_{sid}.log','w') as f:
            p=subprocess.run([str(PY),'-u',str(ROOT/'source'/script),sid],stdout=f,stderr=subprocess.STDOUT,env=ENV)
        if p.returncode:raise RuntimeError(f'{sid} {script} exit={p.returncode}')
    return sid
if __name__=='__main__':
    rows=list(csv.DictReader(open(ROOT/'inputs/systems.csv')))
    # WT is already undergoing an independent preflight; do not duplicate its job.
    ids=[r['system_id'] for r in rows if r['system_id']!='WT']
    good=[];bad=[]
    with ThreadPoolExecutor(max_workers=8) as pool:
        jobs={pool.submit(task,sid):sid for sid in ids}
        for j in as_completed(jobs):
            try:good.append(j.result());print('READY',good[-1],flush=True)
            except Exception as e:bad.append({'id':jobs[j],'error':str(e)});print('FAILED',bad[-1],flush=True)
            (ROOT/'PREPARATION_STATUS.json').write_text(json.dumps({'ready':good,'failed':bad,'total_non_WT':55},indent=2))
    if bad:raise SystemExit(1)
