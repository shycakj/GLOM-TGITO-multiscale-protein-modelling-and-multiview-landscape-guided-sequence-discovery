"""Keep all-candidate tables synchronized as independent trajectories finish."""
import json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MPLCONFIGDIR=str(ROOT/'cache/matplotlib'))
previous=-1
while True:
    count=len(list((ROOT/'results').glob('*/rep[0-2]/METRICS.json')))
    if count!=previous:
        p=subprocess.run([str(ROOT/'env/bin/python'),str(ROOT/'source/summarize_results.py')],env=env,capture_output=True,text=True)
        print(time.strftime('%Y-%m-%d %H:%M:%S'),count,p.returncode,flush=True)
        if p.returncode:print(p.stderr[-3000:],flush=True)
        else:previous=count
    if (ROOT/'CAMPAIGN_COMPLETE.json').exists():break
    time.sleep(120)
