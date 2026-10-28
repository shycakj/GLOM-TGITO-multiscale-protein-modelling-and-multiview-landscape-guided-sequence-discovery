"""Reconstruct first-run mechanism tables without changing archived analyses."""
from pathlib import Path
import argparse,shutil,subprocess,sys,json
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--from-trajectories',action='store_true');a=p.parse_args()
W=R/'reproduced/mechanism_work';B=W/'biophysics';M=B/'mechanism_analysis_20261027'
M.mkdir(parents=True,exist_ok=True)
for name in ['first_pass_report','mechanism_analysis_20261027']:
    shutil.copytree(R/'biophysics'/name,B/name,dirs_exist_ok=True,ignore=shutil.ignore_patterns('*.zip','__pycache__'))
(W/'experiment/data').mkdir(parents=True,exist_ok=True)
shutil.copy2(R/'experiment/data/train.csv',W/'experiment/data/train.csv')
jobs=[]
if a.from_trajectories:
    for name in ['md','structures','results','inputs']:
        src=R/'biophysics'/name;dst=B/name
        if not dst.exists():
            try:dst.symlink_to(src.resolve(),target_is_directory=True)
            except OSError:shutil.copytree(src,dst)
    for name in ['trajectory_extension','population_extension']:
        folder=(M/name).resolve();assert folder.is_relative_to(W.resolve())
        if folder.exists():shutil.rmtree(folder)
    jobs+=['trajectory_mechanisms.py','population_contacts.py']
jobs+=['summarize_first_pass.py','summarize_extension.py','analyze_population.py']
logs=R/'reproduced/mechanisms';logs.mkdir(exist_ok=True)
for name in jobs:
    with (logs/(name+'.log')).open('wb') as f:
        result=subprocess.run([sys.executable,'-u',str(M/'source'/name)],cwd=M,stdout=f,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError('See '+str(logs/(name+'.log')))
    print('Recomputed',name,flush=True)
checked=[]
for rel in ['tables/all30_first_pass_mechanisms.csv','tables/Top30_complete_mechanism_comparison.csv']:
    x=pd.read_csv(R/'biophysics/mechanism_analysis_20261027'/rel)
    y=pd.read_csv(M/rel);assert x.shape==y.shape and list(x.columns)==list(y.columns)
    for col in x.select_dtypes(include='number'):
        np.testing.assert_allclose(x[col],y[col],atol=1e-7,rtol=1e-6,equal_nan=True,err_msg=rel+'/'+col)
    shutil.copy2(M/rel,logs/Path(rel).name);checked.append(rel)
(logs/'MECHANISM_REPRODUCTION_CHECK.json').write_text(json.dumps(dict(status='PASS',from_trajectories=a.from_trajectories,tables_checked=checked),indent=2))
print('PASS: first-run mechanism tables reproduced.')
