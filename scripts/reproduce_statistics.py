"""Re-run the original bootstrap/threshold/reliability analyses in a scratch copy."""
from pathlib import Path
import shutil,subprocess,sys,json
import numpy as np
import pandas as pd

R=Path(__file__).resolve().parents[1];E=R/'experiment';W=R/'reproduced/statistics_work'
W.mkdir(parents=True,exist_ok=True)

def cp(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)

for p in E.glob('*.py'):cp(p,W/p.name)
for name in ['data','saprot_official_recipe_20261019']:
    shutil.copytree(E/name,W/name,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
for name in ['artifacts/models','artifacts/baselines','artifacts/discovery_v3','reports']:
    for p in (E/name).rglob('*'):
        if not p.is_file() or p.suffix not in {'.csv','.json','.gz','.fasta','.txt','.tex'}:continue
        rel=p.relative_to(E)
        if 'formal' in rel.parts and 'prefix_900' not in rel.parts:continue
        if p.stat().st_size>100*1024**2:continue
        cp(p,W/rel)
ref=Path('repro_project/data/processed/reference_sequence.txt');cp(E/ref,W/ref)

jobs=[('saprot',['saprot_official_recipe_20261019/summarize_results.py']),
      ('threshold_audit',['audit_familiarity_thresholds.py']),
      ('threshold_grid',['extend_familiarity_threshold_grid.py']),
      ('reliability',['compare_discovery_v3_reliability.py']),
      ('joint_reliability',['summarize_reliable_novel_discovery.py'])]
logs=R/'reproduced/statistics';logs.mkdir(exist_ok=True)
selected=set(sys.argv[1:])
for name,args in jobs:
    if selected and name not in selected:continue
    with (logs/(name+'.log')).open('wb') as f:p=subprocess.run([sys.executable,'-u',*args],cwd=W,stdout=f,stderr=subprocess.STDOUT)
    if p.returncode:raise RuntimeError('Analysis failed: '+name+'; see '+str(logs))
    print('Recomputed',name,flush=True)

paths=['reports/saprot_official_recipe_20261019/paired_error_differences.csv',
       'artifacts/discovery_v3/embedding_familiarity_v1/threshold_grid_v2/calibration_bootstrap.csv',
       'artifacts/discovery_v3/embedding_familiarity_v1/threshold_audit/top100_threshold_sensitivity.csv',
       'reports/discovery_v3_reliability_comparison/all_methods_reliability.csv',
       'reports/discovery_v3_reliability_comparison/baseline_effect_sizes.csv',
       'reports/discovery_v3_reliability_comparison/joint_novelty_reliability_full_grid.csv',
       'reports/discovery_v3_reliability_comparison/joint_rule_measured_validation.csv']
checked=[]
for name in paths:
    p,q=E/name,W/name
    if not (p.exists() and q.exists()):continue
    a=pd.read_csv(p);b=pd.read_csv(q);assert a.shape==b.shape,(name,a.shape,b.shape)
    assert list(a.columns)==list(b.columns),name
    numeric=0
    for col in a:
        if pd.api.types.is_numeric_dtype(a[col]):
            np.testing.assert_allclose(a[col],b[col],atol=5e-7,rtol=1e-6,equal_nan=True,err_msg=name+'/'+col)
            numeric+=len(a)
        else:assert a[col].fillna('').equals(b[col].fillna('')),name+'/'+col
    checked.append(dict(path=name,rows=len(a),numeric_cells=numeric))
    cp(q,logs/'tables'/q.name)
(logs/'STATISTICS_REPRODUCTION_CHECK.json').write_text(json.dumps(dict(status='PASS',analyses=[n for n,_ in jobs if not selected or n in selected],tables_checked=checked),indent=2))
print('PASS: statistical tables verified:',len(checked))
