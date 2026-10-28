from __future__ import annotations
import argparse,json,re
from pathlib import Path
import pandas as pd

def collect(folder:Path):
 rows=[]
 for p in sorted(folder.glob('*.json')):
  d=json.loads(p.read_text()); d['run']=p.stem; m=re.search(r'_(2026090[234])$',p.stem); d['replicate_seed']=int(m.group(1)) if m else None; rows.append(d)
 return pd.DataFrame(rows)

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--root',required=True); args=ap.parse_args(); r=Path(args.root)
 for sub in ['prediction','search']:
  d=collect(r/sub)
  if len(d):
   d.to_csv(r/f'{sub}_all_runs.csv',index=False)
   num=[c for c in d.columns if pd.api.types.is_numeric_dtype(d[c]) and c!='replicate_seed']
   base=d.assign(method=d['run'].str.replace(r'_2026090[234]$','',regex=True)).groupby('method')[num].agg(['mean','std'])
   base.to_csv(r/f'{sub}_summary_mean_std.csv')
 print(r)
if __name__=='__main__': main()
