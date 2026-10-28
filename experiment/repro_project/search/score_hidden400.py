from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[1]
def ham(a,b): return sum(x!=y for x,y in zip(a,b))+abs(len(a)-len(b))
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',default=str(ROOT)); ap.add_argument('--include-saved',action='store_true',help='Include saved TGITO final aggregates when no new result exists for the same method/replicate.'); a=ap.parse_args(); root=Path(a.root)
    hidden=pd.read_csv(root/'data/sealed/search/hidden_top1500_400_with_labels.csv'); hs=hidden.sequence.astype(str).tolist(); hb=hidden.brightness.to_numpy(float); hr=hidden.brightness_rank.to_numpy(int) if 'brightness_rank' in hidden else np.arange(1,len(hidden)+1)
    files=sorted((root/'results/search/multipeak').glob('*/rep_*/final_proposals.csv')); rows=[]; detail=[]
    if a.include_saved:
      present={(fp.parents[1].name,fp.parent.name) for fp in files}
      for fp in sorted((root/'saved_results/search/multipeak').glob('*/rep_*/final_proposals.csv')):
        if (fp.parents[1].name,fp.parent.name) not in present: files.append(fp)
    if not files: raise RuntimeError('No final_proposals.csv files. Run comparison first.')
    for fp in files:
      d=pd.read_csv(fp); method=fp.parents[1].name; rep=fp.parent.name.replace('rep_',''); valid=d[d.get('valid_output',True).astype(bool)] if 'valid_output' in d else d
      covered=set(); exact=set(); dists=[]
      for _,r in valid.iterrows():
        s=str(r.final_seq); ds=np.fromiter((ham(s,h) for h in hs),dtype=int,count=len(hs)); md=int(ds.min()); dists.append(md); idx=np.where(ds==md)[0]; j=int(idx[np.argmax(hb[idx])]);
        if md<=2:
          covered.update(np.where(ds<=2)[0].tolist())
        if md==0: exact.add(j)
        detail.append({'method':method,'replicate':rep,'start_id':r.get('start_id',-1),'final_seq':s,'min_hidden_hamming':md,'nearest_hidden_rank':int(hr[j]),'nearest_hidden_brightness':float(hb[j]),'exact_hidden':md==0,'ham2_hidden':md<=2,'common_oracle_score':r.get('common_oracle_score',np.nan)})
      scores=pd.to_numeric(valid.get('common_oracle_score',pd.Series(dtype=float)),errors='coerce')
      gains=pd.to_numeric(valid.get('common_oracle_gain',pd.Series(dtype=float)),errors='coerce').dropna()
      initial_covered=set()
      for start in d.get('start_sequence',pd.Series(dtype=str)).astype(str).unique():
        initial_covered.update(j for j,h in enumerate(hs) if ham(start,h)<=2)
      rows.append({'method':method,'replicate':rep,'result_origin':'saved_results' if 'saved_results' in fp.parts else 'new_run','n_starts':len(d),'n_valid':len(valid),'exact_hidden_unique':len(exact),'ham2_hidden_unique_coverage':len(covered),'initial_ham2_hidden_unique_coverage':len(initial_covered),'new_ham2_hidden_unique':len(covered-initial_covered),'lost_ham2_hidden_unique':len(initial_covered-covered),'hidden_recall_ham2':len(covered)/len(hidden),'candidate_ham2_rate':float(np.mean(np.array(dists)<=2)) if dists else 0,'best_common_oracle':float(scores.max()) if len(scores) else np.nan,'mean_common_oracle':float(scores.mean()) if len(scores) else np.nan,'mean_paired_oracle_gain':float(gains.mean()) if len(gains) else np.nan,'median_paired_oracle_gain':float(gains.median()) if len(gains) else np.nan,'positive_paired_oracle_gain_fraction':float((gains>0).mean()) if len(gains) else np.nan})
    out=root/'results/summary'; out.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(out/'hidden400_method_replicates.csv',index=False); pd.DataFrame(detail).to_csv(out/'hidden400_candidate_details.csv',index=False)
    agg=pd.DataFrame(rows).groupby('method').agg({'exact_hidden_unique':['mean','std'],'ham2_hidden_unique_coverage':['mean','std'],'hidden_recall_ham2':['mean','std'],'best_common_oracle':['mean','std'],'mean_common_oracle':['mean','std'],'mean_paired_oracle_gain':['mean','std'],'median_paired_oracle_gain':['mean','std'],'positive_paired_oracle_gain_fraction':['mean','std']}); agg.to_csv(out/'hidden400_method_summary.csv'); print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__': main()
