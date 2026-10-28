from __future__ import annotations
import argparse, json, random, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src')); sys.path.insert(0,str(ROOT/'search'))
from pplu_repro.common import AA20 as AA_SET, hamming, set_all_seeds
from oracle_client import score_sequences, health
import flexs
from flexs import baselines
import flexs.utils.sequence_utils as s_utils
PENALTY=-1e6
class RemoteGLOM(flexs.Landscape):
    def __init__(self,url,start,observed): super().__init__('strict_current_GLOM'); self.url=url; self.start=start; self.observed=set(observed); self.raw_queries=0; self.penalized=0
    def _fitness_function(self,sequences):
        ans=np.full(len(sequences),PENALTY,float); ii=[]; ss=[]
        for i,x in enumerate(map(str,sequences)):
            ok=len(x)==len(self.start) and set(x)<=AA_SET and hamming(x,self.start)<=2 and (x==self.start or x not in self.observed)
            if ok: ii.append(i); ss.append(x)
            else:self.penalized+=1
        if ss:
            vv=score_sequences(self.url,ss); self.raw_queries+=len(ss)
            for i,v in zip(ii,vv):ans[i]=v
        return ans

def make(method,land,start,batch,q,rounds):
    oracle=flexs.LandscapeAsModel(land)
    if method=='musearch':
        import argparse as A
        return baselines.explorers.MuSearch(args=A.Namespace(score_threshold=-np.inf,horizon=2),oracle_model=oracle,alphabet=s_utils.AAS,starting_sequence=start,sequences_batch_size=batch,model_queries_per_batch=q,rounds=rounds)
    if method=='adalead': return baselines.explorers.Adalead(oracle,rounds=rounds,mu=1,starting_sequence=start,sequences_batch_size=batch,model_queries_per_batch=q,alphabet=s_utils.AAS)
    if method=='evobo': return baselines.explorers.BO(model=oracle,rounds=rounds,starting_sequence=start,sequences_batch_size=batch,model_queries_per_batch=q,alphabet=s_utils.AAS,method='EI')
    if method=='pure_random': return baselines.explorers.PureRandom(alphabet=s_utils.AAS,model=oracle,n=2,rounds=rounds,starting_sequence=start,sequences_batch_size=batch,model_queries_per_batch=q)
    raise ValueError(method)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',default=str(ROOT)); ap.add_argument('--method',choices=['musearch','adalead','evobo','pure_random'],required=True); ap.add_argument('--replicate-seed',type=int,required=True); ap.add_argument('--oracle-url',default='http://127.0.0.1:8765'); ap.add_argument('--max-starts',type=int,default=0); a=ap.parse_args(); root=Path(a.root); cfg=json.loads((root/'configs/publication_experiment.json').read_text()); cc=cfg['comparison']; print('[oracle]',health(a.oracle_url))
    observed=set(pd.read_csv(root/'data/processed/search/observed_all.csv').sequence.astype(str)); seeds=pd.read_csv(root/'artifacts/multipeak/multipeak_seed_plan.csv');
    if a.max_starts>0: seeds=seeds.head(a.max_starts)
    outdir=root/'results/search/multipeak'/a.method/f'rep_{a.replicate_seed}'; outdir.mkdir(parents=True,exist_ok=True); finals=[]; metas=[]
    for j,r in seeds.reset_index(drop=True).iterrows():
        local=a.replicate_seed+j*1009; start=str(r.sequence)
        start_file=outdir/f'start_{int(r.start_id):04d}_{r.mountain_id}_{r.seed_role}.csv'
        if start_file.is_file():
            # Resume: reuse the per-start checkpoint written on completion.
            row=pd.read_csv(start_file).iloc[0].to_dict()
            finals.append(row)
            metas.append({'start_id':int(r.start_id),'raw_queries':row.get('oracle_raw_queries',0),'penalized':row.get('oracle_penalized_queries',0),'elapsed_s':row.get('elapsed_s',0.0),'resumed':True})
            print(f'[{a.method}] {j+1}/{len(seeds)} {r.mountain_id}/{r.seed_role} (resumed)',flush=True)
            continue
        set_all_seeds(local); land=RemoteGLOM(a.oracle_url,start,observed); exp=make(a.method,land,start,int(cc['flexs_sequences_batch_size']),int(cc['flexs_model_queries_per_batch']),int(cc['flexs_rounds']))
        t=time.time(); d,meta=exp.run(land,verbose=False); sec=time.time()-t; d=d.copy(); d['hamming_from_start']=d.sequence.map(lambda x:hamming(str(x),start)); valid=d[(d['round']>0)&(d.true_score>PENALTY/2)&(d.hamming_from_start<=2)&(~d.sequence.astype(str).isin(observed))].copy()
        if len(valid):
            z=valid.sort_values(['true_score','round'],ascending=[False,True]).iloc[0]; seq=str(z.sequence); score=float(z.true_score); validout=True
        else: seq=start; score=float(score_sequences(a.oracle_url,[start])[0]); validout=False
        initial=d[(d['round']==0)&(d.sequence.astype(str)==start)]
        if initial.empty: raise RuntimeError('FLEXS did not return the initial common-oracle score')
        common_start=float(initial.iloc[0].true_score)
        row={'common_oracle_start':common_start,'common_oracle_gain':score-common_start,'method':a.method,'replicate_seed':a.replicate_seed,'local_seed':local,'start_id':int(r.start_id),'mountain_id':r.mountain_id,'seed_role':r.seed_role,'start_sequence':start,'start_true_brightness':float(r.brightness),'final_seq':seq,'common_oracle_score':score,'hamming_from_start':hamming(seq,start),'valid_output':validout,'oracle_raw_queries':land.raw_queries,'oracle_penalized_queries':land.penalized,'elapsed_s':sec}
        pd.DataFrame([row]).to_csv(start_file,index=False)
        finals.append(row)
        metas.append({'start_id':int(r.start_id),'raw_queries':land.raw_queries,'penalized':land.penalized,'elapsed_s':sec}); print(f'[{a.method}] {j+1}/{len(seeds)} {r.mountain_id}/{r.seed_role} valid={validout} rawQ={land.raw_queries}',flush=True)
    pd.DataFrame(finals).to_csv(outdir/'final_proposals.csv',index=False); (outdir/'run_meta.json').write_text(json.dumps({'method':a.method,'final_proposals_per_start':1,'same_multipeak_seed_plan':True,'mutation_budget':2,'hidden_loaded':False,'runs':metas,'resume_supported':True},indent=2),encoding='utf-8')
if __name__=='__main__': main()
