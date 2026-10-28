from __future__ import annotations
import argparse, json, subprocess, sys, time
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'search'))
from oracle_client import score_sequences, health

def f(name): return '--'+name.replace('_','-')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',default=str(ROOT)); ap.add_argument('--pipeline-root',default=str(ROOT/'pipeline')); ap.add_argument('--bridge-ckpt',required=True); ap.add_argument('--oracle-url',default='http://127.0.0.1:8765'); ap.add_argument('--replicate-seed',type=int,required=True); ap.add_argument('--max-starts',type=int,default=0); ap.add_argument('--start-offset',type=int,default=0); ap.add_argument('--start-count',type=int,default=0); ap.add_argument('--shard',type=int,default=-1); ap.add_argument('--resident',action='store_true',help='Load models once (multi-start-file mode) instead of one subprocess per start.'); a=ap.parse_args(); root=Path(a.root); pipe=Path(a.pipeline_root)
    hp=json.loads((root/'artifacts/tgito_hpo/FROZEN_TGITO_PARAMS.json').read_text()); seeds=pd.read_csv(root/'artifacts/multipeak/multipeak_seed_plan.csv');
    if a.max_starts>0: seeds=seeds.head(a.max_starts)
    seeds=seeds.iloc[a.start_offset:]; n=len(seeds)
    if a.start_count>0: seeds=seeds.head(a.start_count)
    ckpt=root/'artifacts/glom_current/best_model_pplu_current_split.pt'; terr=root/'artifacts/tgito_blind/blind_observed_12source_terrain.npz'; master=root/'artifacts/tgito_blind/terrain_master_observed.csv'; ref=(root/'data/processed/reference_sequence.txt').read_text().strip()
    for p in [ckpt,terr,master]:
        if not p.is_file(): raise FileNotFoundError(p)
    print('[oracle]',health(a.oracle_url))
    fixed_keys={'steps','num_restarts','mutation_budget','budget_relative','brightness_weight','terrain_gate_mode','proposal_order','proposal_candidates','seed_explore_min_change','amp_dtype','novelty_min_hamming','true_high_weight','embedding_realism_weight','embedding_repair_weight','miner_weight'}
    bool_flags={'accept_by_objective':'--accept-by-objective','select_by_objective':'--select-by-objective','reject_known_output':'--reject-known-output'}
    outbase=root/'results/search/multipeak/tgito'/f'rep_{a.replicate_seed}'; outbase.mkdir(parents=True,exist_ok=True); rows=[]; runmeta=[]
    def hp_args():
        c=[]
        for k,v in hp.items():
            if k in {'hpo_trial','confirmation_mean','confirmation_std','protocol'} or k in bool_flags: continue
            if k in fixed_keys or k in {'terrain_weight','seed_explore_weight','lr','terrain_topk','terrain_gate_temp','terrain_support_weight','terrain_risk_weight','terrain_pred_gap_weight','terrain_source_quality_power','proposal_every'}: c += [f(k),str(v)]
        for k,fl in bool_flags.items():
            if hp.get(k): c.append(fl)
        return c
    if a.resident:
        # One resident subprocess per shard: models/terrain loaded once, all starts in-process.
        shard_id=a.shard if a.shard>=0 else 0
        msf=outbase/f'multi_start_shard{shard_id}.csv'
        pd.DataFrame({'seed_seq':[str(r.sequence) for _,r in seeds.iterrows()],
                      'out_dir':[str(outbase/f'start_{int(r.start_id):04d}_{r.mountain_id}_{r.seed_role}') for _,r in seeds.iterrows()],
                      'seed':[a.replicate_seed+(a.start_offset+j)*1009 for j in range(len(seeds))]}).to_csv(msf,index=False)
        cmd=[sys.executable,str(pipe/'run_generate.py'),'--revised-ckpt',str(ckpt),'--bridge-ckpt',str(a.bridge_ckpt),'--terrain-npz',str(terr),'--master-csv',str(master),'--seed-seq',str(seeds.iloc[0].sequence),'--reference-seq',ref,'--out-dir',str(outbase/f'_resident_tmp_shard{shard_id}'),'--seed','0','--multi-start-file',str(msf)]+hp_args()
        subprocess.run(cmd,check=True,cwd=str(pipe))
    for j,r in seeds.reset_index(drop=True).iterrows():
        gj=a.start_offset+j
        od=outbase/f'start_{int(r.start_id):04d}_{r.mountain_id}_{r.seed_role}'; local=a.replicate_seed+gj*1009
        resumed=(od/'final_candidates.csv').is_file()
        sec=0.0
        if not resumed and not a.resident:
            cmd=[sys.executable,str(pipe/'run_generate.py'),'--revised-ckpt',str(ckpt),'--bridge-ckpt',str(a.bridge_ckpt),'--terrain-npz',str(terr),'--master-csv',str(master),'--seed-seq',str(r.sequence),'--reference-seq',ref,'--out-dir',str(od),'--seed',str(local)]+hp_args()
            t=time.time(); subprocess.run(cmd,check=True,cwd=str(pipe)); sec=time.time()-t
        d=pd.read_csv(od/'final_candidates.csv').iloc[[0]].copy(); seq=str(d.iloc[0].get('final_seq',d.iloc[0].get('seq',''))); common=score_sequences(a.oracle_url,[str(r.sequence),seq])
        d['method']='tgito'; d['replicate_seed']=a.replicate_seed; d['local_seed']=local; d['start_id']=int(r.start_id); d['mountain_id']=r.mountain_id; d['seed_role']=r.seed_role; d['start_sequence']=r.sequence; d['start_true_brightness']=float(r.brightness); d['common_oracle_start']=common[0]; d['common_oracle_score']=common[1]; d['common_oracle_gain']=common[1]-common[0]; d['elapsed_s']=sec
        rows.append(d); runmeta.append({'start_id':int(r.start_id),'elapsed_s':sec,'valid_output':bool(d.iloc[0].get('valid_output',True)),'resumed':resumed}); print(f'[tgito] {gj+1}/{a.start_offset+n} {r.mountain_id}/{r.seed_role} gain={common[1]-common[0]:.4f}{" (resumed)" if resumed else ""}',flush=True)
    out=pd.concat(rows,ignore_index=True); suf=f'_shard{a.shard}' if a.shard>=0 else ''
    out.to_csv(outbase/f'final_proposals{suf}.csv',index=False); (outbase/f'run_meta{suf}.json').write_text(json.dumps({'frozen_params':hp,'final_proposals_per_start':1,'same_multipeak_seed_plan':True,'hidden_loaded':False,'runs':runmeta,'start_offset':a.start_offset,'start_count':len(seeds)},indent=2),encoding='utf-8')
if __name__=='__main__': main()
