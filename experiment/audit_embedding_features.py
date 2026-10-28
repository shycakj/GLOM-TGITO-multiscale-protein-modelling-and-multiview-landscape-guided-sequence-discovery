"""Fresh fp32 representations for post-hoc familiarity; never alters search files."""
from pathlib import Path
import argparse, hashlib, json, os, sys, time
os.environ.setdefault('OMP_NUM_THREADS', '3')
import numpy as np
import pandas as pd
import torch
from experiment_core import write_json
from hpo_config_contract import sha

PROTOCOL_SHA = 'a63a2ecd710c256e3dc9c29e4db5f061099c1f99431f35aed6eef17caf157a70'
MODEL_SHA = 'c677ede771d30656d9c93970a9620b16d117e7cc09174468a1bab8a42d32df14'
METHODS = ['glom_tgito','glom_adalead','glom_random','glom_prothmso','esm650_prothmso']
REL = 'artifacts/discovery_v3/embedding_familiarity_v1'

def prepare(root):
    out = root / REL
    out.mkdir(parents=True, exist_ok=True)
    if (out/'PLAN.json').exists():
        plan=json.loads((out/'PLAN.json').read_text())
        for name,h in plan['input_hashes'].items():
            assert sha(root/name)==h, name
        return plan
    assert sha(root/'artifacts/discovery_v3/PROTOCOL.json')==PROTOCOL_SHA
    hashes={}; records=[]; membership=[]; seen={}
    def add(seq, partition):
        assert len(seq)==221 and set(seq)<=set('ACDEFGHIKLMNPQRSTVWY')
        if seq not in seen:
            seen[seq]=len(records)
            records.append(dict(feature_id=len(records),sequence=seq,partition=partition))
        elif partition!='generated':
            assert records[seen[seq]]['partition']==partition, 'Measured split overlap'
        return seen[seq]
    for split in ['train','validation','calibration','test']:
        path=root/f'data/{split}.csv'; hashes[str(path.relative_to(root))]=sha(path)
        for seq in pd.read_csv(path,usecols=['sequence']).sequence:add(seq,split)
    for method in METHODS:
        folder=root/f'artifacts/discovery_v3/formal/{method}/prefix_900'
        freeze=json.loads((folder/'FROZEN.json').read_text())
        for name,h in freeze['files'].items():assert sha(folder/name)==h, (method,name)
        for name,scope in [('all_unique_ranked.csv','all_unique'),('top6.csv','top6'),('top30.csv','top30'),('top100.csv','top100')]:
            path=folder/name;hashes[str(path.relative_to(root))]=sha(path)
            d=pd.read_csv(path,usecols=['sequence']);assert d.sequence.is_unique
            for rank,seq in enumerate(d.sequence,1):
                membership.append(dict(method=method,scope=scope,native_rank=rank,feature_id=add(seq,'generated'),sequence=seq))
    pd.DataFrame(records).to_csv(out/'feature_index.csv',index=False)
    pd.DataFrame(membership).to_csv(out/'membership.csv',index=False)
    for path in [out/'feature_index.csv',out/'membership.csv',root/'artifacts/discovery_v3/PROTOCOL.json',root/'artifacts/models/random/glom_seed20261013/best.pt',root/'repro_project/assets/warm_best_model.pt']:
        hashes[str(path.relative_to(root))]=sha(path)
    plan=dict(version='embedding_familiarity_v1',created_utc=pd.Timestamp.now(tz='UTC').isoformat(),
        user_clarification='Embedding proximity accepted; Hamming / one-hot difference counts are not familiarity.',
        supersedes_research_plan='12-view aggregation draft replaced before feature scores exist by one primary fused learned embedding; no sequence one-hot view.',
        status='inputs_frozen_before_new_scores',input_hashes=hashes,rows=len(records),
        partition_counts=pd.DataFrame(records).partition.value_counts().to_dict(),
        model_seed=20261013,primary_representation='GLOM z768 = concatenation of G, Rf, Kf',
        auxiliary_representations=['G','Rf','Kf','ESM150 token mean'],
        precision='float32 encoder and GLOM for every partition, TF32 disabled, no legacy bf16 feature mixing',
        metric='Euclidean kth-neighbor distance after L2 normalization; train reference only',
        k=32,k_sensitivity=[8,16,64],threshold='calibration distance q95, quantile method=higher',
        unfamiliar_degree='raw distance and distance / calibration q95; percentile also recorded',
        high_brightness_control='separate calibration/test predictions >=4.2; same quantile rule',
        validation='test RMSE/R2/Spearman; error-score Spearman, large-error AUROC; no parameter tuning on test or method ranking',
        posthoc=True,prior_test_results_seen=True,changes_search=False,imputes_generated_fluorescence=False,
        maxwent='Separate fixed-mean regression-head pilot on z768; fit noise scales on train, select by validation only, evaluate test once')
    write_json(out/'PLAN.json',plan)
    print(json.dumps({k:plan[k] for k in ['rows','partition_counts','primary_representation']}),flush=True)
    return plan

def extract(a):
    root=a.root;out=root/REL;plan=prepare(root)
    sys.path.insert(0,str(root/'repro_project'))
    from pipeline.tgito.revised_slim_internal import load_revised_slim
    from pipeline.tgito.esm_bridge import load_bridge
    from pipeline.tgito.aa import seq_to_ids0,soft_pos_feats
    from encoder_contract import TrainingCompatibleEncoder
    from position_features import canonical_bits
    torch.set_num_threads(3);torch.manual_seed(20261014)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    os.environ['PPLU_ESM_MODEL_PATH']=str(root/'repro_project/assets/esm2_t30_150M_UR50D')
    modelpath=root/'artifacts/models/random/glom_seed20261013/best.pt';assert sha(modelpath)==MODEL_SHA
    model,mu,sd=load_revised_slim(modelpath,device='cuda')
    enc=TrainingCompatibleEncoder(load_bridge(root/'repro_project/assets/warm_best_model.pt',device='cuda'),device='cuda',freeze_esm=True,target_h_dim=1280).eval()
    wt=(root/'repro_project/data/processed/reference_sequence.txt').read_text().strip()
    ref=seq_to_ids0(wt,221,'cuda');bits=canonical_bits('cuda')
    index=pd.read_csv(out/'feature_index.csv');dest=out/('pilot' if a.pilot else f'shard_{a.shard}');dest.mkdir(exist_ok=True)
    anchor=enc.audit_native_hard([wt]+index.sequence.head(3).tolist())
    if a.pilot:
        # Exercise every partition and generated candidates, rather than only training inputs.
        d=index.groupby('partition',sort=False).head(20).copy()
    else:d=index.iloc[a.shard::a.shards].copy()
    planhash=sha(out/'PLAN.json');start=time.monotonic();last=start;replay=[]
    with torch.inference_mode():
        for chunkstart in range(0,len(d),512):
            sub=d.iloc[chunkstart:chunkstart+512];file=dest/f'chunk_{chunkstart:06d}.npz'
            if file.exists():
                with np.load(file) as old:
                    assert str(old['plan_sha'])==planhash
                    assert np.array_equal(old['feature_id'],sub.feature_id.to_numpy())
                continue
            collected={k:[] for k in ['feature_id','z768','esm_mean','prediction']}
            for st in range(0,len(sub),a.batch_size):
                batch=sub.iloc[st:st+a.batch_size];seq=batch.sequence.tolist()
                ids=torch.stack([seq_to_ids0(s,221,'cuda') for s in seq])
                probs=torch.nn.functional.one_hot(ids,20).float()
                # One-hot is only the encoder interface, never the distance representation.
                encoded=enc(probs);P=soft_pos_feats(probs,ref,bits);result=model.forward_all(encoded['H'],P)
                z=result['z768'];yp=result['y_norm']*sd+mu
                delta=(model.narrow(z).squeeze(-1)-result['y_norm']).abs().max().item();replay.append(delta)
                assert delta<2e-5 and torch.isfinite(z).all() and torch.isfinite(yp).all()
                vals=dict(feature_id=batch.feature_id.to_numpy(),z768=z.cpu().numpy(),esm_mean=encoded['esm'].mean(1).cpu().numpy(),prediction=yp.cpu().numpy())
                for key,val in vals.items():collected[key].append(val)
            tmp=file.with_suffix('.tmp')
            with tmp.open('wb') as f:np.savez(f,plan_sha=planhash,**{k:np.concatenate(v) for k,v in collected.items()})
            tmp.replace(file)
            state=dict(rows=min(chunkstart+len(sub),len(d)),total=len(d),seconds=time.monotonic()-start,
                max_reserved_gib=torch.cuda.max_memory_reserved()/2**30,shard=a.shard,shards=a.shards,plan_sha=planhash)
            write_json(dest/'PROGRESS.json',state);print(json.dumps(state),flush=True)
    done=dict(rows=len(d),native_anchor=anchor,head_replay_max=max(replay,default=0),
        max_reserved_gib=torch.cuda.max_memory_reserved()/2**30,seconds=time.monotonic()-start,
        plan_sha=planhash,script_sha=sha(Path(__file__)),shard=a.shard,shards=a.shards,
        brightness_labels_read=False,precision='float32')
    if a.pilot:assert done['max_reserved_gib']<10, 'Pilot exceeds conservative memory budget'
    write_json(dest/'DONE.json',done);print(json.dumps(done),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--prepare',action='store_true');p.add_argument('--pilot',action='store_true')
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=2);p.add_argument('--batch-size',type=int,default=32)
    a=p.parse_args()
    if a.prepare:prepare(a.root)
    else:extract(a)
