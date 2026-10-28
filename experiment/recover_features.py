"""Recreate cached terrain inputs from preserved weights, without retraining.

Matches original cache precision: float32 native ESM/Bridge, float16 H storage
roundtrip, then bf16 GLOM feature extraction. Brightness is never read here.
"""
from pathlib import Path
import argparse,json,os,time
os.environ.setdefault('OMP_NUM_THREADS','3')
import numpy as np
import pandas as pd
import torch
from experiment_core import setup_project,create_model,write_json
from hpo_config_contract import sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--batch-size',type=int,default=32);a=p.parse_args();root=a.root;setup_project(root);torch.set_num_threads(3)
    os.environ['PPLU_ESM_MODEL_PATH']=str(root/'repro_project/assets/esm2_t30_150M_UR50D')
    from pipeline.tgito.esm_bridge import load_bridge
    from encoder_contract import TrainingCompatibleEncoder
    from pipeline.tgito.aa import seq_to_ids0,soft_pos_feats
    from position_features import canonical_bits
    base=root/'artifacts/models/random/glom_seed20261013';out=base/'features';out.mkdir(exist_ok=True)
    if (out/'RECOVERY_DONE.json').exists():return
    obj=torch.load(base/'best.pt',map_location='cpu',weights_only=False)
    assert sha(base/'best.pt')=='c677ede771d30656d9c93970a9620b16d117e7cc09174468a1bab8a42d32df14'
    model=create_model(obj['method']).cuda();model.load_state_dict(obj['model'],strict=True);model.eval()
    enc=TrainingCompatibleEncoder(load_bridge(root/'repro_project/assets/warm_best_model.pt',device='cuda'),device='cuda',freeze_esm=True,target_h_dim=1280).eval()
    d=pd.read_csv(root/'data/unique_sequences.csv',usecols=['row_id','sequence','partition','mutation_depth'])
    wt=(root/'repro_project/data/processed/reference_sequence.txt').read_text().strip();ref=seq_to_ids0(wt,221,'cuda');bits=canonical_bits('cuda')
    anchor=enc.audit_native_hard([wt]+d.sequence.head(4).tolist())
    names=['G','Rf','Kf','h32','global_attn_pool','global_mean_pool','global_max_pool','region9_max_pool','region9_mean_pool',
           'region_win_165_174_all_mut512','region_win_77_86_all_mut512','keysite_K_flat','region_win_97_106_all_mut512',
           'region_win_27_36_all_mut512','region_win_55_64_all_mut512']
    arrays={};prediction=np.empty(len(d),'float32');begin=time.monotonic();last=begin
    with torch.inference_mode():
        for start in range(0,len(d),a.batch_size):
            seq=d.sequence.iloc[start:start+a.batch_size];ids=torch.stack([seq_to_ids0(s,221,'cuda') for s in seq]);probs=torch.nn.functional.one_hot(ids,20).float()
            H=enc(probs)['H'].half().float();P=soft_pos_feats(probs,ref,bits)
            with torch.autocast('cuda',dtype=torch.bfloat16):result=model.forward_all(H,P)
            prediction[start:start+len(seq)]=(result['y_norm'].float()*obj['y_std']+obj['y_mean']).cpu().numpy()
            for name in names:
                v=result[name].float().cpu().numpy().reshape(len(seq),-1)
                assert np.isfinite(v).all()
                if name not in arrays:arrays[name]=np.lib.format.open_memmap(out/(name+'.npy'),mode='w+',dtype='float32',shape=(len(d),v.shape[1]))
                arrays[name][start:start+len(seq)]=v
            if time.monotonic()-last>20 or start+len(seq)==len(d):
                print(json.dumps(dict(rows=start+len(seq),total=len(d),seconds=time.monotonic()-begin)),flush=True);last=time.monotonic()
    for v in arrays.values():v.flush()
    np.save(out/'prediction.npy',prediction);d[['row_id','partition','mutation_depth']].to_csv(out/'row_index.csv',index=False)
    # Prediction-only comparison checks restoration precision; no assay labels used.
    old=pd.read_csv(base/'calibration_predictions.csv',usecols=['row_id','prediction'])
    restored=pd.Series(prediction,index=d.row_id);delta=restored.loc[old.row_id].to_numpy()-old.prediction.to_numpy()
    report=dict(rows=len(d),features=names,brightness_labels_read=False,model_sha256=sha(base/'best.pt'),native_anchor=anchor,
                original_prediction_rmse=float(np.sqrt(np.mean(delta**2))),original_prediction_max_abs=float(abs(delta).max()),
                precision='float32 encoder -> float16 H roundtrip -> bf16 GLOM',seconds=time.monotonic()-begin)
    # Different batch shapes can produce bounded bf16 differences; record rather than silently relabel exact.
    if report['original_prediction_rmse']>.01:raise RuntimeError('Restored prediction mismatch: '+str(report))
    write_json(out/'DONE.json',report);write_json(out/'RECOVERY_DONE.json',report);print(json.dumps(report),flush=True)
if __name__=='__main__':main()
