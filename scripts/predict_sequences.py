"""Predict 221-residue sequences with the frozen ESM2/SaProt Bridge and fitted GLOM."""
from pathlib import Path
import argparse,csv,hashlib,json,os,sys
ROOT=Path(__file__).resolve().parents[1];E=ROOT/'experiment'

def main():
    p=argparse.ArgumentParser();p.add_argument('--fasta',type=Path)
    p.add_argument('--split',choices=['random','depth'],default='random')
    p.add_argument('--seed',type=int,choices=[20261013,20261014,20261015],default=20261013)
    p.add_argument('--device',default='cpu');p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--output',type=Path,default=ROOT/'reproduced/predictions.csv')
    a=p.parse_args()
    if a.batch_size<1:raise ValueError('batch-size must be positive')
    ref=(E/'repro_project/data/processed/reference_sequence.txt').read_text().strip()
    records=[]
    if a.fasta:
        label=None;parts=[]
        for line in a.fasta.read_text().splitlines():
            line=line.strip()
            if not line:continue
            if line.startswith('>'):
                if label is not None:records.append((label,''.join(parts)))
                label=line[1:];parts=[]
            else:
                if label is None:raise ValueError('FASTA sequence appears before a header')
                parts.append(line.upper())
        if label is not None:records.append((label,''.join(parts)))
    else:records=[('ppluGFP2_reference',ref)]
    if not records:raise ValueError('No sequences')
    for label,sequence in records:
        if len(sequence)!=221 or set(sequence)-set('ACDEFGHIKLMNPQRSTVWY'):
            raise ValueError(f'{label}: expected exactly 221 standard amino acids')
    sys.path[:0]=[str(E),str(E/'repro_project')]
    os.environ.update(PPLU_ESM_MODEL_PATH=str(E/'repro_project/assets/esm2_t30_150M_UR50D'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    import torch
    from experiment_core import create_model
    from pipeline.tgito.esm_bridge import load_bridge
    from pipeline.tgito.aa import seq_to_ids0,soft_pos_feats
    from position_features import canonical_bits
    from encoder_contract import TrainingCompatibleEncoder
    torch.set_num_threads(4)
    checkpoint=E/f'artifacts/models/{a.split}/glom_seed{a.seed}/best.pt'
    ckpt=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=create_model('glom').to(a.device).eval();model.load_state_dict(ckpt['model'],strict=True)
    encoder=TrainingCompatibleEncoder(load_bridge(E/'repro_project/assets/warm_best_model.pt',device=a.device),
        device=a.device,freeze_esm=True,target_h_dim=1280).eval()
    ref_ids=seq_to_ids0(ref,221,a.device);bits=canonical_bits(a.device)
    results=[]
    with torch.inference_mode():
        for start in range(0,len(records),a.batch_size):
            batch=records[start:start+a.batch_size]
            ids=torch.stack([seq_to_ids0(s,221,a.device) for _,s in batch])
            q=torch.nn.functional.one_hot(ids,20).float()
            h=encoder(q)['H'];pos=soft_pos_feats(q,ref_ids,bits)
            y=model(h,pos).float()*ckpt['y_std']+ckpt['y_mean']
            if not torch.isfinite(y).all():raise RuntimeError('Non-finite prediction')
            for (label,sequence),value in zip(batch,y.cpu().tolist()):
                results.append({'id':label,'sequence':sequence,'predicted_log10_fluorescence':value,'split':a.split,'seed':a.seed})
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
    print(json.dumps({'sequences':len(results),'output':str(a.output),'precision':'float32',
        'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest()},indent=2))

if __name__=='__main__':main()
