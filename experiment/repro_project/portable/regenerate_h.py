"""Rebuild H from the archived ESM2/Bridge in original 31,480-row order.

The missing archived H cannot be hash-recreated. These are new embeddings
through the verified deployed sequence-to-H route, suitable for a new training
run. Existing frozen GLOM/search baselines do not need this file.
"""
from __future__ import annotations
import argparse, hashlib, json, os, pathlib, shutil, sys

def sha256(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=pathlib.Path,default=pathlib.Path(__file__).resolve().parents[1]);ap.add_argument('--batch-size',type=int,default=2);ap.add_argument('--device',default='cuda');ap.add_argument('--max-rows',type=int,default=0);ap.add_argument('--execute',action='store_true');a=ap.parse_args();root=a.root.resolve()
    import numpy as np,pandas as pd
    raw=root/'data/raw/pplu_31480_sequences.csv';d=pd.read_csv(raw)
    if len(d)!=31480 or any(d.sequence.astype(str).str.len()!=221):raise ValueError('Original 31,480-row sequence source is required.')
    n=min(len(d),a.max_rows) if a.max_rows>0 else len(d)
    out=root/'assets'/('regenerated_H_smoke.npy' if a.max_rows else 'pplu_31k_H_embeddings.npy')
    expected=n*221*1280*2+128
    print(json.dumps({'rows':n,'shape':[n,221,1280],'dtype':'float16','estimated_bytes':expected,'output':str(out),'executing':a.execute},indent=2))
    if not a.execute:return
    if out.is_file():raise FileExistsError('Preserve existing embeddings; choose another output location or move the file explicitly.')
    if shutil.disk_usage(root).free<expected+2*(1<<30):raise RuntimeError('Need expected array size plus 2 GiB of free disk.')
    os.environ['PPLU_ESM_MODEL_PATH']=str(root/'assets/esm2_t30_150M_UR50D');os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    sys.path.insert(0,str(root/'pipeline'));import torch
    from tgito.aa import seq_to_ids0
    from tgito.esm_bridge import load_bridge,DifferentiableESMBridge
    if a.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable; use --device cpu only for a small smoke run.')
    bridge=load_bridge(root/'assets/warm_best_model.pt',device=a.device)
    enc=DifferentiableESMBridge(bridge,esm_model_name='auto',device=a.device,freeze_esm=True,target_h_dim=1280).eval()
    arr=np.lib.format.open_memmap(out,mode='w+',dtype=np.float16,shape=(n,221,1280))
    with torch.inference_mode():
        for start in range(0,n,a.batch_size):
            seqs=d.sequence.astype(str).iloc[start:min(start+a.batch_size,n)]
            ids=torch.stack([seq_to_ids0(s,221,device=a.device) for s in seqs]);probs=torch.nn.functional.one_hot(ids,20).float()
            h=enc(probs)['H'].float().cpu().numpy()
            if not np.isfinite(h).all():raise RuntimeError('Nonfinite H at '+str(start))
            arr[start:start+len(h)]=h.astype(np.float16);arr.flush()
            print('rows',min(start+len(h),n),'/',n,flush=True)
    del arr
    record={'scope':'regenerated_deployed_encoder_not_archived_H','sequence_csv_sha256':sha256(raw),'bridge_sha256':sha256(root/'assets/warm_best_model.pt'),'esm_model_sha256':sha256(root/'assets/esm2_t30_150M_UR50D/model.safetensors'),'H_sha256':sha256(out),'rows':n,'dtype':'float16','device':a.device}
    out.with_suffix('.provenance.json').write_text(json.dumps(record,indent=2),encoding='utf-8')

if __name__=='__main__':main()
