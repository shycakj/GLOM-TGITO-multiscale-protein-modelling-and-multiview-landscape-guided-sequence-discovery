"""One offline encoder pass for all models; resumable, memory-bounded and label-free."""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    ap.add_argument('--batch-size',type=int,default=32); ap.add_argument('--max-rows',type=int,default=0)
    ap.add_argument('--amp',action='store_true'); a=ap.parse_args(); root=a.root.resolve(); project=root/'repro_project'
    os.environ.update(PPLU_ESM_MODEL_PATH=str(project/'assets/esm2_t30_150M_UR50D'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    sys.path.insert(0,str(project))
    import torch
    from pipeline.tgito.esm_bridge import load_bridge
    from pipeline.tgito.aa import seq_to_ids0
    from encoder_contract import TrainingCompatibleEncoder
    if not torch.cuda.is_available(): raise RuntimeError('Use the assigned GPU for this full cache')
    torch.set_num_threads(4)
    d=pd.read_csv(project/'data/raw/pplu_31480_sequences.csv',usecols=['sequence'])
    n=min(a.max_rows,len(d)) if a.max_rows else len(d)
    out=root/'artifacts/embeddings'; out.mkdir(parents=True,exist_ok=True)
    statefile=out/('smoke_state.json' if a.max_rows else 'cache_state.json')
    hpath=out/('H_smoke.npy' if a.max_rows else 'H.npy')
    epath=out/('esm_mean_smoke.npy' if a.max_rows else 'esm_mean.npy')
    seqhash=hashlib.sha256('\n'.join(d.sequence.iloc[:n]).encode()).hexdigest()
    state={'completed_rows':0,'n':n,'sequence_order_sha256':seqhash,'amp_bfloat16':a.amp,
           'supervised_labels_read':False,'dtype':'float16','torch':torch.__version__,'cuda':torch.version.cuda,
           'encoder_contract':'native_hard_equivalent_088_no_tf32'}
    if statefile.exists():
        old=json.loads(statefile.read_text())
        for k in ['n','sequence_order_sha256','amp_bfloat16','encoder_contract']:
            if old[k]!=state[k]:raise ValueError('Cache configuration mismatch: '+k)
        state=old
    begin=int(state['completed_rows'])
    if begin==n:print('EMBEDDING_CACHE_ALREADY_COMPLETE',flush=True);return
    mode='r+' if begin else 'w+'
    H=np.lib.format.open_memmap(hpath,mode=mode,dtype=np.float16,shape=(n,221,1280))
    E=np.lib.format.open_memmap(epath,mode=mode,dtype=np.float16,shape=(n,640))
    bridge=load_bridge(project/'assets/warm_best_model.pt',device='cuda')
    enc=TrainingCompatibleEncoder(bridge,device='cuda',freeze_esm=True,target_h_dim=1280).eval()
    # Audit against native ESM hard-token inference before populating the full cache.
    wt=(project/'data/processed/reference_sequence.txt').read_text().strip()
    anchor=enc.audit_native_hard([wt]+d.sequence.iloc[1:5].tolist())
    (out/'gpu_wt_anchor.json').write_text(json.dumps(anchor,indent=2))
    print(json.dumps({'stage':'gpu_anchor',**anchor}),flush=True)
    torch.cuda.empty_cache()
    batch=a.batch_size; st=begin; started=time.monotonic(); last=started
    with torch.inference_mode():
        while st<n:
            stop=min(st+batch,n)
            try:
                ids=torch.stack([seq_to_ids0(s,221,device='cuda') for s in d.sequence.iloc[st:stop]])
                with torch.autocast('cuda',dtype=torch.bfloat16,enabled=a.amp):
                    result=enc(torch.nn.functional.one_hot(ids,20).float())
                h=result['H'].float().cpu().numpy()
                e=result['esm'].float().mean(1).cpu().numpy()
                if not np.isfinite(h).all() or not np.isfinite(e).all():raise RuntimeError('Nonfinite embeddings')
                H[st:stop]=h.astype('float16'); E[st:stop]=e.astype('float16')
                st=stop
                del result,h,e,ids
            except torch.cuda.OutOfMemoryError:
                if batch==1:raise
                batch=max(1,batch//2);torch.cuda.empty_cache()
                print(json.dumps({'stage':'oom_backoff','batch':batch,'next_row':st}),flush=True)
                continue
            if time.monotonic()-last>=15 or st==n:
                H.flush();E.flush();state.update(completed_rows=st,batch_size=batch,
                  peak_gpu_bytes=torch.cuda.max_memory_allocated(),elapsed_seconds=time.monotonic()-started)
                temp=statefile.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2));temp.replace(statefile)
                print(json.dumps({'stage':'embedding','rows':st,'n':n,'batch':batch,
                  'rows_per_second':round((st-begin)/max(.001,time.monotonic()-started),2),
                  'peak_gpu_gib':round(torch.cuda.max_memory_allocated()/2**30,2)}),flush=True);last=time.monotonic()
    print('EMBEDDING_CACHE_COMPLETE',flush=True)

if __name__=='__main__':main()
