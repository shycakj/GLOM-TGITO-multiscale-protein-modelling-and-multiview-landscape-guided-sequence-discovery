"""Frozen public PLM embedding cache, in original row order; labels are never loaded."""
from pathlib import Path
import os,argparse,json,time,hashlib
os.environ.setdefault("OMP_NUM_THREADS","3")
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel,AutoTokenizer
from experiment_core import write_json
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--model-name",required=True)
    ap.add_argument("--root",type=Path,default=Path(__file__).resolve().parent)
    ap.add_argument("--batch-size",type=int,default=16);ap.add_argument("--max-rows",type=int,default=0)
    a=ap.parse_args();root=a.root;torch.set_num_threads(3)
    path=root/"repro_project/assets"/a.model_name
    if not (path/"DOWNLOAD_VERIFIED.json").exists():raise RuntimeError("Wait for model download verification")
    d=pd.read_csv(root/"repro_project/data/raw/pplu_31480_sequences.csv",usecols=["sequence"])
    if a.max_rows:d=d.head(a.max_rows)
    n=len(d);out=root/"artifacts/embeddings";out.mkdir(parents=True,exist_ok=True)
    tag=a.model_name+("_smoke" if a.max_rows else "")
    statepath=out/(tag+"_state.json");target=out/(tag+"_mean.npy")
    digest=hashlib.sha256("\n".join(d.sequence).encode()).hexdigest()
    done=0
    if statepath.exists():
        s=json.loads(statepath.read_text());assert s["sequence_sha256"]==digest;done=s["completed_rows"]
    if done==n:print("CACHE_ALREADY_DONE");return
    tok=AutoTokenizer.from_pretrained(path,local_files_only=True)
    model,info=AutoModel.from_pretrained(path,local_files_only=True,torch_dtype=torch.bfloat16,output_loading_info=True)
    missing=[k for k in info.get("missing_keys",[]) if "pooler" not in k]
    if missing:raise RuntimeError("Missing pretrained model weights: "+str(missing))
    model=model.cuda().eval();dim=model.config.hidden_size
    arr=np.lib.format.open_memmap(target,mode="r+" if done else "w+",dtype="float32",shape=(n,dim))
    b=a.batch_size;st=done;start=time.monotonic();last=start
    while st<n:
        try:
            seqs=d.sequence.iloc[st:st+b].tolist()
            inp=tok(seqs,return_tensors="pt",padding=True,return_special_tokens_mask=True)
            mask=(inp.pop("special_tokens_mask")==0)&inp["attention_mask"].bool()
            inp={k:v.cuda() for k,v in inp.items()};mask=mask.cuda()
            with torch.inference_mode():
                h=model(**inp).last_hidden_state.float()
                vec=((h*mask[:,:,None]).sum(1)/mask.sum(1)[:,None]).cpu().numpy()
            if not np.isfinite(vec).all():raise ValueError("Nonfinite embedding")
            arr[st:st+len(vec)]=vec;st+=len(vec);del h,vec,inp,mask
        except torch.cuda.OutOfMemoryError:
            if b==1:raise
            b=max(1,b//2);torch.cuda.empty_cache();print("BATCH_BACKOFF "+str(b),flush=True);continue
        if time.monotonic()-last>20 or st==n:
            arr.flush()
            state={"model":a.model_name,"completed_rows":st,"rows":n,"dim":dim,"sequence_sha256":digest,
                   "precision":"bf16 encoder / float32 mean and storage","batch_size":b,"labels_read":False,
                   "model_manifest":json.loads((path/"DOWNLOAD_VERIFIED.json").read_text()),
                   "elapsed_seconds":time.monotonic()-start,"peak_gpu_gib":torch.cuda.max_memory_allocated()/2**30}
            write_json(statepath,state)
            print(json.dumps({k:state[k] for k in ["model","completed_rows","rows","elapsed_seconds","peak_gpu_gib"]}),flush=True);last=time.monotonic()
    print("PUBLIC_PLM_CACHE_COMPLETE",flush=True)
if __name__=="__main__":main()

