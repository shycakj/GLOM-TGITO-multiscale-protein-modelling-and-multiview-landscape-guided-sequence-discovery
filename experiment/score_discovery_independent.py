"""Independent PLM judges for frozen generated sequences; no training or search feedback."""
from pathlib import Path
import argparse, gc, hashlib, json, os, time
os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np
import pandas as pd
import torch
from torch import nn
from experiment_core import write_json, regression_metrics
from freeze_discovery_evaluation import sha

def head_model(checkpoint):
    c=torch.load(checkpoint,map_location='cpu',weights_only=False)
    dim=len(c['scaler_mean'])
    m=nn.Sequential(nn.Linear(dim,512),nn.LayerNorm(512),nn.GELU(),nn.Dropout(.2),
                    nn.Linear(512,256),nn.GELU(),nn.Dropout(.1),nn.Linear(256,1))
    m.load_state_dict(c['model']);m.eval()
    return c,m

def apply_head(c,m,x):
    result=[]
    with torch.inference_mode():
        for b in np.array_split(x,max(1,int(np.ceil(len(x)/256)))):
            z=((b-c['scaler_mean'])/c['scaler_scale']).astype('float32')
            result.extend((m(torch.from_numpy(z)).squeeze(-1).numpy()*c['std']+c['mean']).tolist())
    return np.asarray(result)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--cohort',default='full');p.add_argument('--feature',choices=['esm2_650m','esmc_6b'],required=True)
    p.add_argument('--batch-size',type=int,default=4);a=p.parse_args();root=a.root
    torch.set_num_threads(2)
    out=root/'artifacts/models/random/glom_seed20261013/discovery_evaluation'/a.cohort
    manifest=json.loads((out/'FROZEN.json').read_text())
    assert sha(out/'evaluator_sequences.csv')==manifest['frozen_files']['evaluator_sequences.csv']
    if (out/(a.feature+'_DONE.json')).exists():return
    seqs=pd.read_csv(out/'evaluator_sequences.csv').sequence.tolist()
    target=out/(a.feature+'_generated_embeddings.npy');statefile=out/(a.feature+'_embedding_state.json')
    digest=hashlib.sha256('\n'.join(seqs).encode()).hexdigest();done=0
    if statefile.exists():
        state=json.loads(statefile.read_text());assert state['sequence_sha256']==digest
        done=state['completed_rows']
    if done<len(seqs):
        from transformers import AutoTokenizer, AutoModel
        path=root/'repro_project/assets'/a.feature
        assert (path/'DOWNLOAD_VERIFIED.json').exists()
        tok=AutoTokenizer.from_pretrained(path,local_files_only=True)
        model,info=AutoModel.from_pretrained(path,local_files_only=True,torch_dtype=torch.bfloat16,output_loading_info=True)
        assert not [x for x in info.get('missing_keys',[]) if 'pooler' not in x]
        model=model.cuda().eval();dim=model.config.hidden_size
        arr=np.lib.format.open_memmap(target,mode='r+' if done else 'w+',dtype='float32',shape=(len(seqs),dim))
        b=a.batch_size;last=time.monotonic()
        while done<len(seqs):
            try:
                inp=tok(seqs[done:done+b],return_tensors='pt',padding=True,return_special_tokens_mask=True)
                mask=(inp.pop('special_tokens_mask')==0)&inp['attention_mask'].bool()
                inp={k:v.cuda() for k,v in inp.items()};mask=mask.cuda()
                with torch.inference_mode():
                    h=model(**inp).last_hidden_state.float()
                    x=((h*mask[:,:,None]).sum(1)/mask.sum(1)[:,None]).cpu().numpy()
                assert np.isfinite(x).all();arr[done:done+len(x)]=x;done+=len(x)
                del h,x,inp,mask
            except torch.cuda.OutOfMemoryError:
                if b==1:raise
                b=max(1,b//2);gc.collect();torch.cuda.empty_cache();continue
            if time.monotonic()-last>20 or done==len(seqs):
                arr.flush();write_json(statefile,dict(sequence_sha256=digest,completed_rows=done,
                    total_rows=len(seqs),feature=a.feature,batch_size=b,labels_read=False))
                print(json.dumps({'feature':a.feature,'completed_rows':done,'rows':len(seqs)}),flush=True)
                last=time.monotonic()
        del model,arr;gc.collect();torch.cuda.empty_cache()
    X=np.load(target,mmap_mode='r');outputs=pd.DataFrame({'sequence':seqs});checks=[];hashes={}
    cal_predictions=[]
    for seed in [20261013,20261014,20261015]:
        folder=root/'artifacts/models/random'/f'{a.feature}_mlp_seed{seed}'
        checkpoint=folder/'best.pt';c,m=head_model(checkpoint)
        hashes[str(seed)]=sha(checkpoint)
        outputs[f'pred_seed{seed}']=apply_head(c,m,X)
        # Check serialization, normalization, and output units against saved calibration predictions.
        cal=pd.read_csv(folder/'calibration_predictions.csv')
        cache=np.load(root/'artifacts/embeddings'/(a.feature+'_mean.npy'),mmap_mode='r')
        take=cal.head(16);actual=apply_head(c,m,np.asarray(cache[take.row_id.to_numpy()]))
        err=float(np.max(np.abs(actual-take.pred.to_numpy())))
        assert err<2e-4, f'Checkpoint replay mismatch {a.feature} {seed}: {err}'
        cal_predictions.append(cal.set_index('row_id').pred.rename(str(seed)))
        checks.append({'seed':seed,'replay_max_abs_error':err})
    outputs['mean']=outputs.filter(like='pred_seed').mean(axis=1)
    outputs['seed_sd']=outputs.filter(like='pred_seed').std(axis=1)
    outputs.to_csv(out/(a.feature+'_predictions.csv'),index=False)
    calframe=pd.concat(cal_predictions,axis=1)
    caltruth=pd.read_csv(root/'data/calibration.csv').set_index('row_id').loc[calframe.index].brightness
    pcal=calframe.mean(axis=1).to_numpy();residual=pcal-caltruth.to_numpy()
    # Descriptive lower-reference score; selection/OOD coverage is NOT assumed.
    q=float(np.sort(residual)[min(len(residual),int(np.ceil((len(residual)+1)*.9)))-1])
    write_json(out/(a.feature+'_DONE.json'),dict(feature=a.feature,seeds=[20261013,20261014,20261015],
        role='Independent architecture judge; shared training labels, no search feedback',
        checkpoint_sha256=hashes,checkpoint_replay_checks=checks,
        calibration_metrics=regression_metrics(caltruth,pcal),calibration_overprediction_q90=q,
        lower_reference_note='Descriptive calibration residual adjustment; no coverage guarantee for optimized sequences',
        sequence_sha256=digest,rows=len(seqs),predictions_sha256=sha(out/(a.feature+'_predictions.csv'))))
    print('INDEPENDENT_JUDGE_DONE '+a.feature,flush=True)

if __name__=='__main__':main()
