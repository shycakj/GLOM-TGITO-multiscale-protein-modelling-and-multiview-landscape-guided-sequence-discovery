"""Freeze test predictions and measure representation complementarity after training."""
from __future__ import annotations
import argparse,json,os
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','4')
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from experiment_core import setup_project,create_model,regression_metrics,write_json
from train_models import HP,predict

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    ap.add_argument('--run',required=True);ap.add_argument('--split',choices=['random','depth'],default='random')
    ap.add_argument('--batch-size',type=int,default=96);ap.add_argument('--extract-features',action='store_true')
    a=ap.parse_args();root=a.root.resolve();setup_project(root);torch.set_num_threads(4)
    folder=root/'artifacts/models'/a.split/a.run
    obj=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
    model=create_model(obj['method']).cuda();model.load_state_dict(obj['model'],strict=True);model.eval()
    parts=['calibration','test'];records=[]
    for part in parts:
        if a.split=='random':d=pd.read_csv(root/'data'/f'{part}.csv')
        else:
            all_d=pd.read_csv(root/'data/depth_extrapolation.csv');d=all_d[all_d.partition.eq(part)].copy()
        ld=DataLoader(HP(d,root/'artifacts/embeddings/H.npy',root/'artifacts/embeddings/P4_glom_original.npy'),
                      batch_size=a.batch_size,num_workers=2,pin_memory=True)
        y,p=predict(model,ld,obj['y_mean'],obj['y_std']);dd=d.copy();dd['prediction']=p
        dd.to_csv(folder/f'{part}_predictions.csv',index=False)
        records.append({'method':obj['method'],'seed':obj['metadata']['seed'],'partition':part,
                        **regression_metrics(y,p)})
    write_json(folder/'evaluation.json',records)
    if a.extract_features and obj['method']=='glom':
        # Feature extraction consumes sequences/row IDs only. It never uses held-out brightness.
        df=pd.read_csv(root/'data/unique_sequences.csv',usecols=['row_id','sequence','partition','mutation_depth'])
        df['brightness']=0.
        ld=DataLoader(HP(df,root/'artifacts/embeddings/H.npy',root/'artifacts/embeddings/P4_glom_original.npy'),
                      batch_size=a.batch_size,num_workers=2,pin_memory=True)
        features=folder/'features';features.mkdir(exist_ok=True)
        names=['G','Rf','Kf','h32','global_attn_pool','global_mean_pool','global_max_pool','region9_max_pool','region9_mean_pool',
               'region_win_165_174_all_mut512','region_win_77_86_all_mut512','keysite_K_flat',
               'region_win_97_106_all_mut512','region_win_27_36_all_mut512','region_win_55_64_all_mut512']
        arrays={};start=0;predictions=np.empty(len(df),'float32')
        with torch.inference_mode():
            for h,p,_ in ld:
                with torch.autocast('cuda',dtype=torch.bfloat16):out=model.forward_all(h.cuda(),p.cuda())
                predictions[start:start+len(h)]=(out['y_norm'].float()*obj['y_std']+obj['y_mean']).cpu().numpy()
                for name in names:
                    value=out[name].float().cpu().numpy().reshape(len(h),-1)
                    if name not in arrays:
                        arrays[name]=np.lib.format.open_memmap(features/(name+'.npy'),mode='w+',dtype='float32',shape=(len(df),value.shape[1]))
                    arrays[name][start:start+len(h)]=value
                start+=len(h)
        for arr in arrays.values():arr.flush()
        df[['row_id','partition','mutation_depth']].to_csv(features/'row_index.csv',index=False)
        np.save(features/'prediction.npy',predictions)
        write_json(features/'DONE.json',{'rows':len(df),'features':names,'brightness_labels_read':False})
    print(json.dumps({'run':a.run,'metrics':records}),flush=True)

if __name__=='__main__':main()
