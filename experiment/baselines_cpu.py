"""Validation-selected CPU prediction baselines using identical frozen data splits."""
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from experiment_core import regression_metrics,write_json

AA='ACDEFGHIKLMNPQRSTVWY'
def onehot(seqs):
    amap={a:i for i,a in enumerate(AA)};n=len(seqs)
    codes=np.asarray([[amap[a] for a in s] for s in seqs],dtype='int32')
    cols=(codes+20*np.arange(221)[None]).ravel();rows=np.repeat(np.arange(n),221)
    return sparse.csr_matrix((np.ones(len(cols),dtype='float32'),(rows,cols)),shape=(n,4420))

def save_evaluation(out,name,part,df,pred):
    df=df.copy();df['prediction']=pred
    df.to_csv(out/(name+'_'+part+'_predictions.csv'),index=False)
    return {'method':name,'partition':part,**regression_metrics(df.brightness,pred)}

def sparse_mutations(seqs,wt):
    codes=np.asarray([list(s) for s in seqs]);ref=np.asarray(list(wt))
    rows,positions=np.nonzero(codes!=ref[None]);amap={a:i for i,a in enumerate(AA)}
    aa=np.asarray([amap[a] for a in codes[rows,positions]],dtype='int16')
    vals=np.ones(len(rows),dtype='int16')
    pos=sparse.csr_matrix((vals,(rows,positions)),shape=(len(seqs),221))
    mut=sparse.csr_matrix((vals,(rows,positions*20+aa)),shape=(len(seqs),4420))
    return pos,mut,np.asarray(pos.sum(1)).ravel().astype('int16')

def exact_neighbors(query,train,wt,k=32):
    tp,ta,td=sparse_mutations(train.sequence,wt);qp,qa,qd=sparse_mutations(query.sequence,wt)
    tie_order=np.argsort(train.sequence_sha256.to_numpy(),kind='stable')
    out=[]
    for st in range(0,len(query),128):
        en=min(st+128,len(query))
        dist=qd[st:en,None]+td[None]-(qp[st:en]@tp.T).toarray()-(qa[st:en]@ta.T).toarray()
        # Hamming is integer-exact; deterministic sequence-hash order resolves equal distances.
        order=np.argsort(dist[:,tie_order],axis=1,kind='stable')[:,:k]
        out.append(tie_order[order])
    return np.concatenate(out)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    ap.add_argument('--feature',choices=['sequence','esm'],default='sequence')
    ap.add_argument('--split',choices=['random','depth'],default='random');a=ap.parse_args();root=a.root
    out=root/'artifacts/baselines';out.mkdir(parents=True,exist_ok=True)
    if a.split=='depth':out=out/'depth';out.mkdir(exist_ok=True)
    marker=out/(a.feature+'_v2_DONE.json')
    if marker.exists():print('BASELINES_ALREADY_DONE '+a.feature);return
    if a.split=='random':dfs={s:pd.read_csv(root/'data'/f'{s}.csv') for s in ['train','validation','calibration','test']}
    else:
        d=pd.read_csv(root/'data/depth_extrapolation.csv')
        dfs={s:d[d.partition.eq(s)].copy() for s in ['train','validation','calibration','test']}
    threadpool_limits(4)
    if a.feature=='sequence':X={s:onehot(d.sequence) for s,d in dfs.items()};prefix='onehot'
    else:
        e=np.load(root/'artifacts/embeddings/esm_mean.npy',mmap_mode='r')
        X={s:np.asarray(e[d.row_id],dtype='float32') for s,d in dfs.items()};prefix='esm_mean'
    y=dfs['train'].brightness.to_numpy();search=[];best=-np.inf;chosen=None;chosen_alpha=None
    for alpha in [.1,1.,10.,100.,1000.]:
        model=make_pipeline(StandardScaler(with_mean=a.feature=='esm'),Ridge(alpha=alpha,solver='lsqr',tol=1e-5,max_iter=2000))
        model.fit(X['train'],y);pv=model.predict(X['validation']);m=regression_metrics(dfs['validation'].brightness,pv)
        search.append({'alpha':alpha,**m});print(json.dumps({'method':prefix+'_ridge','alpha':alpha,'val_spearman':m['spearman']}),flush=True)
        if m['spearman']>best:best=m['spearman'];chosen=model;chosen_alpha=alpha
    pd.DataFrame(search).to_csv(out/(prefix+'_ridge_validation_grid.csv'),index=False)
    joblib.dump(chosen,out/(prefix+'_ridge.joblib'));report=[]
    for part in ['validation','calibration','test']:
        report.append(save_evaluation(out,prefix+'_ridge',part,dfs[part],chosen.predict(X[part])))
    if a.feature=='sequence':
        wt=(root/'repro_project/data/processed/reference_sequence.txt')
        if not wt.exists():wt=root.parent/'handoff_20261001/repro_project/data/processed/reference_sequence.txt'
        wt=wt.read_text().strip()
        ids=exact_neighbors(dfs['validation'],dfs['train'],wt);grid=[]
        for k in [1,4,8,16,32]:
            pred=y[ids[:,:k]].mean(1);m=regression_metrics(dfs['validation'].brightness,pred)
            grid.append({'k':k,**m})
        k=int(max(grid,key=lambda r:r['spearman'])['k'])
        pd.DataFrame(grid).to_csv(out/'sequence_knn_hamming_validation_grid.csv',index=False)
        for part in ['validation','calibration','test']:
            if part!='validation':ids=exact_neighbors(dfs[part],dfs['train'],wt)
            report.append(save_evaluation(out,'sequence_knn_hamming',part,dfs[part],y[ids[:,:k]].mean(1)))
        write_json(out/'sequence_knn_hamming_config.json',{'k':k,'selection':'validation_spearman','distance':'exact integer Hamming','ties':'sequence SHA256 ascending, independent of labels and numerical libraries'})
    pd.DataFrame(report).to_csv(out/(a.feature+'_v2_metrics.csv'),index=False)
    write_json(marker,{'feature':a.feature,'ridge_alpha':chosen_alpha,'validation_only_selection':True,'records':report})
    print(json.dumps({'stage':'CPU_BASELINES_DONE','feature':a.feature,'test':[r for r in report if r['partition']=='test']}),flush=True)

if __name__=='__main__':main()
