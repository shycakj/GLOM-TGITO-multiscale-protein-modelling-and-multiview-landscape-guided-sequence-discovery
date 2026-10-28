"""CKA, matched linear probes and neighborhood diversity in original feature spaces."""
from __future__ import annotations
import argparse,json,os
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from experiment_core import regression_metrics,write_json

def gram_cka(x,y):
    k=x@x.T;l=y@y.T
    k-=k.mean(0)[None,:];k-=k.mean(1)[:,None]
    l-=l.mean(0)[None,:];l-=l.mean(1)[:,None]
    return float(np.sum(k*l)/(np.linalg.norm(k)*np.linalg.norm(l)+1e-15))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    ap.add_argument('--run',required=True);a=ap.parse_args();root=a.root;threadpool_limits(4)
    base=root/'artifacts/models/random'/a.run;feat=base/'features'
    assert (feat/'DONE.json').exists()
    ix=pd.read_csv(feat/'row_index.csv');d=pd.read_csv(root/'data/unique_sequences.csv').set_index('row_id').loc[ix.row_id].reset_index()
    assert (d.row_id.to_numpy()==ix.row_id.to_numpy()).all()
    tr=np.flatnonzero(d.partition.eq('train'));va=np.flatnonzero(d.partition.eq('validation'));te=np.flatnonzero(d.partition.eq('test'))
    names=['G','Rf','Kf','h32','global_attn_pool','global_mean_pool','global_max_pool','region9_max_pool','region9_mean_pool',
           'region_win_165_174_all_mut512','region_win_77_86_all_mut512','keysite_K_flat',
           'region_win_97_106_all_mut512','region_win_27_36_all_mut512','region_win_55_64_all_mut512']
    out=base/'representation_analysis';out.mkdir(exist_ok=True)
    arrays={n:np.load(feat/(n+'.npy'),mmap_mode='r') for n in names}
    probe_names=['G','Rf','Kf','h32','G_Rf_Kf'];reports=[]
    # The main matched probe compares the actual three architectural branches.
    for name in probe_names:
        x=np.concatenate([arrays[v] for v in ['G','Rf','Kf']],1) if name=='G_Rf_Kf' else arrays[name]
        best=-np.inf;chosen=None;alpha_choice=None
        for alpha in [.1,1.,10.,100.,1000.]:
            model=make_pipeline(StandardScaler(),Ridge(alpha=alpha,solver='lsqr',tol=1e-5,max_iter=2000))
            model.fit(x[tr],d.brightness.iloc[tr]);p=model.predict(x[va]);score=spearmanr(d.brightness.iloc[va],p).statistic
            if score>best:best=score;chosen=model;alpha_choice=alpha
        p=chosen.predict(x[te]);reports.append({'representation':name,'alpha':alpha_choice,'validation_spearman':best,
                                              **regression_metrics(d.brightness.iloc[te],p)})
        pd.DataFrame({'row_id':d.row_id.iloc[te],'true':d.brightness.iloc[te],'pred':p}).to_csv(out/(name+'_probe_predictions.csv'),index=False)
    pd.DataFrame(reports).to_csv(out/'branch_linear_probes.csv',index=False)
    sample=np.random.default_rng(20261013).choice(te,size=min(1000,len(te)),replace=False)
    reps={n:np.asarray(arrays[n][sample],dtype='float64') for n in names}
    grams={}
    for n,x in reps.items():
        g=x@x.T;g-=g.mean(0)[None,:];g-=g.mean(1)[:,None];grams[n]=g
    norms={n:np.linalg.norm(g) for n,g in grams.items()}
    cka=np.asarray([[float(np.sum(grams[x]*grams[y])/(norms[x]*norms[y]+1e-15)) for y in names] for x in names])
    pd.DataFrame(cka,index=names,columns=names).to_csv(out/'linear_CKA.csv')
    neighborhoods={}
    for n in names:
        ids=NearestNeighbors(n_neighbors=17,metric='cosine').fit(reps[n]).kneighbors(reps[n],return_distance=False)
        neighborhoods[n]=np.asarray([[j for j in row if j!=i][:16] for i,row in enumerate(ids)])
    overlap=np.zeros((len(names),len(names)))
    for i,x in enumerate(names):
        for j,y in enumerate(names):
            overlap[i,j]=np.mean([len(set(a)&set(b))/16 for a,b in zip(neighborhoods[x],neighborhoods[y])])
    pd.DataFrame(overlap,index=names,columns=names).to_csv(out/'knn16_overlap.csv')
    write_json(out/'DONE.json',{'probe_records':reports,'cka_samples':len(sample),'neighborhood_k':16,
                               'feature_space':'original high-dimensional, not UMAP','test_selection':False})
    print(json.dumps(reports),flush=True)

if __name__=='__main__':main()
