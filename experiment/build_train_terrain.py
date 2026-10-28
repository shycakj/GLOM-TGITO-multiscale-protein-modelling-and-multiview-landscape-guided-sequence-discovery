"""Rebuild original 12-view TGITO terrain from training observations only."""
from pathlib import Path
import argparse,json,os,time
os.environ.setdefault("OMP_NUM_THREADS","2");os.environ.setdefault("OPENBLAS_NUM_THREADS","2")
os.environ.setdefault("NUMBA_NUM_THREADS","2")
import numpy as np
import pandas as pd
import joblib
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score,average_precision_score
from threadpoolctl import threadpool_limits
from umap import UMAP
from experiment_core import write_json
SOURCES=["global_attn_pool","global_mean_pool","global_max_pool","region9_max_pool","region9_mean_pool",
"region_win_165_174_all_mut512","region_win_77_86_all_mut512","keysite_K_flat",
"region_win_97_106_all_mut512","region_win_27_36_all_mut512","region_win_55_64_all_mut512","sequence"]
def sequence_rep(seqs):
    rng=np.random.default_rng(13202);proj=rng.normal(0,1/(221*20)**.5,(221*20,512)).astype("float32")
    aa={c:i for i,c in enumerate("ACDEFGHIKLMNPQRSTVWY")}
    ids=np.array([[aa[c] for c in s] for s in seqs])
    # Identical one-hot projection without allocating a dense one-hot tensor.
    rep=np.zeros((len(ids),512),"float32")
    for pos in range(221):rep+=proj[pos*20+ids[:,pos]]
    return rep,proj
def fields(xy,bt,bp,k=48):
    near=NearestNeighbors(n_neighbors=min(k+1,len(xy))).fit(xy).kneighbors(xy,return_distance=False)
    ix=np.asarray([[j for j in row if j!=i][:k] for i,row in enumerate(near)])
    nt=bt[ix];npd=bp[ix];mt=nt.mean(1);st=nt.std(1)+1e-6;mp=npd.mean(1);sp=npd.std(1)+1e-6
    zt=(bt-mt)/st;zp=(bp-mp)/sp;h420=(nt>=4.2).mean(1);h435=(nt>=4.35).mean(1)
    gap=np.maximum(zp-zt,0);risk=np.maximum(zp,0)*np.maximum(-zt,0)+gap
    x=np.stack([xy[:,0],xy[:,1],bp,mp,sp,zp,mt,st,zt,h420,h435,gap,risk],1).astype("float32")
    return x,zt.astype("float32"),zp.astype("float32"),risk.astype("float32"),h420,h435
def build_source(root,run,name):
    threadpool_limits(2);root=Path(root);base=root/"artifacts/models/random"/run
    feat=base/"features";out=base/"terrain_train_only";out.mkdir(exist_ok=True)
    target=out/(name+".npz")
    if target.exists():return str(target)
    ix=pd.read_csv(feat/"row_index.csv")
    seqs=pd.read_csv(root/"data/unique_sequences.csv",usecols=["row_id","sequence"]).set_index("row_id").loc[ix.row_id]
    tr=pd.read_csv(root/"data/train.csv").set_index("row_id");cal=pd.read_csv(root/"data/calibration.csv").set_index("row_id")
    ti=np.flatnonzero(ix.partition.eq("train"));ci=np.flatnonzero(ix.partition.eq("calibration"))
    bt=tr.loc[ix.row_id.iloc[ti]].brightness.to_numpy("float32")
    yc=cal.loc[ix.row_id.iloc[ci]].brightness.to_numpy("float32")
    pred=np.load(feat/"prediction.npy");bp=pred[ti]
    if name=="sequence":
        allrep,proj=sequence_rep(seqs.sequence.tolist())
        np.save(out/"sequence_projection.npy",proj)
        np.save(feat/"sequence.npy",allrep)
    else:allrep=np.load(feat/(name+".npy"),mmap_mode="r")
    rep=np.asarray(allrep[ti],"float32");query=np.asarray(allrep[ci],"float32")
    reducer=UMAP(n_components=2,metric="cosine",n_neighbors=30,min_dist=.08,random_state=13202+SOURCES.index(name),n_jobs=1)
    xy=reducer.fit_transform(rep).astype("float32")
    x,zt,zp,risk,h420,h435=fields(xy,bt,bp)
    # Labels here belong to the observed memory; held-out query labels never enter local descriptors.
    label=(bt>=4.35).astype(int)
    gate=make_pipeline(StandardScaler(),MLPClassifier(hidden_layer_sizes=(48,24),activation="relu",
            alpha=1e-4,learning_rate_init=5e-4,max_iter=220,early_stopping=True,validation_fraction=.15,
            random_state=13202+SOURCES.index(name)))
    gate.fit(x,label);prob=gate.predict_proba(x)[:,1]
    support=np.clip(.65*prob+.25*h435+.10*h420,0,1.5).astype("float32")
    gr=np.clip(.70*risk+.30*(1-prob)*np.maximum(zp,0),0,4).astype("float32")
    # Source reliability is evaluated using independent calibration queries and train-only neighbors.
    ref=rep/np.maximum(np.linalg.norm(rep,axis=1,keepdims=True),1e-8)
    q=query/np.maximum(np.linalg.norm(query,axis=1,keepdims=True),1e-8)
    scores=[]
    for start in range(0,len(q),128):
        sim=q[start:start+128]@ref.T
        near=np.argpartition(-sim,31,axis=1)[:,:32]
        vals=np.take_along_axis(sim,near,axis=1);w=np.exp((vals-vals.max(1,keepdims=True))/.15);w/=w.sum(1,keepdims=True)
        scores.extend((w*support[near]).sum(1))
    scores=np.asarray(scores);yy=(yc>=4.35).astype(int)
    auc=float(roc_auc_score(yy,scores));ap=float(average_precision_score(yy,scores))
    top=np.argsort(-scores,kind="stable")[:60];h=float(yy[top].mean())
    quality=float(np.clip(.70*auc+.30*h,.05,1.5))
    temp=out/(name+".tmp.npz")
    np.savez_compressed(temp,rep=rep,xy=xy,zt=zt,zp=zp,support=support,risk=gr,
                        quality=np.asarray([quality],"float32"),calibration_support=scores.astype("float32"))
    temp.replace(target);joblib.dump(reducer,out/(name+"_umap.joblib"));joblib.dump(gate,out/(name+"_gate.joblib"))
    write_json(out/(name+"_quality.json"),{"source":name,"calibration_auc":auc,"calibration_ap":ap,
        "calibration_top60_high435":h,"source_quality":quality,"memory_n":len(ti),"calibration_n":len(ci),
        "test_labels_read":False,"quality_features_use_query_true_label":False})
    print("SOURCE_COMPLETE "+name,flush=True);return str(target)
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,default=Path(__file__).resolve().parent)
    ap.add_argument("--run",default="glom_seed20261013");ap.add_argument("--workers",type=int,default=2)
    a=ap.parse_args();root=a.root;feat=root/"artifacts/models/random"/a.run/"features"
    assert (feat/"DONE.json").exists()
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        tasks=[pool.submit(build_source,str(root),a.run,s) for s in SOURCES]
        for task in tasks:task.result()
    ix=pd.read_csv(feat/"row_index.csv");ti=np.flatnonzero(ix.partition.eq("train"))
    tr=pd.read_csv(root/"data/train.csv").set_index("row_id");ids=ix.row_id.iloc[ti].to_numpy()
    out=feat.parent/"terrain_train_only"
    arrays={"sample_idx":ids,"brightness_true":tr.loc[ids].brightness.to_numpy("float32"),
            "brightness_pred_model":np.load(feat/"prediction.npy")[ti],
            "sequence_projection":np.load(out/"sequence_projection.npy"),"sources":np.asarray(SOURCES,dtype=object)}
    quality=[]
    for s in SOURCES:
        part=np.load(out/(s+".npz"))
        for dest,src in [(f"rep_{s}","rep"),(f"xy_{s}","xy"),(f"local_z_{s}_true","zt"),
                         (f"local_z_{s}_pred","zp"),(f"gate_support_{s}","support"),
                         (f"gate_risk_{s}","risk"),(f"source_quality_{s}","quality")]:arrays[dest]=part[src]
        quality.append(json.loads((out/(s+"_quality.json")).read_text()))
    np.savez_compressed(out/"terrain.npz",**arrays)
    pd.DataFrame(quality).to_csv(out/"source_quality.csv",index=False)
    write_json(out/"DONE.json",{"memory_partition":"train","quality_partition":"calibration",
                "umap_fit_partition":"train","test_labels_read":False,"memory_n":len(ti),"sources":SOURCES})
if __name__=="__main__":main()

