from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import joblib, numpy as np, pandas as pd, torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from umap import UMAP
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'src'))
from pipeline.tgito.revised_slim_model import RevisedSlimModel
from pplu_repro.common import set_all_seeds
SOURCES=['global_attn_pool','global_mean_pool','global_max_pool','region9_max_pool','region9_mean_pool','region_win_165_174_all_mut512','region_win_77_86_all_mut512','keysite_K_flat','region_win_97_106_all_mut512','region_win_27_36_all_mut512','region_win_55_64_all_mut512','sequence']

def p10(p):
    if p.shape[-1]==10:return p.astype('float32')
    o=np.zeros((*p.shape[:-1],10),dtype='float32'); o[..., [0,7,8,9]]=p.astype('float32'); return o

def knn_feats(xy,bt,bp,k=48):
    nn=NearestNeighbors(n_neighbors=min(k+1,len(xy))).fit(xy); ix=nn.kneighbors(xy,return_distance=False)[:,1:]; nt=bt[ix]; npd=bp[ix]; mt=nt.mean(1); st=nt.std(1)+1e-6; mp=npd.mean(1); sp=npd.std(1)+1e-6; zt=(bt-mt)/st; zp=(bp-mp)/sp; h420=(nt>=4.20).mean(1); h435=(nt>=4.35).mean(1); pg=np.maximum(zp-zt,0); risk=np.maximum(zp,0)*np.maximum(-zt,0)+pg
    X=np.stack([xy[:,0],xy[:,1],bp,mp,sp,zp,mt,st,zt,h420,h435,pg,risk],1).astype('float32'); return X,zt.astype('float32'),zp.astype('float32'),risk.astype('float32'),h420.astype('float32'),h435.astype('float32')

def mk(seed): return make_pipeline(StandardScaler(),MLPClassifier(hidden_layer_sizes=(48,24),activation='relu',alpha=1e-4,learning_rate_init=5e-4,max_iter=220,early_stopping=True,validation_fraction=.15,random_state=seed))
def fit_gate_oof(X,bt,tidx,source,seed,folds=5):
    y=(bt>=4.35).astype(int)
    if y[tidx].sum()<20:y=(bt>=4.30).astype(int)
    yt=y[tidx]
    if len(np.unique(yt))<2: raise RuntimeError(f'one class in tune300: {source}')
    sk=StratifiedKFold(n_splits=min(folds,int(np.bincount(yt).min())),shuffle=True,random_state=seed); oof=np.full(len(tidx),np.nan,'float32')
    for fi,(tr,va) in enumerate(sk.split(tidx,yt)):
        m=mk(seed+fi*101); m.fit(X[tidx[tr]],yt[tr]); oof[va]=m.predict_proba(X[tidx[va]])[:,1]
    auc=float(roc_auc_score(yt,oof)); ap=float(average_precision_score(yt,oof)); topn=max(10,min(60,int(.1*len(tidx)))); ii=tidx[np.argsort(-oof)[:topn]]
    final=mk(seed+999); final.fit(X[tidx],yt); prob=final.predict_proba(X)[:,1].astype('float32')
    return final,prob,{'auc_oof_tune':auc,'ap_oof_tune':ap,'top_true_mean_oof_tune':float(bt[ii].mean()),'top_high435_oof_tune':float((bt[ii]>=4.35).mean()),'topn_tune':topn}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',default=str(ROOT)); ap.add_argument('--config',default=str(ROOT/'configs/publication_experiment.json')); ap.add_argument('--seed',type=int,default=20260902); ap.add_argument('--batch',type=int,default=64); a=ap.parse_args(); root=Path(a.root); cfg=json.loads(Path(a.config).read_text()); set_all_seeds(a.seed)
    tr=pd.read_csv(root/'data/processed/search/glom_train.csv'); va=pd.read_csv(root/'data/processed/search/glom_val.csv'); tune=pd.read_csv(root/'data/processed/search/tgito_tune_300.csv'); fit=pd.concat([tr,va],ignore_index=True); all_df=pd.concat([fit,tune],ignore_index=True); tidx=np.arange(len(fit),len(all_df))
    # Strict: no sealed hidden file is opened here. Reuse the original row-aligned H/P arrays.
    Hpath=Path(os.getenv('PPLU_H_NPY',cfg['data']['h_embeddings'])); Ppath=Path(os.getenv('PPLU_P4_NPY',cfg['data']['p4_features'])); # _safe_mmap for raw binary H/P
    def _sm(p,shape,dtype):
        ed=np.dtype(dtype); eb=int(np.prod(shape))*ed.itemsize
        with open(str(p),'rb') as f: mg=f.read(6)
        if mg==b'\x93NUMPY': return np.load(str(p),mmap_mode='r')
        if p.stat().st_size!=eb: raise ValueError(f'{p}: size mismatch')
        return np.memmap(str(p),mode='r',dtype=ed,shape=tuple(shape))
    N4=int(Ppath.stat().st_size//(221*4*4)); H=_sm(Hpath,[N4,221,1280],np.float16); P=_sm(Ppath,[N4,221,4],np.float32); ids=all_df.row_id.to_numpy(int)
    ck=root/'artifacts/glom_current/best_model_pplu_current_split.pt'; obj=torch.load(ck,map_location='cpu',weights_only=False); dev='cuda' if torch.cuda.is_available() else 'cpu'; model=RevisedSlimModel(D=1280).to(dev); model.load_state_dict(obj['model'],strict=True); model.eval(); ym=float(obj['y_mean']); ys=float(obj['y_std'])
    buckets={s:[] for s in SOURCES if s!='sequence'}; preds=[]
    with torch.inference_mode():
      for st in range(0,len(ids),a.batch):
        jj=ids[st:st+a.batch]; hb=torch.from_numpy(np.asarray(H[jj],dtype='float32')).to(dev); pb=torch.from_numpy(p10(np.asarray(P[jj]))).to(dev); oo=model.forward_all(hb,pb); preds.append((oo['y_norm']*ys+ym).cpu().numpy())
        for s in buckets:buckets[s].append(oo[s].float().cpu().numpy())
    reps={s:np.concatenate(v).astype('float32') for s,v in buckets.items()}; bp=np.concatenate(preds).astype('float32'); bt=all_df.brightness.to_numpy('float32'); L=221
    # Sequence view keeps a fixed random projection of one-hot sequence, exactly reproducible.
    rng=np.random.default_rng(13202); proj=rng.normal(0,1/(L*20)**.5,size=(L*20,512)).astype('float32'); amap={x:i for i,x in enumerate('ACDEFGHIKLMNPQRSTVWY')}; one=np.zeros((len(all_df),L,20),'float32')
    for i,s in enumerate(all_df.sequence.astype(str)):
      for j,x in enumerate(s):one[i,j,amap[x]]=1
    reps['sequence']=one.reshape(len(all_df),-1)@proj; del one
    out=root/'artifacts/tgito_blind'; out.mkdir(parents=True,exist_ok=True); all_df.to_csv(out/'terrain_master_observed.csv',index=False); models=out/'gate_models'; models.mkdir(exist_ok=True)
    arrays={'sample_idx':ids.astype('int64'),'brightness_true':bt,'brightness_pred_model':bp,'sequence_projection':proj,'sources':np.asarray(SOURCES,dtype=object),'blind_note':np.asarray(['Strict observed-only terrain. GLOM fit rows + tune300 only; hidden400 never opened. Gate quality uses OOF tune300; source prior is uniform.'],dtype=object)}; quality=[]; tc=cfg['terrain']
    for si,s in enumerate(SOURCES):
      rep=reps[s]; reducer=UMAP(n_components=2,metric='cosine',n_neighbors=int(tc['umap_neighbors']),min_dist=float(tc['umap_min_dist']),random_state=13202+si,n_jobs=1); xy=reducer.fit_transform(rep).astype('float32'); X,zt,zp,risk,h420,h435=knn_feats(xy,bt,bp,int(tc['local_knn'])); gate,prob,met=fit_gate_oof(X,bt,tidx,s,13202+si,int(tc['gate_oof_folds'])); support=np.clip(.65*prob+.25*h435+.10*h420,0,1.5).astype('float32'); gr=np.clip(.70*risk+.30*(1-prob)*np.maximum(zp,0),0,4).astype('float32')
      # Headline has no historical source prior. Quality comes only from OOF tune300 performance.
      lq=float(np.clip(.70*np.nan_to_num(met['auc_oof_tune'],nan=.5)+.30*met['top_high435_oof_tune'],.05,1.5))
      arrays[f'rep_{s}']=rep; arrays[f'xy_{s}']=xy
      arrays[f'local_z_{s}_true']=zt; arrays[f'local_z_{s}_pred']=zp
      arrays[f'gate_support_{s}']=support; arrays[f'gate_risk_{s}']=gr
      arrays[f'source_quality_{s}']=np.asarray([lq],'float32'); joblib.dump(gate,models/f'gate_{s}.joblib'); quality.append({'source':s,'learned_quality':lq,'prior_used':1.0,**met}); print(f'[terrain] {s} q={lq:.3f} auc_oof={met["auc_oof_tune"]:.3f}',flush=True)
    np.savez_compressed(out/'blind_observed_12source_terrain.npz',**arrays); pd.DataFrame(quality).to_csv(out/'source_quality.csv',index=False); rep={'hidden_loaded':False,'hidden_sequence_used':False,'glom_fit_n':len(fit),'tgito_tune_n':len(tune),'terrain_memory_n':len(all_df),'H_reused':str(Hpath),'P_reused':str(Ppath),'gate_quality':'5-fold OOF on tune300','source_prior_mode':'uniform','umap_fit_rows':'observed only'}; (out/'report.json').write_text(json.dumps(rep,indent=2),encoding='utf-8'); print(json.dumps(rep,indent=2))
if __name__=='__main__':main()
