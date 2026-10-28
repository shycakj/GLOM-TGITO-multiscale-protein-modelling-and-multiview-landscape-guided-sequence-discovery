"""v13.5 summit/ridge/edge discovery design, rebuilt on current training data.

Historical thresholds, source priors, peak/group scores and micro-dedup are kept.
The memory is now train-only; source quality comes from the frozen risk layer.
No validation, calibration or test sequence is eligible as a starting sequence.
"""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import pandas as pd
from experiment_core import write_json

SOURCES=['global_attn_pool','global_mean_pool','global_max_pool','region9_max_pool',
 'region9_mean_pool','region_win_165_174_all_mut512','region_win_77_86_all_mut512',
 'keysite_K_flat','region_win_97_106_all_mut512','region_win_27_36_all_mut512',
 'region_win_55_64_all_mut512','sequence']
PRIOR=dict(zip(SOURCES,[1.15,1.20,.85,1.10,.95,1.05,.90,.85,.80,.90,.75,1.25]))

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def find_peaks(xy,y,n=50):
    bins=[np.linspace(xy[:,j].min()-.01,xy[:,j].max()+.01,n+1) for j in range(2)]
    cells=np.stack([np.clip(np.digitize(xy[:,j],bins[j])-1,0,n-1) for j in range(2)],1)
    maxima={}
    for i,v in enumerate(y):
        c=tuple(cells[i])
        if c not in maxima or v>maxima[c][1]:maxima[c]=(i,float(v))
    peaks=[]
    for (cx,cy),(i,v) in maxima.items():
        if all((cx+dx,cy+dy) not in maxima or maxima[(cx+dx,cy+dy)][1]<v
               for dx in (-1,0,1) for dy in (-1,0,1) if dx or dy):
            rows=np.flatnonzero((np.abs(cells-[cx,cy])<=1).all(1))
            peaks.append((i,rows))
    return peaks

def build(root,target=900,output=None,source_prior=None):
    base=root/'artifacts/models/random/glom_seed20261013'
    out=Path(output) if output is not None else base/'wide_search';out.mkdir(parents=True,exist_ok=True)
    prior=dict(PRIOR if source_prior is None else source_prior)
    assert set(prior)==set(SOURCES) and all(v>0 for v in prior.values())
    if (out/'PLAN_FROZEN.json').exists():
        info=json.loads((out/'PLAN_FROZEN.json').read_text())
        assert sha(out/'starting_sequences.csv')==info['plan_sha256']
        print(json.dumps(info));return
    memory=base/'terrain_train_only/terrain.npz'
    z=np.load(memory,allow_pickle=True)
    master=pd.read_csv(root/'data/train.csv').set_index('row_id').loc[z['sample_idx']].reset_index()
    assert len(master)==len(z['brightness_true']) and master.sequence.is_unique
    y=master.brightness.to_numpy(float)
    assert np.allclose(y,z['brightness_true'],atol=1e-6)
    top=set(np.argsort(-y)[:200]);raw=[];source_counts={}
    q=pd.read_csv(base/'terrain_train_only/source_quality.csv').set_index('source')
    xy_by={}
    for s in SOURCES:
        xy=xy_by[s]=np.asarray(z['xy_'+s],float);quality=float(q.loc[s,'source_quality'])
        # Historical code uses each source's mean risk over observed high rows.
        risk=float(np.asarray(z['gate_risk_'+s])[y>=4.2].mean())
        found=find_peaks(xy,y);kept=[]
        for pi,rr in found:
            yy=y[rr];has_top=bool(top.intersection(rr));high=rr[yy>=4.2]
            if (yy.mean()<=3.9 and yy.max()<4.2 and not has_top) or not len(high):continue
            score=(.22*quality+.18*np.mean(yy>=4.30)+.16*np.clip((yy.max()-4.2)/.25,0,1)
                   +.12*np.mean(yy>=4.35)+.12/(1+risk)+.10*prior[s]
                   +.06*has_top+.04*np.clip((y[pi]-4.2)/.25,0,1))
            kept.append(dict(source=s,peak_index=int(pi),mean_true=float(yy.mean()),
                max_true=float(yy.max()),q90_true=float(np.quantile(yy,.9)),
                frac_ge435=float(np.mean(yy>=4.35)),score=float(score),quality=quality,
                source_risk=risk,rowset=set(map(int,rr)),high_rows=set(map(int,high))))
        kept=sorted(kept,key=lambda x:x['score'],reverse=True)[:120]
        raw.extend(kept);source_counts[s]=dict(raw=len(found),kept=len(kept))
    parent=list(range(len(raw)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i,a in enumerate(raw):
        for j in range(i+1,len(raw)):
            b=raw[j]
            if (abs(a['mean_true']-b['mean_true'])<=.035 and abs(a['q90_true']-b['q90_true'])<=.05
                and abs(a['frac_ge435']-b['frac_ge435'])<=.06
                and len(a['rowset']&b['rowset'])/max(1,len(a['rowset']|b['rowset']))>=.92):
                parent[find(j)]=find(i)
    groups={}
    for i,r in enumerate(raw):groups.setdefault(find(i),[]).append(r)
    mountains=[];seeds=[]
    for mi,members in enumerate(groups.values()):
        rep=max(members,key=lambda x:x['score']);s=rep['source'];mid=f'W{mi:04d}'
        rows=sorted(set.union(*(r['rowset'] for r in members)))
        high=np.array(sorted(set.union(*(r['high_rows'] for r in members))),int)
        yy=y[rows];ns=len(set(r['source'] for r in members))
        score=(.22*np.mean([r['quality'] for r in members])+.18*np.mean(yy>=4.30)
               +.16*np.clip((yy.max()-4.2)/.25,0,1)+.12*np.mean(yy>=4.35)
               +.12/(1+np.mean([r['source_risk'] for r in members]))
               +.10*np.mean([prior[r['source']] for r in members])
               +.06*(1 if ns==1 else min(.6,.10*ns))+.04*min(1,ns/4))
        mountains.append(dict(mountain_id=mid,representative_source=s,mountain_score=float(score),
             n_sources=ns,n_local_rows=len(rows),n_high_rows=len(high),summit_true=float(yy.max())))
        summit=int(high[np.argmax(y[high])]);xy=xy_by[s];dist=np.linalg.norm(xy[high]-xy[summit],axis=1)
        ranked=high[np.argsort(-(np.clip((y[high]-4.2)/.25,0,1)+.30*dist/(np.percentile(dist,90)+1e-6)))]
        ridge=next((int(k) for k in ranked if k!=summit),summit)
        edge_pool=high[y[high]>=max(4.2,min(4.35,float(np.median(y[high]))))]
        edge=int(edge_pool[np.argmax(np.linalg.norm(xy[edge_pool]-xy[summit],axis=1))])
        for role,k in [('summit',summit),('ridge',ridge),('edge',edge)]:
            r=master.iloc[k]
            seeds.append(dict(mountain_id=mid,representative_source=s,mountain_score=float(score),
                role=role,row_id=int(r.row_id),sequence=r.sequence,brightness=float(r.brightness)))
    mdf=pd.DataFrame(mountains).sort_values(['mountain_score','summit_true'],ascending=False)
    by={s:g.mountain_id.tolist() for s,g in mdf.groupby('representative_source')}
    scores=mdf.set_index('mountain_id').mountain_score.to_dict();ptr={s:0 for s in by};selected=[]
    while len(selected)<min(target,len(mdf)):
        order=sorted(by,key=lambda s:(prior[s],scores[by[s][ptr[s]]] if ptr[s]<len(by[s]) else -1),reverse=True)
        for s in order:
            if ptr[s]<len(by[s]):selected.append(by[s][ptr[s]]);ptr[s]+=1
            if len(selected)>=min(target,len(mdf)):break
    sdf=pd.DataFrame(seeds);sdf=sdf[sdf.mountain_id.isin(selected)].copy()
    sdf['mountain_order']=sdf.mountain_id.map({m:i for i,m in enumerate(selected)})
    sdf['role_order']=sdf.role.map({'summit':0,'ridge':1,'edge':2})
    sdf=sdf.sort_values(['mountain_order','role_order']).reset_index(drop=True)
    sdf['start_id']=np.arange(len(sdf));sdf['cluster']=sdf.mountain_order
    sdf['random_seed']=20261013+sdf.start_id*1009
    # Fixed source-balanced prefixes are set before any generated/test outcomes.
    sdf['ablation_subset']=sdf.mountain_order<180
    assert len(sdf)==3*len(selected) and set(sdf.sequence)<=set(master.sequence)
    sdf.to_csv(out/'starting_sequences.csv',index=False)
    mdf[mdf.mountain_id.isin(selected)].to_csv(out/'mountains_selected.csv',index=False)
    info=dict(target_mountains=target,selected_mountains=len(selected),seed_contexts=len(sdf),
        unique_start_sequences=int(sdf.sequence.nunique()),roles=sdf.role.value_counts().to_dict(),
        raw_retained_peaks=len(raw),dedup_mountains=len(mdf),per_source=source_counts,
        terrain_rows=len(master),starting_partition='train',test_labels_read=False,source_prior=prior,
        plan_sha256=sha(out/'starting_sequences.csv'),model_sha256=sha(base/'best.pt'),
        terrain_sha256=sha(memory),risk_config_sha256=sha(base/'risk_calibration/FROZEN_CONFIG.json'),
        split_manifest_sha256=sha(root/'data/split_manifest.json'),
        historical_design='v13.5: 50x50, 3x3 cells, strict local maxima, conservative micro-dedup, source balance, summit/ridge/edge',
        differences=['train-only memory instead of production all-label memory',
            'current frozen calibration quality and mean train-high risk',
            'execution order follows source-balanced selection for valid nested prefixes'],
        ablation_mountains=min(180,len(selected)),
        independence='2700 contexts are not 2700 unique proteins; report unique starts and cluster by shared starts')
    write_json(out/'PLAN_FROZEN.json',info);print(json.dumps(info,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--target',type=int,default=900);a=p.parse_args();build(a.root,a.target)
