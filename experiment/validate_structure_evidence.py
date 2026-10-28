"""Validate a fixed ProteinMPNN metric on real assays, then score frozen designs.

This evaluates compatibility with an experimental backbone, not brightness.
No sequence is sent to an external inference service; all scoring runs locally.
"""
from pathlib import Path
import argparse,json,sys,time,os
os.environ.setdefault('OMP_NUM_THREADS','3')
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score,average_precision_score
from experiment_core import write_json
from build_wide_peak_plan import sha


class Scorer:
    def __init__(self,root):
        sys.path.insert(0,str(root/'third_party/ProteinMPNN'))
        from protein_mpnn_utils import ProteinMPNN
        self.root=root;self.out=root/'audit/structure_validation';torch.set_num_threads(3)
        self.device='cuda';b=np.load(self.out/'reference_backbone.npz')
        self.wt=str(b['wt']);self.mask=b['mask'];self.distance=b['chromophore_distance']
        self.X=torch.tensor(b['X'],device=self.device)[None]
        self.M=torch.tensor(self.mask,device=self.device)[None]
        self.idx=torch.arange(221,device=self.device)[None]
        self.chain=torch.ones((1,221),dtype=torch.long,device=self.device)
        self.code={a:i for i,a in enumerate('ACDEFGHIKLMNPQRSTVWYX')}
        ckpt=torch.load(root/'third_party/ProteinMPNN/vanilla_model_weights/v_48_020.pt',map_location='cpu',weights_only=False)
        self.model=ProteinMPNN(num_letters=21,node_features=128,edge_features=128,hidden_dim=128,
            num_encoder_layers=3,num_decoder_layers=3,augment_eps=0.,k_neighbors=ckpt['num_edges']).to(self.device).eval()
        self.model.load_state_dict(ckpt['model_state_dict'],strict=True)
        rng=np.random.default_rng(20261013)
        self.orders=torch.tensor(rng.standard_normal((4,221)).astype('float32'),device=self.device)
        self.reference_ll=self.per_site([self.wt])[0]

    @torch.inference_mode()
    def per_site(self,sequences):
        n=len(sequences);S=torch.tensor([[self.code[a] for a in s] for s in sequences],device=self.device)
        values=[]
        for order in self.orders:
            lp=self.model(self.X.expand(n,-1,-1,-1),S,self.M.expand(n,-1),self.M.expand(n,-1),
                self.idx.expand(n,-1),self.chain.expand(n,-1),order[None].expand(n,-1))
            values.append(torch.gather(lp,2,S[:,:,None]).squeeze(-1).cpu().numpy())
        return np.mean(values,axis=0)

    def score(self,sequences):
        rows=[];wtchars=np.asarray(list(self.wt))
        for st in range(0,len(sequences),16):
            seqs=sequences[st:st+16];values=self.per_site(seqs)
            for sequence,ll in zip(seqs,values):
                changed=np.asarray(list(sequence))!=wtchars;scored=changed&(self.mask==1)
                n=int(scored.sum());excluded=int((changed&(self.mask==0)).sum())
                rows.append(dict(sequence=sequence,mpnn_log_likelihood=float(np.sum(ll*self.mask)/self.mask.sum()),
                    mpnn_delta_log_likelihood_vs_wt=float(np.sum((ll-self.reference_ll)*self.mask)/self.mask.sum()),
                    mpnn_mutated_site_log_likelihood=float(ll[scored].mean()) if n else None,
                    mutation_depth=int(changed.sum()),scored_mutations=n,unscored_mutations=excluded,
                    canonical_chromophore_preserved=sequence[55:58]==self.wt[55:58],
                    mutations_within_8A_reference_chromophore=int((changed&(self.distance<=8)).sum())))
            if st%512==0:print(json.dumps(dict(scored=min(st+16,len(sequences)),total=len(sequences))),flush=True)
        return pd.DataFrame(rows)


def validate(root,scorer):
    out=scorer.out
    if (out/'ASSAY_VALIDATION_DONE.json').exists():return
    protocol=dict(metric='ProteinMPNN v_48_020 fixed experimental backbone log likelihood',
        reference='2G3O chain A exactly mapped to 221-aa pplu reference',decoding_orders=4,
        order_seed=20261013,noise=0.,thresholds=[4.2,4.35],splits=['calibration','test'],
        frozen_subsets=['all','depth_le2','depth_ge3','search_pred_ge4.31'],
        no_functional_labels_used_for_scores=True,no_selection_of_metric_after_test=True,
        interpretations=['Backbone compatibility','Empirical association with measured fluorescence'],
        excluded_residues=json.loads((out/'REFERENCE_MAPPING.json').read_text())['excluded_positions'],
        model_manifest=json.loads((root/'third_party/ProteinMPNN/DOWNLOAD_VERIFIED.json').read_text()),
        source_sha256=sha(root/'validate_structure_evidence.py'))
    write_json(out/'VALIDATION_PROTOCOL.json',protocol)
    sequence_rows=pd.concat([pd.read_csv(root/f'data/{p}.csv',usecols=['row_id','sequence']).assign(partition=p) for p in protocol['splits']])
    unique=sorted(set(sequence_rows.sequence));scores=scorer.score(unique).set_index('sequence')
    scored=sequence_rows.join(scores,on='sequence');scored.to_csv(out/'frozen_assay_scores.csv',index=False)
    write_json(out/'ASSAY_SCORES_FROZEN.json',dict(sha256=sha(out/'frozen_assay_scores.csv'),rows=len(scored),labels_read=False))
    metrics=[]
    for part in protocol['splits']:
        d=scored[scored.partition.eq(part)].copy()
        truth=pd.read_csv(root/f'data/{part}.csv').set_index('row_id');d['brightness']=d.row_id.map(truth.brightness)
        pred=pd.read_csv(root/f'artifacts/models/random/glom_seed20261013/{part}_predictions.csv').set_index('row_id')
        column='prediction' if 'prediction' in pred else 'pred';d['search_pred']=d.row_id.map(pred[column])
        for subset,mask in dict(all=np.ones(len(d),bool),depth_le2=d.mutation_depth.le(2),depth_ge3=d.mutation_depth.ge(3),search_pred_ge4_31=d.search_pred.ge(4.31)).items():
            x=d[mask]
            for metric in ['mpnn_log_likelihood','mpnn_mutated_site_log_likelihood']:
                q=x.dropna(subset=[metric]);y=q.brightness.to_numpy();s=q[metric].to_numpy()
                if len(q)<3:continue
                for threshold in protocol['thresholds']:
                    event=y>=threshold
                    metrics.append(dict(partition=part,subset=subset,metric=metric,n=len(q),threshold=threshold,
                        prevalence=float(event.mean()),auroc=float(roc_auc_score(event,s)) if len(set(event))==2 else None,
                        auprc=float(average_precision_score(event,s)) if event.any() else None,
                        spearman=float(spearmanr(y,s).statistic),unscored_mutation_fraction=float(q.unscored_mutations.gt(0).mean())))
        d.to_csv(out/f'{part}_scores_with_assays.csv',index=False)
    pd.DataFrame(metrics).to_csv(out/'assay_validation_metrics.csv',index=False)
    write_json(out/'ASSAY_VALIDATION_DONE.json',dict(rows=len(scored),metric_rows=len(metrics),
        purpose='Validate discrimination before using this auxiliary metric to interpret designs',
        not_measured_generated_activity=True))


def score_prefix(root,scorer,mountains):
    dest=root/f'artifacts/models/random/glom_seed20261013/full_discovery_v2/prefix_{mountains:03d}'
    if (dest/'STRUCTURE_DONE.json').exists():return
    frozen=json.loads((dest/'FROZEN.json').read_text())
    for name,digest in frozen['files'].items():assert sha(dest/name)==digest
    chosen=pd.read_csv(dest/'selected.csv')
    starts=pd.read_csv(dest.parent/'starting_sequences.csv').set_index('start_id')
    seqs=sorted(set(chosen.sequence)|set(starts.loc[chosen.start_id].sequence)|{scorer.wt})
    values=scorer.score(seqs).set_index('sequence');values.to_csv(dest/'structure_compatibility_scores.csv')
    d=chosen.join(values.drop(columns=['mutation_depth']),on='sequence')
    d['mpnn_gain_vs_start']=d.mpnn_log_likelihood-d.start_id.map(starts.sequence).map(values.mpnn_log_likelihood)
    d.to_csv(dest/'selected_structure_evaluated.csv',index=False)
    summary=d.groupby('batch_size').agg(selected_n=('sequence','size'),mpnn_ll_mean=('mpnn_log_likelihood','mean'),
        mpnn_gain_vs_start=('mpnn_gain_vs_start','mean'),unscored_mutations_mean=('unscored_mutations','mean'),
        canonical_chromophore_fraction=('canonical_chromophore_preserved','mean'))
    summary.to_csv(dest/'structure_summary.csv')
    write_json(dest/'STRUCTURE_DONE.json',dict(mountains=mountains,rows=len(d),scores_sha256=sha(dest/'structure_compatibility_scores.csv'),
        validation='audit/structure_validation/assay_validation_metrics.csv',evaluated_after_freeze=True,
        fixed_template=True,candidate_structure_not_predicted=True,source_sha256=sha(root/'validate_structure_evidence.py')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--follow',action='store_true');a=p.parse_args();scorer=Scorer(a.root);validate(a.root,scorer)
    deadline=time.monotonic()+36*3600;completed=set()
    while True:
        for m in [12,60,180,900]:
            f=a.root/f'artifacts/models/random/glom_seed20261013/full_discovery_v2/prefix_{m:03d}/FROZEN.json'
            if m not in completed and f.exists():score_prefix(a.root,scorer,m);completed.add(m)
        if not a.follow or len(completed)==4:break
        if time.monotonic()>deadline:raise TimeoutError('36h follow bound; scores retained')
        time.sleep(30)
