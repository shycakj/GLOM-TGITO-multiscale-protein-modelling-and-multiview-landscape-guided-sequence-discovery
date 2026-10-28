"""Check that the weighted multiterrain term changes AA-logit gradients."""
from pathlib import Path
import json
import pandas as pd
import torch
from discovery_v3 import paths,load_generator
from experiment_core import seed_all,write_json
root=Path(__file__).resolve().parent;g,cfg,effective=load_generator(root,'trial12')
from pipeline.tgito.aa import seq_to_ids0
wt=(root/'repro_project/data/processed/reference_sequence.txt').read_text().strip()
ref=seq_to_ids0(wt,221,'cuda');mask=torch.ones(221,dtype=torch.bool,device='cuda')
plan=pd.read_csv(paths(root)[1]/'pilot_starting_sequences.csv');rows=[]
for start in plan.drop_duplicates('role').itertuples():
    seed_all(int(start.random_seed));ids=seq_to_ids0(start.sequence,221,'cuda')
    logits=(torch.nn.functional.one_hot(ids,20).float()*cfg.mutable_anchor_logit+torch.randn(221,20,device='cuda')*cfg.noise_scale).requires_grad_(True)
    probs=(logits/cfg.temp_start).softmax(-1)[None]
    y,terrain,_,_=g._forward_from_probs(probs,ids,ref,mask,cfg)
    gy=torch.autograd.grad(y.sum(),logits,retain_graph=True)[0]
    gt=torch.autograd.grad(terrain['terrain'].sum(),logits)[0]
    assert torch.isfinite(gy).all() and torch.isfinite(gt).all() and gt.norm()>0 and gy.norm()>0
    rows.append(dict(role=start.role,start_id=int(start.start_id),brightness_gradient_norm=float(gy.norm()),
                     terrain_gradient_norm=float(gt.norm()),weighted_terrain_gradient_norm=float((gt*cfg.terrain_weight).norm()),
                     weighted_terrain_to_brightness_gradient_ratio=float((gt*cfg.terrain_weight).norm()/gy.norm()),
                     cosine_between_gradient_terms=float(torch.nn.functional.cosine_similarity(gy.flatten(),gt.flatten(),dim=0))))
report=dict(passed=True,checks=rows,meaning='Nonzero weighted terrain gradient enters the native sequence objective, in addition to brightness.',
            terrain_weight=cfg.terrain_weight,topk_sources=effective['terrain_config']['topk_sources'],test_labels_read=False)
write_json(root/'audit/v3_terrain_gradient_contract.json',report);print(json.dumps(report),flush=True)
