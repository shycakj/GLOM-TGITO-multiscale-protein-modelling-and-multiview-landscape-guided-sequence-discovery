"""Audited numerical corrections, opt-in only. Historical optimizer stays intact.

Remove the softplus offset at the relaxed initialization and use a hard-sequence
objective when comparing/returning hard sequences. No terrain source is removed.
"""
import torch
import torch.nn.functional as F


class ConsistentObjectiveMixin:
    def _confident_change_mass(self, probs, ids0, mutable_mask, cfg, topk=None):
        if probs.ndim == 2:probs=probs[None]
        same=(probs*F.one_hot(ids0.to(probs.device).long(),20)[None]).sum(-1)
        change=(1-same).clamp(0,1)
        baseline=self._init_leakage_prob(cfg)
        threshold=max(.05,min(.75,baseline+float(cfg.change_count_threshold_margin)))
        tau=max(float(cfg.change_count_temp),1e-4)
        offset=F.softplus(change.new_tensor((baseline-threshold)/tau))*tau
        maximum=F.softplus(change.new_tensor((1-threshold)/tau))*tau
        site=((F.softplus((change-threshold)/tau)*tau-offset)/(maximum-offset)).clamp(0,1)
        site=site*mutable_mask.to(probs.device).bool()[None]
        if topk is None:topk=int(cfg.change_count_topk)
        if topk and 0<int(topk)<site.shape[1]:site=site.topk(int(topk),dim=1).values
        return site.sum(1)

    def _record_current(self,logits,base_ids,reference_ids,mutable_mask,cfg,*args,**kwargs):
        row,seq=super()._record_current(logits,base_ids,reference_ids,mutable_mask,cfg,*args,**kwargs)
        ids=logits.detach().argmax(-1).reshape(1,-1)
        _,score=self._score_hard_ids_batch(ids,base_ids,reference_ids,mutable_mask,cfg)
        row['soft_objective']=row['objective']
        row['hard_objective']=float(score[0].detach().cpu())
        row['objective']=row['hard_objective']
        return row,seq
