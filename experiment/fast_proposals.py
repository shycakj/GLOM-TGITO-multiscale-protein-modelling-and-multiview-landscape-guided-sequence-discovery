"""Same TGITO discrete proposals, enumerated on CPU after one batched GPU top-k.

The original implementation synchronizes CUDA for every residue, candidate and
candidate pair. This adapter keeps the proposal policy and GPU scoring unchanged.
"""
import torch

class FastProposalMixin:
    def _try_discrete_proposals(self,logits,grad,base_ids,reference_ids,mutable_mask,cfg,
            current_hard_pred,current_objective=None,reference_seq="",avoid_signatures=None,archive_sequences=None):
        from pipeline.tgito.optimizer import format_mutations,normalize_mutation_signature,hamming
        from pipeline.tgito.aa import ids0_to_seq,make_relaxed_logits_from_ids
        if int(cfg.proposal_candidates)<=0:return logits,False,current_hard_pred,""
        avoid_signatures=avoid_signatures or set();archive_sequences=archive_sequences or []
        with torch.no_grad():
            cur_ids=logits.argmax(-1).detach();positions=torch.where(mutable_mask.bool().to(logits.device))[0]
            if positions.numel()==0:return logits,False,current_hard_pred,""
            scores=(logits-float(cfg.lr)*grad).detach().clone()
            scores.scatter_(1,cur_ids[:,None],-1e9)
            values,amino=torch.topk(scores,k=min(max(1,int(cfg.proposal_alts_per_pos)),20),dim=1)
            # Only these compact arrays cross the device boundary.
            values=values.cpu().tolist();amino=amino.cpu().tolist();current=cur_ids.cpu().tolist()
            budget=(reference_ids if cfg.budget_relative=="reference" else base_ids).cpu().tolist()
            def ok(test):
                if sum(a!=b for a,b in zip(test,budget))>float(cfg.mutation_budget)+1e-6:return False
                seq=ids0_to_seq(test)
                if normalize_mutation_signature(format_mutations(reference_seq,seq)) in avoid_signatures:return False
                if int(cfg.novelty_min_hamming)>0 and archive_sequences:
                    if min(hamming(seq,a) for a in archive_sequences)<int(cfg.novelty_min_hamming):return False
                return True
            singles=[]
            for i in positions.cpu().tolist():
                cur=current[i]
                for value,aa in zip(values[i],amino[i]):
                    test=current.copy();test[i]=aa
                    if ok(test):singles.append((float(value),[(int(i),cur,int(aa))],test))
            proposals=list(singles)
            if int(cfg.proposal_order)>=2 and len(singles)>=2:
                pool=sorted(singles,key=lambda x:x[0],reverse=True)[:64]
                seen={tuple(p[-1]) for p in proposals}
                for a in range(len(pool)):
                    for b in range(a+1,len(pool)):
                        ma,mb=pool[a][1],pool[b][1]
                        if ma[0][0]==mb[0][0]:continue
                        test=current.copy()
                        for pos,_,aa in ma+mb:test[pos]=aa
                        key=tuple(test)
                        if key in seen:continue
                        seen.add(key)
                        if ok(test):proposals.append((float(pool[a][0]+pool[b][0]),ma+mb,test))
            if not proposals:return logits,False,current_hard_pred,""
            proposals.sort(key=lambda x:x[0],reverse=True);proposals=proposals[:int(cfg.proposal_candidates)]
            ids_batch=torch.tensor([p[-1] for p in proposals],dtype=torch.long,device=self.device)
            preds,scores=self._score_hard_ids_batch(ids_batch,base_ids,reference_ids,mutable_mask,cfg)
            best=int(torch.argmax(scores).item());best_pred=float(preds[best].cpu());best_score=float(scores[best].cpu())
            _,mutations,_=proposals[best];best_ids=ids_batch[best]
            accept=best_pred>=float(self._anchor_pred)-float(cfg.anchor_drop)
            if current_objective is not None:accept=accept and best_score>=float(current_objective)+float(cfg.proposal_accept_margin)
            else:accept=accept and best_pred>=float(current_hard_pred)+float(cfg.proposal_accept_margin)
            if not accept:return logits,False,current_hard_pred,""
            new_logits=make_relaxed_logits_from_ids(best_ids,cfg.length,mutable_mask,cfg.mutable_anchor_logit,
                cfg.mutable_alt_logit,cfg.lock_scale,self.device)
            mutation=";".join(f"{pos+1}:{old}->{new}" for pos,old,new in mutations)
            return new_logits.detach(),True,best_pred,mutation
