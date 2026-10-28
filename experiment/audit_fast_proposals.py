"""Compare proposal order and accepted updates against the unchanged native implementation."""
from pathlib import Path
import time,dataclasses
import torch
from experiment_core import setup_project,write_json
ROOT=Path(__file__).resolve().parent;setup_project(ROOT)
from pipeline.tgito.optimizer import TGITOGenerator,GeneratorConfig
from pipeline.tgito.aa import seq_to_ids0,make_relaxed_logits_from_ids
from fast_proposals import FastProposalMixin

class Capture(TGITOGenerator):
    def __init__(self):self.device="cuda";self._anchor_pred=0.;self.candidates=None
    def _score_hard_ids_batch(self,ids,*args):
        self.candidates=ids.clone()
        scores=(ids.float()*torch.arange(1,222,device="cuda")[None,:]).sum(1)/1e5
        return scores,scores
class Fast(FastProposalMixin,Capture):pass

wt=(ROOT/"repro_project/data/processed/reference_sequence.txt").read_text().strip()
base=seq_to_ids0(wt,device="cuda");records=[]
for index,(flat,mutated,restricted) in enumerate([(False,False,False),(True,False,True),(False,True,True)]):
    torch.manual_seed(20261013+index);mask=torch.ones(221,dtype=torch.bool,device="cuda")
    if restricted:mask[40:]=False
    ids=base.clone()
    if mutated:ids[3]=(ids[3]+1)%20;ids[11]=(ids[11]+1)%20
    logits=make_relaxed_logits_from_ids(ids,221,mask,5.,0.,30.,"cuda")
    if not flat:logits+=torch.randn_like(logits)*.15
    grad=torch.zeros_like(logits) if flat else torch.randn_like(logits)*.01
    cfg=GeneratorConfig(mutation_budget=2,budget_relative="reference",proposal_candidates=64,proposal_order=2)
    native=Capture();fast=Fast();start=time.monotonic()
    left=native._try_discrete_proposals(logits,grad,base,base,mask,cfg,-1.,reference_seq=wt)
    native_seconds=time.monotonic()-start;start=time.monotonic()
    right=fast._try_discrete_proposals(logits,grad,base,base,mask,cfg,-1.,reference_seq=wt)
    fast_seconds=time.monotonic()-start
    assert torch.equal(native.candidates,fast.candidates),(index,"candidate order differs")
    assert torch.equal(left[0],right[0]) and left[1:]==right[1:],(index,"accepted update differs")
    records.append({"case":index,"flat_ties":flat,"preexisting_mutations":mutated,"restricted_sites":restricted,
        "ordered_candidates_identical":True,"accepted_logits_identical":True,
        "native_seconds":native_seconds,"fast_seconds":fast_seconds})
    print(records[-1],flush=True)
write_json(ROOT/"audit/fast_proposal_equivalence.json",{"passed":True,"cases":records})
print("FAST_PROPOSALS_EXACT_EQUIVALENCE_PASSED",flush=True)
