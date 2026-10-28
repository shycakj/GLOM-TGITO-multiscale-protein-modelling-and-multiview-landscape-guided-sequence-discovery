"""Canonical P4 inputs reconstructed from the original GLOM training source."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pandas as pd
AA20="ACDEFGHIKLMNPQRSTVWY"
AAP={"A":"H","V":"H","I":"H","L":"H","M":"H","F":"H","W":"H","P":"H","G":"H",
     "S":"P","T":"P","N":"P","Q":"P","C":"P","Y":"P",
     "D":"C","E":"C","K":"C","R":"C","H":"C"}
TABLE=np.asarray([[float(AAP[a]==g) for g in "HPC"] for a in AA20],"float32")
def canonical_bits(device="cpu"):
    import torch
    return torch.tensor(TABLE,device=device)
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,default=Path(__file__).resolve().parent)
    ap.add_argument("--project",type=Path);a=ap.parse_args();root=a.root
    project=a.project or root/"repro_project"
    d=pd.read_csv(project/"data/raw/pplu_31480_sequences.csv",usecols=["sequence"])
    wt=(project/"data/processed/reference_sequence.txt").read_text().strip()
    code={c:i for i,c in enumerate(AA20)}
    ids=np.asarray([[code[c] for c in s] for s in d.sequence],dtype="int64")
    ref=np.asarray([code[c] for c in wt]);mutation=ids!=ref[None]
    p=np.zeros((len(d),221,4),"float32");p[:,:,0]=mutation
    p[:,:,1:]=TABLE[ids]*mutation[:,:,None]
    out=root/"artifacts/embeddings";out.mkdir(parents=True,exist_ok=True)
    np.save(out/"P4_glom_original.npy",p)
    # Independent differentiable implementation must match every row, not just WT.
    import sys,torch
    sys.path.insert(0,str(project))
    from pipeline.tgito.aa import soft_pos_feats
    maxdiff=0.
    for start in range(0,len(ids),256):
        t=torch.from_numpy(ids[start:start+256]);prob=torch.nn.functional.one_hot(t,20).float()
        live=soft_pos_feats(prob,torch.from_numpy(ref),canonical_bits())[:,:,[0,7,8,9]].numpy()
        maxdiff=max(maxdiff,float(np.max(np.abs(live-p[start:start+len(t)]))))
    if maxdiff!=0:raise AssertionError("Canonical P4 differs between training and live generation")
    original=project/"assets/pplu_pmutation_p4.npy"
    old=np.load(original,mmap_mode="r") if original.exists() else None
    report={"rows":len(d),"shape":list(p.shape),"columns":["mutation_flag","mutant_hydrophobic","mutant_polar","mutant_charged"],
        "definition_source":"pipeline/glom_model/train_pplu_full_model.py, AAP and P_all construction",
        "history_finding":"pplu_pmutation_p4 was built for v12.5 miner on 2026-06-26; it encodes full-residue hydrophobic/polar flags with final column zero, not original GLOM inputs",
        "old_cache_different_entries_by_column":np.sum(p!=old,axis=(0,1)).tolist() if old is not None else None,
        "live_vs_cached_max_difference_all_31480":maxdiff,
        "sha256":hashlib.sha256((out/"P4_glom_original.npy").read_bytes()).hexdigest(),
        "brightness_labels_read":False}
    (out/"P4_feature_audit.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)
if __name__=="__main__":main()

