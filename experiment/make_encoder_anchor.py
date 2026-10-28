from pathlib import Path
import os,sys,json,numpy as np
BASE=Path(__file__).resolve().parent;PROJECT=BASE.parent/'handoff_20261001/repro_project'
os.environ.update(PPLU_ESM_MODEL_PATH=str(PROJECT/'assets/esm2_t30_150M_UR50D'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
sys.path.insert(0,str(PROJECT/'pipeline'))
import torch
from tgito.esm_bridge import load_bridge,DifferentiableESMBridge
from tgito.aa import seq_to_ids0
torch.set_num_threads(6)
bridge=load_bridge(PROJECT/'assets/warm_best_model.pt',device='cpu')
enc=DifferentiableESMBridge(bridge,device='cpu').eval()
wt=(PROJECT/'data/processed/reference_sequence.txt').read_text().strip()
with torch.inference_mode():
    ids=seq_to_ids0(wt,221,device='cpu')[None]
    out=enc(torch.nn.functional.one_hot(ids,20).float())
np.savez_compressed(BASE/'encoder_anchor.npz',H=out['H'].numpy(),esm_mean=out['esm'].mean(1).numpy())
print(json.dumps({'anchor_bytes':(BASE/'encoder_anchor.npz').stat().st_size,'H_mean':float(out['H'].mean()),'H_std':float(out['H'].std())}),flush=True)
