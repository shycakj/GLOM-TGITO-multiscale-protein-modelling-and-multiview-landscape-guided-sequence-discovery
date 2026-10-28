"""Exercise native search/ablation/AdaLead interfaces while the final model is training.

Uses a separate training-only 256-row smoke bank with PCA coordinates. Its results
are explicitly excluded from evidence tables; the full experiment uses train-only UMAP.
"""
from pathlib import Path
import os,sys,time,shutil,json,subprocess
os.environ.setdefault("OMP_NUM_THREADS","2")
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from experiment_core import setup_project,create_model,write_json
from train_models import HP
from build_train_terrain import SOURCES,sequence_rep,fields
ROOT=Path(__file__).resolve().parent;setup_project(ROOT);torch.set_num_threads(2)
source=ROOT/"artifacts/models/random/glom_seed20261013/best.pt"
base=ROOT/"artifacts/models/random/pipeline_smoke";base.mkdir(exist_ok=True)
for attempt in range(8):
    try:
        checkpoint=torch.load(source,map_location="cpu",weights_only=False);break
    except (RuntimeError,EOFError):time.sleep(1)
else:raise RuntimeError("Could not load live checkpoint")
torch.save(checkpoint,base/"best.pt")
model=create_model("glom").cuda();model.load_state_dict(checkpoint["model"],strict=True);model.eval()
d=pd.read_csv(ROOT/"data/train.csv").head(256).copy();feat=base/"features";feat.mkdir(exist_ok=True)
loader=DataLoader(HP(d,ROOT/"artifacts/embeddings/H.npy",ROOT/"artifacts/embeddings/P4_glom_original.npy"),batch_size=32)
reps={s:[] for s in SOURCES if s!="sequence"};reps["h32"]=[];pred=[]
with torch.inference_mode():
    for h,p,_ in loader:
        out=model.forward_all(h.cuda(),p.cuda())
        pred.extend((out["y_norm"]*checkpoint["y_std"]+checkpoint["y_mean"]).cpu().numpy())
        for s in reps:reps[s].append(out[s].cpu().numpy().reshape(len(h),-1))
reps={s:np.concatenate(v) for s,v in reps.items()};reps["sequence"],projection=sequence_rep(d.sequence.tolist())
for s,v in reps.items():np.save(feat/(s+".npy"),v)
d[["row_id","partition","mutation_depth"]].to_csv(feat/"row_index.csv",index=False)
np.save(feat/"prediction.npy",np.asarray(pred,"float32"))
write_json(feat/"DONE.json",{"smoke":True,"rows":len(d),"partition":"train"})
bankdir=base/"terrain_train_only";bankdir.mkdir(exist_ok=True)
arrays={"brightness_true":d.brightness.to_numpy("float32"),"brightness_pred_model":np.asarray(pred,"float32"),
        "sample_idx":d.row_id.to_numpy(),"sequence_projection":projection}
quality=[]
for s in SOURCES:
    xy=PCA(2,svd_solver="randomized",random_state=20261013).fit_transform(reps[s]).astype("float32")
    _,zt,zp,risk,h420,h435=fields(xy,arrays["brightness_true"],arrays["brightness_pred_model"])
    arrays.update({"rep_"+s:reps[s],"xy_"+s:xy,"local_z_"+s+"_true":zt,"local_z_"+s+"_pred":zp,
        "gate_support_"+s:(.65*(d.brightness.to_numpy()>=4.35)+.25*h435+.1*h420).astype("float32"),
        "gate_risk_"+s:np.clip(risk,0,4),"source_quality_"+s:np.asarray([.5],"float32")})
    quality.append({"source":s,"source_quality":.5})
np.savez_compressed(bankdir/"terrain.npz",**arrays)
pd.DataFrame(quality).to_csv(bankdir/"source_quality.csv",index=False)
write_json(bankdir/"DONE.json",{"smoke":True,"coordinates":"PCA for interface verification only","rows":len(d)})
del model,checkpoint;torch.cuda.empty_cache()
rc=subprocess.run([sys.executable,"run_search_comparison.py","--run","pipeline_smoke","--pilot"],cwd=ROOT).returncode
write_json(base/"smoke_status.json",{"exit_code":rc,"included_in_main_results":False})
if rc:raise RuntimeError("Search integration smoke failed")
print("NATIVE_SEARCH_INTEGRATION_SMOKE_PASSED",flush=True)
