"""Repeat the exact full six-method smoke fixture after proposal enumeration acceleration."""
from pathlib import Path
import subprocess,sys,json
import numpy as np
import pandas as pd
from experiment_core import write_json
ROOT=Path(__file__).resolve().parent;base=ROOT/"artifacts/models/random/pipeline_smoke"
native=base/"search_pilot_native";fast=base/"search_pilot"
assert json.loads((ROOT/"audit/fast_proposal_equivalence.json").read_text())["passed"]
if not native.exists():
    assert (fast/"DONE.json").exists();fast.rename(native)
rc=subprocess.run([sys.executable,"run_search_comparison.py","--run","pipeline_smoke","--pilot"],cwd=ROOT).returncode
if rc:raise RuntimeError("Accelerated search replay failed")
records=[]
for p in sorted(native.glob("*/candidate_pool.csv")):
    q=fast/p.relative_to(native);a=pd.read_csv(p).sort_values("sequence").reset_index(drop=True)
    b=pd.read_csv(q).sort_values("sequence").reset_index(drop=True)
    assert a.sequence.equals(b.sequence),(p.parent.name,"different candidates")
    cols=[c for c in a.select_dtypes(include="number").columns if c in b]
    assert np.allclose(a[cols],b[cols],rtol=1e-6,atol=1e-6,equal_nan=True),(p.parent.name,"different scores")
    x=json.loads((p.parent/"DONE.json").read_text());y=json.loads((q.parent/"DONE.json").read_text())
    for key in ["forward_sequences","gradient_forward_sequences","valid_unique_candidates"]:assert x[key]==y[key]
    records.append({"run":p.parent.name,"candidates":len(a),"identical_sequences":True,"scores_agree_1e6":True,
        "native_seconds":x["seconds"],"fast_seconds":y["seconds"]})
write_json(ROOT/"audit/full_search_replay_equivalence.json",{"passed":True,"methods":len(records),"records":records,
    "fixture":"fixed intermediate model and training-only 256-row smoke memory; excluded from evidence results"})
print("FULL_SIX_METHOD_SEARCH_REPLAY_IDENTICAL",flush=True)
