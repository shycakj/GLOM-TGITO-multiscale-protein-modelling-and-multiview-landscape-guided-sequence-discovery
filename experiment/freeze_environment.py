"""Record the actual runtime and complete code provenance without credentials."""
from pathlib import Path
import subprocess,json,hashlib,sys,platform,time
import torch
ROOT=Path(__file__).resolve().parent;out=ROOT/"reports";out.mkdir(exist_ok=True)
def command(args):
    r=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    return {"exit_code":r.returncode,"output":r.stdout.strip()}
source={}
for directory,pattern in [(ROOT,"*.py"),(ROOT/"repro_project/pipeline","**/*.py")]:
    for p in directory.glob(pattern):source[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
third={}
for name in ["ProteinTTT","ProteinGym_ProteinTTT","EvolvePro","FLEXS","ProtHMSO"]:
    p=ROOT/"third_party"/name
    if p.exists():third[name]=command(["git","-C",str(p),"rev-parse","HEAD"])
packages={}
for name in ["venv","venv_recent"]:
    p=ROOT/name/"bin/python"
    if p.exists():
        result=command([str(p),"-m","pip","freeze"])
        (out/(name+"_pip_freeze.txt")).write_text(result["output"]+"\n")
        packages[name]={"file":name+"_pip_freeze.txt","exit_code":result["exit_code"]}
record={"unix_time":time.time(),"python":sys.version,"platform":platform.platform(),
    "torch":torch.__version__,"cuda":torch.version.cuda,"cudnn":torch.backends.cudnn.version(),
    "gpu":command(["nvidia-smi","--query-gpu=name,memory.total,driver_version","--format=csv,noheader"]),
    "source_sha256":source,"third_party_commits":third,"environments":packages,
    "split_manifest":json.loads((ROOT/"data/split_manifest.json").read_text()),
    "input_contract":"original GLOM AAP mutant-only P4; ESM native unmasked scaling 0.88; TF32 disabled in differentiable encoder",
    "training_seeds":[20261013,20261014,20261015],"gpu_admission":"memory free checked by queue; bounded concurrent processes",
    "public_weights":"pinned Hugging Face revisions and file hashes in model DOWNLOAD_VERIFIED.json",
    "private_weights":"user-provided Bridge only; fluorescence prediction heads trained anew"}
(out/"reproduction_manifest.json").write_text(json.dumps(record,indent=2))
print(json.dumps({"source_files":len(source),"runtime":record["torch"],"cuda":record["cuda"]}))
