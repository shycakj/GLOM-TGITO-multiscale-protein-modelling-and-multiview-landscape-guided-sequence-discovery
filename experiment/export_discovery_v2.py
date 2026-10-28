"""Small complete-prefix download with a hash manifest, including FASTA files."""
from pathlib import Path
import hashlib,json,zipfile,time
root=Path(__file__).resolve().parent;base=root/'artifacts/models/random/glom_seed20261013/full_discovery_v2'
files=[]
for name in ['PROTOCOL.json','PROGRESS.json','PLAN_FROZEN.json','starting_sequences.csv','DONE.json']:
    if (base/name).exists():files.append(base/name)
for folder in base.glob('prefix_*'):
    if (folder/'FROZEN.json').exists():files.extend(f for f in folder.iterdir() if f.suffix in ['.csv','.json','.fasta'])
for name in ['NUMERICAL_CONTRACT.json','replay_metrics.csv','DONE.json','PROTOCOL.json']:
    f=root/'audit/optimizer_consistency_replay'/name
    if f.exists():files.append(f)
for name in ['ASSAY_VALIDATION_DONE.json','assay_validation_metrics.csv','VALIDATION_PROTOCOL.json','REFERENCE_MAPPING.json']:
    f=root/'audit/structure_validation'/name
    if f.exists():files.append(f)
for name in ['DISCOVERY_REVIEW_HOLD.json','FULL_DISCOVERY_V2_LAUNCH.json','STRUCTURE_VALIDATION_LAUNCH.json']:
    if (root/'audit'/name).exists():files.append(root/'audit'/name)
records=[];target=root/'discovery_v2_snapshot.zip'
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(set(files)):
        rel=p.relative_to(root).as_posix();b=p.read_bytes();z.writestr(rel,b)
        records.append(dict(path=rel,bytes=len(b),sha256=hashlib.sha256(b).hexdigest()))
    z.writestr('DISCOVERY_SNAPSHOT_MANIFEST.json',json.dumps(dict(unix_time=time.time(),files=records),indent=2))
print(json.dumps(dict(bytes=target.stat().st_size,files=len(records),sha256=hashlib.sha256(target.read_bytes()).hexdigest())))
