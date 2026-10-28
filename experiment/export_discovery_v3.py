"""Compact backup of completed records only; raw query logs remain persistent."""
from pathlib import Path
import hashlib,json,time,zipfile
ROOT=Path(__file__).resolve().parent;out=ROOT/'artifacts/discovery_v3';files=set()
for p in out.iterdir():
    if p.is_file() and p.suffix in ['.json','.csv']:files.add(p)
for p in out.glob('*/**/DONE.json'):
    files.add(p)
for folder in out.glob('formal/*/prefix_*'):
    if (folder/'FROZEN.json').exists():files.update(p for p in folder.iterdir() if p.is_file() and p.suffix in ['.json','.csv','.fasta'])
if (out/'evaluation/DONE.json').exists():files.update(p for p in (out/'evaluation').iterdir() if p.is_file())
for p in [ROOT/'RESTORE_READY.json',ROOT/'audit/persistent_runtime_smoke.json',ROOT/'artifacts/models/random/glom_seed20261013/features/RECOVERY_DONE.json']:
    if p.exists():files.add(p)
target=ROOT/'incoming/discovery_v3_snapshot.zip';temp=target.with_suffix('.building')
manifest=[]
with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(files):
        data=p.read_bytes();name=p.relative_to(ROOT).as_posix();z.writestr(name,data)
        manifest.append(dict(path=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
    z.writestr('reports/discovery_v3_snapshot_manifest.json',json.dumps(dict(unix_time=time.time(),files=manifest)))
temp.replace(target)
print(json.dumps(dict(path=str(target),bytes=target.stat().st_size,sha256=hashlib.file_digest(target.open('rb'),'sha256').hexdigest(),
                      files=len(files),campaign_complete=(out/'CAMPAIGN_DONE.json').exists())))
