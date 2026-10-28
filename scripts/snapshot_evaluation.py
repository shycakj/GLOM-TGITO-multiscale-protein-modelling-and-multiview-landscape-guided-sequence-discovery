"""Record fresh evaluation file identities before reliability analysis."""
from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parent
target=R/'reports/discovery_v3_snapshot_manifest.json';target.parent.mkdir(exist_ok=True)
rows=[]
for p in sorted((R/'artifacts/discovery_v3/evaluation').rglob('*')):
    if p.is_file():
        with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
        rows.append(dict(path=p.relative_to(R).as_posix(),sha256=h,bytes=p.stat().st_size))
assert rows
if target.exists():assert json.loads(target.read_text())['files']==rows
else:target.write_text(json.dumps(dict(files=rows),indent=2))
print('Evaluation snapshot verified:',len(rows),'files')
