"""Snapshot completed SaProt results, excluding unfinished checkpoint files."""
from pathlib import Path
import hashlib,json,zipfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
def main():
    files={}
    for split in ['random','depth']:
        for p in (ROOT/'artifacts/models'/split).glob('saprot650_*/DONE.json'):
            if '_smoke' in p.parent.name:continue
            for f in p.parent.iterdir():
                if f.is_file() and f.suffix in {'.json','.csv'}:
                    files[f.relative_to(ROOT).as_posix()]=f
    for name in ['PROTOCOL.json','GPU_PREFLIGHT.json','QUEUE_STATUS.json','INPUT_AUDIT.json','TRAINER_DERIVATION.json']:
        p=HERE/name
        if p.exists():files[p.relative_to(ROOT).as_posix()]=p
    for name in ['saprot650_pred3di_state.json','saprot650_pred3di_3di_state.json',
                 'saprot650_pred3di_native_parity.json']:
        p=ROOT/'artifacts/embeddings'/name
        if p.exists():files[p.relative_to(ROOT).as_posix()]=p
    # Read once, so both the manifest and archive describe the same snapshot.
    content={name:p.read_bytes() for name,p in files.items()}
    manifest={name:hashlib.sha256(b).hexdigest() for name,b in content.items()}
    target=HERE/'completed_results_snapshot.zip'
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,b in content.items():z.writestr(name,b)
        z.writestr('SAPROT_SNAPSHOT_SHA256.json',json.dumps(manifest,indent=2))
    print(json.dumps({'archive':str(target),'files':len(files),'bytes':target.stat().st_size,
                      'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}))

if __name__=='__main__':main()
