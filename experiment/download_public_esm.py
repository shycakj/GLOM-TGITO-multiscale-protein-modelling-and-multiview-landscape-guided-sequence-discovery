"""Fetch the public ESM snapshot directly on the GPU host and verify the original hash."""
from pathlib import Path
import hashlib,json,time,urllib.request
ROOT=Path(__file__).resolve().parent
dest=ROOT/'repro_project/assets/esm2_t30_150M_UR50D';dest.mkdir(parents=True,exist_ok=True)
expected=json.loads((ROOT/'public_esm_manifest.json').read_text())
for item in expected:
    p=dest/item['name']
    if p.exists() and p.stat().st_size==item['bytes']:
        digest=hashlib.sha256(p.read_bytes()).hexdigest()
        if digest==item['sha256']:continue
    last=None
    for base in ['https://huggingface.co/facebook/esm2_t30_150M_UR50D/resolve/main/',
                 'https://hf-mirror.com/facebook/esm2_t30_150M_UR50D/resolve/main/']:
        try:
            tmp=p.with_suffix(p.suffix+'.part')
            with urllib.request.urlopen(base+item['name'],timeout=45) as response,tmp.open('wb') as out:
                h=hashlib.sha256();count=0;notice=time.monotonic()
                while chunk:=response.read(4<<20):
                    out.write(chunk);h.update(chunk);count+=len(chunk)
                    if time.monotonic()-notice>15:
                        print(json.dumps({'file':p.name,'download_bytes':count}),flush=True);notice=time.monotonic()
            if h.hexdigest()!=item['sha256']:raise ValueError('Hash differs from the packaged original: '+p.name)
            tmp.replace(p);print('VERIFIED '+p.name,flush=True);last=None;break
        except Exception as e:
            last=e;print('Download attempt: '+type(e).__name__+': '+str(e),flush=True)
    if last:raise last
(dest/'DOWNLOAD_VERIFIED.json').write_text(json.dumps(expected,indent=2))
print('PUBLIC_ESM_DOWNLOAD_COMPLETE',flush=True)
