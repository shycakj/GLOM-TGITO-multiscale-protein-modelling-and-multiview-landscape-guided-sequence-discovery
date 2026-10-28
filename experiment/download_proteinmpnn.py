"""Fetch public official ProteinMPNN code/weights at one recorded revision."""
from pathlib import Path
import urllib.request,json,hashlib,argparse
root=Path(__file__).resolve().parent;dest=root/'third_party/ProteinMPNN';dest.mkdir(parents=True,exist_ok=True)
p=argparse.ArgumentParser();p.add_argument('--revision');a=p.parse_args()
if a.revision:revision=a.revision
else:
    request=urllib.request.Request('https://api.github.com/repos/dauparas/ProteinMPNN/commits/main',headers={'User-Agent':'GFP-validation-research'})
    revision=json.load(urllib.request.urlopen(request,timeout=45))['sha']
records=[]
for name in ['protein_mpnn_utils.py','protein_mpnn_run.py','LICENSE','vanilla_model_weights/v_48_020.pt']:
    path=dest/name;path.parent.mkdir(exist_ok=True)
    url=f'https://raw.githubusercontent.com/dauparas/ProteinMPNN/{revision}/{name}'
    data=urllib.request.urlopen(url,timeout=90).read();path.write_bytes(data)
    records.append(dict(file=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),url=url))
    print(name,len(data),flush=True)
(dest/'DOWNLOAD_VERIFIED.json').write_text(json.dumps(dict(repository='https://github.com/dauparas/ProteinMPNN',revision=revision,files=records),indent=2))
