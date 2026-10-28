"""Download content-verified public backbones. Never downloads task-finetuned SaProt."""
from pathlib import Path
import argparse,json,hashlib,urllib.request,time,os
ROOT=Path(__file__).resolve().parents[1]
def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def fetch(url,path,expected,size=None):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and (size is None or path.stat().st_size==size) and digest(path)==expected:
        print('VERIFIED',path.relative_to(ROOT));return
    tmp=path.with_name(path.name+'.download')
    with urllib.request.urlopen(url,timeout=120) as response,tmp.open('wb') as f:
        last=time.monotonic();count=0
        while block:=response.read(4*1024*1024):
            f.write(block);count+=len(block)
            if time.monotonic()-last>15:print(path.name,count,flush=True);last=time.monotonic()
    assert (size is None or tmp.stat().st_size==size) and digest(tmp)==expected, 'Downloaded identity differs: '+str(path)
    tmp.replace(path)

def main():
    p=argparse.ArgumentParser();p.add_argument('assets',nargs='*',default=['esm2_150m'])
    a=p.parse_args();base=ROOT/'experiment/repro_project/assets'
    for name in a.assets:
        if name in ['esm2_150m','esm2_650m','esmc_600m','esmc_6b']:
            folder=base/('esm2_t30_150M_UR50D' if name=='esm2_150m' else name);manifest=json.loads((folder/'DOWNLOAD_VERIFIED.json').read_text())
            for item in manifest['files']:
                url=f"https://huggingface.co/{manifest['repo']}/resolve/{manifest['revision']}/{item['file']}"
                fetch(url,folder/item['file'],item['sha256'],item['bytes'])
        elif name=='saprot_650m_af2':
            spec=json.loads((ROOT/'experiment/saprot_baseline_20261019/PROTOCOL.json').read_text())
            folder=base/name;metadata=ROOT/'experiment/saprot_baseline_20261019/model_metadata/METADATA_VERIFIED.json'
            info=json.loads(metadata.read_text());rows=list(info['files'])
            for item in rows:
                fetch(f"https://huggingface.co/{spec['model_repo']}/resolve/{spec['model_revision']}/{item['file']}",folder/item['file'],item['sha256'],item['bytes'])
            namefile='pytorch_model.bin';expected=info['expected_weight_sha256']
            fetch(f"https://huggingface.co/{spec['model_repo']}/resolve/{spec['model_revision']}/{namefile}",folder/namefile,expected)
            rows.append({'file':namefile,'bytes':(folder/namefile).stat().st_size,'sha256':expected})
            (folder/'DOWNLOAD_VERIFIED.json').write_text(json.dumps({'repo':spec['model_repo'],'revision':spec['model_revision'],'files':rows,
                'expected_weight_sha256':expected,'weights_downloaded':True},indent=2)+'\n')
        elif name=='esm2_native650':
            meta=json.loads((ROOT/'experiment/artifacts/proteinttt_650m/DONE.json').read_text())
            fetch('https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt',ROOT/'experiment/public_checkpoints/esm2_t33_650M_UR50D.pt',meta['checkpoint_sha256'])
        elif name=='spurs':
            rows=json.loads((ROOT/'biophysics/tools/SPURS_hub_file_manifest.json').read_text())
            item=next(r for r in rows if r['path'].endswith('/best.ckpt'))
            fetch('https://huggingface.co/cyclization9/SPURS/resolve/main/'+item['path'],ROOT/'biophysics/tools/SPURS_multi.ckpt',item['lfs']['oid'],item['size'])
        else:raise ValueError('Unknown public asset: '+name)
    print('Requested public assets are verified.')

if __name__=='__main__':main()
