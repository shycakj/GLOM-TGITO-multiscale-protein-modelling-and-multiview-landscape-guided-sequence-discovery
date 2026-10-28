"""Install checksummed reproducibility assets without replacing changed files."""
import argparse, hashlib, json, os, shutil, time, urllib.request, zipfile
from pathlib import Path, PurePosixPath
ROOT=Path(__file__).resolve().parents[1]

def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def safe_path(name):
    rel=PurePosixPath(name)
    if rel.is_absolute() or '..' in rel.parts or '\\' in name or ':' in name:
        raise ValueError('Unsafe archive path: '+name)
    path=(ROOT/Path(*rel.parts)).resolve()
    if not path.is_relative_to(ROOT):raise ValueError('Path escapes repository: '+name)
    return path

def fetch(url,dest,expected,size):
    if dest.exists() and dest.stat().st_size==size and digest(dest)==expected:return
    dest.parent.mkdir(parents=True,exist_ok=True)
    tmp=dest.with_suffix(dest.suffix+'.part')
    last=time.monotonic();count=0
    request=urllib.request.Request(url,headers={'User-Agent':'GLOM-TGITO-reproduction'})
    with urllib.request.urlopen(request,timeout=120) as response,tmp.open('wb') as out:
        while block:=response.read(4<<20):
            out.write(block);count+=len(block)
            if time.monotonic()-last>15:
                print(f'{dest.name}: {count/1e6:.1f}/{size/1e6:.1f} MB',flush=True);last=time.monotonic()
    if tmp.stat().st_size!=size or digest(tmp)!=expected:raise ValueError('Downloaded SHA256 or size differs: '+dest.name)
    tmp.replace(dest)

def main():
    spec=json.loads((ROOT/'manifests/ASSETS.json').read_text(encoding='utf8'))
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('groups',nargs='*',default=['data'])
    p.add_argument('--list',action='store_true')
    p.add_argument('--from-dir',type=Path,help='Use matching release ZIPs already downloaded to this directory.')
    a=p.parse_args()
    if a.list:
        for name,g in spec['groups'].items():print(f"{name:18} {g['bytes']/1e6:9.2f} MB  {len(g['files'])} files")
        return
    for name in a.groups:
        if name not in spec['groups']:raise ValueError('Unknown group: '+name)
        g=spec['groups'][name]
        missing=[]
        for rel,v in g['files'].items():
            dest=safe_path(rel)
            if dest.exists():
                if not dest.is_file() or dest.stat().st_size!=v['bytes'] or digest(dest)!=v['sha256']:
                    raise ValueError('Existing file differs; use a clean checkout: '+rel)
            else:missing.append(rel)
        if not missing:print('VERIFIED',name,flush=True);continue
        source=(ROOT/g['local_path']) if 'local_path' in g else ((a.from_dir/g['filename']) if a.from_dir else ROOT/'.downloads'/g['filename'])
        if not source.exists():
            if a.from_dir or 'local_path' in g:raise FileNotFoundError(source)
            fetch(g['url'],source,g['sha256'],g['bytes'])
        if source.stat().st_size!=g['bytes'] or digest(source)!=g['sha256']:raise ValueError('Archive identity differs: '+source.name)
        with zipfile.ZipFile(source) as z:
            if set(z.namelist())!=set(g['files']):raise ValueError('Archive membership differs: '+name)
            for rel in missing:
                dest=safe_path(rel);dest.parent.mkdir(parents=True,exist_ok=True)
                tmp=dest.with_name(dest.name+'.extracting')
                with z.open(rel) as src,tmp.open('wb') as out:shutil.copyfileobj(src,out,4<<20)
                v=g['files'][rel]
                if tmp.stat().st_size!=v['bytes'] or digest(tmp)!=v['sha256']:raise ValueError('Extracted identity differs: '+rel)
                tmp.replace(dest)
        print('INSTALLED',name,len(missing),'files',flush=True)

if __name__=='__main__':main()
