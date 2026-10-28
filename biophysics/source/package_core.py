"""Local-downloadable prepared systems, source, and initial complete scores."""
import hashlib,json,tarfile,io
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
keep=['source','inputs','INPUT_AUDIT.json','PREFLIGHT_PASSED.json','EXPERIMENT_README_zh.txt',
      'results','PREPARATION_STATUS.json','CAMPAIGN_STATUS.json']
tools=['SPURS','SPURS_multi_config.yaml','SPURS_hub_file_manifest.json','SPURS_commit.txt',
       'SPURS_inference_compatibility.patch','SoDoPE_functions.py','conda_explicit.txt','mlenv_pip_freeze.txt']
files=[]
for n in keep:
    p=ROOT/n
    if p.is_dir():files.extend(q for q in p.rglob('*') if q.is_file() and '__pycache__' not in q.parts)
    elif p.exists():files.append(p)
for n in tools:
    p=ROOT/'tools'/n
    if p.is_dir():files.extend(q for q in p.rglob('*') if q.is_file() and '.git' not in q.parts and '__pycache__' not in q.parts)
    elif p.exists():files.append(p)
for folder in (ROOT/'structures').iterdir():
    if not folder.is_dir():continue
    for n in ['mature_input.pdb','precursor_for_structure_predictors.pdb','dry.prmtop','dry.inpcrd',
              'dry.pdb','solvated.prmtop','solvated.inpcrd','residue_mapping.json',
              'BUILD_AUDIT.json','PARAMETER_AUDIT.json','leap_solvate.in','leap_salt.in']:
        p=folder/n
        if p.exists():files.append(p)
manifest={}
with tarfile.open(ROOT/'REPRODUCTION_CORE.tar.gz','w:gz',compresslevel=3) as archive:
    for p in sorted(set(files)):
        data=p.read_bytes();name=str(p.relative_to(ROOT)).replace('\\','/')
        manifest[name]=hashlib.sha256(data).hexdigest()
        info=archive.gettarinfo(str(p),arcname=name);info.size=len(data)
        archive.addfile(info,io.BytesIO(data))
    path=ROOT/'LOCAL_BACKUP_MANIFEST.json';path.write_text(json.dumps(manifest,indent=2))
    archive.add(path,arcname=path.name)
print(json.dumps({'files':len(files),'archive_bytes':(ROOT/'REPRODUCTION_CORE.tar.gz').stat().st_size,
                  'archive_sha256':hashlib.sha256((ROOT/'REPRODUCTION_CORE.tar.gz').read_bytes()).hexdigest()}),flush=True)
