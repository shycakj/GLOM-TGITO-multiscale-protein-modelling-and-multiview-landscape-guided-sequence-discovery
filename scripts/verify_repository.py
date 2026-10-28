"""Verify fixed splits, published parameter routing and optional downloaded assets."""
import argparse, csv, hashlib, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];E=ROOT/'experiment'

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',action='store_true')
    p.add_argument('--all-installed',action='store_true');a=p.parse_args()
    tracked=json.loads((ROOT/'manifests/REPOSITORY_FILES.json').read_text(encoding='utf8'))
    for rel,record in tracked.items():
        f=ROOT/rel
        if not f.is_file() or sha(f)!=record['sha256']:raise ValueError('Repository file differs: '+rel)
    manifest=json.loads((E/'data/split_manifest.json').read_text(encoding='utf8'))
    for rel,expected in manifest['files'].items():
        if sha(E/'data'/rel)!=expected:raise ValueError('Fixed partition differs: '+rel)
    raw=E/'repro_project/data/raw/pplu_31480_sequences.csv'
    if sha(raw)!=manifest['raw_sha256']:raise ValueError('Raw data differs')
    with (E/'data/unique_sequences.csv').open(encoding='utf8',newline='') as f:rows=list(csv.DictReader(f))
    sequences=[r['sequence'] for r in rows]
    assert len(rows)==31434 and len(set(sequences))==31434 and all(len(s)==221 for s in sequences)
    roles={role:{r['sequence'] for r in rows if r['partition']==role} for role in ['train','validation','calibration','test']}
    assert {k:len(v) for k,v in roles.items()}==manifest['partition_counts']
    assert sum(map(len,roles.values()))==len(set.union(*roles.values()))
    with (E/'data/depth_extrapolation.csv').open(encoding='utf8',newline='') as f:depth=list(csv.DictReader(f))
    assert all((int(r['mutation_depth'])>=3)==(r['partition']=='test') for r in depth)
    assert {role:sum(r['partition']==role for r in depth) for role in roles}==manifest['extrapolation_counts']
    sys.path.insert(0,str(E))
    from hpo_config_contract import validate_effective
    protocol=json.loads((E/'artifacts/discovery_v3/PROTOCOL.json').read_text(encoding='utf8'))
    profile=protocol['profiles']['trial12'];validate_effective(profile['parameters'],profile['resolved'])
    with (E/'artifacts/discovery_v3/starting_sequences.csv').open(encoding='utf8',newline='') as f:starts=list(csv.DictReader(f))
    assert len(starts)==2700 and len({r['sequence'] for r in starts})==1308
    available=0;absent=[]
    for rel,expected in protocol['input_hashes'].items():
        f=E/rel
        if not f.exists():absent.append(rel);continue
        if sha(f)!=expected:raise ValueError('Frozen search input differs: '+rel)
        available+=1
    if a.experiment and absent:raise FileNotFoundError('Fetch weights, terrain and public ESM2 backbones before discovery: '+', '.join(absent))
    groups={}
    for name,g in json.loads((ROOT/'manifests/ASSETS.json').read_text(encoding='utf8'))['groups'].items():
        existing=[rel for rel in g['files'] if (ROOT/rel).is_file()]
        if a.all_installed:
            for rel in existing:
                if sha(ROOT/rel)!=g['files'][rel]['sha256']:raise ValueError('Installed asset differs: '+rel)
        groups[name]={'installed':len(existing),'total':len(g['files'])}
    print(json.dumps({'status':'PASS','repository_files':len(tracked),'raw_records':31480,'unique_sequences':len(rows),
        'random_split':manifest['partition_counts'],'depth_split':manifest['extrapolation_counts'],
        'training_seeds':manifest['train_seed_list'],'HPO_trial':12,'search_tasks':len(starts),'distinct_starts':1308,
        'search_inputs_verified':available,'search_inputs_not_downloaded':absent,'asset_groups':groups},indent=2))

if __name__=='__main__':main()
