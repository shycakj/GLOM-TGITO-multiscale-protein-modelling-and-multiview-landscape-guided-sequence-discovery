"""Freeze Top30 and their actual search parents without altering original results."""
import csv, hashlib, json, shutil
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT.parent
SOURCE = EXP / 'artifacts/discovery_v3/formal/glom_tgito/prefix_900'
BIO = EXP / 'reports/teacher_revision_20261024/supplementary_data/Supplementary_Data_3_Biological_Context'
OUT = ROOT / 'inputs'
OUT.mkdir(parents=True, exist_ok=True)
wt = (BIO / 'inputs/reference_sequence.txt').read_text().strip()
assert len(wt) == 221 and wt[55:58] == 'GYG'
def read(path):
    with open(path, encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))
def changes(a,b):
    assert len(a)==len(b)==221
    return ';'.join(f'{x}{i}{y}' for i,(x,y) in enumerate(zip(a,b),1) if x!=y)
starts = {int(r['start_id']): r for r in read(EXP/'artifacts/discovery_v3/starting_sequences.csv')}
top = read(SOURCE / 'top30.csv')
assert len(top)==30 and len({r['sequence'] for r in top})==30
systems = {'WT': {'system_id':'WT','sequence':wt,'role':'wild_type','mutations_vs_wt':''}}
byseq = {wt:'WT'}
pairs=[]
def register(seq, suggested, role):
    assert len(seq)==221 and set(seq)<=set('ACDEFGHIKLMNPQRSTVWY')
    assert seq[55:58]=='GYG', 'Mature chromophore differs: needs a different parameterization'
    if seq not in byseq:
        byseq[seq]=suggested
        systems[suggested]={'system_id':suggested,'sequence':seq,'role':role,'mutations_vs_wt':changes(wt,seq)}
    return byseq[seq]
for rank,r in enumerate(top,1):
    s=starts[int(r['start_id'])]
    assert int(r['seed_row'])==int(s['row_id'])
    m=changes(s['sequence'],r['sequence'])
    n=len(m.split(';')) if m else 0
    assert n==int(r['hamming_from_seed']) and n<=2
    start=register(s['sequence'],f'S{int(r["start_id"]):04d}','search_start')
    cand=register(r['sequence'],f'C{rank:02d}','top30_candidate')
    pairs.append({'rank':rank,'candidate_id':cand,'start_system_id':start,'start_id':r['start_id'],
                  'seed_row':r['seed_row'],'mountain_id':r['mountain_id'],'start_role':r['role'],
                  'prediction':r['pred'],'confidence_score':r['confidence_score'],
                  'mutations_vs_wt':changes(wt,r['sequence']),'mutations_vs_start':m,
                  'start_measured_brightness':s['brightness']})
def write(name,rows):
    with open(OUT/name,'w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
write('systems.csv',list(systems.values()));write('candidate_start_pairs.csv',pairs)
(OUT/'systems.fasta').write_text(''.join(f'>{r["system_id"]}\n{r["sequence"]}\n' for r in systems.values()))
for src,dest in [(SOURCE/'top30.csv','frozen_top30.csv'),(SOURCE/'FROZEN.json','original_FROZEN.json'),
                 (BIO/'inputs/reference_sequence.txt','reference_sequence.txt'),
                 (EXP.parent/'manuscript_bioinformatics_20260930/analysis/2G3O_reference.pdb','2G3O_reference.pdb'),
                 (BIO/'structure/2G3O_reference.cif','2G3O_reference.cif')]:
    shutil.copy2(src,OUT/dest)
hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file()}
summary={'candidate_count':30,'unique_systems':len(systems),'unique_starts':len({r['start_system_id'] for r in pairs}),
         'mutation_events_vs_start':sum(len(r['mutations_vs_start'].split(';')) for r in pairs),
         'candidate_substitutions_vs_wt':dict(Counter(m for r in pairs for m in r['mutations_vs_wt'].split(';') if m)),
         'sha256':hashes,'source':str(SOURCE),'all_chromophores_preserved':True,
         'comparison':'each frozen candidate versus its own actual starting sequence and the 221-aa WT',
         'assay_policy':'No new wet-assay results are used.'}
(ROOT/'INPUT_AUDIT.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))

if __name__=='__main__':
    try:
        from Bio.PDB import PDBParser
        model=PDBParser(QUIET=True).get_structure('2G3O',OUT/'2G3O_reference.pdb')[0]
        for chain in model:
            aa=[r for r in chain if r.id[0]==' ']
            cr=[r for r in chain if r.resname=='CR2']
            print('CHAIN',chain.id,len(aa),min(r.id[1] for r in aa),max(r.id[1] for r in aa),
                  'CR2',[(r.id,[a.name for a in r]) for r in cr])
    except ImportError:
        pass
