"""Published SWI weights extracted unchanged from authors' open source code."""
import ast, csv, hashlib, json, urllib.request
from pathlib import Path
import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis
ROOT=Path(__file__).resolve().parents[1]
url='https://raw.githubusercontent.com/Gardner-BinfLab/SoDoPE_paper_2020/master/functions.py'
source=ROOT/'tools/SoDoPE_functions.py'
if not source.exists():
    source.write_bytes(urllib.request.urlopen(url,timeout=60).read())
tree=ast.parse(source.read_text());weights=None
for node in tree.body:
    if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='weights' for t in node.targets):
        weights=ast.literal_eval(node.value);break
assert weights is not None and len(weights)==20
rows=[]
for r in csv.DictReader(open(ROOT/'inputs/systems.csv')):
    seq=r['sequence'];pa=ProteinAnalysis(seq)
    rows.append({'system_id':r['system_id'],'SWI_mean':float(np.mean([weights[a] for a in seq])),
                 'SWI_tail_199_221':float(np.mean([weights[a] for a in seq[198:]])),
                 'sequence_GRAVY':pa.gravy(),'sequence_pI':pa.isoelectric_point(),'sequence_charge_pH7':pa.charge_at_pH(7.),
                 'sequence_aromaticity':pa.aromaticity()})
with open(ROOT/'results/sequence_solubility.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(ROOT/'results/SWI_AUDIT.json').write_text(json.dumps({'method':'Solubility-Weighted Index (SWI)',
    'doi':'10.1093/bioinformatics/btaa578','source_url':url,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
    'formula':'arithmetic mean of published residue weights across the sequence',
    'direction':'larger SWI predicts greater soluble-expression tendency',
    'scope':'sequence-based supporting score developed for recombinant expression in E. coli; not a fluorescence assay',
    'weights':weights,'systems_scored':len(rows)},indent=2))
print(json.dumps({'scored':len(rows),'WT':rows[0]}))
