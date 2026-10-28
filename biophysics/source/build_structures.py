"""Matched experimental-template structures; restore mature CR2 before parameterization.

The transient standard-residue scaffold is used only by PDBFixer to construct
mutated side chains and the three unresolved C-terminal residues. It is never
used for molecular dynamics. The exact crystal CR2 heavy atoms are restored.
"""
import argparse, csv, json, os, io, hashlib
from pathlib import Path
import numpy as np
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1, seq3
from pdbfixer import PDBFixer
from openmm.app import PDBFile
from openmm import Platform

ROOT=Path(__file__).resolve().parents[1]
def atom_line(serial,name,res,pos,xyz,element,het=False,chain='C'):
    record='HETATM' if het else 'ATOM  '
    an=f' {name:<3}' if len(name)<4 else name
    return f'{record}{serial:5d} {an:4s} {res:3s} {chain}{pos:4d}    {xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}{1.:6.2f}{0.:6.2f}          {element:>2s}  \n'

def prepare(system_id):
    records={r['system_id']:r for r in csv.DictReader(open(ROOT/'inputs/systems.csv'))}
    row=records[system_id];seq=row['sequence']
    out=ROOT/'structures'/system_id;out.mkdir(parents=True,exist_ok=True)
    model=PDBParser(QUIET=True).get_structure('ref',ROOT/'inputs/2G3O_reference.pdb')[0]
    chain=model['C'];cr=next(r for r in chain if r.resname=='CR2')
    cryst={a.name:np.array(a.coord,dtype=float) for a in cr}
    original=[]
    for r in chain:
        if r.id[0]==' ' and 2<=r.id[1]<=219:
            original.append((r.id[1]-1,r))
    assert {i for i,r in original}==set(range(1,219))-{56,57,58}
    # CR2 atom names map to the original three residue identities.
    cr_maps={56:('GLY',{'N':'N1','CA':'CA1','C':'C1'}),
             57:('TYR',{'N':'N2','CA':'CA2','C':'C2','O':'O2','CB':'CB2','CG':'CG2','CD1':'CD1','CD2':'CD2','CE1':'CE1','CE2':'CE2','CZ':'CZ','OH':'OH'}),
             58:('GLY',{'N':'N3','CA':'CA3','C':'C3','O':'O3'})}
    data={i:(r.resname,[(a.name,a.coord,a.element) for a in r if a.element!='H' and a.name!='OXT']) for i,r in original}
    for i,(res,names) in cr_maps.items():data[i]=(res,[(name,cryst[source],name[0]) for name,source in names.items()])
    text=[];serial=1
    for i in sorted(data):
        res,atoms=data[i]
        for name,xyz,el in atoms:
            text.append(atom_line(serial,name,res,i,xyz,el));serial+=1
    text.append('TER\nEND\n')
    fixer=PDBFixer(pdbfile=io.StringIO(''.join(text)),platform=Platform.getPlatformByName('CPU'))
    mutations=[f'{data[i][0]}-{i}-{seq3(seq[i-1]).upper()}' for i in range(1,219) if seq1(data[i][0])!=seq[i-1]]
    if mutations:fixer.applyMutations(mutations,'C')
    fixer.missingResidues={(0,218):[seq3(a).upper() for a in seq[218:]]}
    fixer.findMissingAtoms()
    fixer.addMissingAtoms(seed=20261027)
    with open(out/'precursor_for_structure_predictors.pdb','w') as f:PDBFile.writeFile(fixer.topology,fixer.positions,f,keepIds=True)
    built=PDBParser(QUIET=True).get_structure('built',out/'precursor_for_structure_predictors.pdb')[0]
    residues=list(built.get_residues())
    assert len(residues)==221
    assert ''.join(seq1(r.resname) for r in residues)==seq
    # Number mature residues sequentially for Amber; retain explicit project mapping.
    text=[];serial=1;amber_index=0;mapping=[]
    for i,r in enumerate(residues,1):
        if i in (57,58):continue
        amber_index+=1
        if i==56:
            for a in cr:
                if a.element=='H':continue
                text.append(atom_line(serial,a.name,'CR2',amber_index,a.coord,a.element,True));serial+=1
            mapping.append({'amber_residue':amber_index,'project_positions':[56,57,58],'name':'CR2','coordinate_source':'experimental'})
        else:
            # ff14SB default neutral histidine tautomer, identical preparation rule.
            resname='HIE' if r.resname=='HIS' else r.resname
            for a in r:
                if a.element=='H':continue
                text.append(atom_line(serial,a.name,resname,amber_index,a.coord,a.element));serial+=1
            mapping.append({'amber_residue':amber_index,'project_positions':[i],'name':resname,
                            'coordinate_source':'modeled_tail' if i>=219 else 'template_and_sidechain_repair'})
    text.append('TER\nEND\n');(out/'mature_input.pdb').write_text(''.join(text))
    (out/'residue_mapping.json').write_text(json.dumps(mapping,indent=2))
    audit={'system_id':system_id,'sequence':seq,'mutations_vs_wt':row['mutations_vs_wt'],
           'modeled_project_positions':[219,220,221],'mature_residues':amber_index,
           'chromophore':'CR2: Gly56-Tyr57-Gly58; exact experimental heavy-atom coordinates',
           'histidine_rule':'neutral epsilon-protonated HIE; common fixed-protonation approximation',
           'template_chain':'2G3O C','seed':20261027}
    (out/'BUILD_AUDIT.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('system_id');a=p.parse_args();prepare(a.system_id)
