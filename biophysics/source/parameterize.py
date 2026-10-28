import argparse, json, os, subprocess, hashlib
from pathlib import Path
import numpy as np
import parmed as pmd

ROOT=Path(__file__).resolve().parents[1]
ENV=ROOT/'env'
def leap(folder,name,commands):
    path=folder/f'{name}.in';path.write_text(commands+'\nquit\n')
    env=os.environ.copy();env['AMBERHOME']=str(ENV);env['OMP_NUM_THREADS']='1'
    run=subprocess.run([str(ENV/'bin/tleap'),'-f',str(path)],cwd=folder,env=env,capture_output=True,text=True)
    (folder/f'{name}.log').write_text(run.stdout+'\n'+run.stderr)
    if run.returncode or 'FATAL' in run.stdout or 'does not have a type' in run.stdout:
        raise RuntimeError(f'LEaP failed: {folder/name}')
    return run.stdout

def prepare(system_id):
    out=ROOT/'structures'/system_id
    base='source leaprc.protein.ff14SB\nsource leaprc.water.opc\nsource leaprc.xFPchromophores\n'
    leap(out,'leap_solvate',base+'''u = loadpdb mature_input.pdb
check u
saveamberparm u dry.prmtop dry.inpcrd
savepdb u dry.pdb
solvateOct u OPCBOX 10.0
addIonsRand u Na+ 0
addIonsRand u Cl- 0
saveamberparm u neutral.prmtop neutral.inpcrd
saveOff u solvated.lib
''')
    neutral=pmd.load_file(str(out/'neutral.prmtop'),xyz=str(out/'neutral.inpcrd'))
    a,b,c,al,be,ga=neutral.box
    ca,cb,cg=np.cos(np.radians([al,be,ga]))
    volume=float(a*b*c*np.sqrt(1+2*ca*cb*cg-ca*ca-cb*cb-cg*cg))
    n_salt=int(round(volume*0.000602214076*0.15))
    leap(out,'leap_salt',base+f'''loadOff solvated.lib
addIonsRand u Na+ {n_salt} Cl- {n_salt}
check u
saveamberparm u solvated.prmtop solvated.inpcrd
savepdb u solvated.pdb
''')
    dry=pmd.load_file(str(out/'dry.prmtop'),xyz=str(out/'dry.inpcrd'))
    solv=pmd.load_file(str(out/'solvated.prmtop'),xyz=str(out/'solvated.inpcrd'))
    cr=next(r for r in dry.residues if r.name=='CR2')
    names={a.name:a for a in cr.atoms}
    charge=sum(a.charge for a in cr.atoms)
    assert abs(charge+1)<1e-4,charge
    bonds={frozenset((b.atom1.idx,b.atom2.idx)) for b in dry.bonds}
    before=dry.residues[cr.idx-1];after=dry.residues[cr.idx+1]
    linked=[]
    for a,b in [(next(a for a in before.atoms if a.name=='C'),names['N1']),
                (names['C3'],next(a for a in after.atoms if a.name=='N')),
                (names['N3'],names['C1'])]:
        assert frozenset((a.idx,b.idx)) in bonds
        distance=float(np.linalg.norm(dry.coordinates[a.idx]-dry.coordinates[b.idx]))
        assert 1.0<distance<1.8,(a.name,b.name,distance)
        linked.append({'atoms':[f'{a.residue.idx+1}:{a.name}',f'{b.residue.idx+1}:{b.name}'],'angstrom':distance})
    assert len(dry.residues)==219
    assert abs(sum(a.charge for a in solv.atoms))<1e-3
    audit={'system_id':system_id,'dry_atoms':len(dry.atoms),'solvated_atoms':len(solv.atoms),
           'CR2_charge_e':charge,'CR2_bonds_verified':linked,'CR2_atoms':len(cr.atoms),
           'forcefield':'ff14SB + Breyfogle et al. xFPchromophores (AmberTools distributed)',
           'water':'OPC','padding_angstrom':10,'added_NaCl_pairs':n_salt,'box_volume_A3':volume,
           'target_added_salt_M':0.15,'total_charge_e':sum(a.charge for a in solv.atoms),
           'forcefield_sha256':{str(p.relative_to(ENV)):hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in [ENV/'dat/leap/lib/xFPchromophores.lib',ENV/'dat/leap/parm/frcmod.xFPchromophores']}}
    (out/'PARAMETER_AUDIT.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('system_id');a=p.parse_args();prepare(a.system_id)
