"""Per-replicate physical observables; no fluorescence labels are inferred."""
import argparse, csv, json
from pathlib import Path
import numpy as np
import pandas as pd
import mdtraj as md
from scipy.spatial.distance import cdist
from scipy.stats import circstd
from Bio.SeqUtils import ProtParamData, seq1

ROOT=Path(__file__).resolve().parents[1]
def residue_position(res):
    return res.index+1 if res.index<56 else res.index+3
def atom_lookup(top):
    return {(a.residue.index,a.name):a.index for a in top.atoms}
def ca_map(top):
    return {residue_position(a.residue):a.index for a in top.atoms if a.name=='CA'}
def hbond_count(xyz, triples):
    if not triples:return np.zeros((len(xyz),0),bool)
    tr=np.array(triples,dtype=int)
    dh=xyz[:,tr[:,0]]-xyz[:,tr[:,1]]
    ah=xyz[:,tr[:,2]]-xyz[:,tr[:,1]]
    da=xyz[:,tr[:,0]]-xyz[:,tr[:,2]]
    cos=np.sum(dh*ah,axis=-1)/(np.linalg.norm(dh,axis=-1)*np.linalg.norm(ah,axis=-1))
    return (np.linalg.norm(da,axis=-1)<=.35)&(cos<=np.cos(np.deg2rad(150)))

def interface_positions():
    dest=ROOT/'inputs/reference_interface_positions.json'
    if dest.exists() and json.loads(dest.read_text()).get('version')==2:return set(json.loads(dest.read_text())['positions'])
    ref=md.load(str(ROOT/'inputs/2G3O_reference.pdb'))
    # Filter by author residue numbers, retaining the chromophore and excluding waters.
    inds={cid:[a.index for a in ref.topology.atoms if a.residue.chain.chain_id==cid and
               a.element.symbol not in ('H','VS') and (a.residue.is_protein or a.residue.name=='CR2')]
          for cid in ('C','E')}
    assert len(inds['C'])>1500 and len(inds['E'])>1500
    c=ref.atom_slice(inds['C']); ce=ref.atom_slice(inds['C']+inds['E'])
    sas_c=md.shrake_rupley(c,mode='atom',n_sphere_points=960)[0]*100
    ce_order=sorted(inds['C']+inds['E']);ce_index={global_i:i for i,global_i in enumerate(ce_order)}
    sas_ce=md.shrake_rupley(ce,mode='atom',n_sphere_points=960)[0][[ce_index[i] for i in inds['C']]]*100
    buried={}
    for a,delta in zip(c.topology.atoms,sas_c-sas_ce):
        if a.residue.name!='CR2':buried[a.residue.resSeq-1]=buried.get(a.residue.resSeq-1,0.)+float(delta)
    pos=sorted(k for k,v in buried.items() if v>=5)
    assert len(pos)>=5, 'Biological interface mapping failed'
    dest.write_text(json.dumps({'version':2,'positions':pos,'definition':'at least 5 A2 buried by chain E in biological C/E dimer',
                              'buried_sasa_A2':buried},indent=2))
    return set(pos)

def analyze(sid,rep):
    source=ROOT/'structures'/sid;run=ROOT/'md'/sid/f'rep{rep}'
    done=json.loads((run/'DONE.json').read_text())
    target=ROOT/'results'/sid/f'rep{rep}';target.mkdir(parents=True,exist_ok=True)
    # AMBER dry and solvated topologies have an identical solute atom prefix.
    topology=md.load_prmtop(str(source/'dry.prmtop'))
    traj=md.load_dcd(str(run/'solute.dcd'),top=topology)
    if len(traj)<5:raise ValueError('Insufficient saved frames')
    ref=md.load(str(ROOT/'structures/WT/dry.inpcrd'),top=str(ROOT/'structures/WT/dry.prmtop'))
    cam=ca_map(topology);refcam=ca_map(ref.topology)
    positions=sorted(set(cam)&set(refcam))
    alignpos=[p for p in positions if 3<=p<=215]
    traj.superpose(ref,atom_indices=[cam[p] for p in alignpos],ref_atom_indices=[refcam[p] for p in alignpos])
    ca_xyz=traj.xyz[:,[cam[p] for p in positions]]
    ref_ca=ref.xyz[0,[refcam[p] for p in positions]]
    rmsf=np.sqrt(np.mean(np.sum((ca_xyz-ca_xyz.mean(axis=0))**2,axis=-1),axis=0))*10
    ca_rmsd=np.sqrt(np.mean(np.sum((ca_xyz-ref_ca)**2,axis=-1),axis=1))*10
    dssp=md.compute_dssp(ref,simplified=True)[0]
    beta_positions={residue_position(r) for r,ss in zip(ref.topology.residues,dssp) if ss=='E' and r.name!='CR2'}
    beta_pos=sorted(beta_positions&set(positions))
    native=[];native_ref=[]
    for i,p in enumerate(beta_pos):
        for q in beta_pos[i+1:]:
            if abs(p-q)<=3:continue
            distance=np.linalg.norm(ref.xyz[0,refcam[p]]-ref.xyz[0,refcam[q]])
            if distance<.8:native.append((cam[p],cam[q]));native_ref.append(distance)
    native_d=md.compute_distances(traj,native,periodic=False)
    contact_fraction=(native_d<=1.2*np.array(native_ref)).mean(axis=1)
    look=atom_lookup(topology)
    chrom=next(r for r in topology.residues if r.name=='CR2')
    cr_atoms={a.name:a.index for a in chrom.atoms}; cr_heavy=[a.index for a in chrom.atoms if a.element.symbol!='H']
    torsion_atoms=[['N2','CA2','CB2','CG2'],['CA2','CB2','CG2','CD1']]
    torsions=md.compute_dihedrals(traj,[[cr_atoms[n] for n in names] for names in torsion_atoms],periodic=False)
    donors=[]
    for a,b in topology.bonds:
        if a.element.symbol=='H':a,b=b,a
        if b.element.symbol=='H' and a.element.symbol in ('N','O','S'):donors.append((a.index,b.index))
    triples=[];labels=[]
    for acc_name in ('OH','N2','O2','O3'):
        for d,h in donors:
            if topology.atom(d).residue.index==chrom.index:continue
            triples.append((d,h,cr_atoms[acc_name]));labels.append((residue_position(topology.atom(d).residue),acc_name,d,h))
    chrom_hb=hbond_count(traj.xyz,triples)
    hb_rows=[]
    for i,(p,n,d,h) in enumerate(labels):
        occ=float(chrom_hb[:,i].mean())
        if occ>0:hb_rows.append({'project_position':p,'chromophore_acceptor':n,'donor_atom':topology.atom(d).name,
                                 'occupancy':occ})
    pd.DataFrame(hb_rows,columns=['project_position','chromophore_acceptor','donor_atom','occupancy']).to_csv(target/'chromophore_hbonds.csv',index=False)
    phen_idx=[i for i,l in enumerate(labels) if l[1]=='OH']
    phen_any=chrom_hb[:,phen_idx].any(axis=1)
    beta_donors=[(d,h) for d,h in donors if topology.atom(d).name=='N' and residue_position(topology.atom(d).residue) in beta_positions]
    beta_acceptors=[a.index for a in topology.atoms if a.name=='O' and residue_position(a.residue) in beta_positions]
    beta_triples=[(d,h,a) for d,h in beta_donors for a in beta_acceptors
                  if abs(residue_position(topology.atom(d).residue)-residue_position(topology.atom(a).residue))>3]
    beta_hb=hbond_count(traj.xyz,beta_triples).sum(axis=1)
    # Surface descriptors use heavy-atom SASA with standard 1.4-A probe.
    heavy=[a.index for a in topology.atoms if a.element.symbol!='H']
    sampled=traj[::5].atom_slice(heavy)
    sasa=md.shrake_rupley(sampled,probe_radius=.14,n_sphere_points=960,mode='atom')*100
    iface=interface_positions()
    aa_scale=ProtParamData.gravy_scales['BlackMould'];gly=aa_scale['G']
    backbone={'N','CA','C','O','OXT'}
    hydro=np.zeros(sampled.n_atoms,bool);noninterface=np.ones(sampled.n_atoms,bool);side=np.zeros(sampled.n_atoms,bool)
    weights=np.zeros(sampled.n_atoms);observed=np.ones(sampled.n_atoms,bool)
    res_atoms={}
    for a in sampled.topology.atoms:
        p=residue_position(a.residue)
        if a.residue.name=='CR2':continue
        aa=seq1(a.residue.name,custom_map={'HIE':'H','HID':'H','HIP':'H','CYX':'C'})
        res_atoms.setdefault(p,[]).append(a.index)
        noninterface[a.index]=p not in iface
        observed[a.index]=p<=218
        if a.name not in backbone:
            side[a.index]=True;weights[a.index]=aa_scale[aa]-gly
            hydro[a.index]=aa in 'AVILMFWYPC' and a.element.symbol in ('C','S')
    hydro_sasa=sasa[:,hydro].sum(axis=1)
    hydro_noninterface=sasa[:,hydro&noninterface].sum(axis=1)
    weighted_sasa=(sasa*weights).sum(axis=1)
    # Radius-5-A surface patch size, evaluated at exposed hydrophobic side-chain atoms.
    patch_max=[];patch_noniface=[]
    for xyz,area in zip(sampled.xyz,sasa):
        inds=np.flatnonzero(hydro & (area>0.05))
        if len(inds):
            near=cdist(xyz[inds],xyz[inds])<=.5
            patches=near@area[inds];patch_max.append(float(patches.max()))
            inds2=np.flatnonzero(hydro&noninterface&(area>0.05))
            patch_noniface.append(float(((cdist(xyz[inds2],xyz[inds2])<=.5)@area[inds2]).max()) if len(inds2) else 0.)
        else:patch_max.append(0.);patch_noniface.append(0.)
    all_dist={}
    heavy_by_res={r.index:[a.index for a in r.atoms if a.element.symbol!='H'] for r in topology.residues}
    for r in topology.residues:
        if r.name=='CR2':continue
        p=residue_position(r)
        # Distance between any residue heavy atom and any chromophore heavy atom.
        delta=traj.xyz[:,heavy_by_res[r.index],None,:]-traj.xyz[:,None,cr_heavy,:]
        all_dist[p]=np.sqrt((delta*delta).sum(axis=-1)).min(axis=(1,2))*10
    rows=[]
    for p in positions:
        rr=topology.atom(cam[p]).residue
        ix=positions.index(p)
        indices=res_atoms[p]
        rows.append({'project_position':p,'residue_name':rr.name,'ca_rmsf_A':float(rmsf[ix]),
                     'chromophore_distance_mean_A':float(all_dist[p].mean()),
                     'chromophore_distance_sd_A':float(all_dist[p].std()),
                     'sasa_mean_A2':float(sasa[:,indices].sum(axis=1).mean()),
                     'is_reference_beta':p in beta_positions,'is_reference_dimer_interface':p in iface,
                     'is_modeled_tail':p>=219})
    pd.DataFrame(rows).to_csv(target/'per_residue.csv',index=False)
    metrics={'ca_rmsd_mean_A':float(ca_rmsd.mean()),'beta_native_contact_fraction':float(contact_fraction.mean()),
             'beta_backbone_hbond_mean_count':float(beta_hb.mean()),
             'beta_ca_rmsf_mean_A':float(np.mean([rmsf[positions.index(p)] for p in beta_pos])),
             'chromophore_protein_hbond_mean_count':float(chrom_hb.sum(axis=1).mean()),
             'chromophore_phenolate_protein_hbond_occupancy':float(phen_any.mean()),
             'chromophore_I_torsion_circular_sd_deg':float(np.degrees(circstd(torsions[:,0]))),
             'chromophore_P_torsion_circular_sd_deg':float(np.degrees(circstd(torsions[:,1]))),
             'hydrophobic_sidechain_sasa_A2':float(hydro_sasa.mean()),
             'hydrophobic_sidechain_sasa_noninterface_A2':float(hydro_noninterface.mean()),
             'hydrophobic_sidechain_sasa_observed_core_A2':float(sasa[:,hydro&observed].sum(axis=1).mean()),
             'hydrophobic_sidechain_sasa_observed_noninterface_A2':float(sasa[:,hydro&observed&noninterface].sum(axis=1).mean()),
             'BlackMould_weighted_sidechain_sasa_A2':float(weighted_sasa.mean()),
             'largest_hydrophobic_patch_5A_A2':float(np.mean(patch_max)),
             'largest_noninterface_hydrophobic_patch_5A_A2':float(np.mean(patch_noniface))}
    frame=pd.DataFrame({'time_ps':np.arange(1,len(traj)+1)*10,'ca_rmsd_A':ca_rmsd,
                        'beta_native_contact_fraction':contact_fraction,'beta_backbone_hbonds':beta_hb,
                        'chromophore_protein_hbonds':chrom_hb.sum(axis=1),'phenolate_protein_Hbond':phen_any.astype(int),
                        'chromophore_I_torsion_deg':np.degrees(torsions[:,0]),'chromophore_P_torsion_deg':np.degrees(torsions[:,1])})
    frame.to_csv(target/'timeseries.csv',index=False)
    halves={col:{'first_half_mean':float(frame[col].iloc[:len(frame)//2].mean()),
                  'second_half_mean':float(frame[col].iloc[len(frame)//2:].mean())}
            for col in frame if col!='time_ps' and 'torsion' not in col}
    result={'system_id':sid,'replicate':rep,'production_ns':done['production_ns'],'frames':len(traj),
            'metrics':metrics,'split_half_diagnostic':halves,
            'metric_scope':'monomer dynamics; native oligomer interface is separately annotated',
            'inference':'Structural and biophysical evidence; no fluorescent yield is inferred.'}
    (target/'METRICS.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('system_id');p.add_argument('--rep',type=int,required=True)
    a=p.parse_args();analyze(a.system_id,a.rep)
