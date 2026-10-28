"""Describe every Top30 substitution on a common experimental structural template."""
import csv, json
from pathlib import Path
import numpy as np
import pandas as pd
import mdtraj as md
from scipy.spatial.distance import cdist
from analyze_md import interface_positions, residue_position
ROOT=Path(__file__).resolve().parents[1]
def describe(sid):
    p=ROOT/'structures'/sid
    traj=md.load(str(p/'dry.inpcrd'),top=str(p/'dry.prmtop'))
    heavy=[a.index for a in traj.topology.atoms if a.element.symbol!='H']
    traj=traj.atom_slice(heavy);xyz=traj.xyz[0]
    sasa=md.shrake_rupley(traj,mode='residue',n_sphere_points=960)[0]*100
    ss=md.compute_dssp(traj,simplified=False)[0]
    cr=next(r for r in traj.topology.residues if r.name=='CR2');cridx=[a.index for a in cr.atoms]
    bypos={residue_position(r):r for r in traj.topology.residues if r.name!='CR2'}
    iface=interface_positions();out={}
    for pos,r in bypos.items():
        ii=[a.index for a in r.atoms]
        neighbours=[]
        for q,t in bypos.items():
            if abs(pos-q)<=2:continue
            jj=[a.index for a in t.atoms]
            distance=float(cdist(xyz[ii],xyz[jj]).min()*10)
            if distance<=4.5:neighbours.append((q,distance))
        out[pos]={'system_id':sid,'position':pos,'residue':r.name,'sasa_A2':float(sasa[r.index]),
                  'chromophore_distance_A':float(cdist(xyz[ii],xyz[cridx]).min()*10),
                  'secondary_structure':ss[r.index],'dimer_interface':pos in iface,'modeled_tail':pos>=219,
                  'nonadjacent_contacts':';'.join(f'{q}:{d:.2f}' for q,d in sorted(neighbours)),
                  'nonadjacent_contact_count':len(neighbours)}
    return out
if __name__=='__main__':
    pairs=list(csv.DictReader(open(ROOT/'inputs/candidate_start_pairs.csv')))
    systems=list(csv.DictReader(open(ROOT/'inputs/systems.csv')))
    all_data={}
    for s in systems:
        all_data[s['system_id']]=describe(s['system_id']);print('CONTEXT',s['system_id'],flush=True)
    pd.DataFrame([r for d in all_data.values() for r in d.values()]).to_csv(ROOT/'results/structural_context_all_residues.csv',index=False)
    rows=[]
    for pair in pairs:
        for comparison,ref,field in [('WT','WT','mutations_vs_wt'),('start',pair['start_system_id'],'mutations_vs_start')]:
            for mut in pair[field].split(';'):
                pos=int(mut[1:-1]);a=all_data[pair['candidate_id']][pos];b=all_data[ref][pos]
                contexts=[]
                if min(a['chromophore_distance_A'],b['chromophore_distance_A'])<=8:contexts.append('chromophore-proximal packing')
                if a['secondary_structure'] in ('E','B') or b['secondary_structure'] in ('E','B'):contexts.append('beta-barrel scaffold')
                if b['sasa_A2']>=20:contexts.append('solvent-exposed surface')
                if a['dimer_interface']:contexts.append('native dimer interface')
                if a['modeled_tail']:contexts.append('unresolved C-terminal tail (modeled)')
                if not contexts:contexts.append('local scaffold contacts')
                rows.append({'candidate_id':pair['candidate_id'],'rank':int(pair['rank']),'comparison':comparison,
                    'reference_id':ref,'mutation':mut,'position':pos,'structural_context':'; '.join(contexts),
                    'reference_sasa_A2':b['sasa_A2'],'candidate_sasa_A2':a['sasa_A2'],
                    'delta_template_sasa_A2':a['sasa_A2']-b['sasa_A2'],
                    'reference_chromophore_distance_A':b['chromophore_distance_A'],
                    'candidate_chromophore_distance_A':a['chromophore_distance_A'],
                    'reference_contacts':b['nonadjacent_contacts'],'candidate_contacts':a['nonadjacent_contacts'],
                    'dimer_interface':a['dimer_interface'],'modeled_tail':a['modeled_tail'],
                    'scope':'template mapping and constructed side chains; dynamic effects are evaluated separately'})
    pd.DataFrame(rows).to_csv(ROOT/'results/all_top30_mutation_context.csv',index=False)
    grouped=[]
    for p in pairs:
        rr=[r for r in rows if r['candidate_id']==p['candidate_id'] and r['comparison']=='WT']
        grouped.append({**p,'structural_contexts':'; '.join(sorted({x.strip() for r in rr for x in r['structural_context'].split(';')})),
                        'interface_mutations':';'.join(r['mutation'] for r in rr if r['dimer_interface']),
                        'chromophore_proximal_mutations':';'.join(r['mutation'] for r in rr if r['reference_chromophore_distance_A']<=8),
                        'modeled_tail_mutations':';'.join(r['mutation'] for r in rr if r['modeled_tail'])})
    pd.DataFrame(grouped).to_csv(ROOT/'results/top30_biological_context.csv',index=False)
    print(json.dumps({'candidates':len(grouped),'mutation_comparisons':len(rows),'systems':len(all_data)}))
