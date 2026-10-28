"""Mature CR2 explicit-solvent MD with checkpointable, bounded GPU execution."""
import argparse, json, os, time
from pathlib import Path
import numpy as np
import openmm as mm
from openmm import unit
from openmm.app import AmberPrmtopFile, AmberInpcrdFile, Simulation, PME, HBonds, PDBFile, DCDReporter, StateDataReporter, CheckpointReporter

ROOT=Path(__file__).resolve().parents[1]
def run(a):
    source=ROOT/'structures'/a.system_id
    out=ROOT/'md'/a.system_id/f'rep{a.rep}'
    out.mkdir(parents=True,exist_ok=True)
    if (out/'DONE.json').exists(): print('Already complete',out);return
    top=AmberPrmtopFile(str(source/'solvated.prmtop'))
    coords=AmberInpcrdFile(str(source/'solvated.inpcrd'))
    system=top.createSystem(nonbondedMethod=PME,nonbondedCutoff=1.0*unit.nanometer,
                            constraints=HBonds,rigidWater=True,ewaldErrorTolerance=0.0005)
    system.addForce(mm.MonteCarloBarostat(1*unit.bar,300*unit.kelvin,25))
    barostat=system.getForce(system.getNumForces()-1)
    seed=20261027+a.rep*1000
    barostat.setRandomNumberSeed(seed+100)
    # Positional restraints used only during minimization and staged equilibration.
    restraint=mm.CustomExternalForce('k_restraint*((x-x0)^2+(y-y0)^2+(z-z0)^2)')
    restraint.addGlobalParameter('k_restraint',1000.0)
    for name in ('x0','y0','z0'):restraint.addPerParticleParameter(name)
    positions=coords.positions.value_in_unit(unit.nanometer)
    solute_indices=[];heavy=[]
    for atom in top.topology.atoms():
        if atom.residue.index<219:
            solute_indices.append(atom.index)
            if atom.element is not None and atom.element.symbol!='H':
                heavy.append(atom.index);restraint.addParticle(atom.index,positions[atom.index])
    system.addForce(restraint)
    integrator=mm.LangevinMiddleIntegrator(300*unit.kelvin,1/unit.picosecond,0.002*unit.picoseconds)
    integrator.setRandomNumberSeed(seed)
    integrator.setConstraintTolerance(1e-5)
    platform=mm.Platform.getPlatformByName('CUDA')
    props={'DeviceIndex':str(a.gpu),'Precision':'mixed','TempDirectory':str(ROOT/'tmp')}
    sim=Simulation(top.topology,system,integrator,platform,props)
    sim.context.setPositions(coords.positions)
    sim.context.setPeriodicBoxVectors(*coords.boxVectors)
    t0=time.time()
    initial=sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    if not np.isfinite(initial):raise ValueError('Nonfinite initial energy')
    sim.minimizeEnergy(maxIterations=3000,tolerance=10*unit.kilojoule_per_mole/unit.nanometer)
    e_min=sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    sim.context.setVelocitiesToTemperature(300*unit.kelvin,seed)
    sim.reporters.append(StateDataReporter(str(out/'equilibration.csv'),1000,step=True,time=True,
                                          potentialEnergy=True,temperature=True,volume=True,speed=True))
    # 50 ps strong, 50 ps weak, remainder unrestrained NPT. Independent production follows.
    assert a.equil_ns>=.2
    for k,steps in ((1000.,25000),(100.,25000),(0.,int(round((a.equil_ns-.1)*500000)))):
        sim.context.setParameter('k_restraint',k);sim.step(steps)
    eq_seconds=time.time()-t0
    state=sim.context.getState(getPositions=True,getEnergy=True)
    with open(out/'equilibrated.pdb','w') as f:PDBFile.writeFile(top.topology,state.getPositions(),f)
    sim.saveState(str(out/'equilibrated.xml'))
    sim.saveCheckpoint(str(out/'equilibrated.chk'))
    sim.reporters=[]
    sim.reporters.append(DCDReporter(str(out/'solute.dcd'),5000,atomSubset=solute_indices,enforcePeriodicBox=False))
    sim.reporters.append(StateDataReporter(str(out/'production.csv'),5000,step=True,time=True,
                                          potentialEnergy=True,temperature=True,volume=True,speed=True))
    sim.reporters.append(CheckpointReporter(str(out/'latest.chk'),50000))
    audit={'system_id':a.system_id,'replicate':a.rep,'seed':seed,'gpu':a.gpu,'platform':platform.getName(),
           'openmm_version':mm.__version__,'timestep_fs':2,'temperature_K':300,'pressure_bar':1,
           'equilibration_ns':a.equil_ns,'production_target_ns':a.ns,'solute_atoms':len(solute_indices),
           'total_atoms':system.getNumParticles(),'initial_energy_kj_mol':initial,
           'minimized_energy_kj_mol':e_min,'equilibration_wall_seconds':eq_seconds,
           'production_restraints':0,'frame_interval_ps':10}
    (out/'RUNNING.json').write_text(json.dumps(audit,indent=2))
    tprod=time.time();steps=int(round(a.ns*500000))
    sim.step(steps)
    elapsed=time.time()-tprod
    final=sim.context.getState(getPositions=True,getEnergy=True)
    energy=final.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    assert np.isfinite(energy)
    with open(out/'final.pdb','w') as f:PDBFile.writeFile(top.topology,final.getPositions(),f)
    sim.saveState(str(out/'final.xml'));sim.saveCheckpoint(str(out/'final.chk'))
    audit.update({'production_ns':a.ns,'production_wall_seconds':elapsed,'ns_per_day':a.ns/elapsed*86400,
                  'wall_seconds':time.time()-t0,'final_energy_kj_mol':energy})
    (out/'DONE.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('system_id');p.add_argument('--rep',type=int,default=0)
    p.add_argument('--gpu',type=int,default=0);p.add_argument('--ns',type=float,default=.1)
    p.add_argument('--equil-ns',type=float,default=.5)
    run(p.parse_args())
