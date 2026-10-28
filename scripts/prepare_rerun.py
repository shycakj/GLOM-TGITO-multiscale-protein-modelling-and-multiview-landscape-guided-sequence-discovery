"""Create a separate run workspace and an explicit, frozen-configuration job plan."""
from pathlib import Path
import argparse,json,shutil,os,sys

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'experiment'

def copytree(src,dst):
    shutil.copytree(src,dst,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc','.git'))

def asset_link(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists():return
    try:dst.symlink_to(src.resolve(),target_is_directory=True)
    except OSError:copytree(src,dst)

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['training','discovery','md'])
    p.add_argument('--destination',type=Path,required=True);p.add_argument('--plan-only',action='store_true')
    p.add_argument('--reuse-trajectories',action='store_true',help='MD analysis of the supplied trajectories, without resimulation.')
    p.add_argument('--glom-only',action='store_true',help='Train only the six full GLOM fits (training stage).')
    p.add_argument('--tgito-only',action='store_true',help='Run only TGITO and native selection (discovery stage).')
    a=p.parse_args();dest=a.destination.resolve()
    protected=[SOURCE.resolve(),(ROOT/'biophysics').resolve(),(ROOT/'paper').resolve()]
    if any(dest==q or dest.is_relative_to(q) for q in protected):raise ValueError('Use a new run directory, outside the frozen input folders.')
    if dest.exists():raise FileExistsError('Destination exists. Resume its RUN_PLAN.json, or choose a new destination.')
    jobs=[]
    def add(label,args,group,environment='torch'):
        jobs.append(dict(label=label,args=args,group=group,environment=environment))
    if a.stage=='training':
        add('mutation_features',['position_features.py'],0)
        add('frozen_bridge_cache',['cache_embeddings.py','--batch-size','32'],1)
        for model in ([] if a.glom_only else ['esm2_650m','esmc_600m','esmc_6b']):
            add('cache_'+model,['cache_public_plm.py','--model-name',model,'--batch-size','16'],2,'recent' if model.startswith('esmc') else 'torch')
        if not a.glom_only:
            add('saprot_3di',['saprot_baseline_20261019/cache_saprot.py','--stage','3di','--batch-size','16'],3)
            add('saprot_features',['saprot_baseline_20261019/cache_saprot.py','--stage','saprot','--batch-size','8'],4)
        # The experiment summary enumerates every fitted configuration in the paper.
        import csv
        records=list(csv.DictReader((ROOT/'configs/TRAINING_RUNS.csv').open(encoding='utf8')))
        used=set();evaluate=[]
        for row in records:
            m,sp,seed=row['method'],row['split'],row['seed'];key=(m,sp,seed)
            if a.glom_only and m!='glom':continue
            if key in used:continue
            used.add(key);args=None
            if m in ['glom','global_only','no_message','hp_transformer']:
                args=['train_models.py','--method',m,'--split',sp,'--seed',seed,'--epochs','120','--patience','18','--batch-size','64','--lr','3e-5']
                evaluate.append((m,sp,seed))
            elif m=='esm2_150m_lora':
                args=['train_plm_lora.py','--split',sp,'--seed',seed,'--batch-size','32','--accumulation','2','--rank','16']
            elif m=='saprot650_pred3di_lora':
                args=['saprot_baseline_20261019/train_saprot_lora.py','--split',sp,'--seed',seed,'--batch-size','16','--accumulation','4','--rank','16']
            elif m.startswith('saprot650_official_'):
                mode='aa' if '_aa_' in m else 'pred3di'
                args=['saprot_official_recipe_20261019/train_official_recipe.py','--input-mode',mode,'--split',sp,'--seed',seed,'--batch-size','16','--accumulation','4','--rank','8']
            elif m.endswith(('_mlp','_ridge','_rf')) and m.startswith(('esm2_','esmc_','saprot650_')):
                feature,head=m.rsplit('_',1)
                args=['train_cached_heads.py','--feature',feature,'--head',head,'--split',sp,'--seed',seed]
            if args:add(m+'_'+sp+'_'+seed,args,5)
        for m,sp,seed in evaluate:
            args=['evaluate_models.py','--run',m+'_seed'+seed,'--split',sp]
            if (m,sp,seed)==('glom','random','20261013'):args+=['--extract-features']
            add('evaluate_'+m+'_'+sp+'_'+seed,args,6)
        if not a.glom_only:
            for sp in ['random','depth']:
                add('sequence_controls_'+sp,['baselines_cpu.py','--feature','sequence','--split',sp],6)
            add('esm_mean_control',['baselines_cpu.py','--feature','esm'],6)
            add('proteinttt',['run_proteinttt.py'],6)
        add('representation_probes',['probe_representations.py','--run','glom_seed20261013'],7)
        add('collect_reports',['collect_training_results.py'],8)
    elif a.stage=='discovery':
        # Use the archived predictor/terrain/HPO as the published search experiment did.
        add('input_hash_and_hpo_check',['discovery_v3.py','--action','prepare'],0)
        add('trial12_smoke',['discovery_v3.py','--action','run','--phase','pilot','--profile','trial12','--limit','2'],1)
        for shard in range(2):add('tgito_'+str(shard),['discovery_v3.py','--action','run','--phase','formal','--profile','trial12','--shard',str(shard),'--shards','2'],2)
        add('freeze_tgito',['discovery_v3.py','--action','freeze','--method','glom_tgito'],3)
        for method in ['glom_random','glom_adalead','prothmso_pair']:
            for shard in range(2):add(method+'_'+str(shard),['discovery_v3_baselines.py','--method',method,'--shard',str(shard),'--shards','2'],4)
        for method in ['glom_random','glom_adalead','glom_prothmso','esm650_prothmso']:
            add('freeze_'+method,['discovery_v3.py','--action','freeze','--method',method],5)
        add('freeze_evaluation',['freeze_evaluation_v3.py'],6)
        add('evaluate_discovery',['evaluate_discovery_v3.py'],7)
        add('prepare_deviation',['audit_embedding_features.py','--prepare'],8)
        for shard in range(2):add('deviation_features_'+str(shard),['audit_embedding_features.py','--shard',str(shard),'--shards','2'],9)
        add('score_deviation',['score_embedding_familiarity.py'],10)
        add('threshold_audit',['audit_familiarity_thresholds.py'],11)
        add('maxwent_control',['pilot_maxwent_head.py'],11)
        add('threshold_grid',['extend_familiarity_threshold_grid.py'],12)
        add('evaluation_snapshot',['snapshot_evaluation.py'],12)
        add('reliability_comparisons',['compare_discovery_v3_reliability.py'],13)
        add('joint_reliability',['summarize_reliable_novel_discovery.py'],14)
    else:
        import csv
        systems=list(csv.DictReader((ROOT/'biophysics/inputs/systems.csv').open()))
        if not a.reuse_trajectories:
            for sid in ['WT','C01']:add('preflight_'+sid,['source/md_run.py',sid,'--rep','97','--ns','0.1','--equil-ns','0.2','--gpu','0'],0,'md')
            for sid in ['WT','C01']:add('preflight_analysis_'+sid,['source/analyze_md.py',sid,'--rep','97'],1,'md')
            add('check_preflight',['preflight_check.py'],2,'md')
        for rep in range(3):
            for s in systems:
                sid=s['system_id'];label=sid+'_rep'+str(rep)
                if not a.reuse_trajectories:add(label,['source/md_run.py',sid,'--rep',str(rep),'--ns','3','--equil-ns','0.5','--gpu','0'],3+2*rep,'md')
                add('analyse_'+label,['source/analyze_md.py',sid,'--rep',str(rep)],4+2*rep,'md')
        add('summarize',['source/summarize_results.py'],9,'md')
    if a.tgito_only:
        if a.stage!='discovery':raise ValueError('--tgito-only applies to discovery.')
        jobs=[j for j in jobs if j['group']<=3]
    if a.glom_only and a.stage!='training':raise ValueError('--glom-only applies to training.')
    spec={'stage':a.stage,'workspace':str(dest),'jobs':jobs,'gpu_workers_per_device':1,'memory_reserve_MiB':6000,'reuse_trajectories':a.reuse_trajectories,'archived_results_modified':False}
    if a.plan_only:print(json.dumps(spec,indent=2));return
    if a.stage=='discovery':
        sys.path.insert(0,str(SOURCE))
        from hpo_config_contract import verify_input_hashes
        protocol=json.loads((SOURCE/'artifacts/discovery_v3/PROTOCOL.json').read_text())
        verify_input_hashes(SOURCE,protocol['input_hashes'])
    if a.stage=='training':
        for required in ['repro_project/assets/warm_best_model.pt','repro_project/assets/esm2_t30_150M_UR50D/model.safetensors','data/train.csv','artifacts/embeddings/P4_glom_original.npy']:
            if not (SOURCE/required).is_file():raise FileNotFoundError('Fetch data, weights and esm2_150m first: '+required)
    if a.stage=='md':
        if a.reuse_trajectories and not (ROOT/'biophysics/md').exists():raise FileNotFoundError('Trajectories are not distributed in this repository; run fresh MD or provide trajectories locally.')
        for row in systems:
            for name in ['solvated.prmtop','solvated.inpcrd']:
                if not (ROOT/'biophysics/structures'/row['system_id']/name).is_file():raise FileNotFoundError('Fetch md_inputs before preparing MD.')
    dest.mkdir(parents=True)
    if a.stage in ['training','discovery']:
        for f in SOURCE.glob('*.py'):shutil.copy2(f,dest/f.name)
        shutil.copy2(ROOT/'scripts/collect_training_results.py',dest/'collect_training_results.py')
        shutil.copy2(ROOT/'scripts/snapshot_evaluation.py',dest/'snapshot_evaluation.py')
        for name in ['saprot_baseline_20261019','saprot_official_recipe_20261019']:
            copytree(SOURCE/name,dest/name)
        copytree(SOURCE/'data',dest/'data')
        copytree(SOURCE/'repro_project/pipeline',dest/'repro_project/pipeline')
        copytree(SOURCE/'repro_project/data',dest/'repro_project/data')
        copytree(SOURCE/'repro_project/artifacts',dest/'repro_project/artifacts')
        asset_link(SOURCE/'repro_project/assets',dest/'repro_project/assets')
        asset_link(SOURCE/'third_party',dest/'third_party')
        if (SOURCE/'public_checkpoints').exists():asset_link(SOURCE/'public_checkpoints',dest/'public_checkpoints')
        (dest/'artifacts/embeddings').mkdir(parents=True,exist_ok=True)
        # Shared sequence mutation channels are deterministic and do not contain labels.
        shutil.copy2(SOURCE/'artifacts/embeddings/P4_glom_original.npy',dest/'artifacts/embeddings/P4_glom_original.npy')
        if a.stage=='discovery':
            for split in ['random','depth']:
                for folder in (SOURCE/'artifacts/models'/split).iterdir():
                    if not folder.is_dir():continue
                    for f in folder.glob('*'):
                        if f.is_file():
                            d=dest/'artifacts/models'/split/folder.name/f.name;d.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,d)
            copytree(SOURCE/'artifacts/models/random/glom_seed20261013/terrain_train_only',dest/'artifacts/models/random/glom_seed20261013/terrain_train_only')
            copytree(SOURCE/'audit',dest/'audit')
            target=dest/'artifacts/discovery_v3';target.mkdir(parents=True,exist_ok=True)
            for name in ['PROTOCOL.json','PREFLIGHT_PASSED.json','starting_sequences.csv','pilot_starting_sequences.csv','baseline_starting_sequences.csv']:
                shutil.copy2(SOURCE/'artifacts/discovery_v3'/name,target/name)
    else:
        for name in ['source','inputs','structures','tools']:copytree(ROOT/'biophysics'/name,dest/name)
        for name in ['logs','tmp','cache','results']:(dest/name).mkdir(exist_ok=True)
        if a.reuse_trajectories:asset_link(ROOT/'biophysics/md',dest/'md')
        else:(dest/'md').mkdir(exist_ok=True)
        # MD reruns vary velocities; single-sequence/static-model outputs are shared.
        for name in ['sequence_solubility.csv','SPURS_predictions.csv','SPURS_AUDIT.json','SWI_AUDIT.json','structural_context_all_residues.csv','top30_biological_context.csv']:
            f=ROOT/'biophysics/results'/name
            if f.exists():shutil.copy2(f,dest/'results'/name)
        shutil.copy2(ROOT/'scripts/md_preflight_check.py',dest/'preflight_check.py')
    for name in ['tmp','cache','logs']:(dest/name).mkdir(exist_ok=True)
    (dest/'RUN_PLAN.json').write_text(json.dumps(spec,indent=2),encoding='utf8')
    print('Created:',dest/'RUN_PLAN.json')

if __name__=='__main__':main()
