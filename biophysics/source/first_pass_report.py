"""Freeze a complete replicate-0 report without changing the MD campaign."""
import csv
import hashlib
import io
import json
import math
import shutil
import statistics as st
import tarfile
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'first_pass_report'

def read_csv(path):
    with path.open(newline='', encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))

def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def main():
    if (ROOT/'FIRST_PASS_READY.json').exists():
        print('First-pass report already frozen.', flush=True)
        return
    systems = read_csv(ROOT/'inputs/systems.csv')
    pairs = read_csv(ROOT/'inputs/candidate_start_pairs.csv')
    assert len(systems)==56 and len(pairs)==30
    ids = [r['system_id'] for r in systems]
    deadline = time.time()+12*3600
    previous = None
    while True:
        complete = [sid for sid in ids
                    if (ROOT/'md'/sid/'rep0/DONE.json').exists()
                    and (ROOT/'results'/sid/'rep0/METRICS.json').exists()]
        if len(complete)!=previous:
            print('FIRST_PASS_READY_SYSTEMS', len(complete), '/ 56', flush=True)
            previous = len(complete)
        if len(complete)==56:
            break
        if time.time()>deadline:
            raise TimeoutError('First pass incomplete; no report labelled complete.')
        time.sleep(30)

    records = {sid: json.loads((ROOT/'results'/sid/'rep0/METRICS.json').read_text()) for sid in ids}
    for sid, row in records.items():
        done = json.loads((ROOT/'md'/sid/'rep0/DONE.json').read_text())
        assert row['replicate']==0 and row['frames']==300 and row['production_ns']==3, sid
        assert done['production_restraints']==0, sid
        assert all(math.isfinite(v) for v in row['metrics'].values()), sid
    OUT.mkdir(exist_ok=True)
    data = OUT/'data'
    data.mkdir(exist_ok=True)
    source = OUT/'source'
    source.mkdir(exist_ok=True)
    shutil.copy2(__file__, source/'first_pass_report.py')
    for name in ['candidate_start_pairs.csv', 'systems.csv']:
        shutil.copy2(ROOT/'inputs'/name, data/name)
    for name in ['SPURS_predictions.csv', 'sequence_solubility.csv', 'all_top30_mutation_context.csv',
                 'top30_biological_context.csv', 'SPURS_AUDIT.json', 'SWI_AUDIT.json']:
        shutil.copy2(ROOT/'results'/name, data/name)
    for sid in ids:
        dest = data/'replicate0'/sid
        dest.mkdir(parents=True, exist_ok=True)
        for name in ['METRICS.json','per_residue.csv','timeseries.csv']:
            shutil.copy2(ROOT/'results'/sid/'rep0'/name, dest/name)

    values = {sid:r['metrics'] for sid,r in records.items()}
    metrics = list(values['WT'])
    comparisons = []
    for pair in pairs:
        for label, reference in [('WT','WT'),('start',pair['start_system_id'])]:
            for metric in metrics:
                v,b = values[pair['candidate_id']][metric],values[reference][metric]
                comparisons.append({'candidate_id':pair['candidate_id'],'rank':int(pair['rank']),
                    'comparison':label,'reference_id':reference,'metric':metric,'candidate':v,
                    'reference':b,'delta':v-b,'replicate':0,'production_ns':3})
    write_csv(data/'all30_paired_metrics.csv', comparisons)
    summary = {}
    summary_rows = []
    for label in ['WT','start']:
        summary[label] = {}
        for metric in metrics:
            rows = [r for r in comparisons if r['comparison']==label and r['metric']==metric]
            assert len(rows)==30
            item = {'candidate_count':30,'candidate_mean':st.mean(r['candidate'] for r in rows),
                    'reference_mean':st.mean(r['reference'] for r in rows),
                    'mean_delta':st.mean(r['delta'] for r in rows),
                    'median_delta':st.median(r['delta'] for r in rows),
                    'lower_count':sum(r['delta']<0 for r in rows),
                    'higher_count':sum(r['delta']>0 for r in rows)}
            summary[label][metric] = item
            summary_rows.append({'comparison':label,'metric':metric,**item})
    write_csv(data/'group_summary.csv', summary_rows)
    local_rows = []
    for pair in pairs:
        cand = {int(r['project_position']):r for r in read_csv(data/'replicate0'/pair['candidate_id']/'per_residue.csv')}
        for label,ref,field in [('WT','WT','mutations_vs_wt'),('start',pair['start_system_id'],'mutations_vs_start')]:
            reference = {int(r['project_position']):r for r in read_csv(data/'replicate0'/ref/'per_residue.csv')}
            for mutation in pair[field].split(';'):
                pos = int(mutation[1:-1])
                row = {'candidate_id':pair['candidate_id'],'comparison':label,'reference_id':ref,'mutation':mutation}
                for metric in ['ca_rmsf_A','chromophore_distance_mean_A','sasa_mean_A2']:
                    v,b = float(cand[pos][metric]),float(reference[pos][metric])
                    row[metric]=v;row[metric+'_reference']=b;row[metric+'_delta']=v-b
                local_rows.append(row)
    write_csv(data/'mutation_local_changes.csv', local_rows)

    sol = {r['system_id']:float(r['SWI_mean']) for r in read_csv(data/'sequence_solubility.csv')}
    spurs = read_csv(data/'SPURS_predictions.csv')
    static = {}
    for label in ['WT','start']:
        ddg = [float(r['SPURS_ddG_kcal_mol_mean']) for r in spurs if r['comparison']==label]
        diffs = [sol[r['candidate_id']]-sol['WT' if label=='WT' else r['start_system_id']] for r in pairs]
        static[label] = {'SPURS_negative_count':sum(x<0 for x in ddg),'SPURS_mean_ddG':st.mean(ddg),
                         'SWI_higher_count':sum(x>0 for x in diffs),'SWI_mean_delta':st.mean(diffs)}
    audit = {'complete_first_pass':True,'systems':56,'candidates':30,'replicates_per_system':1,
             'replicate_index':0,'production_ns_per_system':3,'equilibration_ns_per_system':0.5,
             'comparison_rows':len(comparisons),'all_systems_finite_and_300_frames':True,
             'completed_utc':datetime.now(timezone.utc).isoformat(),
             'three_replicate_campaign_continues':True,'summary':summary,'static':static,
             'interpretation':'Single-replicate descriptive results; no fluorescence labels or replicate confidence intervals.'}
    (OUT/'SUMMARY.json').write_text(json.dumps(audit,indent=2))
    labels = [
        ('beta_native_contact_fraction','β桶原生接触保持率',100,'%'),
        ('beta_ca_rmsf_mean_A','β桶Cα波动',1,'Å'),
        ('chromophore_I_torsion_circular_sd_deg','发色团I扭转角波动',1,'°'),
        ('chromophore_P_torsion_circular_sd_deg','发色团P扭转角波动',1,'°'),
        ('chromophore_phenolate_protein_hbond_occupancy','发色团酚氧端与蛋白的氢键占据率',100,'%'),
        ('hydrophobic_sidechain_sasa_observed_noninterface_A2','观测核心非界面疏水侧链暴露面积',1,'Å²'),
        ('largest_noninterface_hydrophobic_patch_5A_A2','最大非界面疏水斑块',1,'Å²')]
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M CST')
    lines = ['Top30首轮完整结果报告',stamp,'',
             '全部30条候选、25条不同起点和WT均完成第一次模拟及分析。每个体系0.5 ns平衡、3 ns生产模拟，共56条轨迹。',
             '以下为首次重复的描述性结果。后两次独立重复继续运行；各结果的重复性将在完成后汇总。','',
             '各指标按同一候选分别与WT、实际搜索起点配对比较。候选排序不因这些结果改变。',
             '群体均值（候选 / WT / 起点）与相对起点的方向计数：']
    for key,label,scale,unit in labels:
        a,b = summary['WT'][key],summary['start'][key]
        lines.append(f"{label}：{a['candidate_mean']*scale:.3f} / {a['reference_mean']*scale:.3f} / {b['reference_mean']*scale:.3f} {unit}；低于起点{b['lower_count']}/30，高于起点{b['higher_count']}/30。")
    lines += ['', '稳定性和溶解性独立评分：']
    for label in ['WT','start']:
        a=static[label]
        lines.append(f"相对{label}：SPURS预测ΔΔG均值{a['SPURS_mean_ddG']:+.3f} kcal/mol，预测更稳定（ΔΔG<0）{a['SPURS_negative_count']}/30；SWI均值变化{a['SWI_mean_delta']:+.6f}，升高{a['SWI_higher_count']}/30。")
    lines += ['', '生物学解释范围：',
              'β桶接触、局部波动描述骨架保持及柔性变化；发色团I/P角及蛋白氢键描述局部环境，方向需结合具体位点解释，不能统一按越大或越小越好计分。',
              '疏水暴露同时区分天然二聚体界面和未解析末端；核心非界面面积与最大疏水斑块分别报告。',
              '短时间成熟蛋白模拟不直接测量折叠速度、发色团成熟速率、荧光量子产率或表达产量。',
              'SPURS与SWI为预测评分，所有正负变化均保留。','', '全部30条的具体变化（先对起点，再对WT）：']
    for pair in pairs:
        lines += ['',f"{pair['candidate_id']}；相对WT突变：{pair['mutations_vs_wt']}；搜索引入：{pair['mutations_vs_start']}"]
        for label,ref in [('起点',pair['start_system_id']),('WT','WT')]:
            c,b=values[pair['candidate_id']],values[ref]
            lines.append(f"  对{label}：I/P角波动变化{c['chromophore_I_torsion_circular_sd_deg']-b['chromophore_I_torsion_circular_sd_deg']:+.3f}/{c['chromophore_P_torsion_circular_sd_deg']-b['chromophore_P_torsion_circular_sd_deg']:+.3f}°；蛋白氢键占据率变化{100*(c['chromophore_phenolate_protein_hbond_occupancy']-b['chromophore_phenolate_protein_hbond_occupancy']):+.2f}个百分点；β桶接触变化{100*(c['beta_native_contact_fraction']-b['beta_native_contact_fraction']):+.3f}个百分点；核心非界面疏水暴露变化{c['hydrophobic_sidechain_sasa_observed_noninterface_A2']-b['hydrophobic_sidechain_sasa_observed_noninterface_A2']:+.2f} Å²。")
    (OUT/'首轮完整结果报告.txt').write_text('\n'.join(lines),encoding='utf-8')
    beta=summary['WT']['beta_native_contact_fraction']
    surf=summary['start']['hydrophobic_sidechain_sasa_observed_noninterface_A2']
    methods = ('All 30 selected candidates were compared with the wild type and their actual search starting sequences, yielding 56 distinct systems. '
               'Structures were prepared from the experimental 2G3O template with the mature chromophore retained. '
               'Explicit-solvent simulations used OpenMM, ff14SB, published xFP chromophore parameters and OPC water at 300 K and 1 bar. '
               'Each trajectory comprised 0.5 ns equilibration and 3 ns unrestrained production. '
               'Chromophore torsions and protein hydrogen bonds, beta-barrel contacts and fluctuations, and hydrophobic solvent exposure were analysed separately. '
               'Native dimer-interface residues and the modelled terminal segment were identified in the surface analysis.')
    results = (f'In the first simulation replicate, the 30 candidates retained a mean beta-barrel native-contact fraction of {beta["candidate_mean"]*100:.2f}%, compared with {beta["reference_mean"]*100:.2f}% for the wild type. '
               f'Core hydrophobic solvent exposure outside the native dimer interface was lower than the corresponding starting sequence in {surf["lower_count"]} of 30 candidates; the mean paired change was {surf["mean_delta"]:+.2f} square angstroms. '
               'Chromophore torsional fluctuations and protein hydrogen-bond occupancy are reported independently to distinguish their different responses. '
               'These first-replicate results are descriptive; the two remaining independent replicates are in progress.')
    (OUT/'writing_reference_first_replicate.txt').write_text('METHODS\n'+methods+'\n\nPRELIMINARY RESULTS\n'+results+'\n',encoding='utf-8')

    archive=ROOT/'FIRST_PASS_REPORT.tar.gz'
    manifest={}
    with tarfile.open(archive,'w:gz') as tar:
        for path in sorted(p for p in OUT.rglob('*') if p.is_file()):
            content=path.read_bytes();name=path.relative_to(OUT).as_posix()
            manifest[name]=hashlib.sha256(content).hexdigest()
            info=tar.gettarinfo(str(path),arcname=name);info.size=len(content)
            tar.addfile(info,io.BytesIO(content))
        content=json.dumps(manifest,indent=2).encode()
        info=tarfile.TarInfo('MANIFEST.json');info.size=len(content)
        tar.addfile(info,io.BytesIO(content))
    ready={'first_pass_complete':True,'systems':56,'candidates':30,'replicates':1,
           'files':len(manifest),'archive_bytes':archive.stat().st_size,
           'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}
    temp=ROOT/'FIRST_PASS_READY.tmp';temp.write_text(json.dumps(ready,indent=2));temp.replace(ROOT/'FIRST_PASS_READY.json')
    print('FIRST_PASS_REPORT_READY',json.dumps(ready),flush=True)

if __name__=='__main__':
    main()
