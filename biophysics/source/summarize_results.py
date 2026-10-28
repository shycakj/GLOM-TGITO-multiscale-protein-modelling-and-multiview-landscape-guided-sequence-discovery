"""Generate all-candidate comparisons, preserving missing and unfavorable results."""
import json, os
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR']=str(ROOT/'cache/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=ROOT/'results';PLOT=OUT/'figures';PLOT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'svg.fonttype':'none',
                     'axes.spines.top':False,'axes.spines.right':False})
pairs=pd.read_csv(ROOT/'inputs/candidate_start_pairs.csv').fillna('')
summ=pairs.copy();summ['md_status']='pending'
solpath=OUT/'sequence_solubility.csv'
if solpath.exists():
    sol=pd.read_csv(solpath).set_index('system_id')
    for col in sol:
        summ[col]=summ.candidate_id.map(sol[col])
        summ[f'{col}_delta_vs_WT']=summ[col]-sol.at['WT',col]
        summ[f'{col}_delta_vs_start']=summ[col]-summ.start_system_id.map(sol[col])
sp=OUT/'SPURS_predictions.csv'
if sp.exists():
    stability=pd.read_csv(sp)
    for comp in ['WT','start']:
        s=stability[stability.comparison==comp].set_index('candidate_id')
        summ[f'SPURS_ddG_vs_{comp}_kcal_mol']=summ.candidate_id.map(s.SPURS_ddG_kcal_mol_mean)
raw=[]
for path in OUT.glob('*/rep[0-2]/METRICS.json'):
    r=json.loads(path.read_text());raw.append({'system_id':r['system_id'],'replicate':r['replicate'],
                                            'production_ns':r['production_ns'],**r['metrics']})
long=[];perres=[]
if raw:
    data=pd.DataFrame(raw);data.to_csv(OUT/'md_metrics_by_replicate.csv',index=False)
    metrics=[c for c in data if c not in ('system_id','replicate','production_ns')]
    for sid,dd in data.groupby('system_id'):
        for m in metrics:
            long.append({'system_id':sid,'metric':m,'mean':dd[m].mean(),'sd_across_replicates':dd[m].std(),
                         'replicates_completed':len(dd),'production_ns_per_rep':dd.production_ns.min()})
    agg=pd.DataFrame(long);agg.to_csv(OUT/'md_metrics_summary.csv',index=False)
    means=data.groupby('system_id')[metrics].mean()
    for m in metrics:
        summ[m]=summ.candidate_id.map(means[m])
        if 'WT' in means.index:summ[f'{m}_delta_vs_WT']=summ[m]-means.at['WT',m]
        summ[f'{m}_delta_vs_start']=summ[m]-summ.start_system_id.map(means[m])
    counts=data.groupby('system_id').size()
    for i,r in summ.iterrows():
        n=min(counts.get(r.candidate_id,0),counts.get(r.start_system_id,0),counts.get('WT',0))
        summ.at[i,'md_status']='complete_3_replicates' if n==3 else f'partial_{n}_matched_replicates'
    # Per-mutation local dynamics for every candidate, each compared to WT and its own start.
    locals_by_id={}
    for sid in data.system_id.unique():
        frames=[]
        for p in (OUT/sid).glob('rep[0-2]/per_residue.csv'):
            df=pd.read_csv(p);df['replicate']=int(p.parent.name[3:]);frames.append(df)
        if frames:locals_by_id[sid]=pd.concat(frames).groupby('project_position').mean(numeric_only=True)
    for _,r in pairs.iterrows():
        for comp,reference,field in [('WT','WT','mutations_vs_wt'),('start',r.start_system_id,'mutations_vs_start')]:
            if r.candidate_id not in locals_by_id or reference not in locals_by_id:continue
            c=locals_by_id[r.candidate_id];b=locals_by_id[reference]
            for mutation in r[field].split(';'):
                p=int(mutation[1:-1]);row={'candidate_id':r.candidate_id,'comparison':comp,'reference_id':reference,'mutation':mutation,'position':p}
                for col in ['ca_rmsf_A','chromophore_distance_mean_A','sasa_mean_A2']:
                    row[col]=c.at[p,col];row[col+'_reference']=b.at[p,col];row[col+'_delta']=c.at[p,col]-b.at[p,col]
                perres.append(row)
    if perres:pd.DataFrame(perres).to_csv(OUT/'all_top30_local_dynamic_changes.csv',index=False)
summ.to_csv(OUT/'top30_biophysics_comparison.csv',index=False)
if sp.exists() and solpath.exists():
    fig,axes=plt.subplots(1,3,figsize=(11.4,3.6),layout='constrained')
    x=summ.SPURS_ddG_vs_WT_kcal_mol;y=summ.SPURS_ddG_vs_start_kcal_mol
    ax=axes[0];ax.axhline(0,color='.75',lw=.8);ax.axvline(0,color='.75',lw=.8)
    ax.scatter(x,y,c='#477FA5',s=32,edgecolors='white',linewidths=.5)
    ax.set(xlabel='Predicted ΔΔG vs WT (kcal/mol)',ylabel='Predicted ΔΔG vs start (kcal/mol)',title='a  Combinatorial stability prediction')
    for ax,comp,title in zip(axes[1:],['WT','start'],['b  Solubility tendency vs WT','c  Solubility tendency vs start']):
        d=summ[f'SWI_mean_delta_vs_{comp}']*1000
        ax.axvline(0,color='.65',lw=.7);ax.hlines(summ['rank'],0,d,color='#C0CED8',lw=.8)
        ax.scatter(d,summ['rank'],c=np.where(d>=0,'#447DA3','#B76C66'),s=18)
        ax.set(xlabel='Change in SWI (×10⁻³)',ylabel='Candidate rank',title=title,ylim=(31,0))
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
        ax.set_yticks([1,5,10,15,20,25,30])
    for ext in ['pdf','svg','png']:fig.savefig(PLOT/f'open_tool_stability_solubility.{ext}',dpi=250)
    plt.close(fig)
summary={'candidate_count':len(summ),'md_replicates_completed':len(raw),
         'candidate_comparisons_complete':int((summ.md_status=='complete_3_replicates').sum()),
         'all_directions_preserved':True}
if sp.exists():
    summary['SPURS']={comp:{'scored':int(summ[f'SPURS_ddG_vs_{comp}_kcal_mol'].notna().sum()),
          'negative_ddG_count':int((summ[f'SPURS_ddG_vs_{comp}_kcal_mol']<0).sum()),
          'mean_ddG':float(summ[f'SPURS_ddG_vs_{comp}_kcal_mol'].mean())} for comp in ['WT','start']}
if solpath.exists():
    summary['SWI']={comp:{'higher_count':int((summ[f'SWI_mean_delta_vs_{comp}']>0).sum()),
                         'mean_delta':float(summ[f'SWI_mean_delta_vs_{comp}'].mean())} for comp in ['WT','start']}
(OUT/'SUMMARY.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
