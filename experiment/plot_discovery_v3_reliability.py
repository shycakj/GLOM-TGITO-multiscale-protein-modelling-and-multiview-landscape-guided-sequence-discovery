"""Publication plots for measured validation and frozen candidate comparisons."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from compare_discovery_v3_reliability import OUT, METHODS, NAMES


COLORS=['#187F78','#5C72A3','#A28965','#8A6DAA','#A66371']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,
                     'axes.labelsize':10,'pdf.fonttype':42,'ps.fonttype':42,
                     'axes.spines.top':False,'axes.spines.right':False,
                     'axes.edgecolor':'#B4BBC3','xtick.color':'#39424D','ytick.color':'#39424D',
                     'savefig.facecolor':'white'})


def finish(fig,name):
    for suffix in ['png','pdf','svg']:
        fig.savefig(OUT/f'{name}.{suffix}',dpi=240,bbox_inches='tight')
    plt.close(fig)


def main():
    matched=pd.read_csv(OUT/'matched_measured_validation.csv').set_index('group').loc[['low','high']]
    curves=pd.read_csv(OUT/'risk_coverage_curves.csv')
    coverage=pd.read_csv(OUT/'empirical_bound_test_validation.csv')
    fig,axes=plt.subplots(2,2,figsize=(10.6,7.1),layout='constrained')
    fig.suptitle('Measured evidence for reliability assessment',fontsize=15,x=.03,ha='left')
    for ax,metric,title,ylabel in [(axes[0,0],'rmse','A  Error at matched prediction and distance','RMSE'),
                                  (axes[0,1],'false_bright_fraction','B  Overoptimistic predictions','Measured fluorescence < 4.2 (%)')]:
        prefix='rmse' if metric=='rmse' else 'false_bright'
        scale=1 if metric=='rmse' else 100
        means=matched[metric].to_numpy()*scale
        err=np.vstack([means-matched[prefix+'_ci_low'].to_numpy()*scale,matched[prefix+'_ci_high'].to_numpy()*scale-means])
        ax.bar([0,1],means,color=[COLORS[0],'#B5BDC6'],width=.55,zorder=3)
        ax.errorbar([0,1],means,yerr=err,fmt='none',ecolor='#34414D',capsize=4,lw=1.2,zorder=4)
        ax.set_xticks([0,1],['Low risk (n=210)','High risk (n=295)'])
        ax.set_ylabel(ylabel);ax.set_title(title,loc='left',pad=12)
        ax.set_ylim(0,float((means+err[1]).max())*1.22);ax.grid(axis='y',color='#EDF0F3',zorder=0)
        for i,y in enumerate(means):ax.text(i,y/2,f'{y:.3f}' if metric=='rmse' else f'{y:.1f}%',ha='center',va='center',color='white',weight='bold')
    ax=axes[1,0]
    for score,color,label in [('combined_risk',COLORS[0],'Terrain risk + prediction gap'),('cross_sd',COLORS[1],'Unused-seed disagreement'),('negative_support',COLORS[2],'Inverse terrain support')]:
        d=curves[curves.scope.eq('predicted_high420')&curves.score.eq(score)].sort_values('retained_fraction')
        ax.plot(d.retained_fraction*100,d.rmse,'o-',color=color,label=label,lw=1.7,ms=4)
    ax.set_title('C  Selective prediction on measured sequences',loc='left',pad=12)
    ax.set_xlabel('Measured candidates retained (%)');ax.set_ylabel('RMSE');ax.grid(color='#EDF0F3');ax.legend(frameon=False,fontsize=8,loc='upper left')
    ax=axes[1,1]
    for i,model in enumerate(['constant','risk_scaled']):
        d=coverage[coverage.alpha.eq(.1)&coverage.model.eq(model)&coverage.stratum.isin(['high_d1','high_d2plus'])].set_index('stratum').loc[['high_d1','high_d2plus']]
        x=np.arange(2)+(i-.5)*.19;y=d.lower_coverage.to_numpy()*100
        err=np.vstack([y-d.coverage_wilson_low.to_numpy()*100,d.coverage_wilson_high.to_numpy()*100-y])
        ax.errorbar(x,y,yerr=err,fmt='o',color=[COLORS[1],COLORS[0]][i],capsize=4,label=['Constant error margin','Terrain-scaled margin'][i])
    ax.axhline(90,color='#6C7886',lw=1,ls='--');ax.set_ylim(82,96)
    ax.set_xticks([0,1],['Distance 1\n(n=671)','Distance >=2\n(n=295)'])
    ax.set_ylabel('Empirical lower-bound coverage (%)');ax.set_title('D  Calibration checked on measured test data',loc='left',pad=12)
    ax.legend(frameon=False,fontsize=8,loc='lower left');ax.grid(axis='y',color='#EDF0F3')
    finish(fig,'reliability_measured_validation')

    table=pd.read_csv(OUT/'all_methods_reliability.csv');table=table[table.batch_size.eq(100)].set_index('method').loc[METHODS]
    joint=pd.read_csv(OUT/'joint_novelty_reliability_primary.csv');joint=joint[joint.batch_size.eq(100)].set_index('method').loc[METHODS]
    fig,axes=plt.subplots(2,3,figsize=(13.4,7.1),layout='constrained')
    fig.suptitle('Reliability of frozen Top100 discovery candidates',fontsize=15,x=.03,ha='left')
    panels=[('terrain_risk','A  Terrain risk (lower is better)'),('terrain_support','B  Terrain support (higher is better)'),
            ('cross_sd','C  Unused-seed disagreement'),('risk_scaled_lower90','D  Empirical lower brightness'),
            ('joint','E  Novel candidates passing the reliability rule'),('novel','F  Distance from visible sequences >=2')]
    for i,(ax,(metric,title)) in enumerate(zip(axes.flat,panels)):
        if metric in ['joint','novel']:
            y=joint['novel_and_reliability_pass_n' if metric=='joint' else 'novel_n'].to_numpy()
            err=None
        else:
            y=table[metric+'_mean'].to_numpy()
            err=np.vstack([y-table[metric+'_ci_low'].to_numpy(),table[metric+'_ci_high'].to_numpy()-y])
        for j in range(5):
            if metric=='risk_scaled_lower90':
                ax.scatter(y[j],j,color=COLORS[j],s=44,zorder=4)
            else:
                ax.barh(j,y[j],height=.56,color=COLORS[j],zorder=3)
            if err is not None:ax.errorbar(y[j],j,xerr=err[:,j:j+1],fmt='none',ecolor='#39424D',capsize=2.5,lw=.9,zorder=4)
        ax.invert_yaxis();ax.set_yticks(np.arange(5),[NAMES[m] for m in METHODS] if i%3==0 else ['']*5)
        ax.set_title(title,loc='left',pad=12);ax.grid(axis='x',color='#EDF0F3',zorder=0)
        if metric=='risk_scaled_lower90':ax.set_xlim(4.15,4.43);ax.axvline(4.2,color='#707C89',ls='--',lw=1)
        elif metric in ['joint','novel']:ax.set_xlim(0,105 if metric=='novel' else 46);ax.set_xlabel('Candidate count')
        else:ax.set_xlim(0,max(y+(err[1] if err is not None else 0))*1.2)
        for j,v in enumerate(y):
            label=f'{int(v)}' if metric in ['joint','novel'] else f'{v:.4f}'
            x=v+(err[1,j] if err is not None else 0)
            ax.annotate(label,(x,j),xytext=(4,0),textcoords='offset points',va='center',fontsize=8)
    finish(fig,'reliability_frozen_candidate_comparison')
    captions=(
        'Figure 1. Measured validation of reliability metrics. A-B: common-stratum-weighted comparison of low versus high terrain risk, '\
        'using calibration-defined quartiles within fixed prediction and training-distance strata; bars show stratified sequence bootstrap 95% intervals. '\
        'C: thresholds from calibration only; all measured test candidates have cross-seed predicted fluorescence >=4.2. '\
        'D: observed coverage of nominal 90% empirical lower bounds; Wilson reference intervals.\n\n'\
        'Figure 2. All five frozen Top100 batches. A-D: means with within-campaign peak/anchor-group bootstrap 95% intervals. '\
        'E: count satisfying distance >=2 from all visible sequences, terrain-scaled empirical lower >=4.2, risk <= the calibration median, '\
        'and unused-seed SD <= the calibration 75th percentile; both cutoffs use the cross-predicted >=4.2 calibration subgroup. '\
        'F: number at distance >=2 before the reliability rule. Search and native selection are unchanged. '\
        'These computational measurements do not replace unknown fluorescence values.\n')
    (OUT/'figure_captions.txt').write_text(captions,encoding='utf-8')
    print(json.dumps({'figures':2,'formats':['pdf','png','svg'],'out':str(OUT)}))


if __name__=='__main__':main()
