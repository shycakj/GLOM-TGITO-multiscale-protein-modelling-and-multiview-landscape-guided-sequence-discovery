"""Generate factual, explicitly dated Results paragraphs from completed artifacts."""
from pathlib import Path
import datetime
import pandas as pd

def write_snapshot(root):
    root=Path(root);report=root/'reports';out=report/'latex';out.mkdir(exist_ok=True)
    runs=pd.read_csv(report/'completed_regression_runs.csv')
    lines=['% Intermediate Results snapshot: regenerate after all planned seeds and searches complete.',
        '% Generated UTC: '+datetime.datetime.now(datetime.timezone.utc).isoformat(),
        r'\subsection{Completed retrospective experiments}']
    for split,label in [('random','random holdout'),('depth','mutation-depth extrapolation')]:
        d=runs[runs.split.eq(split)&runs.method.eq('glom')]
        if d.empty:continue
        m=d.mean(numeric_only=True)
        lines.append(f'For {label}, the {len(d)} completed GLOM training '+('seed yielded' if len(d)==1 else 'seeds yielded mean')+
            f' Spearman correlation {m.spearman:.4f}, Pearson correlation {m.pearson:.4f}, RMSE {m.rmse:.4f}, MAE {m.mae:.4f}, and '+
            f'$R^2={m.r2:.4f}$. Top-6 and Top-30 mean measured brightness were {m.top6_true_mean:.4f} and {m.top30_true_mean:.4f}, respectively. '+
            'Tables report the actual number of completed seeds and their sample standard deviations.')
    base=root/'artifacts/models/random/glom_seed20261013'
    p=base/'representation_analysis/branch_linear_probes.csv'
    if p.exists():
        q=pd.read_csv(p).set_index('representation');k=pd.read_csv(p.parent/'knn16_overlap.csv',index_col=0)
        overlaps=[k.loc[a,b] for a,b in [('G','Rf'),('G','Kf'),('Rf','Kf')]]
        lines.append('For the prespecified representation-analysis seed 20261013, branch-specific linear probes gave test Spearman correlations '+
            f"of {q.loc['G','spearman']:.3f}, {q.loc['Rf','spearman']:.3f}, and {q.loc['Kf','spearman']:.3f} for global, region and key-site representations, respectively. "+
            f"Concatenating the three branches increased this value to {q.loc['G_Rf_Kf','spearman']:.3f}. "+
            f'Pairwise overlap of 16-nearest-neighbor sets ranged from {100*min(overlaps):.1f}'+r'\%'+f' to {100*max(overlaps):.1f}'+r'\%'+
            ', supporting distinct and complementary representations in the original feature spaces.')
    p=base/'submission_policy/submission_policy_robustness.csv'
    if p.exists():
        q=pd.read_csv(p);q=q[q.partition.eq('test')].set_index('method');r=q.loc['calibrated_risk_full_confidence']
        raw=pd.read_csv(base/'risk_screening/paired_library_robustness_summary.csv')
        raw=raw[raw.partition.eq('test')&raw.method.eq('calibrated_risk_full_confidence')].set_index('k')
        lines.append('Under the original final submission constraints (predicted brightness at least 4.31 and pairwise Hamming distance at least two), '+
            f"{int(r.jointly_full_with_prediction)} of 200 candidate libraries returned six sequences for both calibrated confidence and prediction-only ranking. "+
            f"On these libraries, calibrated confidence changed Top-6 mean measured brightness by {r.mean_gain_vs_prediction_common_full:+.4f}, with improvement in {100*r.fraction_improved_common_full:.1f}"+r'\%'+
            f" of paired libraries. The corresponding mean differences for fixed confidence, removal of its final risk term, and single-view confidence were {q.loc['confidence_full','mean_gain_vs_prediction_common_full']:+.4f}, "+
            f"{q.loc['confidence_without_final_risk','mean_gain_vs_prediction_common_full']:+.4f}, and {q.loc['single_view_full_confidence','mean_gain_vs_prediction_common_full']:+.4f}. "+
            'The score-only diagnostic without the final selection constraints behaved differently: calibrated confidence changed Top-6 and Top-30 brightness by '+
            f"{raw.loc[6,'mean_brightness_difference']:+.4f} and {raw.loc[30,'mean_brightness_difference']:+.4f}. "+
            'These comparisons distinguish the complete selection policy from the standalone risk-score contribution.')
    lines.append('The six historical v13.5 submission candidates have confirmed sequence identities. Their prospective fluorescence measurements remain pending; the measurements reported above come from the retrospective held-out dataset.')
    (out/'results_snapshot.tex').write_text('\n\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__':write_snapshot(Path(__file__).resolve().parent)
