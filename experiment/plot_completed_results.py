"""Publication-sized plots from completed runs only; reproducible SVG/PNG/PDF exports."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS={"glom":"GLOM","global_only":"GLOM: global only","no_message":"GLOM: no messaging",
 "hp_transformer":"Matched-input Transformer","onehot_ridge":"One-hot Ridge",
 "sequence_knn_hamming":"Hamming kNN","esm_mean_ridge":"ESM2-150M + Ridge",
 "esm2_150m_lora":"ESM2-150M + LoRA","esmc_600m_mlp":"ESMC-600M + MLP",
 "esm2_650m_mlp":"ESM2-650M + MLP","esmc_6b_mlp":"ESMC-6B + MLP",
 "esmc_600m_ridge":"ESMC-600M + Ridge","esm2_650m_ridge":"ESM2-650M + Ridge",
 "esmc_6b_ridge":"ESMC-6B + Ridge","esm2_650m_rf":"ESM2-650M + RF (EVOLVEpro-style)"}
ORDER=list(LABELS)
COLORS={"glom":"#0F766E","global_only":"#5A9D96","no_message":"#91BCB7",
 "hp_transformer":"#626F83","onehot_ridge":"#A5ACB5","sequence_knn_hamming":"#C4CAD1",
 "esm2_150m_lora":"#946DB1","esmc_600m_mlp":"#547CAE","esm2_650m_mlp":"#7B9DBF",
 "esmc_6b_mlp":"#305777"}

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,default=Path(__file__).resolve().parent)
    ap.add_argument("--pdf",action="store_true");a=ap.parse_args();root=a.root
    report=root/"reports";out=report/"figures";out.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(report/"completed_regression_runs.csv")
    from risk_diagnostics import write_diagnostics
    write_diagnostics(root)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.titlesize":11,
        "axes.labelsize":9,"axes.spines.top":False,"axes.spines.right":False,
        "svg.fonttype":"none","pdf.fonttype":42,"savefig.facecolor":"white"})
    files=[]
    from matplotlib.backends.backend_pdf import PdfPages
    combined=PdfPages(out/"dry_experiment_figures.pdf") if a.pdf else None
    def save(fig,name):
        if combined is not None:combined.savefig(fig,bbox_inches="tight")
        for ext in (["svg","png","pdf"] if a.pdf else ["svg","png"]):
            p=out/(name+"."+ext);fig.savefig(p,dpi=220,bbox_inches="tight");files.append(str(p.relative_to(root)))
        plt.close(fig)
    # Main comparisons retain a fixed order; error bars are between-seed standard deviations.
    for split in ["random","depth"]:
        x=d[d.split.eq(split)&d.method.isin(ORDER)]
        if x.empty:continue
        methods=[m for m in ORDER if m in set(x.method)]
        metrics=[("spearman","Spearman rho",True),("pearson","Pearson r",True),
                 ("rmse","RMSE",False),("mae","MAE",False),("r2","R squared",True),
                 ("top6_true_mean","Top-6 measured brightness",True)]
        fig,axs=plt.subplots(1,6,figsize=(15.6,max(3.6,len(methods)*.39)),sharey=True)
        yy=np.arange(len(methods))
        for ax,(key,label,higher) in zip(axs,metrics):
            g=x.groupby("method")[key];mean=g.mean().reindex(methods);sd=g.std().fillna(0).reindex(methods)
            for i,m in enumerate(methods):
                color=COLORS.get(m,"#7B8CA0")
                ax.errorbar(mean.iloc[i],i,xerr=sd.iloc[i],fmt="o",color=color,ecolor=color,
                            markersize=5,capsize=2,elinewidth=1.1)
            ax.set_title(label+ (" ↑" if higher else " ↓"),pad=14);ax.grid(axis="x",color="#E9EDF0",zorder=0)
            ax.set_axisbelow(True);ax.spines["left"].set_visible(False);ax.tick_params(axis="y",length=0)
            ax.set_yticks(yy);ax.set_yticklabels([LABELS[m]+f"  (n={int((x.method==m).sum())})" for m in methods])
            ax.margins(x=.15)
        axs[0].invert_yaxis()
        fig.suptitle("Random holdout" if split=="random" else "Mutation-depth extrapolation: train ≤2, test ≥3",
                     x=.01,y=1.02,ha="left",fontsize=14,fontweight="bold")
        fig.text(.01,-.025,"Completed runs only. Points: seed mean; bars: seed SD. Brightness is reported on the dataset scale.",fontsize=8,color="#5B6570")
        fig.tight_layout(w_pad=1.2);save(fig,"regression_"+split)
    base=root/"artifacts/models/random/glom_seed20261013"
    rep=base/"representation_analysis"
    for filename,title,limits in [("linear_CKA.csv","Representation similarity (linear CKA)",(0,1)),
                                    ("knn16_overlap.csv","Shared nearest neighbors (k = 16)",(0,1))]:
        if not (rep/filename).exists():continue
        x=pd.read_csv(rep/filename,index_col=0)
        names=[n for n in ["G","Rf","Kf","h32","global_attn_pool","global_mean_pool","global_max_pool",
            "region9_max_pool","region9_mean_pool","keysite_K_flat"] if n in x.index]
        labels={"G":"Global branch","Rf":"Region branch","Kf":"Key-site branch","h32":"Joint bottleneck",
            "global_attn_pool":"Global attention","global_mean_pool":"Global mean","global_max_pool":"Global max",
            "region9_max_pool":"Regions: max","region9_mean_pool":"Regions: mean","keysite_K_flat":"Key-site tokens"}
        fig,ax=plt.subplots(figsize=(6.6,5.5));im=ax.imshow(x.loc[names,names],cmap="YlGnBu",vmin=limits[0],vmax=limits[1])
        ax.set_xticks(range(len(names)),[labels[n] for n in names],rotation=50,ha="right")
        ax.set_yticks(range(len(names)),[labels[n] for n in names]);ax.set_title(title,pad=14)
        for i in range(len(names)):
            for j in range(len(names)):
                v=float(x.loc[names[i],names[j]])
                ax.text(j,i,f"{v:.2f}",ha="center",va="center",fontsize=7,color="white" if v>.64 else "#243544")
        fig.colorbar(im,ax=ax,fraction=.04,pad=.025);fig.tight_layout();save(fig,filename.replace(".csv",""))
    risk=base/"risk_screening/test_selection_metrics.csv"
    if risk.exists():
        x=pd.read_csv(risk);names=["prediction","confidence_full","confidence_without_final_risk",
                                  "rule_gate_full_confidence","single_view_full_confidence","calibrated_risk_full_confidence"]
        labels=["Prediction","Fixed full confidence","Without final risk","Rule gate","Single view","Calibrated risk gate"]
        fig,axs=plt.subplots(1,2,figsize=(9.4,3.8),sharey=True)
        for ax,k in zip(axs,[6,30]):
            v=x[x.k.eq(k)].set_index("method").loc[names]
            ax.scatter(v.true_mean,np.arange(len(names)),c=["#8A9DB2","#0F766E","#83B3AD","#9EADBB","#B9C4CE","#133E39"],s=42,zorder=3)
            for i,r in enumerate(v.itertuples()):ax.text(r.true_mean+.012,i,f"{r.true_mean:.3f}",va="center",fontsize=8)
            ax.set_yticks(np.arange(len(names)),labels);ax.set_xlim(v.true_mean.min()-.04,v.true_mean.max()+.13)
            ax.set_title(f"Top-{k} measured brightness");ax.grid(axis="x",color="#E9EDF0");ax.set_axisbelow(True)
        axs[0].invert_yaxis();fig.text(.01,-.025,"Score-only rankings. Final eligibility and diversity constraints are evaluated separately.",fontsize=8)
        fig.tight_layout();save(fig,"risk_candidate_selection")
    robustness=base/"risk_screening/paired_library_robustness_summary.csv"
    if robustness.exists():
        x=pd.read_csv(robustness);x=x[x.partition.eq("test")]
        names=["confidence_full","confidence_without_final_risk","rule_gate_full_confidence",
            "single_view_full_confidence","calibrated_risk_full_confidence"]
        labels=["Fixed full confidence","Without final risk","Rule gate","Single view","Calibrated risk gate"]
        fig,axs=plt.subplots(1,2,figsize=(10,3.8),sharey=True)
        for ax,k in zip(axs,[6,30]):
            v=x[x.k.eq(k)].set_index("method").loc[names]
            ax.hlines(np.arange(len(names)),v.q25_difference,v.q75_difference,color="#97ACA9",lw=3)
            ax.scatter(v.mean_brightness_difference,np.arange(len(names)),color="#0F766E",s=40,zorder=3)
            ax.axvline(0,color="#51606B",lw=1,ls="--");ax.set_yticks(np.arange(len(names)),labels)
            ax.set_title(f"Top-{k}: brightness change vs prediction");ax.grid(axis="x",color="#E9EDF0");ax.set_axisbelow(True)
        axs[0].invert_yaxis();fig.text(.01,-.02,"Score-only diagnostic: 200 libraries of 512 sequences. Points: mean; lines: library interquartile spread (not CI).",fontsize=8)
        fig.tight_layout();save(fig,"risk_library_robustness")
    submission=base/"submission_policy/submission_policy_robustness.csv"
    if submission.exists():
        x=pd.read_csv(submission);x=x[x.partition.eq("test")].set_index("method")
        names=["confidence_full","confidence_without_final_risk","rule_gate_full_confidence",
            "single_view_full_confidence","calibrated_risk_full_confidence"]
        labels=["Fixed full confidence","Without final risk","Rule gate","Single view","Calibrated risk gate"]
        fig,ax=plt.subplots(figsize=(8,3.8));v=x.loc[names]
        ax.scatter(v.mean_gain_vs_prediction_common_full,np.arange(len(names)),color="#0F766E",s=45,zorder=3)
        ax.axvline(0,color="#51606B",lw=1,ls="--");ax.set_yticks(np.arange(len(names)),labels);ax.invert_yaxis()
        ax.set_title("Full v13.5 selection policy: Top-6 change vs prediction")
        ax.set_xlabel("Mean brightness difference on libraries returning six candidates")
        ax.grid(axis="x",color="#E9EDF0");ax.set_axisbelow(True)
        for i,r in enumerate(v.itertuples()):ax.annotate(f"n={int(r.jointly_full_with_prediction)}/200",(r.mean_gain_vs_prediction_common_full,i),xytext=(8,0),textcoords="offset points",va="center",fontsize=8)
        ax.margins(x=.3);fig.text(.01,-.025,"Prediction ≥4.31; pairwise Hamming distance ≥2. Same eligibility and diversity constraints for every ranking.",fontsize=8)
        fig.tight_layout();save(fig,"submission_policy_robustness")
    if combined is not None:
        combined.close();files.append(str((out/"dry_experiment_figures.pdf").relative_to(root)))
    (out/"figure_manifest.json").write_text(json.dumps({"files":files,"rows_in_report":len(d),
       "statistics":"mean and sample SD across completed training seeds; not a confidence interval",
       "test_selection_of_methods":False},indent=2))
    from export_latex_tables import write_tables
    write_tables(root,d)
    print(json.dumps({"figures":files}))
if __name__=="__main__":main()
