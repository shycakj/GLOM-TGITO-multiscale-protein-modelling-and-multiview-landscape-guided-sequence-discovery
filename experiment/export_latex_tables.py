"""LaTeX tables backed by completed-run CSVs, with explicit seed counts and settings."""
from pathlib import Path
import pandas as pd
LABELS={"glom":"GLOM","global_only":"GLOM: global only","no_message":"GLOM: no messaging",
 "hp_transformer":"Matched-input Transformer","onehot_ridge":"One-hot Ridge","sequence_knn_hamming":"Hamming kNN",
 "esm_mean_ridge":"ESM2-150M + Ridge","esm2_150m_lora":"ESM2-150M + LoRA",
 "esmc_600m_mlp":"ESMC-600M + MLP","esm2_650m_mlp":"ESM2-650M + MLP","esmc_6b_mlp":"ESMC-6B + MLP",
 "esmc_600m_ridge":"ESMC-600M + Ridge","esm2_650m_ridge":"ESM2-650M + Ridge","esmc_6b_ridge":"ESMC-6B + Ridge",
 "esm2_650m_rf":"ESM2-650M + RF (EVOLVEpro-style)",
 "proteinttt_650m_wt_marginals":"ProteinTTT: WT marginals",
 "proteinttt_650m_masked_marginals":"ProteinTTT: masked marginals",
 "esm2_650m_unadapted_wt_marginals":"Unadapted ESM2: WT marginals",
 "esm2_650m_unadapted_masked_marginals":"Unadapted ESM2: masked marginals"}
def write_tables(root,d):
    out=Path(root)/"reports/latex";out.mkdir(exist_ok=True);flat=[]
    keys=["spearman","pearson","rmse","mae","r2","top6_true_mean","top30_true_mean"]
    for split in ["random","depth"]:
        for setting in ["supervised","sequence_adaptation"]:
            isttt=d.method.str.contains("proteinttt|unadapted",regex=True)
            x=d[d.split.eq(split)&(isttt if setting=="sequence_adaptation" else ~isttt)]
            if x.empty:continue
            lines=[r"% Generated from completed_regression_runs.csv; requires booktabs and graphicx.",r"\begin{table*}[t]",r"\centering",
                r"\resizebox{\textwidth}{!}{%",r"\begin{tabular}{lrrrrrrrr}",r"\toprule",
                r"Method & Seeds & Spearman $\rho$ & Pearson $r$ & RMSE & MAE & $R^2$ & Top-6 & Top-30 \\",r"\midrule"]
            for method in [m for m in LABELS if m in set(x.method)]:
                group=x[x.method.eq(method)];cells=[];record={"split":split,"method":method,"setting":setting,"seeds":len(group)}
                for key in keys:
                    mu=float(group[key].mean());sd=float(group[key].std())
                    cells.append(f"${mu:.3f}\\,\\pm\\,{sd:.3f}$" if len(group)>1 else f"${mu:.3f}$")
                    record[key+"_mean"]=mu;record[key+"_sd"]=sd
                flat.append(record);lines.append(LABELS[method]+" & "+str(len(group))+" & "+" & ".join(cells)+r" \\")
            lines.extend([r"\bottomrule",r"\end{tabular}}"])
            title="Random holdout" if split=="random" else "Mutation-depth extrapolation (training depth at most two, test depth at least three)"
            note="Mean and sample standard deviation across completed training seeds. Top-6 and Top-30 are mean measured brightness of selected test candidates."
            if setting=="sequence_adaptation":note+=" Adaptation uses sequence only; regression errors use an affine brightness map fitted on training labels. Raw correlations are retained in the accompanying results."
            lines.extend([r"\caption{"+title+". "+note+"}",r"\label{tab:"+split+"_"+setting+"}",r"\end{table*}"])
            (out/("table_"+split+"_"+setting+".tex")).write_text("\n".join(lines)+"\n",encoding="utf-8")
    if flat:pd.DataFrame(flat).to_csv(Path(root)/"reports/summary_flat.csv",index=False)
    write_evidence_tables(Path(root),out)
    from write_results_snapshot import write_snapshot
    write_snapshot(root)

def write_evidence_tables(root,out):
    base=root/"artifacts/models/random/glom_seed20261013"
    def table(filename,headers,rows,caption):
        lines=[r"% Generated from completed experiment artifacts; requires booktabs.",
            r"\begin{table*}[t]",r"\centering",r"\small",
            r"\begin{tabular}{l"+"r"*(len(headers)-1)+"}",r"\toprule",
            " & ".join(headers)+r" \\",r"\midrule"]
        lines.extend(" & ".join(str(c) for c in row)+r" \\" for row in rows)
        lines.extend([r"\bottomrule",r"\end{tabular}",r"\caption{"+caption+"}",
            r"\label{tab:"+filename+"}",r"\end{table*}"])
        (out/("table_"+filename+".tex")).write_text("\n".join(lines)+"\n",encoding="utf-8")
    p=base/"representation_analysis/branch_linear_probes.csv"
    if p.exists():
        d=pd.read_csv(p);names={"G":"Global","Rf":"Region","Kf":"Key-site","h32":"Joint bottleneck","G_Rf_Kf":"Concatenated branches"}
        rows=[[names[r.representation],*[f"{getattr(r,k):.3f}" for k in ["spearman","pearson","rmse","mae","r2"]]] for r in d.itertuples()]
        table("branch_probes",["Representation",r"Spearman $\rho$",r"Pearson $r$","RMSE","MAE",r"$R^2$"],rows,
            "Linear probes of the prespecified GLOM seed 20261013. Ridge regularization was selected on validation data; all reported metrics use the held-out test sequences.")
    p=base/"risk_screening/test_selection_metrics.csv"
    if p.exists():
        d=pd.read_csv(p).set_index(["method","k"])
        names={"prediction":"Prediction", "confidence_full":"Fixed full confidence", "confidence_without_final_risk":"Without final risk",
            "rule_gate_full_confidence":"Rule gate", "single_view_full_confidence":"Single view", "calibrated_risk_full_confidence":"Calibrated risk gate"}
        rows=[]
        for key,label in names.items():
            a=d.loc[(key,6)];b=d.loc[(key,30)]
            rows.append([label,f"{a.true_mean:.3f}",f"{a.true_min:.3f}",f"{a.high435:.3f}",f"{b.true_mean:.3f}",f"{b.true_min:.3f}",f"{b.high435:.3f}"])
        table("risk_screening",["Ranking","Top-6 mean","Top-6 min",r"Top-6 $\geq4.35$","Top-30 mean","Top-30 min",r"Top-30 $\geq4.35$"],rows,
            "Measured held-out candidate screening. Threshold columns are selected fractions. The calibrated risk configuration was chosen on the independent calibration partition; fixed confidence retains the v13.5 weights.")
    p=base/"risk_screening/paired_library_robustness_summary.csv"
    if p.exists():
        d=pd.read_csv(p);d=d[d.partition.eq("test")].set_index(["method","k"])
        names={"confidence_full":"Fixed full confidence","confidence_without_final_risk":"Without final risk",
            "rule_gate_full_confidence":"Rule gate","single_view_full_confidence":"Single view","calibrated_risk_full_confidence":"Calibrated risk gate"}
        rows=[]
        for key,label in names.items():
            a=d.loc[(key,6)];b=d.loc[(key,30)]
            rows.append([label,f"{a.mean_brightness_difference:+.4f}",f"{a.fraction_improved:.3f}",
                f"{b.mean_brightness_difference:+.4f}",f"{b.fraction_improved:.3f}"])
        table("risk_library_robustness",["Ranking",r"Top-6 $\Delta$","Fraction improved",r"Top-30 $\Delta$","Fraction improved"],rows,
            "Robustness across 200 paired candidate libraries of 512 distinct test sequences. Differences are relative to prediction-only ranking within the same library. Candidate subsets may overlap; fractions describe these libraries and are not independent-trial success probabilities.")
    p=base/"submission_policy/submission_policy_robustness.csv"
    if p.exists():
        d=pd.read_csv(p);d=d[d.partition.eq("test")].set_index("method")
        names={"prediction":"Prediction","confidence_full":"Fixed full confidence","confidence_without_final_risk":"Without final risk",
            "rule_gate_full_confidence":"Rule gate","single_view_full_confidence":"Single view","calibrated_risk_full_confidence":"Calibrated risk gate"}
        rows=[]
        for key,label in names.items():
            r=d.loc[key];rows.append([label,str(int(r.full_six_libraries)),str(int(r.jointly_full_with_prediction)),
                f"{r.mean_gain_vs_prediction_common_full:+.4f}",f"{r.fraction_improved_common_full:.3f}"])
        table("submission_policy",["Ranking","Full-six libraries","Paired full-six",r"Mean $\Delta$","Fraction improved"],rows,
            "Original v13.5 final selection: predicted brightness at least 4.31 and pairwise Hamming distance at least two. All 200 libraries are accounted for; brightness differences are computed on libraries where both the method and prediction ranking return six candidates. Score normalization precedes filtering, matching the original pipeline.")
    p=base/"search/runtime_and_queries.csv"
    if p.exists() and (base/"search/DONE.json").exists():
        d=pd.read_csv(p);names={"tgito_risk":"TGITO","tgito_no_risk_gate":"TGITO: no risk gate","tgito_no_terrain":"TGITO: no terrain",
            "tgito_single_view":"TGITO: single view","random":"Random","adalead":"AdaLead"}
        keys=["forward_sequences","gradient_forward_sequences","valid_unique_candidates","seconds"]
        rows=[[names[key],str(len(g)),*[f"{g[k].mean():.1f}" for k in keys]] for key,g in d.groupby("method")]
        table("search_budget",["Method","Runs","Forward sequences","With gradients","Unique candidates","Seconds"],rows,
            "Mean cost per completed search run. Random and AdaLead use the full TGITO forward-sequence budget. Common risk rescoring is recorded separately in the accompanying data. Elapsed times reflect shared-GPU execution.")
if __name__=="__main__":
    root=Path(__file__).resolve().parent;write_tables(root,pd.read_csv(root/"reports/completed_regression_runs.csv"))
