"""Publish a complete calibration-quantile grid without changing native batches."""
from pathlib import Path
import argparse, csv, hashlib, json, shutil, zipfile
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
P=ROOT/'artifacts/discovery_v3/embedding_familiarity_v1/threshold_grid_v2'
PAPER=ROOT/'reports/full_discovery_paper_20261014'
OUT=PAPER/'multithreshold'
METHODS=['glom_tgito','glom_adalead','glom_random','glom_prothmso','esm650_prothmso']
LABELS=['GLOM-TGITO','GLOM-AdaLead','GLOM-random','GLOM-ProtHMSO','ESM650-ProtHMSO']
QS=[.50,.60,.70,.75,.80,.85,.90,.95,.975,.99]

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def qlabel(q):return 'Q'+f'{100*q:g}'
def tables():
    g=pd.read_csv(P/'threshold_comparison.csv')
    v=pd.read_csv(P/'measured_validation.csv')
    b=pd.read_csv(P/'calibration_bootstrap.csv')
    high=g[(g.reference=='predicted_high420')&(g.scope=='top100')]
    matrix=high.pivot(index='method',columns='quantile',values='exceed_n').loc[METHODS,QS]
    highv=v[v.reference=='predicted_high420'].sort_values('q').reset_index(drop=True)
    return g,v,b,matrix,highv

def make_figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    OUT.mkdir(parents=True,exist_ok=True)
    g,v,b,m,highv=tables()
    for name in ['PLAN.json','DONE.json','threshold_comparison.csv','measured_validation.csv','calibration_bootstrap.csv']:
        if (P/name).resolve()!=(OUT/name).resolve():shutil.copy2(P/name,OUT/name)
    m.index=LABELS;m.columns=[qlabel(q) for q in QS];m.to_csv(OUT/'top100_threshold_counts.csv')
    allm=g[(g.reference=='all')&(g.scope=='top100')].pivot(index='method',columns='quantile',values='exceed_n').loc[METHODS,QS]
    allm.index=LABELS;allm.columns=m.columns;allm.to_csv(OUT/'top100_all_calibration_control.csv')
    # The old counts are a directly reproducible subset of this extension.
    oldpath=P.parent/'threshold_audit/top100_threshold_sensitivity.csv'
    if not oldpath.exists():oldpath=PAPER/'source_data/sensitivity.csv'
    old=pd.read_csv(oldpath)
    for _,r in old[old.k==32].iterrows():
        row=g[(g.reference==r.reference)&(g.method==r.method)&(g.scope=='top100')&np.isclose(g['quantile'],r.q)].iloc[0]
        assert row.exceed_n==r.outside_n and np.isclose(row.threshold,r.threshold,atol=1e-12)
    assert (m.diff(axis=1).iloc[:,1:]<=0).all().all()
    # After downloading the compact raw scores, validate every grid cell locally.
    local_raw_verified=False
    compact=P/'compact_sequence_scores.csv.gz'
    if compact.exists():
        s=pd.read_csv(compact);members=pd.read_csv(P/'compact_top100_membership.csv')
        for reference in ['all','predicted_high420']:
            cal=s[s.partition=='calibration'];test=s[s.partition=='test']
            if reference=='predicted_high420':cal=cal[cal.prediction>=4.2];test=test[test.prediction>=4.2]
            for q in QS:
                cut=np.quantile(cal.z768_d32,q,method='higher')
                for method in METHODS:
                    sel=members[members.method==method].merge(s,on='feature_id',validate='many_to_one')
                    count=int((sel.z768_d32>cut).sum())
                    target=g[(g.reference==reference)&(g.method==method)&(g.scope=='top100')&np.isclose(g['quantile'],q)].iloc[0]
                    assert count==target.exceed_n and np.isclose(cut,target.threshold,atol=1e-12)
                val=v[(v.reference==reference)&np.isclose(v.q,q)].iloc[0]
                outside=test[test.z768_d32>cut];inside=test[test.z768_d32<=cut]
                for tag,dd in [('inside',inside),('outside',outside)]:
                    err=dd.prediction-dd.brightness
                    assert len(dd)==val[f'{tag}.n'] and np.isclose(np.sqrt((err**2).mean()),val[f'{tag}.rmse'])
        local_raw_verified=True
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'pdf.fonttype':42,'svg.fonttype':'none','text.color':'#293843','axes.labelcolor':'#293843'})
    fig=plt.figure(figsize=(9.0,6.1));gs=fig.add_gridspec(2,2,height_ratios=[1,1.1],left=.19,right=.98,bottom=.12,top=.91,hspace=.72,wspace=.36)
    ax=fig.add_subplot(gs[0,:]);a=m.to_numpy()
    im=ax.imshow(a,cmap='Blues',vmin=0,vmax=100,aspect='auto')
    ax.set_yticks(range(5),LABELS);ax.set_xticks(range(10),m.columns)
    ax.tick_params(length=0,labelsize=8)
    for i in range(5):
        for j in range(10):ax.text(j,i,str(a[i,j]),ha='center',va='center',fontsize=9,color='white' if a[i,j]>57 else '#233746')
    for s in ax.spines.values():s.set_visible(False)
    ax.set_title('A  Top 100 candidates above each calibration distance threshold (%)',loc='left',fontsize=10,pad=12)
    ax.set_xlabel('High-prediction calibration quantile; all methods use the same boundary',labelpad=9)
    ax=fig.add_subplot(gs[1,0]);x=np.arange(10)
    ax.plot(x,highv['inside.rmse'],'o-',color='#087F8C',ms=4,label='Within threshold')
    ax.plot(x,highv['outside.rmse'],'o-',color='#D27756',ms=4,label='Above threshold')
    ax.set_title('B  Measured predictive error',loc='left',fontsize=10,pad=12)
    ax.set_xticks(x,[qlabel(q) for q in QS],rotation=55,ha='right',fontsize=7);ax.set_ylabel('Test RMSE (log-fluorescence)');ax.set_ylim(0,.43)
    ax.legend(frameon=False,fontsize=7,loc='upper left')
    ax.text(.02,.45,'High-prediction test set\nn = 973',transform=ax.transAxes,fontsize=7,color='#637782')
    ax=fig.add_subplot(gs[1,1])
    ax.plot(x,100*(1-np.array(QS)),ls='--',color='#8D99A5',label='Nominal calibration tail')
    y=100*highv.test_exceedance.to_numpy();lo=100*highv.test_exceedance_ci_low.to_numpy();hi=100*highv.test_exceedance_ci_high.to_numpy()
    ax.errorbar(x,y,yerr=[y-lo,hi-y],fmt='o',color='#087F8C',ms=4,capsize=2,label='Measured test tail')
    ax.set_title('C  Boundary coverage on measured test data',loc='left',fontsize=10,pad=12)
    ax.set_xticks(x,[qlabel(q) for q in QS],rotation=55,ha='right',fontsize=7);ax.set_ylabel('Test sequences above threshold (%)');ax.set_ylim(0,60)
    ax.legend(frameon=False,fontsize=7,loc='upper right')
    for ax in fig.axes[1:]:
        for sp in ['top','right']:ax.spines[sp].set_visible(False)
        ax.spines['left'].set_color('#B7C2C9');ax.spines['bottom'].set_color('#B7C2C9')
    for ext in ['pdf','svg','png']:fig.savefig(OUT/f'figS2_multithreshold.{ext}',bbox_inches='tight',dpi=450)
    plt.close(fig)
    lines=[r'\begin{table*}[t]',r'\centering\small',r'\setlength{\tabcolsep}{3pt}',r'\caption{Top 100 threshold-exceedance percentages across the complete high-prediction calibration quantile grid. Every method uses the same calibration boundary. Lower quantiles measure central support; they are not binary failure labels.}',r'\label{tab:multithreshold}',r'\begin{tabular}{lrrrrrrrrrr}',r'\toprule','Method & '+' & '.join(m.columns)+r' \\',r'\midrule']
    for name,row in m.iterrows():lines.append(name+' & '+' & '.join(str(int(n)) for n in row)+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table*}']
    (OUT/'table_multithreshold.tex').write_text('\n'.join(lines),encoding='utf-8')
    text=('We expanded the secondary boundary analysis to calibration quantiles 50%, 60%, 70%, 75%, 80%, 85%, 90%, 95%, 97.5% and 99%, preserving all native batches and the original mean-deviation endpoint. At Q80, threshold-exceedance fractions were 36%, 50%, 45%, 34% and 46% for GLOM-TGITO, GLOM-AdaLead, GLOM-random, GLOM-ProtHMSO and ESM650-ProtHMSO, respectively; at Q85 they were 25%, 37%, 34%, 27% and 37%. Thus, central-to-upper-support boundaries reveal differences that a Q95 tail count alone obscures. At Q80, 205 measured high-prediction test sequences exceeded the boundary, with RMSE 0.1530 versus 0.0834 within it. At Q85, the corresponding count was 158 and RMSEs were 0.1703 versus 0.0826. Large-error rates outside these boundaries were 2.89 and 3.11 times their inside rates, respectively. The full grid also shows ranking changes: GLOM-ProtHMSO has fewer exceedances at Q80, whereas ESM650-ProtHMSO has fewer at Q60-Q70 but more in the distant tail. These threshold-dependent comparisons complement the lower mean deviation of GLOM-TGITO without replacing it. Lower-quantile exceedance denotes reduced central representation support, not an observed loss of function.')
    (OUT/'results_multithreshold_EN.txt').write_text(text+'\n',encoding='utf-8')
    (OUT/'results_multithreshold.tex').write_text(r'\paragraph{Multi-threshold support analysis.}'+'\n'+text.replace('%',r'\%')+'\n',encoding='utf-8')
    zh='''多阈值补充：采用高预测亮度校准参考，k=32，所有方法共用每档界线。

固定展示 Q50/Q60/Q70/Q75/Q80/Q85/Q90/Q95/Q97.5/Q99 共10档。
平均偏离主指标、其Q95归一化分母、模型和原生候选排名均不改变。
低分位数比较支撑集中程度，高分位数比较稀疏尾部；不能把超过Q50直接叫作蛋白无效或真正分布外。

Q80：GLOM-TGITO 36/100，AdaLead 50，随机45，GLOM-ProtHMSO 34，ESM650-ProtHMSO 46。
Q85：GLOM-TGITO 25/100，AdaLead 37，随机34，GLOM-ProtHMSO 27，ESM650-ProtHMSO 37。
Q80实测高亮测试样本：界外205个、界内768个，RMSE 0.1530 vs 0.0834。
Q85：界外158个、界内815个，RMSE 0.1703 vs 0.0826。
这两档分别对应2.89倍、3.11倍大误差比例（绝对预测误差>0.1），不是依据一两个测试样本。
Q95/97.5/99界外样本较少，继续作为尾部补充，同时报告样本量。
各阈值下优势不是完全一致；完整网格已展示，不能说任何阈值都是本方法最好。

Bootstrap仅反映校准样本造成的界线波动，候选批次固定，不是独立搜索重复的置信区间。
''' 
    (OUT/'多阈值结果说明.txt').write_text(zh,encoding='utf-8')
    (OUT/'VERIFICATION.json').write_text(json.dumps(dict(old40counts_match=True,monotone_counts=True,local_per_sequence_recalculation=local_raw_verified,top100_comparison_cells=100,primary_metric_and_native_batches_unchanged=True),indent=2),encoding='utf-8')
    print(m.to_string());print('LOCAL_RAW_RECALCULATION',local_raw_verified)

def make_pdf():
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import mm
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from pypdf import PdfReader,PdfWriter
    import html
    for n,f in [('ArialT','arial.ttf'),('ArialTB','arialbd.ttf')]:pdfmetrics.registerFont(TTFont(n,'C:/Windows/Fonts/'+f))
    st={k:ParagraphStyle(k,fontName='ArialTB' if k in ['title','h','head'] else 'ArialT',fontSize=size,leading=lead,spaceAfter=6) for k,size,lead in [('title',17,21),('h',10,13),('body',9.2,12.5),('small',8,10.5),('head',7.2,9.2),('cell',7.2,9.2)]}
    def T(s,k='body'):return Paragraph(html.escape(str(s)),st[k])
    def table(head,rows,widths):
        tt=Table([[T(x,'head') for x in head]]+[[T(x,'cell') for x in row] for row in rows],colWidths=[w*mm for w in widths],repeatRows=1)
        tt.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E7EFF2')),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#95AEB7')),('LINEBELOW',(0,-1),(-1,-1),.6,colors.HexColor('#95AEB7')),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]));return tt
    g,v,b,m,hv=tables();img=Image(str(OUT/'figS2_multithreshold.png'));img.drawHeight*=174*mm/img.drawWidth;img.drawWidth=174*mm
    story=[T('Multi-threshold representation-support analysis','title'),T('Complete native Top 100 batches; primary mean-deviation endpoint unchanged','small'),img,Spacer(1,5*mm),T('Figure S2. Central support and tail exceedance across ten calibration boundaries','h'),T('A: Percentage of each native Top 100 above the common calibration threshold (counts equal percentages because n=100). B: Prediction RMSE within and above each boundary on 973 measured sequences predicted at least 4.2. C: Actual test exceedance with Wilson 95% intervals, compared with the nominal calibration tail. Q50-Q85 describe central support; Q90-Q99 describe the more distant tail. Higher exceedance is not a measured biological failure rate.','small'),T('Interpretation','h'),T((OUT/'results_multithreshold_EN.txt').read_text(encoding='utf-8')),PageBreak(),T('Validation and calibration uncertainty','title')]
    rows=[]
    for _,r in hv.iterrows():rows.append([qlabel(r.q),f'{r.threshold:.4f}',str(int(r['outside.n'])),f"{r['inside.rmse']:.4f}",f"{r['outside.rmse']:.4f}",f"{100*r['inside.large_error_rate']:.1f}",f"{100*r['outside.large_error_rate']:.1f}",f'{r.large_error_risk_ratio:.2f}'])
    story += [T('Table S3. Each boundary is evaluated against measured fluorescence','h'),table(['Boundary','Distance','Outside n','Inside RMSE','Outside RMSE','Inside error %','Outside error %','Error ratio'],rows,[17,21,21,24,24,24,24,19]),T('Distance is measured in L2-normalized GLOM embedding space. Error % is the fraction with absolute prediction error >0.1. Total test n=973; the inside count is 973 minus outside n. Ratios are descriptive. In particular, the Q99 tail contains only 17 measured test sequences.','small'),Spacer(1,4*mm),T('Table S4. Candidate counts under calibration-threshold resampling','h')]
    picked=[.75,.80,.85,.90,.95];rows=[]
    for method,label in zip(METHODS,LABELS):
        row=[label]
        for q in picked:
            r=b[(b.reference=='predicted_high420')&(b.method==method)&np.isclose(b['quantile'],q)].iloc[0]
            row.append(f'{int(r.exceed_n)} [{r.exceed_n_ci_low:.0f}, {r.exceed_n_ci_high:.0f}]')
        rows.append(row)
    story += [table(['Method']+[qlabel(q) for q in picked],rows,[44,26,26,26,26,26]),T('Entries are the fixed-batch exceedance count and its central 95% range across 2,000 calibration resamples. All methods use shared resampled boundaries. These intervals quantify calibration uncertainty, not repeated discovery campaigns. Full results include all ten thresholds, both reference populations, and native Top 6/30/100 plus complete output pools.','small'),T('Threshold construction','h'),T('All boundaries are quantiles of the same frozen calibration-distance distribution, using the higher order statistic. The main reference comprises 950 calibration sequences predicted at least 4.2; the full-calibration control contains 3,143 sequences. No generated sequence was used to estimate a boundary, and candidate rankings were not changed. The original Q95 boundary and mean-deviation normalization are retained.','small')]
    def footer(c,d):c.setFont('ArialT',7);c.setFillColor(colors.HexColor('#6D7C86'));c.drawString(18*mm,12*mm,'GLOM-TGITO | Multi-threshold extension');c.drawRightString(192*mm,12*mm,str(d.page))
    pdf=OUT/'multithreshold_results.pdf';SimpleDocTemplate(str(pdf),pagesize=A4,leftMargin=18*mm,rightMargin=18*mm,topMargin=17*mm,bottomMargin=20*mm).build(story,onFirstPage=footer,onLaterPages=footer)
    writer=PdfWriter();writer.append(PAPER/'full_discovery_results.pdf');writer.append(pdf)
    target=PAPER/'full_discovery_results_multithreshold.pdf'
    with target.open('wb') as f:writer.write(f)
    print(json.dumps(dict(addendum_pages=len(PdfReader(pdf).pages),combined_pages=len(PdfReader(target).pages),output=str(target))))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--mode',choices=['figures','pdf'],required=True)
    a.add_argument('--input-directory',type=Path);a.add_argument('--paper-directory',type=Path);args=a.parse_args()
    if args.input_directory:P=args.input_directory.resolve()
    if args.paper_directory:PAPER=args.paper_directory.resolve();OUT=PAPER/'multithreshold'
    make_figures() if args.mode=='figures' else make_pdf()
