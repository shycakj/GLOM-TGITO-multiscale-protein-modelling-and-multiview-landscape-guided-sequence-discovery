"""Render the publication evidence module with reportlab; no data changes."""
from pathlib import Path
import csv, json, argparse, html
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.units import mm
from reportlab.lib.pagesizes import A4
from pypdf import PdfReader

def main():
    here=Path(__file__).resolve().parent
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=here if (here/'manuscript_content.json').exists() else here/'reports/full_discovery_paper_20261014')
    args=ap.parse_args();out=args.output.resolve()
    fonts=Path('C:/Windows/Fonts')
    for name,file in [('Body','arial.ttf'),('BodyBold','arialbd.ttf'),('BodyItalic','ariali.ttf')]:
        pdfmetrics.registerFont(TTFont(name,str(fonts/file)))
    pdfmetrics.registerFontFamily('Body',normal='Body',bold='BodyBold',italic='BodyItalic',boldItalic='BodyBold')
    styles={
      'title':ParagraphStyle('title',fontName='BodyBold',fontSize=20,leading=24,textColor=colors.HexColor('#173D4A'),spaceAfter=10),
      'h1':ParagraphStyle('h1',fontName='BodyBold',fontSize=15,leading=19,textColor=colors.HexColor('#173D4A'),spaceAfter=12),
      'h2':ParagraphStyle('h2',fontName='BodyBold',fontSize=10,leading=13,spaceBefore=9,spaceAfter=5,keepWithNext=True),
      'body':ParagraphStyle('body',fontName='Body',fontSize=9.4,leading=13.2,spaceAfter=8),
      'small':ParagraphStyle('small',fontName='Body',fontSize=8.0,leading=10.8,spaceAfter=7),
      'cell':ParagraphStyle('cell',fontName='Body',fontSize=7.2,leading=9.5),
      'head':ParagraphStyle('head',fontName='BodyBold',fontSize=7.2,leading=9.5,textColor=colors.HexColor('#173D4A')),
    }
    def clean(s):return str(s).replace('–','-').replace('—','-').replace('−','-').replace('\u2011','-')
    def P(s,kind='body'):return Paragraph(html.escape(clean(s)),styles[kind])
    def paragraphs(items):
        ans=[]
        for title,text in items:ans += [P(title,'h2'),P(text)]
        return ans
    def table(headers,rows,widths):
        data=[[P(h,'head') for h in headers]]+[[P(x,'cell') for x in r] for r in rows]
        tab=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
        commands=[('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#EAF0F2')),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#93AAB3')),('LINEBELOW',(0,-1),(-1,-1),.6,colors.HexColor('#93AAB3'))]
        if len(rows)==5:commands += [('BACKGROUND',(0,1),(-1,1),colors.HexColor('#ECF7F5'))]
        tab.setStyle(TableStyle(commands));return tab
    data=json.loads((out/'manuscript_content.json').read_text(encoding='utf-8'))
    with (out/'tables/top100_complete_metrics.csv').open(encoding='utf-8') as f:top=list(csv.DictReader(f))
    W=174*mm
    def plot(name):
        img=Image(str(out/'figures'/f'{name}.png'))
        img.drawHeight*=W/img.drawWidth;img.drawWidth=W
        return img
    def footer(c,doc):
        c.saveState();c.setFont('Body',7.5);c.setFillColor(colors.HexColor('#65747B'))
        c.drawString(18*mm,12*mm,'GLOM-TGITO | Complete discovery evaluation | 14 October 2026')
        c.drawRightString(192*mm,12*mm,str(doc.page));c.restoreState()
    story=[P('Complete protein discovery:\nGLOM-TGITO and native baselines','title'),P('Computational results, publication figures and manuscript text','small'),plot('fig1_complete_discovery'),Spacer(1,7*mm),P('Table 1. Frozen native Top 100 comparison','h2')]
    rows=[]
    for r in top:rows.append([r['method_label'],f"{float(r['highcal_degree_mean']):.4f}",f"{float(r['cross_mean_mean']):.3f}",f"{float(r['terrain_risk_mean']):.4f}",f"{float(r['terrain_support_mean']):.4f}",f"{float(r['mean_pairwise_hamming']):.2f}",f"{int(r['outside_highcal_q95_n'])}/100"])
    story += [table(['Method','Mean deviation ↓','Predicted log-F ↑','Terrain risk ↓','Terrain support ↑','Diversity ↑','Tail count'],rows,[38*mm,25*mm,23*mm,22*mm,24*mm,22*mm,20*mm]),Spacer(1,4*mm),P('Mean deviation uses learned GLOM embeddings and a common high-prediction calibration denominator. Log-F is predicted log-fluorescence from two GLOM seeds unused during search. Diversity is mean pairwise residue difference. Tail counts are secondary. One completed campaign per method.','small'),PageBreak()]
    story += [P('Results','h1')]+paragraphs(data['results'][:3])+[Spacer(1,4*mm),P('Primary metric','h2'),P('D(x) = d32(x) / 0.5837557863;   batch mean = sum D(x) / batch size.','body'),P('d32 is the 32nd-training-neighbor distance in L2-normalized learned GLOM embeddings. The denominator is the calibration 95th percentile among sequences predicted at or above 4.2. The arithmetic mean includes every member of the frozen batch.','small'),PageBreak()]
    story += [P('Results continued','h1')]+paragraphs(data['results'][3:])+[P('Table 2. Complementary Top 100 evidence','h2')]
    rows=[]
    for r in top:rows.append([r['method_label'],f"{float(r['cross_sd_mean']):.4f}",f"{float(r['risk_scaled_lower90_mean']):.3f}",f"{float(r['nearest_visible_hamming_mean']):.2f}",str(int(r['novel_and_reliability_pass_n']))])
    story += [table(['Method','Cross-seed SD ↓','Empirical lower log-F','Nearest visible sequence','Joint count'],rows,[44*mm,31*mm,34*mm,37*mm,28*mm]),Spacer(1,3*mm),P('The joint count is an exploratory computational acceptance count using the documented novelty, lower-prediction, risk and stability thresholds. It is not an observed functional success count. The full threshold grid is supplied.','small'),PageBreak()]
    story += [P('Validation against measured fluorescence','h1'),plot('fig2_measured_validation'),Spacer(1,6*mm),P(data['captions'][1][0],'h2'),P(data['captions'][1][1],'small'),P('Why use the continuous score?','h2'),P('The distance score is associated with measured error across thousands of test sequences. Reporting its mean preserves how far the full batch lies from training support. A threshold count discards most of that information and becomes unstable when only one or two sequences cross the boundary.','body'),PageBreak()]
    story += [P('Methods','h1')]+paragraphs(data['methods'][:3])+[PageBreak()]
    story += [P('Methods continued','h1')]+paragraphs(data['methods'][3:])+[P('Literature supporting the evaluation','h2'),P('Learned nearest-neighbor distance follows Sun et al. (2022). Error-based validation and separate reporting of accuracy and uncertainty follow Greenman et al. (2025). The MaxWEnt head control adapts de Mathelin et al. (2025). The recent ProtHMSO search baseline adapts Wang et al. (2026).','body')]
    for j,ref in enumerate(data['references'],1):story.append(P(f'[{j}] {ref}','small'))
    story += [PageBreak(),P('Sensitivity and representation controls','h1'),plot('figS1_robustness_and_controls'),Spacer(1,5*mm),P(data['captions'][2][0],'h2'),P(data['captions'][2][1],'small'),PageBreak()]
    story += [P('Supplementary batch-size comparison','h1')]
    with (out/'tables/all_selected_batch_metrics.csv').open(encoding='utf-8') as f:allrows=list(csv.DictReader(f))
    rows=[]
    for b in [6,30,100]:
        for r in sorted([r for r in allrows if int(r['batch_size'])==b],key=lambda r:[t['method'] for t in top].index(r['method'])):
            rows.append([r['method_label'],str(b),f"{float(r['highcal_degree_mean']):.4f}",f"{float(r['cross_mean_mean']):.3f}",f"{float(r['terrain_risk_mean']):.4f}",f"{float(r['terrain_support_mean']):.4f}"])
    story += [table(['Method','Batch','Mean deviation ↓','Predicted log-F ↑','Risk ↓','Support ↑'],rows,[45*mm,15*mm,30*mm,30*mm,27*mm,27*mm]),Spacer(1,5*mm),P('Table S2. Common evaluator accuracy on 3,144 measured test sequences','h2')]
    with (out/'tables/common_evaluator_measured_accuracy.csv').open(encoding='utf-8') as f:judges=[r for r in csv.DictReader(f) if r['scope']=='all']
    story += [table(['Evaluator','RMSE','MAE','R-squared','Spearman'],[[r['judge'].replace('_',' '),*[f'{float(r[k]):.4f}' for k in ['rmse','mae','r2','spearman']]] for r in judges],[54*mm,30*mm,30*mm,30*mm,30*mm]),Spacer(1,4*mm),P(data['captions'][0][0],'h2'),P(data['captions'][0][1],'small'),P('Reproducibility','h2'),P('The accompanying bundle contains vector PDF/SVG figures, raster previews, manuscript LaTeX, machine-readable tables, selected candidate sequences, frozen source snapshots, analysis specifications and input SHA256 records. Previous experiment outputs and manuscript drafts are preserved.','small')]
    pdf=out/'full_discovery_results.pdf'
    doc=SimpleDocTemplate(str(pdf),pagesize=A4,rightMargin=18*mm,leftMargin=18*mm,topMargin=17*mm,bottomMargin=20*mm,title='Complete GLOM-TGITO discovery evaluation',author='',pageCompression=1)
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    reader=PdfReader(pdf)
    qa={'pages':len(reader.pages),'text_characters':[len(p.extract_text()) for p in reader.pages],'path':str(pdf)}
    assert min(qa['text_characters'])>100
    (out/'qa/pdf_text_check.json').write_text(json.dumps(qa,indent=2),encoding='utf-8')
    print(json.dumps(qa))

if __name__=='__main__':main()
