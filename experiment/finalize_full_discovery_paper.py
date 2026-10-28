from pathlib import Path
import hashlib, json, re, shutil, zipfile

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'reports/full_discovery_paper_20261014'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,o):p.write_text(json.dumps(o,ensure_ascii=False,indent=2),encoding='utf-8')

for name in ['build_full_discovery_paper.py','render_paper_pdf.py']:
    shutil.copy2(ROOT/name,OUT/name)

# Final source consistency and a transparent record of the compilation scope.
for r in json.loads((OUT/'source_manifest.json').read_text(encoding='utf-8')):
    assert sha(OUT/r['snapshot'])==r['sha256'],r['snapshot']
assert sha(ROOT/'SOURCE_MANIFEST.json')=='97ce3e291423d6790a9d72c5b375141a75e0fbc95eb5139168fd871ef2576945'
text_checks=[]
for p in (OUT/'latex').glob('*.tex'):
    s=p.read_text(encoding='utf-8')
    level=0
    for bracket in re.finditer(r'(?<!\\)[{}]',s):
        level+=1 if bracket.group()=='{' else -1
        assert level>=0,p
    assert level==0,p
    for name in re.findall(r'\\input\{([^}]+)\}',s):assert (p.parent/(name+'.tex')).exists(),name
    for name in re.findall(r'\\includegraphics(?:\[[^]]+\])?\{([^}]+)\}',s):assert (OUT/'figures'/name).exists(),name
    text_checks.append(p.name)

(OUT/'BUILD_NOTES.txt').write_text('''Rebuilding the scientific figures and tables

Python packages used for numerical analysis: numpy 2.1.3, pandas 2.3.3, matplotlib 3.10.0.
The PDF renderer requires reportlab and pypdf; Arial fonts were loaded from C:/Windows/Fonts.
From this package directory:
  python build_full_discovery_paper.py --snapshot --output .
  python render_paper_pdf.py --output .
The source_data snapshots are checked against source_manifest.json before rebuilding.
The package reproduces the evidence figures/tables, not the expensive model-training/search jobs.

LaTeX integration
The latex directory contains manuscript fragments, equations, four tables, captions,
verified bibliographic metadata and a standalone main.tex. It references ../figures.
Build in the latex directory with pdflatex, bibtex, pdflatex, pdflatex in an installed
TeX environment. TeX structure and input/figure references were checked locally.
No usable local LaTeX compiler was present, so the standalone TeX compilation was
not run. The delivered full_discovery_results.pdf was produced with ReportLab and
all eight pages were rendered and visually inspected. The three figure PDFs were
exported directly from Matplotlib as vector graphics.

Statistical scope
The mean embedding endpoint is retrospective and descriptive. The native batches
were frozen before its computation. Candidate means are not independent campaign
replicates. Threshold bootstrap intervals describe calibration uncertainty only.
Measured fluorescence remains missing for generated candidates without assays.
''',encoding='utf-8')
dump(OUT/'qa/visual_validation.json',dict(pdf_pages_reviewed=8,figure_groups_reviewed=['fig1_complete_discovery','fig2_measured_validation','figS1_robustness_and_controls'],final_changed_pages_reviewed=[2,6,8],layout_issues_remaining=0,tex_structure_checks=text_checks,tex_compiled=False,pdf_engine='ReportLab',figures_vector_pdf_svg=True))

# Deliver publication material and verification records, excluding render previews
# and the safely quarantined, misrouted intermediate build.
files=[]
for p in OUT.rglob('*'):
    if not p.is_file():continue
    rel=p.relative_to(OUT)
    if '__pycache__' in rel.parts or 'qa' in rel.parts:continue
    if p.name in ['BUNDLE_MANIFEST.json','BUNDLE_VERIFIED.json']:continue
    files.append(p)
for name in ['numerical_validation.json','visual_validation.json','pdf_text_check.json']:
    files.append(OUT/'qa'/name)
manifest=[dict(path=p.relative_to(OUT).as_posix(),size=p.stat().st_size,sha256=sha(p)) for p in sorted(files)]
dump(OUT/'BUNDLE_MANIFEST.json',dict(files=manifest,primary_endpoint='mean GLOM embedding deviation',native_batch_selection_preserved=True,unknown_fluorescence_imputed=False))
files.append(OUT/'BUNDLE_MANIFEST.json')
archive=OUT.parent/'full_discovery_paper_20261014.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in files:z.write(p,arcname='full_discovery_paper_20261014/'+p.relative_to(OUT).as_posix())
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    for row in manifest:
        b=z.read('full_discovery_paper_20261014/'+row['path'])
        assert len(b)==row['size'] and hashlib.sha256(b).hexdigest()==row['sha256']
verification=dict(archive=str(archive),sha256=sha(archive),bytes=archive.stat().st_size,files=len(files),all_members_verified=True)
dump(OUT/'BUNDLE_VERIFIED.json',verification)

handoff_path=ROOT/'reports/active_run_handoff.json'
handoff=json.loads(handoff_path.read_text(encoding='utf-8'))
handoff['full_discovery_publication_20261014']={
    'status':'complete_local_delivery',
    'directory':str(OUT),'bundle':str(archive),
    'primary_endpoint':'Mean normalized GLOM z768 training-neighbor distance, k=32, high-prediction calibration q95 denominator; native Top100',
    'endpoint_choice':'User requested after results; retrospective descriptive, no reranking',
    'secondary_endpoint':'Exceedance fractions with quantile/k sensitivity and calibration bootstrap; no ranking from 1 vs 2 events',
    'top100_primary_values':{'glom_tgito':0.8048374926,'glom_adalead':0.8289993812,'glom_random':0.8322721636,'glom_prothmso':0.8309699738,'esm650_prothmso':0.8763434175},
    'supporting_evidence':['predicted log-fluorescence','terrain risk and support','cross-seed stability','catalogue novelty and pairwise diversity','measured-error AUROC/RMSE and common-evaluator R2/Spearman'],
    'controls_retained':['all unique output pools','raw ESM150 distance','Top6/30/100','neighborhood sizes 8/16/32/64','threshold quantiles 90/95/97.5/99%'],
    'figures':3,'pdf_pages':8,'native_outputs_changed':False,'wetlab_values_available':False,
    'tex_compilation':'structure checked; no usable local compiler; report PDF rendered independently',
    'remote_upload_performed':False,
}
dump(handoff_path,handoff)
print(json.dumps(verification,ensure_ascii=False))
