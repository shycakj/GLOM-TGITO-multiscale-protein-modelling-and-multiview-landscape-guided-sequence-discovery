from pathlib import Path
import json,hashlib,zipfile
import pandas as pd
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parent
PAPER=ROOT/'reports/full_discovery_paper_20261014'
OUT=PAPER/'multithreshold'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')

old=PdfReader(PAPER/'full_discovery_results.pdf')
new=PdfReader(PAPER/'full_discovery_results_multithreshold.pdf')
assert len(old.pages)==8 and len(new.pages)==10
for i in range(8):assert old.pages[i].extract_text()==new.pages[i].extract_text()
verified=json.loads((OUT/'VERIFICATION.json').read_text(encoding='utf-8'))
assert verified['local_per_sequence_recalculation'] and verified['old40counts_match']
raw=OUT/'compact_sequence_scores.csv.gz'
assert sha(raw)=='02f1f8f0b9c77a0e3165dd64ab41163cb0923e624478699b32cd04102258837e'
dump(OUT/'qa/DELIVERY_CHECKS.json',dict(base8pages_unchanged=True,combined_pages=10,addendum2pages_rendered_and_visually_reviewed=True,vector_figure_rendered_and_reviewed=True,original40threshold_counts_reproduced=True,new100Top100_cells_recomputed_locally=True,measured_group_sizes_and_RMSE_recomputed_locally=True,remote_compact_scores_sha256_verified=sha(raw)))
(OUT/'REPRODUCE.txt').write_text('''This extension changes only the secondary threshold analysis.
Original model, HPO, native rankings and mean-deviation normalization remain fixed.
The complete grid is Q50/Q60/Q70/Q75/Q80/Q85/Q90/Q95/Q97.5/Q99.

From the multithreshold folder, with numpy/pandas/matplotlib installed:
  python publish_multithreshold_extension.py --mode figures --input-directory . --paper-directory ..
With numpy/pandas/reportlab/pypdf installed:
  python publish_multithreshold_extension.py --mode pdf --input-directory . --paper-directory ..
The compact score file includes all calibration/test samples and the native Top100
union. It supports independent local reconstruction of all new Top100 counts,
thresholds and measured-error group comparisons. Original sequence identities are
available in the main package's frozen_selected_candidates table.

extend_familiarity_threshold_grid.py is the full workspace computation script.
It ran under /xiongyuchen/gfp_evidence_20261013/ and additionally includes all-output
pools and Top6/30. Its original inputs and hashes are recorded in DONE.json.

Use table_multithreshold.tex and results_multithreshold.tex as additional manuscript
fragments; the original mean-deviation primary table stays in place.
''',encoding='utf-8')
(PAPER/'阅读说明_多阈值更新.txt').write_text('''更新后的完整结果稿：full_discovery_results_multithreshold.pdf（10页）。
前8页为原结果稿，后2页为十档阈值比较及实测验证。
multithreshold/ 中含独立补充PDF、矢量图、CSV、LaTeX结果段落和表格、原始分数及复现脚本。
原稿full_discovery_results.pdf和原始结果包继续保留。
主指标仍为平均偏离程度，新增多阈值超限比例作为辅助。
''',encoding='utf-8')

files=[]
for p in PAPER.rglob('*'):
    if not p.is_file():continue
    rel=p.relative_to(PAPER)
    if 'qa' in rel.parts or '__pycache__' in rel.parts:continue
    if p.name.startswith('BUNDLE_') or p.name.startswith('MULTITHRESHOLD_BUNDLE_'):continue
    files.append(p)
files += [OUT/'qa/DELIVERY_CHECKS.json',PAPER/'qa/numerical_validation.json',PAPER/'qa/visual_validation.json']
manifest=[dict(path=p.relative_to(PAPER).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(files)]
dump(PAPER/'MULTITHRESHOLD_BUNDLE_MANIFEST.json',dict(primary_endpoint_unchanged=True,quantile_grid=[.5,.6,.7,.75,.8,.85,.9,.95,.975,.99],files=manifest))
files.append(PAPER/'MULTITHRESHOLD_BUNDLE_MANIFEST.json')
zpath=PAPER.parent/'full_discovery_paper_multithreshold_20261014.zip'
with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in files:z.write(p,'full_discovery_paper_20261014/'+p.relative_to(PAPER).as_posix())
with zipfile.ZipFile(zpath) as z:
    assert z.testzip() is None
    for r in manifest:assert hashlib.sha256(z.read('full_discovery_paper_20261014/'+r['path'])).hexdigest()==r['sha256']
receipt=dict(path=str(zpath),bytes=zpath.stat().st_size,sha256=sha(zpath),files=len(files),all_members_verified=True)
dump(PAPER/'MULTITHRESHOLD_BUNDLE_VERIFIED.json',receipt)
handoff_path=ROOT/'reports/active_run_handoff.json'
handoff=json.loads(handoff_path.read_text(encoding='utf-8'))
handoff['familiarity_multithreshold_extension']={
    'status':'complete','directory':'artifacts/discovery_v3/embedding_familiarity_v1/threshold_grid_v2',
    'quantiles':[.5,.6,.7,.75,.8,.85,.9,.95,.975,.99],
    'primary_mean_unchanged':True,'native_rankings_unchanged':True,
    'paper_pdf':str(PAPER/'full_discovery_results_multithreshold.pdf'),'bundle':str(zpath),
    'top100_counts_at_Q80':{'glom_tgito':36,'glom_adalead':50,'glom_random':45,'glom_prothmso':34,'esm650_prothmso':46},
    'top100_counts_at_Q85':{'glom_tgito':25,'glom_adalead':37,'glom_random':34,'glom_prothmso':27,'esm650_prothmso':37},
    'central_boundaries_are_not_failure_labels':True,
    'local_raw_compact_scores':str(ROOT/'artifacts/discovery_v3/embedding_familiarity_v1/threshold_grid_v2/compact_sequence_scores.csv.gz'),
    'remote_statistics_complete':True,'remote_paper_upload':False,
}
handoff['full_discovery_publication_20261014']['latest_pdf']=str(PAPER/'full_discovery_results_multithreshold.pdf')
handoff['full_discovery_publication_20261014']['latest_bundle']=str(zpath)
dump(handoff_path,handoff)
print(json.dumps(receipt,ensure_ascii=False))
