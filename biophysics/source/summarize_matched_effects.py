"""Descriptive candidate/reference effects using only common completed MD seeds."""
import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results'
with (ROOT/'inputs/candidate_start_pairs.csv').open(newline='', encoding='utf-8-sig') as handle:
    pairs = list(csv.DictReader(handle))
records = {}
for path in OUT.glob('*/rep[0-2]/METRICS.json'):
    row = json.loads(path.read_text())
    records.setdefault(row['system_id'], {})[row['replicate']] = row['metrics']
details = []
for pair in pairs:
    candidate = records.get(pair['candidate_id'], {})
    for comparison, sid in [('WT', 'WT'), ('start', pair['start_system_id'])]:
        reference = records.get(sid, {})
        common = sorted(set(candidate) & set(reference))
        if not common:
            continue
        for metric in candidate[common[0]]:
            values = [candidate[rep][metric] for rep in common]
            refs = [reference[rep][metric] for rep in common]
            deltas = [v-r for v,r in zip(values, refs)]
            row = {'candidate_id': pair['candidate_id'], 'rank': int(pair['rank']),
                   'comparison': comparison, 'reference_id': sid, 'metric': metric,
                   'common_replicates': len(common), 'candidate_mean': statistics.mean(values),
                   'reference_mean': statistics.mean(refs), 'delta_mean': statistics.mean(deltas),
                   'delta_sd': statistics.stdev(deltas) if len(common)>1 else None,
                   'mutations_vs_wt': pair['mutations_vs_wt'],
                   'mutations_vs_start': pair['mutations_vs_start']}
            details.append(row)
with (OUT/'matched_md_effects.csv').open('w', newline='', encoding='utf-8') as handle:
    writer = csv.DictWriter(handle, fieldnames=list(details[0]))
    writer.writeheader(); writer.writerows(details)
summary = {}
for comparison in ['WT', 'start']:
    subset = [r for r in details if r['comparison']==comparison]
    summary[comparison] = {}
    for metric in sorted({r['metric'] for r in subset}):
        rows = [r for r in subset if r['metric']==metric]
        summary[comparison][metric] = {
            'candidate_count':len(rows), 'common_replicate_counts':sorted({r['common_replicates'] for r in rows}),
            'lower_count':sum(r['delta_mean']<0 for r in rows),
            'higher_count':sum(r['delta_mean']>0 for r in rows),
            'candidate_mean':statistics.mean(r['candidate_mean'] for r in rows),
            'reference_mean':statistics.mean(r['reference_mean'] for r in rows),
            'mean_delta':statistics.mean(r['delta_mean'] for r in rows),
            'median_delta':statistics.median(r['delta_mean'] for r in rows)}
result = {'description':'Interim descriptive results for completed candidates only; directions are not universal efficacy labels.',
          'comparisons':summary, 'candidate_ids':sorted({r['candidate_id'] for r in details})}
(OUT/'MATCHED_EFFECTS_SUMMARY.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
