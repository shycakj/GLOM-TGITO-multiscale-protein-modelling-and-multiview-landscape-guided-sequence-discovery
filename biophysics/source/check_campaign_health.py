"""Read-only simulation checks plus an atomic, downloadable result snapshot."""
import csv
import hashlib
import io
import json
import math
import statistics
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
status = json.loads((ROOT / 'CAMPAIGN_STATUS.json').read_text())
rows, issues = [], []
for path in sorted((ROOT / 'md').glob('*/rep[0-2]/DONE.json')):
    done = json.loads(path.read_text())
    sid, rep = done['system_id'], done['replicate']
    with (path.parent / 'production.csv').open() as handle:
        production = list(csv.DictReader(handle))
    metrics_path = ROOT / 'results' / sid / f'rep{rep}' / 'METRICS.json'
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else None
    temperatures = [float(r['Temperature (K)']) for r in production]
    numeric_finite = all(math.isfinite(float(v)) for r in production for v in r.values())
    checks = {
        'production_rows_300': len(production) == 300,
        'finite_production_values': numeric_finite,
        'production_duration_ns_3': done['production_ns'] == 3.,
        'no_production_restraints': done['production_restraints'] == 0,
        'temperature_mean_within_5K': abs(statistics.mean(temperatures)-300) < 5,
    }
    row = {'system_id': sid, 'replicate': rep, 'checks': checks,
           'temperature_mean_K': statistics.mean(temperatures),
           'wall_seconds': done['wall_seconds'], 'ns_per_day': done['ns_per_day'],
           'analysis_complete': metrics is not None}
    if metrics:
        checks['analysis_frames_300'] = metrics['frames'] == 300
        checks['finite_metrics'] = all(math.isfinite(v) for v in metrics['metrics'].values())
        row['metrics'] = metrics['metrics']
        row['split_half_diagnostic'] = metrics['split_half_diagnostic']
    for name, value in checks.items():
        if not value:
            issues.append({'system_id': sid, 'replicate': rep, 'check': name})
    rows.append(row)

wall = [r['wall_seconds'] for r in rows]
remaining = status['total'] - len(rows)
analysed = [r for r in rows if r['analysis_complete']]
ranges = {}
if analysed:
    for name in analysed[0]['metrics']:
        values = [r['metrics'][name] for r in analysed]
        ranges[name] = [min(values), max(values)]
report = {'checked_unix': time.time(), 'formal_done_count': len(rows),
          'formal_analysis_count': len(analysed), 'total': status['total'],
          'queue_failures': status['failed'], 'check_issues': issues,
          'running': status['running'], 'metric_ranges': ranges,
          'remaining_hours_at_observed_median': remaining * statistics.median(wall) / 2 / 3600 if wall else None,
          'remaining_hours_at_observed_mean': remaining * statistics.mean(wall) / 2 / 3600 if wall else None,
          'trajectories': rows,
          'scope': 'Numerical and short-trajectory structural checks; not an equilibrium convergence claim.'}
(ROOT / 'HEALTH_CHECK.json').write_text(json.dumps(report, indent=2))
manifest = {}
files = [p for p in (ROOT/'results').rglob('*') if p.is_file()]
files += [ROOT/'HEALTH_CHECK.json', ROOT/'CAMPAIGN_STATUS.json']
with tarfile.open(ROOT/'RESULTS_SNAPSHOT.tar.gz', 'w:gz') as tar:
    for path in sorted(files):
        data = path.read_bytes()
        name = path.relative_to(ROOT).as_posix()
        manifest[name] = hashlib.sha256(data).hexdigest()
        info = tar.gettarinfo(str(path), arcname=name)
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    data = json.dumps(manifest, indent=2).encode()
    info = tarfile.TarInfo('SNAPSHOT_MANIFEST.json')
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))
print(json.dumps({k:v for k,v in report.items() if k != 'trajectories'}, indent=2), flush=True)
print('Snapshot bytes:', (ROOT/'RESULTS_SNAPSHOT.tar.gz').stat().st_size, flush=True)
