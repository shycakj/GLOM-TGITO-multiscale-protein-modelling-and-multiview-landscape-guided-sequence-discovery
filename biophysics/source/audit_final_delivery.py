"""Certify that the persistent final archive contains all 56 x 3 MD results."""
import csv
import hashlib
import json
import math
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def rows(path):
    with path.open(encoding='utf-8-sig',newline='') as handle:
        return list(csv.DictReader(handle))

def run():
    if (ROOT/'DELIVERY_COMPLETE.json').exists():
        print('Delivery already audited.',flush=True)
        return
    assert str(ROOT).startswith('/xiongyuchen/gfp_evidence_20261013/')
    deadline=time.time()+36*3600
    while not (ROOT/'FINAL_EXPORT_READY.json').exists():
        if time.time()>deadline:
            raise TimeoutError('Final archive not ready; completion has not been certified.')
        time.sleep(30)
    complete=json.loads((ROOT/'CAMPAIGN_COMPLETE.json').read_text())
    assert complete['complete'] and complete['trajectories']==168
    systems=rows(ROOT/'inputs/systems.csv')
    assert len(systems)==56 and len({r['system_id'] for r in systems})==56
    required=[]
    for system in systems:
        sid=system['system_id']
        for rep in range(3):
            md=ROOT/'md'/sid/f'rep{rep}'
            analysis=ROOT/'results'/sid/f'rep{rep}'
            done=json.loads((md/'DONE.json').read_text())
            metrics=json.loads((analysis/'METRICS.json').read_text())
            assert done['system_id']==sid and done['replicate']==rep
            assert done['production_ns']==3 and done['production_restraints']==0
            assert metrics['system_id']==sid and metrics['replicate']==rep
            assert metrics['production_ns']==3 and metrics['frames']==300
            assert all(math.isfinite(v) for v in metrics['metrics'].values())
            for path in [md/'DONE.json',md/'solute.dcd',md/'production.csv',
                         md/'equilibration.csv',analysis/'METRICS.json',analysis/'per_residue.csv']:
                assert path.is_file() and path.stat().st_size>0,str(path)
                required.append(path.relative_to(ROOT).as_posix())
    table=rows(ROOT/'results/md_metrics_by_replicate.csv')
    assert len(table)==168
    assert len({(r['system_id'],int(r['replicate'])) for r in table})==168
    candidates=rows(ROOT/'results/top30_biophysics_comparison.csv')
    assert len(candidates)==30
    assert all(r['md_status']=='complete_3_replicates' for r in candidates)
    ready=json.loads((ROOT/'FINAL_EXPORT_READY.json').read_text())
    archive=ROOT/'FINAL_RESULTS_AND_TRAJECTORIES.tar.gz'
    assert archive.stat().st_size==ready['archive_bytes']
    digest=hashlib.sha256()
    with archive.open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b''):
            digest.update(block)
    assert digest.hexdigest()==ready['sha256']
    with tarfile.open(archive) as tar:
        manifest=json.load(tar.extractfile('FINAL_EXPORT_MANIFEST.json'))
        assert all(name in manifest for name in required)
        for name in required:
            assert tar.getmember(name).size>0,name
    report={'complete':True,'checked_utc':datetime.now(timezone.utc).isoformat(),
            'persistent_root':str(ROOT),'systems':56,'replicates_per_system':3,
            'completed_trajectories':168,'analysed_trajectories':168,'candidate_comparisons':30,
            'production_ns_per_trajectory':3,'all_300_frames_and_finite':True,
            'required_files_present_in_archive':len(required),'archive_files':len(manifest),
            'archive_bytes':archive.stat().st_size,'archive_sha256':digest.hexdigest(),
            'archive':str(archive),'result_table':str(ROOT/'results/top30_biophysics_comparison.csv'),
            'raw_trajectory_directory':str(ROOT/'md'),'source_directory':str(ROOT/'source'),
            'prepared_system_directory':str(ROOT/'structures'),
            'archive_integrity':'Full archive SHA256 matches export record; required trajectory and analysis members present. Per-file SHA256 manifest included.'}
    temp=ROOT/'DELIVERY_COMPLETE.tmp'
    temp.write_text(json.dumps(report,indent=2))
    temp.replace(ROOT/'DELIVERY_COMPLETE.json')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
    try:
        run()
    except Exception as exc:
        (ROOT/'DELIVERY_AUDIT_FAILED.json').write_text(json.dumps({'error_type':type(exc).__name__,'message':str(exc),'utc':datetime.now(timezone.utc).isoformat()},indent=2))
        raise
