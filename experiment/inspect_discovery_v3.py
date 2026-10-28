"""Read-only campaign inspection; write a timestamped status outside frozen inputs."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import json
import subprocess

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'artifacts/discovery_v3'


def records(folder):
    return [json.loads(p.read_text()) for p in sorted(folder.glob('*/DONE.json'))]


def main():
    status = {'snapshot_utc': datetime.now(timezone.utc).isoformat(),
              'remote_root': str(ROOT), 'pilot': {}, 'formal': {}}
    for name in ['QUEUE_STATUS', 'PREFLIGHT_PASSED', 'QUEUE_ERROR', 'CAMPAIGN_DONE']:
        p = OUT / (name + '.json')
        status[name] = json.loads(p.read_text()) if p.exists() else None
    for phase in ['pilot', 'formal']:
        for folder in sorted((OUT / phase).iterdir()):
            if not folder.is_dir():
                continue
            rows = records(folder)
            status[phase][folder.name] = {
                'completed': len(rows), 'valid': sum(bool(x['valid']) for x in rows),
                'gate_topk': sorted({x['gate_topk'] for x in rows if 'gate_topk' in x}),
                'dynamic_weights_changed': sum(x.get('dynamic_weight_range', 0) > 1e-7 for x in rows),
                'total_forward_sequences': sum(x['forward_sequences'] for x in rows),
                'backward_steps': sum(x.get('backward_steps', 0) for x in rows),
                'max_reserved_gib': max((x.get('peak_gpu_gib', 0) for x in rows), default=0),
                'protocol_hashes': dict(Counter(x['protocol_sha256'] for x in rows)),
            }
    status['gpu'] = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total',
         '--format=csv,noheader'], text=True).strip().splitlines()
    p = ROOT / 'audit/v3_terrain_gradient_contract.json'
    status['terrain_gradient_audit'] = json.loads(p.read_text()) if p.exists() else None
    dest = ROOT / 'reports/persistent_campaign_last_verified.json'
    dest.write_text(json.dumps(status, indent=2))
    print(json.dumps(status), flush=True)


if __name__ == '__main__':
    main()
