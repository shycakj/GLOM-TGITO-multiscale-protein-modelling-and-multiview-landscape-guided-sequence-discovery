"""Save a compact live snapshot without reading environment variables or credentials."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, importlib.metadata, json, os, subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def main():
    queue = json.loads((HERE / 'QUEUE_STATUS.json').read_text())
    jobs = []
    for task in queue.get('running', []):
        proc = Path('/proc') / str(task['pid'])
        try:
            command = (proc / 'cmdline').read_bytes().decode().split('\0')
            expected = 'train_saprot_lora.py' if task['head'] == 'lora' else 'train_cached_heads.py'
            active = any(Path(part).name == expected for part in command)
        except FileNotFoundError:
            active = False
        log = ROOT / 'logs/saprot_baseline_20261019' / (task['split'] + '_' + task['name'] + '.log')
        lines = log.read_text(errors='replace').splitlines() if log.exists() else []
        jobs.append({**task, 'verified_training_process': active, 'recent_progress': lines[-2:]})
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,utilization.gpu',
                                   '--format=csv,noheader,nounits'], text=True).strip().splitlines()
    precision_file = HERE / 'PRECISION_AUDIT.json'
    status = {
        'verified_utc': datetime.now(timezone.utc).isoformat(),
        'remote_root': str(ROOT),
        'queue_status': queue['status'],
        'running': jobs,
        'completed': len(queue.get('completed', [])),
        'pending': len(queue.get('pending', [])),
        'failed': queue.get('failed', []),
        'gpu_csv_fields': ['index', 'name', 'total_MiB', 'used_MiB', 'utilization_percent'],
        'gpu_rows': [line.split(', ') for line in gpu],
        'precision_audit': json.loads(precision_file.read_text()) if precision_file.exists() else None,
        'versions': {p: importlib.metadata.version(p) for p in ['torch', 'transformers', 'peft', 'numpy', 'scikit-learn']},
        'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')},
        'protocol_sha256': hashlib.sha256((HERE / 'PROTOCOL.json').read_bytes()).hexdigest(),
    }
    (HERE / 'LIVE_STATUS.json').write_text(json.dumps(status, indent=2) + '\n')
    print(json.dumps({k: status[k] for k in ['verified_utc', 'queue_status', 'completed', 'pending', 'failed', 'gpu_rows']}))
    print('verified_training_processes', sum(j['verified_training_process'] for j in jobs))


if __name__ == '__main__':
    main()
