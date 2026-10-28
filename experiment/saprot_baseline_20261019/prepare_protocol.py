"""Freeze the new comparison without changing any earlier experiment."""
from pathlib import Path
import hashlib, json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def main():
    original = json.loads((ROOT / 'data/split_manifest.json').read_text())
    for name, expected in original['files'].items():
        assert sha(ROOT / 'data' / name) == expected, name
    protocol = {
        'created': '2026-10-19', 'experiment': HERE.name,
        'remote_root': '/xiongyuchen/gfp_evidence_20261013',
        'model_repo': 'westlake-repl/SaProt_650M_AF2',
        'model_revision': 'd9b9ad00ef61c0990e611b2b43f2231c7de24b38',
        'feature': 'saprot650_pred3di',
        'structural_input': 'Frozen existing ESM2-150M/Bridge argmax 3Di; checkpoint vocabulary; padding state maps to #',
        'bridge_sha256': '1dcc0d4031184d300c3fd1f8e0fd956626947fd81d71f65265c4474b5bcbb292',
        'splits': ['random', 'depth'], 'seeds': original['train_seed_list'],
        'data_hashes': original['files'],
        'partition_counts': original['partition_counts'],
        'depth_counts': original['extrapolation_counts'],
        'frozen_head': {'source': 'train_cached_heads.py', 'head': 'mlp',
                        'pooling': 'mean of residues, excluding specials',
                        'epochs_limit': 200, 'patience': 20, 'lr': .001},
        'lora': {'rank': 16, 'alpha': 32, 'dropout': .05,
                 'targets': ['query', 'value'], 'lr': .0001,
                 'epochs_limit': 25, 'patience': 6, 'effective_batch': 64,
                 'loss': 'training-standardized Huber',
                 'head': 'LayerNorm(1280), Linear(1280,256), GELU, Dropout(.1), Linear(256,1)'},
        'selection': 'validation Spearman, fixed in advance; no test feedback',
        'primary_metrics': ['rmse', 'mae', 'r2', 'spearman'],
        'secondary_metrics': ['pearson', 'measured_top6_30_100', 'high420', 'high435', 'top10pct_enrichment'],
        'baseline_scope': 'Native SaProt prediction baselines, not a TGITO ablation',
        'native_saprot_not_bridge_surrogate': True,
        'no_p4_or_glom_local_branches_in_saprot': True,
        'existing_results_unchanged': True,
        'source_hashes': {p.relative_to(ROOT).as_posix(): sha(p) for p in
                          [ROOT/'train_cached_heads.py', ROOT/'train_plm_lora.py',
                           ROOT/'experiment_core.py', ROOT/'encoder_contract.py']},
    }
    path = HERE / 'PROTOCOL.json'
    if path.exists():
        assert json.loads(path.read_text()) == protocol, 'Frozen protocol changed'
    else:
        path.write_text(json.dumps(protocol, indent=2)+'\n', encoding='utf8')
    print(json.dumps({'status': 'PREPARED', 'protocol_sha256': sha(path),
                      'gpu_runs_started': 0, 'split_hashes_verified': len(original['files'])}))

if __name__ == '__main__': main()
