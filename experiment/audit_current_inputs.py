"""Read-only provenance and dataset audit before new experiments."""
from pathlib import Path
import csv
import hashlib
import json
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / 'audit'
OUT.mkdir(parents=True, exist_ok=True)
PROJECT = ROOT / 'handoff_20261001/repro_project'

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

ranked = rows(ROOT / 'LL_GF/results/v13_5_all_candidates_ranked_with_confidence.csv')
top6 = rows(ROOT / 'LL_GF/results/v13_5_final_top6_submission_candidates.csv')
seqkey = 'seq_clean' if 'seq_clean' in top6[0] else 'seq'
rankmap = {r[seqkey]: i + 1 for i, r in enumerate(ranked)}
wet = []
for r in top6:
    wet.append({k: r[k] for k in r if k in {
        seqkey, 'mutations', 'mutation_string', 'mutations_vs_reference',
        'mutations_vs_seed', 'confidence_score', 'pred', 'conf_lowrisk',
        'terrain_selected_risk_wmean', 'seed_seq', 'seed_sequence',
        'start_sequence', 'hamming_to_seed', 'hard_hamming_to_seed',
        'hamming_to_reference', 'nearest_31k_hamming', 'exact_in_31k'}} | {'rank': rankmap[r[seqkey]]})

files = {}
for tag, p in {
    'v13_5_wet_top6': ROOT / 'LL_GF/results/v13_5_final_top6_submission_candidates.csv',
    'v13_5_ranked': ROOT / 'LL_GF/results/v13_5_all_candidates_ranked_with_confidence.csv',
    'v13_5_predictor': ROOT / 'LL_GF/GFP_pplu/weights/best_model_pplu_31k.pt',
    'v4_predictor': PROJECT / 'artifacts/glom_current/best_model_pplu_current_split.pt',
    'bridge': PROJECT / 'assets/warm_best_model.pt',
    'raw_labels': PROJECT / 'data/raw/pplu_31480_sequences.csv',
}.items():
    files[tag] = {'path': str(p.relative_to(ROOT)), 'exists': p.is_file(),
                  **({'bytes': p.stat().st_size, 'sha256': sha(p)} if p.is_file() else {})}

comparisons = []
for name in ['tgito/revised_slim_model.py', 'tgito/optimizer.py', 'tgito/terrain_potential.py',
             'tgito/pplu_config.py', 'tgito/esm_bridge.py', 'run_generate.py']:
    paths = {
        'wet_version': ROOT / 'LL_GF/v13_working_code' / name,
        'complete_pipeline': ROOT / 'LL_GF/complete_pipeline_code' / name,
        'handoff': PROJECT / 'pipeline' / name,
    }
    hashes = {k: sha(p) if p.is_file() else None for k, p in paths.items()}
    comparisons.append({'file': name, 'sha256': hashes,
                        'wet_matches_handoff': bool(hashes['wet_version']) and hashes['wet_version'] == hashes['handoff']})

raw = rows(PROJECT / 'data/raw/pplu_31480_sequences.csv')
wt = (PROJECT / 'data/processed/reference_sequence.txt').read_text().strip()
unique = {}
duplicate_conflicts = []
for i, r in enumerate(raw):
    s = r['sequence'].strip()
    y = float(r['medianBrightness'])
    if s in unique and unique[s]['brightness'] != y:
        duplicate_conflicts.append({'row_id': i, 'retained_row_id': unique[s]['row_id']})
    unique.setdefault(s, {'row_id': i, 'brightness': y})
depth = Counter(sum(a != b for a, b in zip(s, wt)) for s in unique)
labels = sorted(r['brightness'] for r in unique.values())

saved = []
for p in sorted((PROJECT / 'saved_results').rglob('final_proposals.csv')):
    d = rows(p)
    cols = list(d[0]) if d else []
    saved.append({'path': str(p.relative_to(ROOT)), 'n': len(d),
                  'has_confidence_score': 'confidence_score' in cols,
                  'has_local_terrain_risk': 'terrain_selected_risk_wmean' in cols,
                  'has_selected_source_weights': any(c.startswith('selected_weight_') for c in cols)})

report = {
    'date': '2026-10-13', 'wet_campaign': 'v13.5',
    'wet_identity_source': 'User confirmation on 2026-10-13; not inferred from the later v4 runs.',
    'ranked_n': len(ranked), 'top6_n': len(top6), 'top6_ranks': [r['rank'] for r in wet],
    'exact_observed_n': sum(r.get('exact_in_31k', '').lower() == 'true' for r in ranked),
    'nearest_observed_hamming_counts': dict(Counter(r.get('nearest_31k_hamming') for r in ranked)),
    'risk_nonempty_n': sum(bool(r.get('terrain_selected_risk_wmean', '')) for r in ranked),
    'wet_candidates': wet, 'file_identity': files, 'source_comparison': comparisons,
    'dataset': {'raw_n': len(raw), 'unique_n': len(unique), 'duplicate_rows': len(raw) - len(unique),
                'duplicate_label_conflicts': duplicate_conflicts, 'mutation_depth_counts': dict(sorted(depth.items())),
                'min_label': labels[0], 'median_label': labels[len(labels)//2], 'max_label': labels[-1]},
    'later_v4_saved_runs': saved,
    'wet_measurements': 'Quantitative fluorescence/controls/replicates not yet located; user confirms all six fluoresced successfully.',
}
(OUT / 'current_version_and_data.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
print(json.dumps({k: report[k] for k in ['ranked_n','top6_n','top6_ranks','exact_observed_n','nearest_observed_hamming_counts','risk_nonempty_n','dataset','source_comparison','later_v4_saved_runs']}, ensure_ascii=True, indent=2))
