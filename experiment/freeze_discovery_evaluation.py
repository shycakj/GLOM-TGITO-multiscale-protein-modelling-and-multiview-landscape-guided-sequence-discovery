"""Freeze complete, paired discovery batches before independent evaluation."""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import pandas as pd
from evaluate_risk_screening import confidence
from evaluate_submission_policy import choose
from experiment_core import write_json

METHODS = ['tgito_risk', 'tgito_no_risk_gate', 'tgito_no_terrain',
           'tgito_single_view', 'random', 'adalead']
RECENT = 'prothmso_top3_2026'
COLS = ['sequence', 'pred', 'seed_row', 'random_seed', 'method', 'hamming_from_seed',
        'terrain', 'terrain_selected_support_wmean', 'terrain_selected_risk_wmean',
        'terrain_selected_pred_gap_wmean', 'terrain_weight_entropy']

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def freeze(root, cohort, max_blocks=0, recent=False):
    base = root/'artifacts/models/random/glom_seed20261013'
    out = base/'discovery_evaluation'/cohort
    out.mkdir(parents=True, exist_ok=True)
    if (out/'FROZEN.json').exists():
        manifest = json.loads((out/'FROZEN.json').read_text())
        for name, digest in manifest['frozen_files'].items():
            assert sha(out/name) == digest
        return out
    methods = METHODS + ([RECENT] if recent else [])
    starts = pd.read_csv(base/'search/starting_sequences.csv')
    frames, budgets, sources, blocks = [], [], {}, []
    for start in starts.itertuples():
        for seed in [20261013, 20261014, 20261015]:
            folders = {m: base/('search_recent' if m == RECENT else 'search')/
                       f'start{int(start.row_id)}_seed{seed}_{m}' for m in methods}
            if not all((p/'DONE.json').exists() for p in folders.values()):
                continue
            blocks.append((int(start.row_id), seed))
            for method, folder in folders.items():
                d = pd.read_csv(folder/'candidate_pool.csv', usecols=COLS)
                assert set(d.method) <= {method}
                assert set(d.random_seed) <= {seed}
                assert set(d.seed_row) <= {int(start.row_id)}
                assert d.hamming_from_seed.between(1, 2).all()
                actual=d.sequence.map(lambda s:sum(x!=y for x,y in zip(s,start.sequence)))
                assert np.array_equal(actual.to_numpy(),d.hamming_from_seed.to_numpy())
                frames.append(d)
                budgets.append(json.loads((folder/'DONE.json').read_text()))
                sources[str(folder.relative_to(root))] = sha(folder/'candidate_pool.csv')
            if max_blocks and len(blocks) >= max_blocks:
                break
        if max_blocks and len(blocks) >= max_blocks:
            break
    if not max_blocks and len(blocks) != 36:
        raise RuntimeError(f'Waiting for all 36 paired blocks: {len(blocks)} complete')
    if not frames:
        raise RuntimeError('No complete paired blocks')
    pools = pd.concat(frames, ignore_index=True)
    known = set()
    for role in ['train', 'validation', 'calibration']:
        known.update(pd.read_csv(root/f'data/{role}.csv', usecols=['sequence']).sequence)
    assert not set(pools.sequence).intersection(known)
    assert pools.sequence.str.fullmatch('[ACDEFGHIKLMNPQRSTVWY]{221}').all()
    selected, summary = [], []
    groups = [('start_seed', f'{int(s)}_{int(r)}', d)
              for (s, r), d in pools.groupby(['seed_row', 'random_seed'], sort=True)]
    if len(blocks) == 36:
        groups += [('campaign_seed', str(int(r)), d) for r, d in pools.groupby('random_seed')]
        groups += [('campaign_all', 'all', pools)]
    for scope, replicate, frame in groups:
        # Every method shares the same unlabeled normalization pool within a paired replicate.
        frame = confidence(frame)
        for method, d in frame.groupby('method', sort=True):
            d = d.sort_values(['confidence_score', 'sequence'], ascending=[False, True],
                              kind='stable').drop_duplicates('sequence').reset_index(drop=True)
            policies = [('final6_confidence', 'confidence_score', 6, True),
                        ('final6_prediction', 'pred', 6, True),
                        ('final6_without_final_risk', 'confidence_without_final_risk', 6, True),
                        ('screen30_prediction', 'pred', 30, False)]
            for policy, ranking, k, final_rule in policies:
                if final_rule:
                    ids = choose(d.sequence.to_numpy(), d.pred.to_numpy(), d[ranking].to_numpy(), k=k)
                else:
                    ids = np.argsort(-d[ranking].to_numpy(), kind='stable')[:k]
                x = d.iloc[ids].copy()
                x['scope'], x['replicate'], x['policy'] = scope, replicate, policy
                x['rank'] = np.arange(1, len(x)+1)
                selected.append(x)
                summary.append(dict(scope=scope, replicate=replicate, method=method, policy=policy,
                                    requested=k, selected_n=len(x), full_batch=len(x)==k,
                                    pool_n=len(d), eligible_n=int(d.pred.ge(4.31).sum())))
    s = pd.concat(selected, ignore_index=True)
    s.to_csv(out/'frozen_selected.csv', index=False)
    pd.DataFrame(summary).to_csv(out/'frozen_selection_counts.csv', index=False)
    # Include training starts so independent models assess improvements in their own units.
    seqs = sorted(set(s.sequence).union(starts.sequence))
    pd.DataFrame({'sequence': seqs}).to_csv(out/'evaluator_sequences.csv', index=False)
    pools.to_csv(out/'frozen_candidate_pools.csv', index=False)
    pd.DataFrame([{k:v for k,v in r.items() if not isinstance(v, dict)} for r in budgets]).to_csv(
        out/'query_budgets.csv', index=False)
    files = ['frozen_selected.csv','frozen_selection_counts.csv','evaluator_sequences.csv',
             'frozen_candidate_pools.csv','query_budgets.csv']
    write_json(out/'FROZEN.json', dict(cohort=cohort, complete_campaign=len(blocks)==36,
        paired_blocks=len(blocks), methods=methods, frozen_files={f:sha(out/f) for f in files},
        source_pools_sha256=sources, risk_config_sha256=sha(base/'risk_calibration/FROZEN_CONFIG.json'),
        split_manifest_sha256=sha(root/'data/split_manifest.json'),
        primary_analysis='campaign_seed, final6_confidence; same final policy for all search methods',
        secondary_analysis='start_seed diagnostics, campaign_all operational list, ranking ablations, screen30',
        truth_read_before_freezing=False, external_judges_read_before_freezing=False,
        no_external_feedback_to_search=True, prediction_floor=4.31, pairwise_hamming=2,
        budget_note='Search forward budget matched; gradient, MLM proposal, and rescoring costs reported separately',
        uncertainty_note='Three seeds share twelve starts; per-start repeats are clustered, not 36 independent proteins'))
    print(json.dumps({'frozen':str(out),'blocks':len(blocks),'selected_rows':len(s),'unique_evaluator_sequences':len(seqs)}))
    return out

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--cohort',default='full');p.add_argument('--max-blocks',type=int,default=0)
    p.add_argument('--recent',action='store_true');a=p.parse_args()
    freeze(a.root,a.cohort,a.max_blocks,a.recent)
