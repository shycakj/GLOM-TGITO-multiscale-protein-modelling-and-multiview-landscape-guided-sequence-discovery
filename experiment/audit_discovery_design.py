"""Read-only audit of existing discovery outputs; never maps unknown fitness to zero.

Produces diagnostics, not a new biological success score. Test labels are not read.
"""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import pandas as pd


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def distances(seqs, targets, wt):
    t = np.array([list(s.encode('ascii')) for s in targets], dtype=np.uint8)
    w = np.frombuffer(wt.encode('ascii'), dtype=np.uint8)
    depth = (t != w).sum(1)
    result = np.empty((len(seqs), len(targets)), dtype=np.int16)
    for i, seq in enumerate(seqs):
        q = np.frombuffer(seq.encode('ascii'), dtype=np.uint8)
        pos = np.flatnonzero(q != w)
        result[i] = depth + len(pos) - (t[:, pos] != w[pos]).sum(1) - (t[:, pos] == q[pos]).sum(1)
    return result


def main(root, mountains):
    base = root/'artifacts/models/random/glom_seed20261013'
    src = base/'wide_evaluation'/f'prefix_{mountains:03d}'
    out = root/'audit/discovery_design_review'/f'prefix_{mountains:03d}'
    out.mkdir(parents=True, exist_ok=True)
    end = pd.read_csv(src/'frozen_endpoints.csv')
    starts = pd.read_csv(src/'frozen_starts.csv').set_index('start_id')
    budget = pd.read_csv(src/'query_budgets.csv')
    selected = pd.read_csv(src/'frozen_selected.csv')
    known = pd.concat([pd.read_csv(root/f'data/{p}.csv', usecols=['sequence'])
                       for p in ['train', 'validation', 'calibration']]).sequence.unique()
    wt = pd.read_csv(root/'data/train.csv').query('mutation_depth == 0').iloc[0].sequence
    allseq = sorted(set(end.sequence))
    nearest = distances(allseq, known, wt).min(1)
    # Validate the optimized exact distance calculation independently.
    for i, seq in enumerate(allseq[:3]):
        assert nearest[i] == min(sum(a != b for a, b in zip(seq, s)) for s in known)
    lookup = dict(zip(allseq, nearest))
    end['nearest_known_distance'] = end.sequence.map(lookup)
    end['wt_distance'] = end.sequence.map(lambda s: sum(a != b for a, b in zip(s, wt)))
    assert end.nearest_known_distance.ge(1).all()
    rows = []
    for method, d in end.groupby('method'):
        b = budget[budget.method.eq(method)]
        u = d.drop_duplicates('sequence')
        pair = distances(u.sequence.tolist(), u.sequence.tolist(), wt)
        diversity = pair[np.triu_indices(len(u), 1)]
        row = dict(method=method, contexts=len(d), unique_endpoints=len(u),
                   pred_mean=float(d.pred.mean()), pred_median=float(d.pred.median()),
                   pred_ge431_fraction=float(d.pred.ge(4.31).mean()),
                   terrain_mean=float(d.terrain.mean()),
                   support_mean=float(d.terrain_selected_support_wmean.mean()),
                   risk_mean=float(d.terrain_selected_risk_wmean.mean()),
                   seed_distance_mean=float(d.hamming_from_seed.mean()),
                   wt_distance_mean=float(d.wt_distance.mean()),
                   nearest_known_mean=float(d.nearest_known_distance.mean()),
                   nearest_known_ge2_fraction=float(d.nearest_known_distance.ge(2).mean()),
                   pairwise_distance_mean=float(diversity.mean()),
                   fitness_forwards=int(b.forward_sequences.sum()),
                   gradient_forwards=int(b.gradient_forward_sequences.sum()),
                   within_context_unique_hard_total=int(b.valid_unique_candidates.sum()),
                   common_rescoring=int(b.common_rescoring_sequences.sum()),
                   search_seconds=float(b.seconds.sum()))
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(out/'generated_candidate_diagnostics.csv', index=False)
    end.to_csv(out/'endpoints_with_novelty.csv', index=False)
    selected['nearest_known_distance'] = selected.sequence.map(lookup)
    selected['wt_distance'] = selected.sequence.map(lambda s: sum(a != b for a, b in zip(s, wt)))
    sel = selected.groupby(['method', 'policy', 'k']).agg(
        selected_n=('sequence', 'size'), pred_mean=('pred', 'mean'), pred_min=('pred', 'min'),
        risk_mean=('terrain_selected_risk_wmean', 'mean'),
        support_mean=('terrain_selected_support_wmean', 'mean'),
        nearest_known_mean=('nearest_known_distance', 'mean'), wt_distance_mean=('wt_distance', 'mean'))
    sel.to_csv(out/'final_batch_diagnostics.csv')
    # Paired descriptive effects; repeated anchors are collapsed for bootstrap.
    paired = []
    rng = np.random.default_rng(20261013)
    for baseline in sorted(set(end.method)-{'tgito_risk'}):
        for metric in ['pred', 'terrain_selected_risk_wmean', 'nearest_known_distance', 'hamming_from_seed']:
            p = end.pivot(index='start_id', columns='method', values=metric)
            delta = (p.tgito_risk-p[baseline]).dropna()
            groups = pd.DataFrame({'delta': delta, 'seed_row': starts.loc[delta.index, 'row_id']})
            vals = groups.groupby('seed_row').delta.mean().to_numpy()
            boots = vals[rng.integers(len(vals), size=(2000, len(vals)))].mean(1)
            paired.append(dict(baseline=baseline, metric=metric, context_n=len(delta),
                distinct_anchor_n=len(vals), context_mean_difference=float(delta.mean()),
                equal_anchor_mean_difference=float(vals.mean()),
                anchor_bootstrap_low=float(np.quantile(boots, .025)),
                anchor_bootstrap_high=float(np.quantile(boots, .975)),
                meaning='TGITO minus baseline; surrogate/novelty diagnostic, not measured fluorescence'))
    pd.DataFrame(paired).to_csv(out/'paired_surrogate_diagnostics.csv', index=False)
    histories=[];source_weights=[];return_checks=[]
    for folder in sorted((base/'wide_search').glob('context*_tgito_risk')):
        start_id=int(folder.name.split('_')[0].removeprefix('context'))
        if start_id not in starts.index or not (folder/'DONE.json').exists(): continue
        hp=folder/'history_restart0.csv'
        if not hp.exists(): continue
        h=pd.read_csv(hp)
        native=pd.read_csv(folder/'native_final.csv').iloc[0]
        pool=pd.read_csv(folder/'candidate_pool.csv')
        cfg=json.loads((folder/'DONE.json').read_text())['config']
        anchor=float(h.anchor_pred_brightness.iloc[0]);floor=anchor-cfg['anchor_drop']
        eligible=pool[pool.pred.ge(floor)].copy()
        assert cfg['embedding_realism_weight']==cfg['embedding_repair_weight']==0
        eligible['hard_objective']=cfg['brightness_weight']*eligible.pred+cfg['terrain_weight']*eligible.terrain
        final=pool[pool.sequence.eq(native.final_seq)]
        returned_score=float(final.pred.iloc[0]+cfg['terrain_weight']*final.terrain.iloc[0]) if len(final) else None
        best=eligible.sort_values(['hard_objective','sequence'],ascending=[False,True]).head(1)
        return_checks.append(dict(start_id=start_id, valid_native=bool(native.valid_output),
            native_reason=native.get('invalid_reason',''), known_recorded_states=int(h.is_known.sum()),
            recorded_states=len(h), unique_valid_pool=len(pool), floor_eligible_pool=len(eligible),
            final_hard_objective=returned_score,
            best_pool_hard_objective=float(best.hard_objective.iloc[0]) if len(best) else None,
            pool_best_pred=float(pool.pred.max()) if len(pool) else None,
            native_pred=float(native.final_pred_brightness),
            valid_but_discarded_pool=bool(not native.valid_output and len(eligible)),
            best_pool_hard_objective_gap=float(best.hard_objective.iloc[0]-returned_score) if len(best) and returned_score is not None else None))
        weights=[c for c in h if c.startswith('selected_weight_')]
        active=(h[weights].to_numpy()>0).astype(int)
        histories.append(dict(folder=folder.name, recorded_states=len(h), unique_hard_states=h.seq.nunique(),
            accepted_proposals=int(h.proposal_accepted.sum()),
            initial_pred=float(h.hard_argmax_pred_brightness.iloc[0]),
            best_hard_pred=float(h.hard_argmax_pred_brightness.max()),
            active_views_min=int(active.sum(1).min()), active_views_max=int(active.sum(1).max()),
            views_used=len(np.flatnonzero(active.sum(0))),
            view_switches=int(np.any(active[1:]!=active[:-1], axis=1).sum()),
            mean_grad_norm=float(h.grad_norm.mean()) if 'grad_norm' in h else None,
            max_soft_hard_gap=float((h.pred_brightness-h.hard_argmax_pred_brightness).abs().max())))
        for col in weights:
            source_weights.append(dict(folder=folder.name, source=col.removeprefix('selected_weight_'),
                active_states=int((h[col]>0).sum()), mean_weight=float(h[col].mean())))
    pd.DataFrame(histories).to_csv(out/'trajectory_diagnostics.csv', index=False)
    pd.DataFrame(source_weights).to_csv(out/'trajectory_source_weights.csv', index=False)
    pd.DataFrame(return_checks).to_csv(out/'native_return_vs_queried_pool.csv', index=False)
    metadata=dict(mountains=mountains, contexts=len(starts), unique_starts=int(starts.sequence.nunique()),
        test_labels_read=False, biological_success_assigned_to_unknown=False,
        exact_match_endpoint='Ancillary measured rediscovery only; not de novo success/failure',
        comparison_scope='Search engines conditional on shared GLOM, multi-view starts and multi-view final scoring',
        single_view_ablation_scope='Search-guidance-only; starts and final scoring still multi-view',
        gradient_budget='All encoder forward sequences matched; hard unique proposals and backward/MLM costs differ',
        history_files_available=len(histories),
        input_hashes={n:sha(src/n) for n in ['frozen_endpoints.csv','frozen_starts.csv','query_budgets.csv','frozen_selected.csv']})
    (out/'AUDIT.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(summary.to_json(orient='records', indent=2))
    print('AUDIT_OUT', out)
    if histories: print('HISTORY', pd.DataFrame(histories).describe().to_json())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--mountains',type=int,default=60);a=p.parse_args();main(a.root,a.mountains)
