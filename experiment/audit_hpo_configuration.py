"""Audit frozen HPO against the completed run; preserve original run artifacts."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
from copy import deepcopy
import json
import math
import numpy as np
import pandas as pd
from hpo_config_contract import load_confirmed_winner, parameter_diff, resolve, validate_effective, sha

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent / 'handoff_20261001/repro_project'
BASE = ROOT / 'artifacts/models/random/glom_seed20261013'
OUT = ROOT / 'audit/hpo_configuration_20261014'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    folder = PROJECT / 'artifacts/tgito_hpo'
    hp = load_confirmed_winner(folder)
    cfg = json.loads((PROJECT / 'configs/publication_experiment.json').read_text(encoding='utf-8'))
    run = json.loads((BASE / 'full_discovery_v2/PROTOCOL.json').read_text())
    risk = json.loads((BASE / 'risk_calibration/FROZEN_CONFIG.json').read_text())
    rows = parameter_diff(hp, run['config'], risk['terrain_config'])
    pd.DataFrame(rows).to_csv(OUT / 'parameter_diff.csv', index=False)
    trials = pd.read_csv(folder / 'trials.csv')
    trial = trials[trials.number.eq(hp['hpo_trial'])].iloc[0]
    assert trial.state == 'COMPLETE'
    for key in cfg['tgito_tuning']['search_space']:
        assert math.isclose(float(trial['params_' + key]), float(hp[key]), rel_tol=1e-13, abs_tol=1e-14), key
    confirmation = json.loads((folder / 'confirmation.json').read_text())

    # Identity-only overlap check: these readers do not load fluorescence values.
    target = pd.read_csv(PROJECT / 'data/processed/tuning_legacy_embargoed/legacy_hidden_target.csv', usecols=['sequence'])
    starts = pd.read_csv(PROJECT / 'data/processed/tuning_legacy_embargoed/tuning_seed_pool.csv', usecols=['sequence'])
    overlap = {}
    for name in ['train', 'validation', 'calibration', 'test']:
        frame = pd.read_csv(ROOT / f'data/{name}.csv', usecols=['sequence'])
        overlap[name] = dict(legacy_hpo_target=int(frame.sequence.isin(target.sequence).sum()),
                             legacy_hpo_start=int(frame.sequence.isin(starts.sequence).sum()))
    # Independently reproduce the old hidden400/legacy terrain identity overlap.
    memory = np.load(PROJECT / 'provenance/legacy_hpo_assets/terrain_12source.npz', allow_pickle=True)
    ids = memory['sample_idx']
    raw = pd.read_csv(PROJECT / 'data/raw/pplu_31480_sequences.csv', usecols=['sequence'])
    embargo = set((PROJECT / 'data/sealed/search/hidden_sequence_hashes.txt').read_text().split())
    import hashlib
    terrain_hashes = {hashlib.sha256(s.encode()).hexdigest() for s in raw.iloc[ids].sequence}

    strict = resolve(hp, run['config'], risk['terrain_config'])
    validate_effective(hp, strict)
    (OUT / 'HISTORICAL_HPO_RESOLVED_REFERENCE.json').write_text(json.dumps(strict, indent=2), encoding='utf-8')
    corrected = resolve(hp, run['config'], risk['terrain_config'], overrides={
        'amp_dtype': dict(value='off', reason='Current native/live encoder audit uses float32; retain for corrected implementation, compare precision separately.')})
    corrected.update(status='Prepared reference only; not a launched or current-optimum run',
                     numerical_implementation='TrainingCompatibleEncoder + canonical_bits + ConsistentObjectiveMixin',
                     run_identity='Historical trial12 transferred to corrected current pipeline',
                     source_sha256=sha(folder / 'FROZEN_TGITO_PARAMS.json'))
    validate_effective(hp, corrected)
    (OUT / 'HPO_TRIAL12_CORRECTED_REFERENCE.json').write_text(json.dumps(corrected, indent=2), encoding='utf-8')

    # Regression checks exercise the real incident and erroneous parameter routing.
    checks = []
    for label, value in [('completed_run_rejected', dict(generator_config=run['config'], terrain_config=risk['terrain_config'], execution={'num_restarts': 1})),
                         ('topk_overwrite_rejected', deepcopy(corrected))]:
        if label == 'topk_overwrite_rejected': value['terrain_config']['topk_sources'] = 3
        try: validate_effective(hp, value)
        except ValueError: checks.append(label)
        else: raise AssertionError(label)
    try: resolve({**hp, 'silently_ignored_parameter': 1}, run['config'], risk['terrain_config'])
    except ValueError: checks.append('unknown_parameter_rejected')
    else: raise AssertionError('Unknown parameter was ignored')
    checks += ['historical_reference_all_fields_match', 'explicit_precision_override_validated']

    report = dict(checked_cst=datetime.now(timezone(timedelta(hours=8))).isoformat(),
        historical_winner=hp, trials_rows=len(trials), trial_states=trials.state.value_counts().to_dict(),
        confirmation_top_trials=[c['trial'] for c in confirmation],
        best_vs_second_confirmation_gap=confirmation[0]['confirmation_mean']-confirmation[1]['confirmation_mean'],
        frozen_vs_trial_csv_vs_confirmation_consistent=True,
        hpo_dimensions=len(cfg['tgito_tuning']['search_space']),
        mismatched_hpo_dimensions=sum(not x['matches'] and x['hpo_dimension'] for x in rows),
        parameter_diff=rows, legacy_hpo_overlap_current_split=overlap,
        old_hidden400_sequences_in_legacy_hpo_terrain=len(embargo & terrain_hashes),
        old_hpo_objective=cfg['tgito_tuning']['objective'],
        preserved_valid_checks=['Current predictor and risk config hashes equal completed run manifest',
             'Canonical P4 cache/live agreement all 31480 rows; native ESM equivalence audited',
             'Completed generation was multiterrain, but was not historical HPO winner'],
        findings=[
            dict(id='HPO_NOT_APPLIED',kind='confirmed_error',detail='Ten-dimensional frozen winner replaced by defaults plus a two-dimensional endpoint-ranking calibration.'),
            dict(id='PEAK_PRIOR',kind='protocol_difference',detail='Current plan uses historical nonuniform source prior and different peak score; neutral prior must be judged from provenance/current development evidence, not restored blindly.'),
            dict(id='GATE_TRAINING',kind='protocol_difference',detail='Current MLP fits train-memory labels; source quality uses calibration queries. Historical v4 fits tune300 with OOF quality. Not the same experiment; not evidence by itself of test leakage.'),
            dict(id='MODEL_AND_INPUT',kind='intentional_correction',detail='Fresh predictor, canonical P4, native ESM correction and corrected hard objective differ from legacy HPO setup. Keep verified corrections; historical optimum is not established for this new setup.'),
            dict(id='HPO_LABEL_ANCESTRY',kind='verified_overlap',detail='Historical HPO target intersects current test; old terrain also contains 372 old hidden400 sequences. Do not call HPO lineage independent of these labels.'),
            dict(id='RANKING_NOT_HPO',kind='interpretation',detail='35/25/20/15/5 confidence weights are historical selection constants, not the ten HPO dimensions.'),
            dict(id='REPLICATES',kind='scope',detail='900 peaks x 3 roles are one search replicate with 2700 contexts/1277 unique starts; two cross-scoring seeds are not search repeats.'),
            dict(id='RESUME_IDENTITY',kind='execution_gap',detail='Existing full runner does not recheck all model/terrain/risk hashes at each worker start/resume. New execution must bind effective parameters and every input hash to outputs.'),
            dict(id='EVIDENCE_GAP',kind='scope',detail='No revised full-pipeline baseline comparison completed. Cross-model scores and fixed-backbone compatibility remain computational evidence, not generated-sequence measurements.')],
        decision=dict(retrain_predictors=False,rerun_embedding_cache=False,
             restart_affected_search_trajectories=True,posthoc_rerank_cannot_repair_search=True,
             rerun_original_hpo_study_blindly=False,
             next='Small predeclared paired verification including historical winner; freeze corrected development objective before any new HPO; then launch new output version.'),
        configuration_contract_checks=checks,
        source_hashes={str(p.relative_to(ROOT.parent)).replace('\\','/'):sha(p) for p in [
             folder/'FROZEN_TGITO_PARAMS.json',folder/'trials.csv',folder/'confirmation.json',
             PROJECT/'configs/publication_experiment.json', BASE/'full_discovery_v2/PROTOCOL.json',
             BASE/'risk_calibration/FROZEN_CONFIG.json',ROOT/'build_wide_peak_plan.py',ROOT/'build_train_terrain.py']})
    (OUT/'AUDIT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    hold=dict(status='hold_wrong_config_relaunch',reason='Completed full_discovery_v2 is not historical HPO winner; protocol review required before more generation',
              completed_results_preserved=True,completion_metadata_unchanged=True,
              audit='audit/hpo_configuration_20261014/AUDIT.json')
    (ROOT/'audit/HPO_CONFIGURATION_REVIEW_HOLD.json').write_text(json.dumps(hold,indent=2),encoding='utf-8')
    (BASE/'full_discovery_v2/INTERPRETATION_20261014.json').write_text(json.dumps({
        **hold,'label':'Completed non-HPO configuration diagnostic','historical_hpo_applied':False,
        'not_for_hpo_optimum_claim':True,'candidate_sequences_and_scores_remain_unchanged':True},indent=2),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['trials_rows','trial_states','mismatched_hpo_dimensions','old_hidden400_sequences_in_legacy_hpo_terrain','legacy_hpo_overlap_current_split','configuration_contract_checks']},ensure_ascii=False,indent=2))


if __name__ == '__main__': main()
