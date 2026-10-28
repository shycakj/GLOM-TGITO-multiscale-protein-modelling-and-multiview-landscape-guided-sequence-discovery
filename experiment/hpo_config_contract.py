"""Explicit routing of historical HPO parameters; no Torch or GPU required.

The frozen historical winner is an input, not a claim of optimality after a
change of predictor, objective or numerical implementation. This module never
loads the later two-dimensional risk-calibration grid as a replacement.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

TUNED_KEYS = (
    'terrain_weight', 'seed_explore_weight', 'lr', 'terrain_topk',
    'terrain_gate_temp', 'terrain_support_weight', 'terrain_risk_weight',
    'terrain_pred_gap_weight', 'terrain_source_quality_power', 'proposal_every',
)
TERRAIN_KEYS = {
    'terrain_topk': 'topk_sources', 'terrain_gate_mode': 'terrain_gate_mode',
    'terrain_gate_temp': 'terrain_gate_temp',
    'terrain_support_weight': 'terrain_support_weight',
    'terrain_risk_weight': 'terrain_risk_weight',
    'terrain_pred_gap_weight': 'terrain_pred_gap_weight',
    'terrain_source_quality_power': 'terrain_source_quality_power',
    'true_high_weight': 'true_high_weight', 'miner_weight': 'miner_weight',
}
EXECUTION_KEYS = {'num_restarts'}
METADATA_KEYS = {'hpo_trial', 'confirmation_mean', 'confirmation_std', 'protocol'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(4 << 20), b''):
            h.update(part)
    return h.hexdigest()


def load_confirmed_winner(folder):
    folder = Path(folder)
    hp = json.loads((folder / 'FROZEN_TGITO_PARAMS.json').read_text(encoding='utf-8'))
    confirmation = json.loads((folder / 'confirmation.json').read_text(encoding='utf-8'))
    best = max(confirmation, key=lambda x: (x['confirmation_mean'], -x['confirmation_std']))
    if hp['hpo_trial'] != best['trial']:
        raise ValueError('Frozen trial does not equal confirmation winner')
    if set(best['params']) != set(TUNED_KEYS):
        raise ValueError('Unexpected historical HPO dimensions')
    for key in TUNED_KEYS:
        if hp.get(key) != best['params'][key]:
            raise ValueError(f'Frozen/confirmation mismatch: {key}')
    for key in ['confirmation_mean', 'confirmation_std']:
        if hp[key] != best[key]:
            raise ValueError(f'Confirmation metadata mismatch: {key}')
    return hp


def parameter_diff(hp, generator_config, terrain_config, num_restarts=1):
    rows = []
    for key, expected in hp.items():
        if key in METADATA_KEYS:
            continue
        if key in TERRAIN_KEYS:
            section, field, actual = 'terrain', TERRAIN_KEYS[key], terrain_config.get(TERRAIN_KEYS[key])
        elif key in EXECUTION_KEYS:
            section, field, actual = 'execution', key, num_restarts
        else:
            section, field, actual = 'generator', key, generator_config.get(key)
        rows.append(dict(parameter=key, section=section, field=field,
                         historical=expected, effective=actual, matches=actual == expected,
                         hpo_dimension=key in TUNED_KEYS))
    return rows


def resolve(hp, generator_defaults, terrain_defaults, *, overrides=None):
    """Map every historical field and make any departure explicit and reasoned."""
    if set(TUNED_KEYS) - set(hp):
        raise ValueError('Incomplete HPO parameters: '+', '.join(sorted(set(TUNED_KEYS)-set(hp))))
    gen, terr, execution = deepcopy(generator_defaults), deepcopy(terrain_defaults), {}
    for key, value in hp.items():
        if key in METADATA_KEYS:
            continue
        if key in TERRAIN_KEYS:
            dest = TERRAIN_KEYS[key]
            if dest not in terr:
                raise ValueError(f'Terrain parameter not supported: {key}')
            terr[dest] = value
        elif key in EXECUTION_KEYS:
            execution[key] = value
        elif key in gen:
            gen[key] = value
        else:
            raise ValueError(f'Unrouted HPO parameter: {key}')
    for key, spec in (overrides or {}).items():
        if key not in hp or key in METADATA_KEYS or not isinstance(spec, dict) or not spec.get('reason'):
            raise ValueError(f'Invalid or unexplained override: {key}')
        if key in TERRAIN_KEYS:
            terr[TERRAIN_KEYS[key]] = spec['value']
        elif key in EXECUTION_KEYS:
            execution[key] = spec['value']
        else:
            gen[key] = spec['value']
    return dict(generator_config=gen, terrain_config=terr, execution=execution,
                explicit_overrides=deepcopy(overrides or {}),
                historical_hpo_trial=hp['hpo_trial'])


def validate_effective(hp, resolved):
    rows = parameter_diff(hp, resolved['generator_config'], resolved['terrain_config'],
                          resolved['execution'].get('num_restarts'))
    allowed = resolved.get('explicit_overrides', {})
    failures = []
    for row in rows:
        key = row['parameter']
        if key in allowed:
            spec = allowed[key]
            if not spec.get('reason') or row['effective'] != spec['value']:
                failures.append(key)
        elif not row['matches']:
            failures.append(key)
    if set(allowed) - {x['parameter'] for x in rows}:
        failures.append('unknown_override')
    if failures:
        raise ValueError('Effective configuration differs from declared HPO: ' + ', '.join(failures))
    return rows


def verify_input_hashes(root, manifest):
    """Call at worker start/resume, not only during initial plan creation."""
    root = Path(root).resolve()
    for relative, expected in manifest.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f'Manifest path escapes root: {relative}')
        if not path.is_file() or sha(path) != expected:
            raise ValueError(f'Input identity mismatch: {relative}')
