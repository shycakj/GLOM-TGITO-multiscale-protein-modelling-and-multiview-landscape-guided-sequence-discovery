"""Freeze evaluation independently of generation before formal candidate readout."""
from pathlib import Path
import json,time
from hpo_config_contract import sha,verify_input_hashes
from experiment_core import write_json
root=Path(__file__).resolve().parent;out=root/'artifacts/discovery_v3'
files=[root/'audit/structure_validation/reference_backbone.npz',root/'audit/structure_validation/REFERENCE_MAPPING.json',
       root/'third_party/ProteinMPNN/protein_mpnn_utils.py',root/'third_party/ProteinMPNN/vanilla_model_weights/v_48_020.pt',
       root/'validate_structure_evidence.py',root/'data/calibration.csv',root/'data/test.csv',root/'data/train.csv']
for method in ['glom','esm2_650m_mlp']:
    for seed in [20261013,20261014,20261015]:files.append(root/f'artifacts/models/random/{method}_seed{seed}/best.pt')
plan=dict(unix_time=time.time(),generation_protocol_sha256=sha(out/'PROTOCOL.json'),evaluation_source_sha256=sha(root/'evaluate_discovery_v3.py'),
          input_hashes={p.relative_to(root).as_posix():sha(p) for p in files},
          metrics=['Native frozen Top6/30/100 batch size','Cross-seed prediction means and gains relative to starts',
                   'Novelty and Hamming diversity','Measured-test RMSE, R2, Spearman, overall/high-fitness/nearest-training distance',
                   'Empirical calibration error envelope and measured-test coverage',
                   'Complete confidence screening versus predictor-only screening on heldout measured libraries',
                   'ProteinMPNN fixed-backbone compatibility and scored residue coverage'],
          labels_are_not_inferred_from_catalogue_absence=True,rankings_are_not_changed_after_judge_scores=True,
          lower_envelope_is_not_a_coverage_guarantee_for_optimized_sequences=True,
          trial12_is_historical_hpo_transfer_not_a_newly_selected_optimum=True,
          campaign_repetitions=1,overlapping_measured_library_subsets_are_not_independent_campaigns=True)
target=out/'EVALUATION_PLAN_FROZEN.json'
if target.exists():
    old=json.loads(target.read_text());assert old['evaluation_source_sha256']==plan['evaluation_source_sha256'];verify_input_hashes(root,old['input_hashes'])
else:write_json(target,plan)
print(json.dumps(dict(frozen=str(target),inputs=len(files))))
