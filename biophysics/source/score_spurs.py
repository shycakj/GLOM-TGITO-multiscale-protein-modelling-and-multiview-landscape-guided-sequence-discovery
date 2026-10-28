"""Official SPURS multi-mutant inference with exhaustive checkpoint loading."""
import os, sys, json, csv, copy, hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ['HF_HOME']=str(ROOT/'cache/huggingface');os.environ['TORCH_HOME']=str(ROOT/'cache/torch')
sys.path.insert(0,str(ROOT/'tools/SPURS'))
import torch
import esm
from omegaconf import OmegaConf
torch.set_num_threads(4)
device='cpu'  # Preserve both GPUs for molecular dynamics; 221-aa inference is bounded.
ckpath=ROOT/'tools/SPURS_multi.ckpt'
hub_manifest=json.loads((ROOT/'tools/SPURS_hub_file_manifest.json').read_text())
expected_hash=next(x['lfs']['oid'] for x in hub_manifest if x['path'].endswith('/best.ckpt'))
assert hashlib.sha256(ckpath.read_bytes()).hexdigest()==expected_hash
checkpoint=torch.load(ckpath,map_location='cpu',weights_only=False)
state={k.removeprefix('model.'):v for k,v in checkpoint['state_dict'].items() if k.startswith('model.')}
# The official SPURS checkpoint contains every ESM parameter. Construct the exact
# public ESM2 architecture locally, then load ALL SPURS parameters strictly. This
# avoids a redundant 2.6-GB base-model download; no random parameter reaches inference.
def construct_esm(name):
    assert name=='esm2_t33_650M_UR50D'
    alphabet=esm.data.Alphabet.from_architecture('ESM-1b')
    model=esm.model.esm2.ESM2(num_layers=33,embed_dim=1280,attention_heads=20,alphabet=alphabet,token_dropout=True)
    return model,alphabet
esm.pretrained.load_model_and_alphabet_hub=construct_esm
from spurs.models.stability.spurs_multi import SPURSMulti
from spurs.inference import parse_pdb,parse_pdb_for_mutation
cfg=OmegaConf.load(ROOT/'tools/SPURS_multi_config.yaml')
del cfg.model['_target_']
torch.manual_seed(20261027)
model=SPURSMulti(cfg.model).to(device)
loaded=model.load_state_dict(state,strict=True)
assert not loaded.missing_keys and not loaded.unexpected_keys
model.eval()
rows=list(csv.DictReader(open(ROOT/'inputs/candidate_start_pairs.csv')))
seqs={r['system_id']:r['sequence'] for r in csv.DictReader(open(ROOT/'inputs/systems.csv'))}
cache={}
def predict(source_id, mutations, seed):
    if source_id not in cache:
        batch=parse_pdb(str(ROOT/'structures'/source_id/'precursor_for_structure_predictors.pdb'),source_id,'C',cfg,device=device)
        cache[source_id]=batch
    batch=copy.deepcopy(cache[source_id])
    # Verify mutation numbering against the full 221-residue input sequence.
    for mut in mutations:
        assert seqs[source_id][int(mut[1:-1])-1]==mut[0]
    ids,aa=parse_pdb_for_mutation([mutations])
    batch['mut_ids']=ids.to(device);batch['append_tensors']=aa.to(device)
    torch.manual_seed(seed)
    with torch.inference_mode():pred=model(batch)
    if isinstance(pred,dict):
        print('OUTPUT_KEYS',pred.keys(),flush=True)
        raise TypeError('Inspect official output mapping before extraction')
    if isinstance(pred,(list,tuple)):
        print('OUTPUT_TYPES',[(type(x).__name__,getattr(x,'shape',None)) for x in pred],flush=True)
        raise TypeError('Inspect official multi-model output before extraction')
    assert pred.numel()==1,pred.shape
    return float(pred.reshape(-1)[0])
results=[]
for row in rows:
    for baseline,source_id,field in [('WT','WT','mutations_vs_wt'),('start',row['start_system_id'],'mutations_vs_start')]:
        muts=row[field].split(';')
        vals=[predict(source_id,muts,20261027+i) for i in range(3)]
        rec={'candidate_id':row['candidate_id'],'rank':int(row['rank']),'comparison':baseline,'reference_id':source_id,
             'mutations':row[field],'SPURS_ddG_kcal_mol_mean':sum(vals)/len(vals),
             'ddG_seed0':vals[0],'ddG_seed1':vals[1],'ddG_seed2':vals[2],
             'sign':'negative predicts stabilization; positive predicts destabilization',
             'structure_representation':'sequence backbone reconstructed from mature crystal; GYG represented as standard residues for predictor input'}
        results.append(rec);print(json.dumps(rec),flush=True)
        out=ROOT/'results/SPURS_predictions.csv'
        with open(out,'w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rec));writer.writeheader();writer.writerows(results)
(ROOT/'results/SPURS_AUDIT.json').write_text(json.dumps({'checkpoint_sha256':hashlib.sha256(ckpath.read_bytes()).hexdigest(),
    'all_726_checkpoint_parameters_loaded':len(state),'strict_load':True,'comparisons':len(results),
    'model':'SPURS multi','inference_seeds':[20261027,20261028,20261029],
    'note':'Inference seed repeats assess decoder-order sensitivity; they are not independent trained models.',
    'sign_evidence':'official megascale_multi.py negates the experimental unfolding ddG_ML target',
    'chemical_scope':'Neural model uses standard-residue backbone; separate MD explicitly retains mature CR2.'},indent=2))
