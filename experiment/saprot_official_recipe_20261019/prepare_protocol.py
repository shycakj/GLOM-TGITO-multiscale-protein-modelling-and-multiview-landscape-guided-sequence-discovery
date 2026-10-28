"""Freeze task-data-only baselines and audit token row identities before training."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    original=json.loads((ROOT/'data/split_manifest.json').read_text())
    sources=json.loads((ROOT/'saprot_official_20261019/SOURCE_MANIFEST.json').read_text())
    for name,digest in original['files'].items():assert sha(ROOT/'data'/name)==digest,name
    spec={'date':'2026-10-19','name':HERE.name,
          'user_clarification':'No externally task-fine-tuned checkpoint. Use general pretrained SaProt and train only on original project training partitions.',
          'base_repo':'westlake-repl/SaProt_650M_AF2','base_revision':'d9b9ad00ef61c0990e611b2b43f2231c7de24b38',
          'base_weight_sha256':'513c39088804b74309d961174d5fc7aeabfb9b79d2814cdc7e465d4fe4b9f216',
          'external_fluorescence_or_eyfp_adapters_used':False,
          'official_source_repo':sources['official_code_repo'],'official_source_revision':sources['official_code_revision'],
          'head':'Native CLS, Dropout, Linear(1280,1280), tanh, Dropout, Linear(1280,1); random task-head initialization',
          'lora':{'rank':8,'alpha':16,'dropout':0.,'targets':['query','key','value','intermediate.dense','output.dense']},
          'input_modes':{'aa':'All structure positions masked with #, matching official fluorescence task input',
                         'pred3di':'Existing frozen ESM2/Bridge predicted 3Di, identical to current native SaProt controls'},
          'splits':['random','depth'],'seeds':original['train_seed_list'],'data_hashes':original['files'],
          'training':{'loss':'MSE on original target log-brightness','lr':1e-4,'betas':[.9,.98],
                      'weight_decay':.01,'effective_batch':64,'max_epochs':25,'patience':6,
                      'scheduler':'cosine to 5% initial learning rate','gradient_clip':1.,'precision':'bf16 autocast; FP32 master weights'},
          'selection':'minimum validation MSE, matching official regression selection rule',
          'adaptations_from_model_card':'25-epoch cap and patience 6 to match existing local SaProt compute budget; bf16 instead of fp16; original data partitions and seeds',
          'test_policy':'Test predictions made only after validation checkpoint selection; no test feedback',
          'metrics':['rmse','mae','r2','spearman','pearson','measured_top6_30_100'],
          'primary_report':'All seeds, both input modes, both splits; no test-based choice of a favorable baseline',
          'original_runs_unchanged':True}
    path=HERE/'PROTOCOL.json'
    if path.exists():assert json.loads(path.read_text())==spec,'Frozen protocol changed'
    else:path.write_text(json.dumps(spec,indent=2)+'\n')
    print('PROTOCOL',sha(path))
    # Input generation only on the remote machine with the already verified model/tokenizer.
    base=ROOT/'repro_project/assets/saprot_650m_af2'
    if not base.exists():return
    from transformers import AutoTokenizer
    assert sha(base/'pytorch_model.bin')==spec['base_weight_sha256']
    tok=AutoTokenizer.from_pretrained(base,local_files_only=True)
    raw=pd.read_csv(ROOT/'repro_project/data/raw/pplu_31480_sequences.csv',usecols=['sequence'])
    existing=json.loads((ROOT/'artifacts/embeddings/saprot650_pred3di_state.json').read_text())
    sequence_hash=hashlib.sha256('\n'.join(raw.sequence).encode()).hexdigest()
    assert len(raw)==31480 and sequence_hash==existing['sequence_sha256']
    unique=pd.read_csv(ROOT/'data/unique_sequences.csv',usecols=['row_id','sequence'])
    assert raw.sequence.iloc[unique.row_id.to_numpy()].tolist()==unique.sequence.tolist()
    parts=[]
    for offset in range(0,len(raw),512):
        seqs=raw.sequence.iloc[offset:offset+512].tolist()
        batch=tok([' '.join(aa+'#' for aa in seq) for seq in seqs],padding=True,return_tensors='np')
        ids=batch['input_ids'];assert ids.shape==(len(seqs),223) and not (ids==tok.unk_token_id).any()
        for row,seq in zip(ids,seqs):
            assert tok.convert_ids_to_tokens(row[1:-1].tolist())==[aa+'#' for aa in seq]
        parts.append(ids)
    arr=np.concatenate(parts).astype('int64')
    np.save(HERE/'aa_input_ids.npy',arr)
    report={'status':'PASS','original_rows':len(raw),'unique_rows':len(unique),
            'sequence_order_sha256':sequence_hash,'every_unique_row_mapping_verified':True,
            'unknown_tokens':0,'shape':list(arr.shape),'aa_tokens_sha256':sha(HERE/'aa_input_ids.npy'),
            'protocol_sha256':sha(path),'general_pretrained_weights_sha256':sha(base/'pytorch_model.bin')}
    (HERE/'INPUT_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n')
    print('INPUT_AUDIT_PASS')

if __name__=='__main__':main()
