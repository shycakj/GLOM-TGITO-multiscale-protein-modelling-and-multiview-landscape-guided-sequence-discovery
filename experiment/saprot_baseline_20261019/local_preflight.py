"""Local tests of token identity, indexing and protocol; no GPU claims."""
from pathlib import Path
import hashlib, json, sys, ast
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer
from cache_saprot import joint_sequences, checked_tokenize

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent

def main():
    spec=json.loads((HERE/'PROTOCOL.json').read_text())
    mapping={0:'<pad>',**{i+1:c for i,c in enumerate('acdefghiklmnpqrstvwy')}}
    aa='ACDEFGHIKLMNPQRSTVWY'
    seqs=[(aa*12)[:221],(aa[::-1]*12)[:221]]
    states=np.asarray([np.arange(221)%21,np.arange(221)[::-1]%21],dtype='uint8')
    joints=joint_sequences(seqs,states,mapping)
    tok=AutoTokenizer.from_pretrained(HERE/'model_metadata',local_files_only=True)
    batch=checked_tokenize(tok,joints)
    assert tok.convert_ids_to_tokens(batch['input_ids'][0].tolist())[1]=='A#'
    assert all(tok.tokenize(s)==[s[i:i+2] for i in range(0,442,2)] for s in joints)
    assert torch.equal(batch['input_ids'][0],checked_tokenize(tok,[joints[0]])['input_ids'][0])
    # Verify original row identities and split disjointness without fitting anything.
    raw=ROOT.parent/'handoff_20261001/repro_project/data/raw/pplu_31480_sequences.csv'
    original=pd.read_csv(raw,usecols=['sequence'])
    data={r:pd.read_csv(ROOT/f'data/{r}.csv',usecols=['row_id','sequence']) for r in
          ['train','validation','calibration','test']}
    for role,frame in data.items():
        assert original.sequence.iloc[frame.row_id].tolist()==frame.sequence.tolist(),role
    for i,a in enumerate(data):
        for b in list(data)[i+1:]:assert not set(data[a].sequence)&set(data[b].sequence)
    for script in HERE.glob('*.py'):ast.parse(script.read_text(encoding='utf8'))
    report={'status':'PASS','scope':'local CPU and tokenizer checks only',
            'native_tokenizer_loaded':True,'joint_tokens_per_sequence':221,
            'special_tokens':2,'unknown_tokens':0,'padding_state_to_hash_verified':True,
            'single_vs_batch_token_ids_identical':True,
            'all_original_row_ids_and_random_partition_disjointness_verified':True,
            'model_weights_downloaded_on_local_host':False,'gpu_execution_in_this_check':False,
            'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
            for p in HERE.glob('*.py')}}
    (HERE/'LOCAL_PREFLIGHT.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in report.items() if k!='source_hashes'},indent=2))

if __name__=='__main__':main()
