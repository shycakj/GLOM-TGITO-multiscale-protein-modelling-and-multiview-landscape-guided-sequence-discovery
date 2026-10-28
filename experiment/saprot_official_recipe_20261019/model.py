"""Official SaProtHub CLS/LoRA architecture, initialized only from general pretraining."""
from pathlib import Path
import json,sys
import numpy as np
import torch
from transformers import EsmForSequenceClassification,AutoTokenizer
from peft import LoraConfig,get_peft_model

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
BASE=ROOT/'repro_project/assets/saprot_650m_af2'

def make_model():
    spec=json.loads((HERE/'PROTOCOL.json').read_text())
    manifest=json.loads((BASE/'DOWNLOAD_VERIFIED.json').read_text())
    assert manifest['revision']==spec['base_revision']
    base,info=EsmForSequenceClassification.from_pretrained(BASE,num_labels=1,
              local_files_only=True,output_loading_info=True,torch_dtype=torch.float32)
    classifier={'classifier.dense.weight','classifier.dense.bias',
                'classifier.out_proj.weight','classifier.out_proj.bias'}
    unused={'esm.contact_head.regression.bias','esm.contact_head.regression.weight',
            'esm.embeddings.position_embeddings.weight'}
    assert set(info['missing_keys'])==classifier|unused,info
    assert not info.get('mismatched_keys') and not info.get('error_msgs'),info
    assert base.config.position_embedding_type=='rotary'
    cfg=spec['lora']
    model=get_peft_model(base,LoraConfig(task_type='SEQ_CLS',r=cfg['rank'],lora_alpha=cfg['alpha'],
          lora_dropout=cfg['dropout'],target_modules=cfg['targets'],modules_to_save=['classifier'],bias='none'))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    # A newly initialized adapter has zero LoRA B matrices: no external task adaptation.
    b=[p for n,p in model.named_parameters() if 'lora_B' in n]
    assert b and all(torch.count_nonzero(p)==0 for p in b)
    return model

def token_cache(mode):
    assert mode in ['aa','pred3di']
    if mode=='aa':return np.load(HERE/'aa_input_ids.npy',mmap_mode='r')
    return np.load(ROOT/'artifacts/embeddings/saprot650_pred3di_input_ids.npy',mmap_mode='r')
