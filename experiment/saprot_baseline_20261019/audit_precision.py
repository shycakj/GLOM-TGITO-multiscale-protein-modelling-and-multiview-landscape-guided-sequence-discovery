"""Check native embedding precision on training sequences only, with no labels."""
from pathlib import Path
import json, numpy as np, pandas as pd, torch
from transformers import EsmModel
from native_saprot import check_loaded_weights

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
def main():
    torch.set_num_threads(3);torch.backends.cuda.matmul.allow_tf32=False
    train=pd.read_csv(ROOT/'data/train.csv',usecols=['row_id'])
    rows=train.row_id.to_numpy()[np.linspace(0,len(train)-1,128,dtype=int)]
    cache=ROOT/'artifacts/embeddings'
    token=np.load(cache/'saprot650_pred3di_input_ids.npy',mmap_mode='r')
    previous=np.asarray(np.load(cache/'saprot650_pred3di_mean.npy',mmap_mode='r')[rows],dtype='float32')
    model,info=EsmModel.from_pretrained(ROOT/'repro_project/assets/saprot_650m_af2',
                   add_pooling_layer=False,local_files_only=True,torch_dtype=torch.float32,output_loading_info=True)
    check_loaded_weights(model,info);model=model.cuda().eval();parts=[]
    with torch.inference_mode():
        for subset in np.array_split(rows,16):
            ids=torch.tensor(np.asarray(token[subset]),device='cuda')
            h=model(input_ids=ids,attention_mask=torch.ones_like(ids)).last_hidden_state[:,1:222]
            parts.append(h.mean(1).cpu().numpy())
    fp32=np.concatenate(parts);delta=previous-fp32
    variation=fp32-fp32.mean(0);centered_delta=delta-delta.mean(0)
    ratio=float(np.sqrt(np.mean(centered_delta**2))/np.sqrt(np.mean(variation**2)))
    report={'rows':len(rows),'selection':'128 evenly spaced training row indices; no fluorescence labels',
            'bf16_vs_fp32_relative_rms':float(np.sqrt(np.mean(delta**2)/np.mean(fp32**2))),
            'centered_precision_error_over_between_sequence_variation':ratio,
            'cached_mean_norm':float(np.linalg.norm(previous,axis=1).mean()),
            'fp32_mean_norm':float(np.linalg.norm(fp32,axis=1).mean()),
            'fp32_between_sequence_rms':float(np.sqrt(np.mean(variation**2))),
            'centered_precision_error_rms':float(np.sqrt(np.mean(centered_delta**2))),
            'tf32_enabled':False,'test_labels_read':False}
    np.savez(HERE/'PRECISION_TRAIN_SAMPLE.npz',rows=rows,bf16=previous,fp32=fp32)
    (HERE/'PRECISION_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
