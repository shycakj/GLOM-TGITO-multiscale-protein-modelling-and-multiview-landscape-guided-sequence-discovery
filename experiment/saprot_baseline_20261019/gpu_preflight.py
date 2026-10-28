"""Real-weight forward/backward checks and measured admission budget, no labels."""
from pathlib import Path
import argparse, json, sys, time
import numpy as np
import torch
from train_saprot_lora import Regressor

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--batch-size',type=int,default=16)
    args=ap.parse_args(); torch.set_num_threads(4)
    torch.manual_seed(20261013)
    states=ROOT/'artifacts/embeddings'
    parity=json.loads((states/'saprot650_pred3di_smoke_native_parity.json').read_text())
    assert parity['relative_rms']<parity['threshold']
    tokens=np.load(states/'saprot650_pred3di_smoke_input_ids.npy')
    ids=torch.tensor(np.tile(tokens,(args.batch_size//len(tokens)+1,1))[:args.batch_size],device='cuda')
    mask=torch.ones_like(ids)
    model=Regressor(ROOT/'repro_project/assets/saprot_650m_af2',16).cuda().train()
    parameters=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(parameters,lr=1e-4,weight_decay=.01)
    torch.cuda.reset_peak_memory_stats(); times=[]
    for step in range(8):
        torch.cuda.synchronize(); begin=time.monotonic(); opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            pred=model(ids,mask)
            # Synthetic targets are an implementation check, not fitted assay data.
            loss=torch.nn.functional.huber_loss(pred.float(),torch.zeros_like(pred).float())
        assert torch.isfinite(loss)
        loss.backward()
        grads=[p.grad for p in parameters if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert any(float(g.abs().sum())>0 for g in grads)
        torch.nn.utils.clip_grad_norm_(parameters,1.); opt.step()
        torch.cuda.synchronize(); times.append(time.monotonic()-begin)
    model.eval()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        assert torch.isfinite(model(torch.cat([ids,ids]),torch.cat([mask,mask]))).all()
    reserved=torch.cuda.max_memory_reserved()/2**30
    report={'status':'PASS','gpu_name':torch.cuda.get_device_name(),
            'batch_size':args.batch_size,'effective_batch_size':64,
            'accumulation':64//args.batch_size,'backward_steps':8,
            'median_train_step_seconds':float(np.median(times[2:])),
            'peak_reserved_gib':reserved,'admission_gib_per_lora':max(6.,reserved*1.35+1.5),
            'finite_outputs_and_gradients':True,'label_source':'synthetic zero targets only',
            'native_saprot_parity':parity,
            'torch':torch.__version__,'cuda':torch.version.cuda}
    (HERE/'GPU_PREFLIGHT.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
