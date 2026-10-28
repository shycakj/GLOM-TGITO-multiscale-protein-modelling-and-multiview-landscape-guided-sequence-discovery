"""Measure real forward/backward memory for both official-style input modes."""
from pathlib import Path
import json,sys,time
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT))
from experiment_core import write_json,seed_all
from model import make_model,token_cache

def main():
    seed_all(20261013);torch.set_num_threads(3)
    model=make_model().cuda();model.train();records=[]
    # No optimizer updates and no fluorescence labels in this implementation check.
    for mode in ['aa','pred3di']:
        ids=torch.tensor(np.asarray(token_cache(mode)[:16]).copy(),device='cuda')
        timings=[]
        for step in range(4):
            model.zero_grad(set_to_none=True);start=time.monotonic()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                pred=model(input_ids=ids,attention_mask=torch.ones_like(ids)).logits.squeeze(-1)
                loss=torch.nn.functional.mse_loss(pred.float(),torch.zeros_like(pred).float())
            loss.backward();torch.cuda.synchronize()
            step_seconds=time.monotonic()-start
            assert torch.isfinite(loss)
            grads=[p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
            assert grads and all(torch.isfinite(g).all() for g in grads)
            timings.append(step_seconds)
        records.append({'mode':mode,'median_step_seconds':float(np.median(timings)),
                        'finite_outputs_and_gradients':True})
    peak=torch.cuda.max_memory_reserved()/2**30
    write_json(HERE/'GPU_PREFLIGHT.json',{'status':'PASS','batch_size':16,'accumulation':4,
               'effective_batch':64,'input_modes':records,'peak_reserved_gib':peak,
               'admission_gib_per_job':peak*1.35+1.5,'no_external_task_adapter':True,
               'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
               'targets':'synthetic zero labels only','gpu':torch.cuda.get_device_name()})
    print('OFFICIAL_RECIPE_GPU_PREFLIGHT_PASS',flush=True)

if __name__=='__main__':main()
