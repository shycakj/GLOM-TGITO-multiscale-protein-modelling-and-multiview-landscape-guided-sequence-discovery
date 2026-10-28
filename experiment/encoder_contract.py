"""Differentiable encoder with the original hard-token ESM input convention."""
import torch
from pipeline.tgito.esm_bridge import DifferentiableESMBridge

class TrainingCompatibleEncoder(DifferentiableESMBridge):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        torch.backends.cuda.matmul.allow_tf32=False
        self.native_unmasked_scale=.88 if self.esm.config.token_dropout else 1.
    def _soft_esm_inputs(self,aa_probs):
        inputs,mask=super()._soft_esm_inputs(aa_probs)
        # Native ESM applies this token-dropout correction even in eval mode.
        return inputs*self.native_unmasked_scale,mask
    def audit_native_hard(self,sequences):
        from pipeline.tgito.aa import seq_to_ids0
        device=next(self.parameters()).device
        with torch.inference_mode():
            ids=torch.stack([seq_to_ids0(s,len(s),device) for s in sequences])
            soft=self(torch.nn.functional.one_hot(ids,20).float())
            tokens=self.tokenizer(sequences,return_tensors='pt',padding=True)
            tokens={k:v.to(device) for k,v in tokens.items()}
            self.esm.embeddings.token_dropout=bool(self.esm.config.token_dropout)
            try:native_esm=self.esm(**tokens).last_hidden_state[:,1:1+ids.shape[1]]
            finally:self.esm.embeddings.token_dropout=False
            native=self.bridge(native_esm)['saprot_hat']
            delta=soft['H'].float()-native.float()
            result={'n':len(sequences),'H_rmse':float(delta.square().mean().sqrt()),
                    'H_max_abs':float(delta.abs().max()),
                    'esm_rmse':float((soft['esm']-native_esm).square().mean().sqrt()),
                    'reference':'native ESM input_ids + original Bridge.forward'}
            if result['H_rmse']>1e-4:raise RuntimeError('Native/soft encoder mismatch: '+str(result))
            return result
