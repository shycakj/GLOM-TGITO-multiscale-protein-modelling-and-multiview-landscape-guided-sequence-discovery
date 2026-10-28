"""MaxWEnt diagonal fixed-mean MSE adaptation for the frozen GLOM regression head.

Based on de Mathelin et al., JMLR 26(4), 2025, Sections 5 and 7.5.
The representation and point predictor are unchanged. This is a conditional
head-uncertainty pilot, not a refit of GLOM or a replacement fluorescence label.
"""
from pathlib import Path
import argparse,copy,json,sys,time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from experiment_core import write_json
from hpo_config_contract import sha
from audit_embedding_features import REL,MODEL_SHA
from score_embedding_familiarity import load_features,validation_table

class NoisyHead(nn.Module):
    def __init__(self,head,initial_scale=1e-4):
        super().__init__();self.head=copy.deepcopy(head).eval();self.rho=nn.ParameterList();self.names=[]
        for name,p in self.head.named_parameters():
            p.requires_grad_(False);self.names.append(name)
            self.rho.append(nn.Parameter(torch.full_like(p,float(np.log(np.expm1(initial_scale))))))
        self.count=sum(p.numel() for p in self.rho)

    def entropy_proxy(self):
        return sum(F.softplus(p).sum() for p in self.rho)/self.count

    def draw(self,no_noise=False):
        params=dict(self.head.named_parameters())
        return {name:params[name] if no_noise else params[name]+F.softplus(r)*torch.randn_like(r)
                for name,r in zip(self.names,self.rho)}

    def apply_draw(self,x,draw):
        return torch.func.functional_call(self.head,draw,(x,)).squeeze(-1)

def mc_predictions(net,x,draws,seed,batch=2048):
    ys=[]
    # One sampled network is used for all rows in each draw, including across chunks.
    with torch.random.fork_rng(devices=[x.device.index or 0] if x.is_cuda else []):
        torch.manual_seed(seed)
        with torch.no_grad():
            for _ in range(draws):
                weights=net.draw();ys.append(torch.cat([net.apply_draw(x[st:st+batch],weights) for st in range(0,len(x),batch)]).cpu())
    return torch.stack(ys).numpy()

def main(a):
    root=a.root;out=root/REL/'maxwent_pilot';out.mkdir(parents=True,exist_ok=True)
    if (out/'DONE.json').exists():return
    torch.set_num_threads(3);torch.manual_seed(20261014);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;sys.path.insert(0,str(root/'repro_project'))
    from pipeline.tgito.revised_slim_internal import load_revised_slim
    modelpath=root/'artifacts/models/random/glom_seed20261013/best.pt';assert sha(modelpath)==MODEL_SHA
    model,mu,sd=load_revised_slim(modelpath,device=a.device);index,z,_,point=load_features(root)
    x=torch.as_tensor(z,device=a.device);net=NoisyHead(model.narrow).to(a.device).eval()
    frozen={name:p.detach().clone() for name,p in net.head.named_parameters()}
    with torch.no_grad():replay=(net.apply_draw(x[:128],net.draw(no_noise=True))*sd+mu).cpu().numpy()
    assert np.max(abs(replay-point[:128]))<1e-4
    split={p:index.index[index.partition==p].to_numpy() for p in ['train','validation','calibration','test']}
    labels={p:pd.read_csv(root/f'data/{p}.csv',usecols=['sequence','brightness']).set_index('sequence').brightness for p in ['train','validation']}
    ys={p:((labels[p].loc[index.sequence.iloc[split[p]]].to_numpy()-mu)/sd).astype('float32') for p in labels}
    xt=x[split['train']];yt=torch.as_tensor(ys['train'],device=a.device);xv=x[split['validation']]
    base_loss=((point[split['validation']]-mu)/sd-ys['validation'])**2
    tolerance=float(2*np.sqrt(np.sum((base_loss-base_loss.mean())**2))/len(base_loss))
    limit=float(base_loss.mean()+tolerance)
    config=dict(version='diagonal_fixed_mean_head_maxwent_mse_pilot',paper='https://www.jmlr.org/papers/v26/23-1359.html',
        plan_sha=sha(root/REL/'PLAN.json'),frozen_model_sha=MODEL_SHA,source_sha=sha(Path(__file__)),
        optimizer='Adam',learning_rate=.03,lambda_times_mean_scale=10.,initial_scale=1e-4,
        max_steps=a.steps,batch_size=256,validation_every=100,validation_draws=32,score_draws=64,
        seed=20261014,validation_base_mse=float(base_loss.mean()),validation_risk_limit=limit,
        validation_limit_rule='base per-sample MSE mean + 2/n * sqrt(sum centered squared losses)',
        selection_rule='highest entropy checkpoint satisfying validation risk limit; stop after 3 consecutive violations',
        test_used_in_fitting=False,dropout_disabled=True,mean_weights_frozen=True,representation_frozen=True)
    write_json(out/'CONFIG.json',config)
    opt=torch.optim.Adam(net.rho,lr=.03);history=[];best_entropy=-float('inf');best_state=None;best_step=None;violations=0;start=time.monotonic()
    for step in range(a.steps+1):
        if step:
            ids=torch.randint(len(xt),(256,),device=a.device);pred=net.apply_draw(xt[ids],net.draw())
            mse=F.mse_loss(pred,yt[ids]);entropy=net.entropy_proxy();loss=mse-10*entropy
            assert torch.isfinite(loss);opt.zero_grad();loss.backward();opt.step()
        if step%100==0 or step==a.steps:
            val=mc_predictions(net,xv,32,seed=814);risk=float(np.mean((val-ys['validation'][None,:])**2))
            ent=float(net.entropy_proxy().detach());accepted=bool(risk<=limit)
            rec=dict(step=step,validation_expected_mse=risk,validation_limit=limit,mean_scale=ent,accepted=accepted,seconds=time.monotonic()-start)
            history.append(rec);pd.DataFrame(history).to_csv(out/'training_history.csv',index=False);print(json.dumps(rec),flush=True)
            if accepted and ent>best_entropy:
                best_entropy=ent;best_state={k:v.detach().clone() for k,v in net.state_dict().items()};best_step=step
            violations=0 if accepted else violations+1
            if violations>=3:break
    if best_state is None:
        write_json(out/'DONE.json',dict(status='no_validation_admissible_noise_scale',config=config));return
    net.load_state_dict(best_state)
    for name,p in net.head.named_parameters():assert torch.equal(p,frozen[name]), 'Frozen mean mutated'
    assert sha(modelpath)==MODEL_SHA
    torch.save(dict(state=net.state_dict(),config=config,selected_step=best_step),out/'head_noise_scales.pt')
    values=mc_predictions(net,x,64,seed=271828)*sd+mu
    scored=index.copy();scored['prediction']=point;scored['maxwent_mean']=values.mean(0);scored['maxwent_sd']=values.std(0,ddof=1)
    assert np.isfinite(values).all()
    cal=index.partition.eq('calibration').to_numpy();high=cal&(point>=4.2)
    q=float(np.quantile(scored.maxwent_sd.to_numpy()[cal],.95,method='higher'))
    qh=float(np.quantile(scored.maxwent_sd.to_numpy()[high],.95,method='higher'));assert q>0 and qh>0
    scored['maxwent_q95_ratio']=scored.maxwent_sd/q;scored['maxwent_highcal_q95_ratio']=scored.maxwent_sd/qh
    scored.to_csv(out/'sequence_scores.csv',index=False)
    candidates=pd.read_csv(root/REL/'membership.csv').merge(scored.drop(columns='sequence'),on='feature_id',validate='many_to_one')
    candidates.to_csv(out/'candidate_scores.csv',index=False);summary=[]
    for (method,scope),d in candidates.groupby(['method','scope'],sort=False):
        summary.append(dict(method=method,scope=scope,n=len(d),sd_mean=float(d.maxwent_sd.mean()),
            outside_q95_fraction=float((d.maxwent_q95_ratio>1).mean()),degree_mean=float(d.maxwent_q95_ratio.mean()),
            outside_highcal_q95_fraction=float((d.maxwent_highcal_q95_ratio>1).mean()),highcal_degree_mean=float(d.maxwent_highcal_q95_ratio.mean())))
    pd.DataFrame(summary).to_csv(out/'method_summary.csv',index=False)
    # Test labels first read after checkpoint selection, with no test-dependent refitting.
    test=pd.read_csv(root/'data/test.csv',usecols=['sequence','brightness']).merge(scored,on='sequence',validate='one_to_one')
    validation=validation_table(test,['maxwent_sd']);validation.to_csv(out/'test_score_validation.csv',index=False)
    test.to_csv(out/'test_scores.csv',index=False)
    write_json(out/'DONE.json',dict(status='pilot_complete',selected_step=best_step,mean_scale=best_entropy,calibration_q95_sd=q,
        highcal_q95_sd=qh,validation_risk_limit=limit,all_means_unchanged=True,point_predictor_unchanged=True,
        fitted_parameters=net.count,scope='conditional regression-head uncertainty, not whole-model posterior',
        seconds=time.monotonic()-start,config=config,max_reserved_gib=torch.cuda.max_memory_reserved()/2**30 if x.is_cuda else None))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent);p.add_argument('--device',default='cuda');p.add_argument('--steps',type=int,default=2000);main(p.parse_args())
