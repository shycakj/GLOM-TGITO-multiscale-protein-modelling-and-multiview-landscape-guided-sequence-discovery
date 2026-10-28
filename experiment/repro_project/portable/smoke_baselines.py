"""Toy-landscape check only: not a biological result and does not call GLOM."""
from __future__ import annotations
import argparse, pathlib, sys, json
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'search'))
sys.path.insert(0,str(ROOT/'external/Mu-Protein/mu-search/src/flexs'))
sys.path.insert(0,str(ROOT/'external/Mu-Protein/mu-search/src/flexs/flexs/baselines/explorers/stable_baselines3'))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--methods',nargs='+',default=['pure_random','adalead','evobo']);ap.add_argument('--import-only',action='store_true');ap.add_argument('--allow-new-pandas-smoke',action='store_true',help='Toy check only: adapt removed DataFrame.append on a host with pandas>=2; real baselines retain pandas1.5.3.');a=ap.parse_args()
    import numpy as np
    import pandas as pd
    host_compatibility=False
    if a.allow_new_pandas_smoke and not hasattr(pd.DataFrame,'append'):
        def append_compat(self,other,ignore_index=False,verify_integrity=False,sort=False):
            if isinstance(other,dict):other=pd.DataFrame([other])
            elif isinstance(other,pd.Series):other=other.to_frame().T
            return pd.concat([self,other],ignore_index=ignore_index,verify_integrity=verify_integrity,sort=sort)
        pd.DataFrame.append=append_compat
        host_compatibility=True
    import flexs
    from flexs import baselines
    from run_flexs_multipeak import make
    names={'pure_random':'PureRandom','adalead':'Adalead','evobo':'BO','musearch':'MuSearch'}
    result=[]
    class Toy(flexs.Landscape):
        def __init__(self):super().__init__('portable_toy_only')
        def _fitness_function(self,sequences):return np.asarray([s.count('W')/len(s)+1 for s in sequences],float)
    for name in a.methods:
        cls=getattr(baselines.explorers,names[name])
        row={'method':name,'class':cls.__name__,'imported':True}
        if not a.import_only:
            from pplu_repro.common import set_all_seeds
            set_all_seeds(20260902)
            # Use a small synthetic sequence and budget solely to test dispatch.
            exp=make(name,Toy(),'ACDEFGHIKLMN',batch=4,q=100,rounds=1)
            d,meta=exp.run(Toy(),verbose=False)
            row['rows']=len(d);row['finite']=bool(np.isfinite(d['true_score']).all())
            assert row['finite'] and len(d)>1
        result.append(row)
    print(json.dumps({'scope':'toy_import_and_dispatch_only','host_pandas_compatibility_shim':host_compatibility,'results':result},indent=2))

if __name__=='__main__':main()
