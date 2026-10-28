"""Check fixed split, seed plan, frozen assets and local environment without training."""
from __future__ import annotations
import argparse, csv, hashlib, importlib.metadata, json, pathlib, sys, os

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=pathlib.Path, default=pathlib.Path(__file__).resolve().parents[1])
    ap.add_argument('--hash-all', action='store_true', help='Read every asset in ASSET_COPY_MANIFEST; can take several minutes.')
    ap.add_argument('--cpu-model-smoke', action='store_true')
    ap.add_argument('--sequence-model-smoke', action='store_true', help='CPU WT ESM2→Bridge→GLOM functionality check using only local packaged assets.')
    ap.add_argument('--mode', choices=['assets','glom','baseline'], default='glom', help='assets checks data/files only; glom or baseline also checks that environment.')
    a = ap.parse_args(); root = a.root.resolve(); issues = []; checks = {}
    manifest = json.loads((root/'ASSET_COPY_MANIFEST.json').read_text(encoding='utf-8'))
    for e in manifest['asset_entries']:
        p = root/e['destination']
        if not p.is_file(): issues.append('missing: '+e['destination'])
        elif p.stat().st_size != e['bytes']: issues.append('size: '+e['destination'])
        elif a.hash_all and e.get('source_sha256') and digest(p) != e['source_sha256']:
            issues.append('hash: '+e['destination'])
    hashes_path=root/'ASSET_HASHES.json'
    if a.hash_all and hashes_path.is_file():
        for e in json.loads(hashes_path.read_text(encoding='utf-8')):
            p=root/e['path']
            if p.is_file() and digest(p)!=e['sha256']:issues.append('asset hash: '+e['path'])
    expected = {'glom_train':27656, 'glom_val':3078, 'tgito_tune_300':300}
    sets = {}
    for name, n in expected.items():
        p = root/'data/processed/search'/(name+'.csv')
        if not p.is_file(): continue
        d = rows(p); sets[name] = {r['sequence'] for r in d}
        checks[name+'_rows'] = len(d)
        if len(d) != n: issues.append('split count: '+name)
        if len(d) != len(sets[name]): issues.append('duplicate sequence: '+name)
        if any(len(s)!=221 or set(s)-set('ACDEFGHIKLMNPQRSTVWY') for s in sets[name]): issues.append('invalid sequence: '+name)
    hidden_path = root/'data/sealed/search/hidden_top1500_400_with_labels.csv'
    if hidden_path.is_file():
        hd = rows(hidden_path); sets['hidden400'] = {r['sequence'] for r in hd}; checks['hidden_rows'] = len(hd)
        if len(hd)!=400: issues.append('hidden count')
    for i, (n,s) in enumerate(sets.items()):
        for m,t in list(sets.items())[i+1:]:
            if s&t: issues.append('overlap: '+n+' / '+m)
    op = root/'data/processed/search/observed_all.csv'
    if op.is_file():
        obs = rows(op); observed_seqs = {r['sequence'] for r in obs}; checks['observed_rows'] = len(obs)
        if len(obs)!=31034: issues.append('observed count')
        if all(k in sets for k in expected) and observed_seqs != set().union(*(sets[k] for k in expected)): issues.append('observed union mismatch')
    plan_path = root/'artifacts/multipeak/multipeak_seed_plan.csv'
    if plan_path.is_file():
        plan = rows(plan_path); checks['starts'] = len(plan); checks['unique_start_sequences'] = len({r['sequence'] for r in plan})
        roles = {k:sum(r['seed_role']==k for r in plan) for k in ['summit','ridge','edge']}; checks['roles'] = roles
        if len(plan)!=2700 or len({r['start_id'] for r in plan})!=2700 or roles!={'summit':900,'ridge':900,'edge':900}: issues.append('seed plan')
    expected_hashes = {
        'artifacts/glom_current/best_model_pplu_current_split.pt':'f7da41c4a5bb178e10274aad4ca80d7305baa6a6d1032a25641d35c8646fb95a',
        'assets/warm_best_model.pt':'1dcc0d4031184d300c3fd1f8e0fd956626947fd81d71f65265c4474b5bcbb292',
        'artifacts/tgito_hpo/FROZEN_TGITO_PARAMS.json':'129297379b08852d26cd954297cff5ee3525ff0b63ad0fbac039790bce478609',
        'artifacts/multipeak/multipeak_seed_plan.csv':'adc472a10b48630b6b53a9e269b2b45bc6ac6bae6024836648d6aa025b95022f'
    }
    for name,want in expected_hashes.items():
        p=root/name
        if p.is_file():
            got=digest(p); checks[name+'_sha256']=got
            if got!=want: issues.append('frozen hash: '+name)
    versions={}
    for package in ['torch','transformers','numpy','pandas','scipy','scikit-learn','optuna','umap-learn','gym','stable-baselines3','flexs']:
        try: versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: versions[package]=None
    checks['environment']={'python':sys.version,'packages':versions}
    expected_env={
        'glom':{'torch':'2.5.1','transformers':'4.47.0','numpy':'1.26.4','pandas':'2.3.3'},
        'baseline':{'torch':'1.11.0','numpy':'1.21.5','pandas':'1.5.3','gym':'0.22.0','flexs':'0.2.1','stable-baselines3':'1.7.0a1'}
    }
    for pkg,want in expected_env.get(a.mode,{}).items():
        actual=versions[pkg]
        if actual is None or actual.split('+')[0]!=want:issues.append('environment: '+pkg+' expected '+want+' got '+str(actual))
    if a.mode!='assets':
        want_python=(3,10) if a.mode=='glom' else (3,8)
        if sys.version_info[:2]!=want_python:issues.append('environment: expected Python '+'.'.join(map(str,want_python)))
    if a.cpu_model_smoke:
        sys.path.insert(0,str(root/'pipeline'))
        import torch
        from tgito.revised_slim_internal import load_revised_slim
        model,ym,ys=load_revised_slim(root/'artifacts/glom_current/best_model_pplu_current_split.pt',device='cpu')
        model.eval()
        with torch.inference_mode(): y=model(torch.zeros(1,221,1280),torch.zeros(1,221,10))*ys+ym
        if not bool(torch.isfinite(y).all()): issues.append('CPU model forward nonfinite')
        checks['synthetic_cpu_forward']={'shape':list(y.shape),'finite':bool(torch.isfinite(y).all()),'parameters':sum(p.numel() for p in model.parameters())}
    if a.sequence_model_smoke:
        os.environ['PPLU_ESM_MODEL_PATH']=str(root/'assets/esm2_t30_150M_UR50D')
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        sys.path.insert(0,str(root/'pipeline'))
        import torch
        torch.set_num_threads(min(4,torch.get_num_threads()))
        from tgito.aa import aa_bits_tensor,seq_to_ids0,soft_pos_feats
        from tgito.esm_bridge import load_bridge,DifferentiableESMBridge
        from tgito.revised_slim_internal import load_revised_slim
        reference=(root/'data/processed/reference_sequence.txt').read_text(encoding='utf-8').strip()
        bridge=load_bridge(root/'assets/warm_best_model.pt',device='cpu')
        encoder=DifferentiableESMBridge(bridge,esm_model_name='auto',device='cpu',freeze_esm=True,target_h_dim=1280).eval()
        glom,ym,ys=load_revised_slim(root/'artifacts/glom_current/best_model_pplu_current_split.pt',device='cpu');glom.eval()
        ids=seq_to_ids0(reference,len(reference),device='cpu')
        with torch.inference_mode():
            probs=torch.nn.functional.one_hot(ids,20).float().unsqueeze(0)
            H=encoder(probs)['H'];P=soft_pos_feats(probs,ids,aa_bits_tensor(device='cpu'))
            y=glom(H,P)*ys+ym
        finite=bool(torch.isfinite(y).all())
        if not finite:issues.append('sequence-to-brightness CPU forward nonfinite')
        checks['offline_WT_sequence_forward']={'pipeline':'packaged ESM2-150M → Bridge saprot_hat → GLOM','H_shape':list(H.shape),'P_shape':list(P.shape),'predicted_log10_fluorescence':float(y.item()),'finite':finite,'device':'cpu','scope':'Functionality check; numerical equality to historical CUDA/torch2.5.1/transformers4.47 not inferred from a different environment.'}
    report={'ok':not issues,'mode':a.mode,'root':str(root),'checks':checks,'issues':issues,'archived_H_present':(root/'assets/pplu_31k_H_embeddings.npy').is_file()}
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if issues: raise SystemExit(1)

if __name__=='__main__': main()
