"""Native SaProt features, with audited predicted 3Di inputs and resume guards."""
from pathlib import Path
import argparse, hashlib, json, os, sys, time
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from experiment_core import write_json
from native_saprot import check_loaded_weights

def sha(p):
    with p.open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()

def joint_sequences(sequences, states, mapping):
    """Index zero is padding, never silently interpreted as an amino acid."""
    result = []
    for seq, row in zip(sequences, states):
        assert len(seq) == len(row) == 221
        chars = [mapping[int(i)] for i in row]
        chars = ['#' if x == '<pad>' else x.lower() for x in chars]
        assert all(len(c) == 1 and c in 'acdefghiklmnpqrstvwy#' for c in chars)
        result.append(''.join(a+s for a, s in zip(seq, chars)))
    assert len(result) == len(sequences)
    return result

def checked_tokenize(tok, joints):
    batch = tok(joints, return_tensors='pt', padding=True)
    assert batch['input_ids'].shape == (len(joints), 223)
    assert (batch['attention_mask'].sum(1) == 223).all()
    assert not (batch['input_ids'] == tok.unk_token_id).any()
    for row, seq in zip(batch['input_ids'].tolist(), joints):
        expected = [seq[i:i+2] for i in range(0, len(seq), 2)]
        assert tok.convert_ids_to_tokens(row[1:-1]) == expected
    return batch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['3di', 'saprot'], required=True)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    import torch
    from transformers import AutoTokenizer, EsmModel
    assert torch.cuda.is_available(), 'GPU extraction required; no full CPU fallback'
    torch.set_num_threads(3)
    spec = json.loads((HERE/'PROTOCOL.json').read_text())
    project = ROOT/'repro_project'
    os.environ.update(HF_HOME=str(ROOT/'cache/huggingface'),
                      TORCH_HOME=str(ROOT/'cache/torch'),
                      PPLU_ESM_MODEL_PATH=str(project/'assets/esm2_t30_150M_UR50D'))
    out = ROOT/'artifacts/embeddings'
    out.mkdir(parents=True, exist_ok=True)
    raw = project/'data/raw/pplu_31480_sequences.csv'
    seqs = pd.read_csv(raw, usecols=['sequence']).sequence
    if args.smoke: seqs = seqs.head(8)
    n = len(seqs)
    assert n == (8 if args.smoke else 31480)
    seqhash = hashlib.sha256('\n'.join(seqs).encode()).hexdigest()
    if not args.smoke:
        original_cache=json.loads((HERE/'reference_glom_cache_state.json').read_text())
        assert seqhash==original_cache['sequence_order_sha256'], 'GLOM/SaProt row-order mismatch'
    tag = spec['feature'] + ('_smoke' if args.smoke else '')
    bridge_path = project/'assets/warm_best_model.pt'
    assert sha(bridge_path) == spec['bridge_sha256']
    statepath = out/(tag+('_3di_state.json' if args.stage == '3di' else '_state.json'))
    base = {'stage': args.stage, 'rows': n, 'sequence_sha256': seqhash,
            'protocol_sha256': sha(HERE/'PROTOCOL.json'), 'labels_read': False,
            'bridge_sha256': spec['bridge_sha256'], 'completed_rows': 0}
    previous = json.loads(statepath.read_text()) if statepath.exists() else base
    for key in ['stage','rows','sequence_sha256','protocol_sha256','bridge_sha256']:
        assert previous[key] == base[key], key
    start = int(previous['completed_rows'])
    if start == n:
        print('CACHE_ALREADY_COMPLETE'); return
    if args.stage == '3di':
        sys.path.insert(0, str(project))
        from pipeline.tgito.esm_bridge import load_bridge
        from pipeline.tgito.aa import seq_to_ids0
        from encoder_contract import TrainingCompatibleEncoder
        checkpoint = torch.load(bridge_path, map_location='cpu', weights_only=False)
        vocab = checkpoint['vocab']['idx_to_char']
        mapping = {int(k): v for k, v in vocab.items()}
        assert set(mapping) == set(range(21)) and mapping[0] == '<pad>'
        write_json(out/(tag+'_3di_vocab.json'), mapping)
        del checkpoint
        encoder = TrainingCompatibleEncoder(load_bridge(bridge_path, device='cuda'), device='cuda').eval()
        parity = encoder.audit_native_hard(seqs.head(3).tolist())
        write_json(out/(tag+'_encoder_parity.json'), parity)
        arr = np.lib.format.open_memmap(out/(tag+'_3di.npy'), mode='r+' if start else 'w+',
                                       dtype='uint8', shape=(n,221))
        def step(begin, count):
            ids = torch.stack([seq_to_ids0(s,221,'cuda') for s in seqs.iloc[begin:begin+count]])
            with torch.inference_mode():
                q = encoder(torch.nn.functional.one_hot(ids,20).float())['q3di']
            assert torch.isfinite(q).all() and q.shape[-1] == 21
            return q.argmax(-1).cpu().numpy().astype('uint8')
    else:
        source = json.loads((out/(tag+'_3di_state.json')).read_text())
        assert source['completed_rows'] == source['rows'] == n
        assert source['sequence_sha256'] == seqhash and source['protocol_sha256'] == base['protocol_sha256']
        mapping = {int(k):v for k,v in json.loads((out/(tag+'_3di_vocab.json')).read_text()).items()}
        states = np.load(out/(tag+'_3di.npy'), mmap_mode='r')
        modelpath = project/'assets/saprot_650m_af2'
        manifest = json.loads((modelpath/'DOWNLOAD_VERIFIED.json').read_text())
        assert manifest['revision'] == spec['model_revision'] and manifest['weights_downloaded']
        for rec in manifest['files']:
            assert sha(modelpath/rec['file']) == rec['sha256'], rec['file']
        tok = AutoTokenizer.from_pretrained(modelpath, local_files_only=True)
        model, info = EsmModel.from_pretrained(modelpath, add_pooling_layer=False, local_files_only=True,
                                             torch_dtype=torch.bfloat16, output_loading_info=True)
        loading_check = check_loaded_weights(model, info)
        assert model.config.hidden_size == 1280 and model.config.num_hidden_layers == 33
        model = model.cuda().eval()
        arr = np.lib.format.open_memmap(out/(tag+'_mean.npy'), mode='r+' if start else 'w+',
                                       dtype='float32', shape=(n,1280))
        ids_arr = np.lib.format.open_memmap(out/(tag+'_input_ids.npy'), mode='r+' if start else 'w+',
                                           dtype='int64', shape=(n,223))
        def step(begin,count):
            joints = joint_sequences(seqs.iloc[begin:begin+count].tolist(), states[begin:begin+count], mapping)
            batch = checked_tokenize(tok, joints)
            ids_arr[begin:begin+len(joints)] = batch['input_ids'].numpy()
            with torch.inference_mode():
                h = model(**{k:v.cuda() for k,v in batch.items()}).last_hidden_state[:,1:222].float()
            assert h.shape[1:] == (221,1280) and torch.isfinite(h).all()
            return h.mean(1).cpu().numpy()
        together = step(0,2)
        separately = np.concatenate([step(0,1),step(1,1)])
        # Relative RMS is robust to small absolute near-zero activation values.
        rel = float(np.sqrt(np.mean((together-separately)**2))/max(1e-8,np.sqrt(np.mean(together**2))))
        assert rel < .02, f'Batch-parity failure {rel}'
        write_json(out/(tag+'_native_parity.json'), {'relative_rms':rel,'threshold':.02,
                   'tokens_per_sequence':223,'native_layers':33,'residue_dim':1280,
                   'loading_check': loading_check, 'unknown_tokens':0,
                   'padding_3di_fraction':float((states==0).mean())})
        base.update(dim=1280, model_manifest=manifest, precision='native bf16; mean/storage float32',
                    token_input='amino-acid plus predicted 3Di, no GLOM mutation features')
    b = args.batch_size; pos = start; began = last = time.monotonic()
    while pos < n:
        try:
            values = step(pos,min(b,n-pos)); arr[pos:pos+len(values)] = values
            pos += len(values); del values
        except torch.cuda.OutOfMemoryError:
            if b == 1: raise
            b = max(1,b//2); torch.cuda.empty_cache()
            print(json.dumps({'oom_backoff_batch':b}),flush=True); continue
        if time.monotonic()-last > 20 or pos == n:
            arr.flush()
            if args.stage == 'saprot': ids_arr.flush()
            base.update(completed_rows=pos,batch_size=b,elapsed_seconds=time.monotonic()-began,
                        max_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
            write_json(statepath,base); print(json.dumps({k:base[k] for k in
                       ['stage','completed_rows','rows','batch_size','max_reserved_gib']}),flush=True)
            last = time.monotonic()
    print('SAPROT_CACHE_COMPLETE')

if __name__ == '__main__': main()
