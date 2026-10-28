"""Fetch only the pinned native SaProt HF snapshot and verify its LFS hash."""
from pathlib import Path
import argparse, hashlib, json, os, time, urllib.request

HERE = Path(__file__).resolve().parent
FILES = ['config.json', 'tokenizer_config.json', 'special_tokens_map.json',
         'vocab.txt', 'pytorch_model.bin']

def digest(p):
    with p.open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--metadata-only', action='store_true')
    args = ap.parse_args()
    spec = json.loads((HERE/'PROTOCOL.json').read_text())
    repo, revision = spec['model_repo'], spec['model_revision']
    url = f'https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true'
    with urllib.request.urlopen(url, timeout=45) as r: info = json.load(r)
    assert info['sha'] == revision
    records = {x['rfilename']: x for x in info['siblings']}
    for name in FILES: assert name in records, name
    weight_hash = records['pytorch_model.bin']['lfs']['sha256']
    if args.metadata_only:
        dest = HERE/'model_metadata'
    else:
        dest = HERE.parent/'repro_project/assets/saprot_650m_af2'
    dest.mkdir(parents=True, exist_ok=True)
    output = []
    for name in FILES:
        if args.metadata_only and name.endswith('.bin'): continue
        p = dest/name
        expected_size = records[name]['size']
        expected_hash = records[name].get('lfs', {}).get('sha256')
        if p.exists():
            assert p.stat().st_size == expected_size, f'Wrong cached size: {name}'
            if expected_hash: assert digest(p) == expected_hash, name
        else:
            partial = p.with_suffix(p.suffix+'.part')
            with urllib.request.urlopen(f'https://huggingface.co/{repo}/resolve/{revision}/{name}', timeout=120) as r, partial.open('wb') as f:
                written = 0; last = time.monotonic()
                while block := r.read(4 << 20):
                    f.write(block); written += len(block)
                    if time.monotonic()-last > 20:
                        print(json.dumps({'file': name, 'bytes': written}), flush=True)
                        last = time.monotonic()
                f.flush(); os.fsync(f.fileno())
            assert partial.stat().st_size == expected_size, name
            if expected_hash: assert digest(partial) == expected_hash, name
            partial.replace(p)
        output.append({'file': name, 'bytes': p.stat().st_size, 'sha256': digest(p)})
    record = {'repo': repo, 'revision': revision, 'files': output,
              'expected_weight_sha256': weight_hash, 'weights_downloaded': not args.metadata_only}
    target = 'METADATA_VERIFIED.json' if args.metadata_only else 'DOWNLOAD_VERIFIED.json'
    (dest/target).write_text(json.dumps(record, indent=2)+'\n', encoding='utf8')
    print(json.dumps({'status': 'PASS', 'metadata_only': args.metadata_only,
                      'revision': revision, 'files': len(output)}))

if __name__ == '__main__': main()
