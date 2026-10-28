from __future__ import annotations
import argparse, json, sys, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from pplu_repro.common import load_reference_sequence, validate_sequence

class Scorer:
    def __init__(self,pipeline,bridge_ckpt,revised_ckpt,root,batch=32,device="cuda"):
        sys.path.insert(0,str(Path(pipeline)))
        from tgito.aa import aa_bits_tensor,seq_to_ids0,soft_pos_feats
        from tgito.esm_bridge import load_bridge,DifferentiableESMBridge
        from tgito.revised_slim_internal import load_revised_slim
        if device == "cuda" and not torch.cuda.is_available(): raise RuntimeError('CUDA required for GLOM oracle; use --device cpu for functionality smoke only')
        self.dev=device; self.batch=batch; self.seq_to_ids0=seq_to_ids0; self.soft_pos_feats=soft_pos_feats; self.reference=load_reference_sequence(root); self.L=len(self.reference)
        self.bits=aa_bits_tensor(device=self.dev); self.ref=seq_to_ids0(self.reference,length=self.L,device=self.dev)
        b=load_bridge(bridge_ckpt,device=self.dev); self.enc=DifferentiableESMBridge(b,esm_model_name='auto',device=self.dev,freeze_esm=True,target_h_dim=1280).to(self.dev).eval()
        self.model,self.mean,self.std=load_revised_slim(revised_ckpt,device=self.dev); self.lock=threading.Lock(); self.calls=0; self.seqs=0
    def score(self,seqs):
        seqs=[validate_sequence(s,self.L) for s in seqs]; out=[]
        with self.lock, torch.inference_mode():
            for st in range(0,len(seqs),self.batch):
                ss=seqs[st:st+self.batch]; ids=torch.stack([self.seq_to_ids0(s,self.L,device=self.dev) for s in ss]); probs=F.one_hot(ids,20).float()
                H=self.enc(probs)['H']; P=self.soft_pos_feats(probs,self.ref,self.bits); y=self.model(H,P)*self.std+self.mean; out.extend(y.cpu().numpy().astype(float).tolist())
            self.calls+=1; self.seqs+=len(seqs)
        return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',default=str(ROOT)); ap.add_argument('--pipeline-root',required=True); ap.add_argument('--bridge-ckpt',required=True); ap.add_argument('--revised-ckpt',required=True); ap.add_argument('--host',default='127.0.0.1'); ap.add_argument('--port',type=int,default=8765); ap.add_argument('--batch',type=int,default=32); ap.add_argument('--device',choices=['cuda','cpu'],default='cuda'); args=ap.parse_args()
    scorer=Scorer(args.pipeline_root,args.bridge_ckpt,args.revised_ckpt,args.root,args.batch,args.device)
    class H(BaseHTTPRequestHandler):
        def log_message(self,fmt,*a): return
        def sendj(self,obj,code=200):
            b=json.dumps(obj).encode(); self.send_response(code); self.send_header('Content-Type','application/json'); self.send_header('Content-Length',str(len(b))); self.end_headers(); self.wfile.write(b)
        def do_GET(self):
            if self.path=='/health': self.sendj({'ok':True,'calls':scorer.calls,'sequences_scored':scorer.seqs})
            else: self.sendj({'error':'not found'},404)
        def do_POST(self):
            if self.path!='/score': return self.sendj({'error':'not found'},404)
            try:
                n=int(self.headers.get('Content-Length','0')); obj=json.loads(self.rfile.read(n)); seqs=obj['sequences']; self.sendj({'scores':scorer.score(seqs)})
            except Exception as e: self.sendj({'error':repr(e)},400)
    print(f'[oracle] http://{args.host}:{args.port}',flush=True); ThreadingHTTPServer((args.host,args.port),H).serve_forever()
if __name__=='__main__': main()
