"""Download a pinned public Hugging Face model on the GPU host; record hashes."""
from pathlib import Path
import argparse,json,urllib.request,hashlib,time
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parent
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--repo",required=True);ap.add_argument("--name",required=True)
    ap.add_argument("--revision",help="Pin the exact revision recorded by the original experiment");a=ap.parse_args()
    base="https://huggingface.co"
    api=base+"/api/models/"+a.repo+("/revision/"+a.revision if a.revision else "")
    info=json.load(urllib.request.urlopen(api,timeout=45))
    rev=info["sha"];dest=ROOT/"repro_project/assets"/a.name;dest.mkdir(parents=True,exist_ok=True)
    # Public model configuration and safetensors only; no remote Python is executed.
    files=[s["rfilename"] for s in info["siblings"] if s["rfilename"].endswith((".json",".safetensors",".txt",".model"))]
    records=[]
    def fetch(name):
        p=dest/name;p.parent.mkdir(parents=True,exist_ok=True)
        url=base+"/"+a.repo+"/resolve/"+rev+"/"+name
        if not p.exists():
            tmp=p.with_name(p.name+".part")
            for attempt in range(4):
                try:
                    with urllib.request.urlopen(url,timeout=120) as inp,tmp.open("wb") as out:
                        count=0;last=time.monotonic()
                        while b:=inp.read(4<<20):
                            out.write(b);count+=len(b)
                            if time.monotonic()-last>20:
                                print(json.dumps({"file":name,"bytes":count}),flush=True);last=time.monotonic()
                    tmp.replace(p);break
                except Exception:
                    if attempt==3:raise
                    time.sleep(3)
        h=hashlib.sha256()
        with p.open("rb") as f:
            while b:=f.read(8<<20):h.update(b)
        return {"file":name,"bytes":p.stat().st_size,"sha256":h.hexdigest()}
    with ThreadPoolExecutor(max_workers=3) as pool:
        records=list(pool.map(fetch,files))
    (dest/"DOWNLOAD_VERIFIED.json").write_text(json.dumps({"repo":a.repo,"revision":rev,"files":records},indent=2))
    print("PUBLIC_MODEL_COMPLETE "+a.repo,flush=True)
if __name__=="__main__":main()

