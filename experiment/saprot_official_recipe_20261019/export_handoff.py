"""Verify actual process identities and snapshot compact metadata for local handoff."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,subprocess,zipfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent

def main():
    current=json.loads((HERE/'QUEUE_STATUS.json').read_text())
    original=json.loads((ROOT/'saprot_baseline_20261019/QUEUE_STATUS.json').read_text())
    processes=[]
    names={'run_recipe_queue.py','run_training_queue.py','train_official_recipe.py','train_saprot_lora.py'}
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:args=(proc/'cmdline').read_bytes().decode().split('\0')
        except (FileNotFoundError,PermissionError):continue
        scripts=[Path(a).name for a in args if Path(a).name in names]
        if scripts:processes.append({'pid':int(proc.name),'scripts':scripts})
    assert any('run_recipe_queue.py' in p['scripts'] for p in processes) or current['status']=='COMPLETE'
    status={'verified_utc':datetime.now(timezone.utc).isoformat(),
            'task_checkpoint_policy':'General pretrained SaProt only; all task labels from project training partitions',
            'original_queue':{k:original[k] for k in ['status','pending','running','completed','failed']},
            'new_queue':current,'verified_processes':processes,
            'gpu_csv':subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.total,memory.used,utilization.gpu',
                        '--format=csv,noheader,nounits'],text=True).splitlines(),
            'pending_previous_task':'Mutation embedding diagram correction remains pending; preserve existing manuscript deliverables.'}
    (HERE/'LIVE_HANDOFF.json').write_text(json.dumps(status,indent=2)+'\n')
    files={}
    for p in HERE.iterdir():
        if p.suffix in {'.py','.json','.txt'}:files[p.relative_to(ROOT).as_posix()]=p.read_bytes()
    for mode in ['aa','pred3di']:
        p=ROOT/f'artifacts/models/random/saprot650_official_{mode}_lora_seed20261013_smoke/DONE.json'
        files[p.relative_to(ROOT).as_posix()]=p.read_bytes()
    hashes={name:hashlib.sha256(raw).hexdigest() for name,raw in files.items()}
    target=HERE/'verified_handoff.zip'
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
        for name,raw in files.items():z.writestr(name,raw)
        z.writestr('OFFICIAL_RECIPE_SNAPSHOT_SHA256.json',json.dumps(hashes,indent=2))
    print(json.dumps({'archive':str(target),'bytes':target.stat().st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
                      'new_queue_status':current['status'],'verified_processes':processes}))

if __name__=='__main__':main()
