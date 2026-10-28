"""Replace only the waiting supervisor, after verifying no new training jobs exist."""
from pathlib import Path
from datetime import datetime,timezone
import fcntl,hashlib,json,os,signal,subprocess,sys,time,zipfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    status=json.loads((HERE/'QUEUE_STATUS.json').read_text())
    assert status['status']=='WAITING_FOR_EXISTING_CAMPAIGN',status['status']
    assert not status.get('running')
    candidates=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            args=(proc/'cmdline').read_bytes().decode().split('\0')
            cwd=(proc/'cwd').resolve()
        except (FileNotFoundError,PermissionError):continue
        if cwd==ROOT and any(Path(a).name=='run_recipe_queue.py' for a in args):candidates.append(int(proc.name))
    assert len(candidates)==1,candidates
    before={n:sha(HERE/n) for n in ['PROTOCOL.json','train_official_recipe.py','model.py']}
    old=(HERE/'run_recipe_queue.py').read_bytes()
    backup=HERE/'run_recipe_queue_before_slotfill_20261019.txt'
    if backup.exists():assert backup.read_bytes()==old
    else:backup.write_bytes(old)
    (HERE/'QUEUE_STATUS_before_slotfill.json').write_text(json.dumps(status,indent=2))
    oldpid=candidates[0];os.kill(oldpid,signal.SIGTERM)
    # The advisory lock, not disappearance of a possibly unreaped PID, proves exit.
    lock=(HERE/'QUEUE.lock').open('a');acquired=False
    for _ in range(100):
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);acquired=True;break
        except BlockingIOError:time.sleep(.1)
    assert acquired,'Waiting supervisor did not release its lock'
    approved={'run_recipe_queue.py','scheduling_policy.py','SCHEDULING_AUDIT.json','SOURCE_AUDIT.json','README.txt'}
    with zipfile.ZipFile(ROOT/'saprot_recipe_slotfill_update.zip') as z:
        for name in z.namelist():
            target=(ROOT/name).resolve()
            assert target.parent==HERE and target.name in approved,name
        for name in z.namelist():(ROOT/name).write_bytes(z.read(name))
    assert before=={n:sha(HERE/n) for n in before},'Training files changed'
    fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    env=dict(os.environ,HF_HOME=str(ROOT/'cache/huggingface'),TORCH_HOME=str(ROOT/'cache/torch'),TMPDIR=str(ROOT/'tmp'))
    log=(ROOT/'logs/saprot_official_recipe_20261019/queue_slotfill.log').open('a')
    p=subprocess.Popen([sys.executable,'-u',str(HERE/'run_recipe_queue.py')],cwd=ROOT,env=env,
             stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    report={'utc':datetime.now(timezone.utc).isoformat(),'old_waiting_supervisor_pid':oldpid,
            'new_supervisor_pid':p.pid,'original_queue_and_training_processes_not_stopped':True,
            'unchanged_training_files_sha256':before,
            'old_supervisor_termination_expected':'Previous wrapper may log CalledProcessError after intentional SIGTERM; this is not a training failure.',
            'scheduling_rule':'Only after prior pending list is empty; count both queues; at most 2 live training jobs per GPU'}
    (HERE/'SCHEDULING_RESTART.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))

if __name__=='__main__':main()
