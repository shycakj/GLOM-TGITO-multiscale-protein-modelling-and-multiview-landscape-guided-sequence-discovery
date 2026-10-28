"""Periodic, detached report generation; automatically picks up newly completed experiments."""
from pathlib import Path
import subprocess,time,os,json
ROOT=Path(__file__).resolve().parent;PY=str(ROOT/"venv/bin/python")
ENV=dict(os.environ,OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",MKL_NUM_THREADS="2")
last=""
for _ in range(576):  # bounded to 48 hours, five-minute refresh
    with (ROOT/"logs/reporting.log").open("a") as f:
        # Score finished scratch heads promptly; the main campaign also skips these completed evaluations.
        for folder in sorted((ROOT/"artifacts/models/random").glob("*")):
            if "smoke" in folder.name or folder.name=="glom_seed20261013":continue
            if not (folder/"DONE.json").exists() or (folder/"metrics.csv").exists() or (folder/"evaluation.json").exists():continue
            free=int(subprocess.check_output(["nvidia-smi","--query-gpu=memory.free","--format=csv,noheader,nounits"],text=True).splitlines()[0])
            if free<4000:break
            subprocess.run([PY,"evaluate_models.py","--run",folder.name],cwd=ROOT,env=ENV,stdout=f,stderr=subprocess.STDOUT)
        rc=subprocess.run([PY,"collect_reports.py"],cwd=ROOT,env=ENV,stdout=f,stderr=subprocess.STDOUT).returncode
        if rc==0:
            base=ROOT/"artifacts/models/random/glom_seed20261013"
            if (base/"risk_screening/DONE.json").exists() and not (base/"local_measured_libraries/DONE.json").exists():
                subprocess.run([PY,"evaluate_local_libraries.py"],cwd=ROOT,env=ENV,stdout=f,stderr=subprocess.STDOUT)
            p=ROOT/"reports/completed_regression_runs.csv"
            current=p.read_text() if p.exists() else ""
            for extra in [base/(name+"/DONE.json") for name in ["representation_analysis","risk_screening","local_measured_libraries","search"]]:
                if extra.exists():current+=str(extra.stat().st_mtime_ns)
            if current and current!=last:
                for script in ["paired_statistics.py","plot_completed_results.py"]:
                    result=subprocess.run([PY,script],cwd=ROOT,env=ENV,stdout=f,stderr=subprocess.STDOUT)
                    print(json.dumps({"script":script,"exit_code":result.returncode,"time":time.time()}),flush=True)
                last=current
    time.sleep(300)
