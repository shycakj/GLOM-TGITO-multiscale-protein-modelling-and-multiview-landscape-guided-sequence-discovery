#!/usr/bin/env bash
set -euo pipefail
B=$(cd -- "$(dirname -- "$0")/.." && pwd)
case "$B" in /xiongyuchen/*) ;; *) echo 'Place the reproduction folder under /xiongyuchen/ persistent storage.' >&2; exit 2;; esac
export CONDA_PKGS_DIRS="$B/cache/conda" MAMBA_ROOT_PREFIX="$B/tools/mamba"
export TMPDIR="$B/tmp" XDG_CACHE_HOME="$B/cache" MPLCONFIGDIR="$B/cache/matplotlib"
export OPENMM_CPU_THREADS=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
mkdir -p "$B/tmp" "$B/cache" "$B/logs"
if [ ! -x "$B/env/bin/python" ]; then
  /opt/conda/bin/mamba create -y -p "$B/env" --file "$B/tools/conda_explicit.txt"
fi
cd "$B"
# Prepared, audited coordinates/topologies are supplied. This preserves the
# identical initial solvent/ion arrangement instead of drawing new ion positions.
"$B/env/bin/python" - <<'PY'
import hashlib,json,pathlib
r=pathlib.Path('.')
m=json.loads((r/'LOCAL_BACKUP_MANIFEST.json').read_text())
keys=[k for k in m if k.startswith('inputs/') or k.startswith('structures/')]
for k in keys:
 p=r/k
 assert p.exists() and hashlib.sha256(p.read_bytes()).hexdigest()==m[k],k
print('Verified prepared inputs:',len(keys))
PY
"$B/env/bin/python" -u source/md_run.py WT --gpu 0 --rep 97 --ns .1 --equil-ns .2 > logs/reproduction_preflight_WT.log 2>&1 &
pid_wt=$!
"$B/env/bin/python" -u source/md_run.py C01 --gpu 1 --rep 97 --ns .1 --equil-ns .2 > logs/reproduction_preflight_C01.log 2>&1 &
pid_mt=$!
wait "$pid_wt"
wait "$pid_mt"
"$B/env/bin/python" source/analyze_md.py WT --rep 97 > logs/reproduction_preflight_analysis_WT.log 2>&1
"$B/env/bin/python" source/analyze_md.py C01 --rep 97 > logs/reproduction_preflight_analysis_C01.log 2>&1
"$B/env/bin/python" - <<'PY'
import json,pathlib,pandas as pd
r=pathlib.Path('.')
for sid in ['WT','C01']:
 m=json.loads((r/'results'/sid/'rep97/METRICS.json').read_text())['metrics']
 assert m['ca_rmsd_mean_A']<4 and m['beta_native_contact_fraction']>.8
 temp=pd.read_csv(r/'md'/sid/'rep97/production.csv')['Temperature (K)'].mean()
 assert 280<temp<320
(r/'PREFLIGHT_PASSED.json').write_text(json.dumps({'passed':True,'fresh_preflight':['WT/rep97','C01/rep97'],
 'formal_replicates':3,'production_ns':3,'equilibration_ns':.5},indent=2))
PY
exec "$B/env/bin/python" -u source/run_campaign.py
