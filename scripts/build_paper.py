"""Compile the included final manuscript and supplement in a separate build folder."""
from pathlib import Path
import argparse,shutil,subprocess
R=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--tectonic');p.add_argument('--latexmk',default='latexmk');a=p.parse_args()
for part in ['bioinformatics','supplement']:
    target=R/'reproduced/paper'/part
    shutil.copytree(R/'paper/latex'/part,target,dirs_exist_ok=True)
    if a.tectonic:cmd=[a.tectonic,'--keep-logs','--keep-intermediates','main.tex']
    else:cmd=[a.latexmk,'-xelatex','-interaction=nonstopmode','-halt-on-error','main.tex']
    with (target/'build_stdout.log').open('wb') as f:result=subprocess.run(cmd,cwd=target,stdout=f,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError('Build failed; see '+str(target/'build_stdout.log'))
    print('Built',target/'main.pdf',flush=True)
