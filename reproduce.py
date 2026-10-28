"""Portable entry point for inference, paper-data checks and fresh experiment plans."""
from pathlib import Path
import argparse,subprocess,sys
ROOT=Path(__file__).resolve().parent
MAPPING={'fetch':'fetch_assets.py','verify':'verify_repository.py','predict':'predict_sequences.py',
    'tables':'reproduce_tables.py','statistics':'reproduce_statistics.py','mechanisms':'reproduce_mechanisms.py',
    'figures':'reproduce_figures.py','paper':'build_paper.py','assets':'download_public_assets.py',
    'prepare':'prepare_rerun.py','run':'run_plan.py'}
p=argparse.ArgumentParser();p.add_argument('task',choices=MAPPING)
p.add_argument('arguments',nargs=argparse.REMAINDER);a=p.parse_args()
raise SystemExit(subprocess.run([sys.executable,'-X','utf8',str(ROOT/'scripts'/MAPPING[a.task]),*a.arguments],cwd=ROOT).returncode)
