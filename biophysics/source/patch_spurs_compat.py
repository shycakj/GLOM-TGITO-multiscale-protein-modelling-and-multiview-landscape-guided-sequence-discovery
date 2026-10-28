"""Reapply Python 3.12/Lightning compatibility fixes without changing model math."""
from pathlib import Path
import re,subprocess
root=Path(__file__).resolve().parents[1]/'tools/SPURS'
p=root/'spurs/utils/__init__.py';s=p.read_text()
s=s.replace('from pytorch_lightning.loggers import LightningLoggerBase, TensorBoardLogger',
            'from pytorch_lightning.loggers import TensorBoardLogger\nfrom pytorch_lightning.loggers.logger import Logger as LightningLoggerBase')
s=s.replace('pl.loggers.LightningLoggerBase','LightningLoggerBase')
s=s.replace('from . import strategies','# Training-only strategy registration is not needed for inference.')
p.write_text(s)
for p in root.rglob('*.py'):
    s=p.read_text();t=re.sub(r'field\(default=([A-Za-z_][A-Za-z_0-9]*)\(\)\)',r'field(default_factory=\1)',s)
    if t!=s:p.write_text(t)
p=root/'spurs/__init__.py';s=p.read_text().replace('import spurs.tasks','# Training tasks are not loaded by the inference runner.');p.write_text(s)
for rel in ['spurs/models/__init__.py','spurs/datamodules/__init__.py']:
    p=root/rel;s=p.read_text();s='\n'.join('# Inference-only: '+l if l.startswith('import_modules(') else l for l in s.split('\n'));p.write_text(s)
diff=subprocess.check_output(['git','diff'],cwd=root,text=True)
(root.parent/'SPURS_inference_compatibility.patch').write_text(diff)
(root.parent/'SPURS_commit.txt').write_text(subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True))
print('Compatibility patch recorded; model forward computations unchanged.')
