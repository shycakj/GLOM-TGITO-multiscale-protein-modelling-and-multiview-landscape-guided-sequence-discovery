from pathlib import Path
import json
import pandas as pd
R=Path(__file__).resolve().parent
for sid in ['WT','C01']:
    m=json.loads((R/'results'/sid/'rep97/METRICS.json').read_text())['metrics']
    assert m['ca_rmsd_mean_A']<4 and m['beta_native_contact_fraction']>.8
    t=pd.read_csv(R/'md'/sid/'rep97/production.csv')['Temperature (K)'].mean()
    assert 280<t<320
(R/'PREFLIGHT_PASSED.json').write_text(json.dumps({'passed':True,'systems':['WT','C01'],'preflight_rep':97}))
print('Fresh MD preflight passed.')
