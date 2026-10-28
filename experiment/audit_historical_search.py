"""Extract historical messages and independently count the saved search plan."""
from pathlib import Path
import json,re,hashlib
import sys
import pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
ROOT=Path(__file__).resolve().parent
chat=Path(r'C:\Users\11947\.claude\projects\D-----GFP\b5052361-6e68-4309-b583-f06ab1974ee4.jsonl')
plan=ROOT.parent/'handoff_20261001/repro_project/artifacts/multipeak/multipeak_seed_plan.csv'
hits=[]
pattern=re.compile(r'900.{0,15}(?:峰|山)|2700.{0,15}(?:起点|种子|runs)|wide900|summit.{0,30}ridge.{0,30}edge',re.I)
with chat.open(encoding='utf-8') as f:
    for line_no,line in enumerate(f,1):
        row=json.loads(line);message=row.get('message',{})
        if not isinstance(message,dict):continue
        content=message.get('content',[])
        if isinstance(content,str):content=[dict(type='text',text=content)]
        for item in content:
            if not isinstance(item,dict) or item.get('type')!='text':continue
            text=item.get('text','')
            for m in list(pattern.finditer(text))[:3]:
                hits.append(dict(line=line_no,role=message.get('role'),timestamp=row.get('timestamp'),
                    excerpt=text[max(0,m.start()-150):min(len(text),m.end()+420)]))
d=pd.read_csv(plan)
production=pd.read_csv(ROOT.parent/'LL_GF/cloud_data/v13_5_wide_explore_plan.csv')
info=dict(chat_path=str(chat),chat_sha256=hashlib.sha256(chat.read_bytes()).hexdigest(),
    plan_path=str(plan),plan_sha256=hashlib.sha256(plan.read_bytes()).hexdigest(),
    plan_rows=len(d),mountains=int(d.mountain_id.nunique()),unique_sequences=int(d.sequence.nunique()),
    roles=d.seed_role.value_counts().to_dict(),
    actual_production=dict(rows=len(production),mountains=int(production.mountain_id.nunique()),
        unique_seed_rows=int(production.seed_row.nunique()),
        actual_parameters={c:production[c].unique().tolist() for c in
            ['steps','lr','proposal_candidates','terrain_topk','terrain_risk_weight','amp_dtype']}),
    matched_excerpts=hits)
(ROOT/'audit/historical_wide_search.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in info.items() if k!='matched_excerpts'},ensure_ascii=False))
print('relevant_message_excerpts',len(hits))
for h in hits[-8:]:print(json.dumps(h,ensure_ascii=False))
