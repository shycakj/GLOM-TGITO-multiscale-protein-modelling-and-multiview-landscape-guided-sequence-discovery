"""Extract relevant P4 construction provenance without printing unrelated chat content."""
from pathlib import Path
import json,re
path=Path("C:/Users/11947/.claude/projects/D-----GFP/b5052361-6e68-4309-b583-f06ab1974ee4.jsonl")
out=[]
for lineno,line in enumerate(path.open(encoding="utf-8"),1):
    if "pplu_pmutation_p4" not in line:continue
    item=json.loads(line);message=item.get("message",{})
    content=message.get("content",[])
    if isinstance(content,str):continue
    for block in content:
        if block.get("type") not in {"tool_use","tool_result","text"}:continue
        value=json.dumps(block,ensure_ascii=False)
        if "pplu_pmutation_p4" not in value:continue
        for key in ["AAP","aa_bits","p4 =","p4[","P4 =","P4[","np.save","pmutation"]:
            position=value.find(key)
            if position>=0:
                snippet=value[max(0,position-180):position+2500]
                out.append({"line":lineno,"timestamp":item.get("timestamp"),"type":block.get("type"),"key":key,"snippet":snippet})
                break
target=Path(__file__).resolve().parent/"audit/p4_chat_provenance.json"
target.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
for r in out:
    if r["key"]!="pmutation":print(json.dumps(r,ensure_ascii=False))
print("saved",len(out))

