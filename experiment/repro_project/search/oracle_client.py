from __future__ import annotations
import json
import urllib.request


def score_sequences(base_url: str, sequences: list[str], timeout: float = 300.0) -> list[float]:
    data=json.dumps({'sequences':[str(s) for s in sequences]}).encode('utf-8')
    req=urllib.request.Request(base_url.rstrip('/')+'/score',data=data,headers={'Content-Type':'application/json'},method='POST')
    with urllib.request.urlopen(req,timeout=timeout) as r:
        obj=json.loads(r.read().decode('utf-8'))
    if 'scores' not in obj:
        raise RuntimeError(obj)
    return [float(x) for x in obj['scores']]


def health(base_url: str, timeout: float = 10.0) -> dict:
    with urllib.request.urlopen(base_url.rstrip('/')+'/health',timeout=timeout) as r:
        return json.loads(r.read().decode('utf-8'))
