from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from pplu_repro import esm_patch  # noqa: F401  fix ESM2 mask_token_id on transformers 4.47
except Exception:
    pass

AA20 = set("ACDEFGHIKLMNPQRSTVWY")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict:
    p = Path(path) if path else repo_root() / "configs" / "experiment.json"
    return json.loads(p.read_text(encoding="utf-8"))


def atomic_write_json(obj: object, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, p)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def validate_sequence(seq: object, expected_length: int | None = None) -> str:
    s = str(seq).strip().upper()
    if expected_length is not None and len(s) != int(expected_length):
        raise ValueError(f"Sequence length {len(s)} != expected {expected_length}")
    bad = set(s) - AA20
    if bad:
        raise ValueError(f"Sequence contains non-canonical residues: {sorted(bad)}")
    return s


def hamming(a: str, b: str) -> int:
    if len(a) != len(b):
        return max(len(a), len(b))
    return sum(x != y for x, y in zip(a, b))


def mutation_count_from_reference(seq: str, reference: str) -> int:
    return hamming(seq, reference)


def ensure_disjoint(*sets_: Iterable[str]) -> bool:
    ss = [set(x) for x in sets_]
    for i in range(len(ss)):
        for j in range(i + 1, len(ss)):
            if ss[i] & ss[j]:
                return False
    return True


def set_all_seeds(seed: int) -> None:
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def load_reference_sequence(root: str | Path | None = None) -> str:
    r = Path(root) if root else repo_root()
    p = r / "data" / "processed" / "reference_sequence.txt"
    if not p.is_file():
        raise FileNotFoundError(f"Reference sequence not prepared: {p}. Run scripts/01_prepare_data.sh first.")
    return validate_sequence(p.read_text(encoding="utf-8").strip())
