from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .common import ensure_disjoint, load_config


def seqs(path: Path) -> set[str]:
    d = pd.read_csv(path)
    if "sequence" not in d.columns:
        raise ValueError(f"No sequence column in {path}")
    return set(d.sequence.astype(str))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--data-root", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    root = Path(args.data_root) if args.data_root else Path(__file__).resolve().parents[2] / "data"
    p = root / "processed"
    s = root / "sealed"

    tr, va, te = seqs(p / "prediction/train.csv"), seqs(p / "prediction/val.csv"), seqs(s / "prediction/test_with_labels.csv")
    assert ensure_disjoint(tr, va, te), "Prediction train/val/test overlap"

    gtr = seqs(p / "search/glom_train.csv")
    gva = seqs(p / "search/glom_val.csv")
    tune = seqs(p / "search/tgito_tune_300.csv")
    hid = seqs(s / "search/hidden_top1500_400_with_labels.csv")
    gh = seqs(p / "search/glom_high_800.csv")
    observed = seqs(p / "search/observed_all.csv")
    seeds = seqs(p / "search/common_seed_pool.csv")
    assert ensure_disjoint(gtr, gva, tune, hid), "Search GLOM train/val/tune/hidden overlap"
    # Expected totals come from the protocol manifest (31480 synthetic -> 31080;
    # real data -> 31034 after the 46 beyond-window twin rows are dropped).
    import json as _json
    manifest = _json.loads((root / "audit" / "protocol_manifest.json").read_text(encoding="utf-8"))
    sc = manifest["search_counts"]
    assert len(hid) == 400 and len(tune) == 300 and len(gh) == 800
    assert len(observed) == int(sc["observed_all"]) and len(gtr) + len(gva) == int(sc["glom_train"]) + int(sc["glom_val"])
    assert not (hid & observed), "Hidden appears in observed search data"
    assert seeds <= gh, "Common seed pool must be a subset of GLOM-high 800"

    hd = pd.read_csv(s / "search/hidden_top1500_400_with_labels.csv")
    counts = hd.groupby("rank_stratum").size().to_dict()
    assert len(counts) == int(cfg["search_task"]["num_strata"])
    assert all(v == int(cfg["search_task"]["hidden_per_stratum"]) for v in counts.values())

    out = {
        "prediction": {"train": len(tr), "val": len(va), "test": len(te), "disjoint": True},
        "search": {
            "glom_train": len(gtr), "glom_val": len(gva), "tgito_tune": len(tune),
            "hidden": len(hid), "observed_all": len(observed), "seed_pool": len(seeds),
            "hidden_per_rank_stratum": counts, "disjoint": True,
        },
        "status": "PASS",
    }
    out_path = root / "audit" / "leakage_audit.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
