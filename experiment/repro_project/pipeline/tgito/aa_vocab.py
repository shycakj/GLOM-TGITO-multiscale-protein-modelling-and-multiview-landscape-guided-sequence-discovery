from __future__ import annotations

AA20 = "ACDEFGHIKLMNPQRSTVWY"


def aa_vocab() -> dict[str, int]:
    # 0 is reserved for pad/unknown compatibility with SaProt joint tables.
    return {aa: i + 1 for i, aa in enumerate(AA20)}


def aa_index0() -> dict[str, int]:
    return {aa: i for i, aa in enumerate(AA20)}
