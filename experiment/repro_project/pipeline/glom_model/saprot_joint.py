from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn

from .aa_vocab import AA20, aa_vocab


@dataclass
class SaProtJointBuildReport:
    model_name: str
    embedding_dim: int
    aa_vocab_size: int
    struct_vocab_size: int
    missing_tokens: list[str]
    used_tokens: int


class SaProtJointEmbeddingTable(nn.Module):
    """Expected SaProt joint-token embedding table.

    For each residue i, given an amino-acid id a_i and a predicted Foldseek 3Di
    distribution q_i(s), this returns

        e_i = sum_s q_i(s) * Emb_SaProt(token=(AA_i, 3Di_s)).

    This fixes the earlier surrogate issue where q3Di was multiplied by a custom
    learned 3Di embedding table. Here the structure token is coupled with the AA
    token through SaProt's own structure-aware vocabulary embedding table.
    """
    def __init__(self, table: torch.Tensor, missing_mask: Optional[torch.Tensor] = None, trainable: bool = False):
        super().__init__()
        # table: (AA_V, S_V, D). aa_id=0 and struct pad column should be zeros.
        if table.ndim != 3:
            raise ValueError(f"Expected table (AA_V,S_V,D), got {tuple(table.shape)}")
        if trainable:
            self.table = nn.Parameter(table.float())
        else:
            self.register_buffer("table", table.float(), persistent=True)
        if missing_mask is None:
            missing_mask = torch.zeros(table.shape[:2], dtype=torch.bool)
        self.register_buffer("missing_mask", missing_mask.bool(), persistent=True)

    @property
    def embedding_dim(self) -> int:
        return int(self.table.shape[-1])

    def forward(self, aa_ids: torch.Tensor, q3di: torch.Tensor) -> torch.Tensor:
        if aa_ids.ndim != 2:
            raise ValueError(f"aa_ids should be (B,L), got {tuple(aa_ids.shape)}")
        if q3di.ndim != 3:
            raise ValueError(f"q3di should be (B,L,S), got {tuple(q3di.shape)}")
        if q3di.shape[:2] != aa_ids.shape:
            raise ValueError(f"aa_ids shape {tuple(aa_ids.shape)} incompatible with q3di {tuple(q3di.shape)}")
        if q3di.shape[-1] != self.table.shape[1]:
            raise ValueError(f"q3di S={q3di.shape[-1]} != table S={self.table.shape[1]}")
        aa_ids = aa_ids.clamp(0, self.table.shape[0] - 1).long()
        # token_emb: (B,L,S,D)
        token_emb = self.table[aa_ids]
        return torch.einsum("bls,blsd->bld", q3di.float(), token_emb.float())


def _get_input_embedding_weight(model) -> torch.Tensor:
    emb = model.get_input_embeddings()
    if emb is None:
        raise ValueError("Loaded SaProt model has no input embeddings")
    return emb.weight.detach().cpu().float()


def _token_id(tokenizer, token: str) -> int:
    # Prefer direct vocabulary lookup because SaProt SA tokens are intended to be atomic tokens.
    try:
        idx = tokenizer.convert_tokens_to_ids(token)
        unk = getattr(tokenizer, "unk_token_id", None)
        if idx is not None and int(idx) >= 0 and (unk is None or int(idx) != int(unk)):
            return int(idx)
    except Exception:
        pass
    vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}
    if token in vocab:
        return int(vocab[token])
    raise KeyError(token)


def build_saprot_joint_embedding_table_from_hf(
    model_name: str,
    struct_vocab: Dict[str, int],
    cache_dir: str | None = None,
    trust_remote_code: bool = True,
) -> tuple[SaProtJointEmbeddingTable, SaProtJointBuildReport]:
    """Load official SaProt embedding table and create AA+3Di expected-embedding lookup.

    Requires transformers and the SaProt HF model locally cached or downloadable.
    The 3Di characters must be exactly those used in your saprot_sequences.csv.
    """
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=trust_remote_code)
    try:
        model = AutoModelForMaskedLM.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=trust_remote_code)
    except Exception:
        from transformers import AutoModel
        model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=trust_remote_code)
    weight = _get_input_embedding_weight(model)
    aa_v = aa_vocab()
    s_v = {k: int(v) for k, v in struct_vocab.items()}
    table = torch.zeros((len(aa_v), len(s_v), weight.shape[-1]), dtype=torch.float32)
    missing = torch.zeros((len(aa_v), len(s_v)), dtype=torch.bool)
    missing_tokens: list[str] = []
    used = 0
    for aa, ai in aa_v.items():
        if aa == "<pad>":
            continue
        for s, si in s_v.items():
            if s == "<pad>":
                continue
            # In SaProt / ProstT5 conventions 3Di chars are lowercase; '#' may appear as itself.
            candidates = [aa + str(s).lower(), aa + str(s)]
            tok_id = None
            for tok in candidates:
                try:
                    tok_id = _token_id(tokenizer, tok)
                    break
                except KeyError:
                    continue
            if tok_id is None:
                missing[ai, si] = True
                missing_tokens.append(candidates[0])
                continue
            table[ai, si] = weight[tok_id]
            used += 1
    report = SaProtJointBuildReport(
        model_name=model_name,
        embedding_dim=int(weight.shape[-1]),
        aa_vocab_size=len(aa_v),
        struct_vocab_size=len(s_v),
        missing_tokens=missing_tokens,
        used_tokens=used,
    )
    return SaProtJointEmbeddingTable(table, missing_mask=missing, trainable=False), report


def make_dummy_saprot_joint_table(struct_vocab: Dict[str, int], dim: int = 128, seed: int = 0) -> SaProtJointEmbeddingTable:
    """For sanity checks only. Do not use for real experiments."""
    g = torch.Generator().manual_seed(seed)
    table = torch.randn((len(aa_vocab()), len(struct_vocab), dim), generator=g) * 0.02
    table[0].zero_()
    if "<pad>" in struct_vocab:
        table[:, int(struct_vocab["<pad>"])].zero_()
    return SaProtJointEmbeddingTable(table, trainable=False)
