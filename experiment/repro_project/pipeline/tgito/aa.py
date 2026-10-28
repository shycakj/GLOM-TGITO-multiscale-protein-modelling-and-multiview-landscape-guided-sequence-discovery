from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from .aa_vocab import AA20, aa_index0

# Three mutant-AA property bits. Keep this stable because revised_slim was trained on
# columns [7,8,9] from pos_feats.npy. This table is only used to form *mutant* bits,
# i.e. sum_{a != WT_i} p_i(a) * bit(a), never current-AA bits at all positions.
# Columns are: hydrophobic/aromatic-ish, polar/charged-ish, special/proline/glycine/cys-ish.
# This matches the intended coarse type coding used in TG-ITO; audit can compare to pos_feats.npy.
AA_BITS = {
    "A": (1, 0, 0), "V": (1, 0, 0), "I": (1, 0, 0), "L": (1, 0, 0), "M": (1, 0, 0),
    "F": (1, 0, 0), "W": (1, 0, 0), "Y": (1, 1, 0),
    "S": (0, 1, 0), "T": (0, 1, 0), "N": (0, 1, 0), "Q": (0, 1, 0),
    "D": (0, 1, 0), "E": (0, 1, 0), "K": (0, 1, 0), "R": (0, 1, 0), "H": (0, 1, 0),
    "G": (0, 0, 1), "P": (0, 0, 1), "C": (0, 0, 1),
}


def clean_seq(seq: str, length: int = 221) -> str:
    s = "".join(ch for ch in str(seq).upper() if ch in AA20)
    if len(s) < length:
        # Never silently create biologically meaningful residues; pad by repeating last valid AA only for malformed short input.
        if not s:
            raise ValueError("Empty sequence after AA20 cleaning")
        s = s + s[-1] * (length - len(s))
    return s[:length]


def seq_to_ids0(seq: str, length: int = 221, device=None) -> torch.Tensor:
    table = aa_index0()
    s = clean_seq(seq, length)
    return torch.tensor([table[a] for a in s], dtype=torch.long, device=device)


def ids0_to_seq(ids) -> str:
    if isinstance(ids, torch.Tensor):
        ids = ids.detach().cpu().tolist()
    return "".join(AA20[int(i)] for i in ids)


def aa_bits_tensor(device=None, dtype=torch.float32) -> torch.Tensor:
    return torch.tensor([AA_BITS[a] for a in AA20], dtype=dtype, device=device)


def parse_positions(spec: str, length: int = 221) -> torch.Tensor:
    mask = torch.zeros(length, dtype=torch.bool)
    if spec is None or str(spec).strip().lower() in {"", "none", "all"}:
        if str(spec).strip().lower() == "all":
            mask[:] = True
        return mask
    for part in str(spec).replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            lo, hi = int(a), int(b)
            for p in range(lo, hi + 1):
                if 1 <= p <= length:
                    mask[p - 1] = True
        else:
            p = int(part)
            if 1 <= p <= length:
                mask[p - 1] = True
    return mask


def apply_mutation_mask(logits: torch.Tensor, base_ids: torch.Tensor, mutable_mask: torch.Tensor, lock_scale: float = 30.0) -> torch.Tensor:
    base_ids = base_ids.to(logits.device).long()
    mutable_mask = mutable_mask.to(logits.device).bool()
    locked = torch.full_like(logits, -float(lock_scale))
    locked.scatter_(1, base_ids[:, None], float(lock_scale))
    return torch.where(mutable_mask[:, None], logits, locked)


def make_logits_from_seq(seq: str, length: int = 221, init_scale: float = 9.0, device=None) -> torch.Tensor:
    ids = seq_to_ids0(seq, length=length, device=device)
    logits = torch.full((length, 20), -float(init_scale), dtype=torch.float32, device=device)
    logits.scatter_(1, ids[:, None], float(init_scale))
    return logits


def make_relaxed_logits_from_ids(
    ids: torch.Tensor,
    length: int = 221,
    mutable_mask: Optional[torch.Tensor] = None,
    anchor_logit: float = 5.0,
    alt_logit: float = 0.0,
    lock_scale: float = 30.0,
    device=None,
) -> torch.Tensor:
    ids = ids.to(device=device).long()
    if ids.numel() != length:
        ids = ids[:length]
    logits = torch.full((length, 20), float(alt_logit), dtype=torch.float32, device=ids.device)
    logits.scatter_(1, ids[:, None], float(anchor_logit))
    if mutable_mask is not None:
        logits = apply_mutation_mask(logits, ids, mutable_mask.to(ids.device), lock_scale=lock_scale)
    return logits


def make_relaxed_logits_from_seq(
    seq: str,
    length: int = 221,
    mutable_mask: Optional[torch.Tensor] = None,
    anchor_logit: float = 5.0,
    alt_logit: float = 0.0,
    lock_scale: float = 30.0,
    device=None,
) -> torch.Tensor:
    ids = seq_to_ids0(seq, length=length, device=device)
    return make_relaxed_logits_from_ids(ids, length, mutable_mask, anchor_logit, alt_logit, lock_scale, device)


def soft_pos_feats(aa_probs: torch.Tensor, reference_ids0: torch.Tensor, aa_bits: torch.Tensor) -> torch.Tensor:
    """Build differentiable P [B,L,10] in the original revised_slim training convention.

    revised_slim uses columns [0,7,8,9]:
      P[...,0]    = mutation probability relative to WT/reference.
      P[...,7:10] = mutant-AA type bits, zero for non-mutated/reference residues.

    For soft AA probabilities this is:
      P_i,0    = 1 - p_i(WT_i)
      P_i,7:10 = sum_{a != WT_i} p_i(a) * bit(a)
    """
    if aa_probs.ndim == 2:
        aa_probs = aa_probs[None]
    B, L, A = aa_probs.shape
    if A != 20:
        raise ValueError(f"aa_probs last dim should be 20, got {A}")
    ref_ids = reference_ids0.to(aa_probs.device).long()
    bits = aa_bits.to(aa_probs.device, dtype=aa_probs.dtype)
    P = torch.zeros((B, L, 10), dtype=aa_probs.dtype, device=aa_probs.device)
    ref_onehot = F.one_hot(ref_ids, num_classes=20).to(dtype=aa_probs.dtype)
    same_prob = (aa_probs * ref_onehot[None]).sum(dim=-1)
    P[..., 0] = 1.0 - same_prob
    expected_bits_all = aa_probs @ bits
    ref_bits = bits.index_select(0, ref_ids)[None, :, :]
    P[..., 7:10] = expected_bits_all - same_prob.unsqueeze(-1) * ref_bits
    return P


def mutation_count_soft(aa_probs: torch.Tensor, reference_ids0: torch.Tensor) -> torch.Tensor:
    if aa_probs.ndim == 2:
        aa_probs = aa_probs[None]
    ref_onehot = F.one_hot(reference_ids0.to(aa_probs.device).long(), num_classes=20).to(dtype=aa_probs.dtype)
    same_prob = (aa_probs * ref_onehot[None]).sum(dim=-1)
    return (1.0 - same_prob).sum(dim=-1)
