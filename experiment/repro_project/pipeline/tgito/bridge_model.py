from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .saprot_joint import SaProtJointEmbeddingTable


class ESM2To3DiProjector(nn.Module):
    """Predict Foldseek 3Di token distribution from per-residue ESM embeddings.

    Stage-2 features:
      1) If `saprot_joint_table` is supplied, the bridge uses SaProt official
         AA+3Di joint-token expected embeddings instead of a custom 3Di table.
      2) If no joint table is supplied, it falls back to the original learned 3Di
         embedding table for baseline/control experiments.
    """
    def __init__(
        self,
        input_dim: int = 1280,
        hidden_dim: int = 512,
        num_layers: int = 3,
        num_heads: int = 8,
        vocab_size: int = 21,
        dropout: float = 0.1,
        struct_dim: int = 128,
        saprot_dim: int = 1280,
        use_bridge: bool = False,
        saprot_joint_table: Optional[SaProtJointEmbeddingTable] = None,
        freeze_saprot_joint_table: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.use_bridge = bool(use_bridge)
        self.use_saprot_joint = saprot_joint_table is not None
        self.in_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.token_head = nn.Linear(hidden_dim, vocab_size)

        self.saprot_joint_table = saprot_joint_table
        if self.saprot_joint_table is not None:
            if freeze_saprot_joint_table:
                for p in self.saprot_joint_table.parameters():
                    p.requires_grad_(False)
            bridge_struct_dim = self.saprot_joint_table.embedding_dim
            self.struct_embedding = None
        else:
            bridge_struct_dim = struct_dim
            self.struct_embedding = nn.Embedding(vocab_size, struct_dim)

        if self.use_bridge:
            self.bridge = nn.Sequential(
                nn.Linear(input_dim + bridge_struct_dim + hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.LayerNorm(hidden_dim * 2),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, saprot_dim),
            )
        else:
            self.bridge = None

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, temperature: float = 1.0, aa_ids: Optional[torch.Tensor] = None):
        # x: (B,L,D), mask True for valid residues.
        h0 = self.in_proj(x)
        key_padding_mask = None if mask is None else ~mask.bool()
        h = self.encoder(h0, src_key_padding_mask=key_padding_mask)
        h = self.out_norm(h)
        logits = self.token_head(h)
        q = torch.softmax(logits / max(float(temperature), 1e-6), dim=-1)

        if self.saprot_joint_table is not None:
            if aa_ids is None:
                raise ValueError("aa_ids is required when using SaProt joint-token embedding table")
            soft_struct = self.saprot_joint_table(aa_ids.to(x.device), q)
            soft_name = "soft_saprot_joint_embedding"
        else:
            soft_struct = q @ self.struct_embedding.weight
            soft_name = "soft_3di_embedding"

        out = {"logits": logits, "prob": q, soft_name: soft_struct, "soft_struct_embedding": soft_struct, "hidden": h}
        if self.bridge is not None:
            out["saprot_hat"] = self.bridge(torch.cat([x, soft_struct.to(dtype=x.dtype), h], dim=-1))
        return out


def masked_ce_loss(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    b, l, v = logits.shape
    loss = F.cross_entropy(logits.reshape(b*l, v), target.reshape(b*l), reduction="none")
    loss = loss.reshape(b, l)
    denom = mask.float().sum().clamp_min(1.0)
    return (loss * mask.float()).sum() / denom


def masked_cosine_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    c = 1.0 - F.cosine_similarity(pred.float(), target.float(), dim=-1)
    denom = mask.float().sum().clamp_min(1.0)
    return (c * mask.float()).sum() / denom


def smoothness_loss(soft_struct: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if soft_struct.shape[1] < 2:
        return soft_struct.sum() * 0.0
    diff = (soft_struct[:, 1:] - soft_struct[:, :-1]).pow(2).sum(dim=-1)
    m = (mask[:, 1:] & mask[:, :-1]).float()
    return (diff * m).sum() / m.sum().clamp_min(1.0)


def total_e2s_loss(out, target, mask, target_saprot=None, lambda_saprot=0.0, lambda_smooth=0.0):
    ce = masked_ce_loss(out["logits"], target, mask)
    loss = ce
    logs = {"loss_ce": float(ce.detach().cpu())}
    if target_saprot is not None and "saprot_hat" in out and lambda_saprot > 0:
        sap = masked_cosine_loss(out["saprot_hat"], target_saprot, mask)
        loss = loss + lambda_saprot * sap
        logs["loss_saprot_cos"] = float(sap.detach().cpu())
    if lambda_smooth > 0:
        sm = smoothness_loss(out["soft_struct_embedding"], mask)
        loss = loss + lambda_smooth * sm
        logs["loss_smooth"] = float(sm.detach().cpu())
    logs["loss_total"] = float(loss.detach().cpu())
    return loss, logs
