from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

# ESM2 mask_token_id patch for transformers 4.47 + torch 2.5
# Patch both EsmConfig.__init__ AND EsmEmbeddings (some load paths bypass config __init__).
def _patch_esm_mask_token_id():
    try:
        from transformers.models.esm.configuration_esm import EsmConfig
        if not getattr(EsmConfig, "_pplu_patched", False):
            _orig = EsmConfig.__init__
            def _init(self, *a, **k):
                _orig(self, *a, **k)
                if self.mask_token_id is None:
                    self.mask_token_id = 32
            EsmConfig.__init__ = _init
            EsmConfig._pplu_patched = True
        from transformers.models.esm.modeling_esm import EsmEmbeddings
        if not getattr(EsmEmbeddings, "_pplu_patched", False):
            _orig_emb = EsmEmbeddings.__init__
            def _emb_init(self, config, *a, **k):
                _orig_emb(self, config, *a, **k)
                if getattr(self, "mask_token_id", None) is None:
                    self.mask_token_id = 32
            EsmEmbeddings.__init__ = _emb_init
            EsmEmbeddings._pplu_patched = True
    except Exception:
        pass
_patch_esm_mask_token_id()

# Runtime self-healing: no matter how the model was loaded, fix mask_token_id at forward time.
def _patch_esm_forward_runtime():
    try:
        from transformers.models.esm.modeling_esm import EsmEmbeddings
        if getattr(EsmEmbeddings, "_pplu_fwd_patched", False):
            return
        _orig_fwd = EsmEmbeddings.forward
        def _fwd(self, input_ids=None, *a, **k):
            if getattr(self, "mask_token_id", None) is None:
                self.mask_token_id = 32
            return _orig_fwd(self, input_ids, *a, **k)
        EsmEmbeddings.forward = _fwd
        EsmEmbeddings._pplu_fwd_patched = True
    except Exception:
        pass
_patch_esm_forward_runtime()

from .aa_vocab import AA20, aa_vocab
from .bridge_model import ESM2To3DiProjector
from .saprot_joint import SaProtJointEmbeddingTable, build_saprot_joint_embedding_table_from_hf, make_dummy_saprot_joint_table

DEFAULT_STRUCT_ALPHABET = ["<pad>"] + list("ACDEFGHIKLMNPQRSTVWY")
ESM_BY_DIM = {
    1280: "facebook/esm2_t33_650M_UR50D",
    640: "facebook/esm2_t30_150M_UR50D",
    480: "facebook/esm2_t12_35M_UR50D",
    320: "facebook/esm2_t6_8M_UR50D",
}


def default_struct_vocab() -> Dict[str, int]:
    return {s: i for i, s in enumerate(DEFAULT_STRUCT_ALPHABET)}


def _looks_state(x: Any) -> bool:
    return isinstance(x, dict) and any(isinstance(v, torch.Tensor) for v in x.values())


def _extract_state_dict(obj: Any) -> Dict[str, torch.Tensor]:
    if _looks_state(obj):
        return obj
    if not isinstance(obj, dict):
        raise ValueError(f"Unsupported checkpoint type: {type(obj)}")
    for k in ["model", "state_dict", "model_state", "model_state_dict", "bridge", "bridge_state_dict", "projector", "projector_state_dict"]:
        if k in obj:
            try:
                return _extract_state_dict(obj[k])
            except Exception:
                pass
    stack = [obj]
    seen = set()
    while stack:
        cur = stack.pop()
        if id(cur) in seen or not isinstance(cur, dict):
            continue
        seen.add(id(cur))
        if _looks_state(cur) and any(str(k).endswith("in_proj.0.weight") for k in cur.keys()):
            return cur
        for v in cur.values():
            if isinstance(v, dict):
                stack.append(v)
    raise ValueError("Could not find Bridge state_dict; expected key ending with in_proj.0.weight")


def _normalize_state_keys(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    prefixes = ["module.", "model.", "projector.", "bridge_model.", "net."]
    for k, v in state.items():
        kk = str(k)
        changed = True
        while changed:
            changed = False
            for pref in prefixes:
                if kk.startswith(pref):
                    kk = kk[len(pref):]
                    changed = True
        out[kk] = v
    return out


def infer_bridge_config(state: Dict[str, torch.Tensor]) -> Dict[str, object]:
    state = _normalize_state_keys(state)
    matches = [(k, v) for k, v in state.items() if k.endswith("in_proj.0.weight") and isinstance(v, torch.Tensor) and v.ndim == 2]
    if not matches:
        raise ValueError("Cannot infer Bridge config: no in_proj.0.weight")
    key, w = next(((k, v) for k, v in matches if k == "in_proj.0.weight"), matches[0])
    prefix = key[:-len("in_proj.0.weight")]
    hidden_dim, input_dim = int(w.shape[0]), int(w.shape[1])
    tok_key = prefix + "token_head.weight"
    if tok_key not in state:
        tok_key = next(k for k in state if k.endswith("token_head.weight"))
    vocab_size = int(state[tok_key].shape[0])
    layer_ids = set()
    needle = prefix + "encoder.layers."
    for k in state:
        if k.startswith(needle):
            try:
                layer_ids.add(int(k[len(needle):].split(".", 1)[0]))
            except Exception:
                pass
    num_layers = max(layer_ids) + 1 if layer_ids else 3
    use_bridge = any(k.startswith(prefix + "bridge.") for k in state)
    use_saprot_joint = any(k.endswith("saprot_joint_table.table") for k in state)
    struct_dim = 128
    se_key = prefix + "struct_embedding.weight"
    if se_key in state:
        struct_dim = int(state[se_key].shape[1])
    elif use_bridge:
        b0 = prefix + "bridge.0.weight"
        if b0 in state:
            struct_dim = int(state[b0].shape[1]) - input_dim - hidden_dim
    bridge_out_dim = None
    for k, v in state.items():
        if k.endswith("bridge.4.weight") and isinstance(v, torch.Tensor) and v.ndim == 2:
            bridge_out_dim = int(v.shape[0])
    return dict(input_dim=input_dim, hidden_dim=hidden_dim, vocab_size=vocab_size, num_layers=num_layers,
                use_bridge=use_bridge, use_saprot_joint=use_saprot_joint, struct_dim=struct_dim,
                bridge_out_dim=bridge_out_dim, prefix=prefix)


def strip_prefix_state(state: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    state = _normalize_state_keys(state)
    if not prefix:
        return state
    return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state.items()}


def load_bridge(
    bridge_ckpt: str | Path,
    device: str,
    saprot_model_name: str = "westlake-repl/SaProt_650M_AF2",
    saprot_cache_dir: Optional[str] = None,
    struct_vocab_json: Optional[str] = None,
    trust_remote_code: bool = True,
    allow_dummy_saprot: bool = False,
) -> ESM2To3DiProjector:
    ckpt = torch.load(str(bridge_ckpt), map_location="cpu", weights_only=False)
    raw_state = _extract_state_dict(ckpt)
    cfg = infer_bridge_config(raw_state)
    prefix = str(cfg.pop("prefix"))
    state = strip_prefix_state(raw_state, prefix)
    struct_vocab = default_struct_vocab()
    if struct_vocab_json:
        with open(struct_vocab_json, "r", encoding="utf-8") as f:
            struct_vocab = json.load(f)
    saprot_joint_table = None
    if "saprot_joint_table.table" in state:
        saprot_joint_table = SaProtJointEmbeddingTable(state["saprot_joint_table.table"].float(), trainable=False)
    elif bool(cfg["use_saprot_joint"]):
        try:
            saprot_joint_table, report = build_saprot_joint_embedding_table_from_hf(
                saprot_model_name, struct_vocab, cache_dir=saprot_cache_dir, trust_remote_code=trust_remote_code
            )
            print(f"[SaProt] built joint table dim={report.embedding_dim} used={report.used_tokens} missing={len(report.missing_tokens)}")
        except Exception as e:
            if not allow_dummy_saprot:
                raise RuntimeError("Failed to build SaProt joint table") from e
            saprot_joint_table = make_dummy_saprot_joint_table(struct_vocab, dim=1280)
    print(f"[Bridge] input_dim={cfg['input_dim']} hidden={cfg['hidden_dim']} struct_dim={cfg['struct_dim']} use_bridge={cfg['use_bridge']} out_dim={cfg.get('bridge_out_dim')} joint={saprot_joint_table is not None}")
    model = ESM2To3DiProjector(
        input_dim=int(cfg["input_dim"]), hidden_dim=int(cfg["hidden_dim"]), num_layers=int(cfg["num_layers"]),
        vocab_size=int(cfg["vocab_size"]), struct_dim=int(cfg["struct_dim"]), saprot_dim=1280,
        use_bridge=bool(cfg["use_bridge"]), saprot_joint_table=saprot_joint_table,
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[Bridge] missing keys: {missing[:8]}{'...' if len(missing)>8 else ''}")
    if unexpected:
        print(f"[Bridge] unexpected keys ignored: {unexpected[:8]}{'...' if len(unexpected)>8 else ''}")
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


class DifferentiableESMBridge(nn.Module):
    def __init__(self, bridge: ESM2To3DiProjector, esm_model_name: str = "auto", device: str = "cuda",
                 esm_cache_dir: Optional[str] = None, bridge_output: str = "auto", trust_remote_code: bool = False,
                 freeze_esm: bool = True, target_h_dim: int = 1280):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer
        self.bridge = bridge
        self.bridge_output = bridge_output
        self.target_h_dim = int(target_h_dim)
        input_dim = int(getattr(bridge, "input_dim", 1280))
        if esm_model_name == "auto":
            if input_dim not in ESM_BY_DIM:
                raise ValueError(f"Cannot auto-select ESM2 for Bridge input_dim={input_dim}")
            esm_model_name = ESM_BY_DIM[input_dim]
            print(f"[ESMBridge] auto-selected {esm_model_name}")
        local_esm = os.getenv("PPLU_ESM_MODEL_PATH", "")
        if local_esm and input_dim == 640:
            local_path = Path(local_esm).expanduser().resolve()
            if not (local_path / "config.json").is_file():
                raise FileNotFoundError(f"Offline ESM snapshot missing: {local_path}")
            esm_model_name = str(local_path)
        self.esm_model_name = esm_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(esm_model_name, cache_dir=esm_cache_dir, trust_remote_code=trust_remote_code)
        self.esm = AutoModel.from_pretrained(esm_model_name, cache_dir=esm_cache_dir, trust_remote_code=trust_remote_code).to(device)
        # Instance-level self-heal: some transformers load paths leave mask_token_id=None.
        if getattr(self.esm.embeddings, "mask_token_id", None) is None:
            self.esm.embeddings.mask_token_id = 32
            self.esm.config.mask_token_id = 32
        # CRITICAL: we call esm(inputs_embeds=...) with input_ids=None.
        # With token_dropout=True, forward does (None == mask_id).unsqueeze -> bool crash.
        self.esm.embeddings.token_dropout = False
        self.esm.eval()
        emb_dim = int(self.esm.get_input_embeddings().weight.shape[1])
        hidden_size = int(getattr(getattr(self.esm, "config", object()), "hidden_size", emb_dim))
        print(f"[ESMBridge] emb_dim={emb_dim} hidden_size={hidden_size} bridge_input_dim={input_dim} target_H={target_h_dim} output={bridge_output}")
        if emb_dim != input_dim or hidden_size != input_dim:
            raise ValueError(f"ESM/Bridge mismatch: ESM {emb_dim}/{hidden_size}, Bridge expects {input_dim}")
        if freeze_esm:
            for p in self.esm.parameters():
                p.requires_grad_(False)
        self.device = device
        self.aa_token_ids = self._resolve_aa_token_ids(device)
        self.bos_id = self._resolve_special(["cls_token_id", "bos_token_id"], "<cls>")
        self.eos_id = self._resolve_special(["eos_token_id", "sep_token_id"], "<eos>")

    def _resolve_special(self, attrs, fallback_token: str) -> int:
        for a in attrs:
            v = getattr(self.tokenizer, a, None)
            if v is not None:
                return int(v)
        v = self.tokenizer.convert_tokens_to_ids(fallback_token)
        if v is None or int(v) < 0:
            raise ValueError(f"Cannot resolve token {fallback_token}")
        return int(v)

    def _resolve_aa_token_ids(self, device) -> torch.Tensor:
        ids = []
        for aa in AA20:
            idx = self.tokenizer.convert_tokens_to_ids(aa)
            unk = getattr(self.tokenizer, "unk_token_id", None)
            if idx is None or int(idx) < 0 or (unk is not None and int(idx) == int(unk)):
                raise ValueError(f"ESM tokenizer cannot resolve AA token {aa}")
            ids.append(int(idx))
        return torch.tensor(ids, dtype=torch.long, device=device)

    def _soft_esm_inputs(self, aa_probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, _ = aa_probs.shape
        emb = self.esm.get_input_embeddings().weight
        aa_emb = emb[self.aa_token_ids]
        soft_res = aa_probs.to(emb.dtype) @ aa_emb.to(aa_probs.device)
        bos = emb[self.bos_id].to(aa_probs.device).view(1, 1, -1).expand(B, 1, -1)
        eos = emb[self.eos_id].to(aa_probs.device).view(1, 1, -1).expand(B, 1, -1)
        return torch.cat([bos, soft_res, eos], dim=1), torch.ones((B, L + 2), dtype=torch.long, device=aa_probs.device)

    def _choose_H(self, result: Dict[str, torch.Tensor], soft_struct: torch.Tensor) -> torch.Tensor:
        options = {"soft_struct": soft_struct}
        if "saprot_hat" in result:
            options["saprot_hat"] = result["saprot_hat"]
        dims = {k: int(v.shape[-1]) for k, v in options.items()}
        if self.bridge_output in {"soft_struct", "saprot_hat"}:
            H = options[self.bridge_output]
            if int(H.shape[-1]) != self.target_h_dim:
                raise RuntimeError(f"bridge_output={self.bridge_output} dim={H.shape[-1]} but revised_slim needs {self.target_h_dim}; available={dims}")
            return H
        if dims.get("soft_struct") == self.target_h_dim:
            choice = "soft_struct"
        elif dims.get("saprot_hat") == self.target_h_dim:
            choice = "saprot_hat"
        else:
            raise RuntimeError(f"No Bridge output matches H dim={self.target_h_dim}; available={dims}")
        if not hasattr(self, "_printed_choice"):
            print(f"[ESMBridge] auto-selected H={choice}, dims={dims}")
            self._printed_choice = True
        return options[choice]

    def forward(self, aa_probs: torch.Tensor, bridge_temp: float = 1.0) -> Dict[str, torch.Tensor]:
        if aa_probs.ndim == 2:
            aa_probs = aa_probs[None]
        inputs_embeds, attention_mask = self._soft_esm_inputs(aa_probs)
        out = self.esm(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        esm_res = out.last_hidden_state[:, 1:1 + aa_probs.shape[1], :]
        expected_dim = int(getattr(self.bridge, "input_dim", esm_res.shape[-1]))
        if int(esm_res.shape[-1]) != expected_dim:
            raise RuntimeError(f"Live ESM dim={esm_res.shape[-1]}, Bridge expects {expected_dim}")
        h0 = self.bridge.in_proj(esm_res)
        h = self.bridge.encoder(h0, src_key_padding_mask=None)
        h = self.bridge.out_norm(h)
        logits = self.bridge.token_head(h)
        q3di = torch.softmax(logits / max(float(bridge_temp), 1e-6), dim=-1)
        if self.bridge.saprot_joint_table is not None:
            table = self.bridge.saprot_joint_table.table.to(aa_probs.device, dtype=aa_probs.dtype)
            av = aa_vocab()
            aa1_ids = torch.tensor([av[a] for a in AA20], dtype=torch.long, device=aa_probs.device)
            aa_table = table.index_select(0, aa1_ids)
            soft_struct = torch.einsum("bla,bls,asd->bld", aa_probs.to(table.dtype), q3di.to(table.dtype), aa_table)
        else:
            if self.bridge.struct_embedding is None:
                raise RuntimeError("Bridge has neither saprot_joint_table nor struct_embedding")
            soft_struct = q3di @ self.bridge.struct_embedding.weight.to(q3di.device)
        result = {"esm": esm_res, "bridge_hidden": h, "q3di": q3di, "soft_struct_embedding": soft_struct}
        if self.bridge.bridge is not None:
            result["saprot_hat"] = self.bridge.bridge(torch.cat([esm_res, soft_struct.to(esm_res.dtype), h], dim=-1))
        result["H"] = self._choose_H(result, soft_struct)
        return result
