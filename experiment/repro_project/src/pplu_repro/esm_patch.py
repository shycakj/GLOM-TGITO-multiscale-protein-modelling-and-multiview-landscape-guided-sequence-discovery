"""Patch transformers EsmConfig so ESM2 loads work on transformers 4.47.0 + torch 2.5.

Bug: transformers/models/esm/modeling_esm.py:210 does
    input_ids == self.mask_token_id
with mask_token_id=None. torch 2.5 turns `tensor == None` into scalar False,
then EsmEmbeddings calls .unsqueeze on the bool scalar -> AttributeError.

Fix: default mask_token_id to 32 (ESM2 <mask> id) inside EsmConfig.__init__
before any model load. Import this module FIRST in any script that loads an
EsmModel / EsmForMaskedLM / EsmForSequenceClassification.
"""
from __future__ import annotations

import transformers
from transformers.models.esm.configuration_esm import EsmConfig

_ORIG_INIT = EsmConfig.__init__


def _patched_init(self, *args, **kwargs):
    _ORIG_INIT(self, *args, **kwargs)
    if self.mask_token_id is None:
        self.mask_token_id = 32  # ESM2 <mask> id


EsmConfig.__init__ = _patched_init

# Also patch any class that inherits EsmConfig via the module-level default
# (config classes are created once at import time, so this covers all uses).


def ensure_patched() -> bool:
    return EsmConfig.__init__ is _patched_init
