"""ppluGFP-specific constants for TG-ITO PPLU package."""
from __future__ import annotations

PPLU_WT = (
    "PAMKIECRITGTLNGVEFELVGGGEGTPEQGRMTNKMKSTKGALTFSPYLLSHVMGYGFYHFGTYPSGYENPFLHAINNGGYTNTRIEKYEDGGVLHVSFSYRYEAGRVIGDFKVVGTGFPEDSVIFTDKIIRSNATVEHLHPMGDNVLVGSFARTFSLRDGGYYSFVVDSHMHFKSAIHPSILQNGGPMFAFRRVEELHSNTELGIVEYQHAFKTPIAFA"
)
PPLU_LENGTH = 221

KEY_SITES_1B = [24, 33, 35, 36, 52, 53, 55, 56, 57, 58, 59, 62, 84, 86, 100]
KEY_SITES_0B = [p - 1 for p in KEY_SITES_1B]

REGIONS_1B = {
    "win_53_62":   [53, 62],
    "win_55_64":   [55, 64],
    "win_51_60":   [51, 60],
    "win_207_216": [207, 216],
    "win_203_212": [203, 212],
    "win_77_86":   [77, 86],
    "win_27_36":   [27, 36],
    "win_165_174": [165, 174],
    "win_97_106":  [97, 106],
}
REGION_NAMES = list(REGIONS_1B.keys())
RIDX = [list(range(v[0] - 1, v[1])) for v in REGIONS_1B.values()]

PPLU_DEFAULT_COMPANION_SOURCES = "h32,global_attn_pool,keysite_K_flat,region_win_53_62_all_mut512,region_win_55_64_all_mut512,region_win_77_86_all_mut512"


def clean_pplu_wt() -> str:
    return "".join(ch for ch in PPLU_WT.upper() if ch in "ACDEFGHIKLMNPQRSTVWY")[:PPLU_LENGTH]
