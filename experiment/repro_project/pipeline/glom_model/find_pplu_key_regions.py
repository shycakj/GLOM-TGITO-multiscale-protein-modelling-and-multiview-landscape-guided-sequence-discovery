"""Find key positions for ppluGFP — adapted from find_key_regions_from_results.py"""
import numpy as np, pandas as pd
from collections import defaultdict
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_predict
from scipy.stats import pearsonr
import warnings; warnings.filterwarnings('ignore')

PPLU_WT = 'PAMKIECRITGTLNGVEFELVGGGEGTPEQGRMTNKMKSTKGALTFSPYLLSHVMGYGFYHFGTYPSGYENPFLHAINNGGYTNTRIEKYEDGGVLHVSFSYRYEAGRVIGDFKVVGTGFPEDSVIFTDKIIRSNATVEHLHPMGDNVLVGSFARTFSLRDGGYYSFVVDSHMHFKSAIHPSILQNGGPMFAFRRVEELHSNTELGIVEYQHAFKTPIAFA'
L = len(PPLU_WT)

df = pd.read_csv(r'D:\桌面\GFP\pplu_31480_for_keyfinder.csv')
print(f'Loaded {len(df)} ppluGFP variants')
# Stratified sample for speed: 2000 bright + 2000 mid + 2000 dim
b = df['medianBrightness']
hi, lo = b.quantile(0.80), b.quantile(0.20)
s_hi = df[b >= hi].sample(700, random_state=42)
s_mid = df[(b > lo) & (b < hi)].sample(700, random_state=42)
s_lo = df[b <= lo].sample(600, random_state=42)
train_df = pd.concat([s_hi, s_mid, s_lo]).copy()
print(f'Training on {len(train_df)} samples (stratified 700+700+600)')

# Build pos×AA features
feature_set = set()
rows = []
for _, row in train_df.iterrows():
    seq = str(row['sequence'])
    feats = {}
    for p in range(L):
        wt = PPLU_WT[p]
        mt = seq[p] if p < len(seq) else wt
        if wt != mt:
            key = f'pxa_{p}_{mt}'
            feats[key] = 1.0
            feature_set.add(key)
    rows.append(feats)

feature_list = sorted(feature_set)
feat_idx = {f: i for i, f in enumerate(feature_list)}
X = np.zeros((len(rows), len(feature_list)), dtype=np.float32)
y = np.array(train_df['medianBrightness'].to_numpy(), dtype=np.float32)
for i, feats in enumerate(rows):
    for k, v in feats.items():
        if k in feat_idx: X[i, feat_idx[k]] = v
print(f'Features: {X.shape[1]} pos×AA from {X.shape[0]} sequences')

# Train Ridge (simple, fast)
from sklearn.linear_model import Ridge
scaler = StandardScaler()
X_s = scaler.fit_transform(X)
model = Ridge(alpha=10.0)
model.fit(X_s, y)
from sklearn.model_selection import cross_val_predict as cvp
y_pred = cvp(Ridge(alpha=10.0), X_s, y, cv=3)
cv_r = pearsonr(y_pred, y)[0]
print(f'CV Pearson R = {cv_r:.4f}')

# Extract position importance
coefs = model.coef_
pos_imp = defaultdict(lambda: {'total_abs':0.0, 'n_aa':0, 'best_aa':'', 'best_coef':-99, 'worst_aa':'', 'worst_coef':99})
for feat, coef in zip(feature_list, coefs):
    parts = feat.split('_')
    pos = int(parts[1]); aa = parts[2]
    pi = pos_imp[pos]
    pi['total_abs'] += abs(coef); pi['n_aa'] += 1
    if coef > pi['best_coef']: pi['best_coef'] = coef; pi['best_aa'] = aa
    if coef < pi['worst_coef']: pi['worst_coef'] = coef; pi['worst_aa'] = aa

ranked = sorted(pos_imp.items(), key=lambda x: x[1]['total_abs'], reverse=True)

top_n = 40
print(f'\nTOP {top_n} KEY POSITIONS for ppluGFP (221 AA)')
print(f'{"Pos":<6} {"WT":<4} {"Tot|coef|":<12} {"#AA":<5} {"Best_AA":<8} {"Worst_AA":<8}')
print('-' * 50)
for pos, info in ranked[:top_n]:
    print(f'{pos+1:<6} {PPLU_WT[pos]:<4} {info["total_abs"]:<12.4f} {info["n_aa"]:<5} {info["best_aa"]:<8} {info["worst_aa"]:<8}')

# Save
out_rows = []
for pos, info in ranked:
    out_rows.append({'position': pos+1, 'wt_aa': PPLU_WT[pos],
                     'total_abs_coef': info['total_abs'], 'n_aa': info['n_aa'],
                     'best_aa': info['best_aa'], 'best_coef': info['best_coef'],
                     'worst_aa': info['worst_aa'], 'worst_coef': info['worst_coef']})
pd.DataFrame(out_rows).to_csv(r'D:\桌面\GFP\pplu_key_positions.csv', index=False)

top30 = sorted([pos+1 for pos, _ in ranked[:30]])
print(f'\nKEY_POSITIONS (top 30): {top30}')

# ── Sliding-window segment importance ──
WINDOW = 10
pos_total = {pos: info['total_abs'] for pos, info in ranked}
seg_scores = []
for start in range(0, L - WINDOW + 1, 2):
    end = start + WINDOW
    seg_total = sum(pos_total.get(p, 0.0) for p in range(start, end))
    n_top30 = sum(1 for p in range(start, end) if (p+1) in top30)
    seg_scores.append({'start': start+1, 'end': end, 'window_len': WINDOW,
                       'total_importance': seg_total, 'n_top30': n_top30})
seg_scores.sort(key=lambda x: -x['total_importance'])

print(f'\nTOP 15 WINDOW-BASED KEY SEGMENTS (window={WINDOW}):')
print(f'{"Range":<14} {"TotalImp":<12} {"#Top30":<8}')
print('-' * 40)
for s in seg_scores[:15]:
    st, en, tot, n30 = s['start'], s['end'], s['total_importance'], s['n_top30']
    print(f'{st:3d}-{en:3d}       {tot:<12.4f} {n30:<8}')

pd.DataFrame(seg_scores).to_csv(r'D:\桌面\GFP\pplu_key_segments.csv', index=False)
print(f'\nSaved to pplu_key_positions.csv and pplu_key_segments.csv')
