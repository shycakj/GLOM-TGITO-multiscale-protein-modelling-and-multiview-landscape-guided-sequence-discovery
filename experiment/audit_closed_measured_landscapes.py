"""Check whether this sparse assay set supplies complete WT-rooted binary cubes.

Sequence/partition metadata only. This tests applicability of a common measured-
landscape benchmark, without imputing unmeasured fitness or selecting by activity.
"""
from pathlib import Path
import itertools,json
import pandas as pd
ROOT=Path(__file__).resolve().parent
d=pd.read_csv(ROOT/'data/unique_sequences.csv',usecols=['sequence','mutation_depth','partition'])
wt=d.loc[d.mutation_depth.eq(0),'sequence'].iloc[0]
keys={}
for row in d.itertuples():
    key=frozenset((i,b) for i,(a,b) in enumerate(zip(wt,row.sequence)) if a!=b)
    keys[key]=row.partition
rows=[];examples=[]
for dimension in range(2,7):
    corners=[key for key in keys if len(key)==dimension];complete=[]
    for key in corners:
        parts=[];ok=True
        for n in range(dimension+1):
            for subset in itertools.combinations(sorted(key),n):
                partition=keys.get(frozenset(subset))
                if partition is None:ok=False;break
                parts.append(partition)
            if not ok:break
        if ok:
            complete.append(parts)
            if len(examples)<20:examples.append(dict(dimension=dimension,mutations=sorted(key),partitions=parts))
    rows.append(dict(dimension=dimension,domain_size=2**dimension,observed_max_corners=len(corners),
        complete_cubes=len(complete),complete_with_test_node=sum('test' in x for x in complete),
        complete_all_non_wt_nodes_in_test=sum(all(v=='test' for v in x[1:]) for x in complete)))
out=ROOT/'audit/closed_measured_landscapes.json'
out.write_text(json.dumps(dict(scope='WT-rooted binary substitution cubes of dimensions 2--6 only; not arbitrary-background enumeration',
    no_fluorescence_labels_read=True,dimensions=rows,examples=examples),indent=2))
print(json.dumps(rows,indent=2))
