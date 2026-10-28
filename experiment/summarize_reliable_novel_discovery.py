"""Joint novelty/reliability counts, with a recorded full sensitivity grid."""
from pathlib import Path
from datetime import datetime
import json
import numpy as np
import pandas as pd
from compare_discovery_v3_reliability import OUT, METHODS, sha, write_json, wilson


def main():
    plan=dict(created_local=datetime.now().astimezone().isoformat(),
        status='Supplement added after aggregate risk comparison; before computing joint counts',
        motivation='Original discovery objective requires novelty and credibility jointly',
        distance_thresholds=[1,2,3],lower_brightness_thresholds=[4.2,4.3,4.35],
        risk_quantiles=[.25,.5,.75],stability_quantile=.75,
        primary_display=dict(nearest_visible_distance=2,lower_brightness=4.2,risk_quantile=.5),
        full_grid_reported=True,ranking_changed=False,
        reference='Novelty is distance from train+validation+calibration sequences; test outcomes use distance to train, explicitly distinguished',
        inputs={f:sha(OUT/f) for f in ['candidate_reliability_scores.csv','calibration_reliability_rows.csv','test_reliability_rows.csv']})
    path=OUT/'JOINT_ANALYSIS_RULES.json'
    if path.exists():
        old=json.loads(path.read_text(encoding='utf-8'))
        assert old['inputs']==plan['inputs']
    else:write_json(path,plan)
    s=pd.read_csv(OUT/'candidate_reliability_scores.csv')
    cal=pd.read_csv(OUT/'calibration_reliability_rows.csv')
    test=pd.read_csv(OUT/'test_reliability_rows.csv')
    high=cal[cal.cross_mean.ge(4.2)]
    sd=float(high.cross_sd.quantile(.75))
    rows=[];valid=[]
    for rq in [.25,.5,.75]:
        cutoff=float(high.combined_risk.quantile(rq))
        for bright in [4.2,4.3,4.35]:
            for dist in [1,2,3]:
                for (m,k),d in s.groupby(['method','batch_size']):
                    novel=d.nearest_visible_hamming.ge(dist)
                    reliable=d.risk_scaled_lower90.ge(bright)&d.combined_risk.le(cutoff)&d.cross_sd.le(sd)
                    rows.append(dict(method=m,batch_size=int(k),distance_threshold=dist,lower_brightness_threshold=bright,
                        risk_quantile=rq,risk_cutoff=cutoff,sd_cutoff=sd,
                        novel_n=int(novel.sum()),reliability_pass_n=int(reliable.sum()),
                        novel_and_reliability_pass_n=int((novel&reliable).sum()),
                        joint_fraction=float((novel&reliable).mean())))
                accepted=test.nearest_training_hamming.ge(dist)&test.risk_scaled_lower90.ge(bright)&test.combined_risk.le(cutoff)&test.cross_sd.le(sd)
                d=test[accepted];n=len(d);success=int(d.brightness.ge(bright).sum());lo,hi=wilson(success,n)
                valid.append(dict(distance_to_train_threshold=dist,lower_brightness_threshold=bright,risk_quantile=rq,
                    accepted_test_n=n,actual_success_n=success,actual_success_fraction=success/n if n else np.nan,wilson_low=lo,wilson_high=hi))
    result=pd.DataFrame(rows);validation=pd.DataFrame(valid)
    result.to_csv(OUT/'joint_novelty_reliability_full_grid.csv',index=False)
    validation.to_csv(OUT/'joint_rule_measured_validation.csv',index=False)
    main=result[result.distance_threshold.eq(2)&result.lower_brightness_threshold.eq(4.2)&result.risk_quantile.eq(.5)].copy()
    main['order']=main.method.map({m:i for i,m in enumerate(METHODS)})
    main=main.sort_values(['batch_size','order']).drop(columns='order')
    main.to_csv(OUT/'joint_novelty_reliability_primary.csv',index=False)
    payload=json.loads((OUT/'workbook_data.json').read_text(encoding='utf-8'))
    payload['joint']=json.loads(main.to_json(orient='records'))
    payload['joint_test']=json.loads(validation.to_json(orient='records'))
    # Keep original computation export immutable; the presentation file is separate.
    write_json(OUT/'workbook_presentation_data.json',payload)
    print(main[['method','batch_size','novel_n','reliability_pass_n','novel_and_reliability_pass_n']].to_string(index=False))
    print(validation[validation.distance_to_train_threshold.eq(2)&validation.risk_quantile.eq(.5)].to_string(index=False))


if __name__=='__main__':main()
