"""Validate the local scoring delivery and write a compact experiment report."""
from pathlib import Path
import argparse,hashlib,json,zipfile
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from experiment_core import write_json

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/discovery_v3/embedding_familiarity_v1'
EXPECTED_ARCHIVE='a3d588f75667c668a35a61f7b75670d7b94ce9180d3d27c4465059b27e1edb41'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--existing',action='store_true');args=parser.parse_args()
    archive=ROOT/('embedding_familiarity_v1_delivery.zip' if args.existing else 'embedding_familiarity_v1_delivery_exec.zip')
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==EXPECTED_ARCHIVE
    if not args.existing:
        with zipfile.ZipFile(archive) as z:
            for info in z.infolist():
                dest=(ROOT/info.filename).resolve();assert dest.is_relative_to(ROOT.resolve())
                if dest.exists() and dest.suffix=='.py':assert dest.read_bytes()==z.read(info)
                else:z.extract(info,ROOT)
    manifest=json.loads((OUT/'DELIVERY_MANIFEST.json').read_text())
    for name,item in manifest.items():
        p=ROOT/name;assert p.stat().st_size==item['bytes']
        assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256'],name
    knn=pd.read_csv(OUT/'test_scores.csv');mw=pd.read_csv(OUT/'maxwent_pilot/test_scores.csv')
    data=knn.merge(mw[['sequence','maxwent_sd']],on='sequence',validate='one_to_one')
    ci=[];rng=np.random.default_rng(20261014)
    for scope,d in [('all',data),('predicted_high420',data[data.prediction>=4.2])]:
        e=d.prediction.to_numpy()-d.brightness.to_numpy()
        scores=d[['z768_d32','esm_mean_d32','maxwent_sd']].to_numpy()
        for kind,target in [('absolute_gt0.1',abs(e)>.1),('overprediction_gt0.1',e>.1)]:
            positives=np.flatnonzero(target);negatives=np.flatnonzero(~target);draws=[]
            for _ in range(1000):
                ix=np.r_[rng.choice(positives,len(positives)),rng.choice(negatives,len(negatives))]
                draws.append([roc_auc_score(target[ix],scores[ix,j]) for j in range(3)])
            draws=np.asarray(draws)
            for j,name in enumerate(['z768_d32','esm_mean_d32','maxwent_sd']):
                lo,hi=np.quantile(draws[:,j],[.025,.975]);ci.append(dict(scope=scope,target=kind,score=name,n=len(d),events=int(target.sum()),
                    auroc=roc_auc_score(target,scores[:,j]),ci_low=lo,ci_high=hi,bootstrap=1000))
            for j,name in [(1,'z768_minus_esm_mean'),(2,'z768_minus_maxwent')]:
                lo,hi=np.quantile(draws[:,0]-draws[:,j],[.025,.975]);ci.append(dict(scope=scope,target=kind,score=name,n=len(d),events=int(target.sum()),
                    auroc=roc_auc_score(target,scores[:,0])-roc_auc_score(target,scores[:,j]),ci_low=lo,ci_high=hi,bootstrap=1000))
    ci=pd.DataFrame(ci);ci.to_csv(OUT/'test_auroc_bootstrap.csv',index=False)
    summary=pd.read_csv(OUT/'method_summary.csv');top=summary[summary.scope=='top100'];allpool=summary[summary.scope=='all_unique']
    done=json.loads((OUT/'SCORES_DONE.json').read_text());threshold=done['thresholds']['z768_d32']
    validation=pd.read_csv(OUT/'test_score_validation.csv');groups=pd.read_csv(OUT/'test_group_metrics.csv')
    lines=['GLOM–TGITO 输出的 embedding 熟悉度：实跑结果（2026-10-14）','',
        '本轮完成了什么',
        '冻结的 900 峰 / 2700 上下文整流程及四个原生基线不变；对所有输出补算 embedding 熟悉度，并完成 MaxWEnt 预测头小规模适配试验。',
        '全部 44399 条独立序列重新以 float32 编码。参照库仅含训练集 22003 条；校准集 3143 条，实测测试集 3144 条。',
        '五种方法的全部候选按序列去重后，与实测库并集新增 12965 条；该数为并集新增数，不是任何单个方法的产量。','',
        '评分定义',
        '主表示为 GLOM z768=[全局 G, 区域 Rf, 关键位点 Kf]，L2 归一化后计算第 32 个训练邻居的欧氏距离。',
        '另完整保存 G/Rf/Kf 各自的距离、原始 ESM150 token 均值 embedding 距离及 k=8/16/64 敏感性结果。',
        '这里没有使用 Hamming 或 one-hot 差异计数作为熟悉度。one-hot 仅作为既有 ESM 编码接口的输入格式。',
        f'全部校准序列的距离 q95={threshold["q95"]:.6f}；预测亮度>=4.2 的高亮校准序列 n={threshold["highcal_n"]}，q95={threshold["highcal_q95"]:.6f}。',
        '偏离程度=距离/q95，>1 表示超过相应参照范围；这是特征支持度，不是错误概率，也不是预训练成员判断。','',
        'Top 100 结果（同一冻结排序，未重新选候选）',
        top[['method','n','outside_q95_n','outside_highcal_q95_n','degree_mean','highcal_degree_mean','esm_mean_outside_q95_fraction']].to_string(index=False),
        '', '全部独立输出结果',allpool[['method','n','outside_q95_n','outside_highcal_q95_n','degree_mean','highcal_degree_mean']].to_string(index=False),
        '', '评分是否真的识别模型误差',
        '误差阈值 0.1 使用数据集原始亮度标签尺度。AUROC 衡量大误差识别能力，不是亮蛋白生成成功率。',
        ci[(ci.target=='absolute_gt0.1')&ci.score.isin(['z768_d32','esm_mean_d32','maxwent_sd'])][['scope','score','n','events','auroc','ci_low','ci_high']].to_string(index=False),
        '', '原预测器在实测测试集上的分组 RMSE / R2 / Spearman',
        groups[groups.threshold.isin(['outside_q95','outside_highcal_q95'])].to_string(index=False),
        '', '判断',
        'GLOM embedding 距离能够提供有实测误差检验的熟悉度统计；在高亮测试区仍有误差识别能力。',
        '我们的 Top100 平均偏离程度低于本轮四个基线，但超出高亮参照范围的比例并非最低：2%，AdaLead 为1%。该结果不作“各项全面领先”的表述。',
        'MaxWEnt 试验保持模型权重均值不变，只在训练数据上拟合 568321 个预测头噪声尺度，按验证误差上限选 step300，64 次共同权重采样评分。',
        'MaxWEnt 全测试区的大误差 AUROC 较好，但高亮区仅约0.586，低于主 embedding 指标；保留全部结果为辅助，不采用为主要可信度筛选。',
        '全部生成序列的未知荧光仍缺失。本轮证明的是输出的特征支持程度，以及该评分与实测误差的关系；与已完成的地势风险、支撑和跨种子统计共同解释发现能力。',
        '这是对已冻结候选的回顾性评分。先前测试结果已被查看；本轮没有用测试标签、生成候选排名或方法胜负调 k、阈值或 MaxWEnt 选点。',
        '同一批候选由同一搜索活动产生，方法间比例按描述统计报告，不把2700起点当成独立实验重复。','',
        '实现核验',
        '原生编码 anchor 与当前编码一致；预测头零噪声回放一致；采样权重跨 batch 保持同一抽样；固定均值无梯度；距离以 fp64 运算并与直接相减结果交叉核验。',
        f'与原有 7097 条 live 预测的重放 RMSE={done["live_prediction_replay"]["rmse"]:.3g}，最大绝对差={done["live_prediction_replay"]["max_abs"]:.3g}。',
        '双卡各一个提取任务，约68秒/卡，峰值显存1.36 GiB；所有文件保存在 /xiongyuchen/gfp_evidence_20261013/ 下。','',
        '主要文件',
        'candidate_scores.csv：五种方法的全部输出及 Top6/30/100 逐序列分数；scope 不同会重复出现同一候选，汇总时不可跨scope重复计数。',
        'method_summary.csv：20 个 方法×输出范围 汇总。test_scores.csv / test_group_metrics.csv / test_auroc_bootstrap.csv：实测核验。',
        'maxwent_pilot/：所有辅助试验结果、训练历史、设定及验证记录。远端另保存 head_noise_scales.pt、特征分片及距离缓存。',
        'PLAN.json / DELIVERY_MANIFEST.json / LOCAL_VERIFICATION.json：冻结设定、文件哈希和下载核验。','',
        '文献',
        'Sun et al., ICML 2022, Out-of-Distribution Detection with Deep Nearest Neighbors. https://proceedings.mlr.press/v162/sun22d.html',
        'de Mathelin et al., JMLR 2025, Deep Out-of-Distribution Uncertainty Quantification via Weight Entropy Maximization. https://www.jmlr.org/papers/v26/23-1359.html',
        'Greenman et al., PLOS Computational Biology 2025, protein uncertainty benchmark. https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012639','']
    (OUT/'结果与使用说明.txt').write_text('\n'.join(lines),encoding='utf-8')
    write_json(OUT/('REMOTE_VERIFICATION.json' if args.existing else 'LOCAL_VERIFICATION.json'),dict(status='passed',archive_sha256=EXPECTED_ARCHIVE,files_verified=len(manifest),
        bootstrap_unit='measured test sequence, stratified by binary error label; paired across scores',bootstrap_repeats=1000,
        completed_cst=pd.Timestamp.now(tz='Asia/Shanghai').isoformat(),remote_root='/xiongyuchen/gfp_evidence_20261013'))
    handoff=ROOT/'reports/active_run_handoff.json';h=json.loads(handoff.read_text(encoding='utf-8'))
    h.update(updated_local=pd.Timestamp.now(tz='Asia/Shanghai').isoformat(),embedding_familiarity_status='complete',
        embedding_familiarity_directory='artifacts/discovery_v3/embedding_familiarity_v1',
        embedding_familiarity_report='artifacts/discovery_v3/embedding_familiarity_v1/结果与使用说明.txt',
        embedding_familiarity_definition='Learned z768/PLM proximity; Hamming is not a familiarity metric',
        maxwent_pilot_status='complete; auxiliary only due to weak high-brightness error discrimination',
        live_remote_status='Embedding extraction, common distance scoring and MaxWEnt pilot complete; verified via DONE and GPU idle',
        ssh_connectivity_last_verified='2026-10-14 connection through existing SOCKS VPN succeeded')
    write_json(handoff,h)
    print(top[['method','outside_q95_n','outside_highcal_q95_n','degree_mean']].to_string(index=False))
    print('Verified files:',len(manifest));print('Report:',OUT/'结果与使用说明.txt')

if __name__=='__main__':main()
