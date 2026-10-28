# Complete ppluGFP Pipeline Code (GLOM → TG-ITO)

## 目录
- glom_model/         GLOM 亮度预测模型（训练+架构）
- terrain_umap/       19张UMAP地势图生成+可视化
- tgito/              TG-ITO 核心模块
- run_generate.py     单次攀爬入口
- modal_*.py          Modal 云端入口（训练/批量/评分）
- scripts/      pipeline 脚本

## 流程
1. train_pplu_full_model.py  → 训练 GLOM (31480 pplu)
2. pplu_terrain_pipeline.py  → 生成 19 UMAP + terrain
3. modal_build_allhigh_miner_gate.py → 12源 gate 训练 (7500高亮)
4. modal_make_wide_exploration_plan.py → 900山峰规划
5. modal_tgito_batch.py → 2700次 TG-ITO 攀爬
6. modal_candidate_confidence.py → TOP6 筛选

## 关键参数
- mutation_budget=2, steps=70, restarts=1
- terrain_weight=0.03, gate=risk-top5 (动态选5源)
- cos0 模式: realism=0, repair=0, miner=0
