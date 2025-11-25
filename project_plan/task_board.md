任务看板（当前版本）
====================

说明：这是一个轻量级文本看板，方便在 Git 里追踪任务状态。  
状态分为：`TODO` / `IN_PROGRESS` / `DONE`。  
执行过程中可以直接在 PR 或本地修改此文件。

Phase 1（P1）：数据接口 & 基础特征
----------------------------------

- [DONE] A: 统一 60 天数据加载接口（`DatasetLoader`）
- [DONE] A: EDA 脚本（绘制 mood 曲线、缺失率等）（当前实现为终端统计与验证输出）
- [DONE] B: 从 behaviors/emotions 抽 5–10 个稳定的日级特征（见 `features/basic_features.py`）
- [DONE] B: 产出 `X_daily (60×F)` + `y_daily` 的 Numpy/Pandas 版本（见 `scripts/export_daily_features.py` 输出）
- [DONE] C: 对现有 BFI 工具与 persona 做一次梳理，整理为实验指标规范文档（`project_plan/metrics_and_evaluation.md`）

Phase 2（P2）：时间序列表示 & 人格回归
--------------------------------------

- [DONE] A: 实现 7 天滑动窗口统计 + 指数衰减基线模块（见 `features/temporal_features.py` 与 `scripts/export_temporal_features.py`）
- [DONE] B: 实现 BiLSTM/GRU + 注意力编码器（见 `models/sequence_encoder.py`）
- [DONE] B: 基于 temporal 特征完成一次自监督序列预训练（重构 rolling_stats），见 `scripts/train_sequence_encoder.py` 与 `logs/selfcheck_p2_sequence_autoencoder_isabella_irl_3d_clean.md`
- [TODO] C: 在有足够 persona+BFI 标签后，封装统一训练脚本 `train_personality_regressor.py`，完成「轨迹 → Big Five 预测」基准实验并记录指标

Phase 3（P3）：IRL / 偏好建模 MVP
-------------------------------

- [TODO] A: 与 B/C 共同定义动作空间与奖励 proxy 的编码
- [TODO] A: 将状态/动作/奖励打包为「专家轨迹」数据结构
- [TODO] B: 视时间对序列编码器做小规模改进（phase/day embedding）
- [TODO] C: 实现 `learning/irl_mvp.py`（或等价）并跑通一组 IRL 实验
- [TODO] C: 输出一份 IRL 实验报告，包含与 Big Five / BFI 的对齐分析

备注：当前阶段默认使用 `data/isabella_irl_3d_clean` 作为主要 60 天假数据源。  
后续接入真实数据时，本看板可以扩展出「Real 数据阶段」的专门任务区块。
