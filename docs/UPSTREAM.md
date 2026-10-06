# 上游来源与修改记录

本项目在 [J1mL1/causal-latent-cot](https://github.com/J1mL1/causal-latent-cot) 基础上开发，导入版本为：

```text
20638ecdb56f24a654ca2047b481e8d3943f6c57
```

原论文：Li et al., *Dynamics Within Latent Chain-of-Thought: An Empirical Study of Causal Structure*，[论文链接](https://arxiv.org/abs/2602.08783)。原仓库说明保存在 [UPSTREAM_README.md](UPSTREAM_README.md)。

本项目新增第一轮实验入口、数据适配与评分、恢复一致性检查、逐位置干预、结果导出和离线分析。Coconut wrapper 增加严格权重加载、continuation 与轨迹输出，并修复恢复过程中修改共享 logits 列表的问题。README 与开发文档按本项目流程整理，数据文件从版本控制中移除。

原有 RQ1–RQ4 脚本和其他模型封装保留为参考；当前自动检查只覆盖 `experiments/first_round/`，不表示所有上游实验均已复现。

上游许可证保留在 [LICENSE](../LICENSE)，为 CC BY-NC 4.0；本项目不将继承代码改为其他许可证。外部 Coconut 代码单独从 [facebookresearch/coconut](https://github.com/facebookresearch/coconut) 获取，使用其自身许可证，第一轮固定版本为 `27273cb8cca4bb763c041a63b036d0c3b7cbbb48`。

研究中使用了 Dilgren & Wiegreffe 的 [Are Latent Reasoning Models Easily Interpretable?](https://arxiv.org/abs/2604.04902) 提供的 checkpoint 和配置。模型权重与数据集均单独获取，不随本仓库发布。
