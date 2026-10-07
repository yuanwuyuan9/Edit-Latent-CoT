# Edit-Latent-CoT

[![First-round checks](https://github.com/yuanwuyuan9/Edit-Latent-CoT/actions/workflows/checks.yml/badge.svg)](https://github.com/yuanwuyuan9/Edit-Latent-CoT/actions/workflows/checks.yml)

研究 latent 推理中的自然错误：检查 latent 计算的作用，寻找候选问题位置，并探索局部修改后模型能否自行计算后果。

当前实现第一轮实验平台：**基线评测 → 原样恢复验证 → 逐位置干预 → 离线分析**。首个模型为 GPT-2 + Coconut，数据集为 GSM8K 和 ProsQA。已完成 CPU 检查及 GSM8K 的 20 条基线、5 道题的干预 pilot；完整 benchmark 和 ProsQA 实验仍待验证。成功案例的反馈对照、单向量移植、受控问题变体和同题 donor 搜索入口见 [运行说明](FIRST_ROUND_RUNBOOK.md)。

本项目基于 [causal-latent-cot](https://github.com/J1mL1/causal-latent-cot) 开发，上游来源、许可证及修改记录见 [docs/UPSTREAM.md](docs/UPSTREAM.md)。

## 代码与数据目录

仓库根目录就是原来的 `causal-latent-cot/` 代码目录，不再嵌套同名文件夹：

```text
Edit-Latent-CoT/                 # 服务器上的项目父目录
├── causal-latent-cot/           # 本仓库的 checkout
│   ├── run_baseline.py
│   ├── check_resume.py
│   ├── run_interventions.py
│   ├── analyze_first_round.py
│   ├── experiments/first_round/
│   ├── configs/first_round/
│   ├── common/
│   └── docs/
├── data/                       # 单独准备，不进入 Git
├── models/                     # 单独下载，不进入 Git
└── outputs/                    # 服务器实验结果
```

继承的 RQ1–RQ4 脚本保留供参考。第一轮使用 `requirements-first-round.txt`，其环境与上游 `latent.yaml` 分开管理。

## 首次安装

建议在独立的 Python 3.10/3.11 环境中，先安装适合服务器 CUDA 的 PyTorch，再执行：

```bash
git clone https://github.com/yuanwuyuan9/Edit-Latent-CoT.git causal-latent-cot
cd causal-latent-cot
python -m pip install -r requirements-first-round.txt
git clone https://github.com/facebookresearch/coconut.git external/coconut
git -C external/coconut checkout 27273cb8cca4bb763c041a63b036d0c3b7cbbb48
cp -n configs/first_round/server.example.json configs/first_round/server.json
```

编辑 `server.json` 中的路径与设备。它被 Git 忽略，只需首次创建，后续更新代码时保留。

准备 `../data/gsm8k/test.jsonl` 和 `../data/prosqa/test.json`，然后下载需要的权重：

```bash
python prepare_first_round_models.py --machine-config configs/first_round/server.json --dataset gsm8k
# 开始 ProsQA 实验时，再将 --dataset 改为 prosqa。
```

GSM8K checkpoint 在 GSM8K-Aug 上训练，第一轮评测使用原始 GSM8K test。权重来源与完整实验命令见 [FIRST_ROUND_RUNBOOK.md](FIRST_ROUND_RUNBOOK.md)。

## 第一批实验

先检查环境，再运行 GSM8K 前 20 条固定样本：

```bash
python run_baseline.py --config configs/first_round/gsm8k.json --machine-config configs/first_round/server.json --preflight
python run_baseline.py --config configs/first_round/gsm8k.json --machine-config configs/first_round/server.json --max-samples 20 --run-id gsm8k_baseline_pilot01
```

确认生成与答案解析正常后，依次运行 `check_resume.py` 和 `run_interventions.py`。干预入口要求已通过的恢复验证，并检查对应代码、权重、数据和推理设置。每次运行使用新 `run_id`，保留配置、版本、日志、逐条结果和选中的轨迹。

下载完整运行目录后，本地可不加载模型直接分析：

```bash
python3 analyze_first_round.py --run-dir ../results/gsm8k_baseline_pilot01
```

分析程序验证校验值、重新评分，并分别导出自然错误与解析失败。置零或随机扰动改变答案，只说明干预效应，不能直接认定已经定位或修复语义错误。

## 服务器更新

已有本项目 checkout 后，先确认没有正在使用该目录运行的任务，再拉取：

```bash
cd /data2/lsy/projects/Edit-Latent-CoT/causal-latent-cot
git status --short
git pull --ff-only
```

若有本地代码修改，先保存或提交；若实验正在运行，在另一个 checkout 中更新。依赖变化后重新安装并运行小规模检查。

首次从之前手动上传的上游代码迁移时，在没有运行任务的情况下备份旧目录后重新克隆。下例执行前确认备份名称未被使用：

```bash
cd /data2/lsy/projects/Edit-Latent-CoT
mv causal-latent-cot causal-latent-cot.before-github
git clone https://github.com/yuanwuyuan9/Edit-Latent-CoT.git causal-latent-cot
```

`data/`、`models/` 和 `outputs/` 保持在父目录；旧目录中的自定义文件和机器配置按需迁移。之后按首次安装步骤准备外部依赖和环境。

## 本地与 GitHub 检查

在已经安装 CPU PyTorch、第一轮依赖及固定版 Coconut 的环境中：

```bash
python -m unittest experiments.first_round.test_data experiments.first_round.test_tiny_model experiments.first_round.test_pipeline -v
```

检查涵盖答案解析、实验前置条件、恢复一致性、分支状态隔离、扰动尺度、结果保存及离线评分。真实数据文件缺失时，该项集成检查跳过；小样本适配与微型模型检查继续运行。CI 不下载真实模型权重，不运行 GPU benchmark。

## 文档

- [第一轮开发计划](docs/FIRST_ROUND_DEVELOPMENT_PLAN.md)
- [本地开发与远程实验规范](docs/LOCAL_REMOTE_WORKFLOW.md)
- [服务器运行说明](FIRST_ROUND_RUNBOOK.md)
- [上游来源与修改记录](docs/UPSTREAM.md)

## 许可证

保留上游 [CC BY-NC 4.0 许可证](LICENSE)。外部依赖、数据集和模型权重按各自的来源与许可证管理。
