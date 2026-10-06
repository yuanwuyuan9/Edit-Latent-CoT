# 第一轮实验运行说明

目标：先确认自然错误和恢复推理的正确性，再观察逐位置干预的效果。置零或随机扰动有效不等于已经定位或修复了语义错误。

## 1. 获取代码与机器配置

首次在服务器拉取本项目。已有旧代码目录时，保留旧目录，先克隆到旁边验证；可按 README 的迁移步骤保持原路径：

```text
/data2/lsy/projects/Edit-Latent-CoT/causal-latent-cot
```

首次从 `configs/first_round/server.example.json` 复制生成 `server.json`，之后按机器修改。`server.json` 不受 Git 跟踪，拉取更新不会覆盖它。示例已使用现有服务器的数据路径；模型、结果目录位于项目父目录的 `models/` 和 `outputs/`。

## 2. 准备服务器环境与权重

以下命令在服务器运行。进入实验目录后，使用一个独立环境（建议 Python 3.10 或 3.11），安装适合服务器 CUDA 的 PyTorch，再执行：

```bash
cd /data2/lsy/projects/Edit-Latent-CoT/causal-latent-cot
cp -n configs/first_round/server.example.json configs/first_round/server.json
python -m pip install -r requirements-first-round.txt
git clone https://github.com/facebookresearch/coconut.git external/coconut
git -C external/coconut checkout 27273cb8cca4bb763c041a63b036d0c3b7cbbb48
python prepare_first_round_models.py --machine-config configs/first_round/server.json --dataset gsm8k
```

若 `external/coconut` 已存在，省略 clone，并先检查其未提交修改再切换版本。服务器已上传的 `latent.yaml` 使用另一套 Transformers 版本；第一轮使用本说明的独立环境，因为固定的官方 Coconut 代码采用旧版 tuple cache 接口。

下载脚本获取基础 GPT-2 和一个已训练 checkpoint，不下载整个模型集合；保存实际下载的 Hub revision，已有完整文件会复用。下载后的权重仍要通过严格加载与恢复检查。

来源：[公开 GSM8K 权重](https://huggingface.co/connordilgren/gpt2-gsm8k-coconut)、[公开 ProsQA 权重](https://huggingface.co/connordilgren/gpt2-prosqa-coconut)。两个 checkpoint 均先使用 6 个 latent 步骤，与对应 [GSM8K 配置](https://github.com/connordilgren/are-lrms-easily-interpretable/blob/main/args_coconut-cot-no_cot/gpt2_gsm8k_coconut.yaml) 和 [ProsQA 配置](https://github.com/connordilgren/are-lrms-easily-interpretable/blob/main/args_coconut-cot-no_cot/gpt2_prosqa_coconut.yaml) 一致。

## 3. 第一批只运行 GSM8K 基线

先预检查文件、数据和包版本，再运行前 20 个固定样本：

```bash
python run_baseline.py --config configs/first_round/gsm8k.json --machine-config configs/first_round/server.json --preflight
python run_baseline.py --config configs/first_round/gsm8k.json --machine-config configs/first_round/server.json --max-samples 20 --run-id gsm8k_baseline_pilot01
```

需要检查：权重严格加载是否成功；生成文本是否合理；正确、错误及解析失败数量。只从新生成的 continuation 提取答案，不从包含题目的全文提取数字。

运行目录：`/data2/lsy/projects/Edit-Latent-CoT/outputs/gsm8k_baseline_pilot01/`。每个 `run_id` 只能创建一次，重跑使用新名称。

**先回传这一批基线结果再决定扩大规模。** 20 条样本用于检查流程，不能代表完整 benchmark 的性能或机制结论。

## 4. 原样恢复验证

基线正常后，对其中前 5 个样本遍历全部 latent 位置：

```bash
python check_resume.py --config configs/first_round/gsm8k.json --machine-config configs/first_round/server.json --max-samples 5 --baseline-run /data2/lsy/projects/Edit-Latent-CoT/outputs/gsm8k_baseline_pilot01 --run-id gsm8k_resume_pilot01
```

检查独立官方 forward 与 wrapper 的 latent 向量、最终 prompt logits，以及恢复后的生成 token。随后在同一前缀上执行 identity → zero → identity，检查 embeddings、cache 和 logits 列表是否被修改。

`summary.json` 中 `checks_passed` 必须等于 `checks_total`（本例为 30），且 manifest 状态为 `completed`。未通过时停止干预效果分析，先排查实现。

## 5. 逐位置干预

验证通过后才能运行：

```bash
python run_interventions.py --config configs/first_round/gsm8k.json --machine-config configs/first_round/server.json --max-samples 5 --baseline-run /data2/lsy/projects/Edit-Latent-CoT/outputs/gsm8k_baseline_pilot01 --resume-check /data2/lsy/projects/Edit-Latent-CoT/outputs/gsm8k_resume_pilot01 --run-id gsm8k_interventions_pilot01
```

默认逐位置测试 identity、zero，以及强度为 0.1/0.5/1.0、种子为 0/1/2 的随机扰动（5 × 6 × 11 = 330 条记录）。随机扰动为加性修改，满足 `||delta|| = strength × ||h||`；不同强度共享相同种子的噪声方向。

程序要求有效的恢复验证记录，并检查代码、权重、数据和推理配置与基线一致；对新样本也先检查 identity。首次出现异常时停止并保留失败目录，不将异常结果混入正常统计。

比较不同位置的错误 → 正确、正确 → 错误、答案变化及后续 latent 变化。修复转移率只使用基线答案可解析的样本，另报基线解析失败数量。按操作、位置、强度和种子分组解释，不把不同操作重复记录后的总体准确率当成 benchmark 准确率。

## 6. 回传与本地分析

下载完整的单次运行目录；包括 `manifest.json`、`summary.json`、`predictions.jsonl`、最终配置、环境、样本列表、日志、源码快照、校验清单和选中的轨迹。目录中没有模型权重。

本地无需模型依赖即可重新评分：

```bash
python3 analyze_first_round.py --run-dir /本地路径/results/gsm8k_baseline_pilot01
```

脚本验证文件校验值、检查重复记录、重新评分，并生成汇总。干预结果生成 `by_condition.json`；基线分别生成 `natural_errors.json` 和 `parse_failures.json`。分析结果另存到 `analysis/<run_id>/`。

## 7. ProsQA 与扩大规模

完成 GSM8K 小规模流程后，下载 ProsQA checkpoint，使用 `configs/first_round/prosqa.json` 重复相同流程，并使用独立的 baseline 和恢复记录。

```bash
python prepare_first_round_models.py --machine-config configs/first_round/server.json --dataset prosqa
```

`--max-samples 0` 表示整个 split；默认只取前 5 条。扩大干预范围时，对应 baseline 必须覆盖所有选中样本；可以用同一配置下的小规模恢复验证作为入口检查，每个实际干预样本仍会执行 identity 验证。

第一轮先报告自然错误数量和 latent 干预效应，再选择值得研究的案例。ProsQA 如果几乎没有自然错误或不依赖 latent，则据实报告，不通过制造错误替代自然失败。

## 8. 本地验证入口

```bash
python3 -m unittest experiments.first_round.test_data -v
../.env/bin/python -m unittest experiments.first_round.test_tiny_model -v
../.env/bin/python -m unittest experiments.first_round.test_pipeline -v
```

第二组使用随机初始化的微型 GPT-2，在 CPU 上检查计算与状态处理，不下载 checkpoint，不代表真实模型的正确率或修复效果。

GitHub Actions 使用 CPU 运行相同检查。实际数据未下载时，仅跳过真实数据集检查，数据适配仍通过独立小样本验证。克隆后也可以在已安装依赖的环境中直接使用 `python -m unittest ...`。
