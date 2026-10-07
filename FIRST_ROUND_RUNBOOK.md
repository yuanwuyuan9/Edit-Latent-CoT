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

## 9. 成功案例的后续反馈对照

GSM8K 的 5 条 pilot 中，样本索引 4 的随机干预成功集中在第 1–3 步。后续扫描保存了所有样本的完整向量，可用于检查成功是否依赖后续反馈输入的变化。

`run_path_controls.py` 从已完成的干预目录读取向量，在相同问题上交叉组合四种输入：原轨迹、完整修改轨迹、仅修改当前位置、仅保留修改后的后续向量。所有 latent 输入固定，Transformer 隐藏状态及答案生成重新计算；并不冻结 KV cache 或全部后续计算。

```bash
git pull --ff-only
python run_path_controls.py \
  --config configs/first_round/gsm8k_followup.local.json \
  --machine-config configs/first_round/server.json \
  --input-run /data2/lsy/projects/Edit-Latent-CoT/outputs/gsm8k_interventions_followup01 \
  --sample-index 4 \
  --run-id gsm8k_path_controls_pilot01
```

默认选择第 1–3 步、强度至少 0.5 的所有随机条件，包含成功及失败对照：90 个来源条件 × 4 个分支 = 360 条记录。运行要求当前代码、权重、数据和环境与来源目录的第一轮签名一致。新入口及其测试单独保存源码和校验值，不改变既有恢复接口。

每个条件先验证原轨迹回放与基线一致、完整修改轨迹回放与该条件原始生成一致；任一不一致则停止，不能解释混合轨迹结果。输出目录可使用 `analyze_first_round.py` 校验与重新评分。

该实验使用已观察到的轨迹，是 oracle 机制诊断。比较两种混合轨迹能检查后续反馈向量的作用，但不直接证明语义修复，也不表示冻结输入向量后剩余 Transformer 计算没有变化。

本地验证：`python -m unittest test_path_controls -v`，包含输入组合、全上下文回放、异常输入检查及微型模型端到端导出与评分。

## 10. 单个后续向量移植，恢复自然反馈

前面的混合轨迹对照固定了所有 latent 输入。要检查一个后续向量能否单独传递有用的信息，可把来源轨迹在后续位置的一个向量移植到原始问题的原始前缀，再让 Coconut 自主计算剩余反馈与答案。

```bash
git pull --ff-only
python run_path_controls.py \
  --mode transplant \
  --config configs/first_round/gsm8k_followup.local.json \
  --machine-config configs/first_round/server.json \
  --input-run /data2/lsy/projects/Edit-Latent-CoT/outputs/gsm8k_interventions_followup01 \
  --sample-index 4 \
  --run-id gsm8k_latent_transplants_pilot01
```

仍使用第 1–3 步、强度至少 0.5 的 90 个来源条件，包含 22 个成功来源及 68 个失败来源。逐一尝试来源修改位置之后的每个 latent 位置：第 1 步来源有 5 个目标位置，第 2 步有 4 个，第 3 步有 3 个，共 360 条移植记录。90 个来源条件不是独立题目。

移植前复现来源的自然轨迹；每个目标位置先检查 identity 能复现基线。之后只插入一个来源向量，使用原始前缀与缓存，重新计算全部后续反馈向量。记录来源步骤、目标步骤、来源正确性、实际移植幅度、答案及完整输出轨迹。检查原始前缀未变且目标位置恰好插入指定向量；异常立即停止。

这是同题、oracle 来源的机制诊断。若成功，只支持该后续反馈向量在指定原始前缀下提供了足以改变答案的信息；其语义内容、迁移性和错误起源仍需其他证据。失败来源作为对照，不能仅比较成功来源的最佳结果。

## 11. 固定编辑，改变问题条件

单向量移植已在样本 4 的第 5 步得到一次成功。下一轮固定这一编辑，检查它能否随题目条件产生不同答案，而非重复输出原题答案 20。使用第 1 步、强度 1.0、种子 9 的来源轨迹，定义 `delta = donor_h5 - original_h5`；在每个新问题自己的第 5 步输入上加同一个 delta，保留原始前缀并自然重算后续反馈。

```bash
git pull --ff-only
python run_counterfactual_family.py \
  --config configs/first_round/gsm8k.json \
  --machine-config configs/first_round/server.json \
  --source-run /data2/lsy/projects/Edit-Latent-CoT/outputs/gsm8k_interventions_followup01 \
  --transplant-run /data2/lsy/projects/Edit-Latent-CoT/outputs/gsm8k_latent_transplants_pilot01 \
  --run-id gsm8k_counterfactual_family_pilot01
```

无需手动传输编辑向量或新数据。程序从服务器已有输出提取向量，核对配置中预先固定的来源轨迹及向量校验值，生成问题变体，并保存数据、编辑、协议及代码快照。来源数据、模型、环境和原恢复接口仍须与已有实验签名匹配；新实验单独计算包含变体数据和新代码的签名。

原题 20 只鸡先用于复现检查，之后测试 16、18、22、24、26、28、30 只鸡；每题只替换鸡的数量，正确答案为 `3*n-15-25`，分别为 8、14、26、32、38、44、50。每题运行基线、identity、正向 delta、反向 delta，以及种子 100–109 的 10 个等绝对范数随机方向：8 × 14 = 112 条记录。所有方向与幅度在观察变体结果前固定；不按每个新问题的向量范数重新缩放。

任一问题的 identity、前缀保持、向量插入或共享缓存检查失败即停止；原题正向编辑须复现既有移植答案。回传完整 `outputs/gsm8k_counterfactual_family_pilot01/`。`summary.json` 按条件分开汇总原题和 7 个新变体，报告修复、破坏及输出 20 的次数。比较正向编辑与反向、随机对照的全部答案，不把 112 条记录当作 112 道题。

这是同一道题的合成受控变体，编辑由已知成功案例事后选定。即使结果随数量正确变化，也仅提供行为层面的迁移证据，不能单独证明语义因素分离或自动错误定位。真实模型效果须在服务器上运行；本地 `python -m unittest test_counterfactual_family -v` 使用微型模型检查完整流程、自然反馈及统计口径。

## 12. 每道题自身的第 5 步 donor 可修复性

固定原题编辑在 7 个数量变体上均未修复。下一轮检查：每个变体是否存在由本题候选轨迹产生的第 5 步向量，能在该题原始前缀下修正答案？移植位置保持第 5 步，用于区分该位置的同题可修复性和原题固定方向的迁移能力。

```bash
git pull --ff-only
python run_same_question_donors.py \
  --config configs/first_round/gsm8k.json \
  --machine-config configs/first_round/server.json \
  --family-run /data2/lsy/projects/Edit-Latent-CoT/outputs/gsm8k_counterfactual_family_pilot01 \
  --run-id gsm8k_same_question_donors_pilot01
```

程序读取上一轮已保存的问题、基线与固定方向结果。每题在第 1–3 步分别施加强度 0.5、0.75、1.0、种子 0–9 的随机干预，自然生成 90 条来源轨迹；使用与原题上一轮相同的有限搜索预算。不根据来源回答正确与否筛选候选，也不在成功后提前停止。

对每条来源轨迹，取本题第 5 步 donor，并从本题原始前缀执行三种分支：插入 donor；施加反向位移；施加一个等位移范数的随机方向。随机对照种子预先固定为 `1000 + 候选序号`，每题各 90 次，与 donor 的搜索预算相同。三个分支都自主重算第 6 步与答案，不复制来源轨迹后缀，不将正确答案输入模型。

每题记录基线 1 条、第 1–3 和第 5 步 identity 4 条、原题固定方向复现 1 条，以及 90 ×（来源 + donor + 反向 + 随机）360 条，共 366 条。8 题共 **2,928 条记录**。每个候选保存完整反馈输入及来源链接；检查 identity、原始前缀、单向量插入与共享缓存不变。原题已知成功的 donor 路径还须复现，异常即停止并保存失败目录。

回传完整 `outputs/gsm8k_same_question_donors_pilot01/`。`summary.json` 逐题报告各分支全部尝试的正确率、修复/破坏计数，以及是否至少有一个 donor 成功；成功来源和失败来源均保留。比较 donor、反向和随机分支的相同搜索预算，不把候选数量作为独立题目数，原题与 7 个相关变体分开解释。

若新变体中同题 donor 成功而原题固定方向失败，支持该题在第 5 步存在局部修复，同时显示共享固定方向的限制。若没有成功，只说明此次候选生成方式与预算未找到修复，不能否定其他状态、位置或编辑方式。成功仍属于使用 gold 判定后的 oracle 可行性证据，尚不能定位错误起源或识别具体语义因素。

本地验证：`python -m unittest test_same_question_donors -v`，检查搜索预算、等范数对照、原始前缀与自然反馈、缓存隔离，以及包含失败来源的端到端导出和离线评分。

## 13. 受范数约束的 oracle 单向量优化

同题 donor 搜索在 7 个新变体上未找到成功，来源轨迹也全部回答错误。下一轮直接用正确答案损失寻找第 5 步输入，检验限定幅度内的可修复性。正确答案只用于优化和离线评分；成功仍由不提供答案的自由生成判断。

```bash
git pull --ff-only
python run_oracle_latent_optimization.py \
  --config configs/first_round/gsm8k.json \
  --machine-config configs/first_round/server.json \
  --family-run /data2/lsy/projects/Edit-Latent-CoT/outputs/gsm8k_counterfactual_family_pilot01 \
  --run-id gsm8k_oracle_latent_optimization_pilot01
```

冻结模型参数及第 5 步之前的输入。令修改为 `h5 + ||h5|| * u`，约束 `||u|| <= radius`。相对半径预先固定为 0.1、0.25、0.5、1.0；每个半径使用种子 0、1、2，各执行 80 次更新。种子 0 从零位移开始，其他初始化为半径的 10% 的随机位移。使用归一化梯度下降，每步长度为半径的 5%，更新后投影回约束球。

目标是 `### {gold_answer}` 加 EOS 的平均 token NLL，采用 teacher forcing；所有后续 latent 输入自然重算并保留梯度。每次运行保存 81 个迭代的损失、位移范数和梯度范数，选取最低 NLL 的迭代。最终分支另存每个目标 token 的 NLL，便于区分数字答案、格式和 EOS 的变化。不会按生成正确性挑选迭代，也不会答对后提前停止或调整协议。

现有恢复接口生成后续反馈时会 detach；新的梯度入口按相同的逐段缓存调用顺序生成反馈，同时保留后续反馈和缓存中的梯度。原接口和旧实验签名不变。初版使用完整上下文重算，与缓存路径的浮点计算顺序不同；修订版对齐调用顺序，原数值阈值保持不变。CPU 测试以独立完整上下文参考检查函数值和梯度，服务器每次评估继续核对反馈向量与 teacher-forcing logits。自由生成始终使用原接口。

检查失败时停止，并在 `logs/teacher_mismatch.json` 分别记录反馈和 logits 的误差、未通过元素数、阈值及具体分支；`logs/teacher_mismatch.pt` 保存实际候选、两侧向量和 logits。失败目录保留；修订后使用新的 run-id 重跑，例如 `gsm8k_oracle_latent_optimization_pilot02`。

每个优化候选再评估反向位移及种子 3000–3009 的 10 个等位移范数随机方向；所有分支只修改第 5 步、保持原始前缀并自然重算后续反馈。每题另保存基线、identity 和上一轮固定方向复现。8 题 ×（3 + 4 半径 × 3 初始化 × 12 分支）= **1,176 条生成记录**，另保存 96 次优化的完整历史和向量轨迹。

运行结束检查参数值和梯度、前缀与缓存均未改变。回传完整 `outputs/gsm8k_oracle_latent_optimization_pilot01/`，包含 `optimization/`；按题目、半径和初始化报告自由生成成功率与损失变化，原题和新变体分开。随机对照匹配位移范数，计算搜索预算与梯度优化不同，不能据此宣称算法性能公平优于随机搜索。

若成功，仅说明在该位置和幅度内找到能使答案正确的输入，尚不能证明修复了语义因素或定位了错误起源。若失败，只说明当前目标、优化器和预算未找到修复，不能证明不存在可修复状态。本地 `python -m unittest test_oracle_latent_optimization -v` 验证有限差分梯度、后续反馈梯度、投影约束、缓存/权重保持，以及自由生成与 teacher forcing 的分离。

`pilot02` 已完成并通过离线核验。7 个新变体在相对半径 0.25 下有 6 个找到正确生成，在 0.5 下全部找到；这支持答案可达性，尚不能区分推理修复和答案引导。结果与下一轮指定错误答案对照的设计见 [pilot02 分析](docs/ORACLE_LATENT_OPTIMIZATION_PILOT02.md)。

## 14. 正确与指定错误答案的同预算优化

直接优化使用了正确答案监督。下一轮固定两个错误目标，检查相同接口是否也能导向这些答案。每题优化真实答案 `y`、`y-4`、`y+4`；这两个错误目标在当前 8 题上均为正数，且不同于该题 baseline 答案，不按运行结果选择目标。

```bash
git pull --ff-only
python run_target_answer_controls.py \
  --config configs/first_round/gsm8k.json \
  --machine-config configs/first_round/server.json \
  --oracle-run /data2/lsy/projects/Edit-Latent-CoT/outputs/gsm8k_oracle_latent_optimization_pilot02 \
  --run-id gsm8k_target_answer_controls_pilot01
```

直接继承 `pilot02/protocol.json`：位置为第 5 步，相对半径 0.1、0.25、0.5、1.0，初始化种子 0、1、2，各执行 80 次更新，以最低平均目标 token NLL 选择迭代。优化器和梯度入口复用上一轮代码，不修改原入口或恢复接口。每次搜索从本题原始输入和缓存开始，三个目标使用相同随机初始化方向、相同半径及更新规则，不使用另一目标优化后的向量初始化。

本轮只评估优化候选，不重复反向与随机方向分支；主要比较三个目标的可达性，预算在三个目标间一致。8 题 × 3 目标 × 4 半径 × 3 初始化 = **288 次优化**，每次保存 81 个迭代损失。每题另存 baseline 和 identity，合计 **304 条自由生成记录**。

先分别回放 `pilot02` 保存的 96 个 gold 输入，必须复现其生成 token IDs、完整 latent 输入（原数值阈值）及目标 NLL（误差最多 `1e-4`）；baseline 和 identity 也须复现。重新优化得到的 gold 向量是本轮独立搜索的结果，与旧向量的差异另行记录，不替换成旧候选，也不据此筛除本轮结果。三个目标仍使用相同算法、目标函数形式和更新预算。候选插入、前缀保持、缓存隔离、范数投影及 teacher/cache 一致性继续严格检查。最后校验权重值及梯度未变。推理回放或这些检查异常则停止并保存失败目录，原输出不被覆盖。

初版把“重新优化得到相同向量”和“同一向量复现推理”合并成了停止门槛。修订版分开这两个检查，未放宽任何原数值阈值，也未改优化器、随机种子、迭代选择或成功判据。跨运行搜索结果可能不同，具体原因不能仅凭旧报错判定。每个 gold 条件在 `logs/gold_reproduction/` 保存两种比较：token IDs、分位置向量差值、NLL 差值、最佳迭代以及优化历史首次逐值差异。独立搜索不匹配时另存 `.pt` 张量，汇总为 `gold_search_differences`；真实固定输入回放失败则保存 `logs/gold_replay_mismatch.json` 和 `.pt` 并停止。

每个目标保存实际文本、token IDs、长度及原始输入下的 NLL 到 `optimization/<index>_targets.json`。记录 `target_answer`、`target_offset`、`target_hit` 和 `correct`：`target_hit` 表示自由生成命中所指定的优化目标；`correct` 始终表示命中真实 gold。成功生成错误目标不算修复。`summary.json` 按题、目标和半径分别统计两种结果，并记录最小已测试成功半径；并非真实最小编辑范数。原题与 7 个相关变体分开解释。

回传完整 `outputs/gsm8k_target_answer_controls_pilot01/`，包含全部 `optimization/` 和 `traces/`。若发生 teacher/cache 不匹配，诊断文件继续保存至 `logs/teacher_mismatch.json` 与 `.pt`。目标数值和分词长度可能影响可达难度；两个错误目标的结果不能代表所有错误答案。

初版失败目录保留；修订版使用新的 run-id `gsm8k_target_answer_controls_pilot02` 重跑。生成记录仍为 304 条，额外的 96 次固定输入回放只作一致性核验，不纳入目标命中统计。

若两个错误目标也在相近半径和预算下普遍可达，当前优化成功不足以证明推理修复；若 gold 更容易达到，也只支持当前题族和目标集合下的相对可达性。此对照本身不定位原错误，不识别中间语义因素。CPU `python -m unittest test_target_answer_controls -v` 检查完整导出、旧 gold 搜索复现、协议继承、目标/真实正确性分离、缓存一致性及离线重评分。
