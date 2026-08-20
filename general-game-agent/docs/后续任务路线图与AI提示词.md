# 后续任务路线图 + AI 开发提示词手册（v1 · 2026-08-20 晚）

> 适用：第 2 天晚 → 第 9 天（8/27）结课。基于开发计划 v4.0 + 实测现状（M3 已三跑、M2 500 帧已合并、归档口径已回填）。
> 每个任务给：**精确命令 → 验收标准 → AI 提示词**（提示词按课程"项目背景/技术栈/困难点/期望输出"格式编写，可直接复制）。
> 提示词使用铁律：每次只问一件事；把本手册对应段的【背景】原样贴给 AI；AI 给代码后自己通读一遍再合入（审查职责）。

---

## 0. 当前状态速览（实测核对，2026-08-20 晚）

| 任务 | 状态 | 证据 |
|------|------|------|
| M2 500 帧 / 测试 200 帧数据 | ✅ | `workspace/*.parquet`（M2=500帧10序列，TEST=200帧4序列，25列） |
| M2 统计 + 12 张图 | ✅ | `workspace/m2_analysis/`（含 sequences/ 10 张） |
| serve.py 冒烟 + 探针 | ✅ | `scripts/probe_serve.py` + `probe/probe_stats.csv` |
| m3_eval.py 评测脚本 | ✅ | 支持 `--input/--offset/--limit/--save/--out-dir` |
| M3 评测 200 帧 | ✅ **3/3 run** | `metrics.csv`（r=0.0049）+ `run1/`（r=0.0188）+ `run3/`（r=0.0005）；均值±std 已回填归档口径 |
| M2 500 帧预测 | ✅ **3/3 块已合并** | `m2_blk1`(seq0-3)、`m2_blk2`(seq4-7)、`m2_blk3`(seq8-9) → `predictions_m2.parquet` + `metrics_m2.csv` |
| 指标归档口径 | ✅ | `docs/指标归档与实验口径.md` |
| T5.1 对比表 / T6.1 可视化 / T7.1 集成 / T8.1 自测 / T9.1 报告 | ⏳ | — |

---

## 1. 依赖关系（谁必须先做）

```
T4.1② M2 预测（第三块+合并） ──→  T6.1 viz_diff 20 张图（第 6 天）
T4.2 第三跑（重启 serve）    ──→  归档口径回填"均值±std"（文档第 1 节）
T5.1 对比表 ← 依赖已有的 predictions.parquet（随时可做）
T6.2 帧连续性 ← 只读检查（随时可做，5 分钟）
T7.1 run_all.py ← 依赖 m3_eval.py 与 viz_diff.py 都稳定
T8.1 README+自测表 → T9.1 大报告
```

**结论：今晚两件事（M2 第三块+合并、M3 第三跑）已于 2026-08-20 深夜全部完成，T4 已 100% 收官。** 后续按 Step 3~9 推进即可。

---

## 2. 分步执行

### Step 1（今晚）：T4.1② 收尾——M2 第三块 + 合并

**前提**：serve.py 正在运行（若无则重新启动，见 Step 2 启动命令）。

```powershell
cd C:\Users\HP\Desktop\general-game-agent

# ① 第三块：seq 8-9 共 100 帧（offset=400, limit=100）
E:\tools\Anaconda3\envs\nitrogen\python.exe scripts/m3_eval.py --input workspace/m2_elden_ring_500frames.parquet --offset 400 --limit 100 --save --out-dir workspace/m3_eval/m2_blk3

# ② 合并三块 → predictions_m2.parquet + metrics_m2.csv
E:\tools\Anaconda3\envs\nitrogen\python.exe scripts/merge_predictions.py --parts workspace/m3_eval/m2_blk1 workspace/m3_eval/m2_blk2 workspace/m3_eval/m2_blk3 --label workspace/m2_elden_ring_500frames.parquet --out-dir workspace/m3_eval
```

**验收**：`workspace/m3_eval/predictions_m2.parquet`（500 行）与 `metrics_m2.csv` 存在；`metrics_m2.csv` 的 `n_frames=500`。

> 若第三块评测中途断连：重跑同一条命令即可（`--out-dir` 会覆盖该块产物，无副作用）。

---

### Step 2（今晚）：T4.2 第三跑——重启 serve 独立重跑 M3

**关键**：必须**先关掉旧 serve 再启动新的**（采样随机性在服务端进程内，重启才是"独立 run"）。

```powershell
# ① 在 serve 所在终端按 Ctrl+C 关闭旧服务

# ② 新开终端启动（无条件模式，无 input() 交互）
cd C:\Users\HP\Desktop\general-game-agent\NitroGen-main
$env:HF_ENDPOINT="https://hf-mirror.com"   # 保险起见走镜像
E:\tools\Anaconda3\envs\nitrogen\python.exe scripts/serve.py ..\ng.pt --port 5555

# ③ 回到项目根目录，新开终端跑第三 run
cd C:\Users\HP\Desktop\general-game-agent
E:\tools\Anaconda3\envs\nitrogen\python.exe scripts/m3_eval.py --save --out-dir workspace/m3_eval/run3
```

**验收**：`workspace/m3_eval/run3/metrics.csv` 存在；按键准确率落在 0.87~0.91、r 落在 -0.05~+0.05（与已有两跑一致即"结论稳定"）。

**随后**：把三跑数值合并成"均值±std"，回填 `docs/指标归档与实验口径.md` 第 1 节（把"Run1/Run2"两列扩为"均值±std"或加一行汇总）。git 提交（commit message 含 `T4.2 三跑`）。

---

### Step 3（第 3 天 8/21）：当日报告（主题：主路径数据/对象、成功与失败约定）

第 3 天本身任务已全部提前完成，当日以**补写报告 + 顺手收尾**为主。

**AI 提示词（第 3 天报告）：**

> 【背景】我在做腾讯 IEG"通用游戏智能体"课题（基于 NVIDIA NitroGen），今天是实践第 3 天，需要写当日报告。项目已完成：serve.py 推理服务跑通（无条件模式，`game_mapping_cfg: null`）、M3 评测脚本 m3_eval.py 完成并出数（200 帧 × 2 run：按键准确率 0.9044/0.9056 但全零基线 0.9203 更高，precision 0.103/0.121、recall 0.026/0.030，摇杆 MSE 0.308/0.285，Pearson r 0.005/0.019 未达 0.4）、M2 500 帧预测分块评测中（2/3 块完成）。评测方式：随机噪声图占位 + 无条件先验 + 自回归采样，无逐帧对齐。课程要求今日报告主题是"主路径数据/对象、成功与失败约定、临时假实现"。
> 【技术栈】Windows 11 + conda nitrogen + Python 3.12 + parquet/pandas。
> 【困难点】报告要体现"主路径的数据流"：数据 parquet（17 布尔按键 + 4 摇杆 + seq_id/frame_idx）→ m3_eval.py → ModelClient(predict) → 21 维 buttons + 双摇杆 → 名字级映射+阈值化 → metrics.csv。"成功与失败约定"包括：摇杆尺度校验失败报错、块内 reset() 清空缓冲、指标列与基线列的判定逻辑。"临时假实现"包括：随机噪声图占位（数据集无视频）、无条件模式（无游戏 ID）。
> 【期望输出】一份当日报告 markdown：基本信息 / 今日目标 / 今日完成 / 主路径数据与对象（画 ASCII 流程图）/ 成功与失败约定表 / 临时假实现清单 / 自检 / 问题与明日计划。600 字左右，别堆数字。

---

### Step 4（第 4 天 8/22）：当日报告（主题：启动说明、目录职责、审查纪要）

**AI 提示词（第 4 天报告）：**

> 【背景】我在做腾讯 IEG"通用游戏智能体"课题（NVIDIA NitroGen），实践第 4 天。已完成：M3 评测三跑（均值±std 已汇总，按键准确率约 0.89±0.02 劣于全零基线、摇杆 r≈0.01±0.01 未达 0.4、MSE 约 0.30±0.02）、M2 500 帧预测完成（predictions_m2.parquet + metrics_m2.csv）、指标归档口径文档已写（docs/指标归档与实验口径.md：主报 precision/recall + MSE，r 参考列，未达标原因=无条件先验+无逐帧对齐）。目录结构：scripts/ 下 scan_games、extract_frames_v2、m2_analysis、m3_eval、merge_predictions、probe_serve；workspace/ 下 m2_analysis、m3_eval、probe；NitroGen-main/ 为官方代码。
> 【技术栈】Windows 11 + conda nitrogen + Python 3.12 + ZeroMQ + parquet。
> 【困难点】今日报告主题是"启动说明、目录职责、审查纪要"：启动说明=serve.py（无条件模式无 input()）+ m3_eval.py 两步；目录职责=每个目录/脚本一句话职责；审查纪要=按课程要求通读 m3_eval.py 与 merge_predictions.py，记录审查发现（如：分块评测需 --offset/--limit 且块间 reset() 是否正确的隐患、21→17 映射的维护点）。
> 【期望输出】一份当日报告 markdown：基本信息/今日目标/今日完成/启动说明（命令块）/目录职责表/审查纪要（含审查人、日期、发现与处理）/自检/问题与明日计划。600 字左右。

---

### Step 5（第 5 天 8/23）：T5.1 演示对比表 + 当日报告

**AI 提示词（T5.1 对比表）：**

> 【背景】我在做 NitroGen 通用游戏智能体项目的"第 5 天演示"。已有：`workspace/m3_eval/predictions.parquet`（M3 200 帧模型预测，列：seq_id/frame_idx + pred_*17 个布尔按键 + pred_*4 个摇杆浮点）和 `workspace/test_elden_ring_200frames.parquet`（人工标注，17 布尔按键 + 4 摇杆 + 元数据），两者同 shape 可对齐。目的：生成一张逐帧对比表，让观众直观看到"模型预测 vs 人工标注"的差异。
> 【技术栈】Python 3.12 + pandas，Windows 11，conda nitrogen。
> 【困难点】列结构设计：frame_idx, seq_id, 每个按键拆两列（如 east_pred, east_label 共 34 列），每个摇杆拆两列（如 j_left_x_pred, j_left_x_label 共 8 列），最后一列 frame_diff_score = 按键错误数/17 + 摇杆欧氏距离/2（0~1，越小越像）。
> 【期望输出】一个函数 build_comparison_table(pred_df, label_df) → DataFrame 并保存 `workspace/m3_eval/comparison_table.csv`（utf-8-sig，Excel 可直接打开）。只写这一个脚本，带主函数入口，输出前 10 行预览。

**当日报告主题**：演示步骤与预期现象、差距清单。写法：演示三步（开 serve → 跑 m3_eval → 打开 comparison_table.csv）+ 预期现象（按键列以 0 为主、差异分数普遍 >0.3）+ 差距清单（逐帧对齐达不到 → 演示定位为"分布差异展示"而非"动作模仿"）。

---

### Step 6（第 6 天 8/24）：T6.1 viz_diff 20 张图 + T6.2 帧连续性 + 当日报告

**前置**：`workspace/m3_eval/predictions_m2.parquet` 必须已存在（Step 1 产物）。

**T6.2 帧连续性（先做，5 分钟，直接命令无需 AI）：**

```powershell
E:\tools\Anaconda3\envs\nitrogen\python.exe -c "import pandas as pd; df=pd.read_parquet(r'C:\Users\HP\Desktop\general-game-agent\workspace\m2_elden_ring_500frames.parquet'); g=df.groupby('seq_id').agg(n=('frame_idx','count'),cont=('frame_idx',lambda s: bool((s.diff().dropna()==1).all())),start=('frame_idx','min'),end=('frame_idx','max')); print(g)"
```

**验收**：输出 10 行，看 `cont` 列全 True 还是混有 False。结论写进当日报告——若全连续，viz_diff"动作历史"叙事成立；若不连续，报告措辞留余地（说"活跃度挑帧抽帧，段内为采样帧序列"）。

**AI 提示词（T6.1 viz_diff.py）：**

> 【背景】腾讯 IEG"通用游戏智能体"课题的扩展方向"可视化工具"。我有 500 帧手柄动作标注（`workspace/m2_elden_ring_500frames.parquet`：17 布尔按键 + j_left_x/j_left_y/j_right_x/j_right_y + seq_id/frame_idx）和模型对同 500 帧的预测（`workspace/m3_eval/predictions_m2.parquet`：同 shape，列名 pred_* 前缀）。需要批量导出 20 段对比图：10 条序列 × 每序列对半切 2 段 = 20 张，每张标出差异最大的 5 帧。扩展方向验收标准："批量导出 20 段手柄动作曲线，标出差异最大的 5 帧"。
> 【技术栈】Python 3.12 + matplotlib + numpy + pandas，全部离线导出 PNG（不开浏览器），Windows 11。
> 【困难点】① 按 seq_id 分组、每序列对半切成两段（50 帧/序列 → 25+25），恰好 20 张；② 图内布局：上面 17 个小子图画按键阶梯图（预测红色虚线 / 标注蓝色实线），下面 4 个子图画摇杆折线图；③ 帧差异分数 = 按键错误数/17 + 摇杆欧氏距离/2，每段取 top-5 帧；④ top-5 帧 x 轴画红色竖虚线 + 顶部标注分数；⑤ 输出 `workspace/viz_diff/viz_segment_XX.png`，标题含段号与该段平均差异分。matplotlib 中文字体可能在 Windows 上乱码，标题只用 ASCII。
> 【期望输出】一个脚本 viz_diff.py，可独立运行，跑完打印每张图的保存路径。只写这一个脚本。

**当日报告主题**：两块完成说明与自测（可视化 20 张 + 帧连续性结论）。

---

### Step 7（第 7 天 8/25）：T7.1 run_all.py 端到端 + 当日报告

**AI 提示词（T7.1 run_all.py，注意已更新——input() 不触发了）：**

> 【背景】我的项目有三步独立流程：① 启动 `NitroGen-main/scripts/serve.py`（加载 ng.pt，ZeroMQ 监听 5555；已实测 `game_mapping_cfg: null` 走无条件模式，**启动时不会 input() 等待游戏 ID**，无需交互）；② 运行 `scripts/m3_eval.py`（连 5555 评测 200 帧，约 60s）；③ 运行 `scripts/viz_diff.py`（导出 20 张图）。目前三步手动执行都能跑，需要一个一键脚本。
> 【技术栈】Windows 11 + conda nitrogen（Python 3.12），python 在 `E:\tools\Anaconda3\envs\nitrogen\python.exe`，serve 需从 `NitroGen-main` 目录启动（相对路径 ..\ng.pt）。
> 【困难点】① 用 subprocess 启动 serve.py（Popen），轮询 localhost:5555 端口直到可连（socket 尝试连接，超时重试，最多 60s）；② serve 就绪后依次跑 m3_eval.py 和 viz_diff.py（用同一个 python 可执行）；③ 结束后 terminate serve 进程并等待退出；④ 任一步失败打印清晰错误并返回非零码；⑤ 日志写入 `workspace/run_all.log`。
> 【期望输出】一个 run_all.py，运行后打印三步各自耗时与成功标志。只写这一个脚本。

**当日报告主题**：通检表、问题与处理、遗留。

---

### Step 8（第 8 天 8/26）：T8.1 README 补全 + 自测表 + 当日报告

**先手动核对 README 现有内容**（3.1~3.7 已有，缺 3.8 可视化与对照表更新），再给 AI：

**AI 提示词（T8.1 README + 自测表）：**

> 【背景】腾讯 IEG"通用游戏智能体"课题，第 8 天归档。README.md 已有 3.1~3.7 节（环境、启动、M2、M3 评测）。需要补：① "3.8 扩展可视化工具"节（运行 viz_diff.py 导出 20 张图到 workspace/viz_diff/，命令 + 预期输出 + 注意事项：必须先有 predictions_m2.parquet）；② 把"4. 产出物对照（MVP 六项）"更新为最终版：第 3 条 ✅、第 4 条 🔶 完成·指标未达标（注明原因，引用 docs/指标归档与实验口径.md）、第 5 条 ✅、第 6 条 ✅。另外生成 `docs/self_test_table.md`：对照 MVP 6 条逐条写"完成状态 / 产出物路径 / 自测方法（命令）/ 结果"，其中第 4 条自测方法=重跑 m3_eval.py 看 metrics.csv 的 button_vs_baseline 与 pearson_vs_baseline 两列。
> 【技术栈】Windows 11 + conda nitrogen + Python 3.12 + markdown。
> 【困难点】保持与现有 3.5/3.6 节一致的表格/命令格式；自测表要"他人可跟做"（每条给出可复制命令）。
> 【期望输出】README 的 3.8 节 + 对照表更新段 + 完整 `docs/self_test_table.md`。输出 markdown 代码块让我直接粘贴。

**当日报告主题**：回归、审查自检、他人启动、自测对照表。

---

### Step 9（第 9 天 8/27）：T9.1 结课大报告（3000+ 字）

**AI 提示词（T9.1，融合归档口径）：**

> 【背景】我在写腾讯 IEG"通用游戏智能体"课题结课大报告（3000+ 字，第 9 天）。已完成：① 全链路复现（serve.py + ng.pt，无条件模式）；② M2 500 帧统计与 10 序列可视化（east 键按压率 61.2%、左摇杆 |y| 均值 0.764——Elden Ring 以向前移动为主，见 workspace/m2_analysis/）；③ M3 评测 200 帧 × 3 次独立 run：按键准确率 0.87~0.91 但全零基线 0.92 更高（真实按键率仅 7.97%，类别失衡虚高），precision≈0.10、recall≈0.03（模型几乎不按键），摇杆 MSE≈0.29~0.31（全零基线 0.264、均值基线 0.115），Pearson r≈0.005~0.019 远低于 0.4；④ 扩展方向"可视化工具"：viz_diff 20 段预测 vs 标注对比图；⑤ 归档口径文档 docs/指标归档与实验口径.md（结论：未达标系结构性原因——无条件先验 + 自回归无逐帧对齐 + 无视觉输入，非调参可解；主报 MSE 符合 MVP"均方误差或相关系数"原文）。
> 【技术栈】Windows 11 + conda nitrogen + Python 3.12 + NVIDIA NitroGen（Flow Matching + 500M DiT）+ parquet/matplotlib。
> 【困难点】报告需覆盖 5 块：① 课题背景与目标（必做/不做最终版）；② 完成情况对照表（MVP 6 条逐条：完成/部分完成/未做，第 4 条注明"完成评测·指标未达标·原因已说明"）；③ 方案与实现概要、主路径（数据→serve→评测→可视化，含技术选型理由）；④ 测试与演示说明（三跑可复现 + 对比表 + 20 张图）；⑤ 问题回顾与总结。**"实验结果与分析"章节必须按归档口径写**：指标总览→类别失衡虚高解读→precision/recall 揭示模型几乎不按键→摇杆 MSE 主报 + r 参考→结构性原因（3 条硬伤+源码行号）→评测局限性与"若需 r≈0.4 需注入游戏条件+逐帧对齐，超出当前数据与 MVP 范围"。不要堆数字，每个数字给一句话解读。
> 【期望输出】3000+ 字完整大报告 markdown（含表格），标题层级清晰，直接可提交。

---

## 3. 通用提示词写作模板（教学：以后自己写）

```
【背景】我在做 <课题>，任务 <XX>。已有：<具体文件/产物路径 + 一句内容>。目标：<一句话>。
【技术栈】<操作系统 + 语言 + 关键库 + 可执行路径>。
【困难点】<1~3 个你踩过/预判的坑，越具体越好；AI 不知道你的坑>。
【期望输出】<交付物类型 + 格式 + 验收点；"只写这一个脚本/只给大纲"类约束放在最后>。
```

三条纪律（来自课程"人工智能辅助开发"要求）：
1. **单一问题**：一次只问一件事，不要"帮我搞定整个系统"；
2. **贴真实路径**：给出绝对路径和文件名，AI 才能写可跑代码；
3. **审查合入**：AI 给的代码自己通读一遍，检查硬编码、无关依赖、接口一致性后再提交。

---

## 4. 提交节奏（git）

| 时点 | commit message 建议 |
|------|--------------------|
| 今晚 Step 1+2 完成 | `T4.1② M2 500帧预测合并 + T4.2 三跑（均值±std 回填归档口径）` |
| 第 5 天 | `T5.1 演示对比表 + 第5天报告` |
| 第 6 天 | `T6.1 可视化20张 + T6.2 帧连续性 + 第6天报告` |
| 第 7 天 | `T7.1 run_all 端到端 + 第7天报告` |
| 第 8 天 | `T8.1 README补全 + 自测表 + 第8天报告` |
| 第 9 天 | `T9.1 结课大报告` |

每天结束必须 `git add -A && git commit`（防止第 8 天归档丢历史）。
