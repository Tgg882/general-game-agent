# 后续任务路线图 + AI 开发提示词手册（v3 · 2026-08-20 深夜）

> 适用：第 2 天晚 → 第 9 天（8/27）结课。**基于项目实际**：已完成的执行任务不再列入计划（命令保留在附录作复现），本文只把**未完成任务**按 Step 逐条计划；报告撰写由本人完成，不涉及。
> 每个 Step 给：**目标（对应选题要求）→ 精确命令 / AI 提示词 → 验收标准**。
> 提示词使用铁律：每次只问一件事；把对应段【背景】原样贴给 AI；AI 给代码后自己通读一遍再合入（审查职责）。

---

## 0. 项目实际状态（2026-08-20 深夜核对）

### 已完成 ✅

| 任务 | 证据 |
|------|------|
| M2 500 帧标注（10 序列）+ 分布统计 + 10 序列图 | `workspace/m2_elden_ring_500frames.parquet` + `m2_analysis/` |
| 测试集 200 帧（4 序列） | `workspace/test_elden_ring_200frames.parquet` |
| serve.py 冒烟 + 50 帧探针 | `scripts/probe_serve.py` + `probe/probe_stats.csv` |
| M3 评测脚本（--offset/--limit/--save/--out-dir） | `scripts/m3_eval.py` |
| M3 评测 200 帧 × 3 run（均值±std） | `m3_eval/metrics.csv` + `run1/` + `run3/` |
| M2 500 帧预测 + 合并全量指标 | `m3_eval/predictions_m2.parquet` + `metrics_m2.csv` |
| 指标归档口径（未达标原因论证） | `docs/指标归档与实验口径.md` |
| README 主体（3.1~3.7 + MVP 对照表） | `README.md` |

### 未完成 ⏳（本文的 Step 计划对象）

| # | 任务 | 对应选题要求 |
|---|------|--------------|
| 1 | T5.1 演示对比表 | MVP 5：第 5 天演示模型输出与标注对比 |
| 2 | T6.2 帧连续性检查 | 扩展·可视化工具前置 |
| 3 | T6.1 viz_diff 20 段对比图 | 扩展·可视化工具（立项） |
| 4 | T8.1 README 补全 + 自测表 | MVP 1 / MVP 6 归档 |

> MVP 完成度全景：1️⃣ 主体完成（T8.1 收尾）· 2️⃣ ✅ · 3️⃣ ✅ · 4️⃣ 🔶 完成评测·指标未达标（口径已归档）· 5️⃣ ⏳ Step 1 · 6️⃣ 🔶（Step 4 + 本人报告）。

---

## 1. 未完成任务 · Step 1：T5.1 演示对比表（MVP 5）

**目标**：生成一张"模型预测 vs 人工标注"逐帧对比表，作为第 5 天演示产物（表格形式）。

**AI 提示词：**

> 【背景】我在做 NitroGen 通用游戏智能体项目的"第 5 天演示"。已有：`workspace/m3_eval/predictions.parquet`（M3 200 帧模型预测，列：seq_id/frame_idx + pred_*17 个布尔按键 + pred_*4 个摇杆浮点）和 `workspace/test_elden_ring_200frames.parquet`（人工标注，17 布尔按键 + 4 摇杆 + 元数据），两者同 shape 可对齐。目的：生成一张逐帧对比表，让观众直观看到"模型预测 vs 人工标注"的差异。
> 【技术栈】Python 3.12 + pandas，Windows 11，conda nitrogen。
> 【困难点】列结构设计：frame_idx, seq_id, 每个按键拆两列（如 east_pred, east_label 共 34 列），每个摇杆拆两列（如 j_left_x_pred, j_left_x_label 共 8 列），最后一列 frame_diff_score = 按键错误数/17 + 摇杆欧氏距离/2（0~1，越小越像）。
> 【期望输出】一个函数 build_comparison_table(pred_df, label_df) → DataFrame 并保存 `workspace/m3_eval/comparison_table.csv`（utf-8-sig，Excel 可直接打开）。只写这一个脚本，带主函数入口，输出前 10 行预览。

**验收**：`comparison_table.csv` 存在；每帧一行，含 pred/label 分列 + `frame_diff_score`；Excel 可打开。

**提交**：`git add -A && git commit -m "T5.1 演示对比表"`

---

## 2. 未完成任务 · Step 2：T6.2 帧连续性检查（扩展前置）

**目标**：确认 M2 500 帧各序列 frame_idx 是否连续，为 T6.1 可视化的"动作历史"叙事定调。

**直接命令（无需 AI）：**

```powershell
E:\tools\Anaconda3\envs\nitrogen\python.exe -c "import pandas as pd; df=pd.read_parquet(r'C:\Users\HP\Desktop\general-game-agent\workspace\m2_elden_ring_500frames.parquet'); g=df.groupby('seq_id').agg(n=('frame_idx','count'),cont=('frame_idx',lambda s: bool((s.diff().dropna()==1).all())),start=('frame_idx','min'),end=('frame_idx','max')); print(g)"
```

**验收**：输出 10 行，记录 `cont` 列是否全 True。全连续 → Step 3 措辞用"动作历史"；不连续 → 措辞用"活跃度挑帧抽帧，段内为采样帧序列"。

---

## 3. 未完成任务 · Step 3：T6.1 viz_diff 20 段对比图（扩展·可视化工具）

**前置**：Step 2 结论 + `predictions_m2.parquet`（已完成）。

**AI 提示词：**

> 【背景】腾讯 IEG"通用游戏智能体"课题的扩展方向"可视化工具"。我有 500 帧手柄动作标注（`workspace/m2_elden_ring_500frames.parquet`：17 布尔按键 + j_left_x/j_left_y/j_right_x/j_right_y + seq_id/frame_idx）和模型对同 500 帧的预测（`workspace/m3_eval/predictions_m2.parquet`：同 shape，列名 pred_* 前缀）。需要批量导出 20 段对比图：10 条序列 × 每序列对半切 2 段 = 20 张，每张标出差异最大的 5 帧。扩展方向验收标准："批量导出 20 段手柄动作曲线，标出差异最大的 5 帧"。
> 【技术栈】Python 3.12 + matplotlib + numpy + pandas，全部离线导出 PNG（不开浏览器），Windows 11。
> 【困难点】① 按 seq_id 分组、每序列对半切成两段（50 帧/序列 → 25+25），恰好 20 张；② 图内布局：上面 17 个小子图画按键阶梯图（预测红色虚线 / 标注蓝色实线），下面 4 个子图画摇杆折线图；③ 帧差异分数 = 按键错误数/17 + 摇杆欧氏距离/2，每段取 top-5 帧；④ top-5 帧 x 轴画红色竖虚线 + 顶部标注分数；⑤ 输出 `workspace/viz_diff/viz_segment_XX.png`，标题含段号与该段平均差异分。matplotlib 中文字体可能在 Windows 上乱码，标题只用 ASCII。
> 【期望输出】一个脚本 viz_diff.py，可独立运行，跑完打印每张图的保存路径。只写这一个脚本。

**验收**：`workspace/viz_diff/` 下恰好 20 张 PNG；每张含按键（17 阶梯）+ 摇杆（4 折线）与 top-5 差异帧竖线标注。

**提交**：`git add -A && git commit -m "T6.1 可视化20张 + T6.2 帧连续性"`

---

## 4. 未完成任务 · Step 4：T8.1 README 补全 + 自测表（MVP 1/6）

**前置**：Step 1（对比表）、Step 3（20 张图）完成后做，README 对照表与自测表引用其结果。

**先手动核对 README 现有内容**（3.1~3.7 已有，缺 3.8 可视化与对照表更新），再给 AI：

**AI 提示词：**

> 【背景】腾讯 IEG"通用游戏智能体"课题归档。README.md 已有 3.1~3.7 节（环境、启动、M2、M3 评测）。需要补：① "3.8 扩展可视化工具"节（运行 viz_diff.py 导出 20 张图到 workspace/viz_diff/，命令 + 预期输出 + 注意事项：必须先有 predictions_m2.parquet）；② 把"4. 产出物对照（MVP 六项）"更新为最终版：第 3 条 ✅、第 4 条 🔶 完成·指标未达标（注明原因，引用 docs/指标归档与实验口径.md）、第 5 条 ✅、第 6 条 ✅。另外生成 `docs/self_test_table.md`：对照 MVP 6 条逐条写"完成状态 / 产出物路径 / 自测方法（命令）/ 结果"，其中第 4 条自测方法=重跑 m3_eval.py 看 metrics.csv 的 button_vs_baseline 与 pearson_vs_baseline 两列。
> 【技术栈】Windows 11 + conda nitrogen + Python 3.12 + markdown。
> 【困难点】保持与现有 3.5/3.6 节一致的表格/命令格式；自测表要"他人可跟做"（每条给出可复制命令）。
> 【期望输出】README 的 3.8 节 + 对照表更新段 + 完整 `docs/self_test_table.md`。输出 markdown 代码块让我直接粘贴。

**验收**：README 3.8 节可照做复现 20 张图；`self_test_table.md` 对 MVP 6 条逐条可跟做。

**提交**：`git add -A && git commit -m "T8.1 README补全 + 自测表"`

---

## 附录 A：已完成任务的可复现命令（备用）

### M3 评测（连 serve，200 帧测试集）

```powershell
# ① 启动 serve（无条件模式，无 input()）
cd C:\Users\HP\Desktop\general-game-agent\NitroGen-main
$env:HF_ENDPOINT="https://hf-mirror.com"
E:\tools\Anaconda3\envs\nitrogen\python.exe scripts/serve.py ..\ng.pt --port 5555

# ② 跑评测
cd C:\Users\HP\Desktop\general-game-agent
E:\tools\Anaconda3\envs\nitrogen\python.exe scripts/m3_eval.py --save --out-dir workspace/m3_eval/<run_name>
```

### M2 500 帧分块评测 + 合并

```powershell
E:\tools\Anaconda3\envs\nitrogen\python.exe scripts/m3_eval.py --input workspace/m2_elden_ring_500frames.parquet --offset <N> --limit <M> --save --out-dir workspace/m3_eval/m2_blk<N>
E:\tools\Anaconda3\envs\nitrogen\python.exe scripts/merge_predictions.py --parts workspace/m3_eval/m2_blk1 workspace/m3_eval/m2_blk2 workspace/m3_eval/m2_blk3 --label workspace/m2_elden_ring_500frames.parquet --out-dir workspace/m3_eval
```

---

## 附录 B：通用提示词写作模板

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
