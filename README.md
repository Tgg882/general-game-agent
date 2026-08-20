# 通用游戏智能体（腾讯 IEG 课题）

基于 NVIDIA NitroGen 的通用游戏智能体复现与评估项目。

- 参考项目：https://nitrogen.minedojo.org/ （NitroGen）
- 参考代码：https://github.com/MineDojo/NitroGen
- 数据集：nvidia/NitroGen（Parquet 手柄标注，不含视频画面）
- 预训练权重：`ng.pt`（CC BY-NC 4.0）

> 课题约束：不得下载 full-data 全库 / 数据集全库。本项目仅下载 `SHARD_0000` 一个分片。

---

## 1. 环境要求

| 项目 | 要求 |
|------|------|
| 操作系统 | Windows 11（推理服务 + 游戏游玩均支持） |
| Python | ≥ 3.10（官方测试 3.12，本项目使用 conda 环境 `nitrogen`） |
| GPU | NVIDIA 显卡，CUDA 可用（模型推理必须 `cuda`，`ng.pt` 为 500M 参数 DiT） |
| 网络 | 首次加载需访问 HuggingFace 下载视觉编码器配置 |

本机 conda 环境：`E:\tools\Anaconda3\envs\nitrogen`

---

## 2. 目录结构

```
general-game-agent/
├── ng.pt                          # 预训练权重（500M Flow Matching Transformer）
├── NitroGen-main/                 # 官方参考代码（pip 可安装包 nitrogen）
│   ├── nitrogen/                  # 核心包：模型 / tokenizer / 推理会话 / 游戏环境
│   └── scripts/
│       ├── serve.py               # 推理服务器（ZeroMQ，加载 ng.pt 提供预测）
│       └── play.py                # 游玩客户端（连服务器，操控 Windows 游戏）
├── dataset/
│   └── SHARD_0000/                # 数据集分片（视频ID/chunk/actions_raw.parquet + metadata.json）
├── scripts/                       # 本项目自写脚本
│   ├── scan_games.py              # 任务1：扫描全库游戏分布 → game_stats.csv
│   ├── find_elden_ring.py         # 辅助：定位 elden_ring 的 chunk 列表
│   ├── extract_frames.py          # 任务2a：M2 500帧 / 测试集 200帧 提取
│   ├── extract_frames_v2.py       # 任务2a：改进版（按活跃度挑帧，避免挂机帧）
│   ├── m2_analysis.py             # 任务2b：按键/摇杆统计 + 10条序列可视化
│   ├── m3_eval.py                 # 任务3/4：M3 零样本评测（支持分块）
│   ├── merge_predictions.py       # 合并分块预测 → predictions_m2.parquet
│   ├── build_comparison_table.py  # 任务5：预测 vs 标注 逐帧对比表
│   └── viz_diff.py                # 扩展：20 段手柄动作对比图
└── workspace/
    ├── game_stats.csv             # 86 款游戏分布统计
    ├── elden_ring_chunks.json     # elden_ring 全部 chunk 索引
    ├── m2_elden_ring_500frames.parquet  # M2 数据集（500帧标注）
    ├── test_elden_ring_200frames.parquet # M3 测试集（200帧标注）
    ├── m2_analysis/               # 统计 CSV + 12 张序列可视化图
    ├── m3_eval/                   # 评测产物（三跑指标 + 预测 + 对比表 + M2 合并）
    └── viz_diff/                  # 扩展：20 段对比图
```

---

## 3. 复现步骤

### 3.1 准备代码与权重

```bash
# 1) 获取参考代码（本仓库已含 NitroGen-main/，可跳过 clone）
git clone https://github.com/MineDojo/NitroGen.git

# 2) 安装依赖（conda 环境 nitrogen）
conda activate nitrogen
cd NitroGen-main
pip install -e .

# 3) 下载权重 ng.pt（CC BY-NC 4.0，约 2GB）
hf download nvidia/NitroGen ng.pt
#   本仓库已将 ng.pt 放在根目录，可跳过此步
```

### 3.2 跑通推理服务（MVP 任务 1）

```bash
cd NitroGen-main
python scripts/serve.py ..\ng.pt --port 5555
```

启动后预期输出：

```
Checkpoint args: {...}
Available games in tokenizer mapping:   # 若出现此提示
Enter the game ID to use (leave empty for unconditional):
```

> 若出现 `Enter the game ID to use` 提示：直接回车（空输入 = 无条件生成），或输入列表中的游戏 ID。
> 首次运行会联网下载视觉编码器配置（HuggingFace），需保持网络畅通。
> 模型加载到 GPU 后显示：

```
Server running on port 5555
Waiting for requests...
```

### 3.3 验证推理服务（可选）

另开一个终端，发送一张图片验证服务器返回手柄动作：

```bash
conda activate nitrogen
cd NitroGen-main
python -c "import numpy as np; from nitrogen.inference_client import ModelClient; \
c = ModelClient(port=5555); \
img = np.random.randint(0,255,(256,256,3),dtype=np.uint8); \
print(c.predict(img)); c.close()"
```

预期返回 `{'j_left': [...], 'j_right': [...], 'buttons': [...]}`。

### 3.4 连接真实游戏游玩（可选，需本机安装对应游戏）

```bash
cd NitroGen-main
python scripts/play.py --process 'elden_ring.exe'
```

> `--process` 必须是任务管理器中的精确进程名（含 `.exe`）。
> 需要 `vgamepad` 虚拟手柄驱动；模型预测的动作会以 60 FPS 下发并录制 DEBUG/CLEAN 视频。
> 若报缺少 dxcam / pywinctl / vgamepad 等 Windows 依赖，执行 `pip install -e ".[play]"`。

### 3.5 数据扫描（MVP 任务 1 补充）

```bash
conda activate nitrogen
python scripts/scan_games.py          # → workspace/game_stats.csv（86 款游戏分布）
python scripts/find_elden_ring.py     # → workspace/elden_ring_chunks.json
```

### 3.6 M2 数据提取与统计（MVP 任务 2）

```bash
python scripts/extract_frames_v2.py   # → M2 500帧 + 测试集 200帧 parquet
python scripts/m2_analysis.py         # → workspace/m2_analysis/（统计表 + 12 张图）
```

### 3.7 M3 零样本评测（MVP 任务 3/4，已完成）

```bash
# 先启动推理服务（3.2），再运行：
python scripts/m3_eval.py             # → workspace/m3_eval/（指标表）
```

评测脚本将对 `test_elden_ring_200frames.parquet` 中的 200 帧逐帧调用推理服务，
计算：按键准确率、摇杆 MSE / 相关系数，并与基线（按键 50% / 摇杆相关 0.4）对比。

独立采样验证：重启服务跑 3 次，按键准确率 `0.9052 ± 0.0007`、Pearson r `0.0081 ± 0.0096`，结论稳定可复现（三跑汇总见 `docs/指标归档与实验口径.md`）。

M2 500 帧全量评测（分块防显存溢出）：

```bash
python scripts/m3_eval.py --input workspace/m2_elden_ring_500frames.parquet --offset 0 --limit 200 --save --out-dir workspace/m3_eval/m2_blk1
python scripts/m3_eval.py --input workspace/m2_elden_ring_500frames.parquet --offset 200 --limit 200 --save --out-dir workspace/m3_eval/m2_blk2
python scripts/m3_eval.py --input workspace/m2_elden_ring_500frames.parquet --offset 400 --limit 100 --save --out-dir workspace/m3_eval/m2_blk3
python scripts/merge_predictions.py --parts workspace/m3_eval/m2_blk1 workspace/m3_eval/m2_blk2 workspace/m3_eval/m2_blk3 --label workspace/m2_elden_ring_500frames.parquet --out-dir workspace/m3_eval
```

### 3.8 演示对比表 + 扩展可视化（MVP 任务 5 + 扩展）

```bash
# 预测 vs 人工标注 逐帧对比表（MVP 5 演示）→ workspace/m3_eval/comparison_table.csv
python scripts/build_comparison_table.py

# 扩展可视化：20 段手柄动作对比图（标出差异最大 5 帧）→ workspace/viz_diff/viz_segment_01~20.png
python scripts/viz_diff.py
```

> 前置：`viz_diff.py` 需要 `predictions_m2.parquet`（3.7 合并产物）存在；`build_comparison_table.py` 需要 M3 `predictions.parquet`（默认输出即 3.7 产物）。

---

## 4. 产出物对照（MVP 六项）

| # | 必做项 | 产出物 | 状态 |
|---|--------|--------|------|
| 1 | 跑通 ng.pt 推理，README 可复现 | 本 README + 推理服务运行 | ✅ |
| 2 | 500 帧统计 + 10 条序列可视化 | `workspace/m2_analysis/` | ✅ |
| 3 | 200 帧测试集指标 | `workspace/m3_eval/metrics.csv` + `predictions.parquet`（三跑） | ✅ |
| 4 | zero-shot 基线对比（50% / 0.4） | `metrics.csv` 含 `button_baseline_50` / `pearson_baseline_04` 对比列。按键准确率虚高（模型 0.90 < 全零基线 0.92，已改主报 precision/recall）；摇杆 r≈0.008±0.010 未达 0.4，系评测方式结构性所致（无条件先验 + 无逐帧对齐），归档口径见 `docs/指标归档与实验口径.md` | 🔶 完成·未达标（原因已说明） |
| 5 | 第 5 天演示（表格或录屏） | `workspace/m3_eval/comparison_table.csv`（200 帧逐帧对比 + 差异分） | ✅ |
| 6 | 归档代码 + 指标表 + 3000 字报告 | 本仓库（代码/指标/自测表已归档）+ 结课报告 | 🔶 报告待写 |

---

## 5. 已知问题与注意事项

1. **推理必须 GPU**：`inference_session.py` 中显式 `model.to("cuda")`，无 GPU 会直接报错。
2. **首次加载联网**：`AutoImageProcessor.from_pretrained(model_cfg.vision_encoder_name)` 需从 HuggingFace 下载视觉编码器配置。
3. **serve.py 是交互式的**：若权重含游戏映射表，会阻塞等待输入；直接回车即可跳过。
4. **数据集不含视频**：`actions_raw.parquet` 只有手柄动作标注（帧序号 + 按键 + 摇杆），无画面帧。
5. **扩展方向（已选）**：可视化工具——批量导出 20 段手柄动作曲线，标出差异最大的 5 帧（`viz_diff.py`，已完成，输出 `workspace/viz_diff/`）。

---

## 6. 参考

- NitroGen 官网：https://nitrogen.minedojo.org/
- 模型权重：https://huggingface.co/nvidia/NitroGen
- 数据集：https://huggingface.co/datasets/nvidia/NitroGen
- 论文：NitroGen: An Open Foundation Model for Generalist Gaming Agents（arXiv:2601.02427）
