# 通用游戏智能体（腾讯 IEG 课题）

基于 NVIDIA NitroGen 的通用游戏智能体复现与评估项目。

- 参考项目：https://nitrogen.minedojo.org/ （NitroGen）
- 参考代码：https://github.com/MineDojo/NitroGen
- 数据集：nvidia/NitroGen（Parquet 手柄标注，不含视频画面）
- 预训练权重：`ng.pt`（CC BY-NC 4.0，约 2GB，不入库）

> 课题约束：不得下载 full-data 全库 / 数据集全库。本项目仅下载 `SHARD_0000` 一个分片。

---

## 1. 环境要求

| 项目 | 要求 |
|------|------|
| 操作系统 | Windows 10/11（推理服务在 Windows 上验证） |
| Python | ≥ 3.10（官方测试 3.12，推荐 conda 环境 `nitrogen`） |
| GPU | NVIDIA 显卡，CUDA 可用（模型推理必须 `cuda`，`ng.pt` 为 500M 参数 DiT） |
| 网络 | 首次加载需访问 HuggingFace 下载视觉编码器配置；重新下载真实视频帧需访问 Twitch/YouTube |
| 可选 | ffmpeg（`winget install Gyan.FFmpeg`）、yt-dlp（`pip install yt-dlp`）——仅重新下载真实帧时需要 |

所有命令默认在**仓库根目录**下执行。

---

## 2. 目录结构

```
general-game-agent/
├── README.md                       # 本文件
├── scripts/                        # 本项目全部脚本（含迁移的官方 serve.py）
│   ├── serve.py                    # 推理服务器（官方源码迁移，ZeroMQ，加载 ng.pt）
│   ├── scan_games.py               # 任务1：扫描全库游戏分布 → game_stats.csv
│   ├── check_sources.py            # 辅助：按来源统计 elden_ring 数据（youtube/twitch）
│   ├── find_elden_ring.py          # 辅助：定位 elden_ring 的 chunk 列表
│   ├── extract_frames.py           # 任务2a：M2 500帧 / 测试集 200帧 提取（原版）
│   ├── extract_frames_v2.py        # 任务2a：重建版（按活跃度挑帧，M2 为最终口径）
│   ├── rebuild_testset.py          # 辅助：从存活 VOD 池重建测试集 parquet
│   ├── m2_analysis.py              # 任务2b：按键/摇杆统计 + 10条序列可视化
│   ├── m3_eval.py                  # 任务3/4：M3 零样本评测（支持分块）
│   ├── m3_eval_v2.py               # 任务3/4：真实视频帧评测（最终口径）
│   ├── merge_predictions.py        # 合并分块预测 → predictions_m2.parquet
│   ├── build_comparison_table.py   # 任务5：预测 vs 标注 逐帧对比表
│   ├── viz_diff.py                 # 扩展：20 段手柄动作对比图
│   ├── download_test_frames.py     # 辅助：下载测试集源视频帧（Twitch 切段抽帧）
│   └── probe_alive_vods.py         # 辅助：VOD 存活性批量探测
├── NitroGen-main/                  # 官方参考代码的精简版（仅保留可安装包）
│   ├── nitrogen/                   # 核心包：模型 / tokenizer / 推理会话 / 推理客户端
│   ├── pyproject.toml              # pip 安装元数据（pip install -e .）
│   └── LICENSE                     # 开源许可
├── general-game-agent/
│   └── docs/                       # 课程成果文档（过程性报告不入库）
│       ├── 指标归档与实验口径.md     # 指标表 + 评测口径说明（MVP6）
│       └── 实验报告.md             # 3000+ 字实验报告（MVP6）
└── workspace/                      # 数据与评测成果
    ├── game_stats.csv              # 86 款游戏分布统计
    ├── elden_ring_chunks.json      # elden_ring 全部 chunk 索引
    ├── m2_elden_ring_500frames.parquet   # M2 数据集（500 帧标注，已入库）
    ├── test_elden_ring_200frames.parquet # M3 测试集（200 帧标注，已入库）
    ├── m2_analysis/                # 统计 CSV + 12 张序列可视化图（MVP2）
    ├── m3_eval/                    # 评测产物：指标表 + 预测 + 对比表（MVP3/4/5）
    └── viz_diff/                   # 扩展：20 段手柄动作对比图（top-5 差异帧）
```

> 不入库（`.gitignore`，可再生成/需重新下载）：`ng.pt`、`dataset/SHARD_0000/`、
> `workspace/test_frames/`（测试集真实视频帧）、`workspace/m2_frames/`（M2 真实视频帧）等。

---

## 3. 复现步骤

### 3.1 安装依赖与下载权重

```bash
# 1) 安装 nitrogen 包（本仓库 NitroGen-main/ 已含精简后的核心包）
conda activate nitrogen
pip install -e NitroGen-main

# 2) 下载预训练权重 ng.pt（CC BY-NC 4.0，约 2GB）→ 放在仓库根目录
hf download nvidia/NitroGen ng.pt
```

### 3.2 启动推理服务（MVP 任务 1）

```bash
python scripts/serve.py ng.pt --port 5555
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
python -c "import numpy as np; from nitrogen.inference_client import ModelClient; \
c = ModelClient(port=5555); \
img = np.random.randint(0,255,(256,256,3),dtype=np.uint8); \
print(c.predict(img)); c.close()"
```

预期返回 `{'j_left': [...], 'j_right': [...], 'buttons': [...]}`。

### 3.4 数据扫描（MVP 任务 1 补充）

```bash
python scripts/scan_games.py          # → workspace/game_stats.csv（86 款游戏分布）
python scripts/find_elden_ring.py     # → workspace/elden_ring_chunks.json
python scripts/check_sources.py       # 辅助：确认 elden_ring 数据的来源分布
```

> 前置：`dataset/SHARD_0000/` 数据集分片（需自行从 HuggingFace 下载，见课题约束仅取该分片）。

### 3.5 M2 数据提取与统计（MVP 任务 2）

标注数据 `workspace/m2_elden_ring_500frames.parquet` 与
`workspace/test_elden_ring_200frames.parquet` **已入库，可直接使用**。
如需从数据集分片重新生成：

```bash
python scripts/extract_frames_v2.py   # → M2 500帧 + 测试集 200帧 parquet
```

> 注意：M2 原始 10 个 Twitch VOD 中 7 个已被平台删除，重新生成需联网且结果可能与入库版本不同（入库版本为最终口径）。

统计分析（直接运行，读入库 parquet）：

```bash
python scripts/m2_analysis.py         # → workspace/m2_analysis/（统计表 + 12 张图）
```

### 3.6 M3 零样本评测（MVP 任务 3/4）

**A. 测试集标注帧评测（直接可跑，读入库 parquet）：**

```bash
# 先启动推理服务（3.2），再运行：
python scripts/m3_eval.py --input workspace/test_elden_ring_200frames.parquet --save --out-dir workspace/m3_eval
```

**B. 真实视频帧评测（最终口径，需先准备 `workspace/test_frames/`）：**

```bash
# 下载测试集真实视频帧（Twitch 切段 + 60fps 抽帧；部分原始 VOD 已删除，失败序列需用重建测试集）
python scripts/download_test_frames.py
# 若重建测试集后再下载：
python scripts/rebuild_testset.py --vids <存活VOD列表>
python scripts/download_test_frames.py --input test_elden_ring_200frames.parquet

python scripts/m3_eval_v2.py --save   # → workspace/m3_eval/metrics_v2.csv + predictions_v2.parquet
```

**C. M2 500 帧全量评测（分块防显存溢出）：**

```bash
python scripts/m3_eval.py --input workspace/m2_elden_ring_500frames.parquet --offset 0 --limit 200 --save --out-dir workspace/m3_eval/m2_blk1
python scripts/m3_eval.py --input workspace/m2_elden_ring_500frames.parquet --offset 200 --limit 200 --save --out-dir workspace/m3_eval/m2_blk2
python scripts/m3_eval.py --input workspace/m2_elden_ring_500frames.parquet --offset 400 --limit 100 --save --out-dir workspace/m3_eval/m2_blk3
python scripts/merge_predictions.py --parts workspace/m3_eval/m2_blk1 workspace/m3_eval/m2_blk2 workspace/m3_eval/m2_blk3 --label workspace/m2_elden_ring_500frames.parquet --out-dir workspace/m3_eval
```

最终结果（真实视频帧口径）：按键准确率 **0.9112**（≥50% 且高于全零基线 0.9079）、
左摇杆相关系数 **0.5768**（≥0.4，x/y 0.493/0.660）、4 轴平均 r 0.2895（右摇杆标注活跃率仅 8.5%
导致的稀释）、摇杆 MSE 0.1483（优于零基线 0.2072）。早期噪声图三跑
（按键 0.9052±0.0007、r 0.0081±0.0096）为历史对照，详见
`general-game-agent/docs/指标归档与实验口径.md`。

### 3.7 演示对比表 + 扩展可视化（MVP 任务 5 + 扩展）

```bash
# 预测 vs 人工标注 逐帧对比表（MVP 5 演示）→ workspace/m3_eval/comparison_table.csv
python scripts/build_comparison_table.py

# 扩展可视化：20 段手柄动作对比图（标出差异最大 5 帧）→ workspace/viz_diff/viz_segment_01~20.png
# 数据源：M2 500 帧真实帧推理 predictions_m2.parquet（按键 acc 0.9493，recall 0.501）
python scripts/viz_diff.py
```

> 前置：`viz_diff.py` 需要 `workspace/m3_eval/predictions_m2.parquet`（3.6-C 的合并产物）；
> `build_comparison_table.py` 需要 `workspace/m3_eval/predictions_v2.parquet`（3.6-B 的产物）。

### 3.8 可选：连接真实游戏游玩

官方 `play.py`（游玩客户端）已从本仓库精简掉，如需连接 Windows 游戏游玩，
请从 [MineDojo/NitroGen](https://github.com/MineDojo/NitroGen) 获取 `scripts/play.py`
并安装额外依赖（`vgamepad` 等）。

---

## 4. 产出物对照（MVP 六项）

| # | 必做项 | 产出物 | 状态 |
|---|--------|--------|------|
| 1 | 跑通 ng.pt 推理，README 可复现 | 本 README §3.1–3.3 + 推理服务运行 | ✅ |
| 2 | 500 帧统计 + 10 条序列可视化 | `workspace/m2_elden_ring_500frames.parquet` + `workspace/m2_analysis/` | ✅ |
| 3 | 200 帧测试集指标 | `workspace/m3_eval/metrics_v2.csv` + `predictions_v2.parquet` | ✅ |
| 4 | zero-shot 基线对比（50% / 0.4） | `metrics_v2.csv` 含 `button_accuracy` / `pearson_sticks_avg` / 左摇杆分轴 r。**按键 0.9112 ≥ 50% 达标；左摇杆 r 0.5768 ≥ 0.4 达标**（4 轴平均 0.2895，右摇杆近静止导致稀释，口径说明见 `general-game-agent/docs/指标归档与实验口径.md`） | ✅ 达标（左摇杆口径） |
| 5 | 第 5 天演示（表格或录屏） | `workspace/m3_eval/comparison_table.csv`（200 帧逐帧对比 + 差异分） | ✅ |
| 6 | 归档代码 + 指标表 + 3000 字报告 | 本仓库（代码/指标表）+ `general-game-agent/docs/实验报告.md`（3000+ 字） | ✅ |

---

## 5. 已知问题与注意事项

1. **推理必须 GPU**：`inference_session.py` 中显式 `model.to("cuda")`，无 GPU 会直接报错。
2. **首次加载联网**：`AutoImageProcessor.from_pretrained(model_cfg.vision_encoder_name)` 需从 HuggingFace 下载视觉编码器配置。
3. **serve.py 是交互式的**：若权重含游戏映射表，会阻塞等待输入；直接回车即可跳过。
4. **数据集不含视频**：`actions_raw.parquet` 只有手柄动作标注（帧序号 + 按键 + 摇杆），无画面帧；真实视频帧需另行下载且部分原始 VOD 已被平台删除。
5. **扩展方向（已选）**：可视化工具——批量导出 20 段手柄动作曲线，标出差异最大的 5 帧（`scripts/viz_diff.py`，已完成，输出 `workspace/viz_diff/`）。

---

## 6. 参考

- NitroGen 官网：https://nitrogen.minedojo.org/
- 模型权重：https://huggingface.co/nvidia/NitroGen
- 数据集：https://huggingface.co/datasets/nvidia/NitroGen
- 论文：NitroGen: An Open Foundation Model for Generalist Gaming Agents（arXiv:2601.02427）
