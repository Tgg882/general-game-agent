# MVP 六项 · 自测表（他人可跟做）

> 项目根目录：`C:\Users\HP\Desktop\general-game-agent`
> Python 环境：`E:\tools\Anaconda3\envs\nitrogen\python.exe`（conda `nitrogen`）
> 核对日期：2026-08-20 深夜

| # | 必做项 | 完成状态 | 产出物路径 | 自测方法（命令） | 预期结果 |
|---|--------|----------|------------|------------------|----------|
| 1 | 跑通 ng.pt 推理，README 可复现 | ✅ 主体完成（README 3.1~3.8） | `README.md` + `NitroGen-main/scripts/serve.py` | ① `cd NitroGen-main` ② `python scripts/serve.py ..\ng.pt --port 5555` ③ 另开终端执行：`python -c "import numpy as np; from nitrogen.inference_client import ModelClient; c=ModelClient(port=5555); img=np.random.randint(0,255,(256,256,3),dtype=np.uint8); print(c.predict(img)); c.close()"` | ② 输出 `Server running on port 5555`；③ 返回 `{'j_left': [...], 'j_right': [...], 'buttons': [...]}` |
| 2 | 500 帧标注 + 分布统计 + 10 序列可视化 | ✅ 完成 | `workspace/m2_elden_ring_500frames.parquet` + `workspace/m2_analysis/` | `python -c "import pandas as pd; df=pd.read_parquet('workspace/m2_elden_ring_500frames.parquet'); print(len(df), df['seq_id'].nunique()); print(df.groupby('seq_id').size())"` | 输出 `500 10`，且每序列 50 帧；`m2_analysis/` 含 `sequences/` 10 张图 + 分布图 + 统计 CSV |
| 3 | 200 帧测试集指标（按键准确率 + 摇杆 MSE/相关系数） | ✅ 完成 | `workspace/m3_eval/metrics.csv`、`run1/metrics.csv`、`run3/metrics.csv` | ① 按第 1 条启动 serve ② `python scripts/m3_eval.py --save --out-dir workspace/m3_eval/selfcheck` | `selfcheck/metrics.csv` 出现且 `n_frames=200`；按键准确率落在 0.87~0.91、Pearson r 落在 -0.05~+0.05（与三跑存档一致即通过） |
| 4 | zero-shot 基线对比（按键 50% / 摇杆 r 0.4） | 🔶 完成·指标未达标 | `workspace/m3_eval/metrics.csv`（`button_baseline_50` / `pearson_baseline_04` 两列） | 重跑第 3 条命令后查看指标：`python -c "import pandas as pd; m=pd.read_csv('workspace/m3_eval/selfcheck/metrics.csv'); print('button_vs_baseline:', m['button_vs_baseline'][0]); print('pearson_vs_baseline:', m['pearson_vs_baseline'][0])"` | 按键 `button_vs_baseline` > 0（数值达标）；`pearson_vs_baseline` < 0（未达 0.4，结构性原因见 `docs/指标归档与实验口径.md`，此处应复现"未达标"这一结论本身） |
| 5 | 第 5 天演示（模型输出与标注对比表格） | ✅ 完成 | `workspace/m3_eval/comparison_table.csv`（200 行 × 45 列） | `python scripts/build_comparison_table.py --out workspace/m3_eval/selfcheck_cmp.csv` | 生成 200 行对比表，每帧含 34 个按键 pred/label 分列 + 8 个摇杆 pred/label 分列 + `frame_diff_score`（0~1 越小越像） |
| 6 | 归档代码 + 指标表 + 3000 字报告 | 🔶 代码/指标已归档，报告本人撰写中 | `scripts/`、`docs/指标归档与实验口径.md`、`README.md` | `git log --oneline` | 可见 T1~T8 分步提交历史；指标口径、README 复现文档、自测表齐全 |

## 说明

- **第 4 条"自测"的特殊性**：该条目本身是"对比基线后未达标"，自测目的是**复现"未达标"这一结论**（r≈0、按键虚高），并核对报告中引用的原因说明与实测一致；若复现出"达标"，反而说明测试不可复现，需排查。
- 三跑存档（Run1/Run2/Run3）为验收基准，`selfcheck` 为他人复现用临时目录，可自删。
- 首次运行需联网下载视觉编码器配置（HuggingFace），且必须 GPU（`inference_session.py` 显式 `model.to("cuda")`）。
