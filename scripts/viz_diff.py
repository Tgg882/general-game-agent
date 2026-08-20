"""viz_diff.py - T6.1 扩展可视化工具: 批量导出 20 段手柄动作对比图.

输入:
  - 标注: workspace/m2_elden_ring_500frames.parquet (17 布尔按键 + 4 摇杆 + seq_id/frame_idx)
  - 预测: workspace/m3_eval/predictions_m2.parquet (同 shape, pred_* 前缀)

输出: workspace/viz_diff/viz_segment_XX.png, 共 10 序列 x 2 段 = 20 张.
每张: 上 17 子图按键阶梯(预测红虚线/标注蓝实线), 下 4 子图摇杆折线,
      红色竖虚线标注差异最大的 top-5 帧, 标题含段号与平均差异分.

用法:
  python scripts/viz_diff.py [--out-dir workspace/viz_diff]
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 离线导出, 不开 GUI
import matplotlib.pyplot as plt

BTN_COLS = [
    "back", "dpad_down", "dpad_left", "dpad_right", "dpad_up",
    "east", "guide", "left_shoulder", "left_thumb", "left_trigger",
    "north", "right_shoulder", "right_thumb", "right_trigger",
    "south", "start", "west",
]
STICK_COLS = ["j_left_x", "j_left_y", "j_right_x", "j_right_y"]
N_TOP = 5          # 每段标出差异最大的帧数
SEG_SPLIT = 2      # 每序列对半切成 2 段


def parse_args():
    p = argparse.ArgumentParser(description="T6.1 viz_diff: 20 段手柄动作对比图")
    p.add_argument("--label", default="workspace/m2_elden_ring_500frames.parquet")
    p.add_argument("--pred", default="workspace/m3_eval/predictions_m2.parquet")
    p.add_argument("--out-dir", default="workspace/viz_diff")
    return p.parse_args()


def _resolve(path: str) -> str:
    if os.path.exists(path):
        return path
    p = os.path.join(os.getcwd(), path)
    if os.path.exists(p):
        return p
    raise FileNotFoundError(f"找不到文件: {path}")


def diff_score(pred_btn: np.ndarray, label_btn: np.ndarray,
               pred_stick: np.ndarray, label_stick: np.ndarray) -> float:
    """帧差异分数 = 按键错误数/17 + 摇杆欧氏距离/2 (越小越像)."""
    btn_err = np.mean(pred_btn != label_btn)
    stick_dist = np.sqrt(((pred_stick - label_stick) ** 2).sum()) / 2.0
    return float(btn_err + stick_dist)


def plot_segment(ax_main_title: str, frame_idx: np.ndarray,
                 pred_btn: np.ndarray, label_btn: np.ndarray,
                 pred_stick: np.ndarray, label_stick: np.ndarray,
                 top_idxs: np.ndarray, top_scores: np.ndarray,
                 out_path: str):
    """绘制单段对比图."""
    fig, axes = plt.subplots(
        len(BTN_COLS) + len(STICK_COLS), 1,
        figsize=(12, 3.2 * (len(BTN_COLS) + len(STICK_COLS))),
        sharex=True,
    )
    # 顶部主标题
    fig.suptitle(ax_main_title, fontsize=13, y=0.998)

    # 按键阶梯图 (17)
    for i, c in enumerate(BTN_COLS):
        ax = axes[i]
        ax.step(frame_idx, label_btn[i], where="mid", color="blue",
                linewidth=1.0, label="label")
        ax.step(frame_idx, pred_btn[i], where="mid", color="red",
                linewidth=0.8, linestyle="--", label="pred")
        ax.set_ylim(-0.1, 1.1)
        ax.set_yticks([0, 1])
        ax.set_ylabel(c, fontsize=7)
        ax.tick_params(labelsize=6)

    # 摇杆折线图 (4)
    for i, c in enumerate(STICK_COLS):
        ax = axes[len(BTN_COLS) + i]
        ax.plot(frame_idx, label_stick[i], color="blue", linewidth=1.0, label="label")
        ax.plot(frame_idx, pred_stick[i], color="red", linewidth=0.8,
                linestyle="--", label="pred")
        ax.set_ylim(-1.2, 1.2)
        ax.set_ylabel(c, fontsize=7)
        ax.tick_params(labelsize=6)

    # top-5 差异帧竖线 + 顶部分数标注
    for j, (idx, s) in enumerate(zip(top_idxs, top_scores)):
        for ax in axes:
            ax.axvline(idx, color="red", linewidth=0.6, alpha=0.35,
                       linestyle=":")
        axes[0].text(idx, 1.35, f"#{idx} {s:.2f}", color="darkred",
                     fontsize=7, ha="center", rotation=45)

    axes[-1].set_xlabel("frame_idx", fontsize=8)
    axes[0].legend(loc="upper left", fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main():
    args = parse_args()
    label_df = pd.read_parquet(_resolve(args.label))
    pred_df = pd.read_parquet(_resolve(args.pred))
    os.makedirs(args.out_dir, exist_ok=True)

    # 校验列
    for c in BTN_COLS + STICK_COLS:
        if c not in label_df.columns:
            raise ValueError(f"标注表缺少列: {c}")
    for c in BTN_COLS + STICK_COLS:
        if f"pred_{c}" not in pred_df.columns:
            raise ValueError(f"预测表缺少列: pred_{c}")

    # 对齐
    merged = pd.merge(
        label_df[["seq_id", "frame_idx"] + BTN_COLS + STICK_COLS],
        pred_df[["seq_id", "frame_idx"] + [f"pred_{c}" for c in BTN_COLS + STICK_COLS]],
        on=["seq_id", "frame_idx"], how="inner",
    )

    n_seg = 0
    for seq in sorted(merged["seq_id"].unique()):
        sub = merged[merged["seq_id"] == seq].sort_values("frame_idx").reset_index(drop=True)
        n = len(sub)
        seg_len = int(np.ceil(n / SEG_SPLIT))
        for k in range(SEG_SPLIT):
            seg = sub.iloc[k * seg_len:(k + 1) * seg_len]
            if len(seg) == 0:
                continue
            n_seg += 1

            frame_idx = seg["frame_idx"].to_numpy()
            label_btn = seg[BTN_COLS].to_numpy().astype(float).T      # (17, n)
            pred_btn = seg[[f"pred_{c}" for c in BTN_COLS]].to_numpy().astype(float).T
            label_stick = seg[STICK_COLS].to_numpy().astype(float).T  # (4, n)
            pred_stick = seg[[f"pred_{c}" for c in STICK_COLS]].to_numpy().astype(float).T

            # 每帧差异分数
            scores = np.array([
                diff_score(pred_btn[:, i], label_btn[:, i],
                           pred_stick[:, i], label_stick[:, i])
                for i in range(len(seg))
            ])
            top_idxs_all = np.argsort(scores)[::-1][:N_TOP]
            top_idxs = frame_idx[top_idxs_all]
            top_scores = scores[top_idxs_all]

            title = (f"seq {seq} seg {k+1} | frames {frame_idx[0]}-{frame_idx[-1]} "
                     f"| mean_diff {scores.mean():.3f}")
            out_path = os.path.join(args.out_dir, f"viz_segment_{n_seg:02d}.png")
            plot_segment(title, frame_idx, pred_btn, label_btn,
                         pred_stick, label_stick, top_idxs, top_scores, out_path)
            print(f"saved {out_path} (n={len(seg)})")

    print(f"\n完成: 共导出 {n_seg} 张 -> {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
