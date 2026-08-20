"""merge_predictions.py - 合并分块评测预测并输出全量指标表.

用于 T4.1②: m3_eval.py 按 --offset/--limit 分块跑完后, 把各块
predictions.parquet 合并为全量 predictions_m2.parquet, 再对全量帧
计算与标注的对比指标, 落盘 metrics_m2.csv.

用法:
  python scripts/merge_predictions.py --parts workspace/m3_eval/m2_blk1 workspace/m3_eval/m2_blk2 \
      --label workspace/m2_elden_ring_500frames.parquet \
      --out-dir workspace/m3_eval
"""
import argparse
import os

import numpy as np
import pandas as pd

from m3_eval import BTN_COLS, STICK_COLS  # 复用按键列序与摇杆列定义


def parse_args():
    p = argparse.ArgumentParser(description="合并分块评测预测并输出全量指标表")
    p.add_argument("--parts", nargs="+", required=True,
                   help="各块预测目录 (每个目录含 predictions.parquet), 按顺序合并")
    p.add_argument("--label", required=True, help="原始标注 parquet (M2/M3 数据)")
    p.add_argument("--out-dir", default="workspace/m3_eval", help="输出目录")
    p.add_argument("--out-name", default="predictions_m2.parquet", help="合并预测文件名")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    frames = []
    for d in args.parts:
        fp = os.path.join(d, "predictions.parquet")
        if not os.path.exists(fp):
            raise FileNotFoundError(f"缺少分块预测: {fp}")
        frames.append(pd.read_parquet(fp))
    merged = pd.concat(frames, ignore_index=True)

    # 按 seq_id + frame_idx 排序还原原始顺序
    sort_by = ["seq_id"]
    if "frame_idx" in merged.columns:
        sort_by.append("frame_idx")
    merged = merged.sort_values(sort_by).reset_index(drop=True)

    out_parquet = os.path.join(args.out_dir, args.out_name)
    merged.to_parquet(out_parquet)
    print(f"合并预测 -> {out_parquet} ({len(merged)} 帧)")

    # ------------------------------------------------------------ 全量指标
    label_df = pd.read_parquet(args.label)
    label_df = label_df.sort_values(sort_by).reset_index(drop=True)
    if len(label_df) != len(merged):
        raise ValueError(f"标注 {len(label_df)} 帧 != 预测 {len(merged)} 帧")

    label_btns = label_df[BTN_COLS].to_numpy().astype(np.float32)
    pred_btns = merged[[f"pred_{c}" for c in BTN_COLS]].to_numpy().astype(np.float32)
    label_sticks = label_df[STICK_COLS].to_numpy().astype(np.float32)
    pred_sticks = merged[[f"pred_{c}" for c in STICK_COLS]].to_numpy().astype(np.float32)

    btn_acc = float((pred_btns == label_btns).mean())
    zero_acc = float((label_btns == 0).mean())
    tp = int(((pred_btns == 1) & (label_btns == 1)).sum())
    fp = int(((pred_btns == 1) & (label_btns == 0)).sum())
    fn = int(((pred_btns == 0) & (label_btns == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    mse_cols, zero_mse_cols, mean_mse_cols, r_cols = {}, {}, {}, {}
    for j, c in enumerate(STICK_COLS):
        p, l = pred_sticks[:, j], label_sticks[:, j]
        mse_cols[f"mse_{c}"] = float(((p - l) ** 2).mean())
        zero_mse_cols[f"zero_mse_{c}"] = float((l ** 2).mean())
        mean_mse_cols[f"mean_mse_{c}"] = float(((l - l.mean()) ** 2).mean())
        if np.std(p) == 0 or np.std(l) == 0:
            r_cols[f"r_{c}"] = 0.0
        else:
            r_cols[f"r_{c}"] = float(np.corrcoef(p, l)[0, 1])

    mse_avg = float(np.mean(list(mse_cols.values())))
    zero_mse_avg = float(np.mean(list(zero_mse_cols.values())))
    mean_mse_avg = float(np.mean(list(mean_mse_cols.values())))
    r_avg = float(np.mean(list(r_cols.values())))

    metrics = {
        "n_frames": len(merged),
        "n_sequences": int(label_df["seq_id"].nunique()),
        "button_accuracy": btn_acc,
        "button_baseline_50": 0.50,
        "button_vs_baseline": btn_acc - 0.50,
        "button_zero_baseline": zero_acc,
        "button_precision": precision,
        "button_recall": recall,
        "mse_sticks_avg": mse_avg,
        "mse_zero_baseline": zero_mse_avg,
        "mse_mean_baseline": mean_mse_avg,
        "pearson_sticks_avg": r_avg,
        "pearson_baseline_04": 0.40,
        "pearson_vs_baseline": r_avg - 0.40,
        **mse_cols, **zero_mse_cols, **mean_mse_cols, **r_cols,
    }
    out_csv = os.path.join(args.out_dir, "metrics_m2.csv")
    pd.DataFrame([metrics]).to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"指标已保存 -> {out_csv}")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
