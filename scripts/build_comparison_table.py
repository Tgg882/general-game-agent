"""build_comparison_table.py - T5.1 演示对比表 (MVP 5).

将 M3 评测预测与人工标注按 (seq_id, frame_idx) 对齐, 生成逐帧对比表:
  - 17 个按键各拆两列: <btn>_pred / <btn>_label (共 34 列)
  - 4 个摇杆各拆两列: <stick>_pred / <stick>_label (共 8 列)
  - 最后一列 frame_diff_score = 按键错误数/17 + 摇杆欧氏距离/2 (0~1, 越小越像)

用法:
  python scripts/build_comparison_table.py [--pred workspace/m3_eval/predictions.parquet]
                                           [--label workspace/test_elden_ring_200frames.parquet]
                                           [--out workspace/m3_eval/comparison_table.csv]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

# 与 m3_eval.py 保持一致
BTN_COLS = [
    "back", "dpad_down", "dpad_left", "dpad_right", "dpad_up",
    "east", "guide", "left_shoulder", "left_thumb", "left_trigger",
    "north", "right_shoulder", "right_thumb", "right_trigger",
    "south", "start", "west",
]
STICK_COLS = ["j_left_x", "j_left_y", "j_right_x", "j_right_y"]


def parse_args():
    p = argparse.ArgumentParser(description="T5.1 演示对比表: 预测 vs 人工标注 逐帧对比")
    p.add_argument("--pred", default="workspace/m3_eval/predictions.parquet",
                   help="预测 parquet (M3 评测产物)")
    p.add_argument("--label", default="workspace/test_elden_ring_200frames.parquet",
                   help="人工标注 parquet")
    p.add_argument("--out", default="workspace/m3_eval/comparison_table.csv",
                   help="输出 CSV 路径 (utf-8-sig)")
    return p.parse_args()


def _resolve(path: str) -> str:
    if os.path.exists(path):
        return path
    p = os.path.join(os.getcwd(), path)
    if os.path.exists(p):
        return p
    raise FileNotFoundError(f"找不到文件: {path}")


def build_comparison_table(pred_df: pd.DataFrame, label_df: pd.DataFrame) -> pd.DataFrame:
    """对齐两表并生成逐帧对比 DataFrame."""
    # 校验必需列
    missing_pred = [f"pred_{c}" for c in BTN_COLS + STICK_COLS]
    missing_pred = [c for c in missing_pred if c not in pred_df.columns]
    if missing_pred:
        raise ValueError(f"预测表缺少列: {missing_pred}")
    missing_label = [c for c in BTN_COLS + STICK_COLS if c not in label_df.columns]
    if missing_label:
        raise ValueError(f"标注表缺少列: {missing_label}")

    # 对齐键
    keys = ["seq_id", "frame_idx"]
    pred = pred_df[keys + [f"pred_{c}" for c in BTN_COLS + STICK_COLS]].copy()
    label = label_df[keys + BTN_COLS + STICK_COLS].copy()
    merged = pd.merge(pred, label, on=keys, how="inner", suffixes=("_x", "_y"))
    if len(merged) != len(pred_df):
        raise ValueError(f"对齐后行数 {len(merged)} != 预测行数 {len(pred_df)} (可能存在未对齐帧)")

    # 列结构: frame_idx, seq_id, 17 键 pred/label, 4 摇杆 pred/label, diff_score
    btn_pred_cols = [f"{c}_pred" for c in BTN_COLS]
    btn_label_cols = [f"{c}_label" for c in BTN_COLS]
    stick_pred_cols = [f"{c}_pred" for c in STICK_COLS]
    stick_label_cols = [f"{c}_label" for c in STICK_COLS]
    merged = merged.rename(columns={
        f"pred_{c}": f"{c}_pred" for c in BTN_COLS + STICK_COLS
    })
    out = pd.DataFrame(index=merged.index)
    out["frame_idx"] = merged["frame_idx"]
    out["seq_id"] = merged["seq_id"]
    for c in BTN_COLS:
        out[f"{c}_pred"] = merged[f"{c}_pred"]
        out[f"{c}_label"] = merged[c]  # 标注列保持原始名 (无前缀)
    for c in STICK_COLS:
        out[f"{c}_pred"] = merged[f"{c}_pred"]
        out[f"{c}_label"] = merged[c]

    # frame_diff_score = 按键错误数/17 + 摇杆欧氏距离/2 (0~1, 越小越像)
    btn_err = (out[btn_pred_cols].to_numpy() != out[btn_label_cols].to_numpy()).sum(axis=1).astype(float) / len(BTN_COLS)
    stick_dist = np.sqrt(((out[stick_pred_cols].to_numpy() - out[stick_label_cols].to_numpy()) ** 2).sum(axis=1)) / 2.0
    out["frame_diff_score"] = btn_err + stick_dist
    return out.reset_index(drop=True)


def main():
    args = parse_args()
    pred_df = pd.read_parquet(_resolve(args.pred))
    label_df = pd.read_parquet(_resolve(args.label))
    print(f"预测: {pred_df.shape}, 标注: {label_df.shape}")

    table = build_comparison_table(pred_df, label_df)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    table.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"对比表已保存 -> {args.out} ({len(table)} 行 x {table.shape[1]} 列)")
    print("\n前 10 行预览 (节选 frame_idx/seq_id/部分列/frame_diff_score):")
    preview_cols = ["frame_idx", "seq_id", "south_pred", "south_label",
                    "j_left_y_pred", "j_left_y_label", "frame_diff_score"]
    preview_cols = [c for c in preview_cols if c in table.columns]
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(table[preview_cols].head(10).to_string(index=False))
    print(f"\nframe_diff_score 统计: min={table['frame_diff_score'].min():.4f}, "
          f"mean={table['frame_diff_score'].mean():.4f}, max={table['frame_diff_score'].max():.4f}")


if __name__ == "__main__":
    main()
