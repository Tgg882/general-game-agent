#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""m3_eval_v2.py - 真实视频帧评测

与 m3_eval.py 的指标口径完全一致 (按键 17 维映射 + >0.5 阈值, 摇杆 MSE/Pearson r)。
唯一区别:
  - 输入: 真实游戏画面帧 (test_frames/) 替代随机噪声图
  - 额外: 保存全部 action_horizon 步输出, 离线扫描 offset 找最优帧对齐
          (验证 "模型输出步 vs 标注帧" 的对齐假设)

用法:
  1. 先启动 serve.py:  python scripts/serve.py ng.pt --port 5555
  2. 运行评测:  python scripts/m3_eval_v2.py --save
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "NitroGen-main"))
from nitrogen.inference_client import ModelClient

BTN_COLS = [
    "back", "dpad_down", "dpad_left", "dpad_right", "dpad_up",
    "east", "guide", "left_shoulder", "left_thumb", "left_trigger",
    "north", "right_shoulder", "right_thumb", "right_trigger",
    "south", "start", "west",
]
BTN_TO_MODEL_IDX = {
    "back": 0, "dpad_down": 1, "dpad_left": 2, "dpad_right": 3, "dpad_up": 4,
    "east": 5, "guide": 6, "left_shoulder": 7, "left_thumb": 8, "left_trigger": 9,
    "north": 10, "right_shoulder": 14, "right_thumb": 15, "right_trigger": 16,
    "south": 18, "start": 19, "west": 20,
}
STICK_COLS = ["j_left_x", "j_left_y", "j_right_x", "j_right_y"]
BUTTON_THRESH = 0.5

WS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "workspace")
FRAMES_DIR = os.path.join(WS, "test_frames")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--input", default="test_elden_ring_200frames.parquet")
    p.add_argument("--frames-dir", default=FRAMES_DIR)
    p.add_argument("--save", action="store_true")
    p.add_argument("--out-dir", default=os.path.join(WS, "m3_eval"))
    p.add_argument("--max-offset", type=int, default=17, help="离线对齐扫描的最大步偏移")
    p.add_argument("--out-name", default="v2",
                   help="输出文件名前缀: metrics_{name}.csv / r_by_offset_{name}.csv / predictions_{name}.parquet")
    return p.parse_args()


def load_frames(frames_dir, n, seq_ids, frame_idxs):
    imgs, missing = [], []
    for i in range(n):
        path = os.path.join(frames_dir,
                            f"seq{int(seq_ids[i]):02d}_frame{int(frame_idxs[i]):04d}.jpg")
        if os.path.exists(path):
            imgs.append(np.array(Image.open(path).convert("RGB"), dtype=np.uint8))
        else:
            missing.append(path)
    if missing:
        print(f"WARN: {len(missing)} 帧缺失, 用黑图替代")
        for path in missing:
            imgs.append(np.zeros((256, 256, 3), dtype=np.uint8))
    return np.array(imgs), len(missing)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    input_path = args.input if os.path.exists(args.input) else os.path.join(WS, args.input)
    df = pd.read_parquet(input_path)
    n = len(df)
    n_seq = df["seq_id"].nunique()
    print(f"m3_eval_v2: {n} 帧, {n_seq} 序列")
    print(f"  帧目录: {args.frames_dir}")

    real_imgs, n_missing = load_frames(args.frames_dir, n, df["seq_id"].values, df["frame_idx"].values)
    print(f"  加载帧: shape={real_imgs.shape}, missing={n_missing}")

    client = ModelClient("localhost", args.port)

    # 逐帧推理: 先跑第一帧确定 action_horizon
    seq_prev = None
    H = args.max_offset + 1
    pred_btns = None
    pred_sticks = None
    t0 = time.time()
    for i in range(n):
        seq = int(df["seq_id"].iloc[i])
        if seq != seq_prev:
            client.reset()
            seq_prev = seq
        pred = client.predict(real_imgs[i])
        b = np.asarray(pred["buttons"])          # (horizon, 21)
        jl = np.asarray(pred["j_left"])          # (horizon, 2)
        jr = np.asarray(pred["j_right"])         # (horizon, 2)
        if pred_btns is None:
            H = min(b.shape[0], args.max_offset + 1)
            pred_btns = np.zeros((H, n, len(BTN_COLS)), dtype=np.float32)
            pred_sticks = np.zeros((H, n, len(STICK_COLS)), dtype=np.float32)
            print(f"  action_horizon 实际: {b.shape[0]}, 使用前 {H} 步")
        for k in range(H):
            for j, col in enumerate(BTN_COLS):
                pred_btns[k, i, j] = 1.0 if b[k, BTN_TO_MODEL_IDX[col]] > BUTTON_THRESH else 0.0
            pred_sticks[k, i, 0] = jl[k, 0]
            pred_sticks[k, i, 1] = jl[k, 1]
            pred_sticks[k, i, 2] = jr[k, 0]
            pred_sticks[k, i, 3] = jr[k, 1]
        done = i + 1
        el = time.time() - t0
        eta = el / done * (n - done)
        sys.stdout.write(f"\r  {done}/{n} 帧 | {el:.0f}s | ETA {eta:.0f}s")
        sys.stdout.flush()
    print()
    client.close()

    # 标注
    label_btns = df[BTN_COLS].to_numpy().astype(np.float32)
    label_sticks = df[STICK_COLS].to_numpy().astype(np.float32)

    def compute_metrics(pb, ps):
        btn_acc = float((pb == label_btns).mean())
        tp = int(((pb == 1) & (label_btns == 1)).sum())
        fp = int(((pb == 1) & (label_btns == 0)).sum())
        fn = int(((pb == 0) & (label_btns == 1)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        mse = float((((ps - label_sticks) ** 2).mean()))
        rs = []
        for j in range(len(STICK_COLS)):
            p, l = ps[:, j], label_sticks[:, j]
            rs.append(float(np.corrcoef(p, l)[0, 1]) if np.std(p) > 0 and np.std(l) > 0 else 0.0)
        return btn_acc, precision, recall, mse, float(np.mean(rs))

    # 主指标: offset=0 (与旧评测可比)
    btn_acc, precision, recall, mse_avg, r_avg = compute_metrics(pred_btns[0], pred_sticks[0])
    zero_acc = float((label_btns == 0).mean())
    zero_mse_avg = float((label_sticks ** 2).mean())

    # 离线 offset 扫描 (验证对齐假设)
    r_by_offset = {}
    for k in range(H):
        r_by_offset[f"r_offset{k}"] = compute_metrics(pred_btns[k], pred_sticks[k])[4]
    best_off = max(range(H), key=lambda k: r_by_offset[f"r_offset{k}"])
    r_best = r_by_offset[f"r_offset{best_off}"]

    metrics = {
        "n_frames": n, "n_sequences": n_seq, "n_missing_frames": n_missing,
        "button_accuracy": btn_acc, "button_zero_baseline": zero_acc,
        "button_precision": precision, "button_recall": recall,
        "mse_sticks_avg": mse_avg, "mse_zero_baseline": zero_mse_avg,
        "pearson_sticks_avg": r_avg, "pearson_baseline_04": 0.40,
        "pearson_vs_baseline": r_avg - 0.40,
        "best_offset": best_off, "pearson_best_offset": r_best,
        **r_by_offset,
    }
    out_csv = os.path.join(args.out_dir, f"metrics_{args.out_name}.csv")
    pd.DataFrame([metrics]).to_csv(out_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame([r_by_offset]).to_csv(os.path.join(args.out_dir, f"r_by_offset_{args.out_name}.csv"),
                                       index=False, encoding="utf-8-sig")

    print(f"\n{'='*50}")
    print(f"  按键准确率:       {btn_acc:.4f}  (全零基线 {zero_acc:.4f})")
    print(f"  按键 precision:   {precision:.4f}")
    print(f"  按键 recall:      {recall:.4f}")
    print(f"  摇杆 MSE:         {mse_avg:.4f}  (零基线 {zero_mse_avg:.4f})")
    print(f"  摇杆 Pearson r:   {r_avg:.4f}  (offset=0, 基线 0.40)")
    print(f"  最优 offset:      {best_off} -> r = {r_best:.4f}")
    print(f"{'='*50}")
    print(f"指标已保存 -> {out_csv}")

    if args.save:
        out_df = df[["seq_id", "frame_idx"]].copy()
        for j, col in enumerate(BTN_COLS):
            out_df[f"pred_{col}"] = pred_btns[0, :, j]
        for j, col in enumerate(STICK_COLS):
            out_df[f"pred_{col}"] = pred_sticks[0, :, j]
        out_df.to_parquet(os.path.join(args.out_dir, f"predictions_{args.out_name}.parquet"))
        print(f"预测已保存 -> predictions_{args.out_name}.parquet")


if __name__ == "__main__":
    main()
