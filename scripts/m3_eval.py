"""m3_eval.py - NitroGen 无条件先验评测脚本.

连接 serve.py (localhost:5555), 对任意动作 parquet 逐帧评测:
  - 17 个按键布尔列 vs 模型 buttons (21 维) 名字级映射 + >0.5 阈值化
  - 4 个摇杆浮点列 vs j_left/j_right, 尺度校验 ([-1,1])
  - 输入为固定 seed 随机噪声图 (数据集无视频, 占位输入)

用法:
  python scripts/m3_eval.py [--port 5555] [--input test_elden_ring_200frames.parquet]
                            [--save] [--img-size 256] [--seq-col seq_id] [--out-dir workspace/m3_eval]
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "NitroGen-main"))
from nitrogen.inference_client import ModelClient

# ---------------------------------------------------------------- 常量
# dataset 17 键列序 (字母序, 与测试集 parquet 列一致)
BTN_COLS = [
    "back", "dpad_down", "dpad_left", "dpad_right", "dpad_up",
    "east", "guide", "left_shoulder", "left_thumb", "left_trigger",
    "north", "right_shoulder", "right_thumb", "right_trigger",
    "south", "start", "west",
]

# 模型 buttons 21 维 = BUTTON_ACTION_TOKENS 字母序 (shared.py)
# 索引: 0 BACK,1 DPAD_DOWN,2 DPAD_LEFT,3 DPAD_RIGHT,4 DPAD_UP,5 EAST,6 GUIDE,
#       7 LEFT_SHOULDER,8 LEFT_THUMB,9 LEFT_TRIGGER,10 NORTH,11 RIGHT_BOTTOM,
#       12 RIGHT_LEFT,13 RIGHT_RIGHT,14 RIGHT_SHOULDER,15 RIGHT_THUMB,
#       16 RIGHT_TRIGGER,17 RIGHT_UP,18 SOUTH,19 START,20 WEST
# dataset 17 键在 21 维中的索引 (跳过 11,12,13,17 四个 RIGHT_* 方向)
BTN_TO_MODEL_IDX = {
    "back": 0, "dpad_down": 1, "dpad_left": 2, "dpad_right": 3, "dpad_up": 4,
    "east": 5, "guide": 6, "left_shoulder": 7, "left_thumb": 8, "left_trigger": 9,
    "north": 10, "right_shoulder": 14, "right_thumb": 15, "right_trigger": 16,
    "south": 18, "start": 19, "west": 20,
}

STICK_COLS = ["j_left_x", "j_left_y", "j_right_x", "j_right_y"]
BUTTON_THRESH = 0.5


def parse_args():
    p = argparse.ArgumentParser(description="NitroGen 无条件先验评测")
    p.add_argument("--port", type=int, default=5555, help="serve.py 端口")
    p.add_argument("--input", default="test_elden_ring_200frames.parquet",
                   help="待评测 parquet 路径 (相对 workspace 或绝对)")
    p.add_argument("--save", action="store_true", help="保存 predictions.parquet")
    p.add_argument("--img-size", type=int, default=256, help="噪声占位图边长")
    p.add_argument("--seq-col", default="seq_id", help="序列分组列, 每条序列前 reset()")
    p.add_argument("--out-dir", default="workspace/m3_eval", help="输出目录")
    return p.parse_args()


def load_input(path: str) -> pd.DataFrame:
    """解析输入 parquet: 相对 workspace 或绝对路径."""
    if os.path.exists(path):
        p = path
    else:
        p = os.path.join(os.getcwd(), "workspace", path)
        if not os.path.exists(p):
            raise FileNotFoundError(f"找不到输入文件: {path} (相对 workspace 或绝对均可)")
    df = pd.read_parquet(p)
    missing = [c for c in BTN_COLS + STICK_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"输入缺少列: {missing}")
    if "seq_id" not in df.columns:
        df["seq_id"] = 0  # 无序列列则视为单条序列
    return df


def make_noise_image(rng: np.random.Generator, size: int) -> np.ndarray:
    """固定 seed 随机噪声占位图 (H,W,3) uint8."""
    return rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = load_input(args.input)
    n = len(df)
    print(f"m3_eval: {n} 帧, {df['seq_id'].nunique()} 条序列, 图像占位 {args.img_size}x{args.img_size}")

    # 摇杆尺度校验: 标注应为 [-1,1]
    for c in STICK_COLS:
        mx = float(df[c].abs().max())
        if mx > 1.5:
            raise ValueError(f"摇杆列 {c} 尺度异常 max|.|={mx:.2f} (预期 [-1,1])")
    print(f"摇杆尺度校验通过: 4 列均在 [-1,1]")

    client = ModelClient("localhost", args.port)

    # 预生成固定 seed 噪声图 (逐帧一张, 可复现)
    rng = np.random.default_rng(42)
    noise_imgs = [make_noise_image(rng, args.img_size) for _ in range(n)]

    pred_btns = np.zeros((n, len(BTN_COLS)), dtype=np.float32)
    pred_sticks = np.zeros((n, len(STICK_COLS)), dtype=np.float32)

    t0 = time.time()
    seq_prev = None
    for i in range(n):
        seq = int(df["seq_id"].iloc[i])
        if seq != seq_prev:
            client.reset()  # 每条序列前 reset, 清空 obs/action buffer
            seq_prev = seq

        pred = client.predict(noise_imgs[i])
        b = np.asarray(pred["buttons"])[0]      # (21,)
        jl = np.asarray(pred["j_left"])[0]      # (2,)
        jr = np.asarray(pred["j_right"])[0]     # (2,)
        for j, col in enumerate(BTN_COLS):
            pred_btns[i, j] = 1.0 if b[BTN_TO_MODEL_IDX[col]] > BUTTON_THRESH else 0.0
        pred_sticks[i, 0] = jl[0]
        pred_sticks[i, 1] = jl[1]
        pred_sticks[i, 2] = jr[0]
        pred_sticks[i, 3] = jr[1]

        done = i + 1
        el = time.time() - t0
        eta = el / done * (n - done)
        sys.stdout.write(f"\r  {done}/{n} 帧 | {el:.0f}s | ETA {eta:.0f}s")
        sys.stdout.flush()
    print()

    client.close()

    # ------------------------------------------------------------ 指标
    label_btns = df[BTN_COLS].to_numpy().astype(np.float32)
    label_sticks = df[STICK_COLS].to_numpy().astype(np.float32)

    # 按键准确率 = 逐帧逐按键正确 / (N*17)
    btn_acc = float((pred_btns == label_btns).mean())

    # 基线: 全零预测 (无条件先验"安静"基线的上限, 防类别失衡虚高)
    zero_acc = float((label_btns == 0).mean())  # 全零预测的准确率
    tp = int(((pred_btns == 1) & (label_btns == 1)).sum())
    fp = int(((pred_btns == 1) & (label_btns == 0)).sum())
    fn = int(((pred_btns == 0) & (label_btns == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    # 摇杆 MSE (4 列各一 + 平均)
    mse_cols = {}
    for j, c in enumerate(STICK_COLS):
        mse_cols[f"mse_{c}"] = float(((pred_sticks[:, j] - label_sticks[:, j]) ** 2).mean())
    mse_avg = float(np.mean(list(mse_cols.values())))

    # 摇杆基线 MSE: 全零预测 / 均值预测
    zero_mse_cols, mean_mse_cols = {}, {}
    for j, c in enumerate(STICK_COLS):
        l = label_sticks[:, j]
        zero_mse_cols[f"zero_mse_{c}"] = float((l ** 2).mean())
        mean_mse_cols[f"mean_mse_{c}"] = float(((l - l.mean()) ** 2).mean())
    zero_mse_avg = float(np.mean(list(zero_mse_cols.values())))
    mean_mse_avg = float(np.mean(list(mean_mse_cols.values())))

    # 摇杆 Pearson r (参考列)
    r_cols = {}
    for j, c in enumerate(STICK_COLS):
        p, l = pred_sticks[:, j], label_sticks[:, j]
        if np.std(p) == 0 or np.std(l) == 0:
            r_cols[f"r_{c}"] = 0.0
        else:
            r_cols[f"r_{c}"] = float(np.corrcoef(p, l)[0, 1])
    r_avg = float(np.mean(list(r_cols.values())))

    metrics = {
        "n_frames": n,
        "n_sequences": int(df["seq_id"].nunique()),
        "button_accuracy": btn_acc,
        "button_zero_baseline": zero_acc,      # 全零预测准确率
        "button_precision": precision,          # 按键=1 的精确率
        "button_recall": recall,                # 按键=1 的召回率
        "mse_sticks_avg": mse_avg,
        "mse_zero_baseline": zero_mse_avg,      # 全零摇杆基线
        "mse_mean_baseline": mean_mse_avg,      # 均值摇杆基线
        "pearson_sticks_avg": r_avg,
        **mse_cols, **zero_mse_cols, **mean_mse_cols,
        **r_cols,
    }
    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, "metrics.csv")
    pd.DataFrame([metrics]).to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n指标已保存 -> {out_csv}")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    if args.save:
        out_df = df[["seq_id", "frame_idx"]].copy() if "frame_idx" in df.columns else df[["seq_id"]].copy()
        for j, col in enumerate(BTN_COLS):
            out_df[f"pred_{col}"] = pred_btns[:, j]
        for j, col in enumerate(STICK_COLS):
            out_df[f"pred_{col}"] = pred_sticks[:, j]
        out_parquet = os.path.join(args.out_dir, "predictions.parquet")
        out_df.to_parquet(out_parquet)
        print(f"预测已保存 -> {out_parquet}")


if __name__ == "__main__":
    main()
