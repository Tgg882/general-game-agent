"""probe_serve.py — T2.1 冒烟测试后半段：50 帧探针。

连 serve.py，用固定 seed 生成的随机图像序列连发 N 次 predict
（每次返回 action_horizon=18 帧动作），统计：
  - 按键按压率（>0.5 阈值化）、逐键按压率
  - 左/右摇杆 x/y 的均值、标准差
重跑 R 次（相同图像序列）对比统计量波动，判断模型输出是否稳定
（Flow Matching 采样随机，预期逐帧不同但分布应接近）。

用法: python scripts/probe_serve.py [--port 5555] [--predicts 50] [--runs 2]
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "NitroGen-main"))

from nitrogen.inference_client import ModelClient

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "..", "workspace", "probe")


def summarize(B, JL, JR):
    """输入: B (N,18,21) buttons, JL/JR (N,18,2) 摇杆。返回统计 dict。"""
    pressed = B > 0.5
    return {
        "n_actions": int(B.shape[0] * B.shape[1]),
        "total_press_rate": float(pressed.mean()),
        "buttons_abs_mean": float(np.abs(B).mean()),
        "jl_x_mean": float(JL[..., 0].mean()),
        "jl_y_mean": float(JL[..., 1].mean()),
        "jl_x_std": float(JL[..., 0].std()),
        "jl_y_std": float(JL[..., 1].std()),
        "jr_x_mean": float(JR[..., 0].mean()),
        "jr_y_mean": float(JR[..., 1].mean()),
        "jr_x_std": float(JR[..., 0].std()),
        "jr_y_std": float(JR[..., 1].std()),
        "per_button": pressed.mean(axis=(0, 1)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--predicts", type=int, default=50, help="predict 次数（每次返回 18 帧动作）")
    ap.add_argument("--runs", type=int, default=2, help="重跑次数（相同输入序列）")
    args = ap.parse_args()

    # 固定 seed 预生成图像序列：保证两次 run 输入完全一致
    imgs = np.random.default_rng(0).integers(0, 255, size=(args.predicts, 256, 256, 3), dtype=np.uint8)
    print(f"probe: {args.predicts} predicts x {args.runs} runs, action_horizon=18/次")

    rows = []
    with ModelClient("localhost", args.port) as client:
        for run in range(1, args.runs + 1):
            print(f"=== run {run}/{args.runs} ===")
            client.reset()
            B, JL, JR = [], [], []
            for i, img in enumerate(imgs, 1):
                pred = client.predict(img)
                B.append(pred["buttons"])
                JL.append(pred["j_left"])
                JR.append(pred["j_right"])
                if i % 10 == 0:
                    print(f"  predicted {i}/{args.predicts}")
            B = np.asarray(B)
            JL = np.asarray(JL)
            JR = np.asarray(JR)
            s = summarize(B, JL, JR)
            print(f"  press_rate={s['total_press_rate']:.4f} | buttons|mean={s['buttons_abs_mean']:.4f}")
            print(f"  j_left  mean=({s['jl_x_mean']:+.4f},{s['jl_y_mean']:+.4f}) std=({s['jl_x_std']:.4f},{s['jl_y_std']:.4f})")
            print(f"  j_right mean=({s['jr_x_mean']:+.4f},{s['jr_y_mean']:+.4f}) std=({s['jr_x_std']:.4f},{s['jr_y_std']:.4f})")
            print("  per-button press rates:", " ".join(f"{v:.2f}" for v in s["per_button"]))
            row = {k: v for k, v in s.items() if k != "per_button"}
            row["run"] = run
            row["n_predicts"] = args.predicts
            rows.append(row)

    # 保存摘要 CSV
    os.makedirs(OUT_DIR, exist_ok=True)
    fields = ["run", "n_predicts", "n_actions", "total_press_rate", "buttons_abs_mean",
              "jl_x_mean", "jl_y_mean", "jl_x_std", "jl_y_std",
              "jr_x_mean", "jr_y_mean", "jr_x_std", "jr_y_std"]
    out = os.path.join(OUT_DIR, "probe_stats.csv")
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"Saved -> {out}")

    # 两次 run 波动
    if len(rows) >= 2:
        for col in ["total_press_rate", "jl_x_mean", "jl_y_mean", "jr_x_mean", "jr_y_mean"]:
            vals = [r[col] for r in rows]
            diff = abs(vals[0] - vals[1])
            print(f"  stability {col}: run1={vals[0]:.4f} run2={vals[1]:.4f} |diff|={diff:.4f}")


if __name__ == "__main__":
    main()
