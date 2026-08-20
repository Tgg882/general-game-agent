# -*- coding: utf-8 -*-
"""任务2b：M2 统计分析 —— 按键分布、摇杆分布、10条序列可视化。"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

WS = Path(r"C:\Users\HP\Desktop\general-game-agent\workspace")
OUT = WS / "m2_analysis"
(OUT / "sequences").mkdir(parents=True, exist_ok=True)

BTN_COLS = ["back", "dpad_down", "dpad_left", "dpad_right", "dpad_up", "east", "guide",
            "left_shoulder", "left_thumb", "left_trigger", "north", "right_shoulder",
            "right_thumb", "right_trigger", "south", "start", "west"]
BTN_ZH = {
    "dpad_up": "十字键上", "dpad_down": "十字键下", "dpad_left": "十字键左", "dpad_right": "十字键右",
    "left_shoulder": "左肩键L1", "left_trigger": "左扳机L2", "left_thumb": "左摇杆按下L3",
    "right_shoulder": "右肩键R1", "right_trigger": "右扳机R2", "right_thumb": "右摇杆按下R3",
    "south": "南键(○/A)", "east": "东键(×/B)", "north": "北键(△/Y)", "west": "西键(□/X)",
    "back": "返回键", "start": "开始键", "guide": "PS键",
}

df = pd.read_parquet(WS / "m2_elden_ring_500frames.parquet")
print(f"M2 数据: {len(df)} 帧, {df['seq_id'].nunique()} 条序列")

# ============ 1. 按键分布 ============
btn_rate = df[BTN_COLS].mean().sort_values(ascending=False)
idle_rate = (df[BTN_COLS].sum(axis=1) == 0).mean()
n_btns_per_frame = df[BTN_COLS].sum(axis=1)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
# 左：按键按压率
colors = ["#d62728" if v > 0.1 else "#7f7f7f" for v in btn_rate.values]
axes[0].bar(range(len(btn_rate)), btn_rate.values * 100, color=colors)
axes[0].set_xticks(range(len(btn_rate)))
axes[0].set_xticklabels([BTN_ZH.get(c, c) for c in btn_rate.index], rotation=60, ha="right", fontsize=9)
axes[0].set_ylabel("按压率 (%)")
axes[0].set_title("Elden Ring 500帧 · 17个按键按压率分布")
for i, v in enumerate(btn_rate.values * 100):
    axes[0].text(i, v + 0.3, f"{v:.1f}", ha="center", fontsize=8)
# 右：每帧按键数直方图
axes[1].hist(n_btns_per_frame, bins=range(0, n_btns_per_frame.max() + 2), color="#1f77b4", edgecolor="white")
axes[1].set_xlabel("每帧同时按下的按键数")
axes[1].set_ylabel("帧数")
axes[1].set_title(f"每帧按键数分布（空按键帧占比 {idle_rate*100:.1f}%）")
plt.tight_layout()
plt.savefig(OUT / "button_distribution.png", dpi=150)
plt.close()
print("Saved button_distribution.png")

# ============ 2. 摇杆分布 ============
js = df[["j_left_x", "j_left_y", "j_right_x", "j_right_y"]]
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
titles = [("j_left_x", "左摇杆 X"), ("j_left_y", "左摇杆 Y"), ("j_right_x", "右摇杆 X"),
          ("j_right_y", "右摇杆 Y")]
for i, (col, name) in enumerate(titles):
    ax = axes[i // 3][i % 3]
    ax.hist(js[col], bins=40, color="#2ca02c", edgecolor="white")
    ax.axvline(js[col].mean(), color="#d62728", linestyle="--", label=f"均值 {js[col].mean():.3f}")
    ax.set_title(f"{name} 分布 (std={js[col].std():.3f})")
    ax.legend()
# 右上：左摇杆轨迹散点
ax = axes[0][2]
ax.scatter(df["j_left_x"], df["j_left_y"], s=4, alpha=0.4, c="#1f77b4")
ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_title("左摇杆位置散点")
# 右中：右摇杆轨迹散点
ax = axes[1][2]
ax.scatter(df["j_right_x"], df["j_right_y"], s=4, alpha=0.4, c="#ff7f0e")
ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_title("右摇杆位置散点")
axes[1][0].remove(); axes[1][1].remove()
fig.suptitle("Elden Ring 500帧 · 摇杆分布", fontsize=14)
plt.tight_layout()
plt.savefig(OUT / "joystick_distribution.png", dpi=150)
plt.close()
print("Saved joystick_distribution.png")

# ============ 3. 10条序列可视化 ============
for seq_id in range(10):
    seg = df[df["seq_id"] == seq_id].reset_index(drop=True)
    vid = seg["video_id"].iloc[0]
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), gridspec_kw={"height_ratios": [17, 4]})
    # 上：按键事件栅格图
    pressed = seg[BTN_COLS].values.T
    for bi, b in enumerate(BTN_COLS):
        xs = np.where(pressed[bi] == 1)[0]
        if len(xs) > 0:
            axes[0].scatter(xs, [bi] * len(xs), s=12, marker="s", color="#d62728")
    axes[0].set_yticks(range(len(BTN_COLS)))
    axes[0].set_yticklabels([BTN_ZH.get(c, c) for c in BTN_COLS], fontsize=8)
    axes[0].set_xlabel("帧序号")
    axes[0].set_title(f"序列 {seq_id}（视频 {vid} · 50帧）· 按键事件")
    axes[0].set_xlim(-1, 50)
    axes[0].grid(axis="x", alpha=0.2)
    # 下：摇杆轨迹
    axes[1].plot(seg["frame_idx"], seg["j_left_x"], label="左X", color="#1f77b4")
    axes[1].plot(seg["frame_idx"], seg["j_left_y"], label="左Y", color="#2ca02c")
    axes[1].plot(seg["frame_idx"], seg["j_right_x"], label="右X", color="#ff7f0e")
    axes[1].plot(seg["frame_idx"], seg["j_right_y"], label="右Y", color="#d62728")
    axes[1].legend(ncol=4, fontsize=8, loc="upper right")
    axes[1].set_ylim(-1.1, 1.1)
    axes[1].set_xlabel("帧序号")
    axes[1].set_title("摇杆轨迹")
    plt.tight_layout()
    plt.savefig(OUT / "sequences" / f"seq_{seq_id:02d}.png", dpi=130)
    plt.close()
print("Saved 10 sequence figures")

# ============ 4. 统计摘要 ============
summary_lines = [
    "# M2 统计摘要：Elden Ring 500帧\n",
    f"- 总帧数: {len(df)}，序列数: {df['seq_id'].nunique()}（每条50帧，来自10个不同视频）\n",
    f"- 空按键帧占比: {idle_rate*100:.1f}%\n",
    f"- 平均每帧按键数: {n_btns_per_frame.mean():.2f}，最大: {n_btns_per_frame.max()}\n",
    "\n## 按键按压率 Top 10\n",
    "\n".join(f"- {BTN_ZH.get(c, c)} ({c}): {r*100:.2f}%" for c, r in btn_rate.head(10).items()),
    "\n\n## 摇杆统计\n",
    f"- 左摇杆: |x|均值 {df['j_left_x'].abs().mean():.3f}, |y|均值 {df['j_left_y'].abs().mean():.3f}\n",
    f"- 右摇杆: |x|均值 {df['j_right_x'].abs().mean():.3f}, |y|均值 {df['j_right_y'].abs().mean():.3f}\n",
    f"- 左摇杆死区(|v|<0.1)占比: {((df['j_left_x'].abs()<0.1)&(df['j_left_y'].abs()<0.1)).mean()*100:.1f}%\n",
    f"- 右摇杆死区占比: {((df['j_right_x'].abs()<0.1)&(df['j_right_y'].abs()<0.1)).mean()*100:.1f}%\n",
]
(OUT / "m2_stats_summary.md").write_text("".join(summary_lines), encoding="utf-8")
btn_rate.to_csv(OUT / "button_press_rate.csv", encoding="utf-8-sig")
js.describe().to_csv(OUT / "joystick_stats.csv", encoding="utf-8-sig")
print("Saved summary + CSVs")
print("\n=== 控制台摘要 ===")
print(f"空按键帧占比: {idle_rate*100:.1f}%")
print("按键按压率 Top 5:")
for c, r in btn_rate.head(5).items():
    print(f"  {c:<16} {r*100:6.2f}%")
