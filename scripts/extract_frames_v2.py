# -*- coding: utf-8 -*-
"""任务2a (重做)：基于"活跃度"挑选 10 个视频中按键+摇杆都活跃的窗口作为 M2 500 帧。"""
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

WS = Path(r"C:\Users\HP\Desktop\general-game-agent\workspace")
CHUNK_LIST = WS / "elden_ring_chunks.json"
random.seed(42)

BTN_COLS = ["back", "dpad_down", "dpad_left", "dpad_right", "dpad_up", "east", "guide",
            "left_shoulder", "left_thumb", "left_trigger", "north", "right_shoulder",
            "right_thumb", "right_trigger", "south", "start", "west"]

with open(CHUNK_LIST) as f:
    chunks = json.load(f)

from collections import defaultdict
by_video = defaultdict(list)
for r in chunks:
    by_video[r["video_id"]].append(r)

videos_sorted = sorted(by_video.items(), key=lambda kv: -sum(c["chunk_size"] for c in kv[1]))
m2_videos = [v for v, _ in videos_sorted[:10]]
test_videos = [v for v, _ in videos_sorted[10:14]]


def activity_score(pq_path):
    """读 parquet，计算活跃度分数：按键按压率 + 摇杆非零比例。"""
    tbl = pq.read_table(pq_path, columns=BTN_COLS + ["j_left", "j_right"])
    df = tbl.to_pandas()
    btn_rate = df[BTN_COLS].mean().mean()
    n_btns = df[BTN_COLS].sum(axis=1)
    any_btn = (n_btns > 0).mean()
    jl = np.array([list(v) for v in df["j_left"]], dtype=float)
    jr = np.array([list(v) for v in df["j_right"]], dtype=float)
    l_active = ((np.abs(jl) > 0.1).any(axis=1)).mean()
    r_active = ((np.abs(jr) > 0.1).any(axis=1)).mean()
    return btn_rate + 0.5 * any_btn + 0.3 * (l_active + r_active)


def pick_active_window(pq_path, win=50, step=50, top_k=3):
    """在 chunk 中按 50 帧滑动窗口扫，挑最活跃的 1 个窗口。"""
    tbl = pq.read_table(pq_path)
    df = tbl.to_pandas()
    n = len(df)
    best_score, best_start = -1, 200
    for s in range(200, n - win - 1, step):
        seg = df.iloc[s:s + win]
        score = seg[BTN_COLS].sum().sum()
        jl = np.array([list(v) for v in seg["j_left"]], dtype=float)
        jr = np.array([list(v) for v in seg["j_right"]], dtype=float)
        score += 0.1 * (np.abs(jl).sum() + np.abs(jr).sum())
        if score > best_score:
            best_score, best_start = score, s
    return best_start, best_score


def extract_sequences(video_ids, seq_frames=50, tag=""):
    rows = []
    seq_id = 0
    for vid in video_ids:
        # 选 1 个 chunk：抽几个候选，跑 activity_score 取最高
        cand = by_video[vid]
        cand = [c for c in cand if c["chunk_size"] >= seq_frames + 400]
        random.shuffle(cand)
        sample = cand[:6]
        scored = [(activity_score(Path(c["chunk_dir"]) / "actions_raw.parquet"), c) for c in sample]
        scored.sort(key=lambda x: -x[0])
        c = scored[0][1]
        start, score = pick_active_window(Path(c["chunk_dir"]) / "actions_raw.parquet",
                                           win=seq_frames, step=50)
        pq_path = Path(c["chunk_dir"]) / "actions_raw.parquet"
        df = pq.read_table(pq_path).to_pandas()
        seg = df.iloc[start:start + seq_frames].reset_index(drop=True)
        jl = np.array([list(v) for v in seg["j_left"]], dtype=float)
        jr = np.array([list(v) for v in seg["j_right"]], dtype=float)
        for i in range(seq_frames):
            row = {b: int(seg.loc[i, b]) for b in BTN_COLS}
            row.update({
                "j_left_x": jl[i, 0], "j_left_y": jl[i, 1],
                "j_right_x": jr[i, 0], "j_right_y": jr[i, 1],
                "seq_id": seq_id, "frame_idx": i,
                "video_id": vid, "chunk_id": c["chunk_id"],
            })
            rows.append(row)
        print(f"  [{tag}] seq {seq_id}: video={vid} chunk={c['chunk_id']} frames[{start}:{start+seq_frames}] score={score:.1f}")
        seq_id += 1
    return pd.DataFrame(rows)


print("Extracting M2 (500 frames, activity-picked)...")
m2_df = extract_sequences(m2_videos, seq_frames=50, tag="M2")
print(f"M2 total: {len(m2_df)} frames, {m2_df['seq_id'].nunique()} sequences\n")

print("Extracting Test (200 frames, same activity-pick strategy)...")
test_df = extract_sequences(test_videos, seq_frames=50, tag="TEST")
print(f"Test total: {len(test_df)} frames\n")

m2_keys = set(zip(m2_df["video_id"], m2_df["chunk_id"]))
test_keys = set(zip(test_df["video_id"], test_df["chunk_id"]))
assert not (m2_keys & test_keys), "M2 与测试集重叠！"
print("Check passed: M2 and test set use different videos/chunks.\n")

m2_out = WS / "m2_elden_ring_500frames.parquet"
test_out = WS / "test_elden_ring_200frames.parquet"
m2_df.to_parquet(m2_out, index=False)
test_df.to_parquet(test_out, index=False)
print(f"Saved: {m2_out}")
print(f"Saved: {test_out}")

# 简单校验
print("\nM2 帧级校验:")
n_btns = m2_df[BTN_COLS].sum(axis=1)
print(f"  空按键帧占比: {(n_btns==0).mean()*100:.1f}%")
print(f"  平均每帧按键数: {n_btns.mean():.2f}")
print(f"  左摇杆活跃(|v|>0.1)占比: {((m2_df['j_left_x'].abs()>0.1)|(m2_df['j_left_y'].abs()>0.1)).mean()*100:.1f}%")
print(f"  右摇杆活跃(|v|>0.1)占比: {((m2_df['j_right_x'].abs()>0.1)|(m2_df['j_right_y'].abs()>0.1)).mean()*100:.1f}%")
