# -*- coding: utf-8 -*-
"""任务2a：从 elden_ring 提取 M2 的 500 帧（10条序列×50帧）和 M3 的 200 帧测试集（4条序列×50帧）。"""
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

WS = Path(r"C:\Users\HP\Desktop\general-game-agent\workspace")
CHUNK_LIST = WS / "elden_ring_chunks.json"

BTN_COLS = ["back", "dpad_down", "dpad_left", "dpad_right", "dpad_up", "east", "guide",
            "left_shoulder", "left_thumb", "left_trigger", "north", "right_shoulder",
            "right_thumb", "right_trigger", "south", "start", "west"]

random.seed(42)

with open(CHUNK_LIST, "r", encoding="utf-8") as f:
    chunks = json.load(f)

# 按视频分组，取帧数最多的 14 个视频：前 10 个给 M2，后 4 个给测试集
from collections import defaultdict
by_video = defaultdict(list)
for r in chunks:
    by_video[r["video_id"]].append(r)

videos_sorted = sorted(by_video.items(), key=lambda kv: -sum(c["chunk_size"] for c in kv[1]))
m2_videos = [v for v, _ in videos_sorted[:10]]
test_videos = [v for v, _ in videos_sorted[10:14]]
print(f"M2 videos (10): {m2_videos}")
print(f"Test videos (4): {test_videos}")


def extract_sequences(video_ids, seq_frames=50, n_seq_per_video=1, tag=""):
    """每个视频随机挑一个 chunk，从该 chunk 中间取 seq_frames 帧连续序列。"""
    rows = []
    seq_id = 0
    for vid in video_ids:
        cand = by_video[vid]
        # 挑 chunk_size 足够大的 chunk
        cand = [c for c in cand if c["chunk_size"] >= seq_frames + 200]
        c = random.choice(cand)
        parquet_path = Path(c["chunk_dir"]) / "actions_raw.parquet"
        df = pq.read_table(parquet_path).to_pandas()
        n = len(df)
        # 从中段开始，避开开头可能的挂机
        start = random.randint(200, n - seq_frames - 1)
        seg = df.iloc[start:start + seq_frames].reset_index(drop=True)
        # 展开摇杆
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
        print(f"  [{tag}] seq {seq_id}: video={vid} chunk={c['chunk_id']} frames[{start}:{start+seq_frames}]")
        seq_id += 1
    return pd.DataFrame(rows)


print("\nExtracting M2 (500 frames)...")
m2_df = extract_sequences(m2_videos, seq_frames=50, n_seq_per_video=1, tag="M2")
print(f"M2 total: {len(m2_df)} frames, {m2_df['seq_id'].nunique()} sequences")

print("\nExtracting Test (200 frames)...")
test_df = extract_sequences(test_videos, seq_frames=50, n_seq_per_video=1, tag="TEST")
print(f"Test total: {len(test_df)} frames, {test_df['seq_id'].nunique()} sequences")

# 校验 M2 和测试集不重叠
m2_keys = set(zip(m2_df["video_id"], m2_df["chunk_id"]))
test_keys = set(zip(test_df["video_id"], test_df["chunk_id"]))
assert not (m2_keys & test_keys), "M2 与测试集有重叠！"
print("\nCheck passed: M2 and test set use different videos/chunks.")

m2_out = WS / "m2_elden_ring_500frames.parquet"
test_out = WS / "test_elden_ring_200frames.parquet"
m2_df.to_parquet(m2_out, index=False)
test_df.to_parquet(test_out, index=False)
print(f"\nSaved: {m2_out}")
print(f"Saved: {test_out}")
