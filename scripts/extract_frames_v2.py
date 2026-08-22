# -*- coding: utf-8 -*-
"""任务2a (重建版)：基于"活跃度"挑选 10 个存活视频中按键+摇杆都活跃的窗口作为 M2 500 帧。

背景: 原 M2 的 10 个 Twitch VOD 中 7 个已被平台删除, 无法下载真实帧。
本重建版从"存活 VOD 池"重新挑选 10 个视频生成 M2 (与 M3 测试集 chunk 不重叠)。
"""
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PROJECT = Path(__file__).resolve().parent.parent  # 仓库根目录
WS = PROJECT / "workspace"
CHUNK_LIST = WS / "elden_ring_chunks.json"
random.seed(42)

BTN_COLS = ["back", "dpad_down", "dpad_left", "dpad_right", "dpad_up", "east", "guide",
            "left_shoulder", "left_thumb", "left_trigger", "north", "right_shoulder",
            "right_thumb", "right_trigger", "south", "start", "west"]

# 已确认被 Twitch/YouTube 删除的 VOD (2026-08-22 探测)
DEAD_VIDS = {
    "v2390341180", "v2469787272", "v2142427275", "v2002470506",
    "v2578903257", "v2465764623", "v2434016041", "v2583251552",
    "v2511525153",  # 8/22 连通性测试确认已删除
}

with open(CHUNK_LIST) as f:
    chunks = json.load(f)

from collections import defaultdict
by_video = defaultdict(list)
for r in chunks:
    by_video[r["video_id"]].append(r)

# 存活池按 chunk 总量排序
videos_sorted = sorted(
    [(v, cs) for v, cs in by_video.items() if v not in DEAD_VIDS],
    key=lambda kv: -sum(c["chunk_size"] for c in kv[1]),
)
# M3 测试集 (已有真实帧) 的 video/chunk, M2 必须 chunk 级不重叠
test_df = pd.read_parquet(WS / "test_elden_ring_200frames.parquet")
TEST_KEYS = set(zip(test_df["video_id"], test_df["chunk_id"]))

print(f"存活 VOD 池: {len(videos_sorted)} 个")
for v, cs in videos_sorted:
    print(f"  {v}: total_chunk={sum(c['chunk_size'] for c in cs)}")

m2_videos = [v for v, _ in videos_sorted[:10]]
print(f"\nM2 重建选用前 10: {m2_videos}")


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


def pick_active_window(pq_path, win=50, step=50):
    """在 chunk 中按 50 帧滑动窗口扫，挑最活跃的 1 个窗口 (从 row 200 开始, 确定性)."""
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


def extract_sequences(video_ids, seq_frames=50):
    rows = []
    seq_id = 0
    for vid in video_ids:
        # 候选 chunk: 排除 M3 测试集已用的 chunk
        cand = [c for c in by_video[vid]
                if c["chunk_size"] >= seq_frames + 400
                and (vid, c["chunk_id"]) not in TEST_KEYS]
        if not cand:
            print(f"  !! {vid} 无可候选 chunk (排除 TEST 后为空), 跳过")
            continue
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
        print(f"  [M2] seq {seq_id}: video={vid} chunk={c['chunk_id']} "
              f"frames[{start}:{start+seq_frames}] score={score:.1f}")
        seq_id += 1
    return pd.DataFrame(rows)


print("\nExtracting M2 (rebuilt, 500 frames, activity-picked, alive VODs only)...")
m2_df = extract_sequences(m2_videos, seq_frames=50)
print(f"M2 total: {len(m2_df)} frames, {m2_df['seq_id'].nunique()} sequences\n")

# 校验: 与 M3 测试集 chunk 级不重叠
m2_keys = set(zip(m2_df["video_id"], m2_df["chunk_id"]))
overlap = m2_keys & TEST_KEYS
assert not overlap, f"M2 与测试集 chunk 重叠: {overlap}"
print(f"Check passed: M2 ({len(m2_keys)} chunks) 与测试集 ({len(TEST_KEYS)} chunks) 不重叠.\n")

m2_out = WS / "m2_elden_ring_500frames.parquet"
m2_df.to_parquet(m2_out, index=False)
print(f"Saved: {m2_out}")

# 简单校验
print("\nM2 帧级校验:")
n_btns = m2_df[BTN_COLS].sum(axis=1)
print(f"  空按键帧占比: {(n_btns==0).mean()*100:.1f}%")
print(f"  平均每帧按键数: {n_btns.mean():.2f}")
print(f"  左摇杆活跃(|v|>0.1)占比: {((m2_df['j_left_x'].abs()>0.1)|(m2_df['j_left_y'].abs()>0.1)).mean()*100:.1f}%")
print(f"  右摇杆活跃(|v|>0.1)占比: {((m2_df['j_right_x'].abs()>0.1)|(m2_df['j_right_y'].abs()>0.1)).mean()*100:.1f}%")
