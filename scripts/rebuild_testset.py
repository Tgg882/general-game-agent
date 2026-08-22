#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""rebuild_testset.py - 从存活 VOD 池重建测试集 test_elden_ring_200frames.parquet

背景: 原测试集 seq1(v2583251552)/seq2(v2511525153) 被 Twitch 删除, 无法下载帧。
按原 extract_frames_v2.py 相同逻辑 (activity_score 挑 chunk + pick_active_window
挑窗口), 从存活 VOD 池重新抽 4 段 × 50 帧, 生成新测试集。

用法:
  python scripts/rebuild_testset.py --vids v2297468680,v2251390311,v2241243379,v2408179393
  # --vids 顺序即优先级; 原测试集仍存活的视频放最前可优先保留
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PROJECT = Path(__file__).resolve().parent.parent  # 仓库根目录
WS = PROJECT / "workspace"
CHUNK_LIST = WS / "elden_ring_chunks.json"
TEST_OUT = WS / "test_elden_ring_200frames.parquet"
random.seed(42)

BTN_COLS = ["back", "dpad_down", "dpad_left", "dpad_right", "dpad_up", "east", "guide",
            "left_shoulder", "left_thumb", "left_trigger", "north", "right_shoulder",
            "right_thumb", "right_trigger", "south", "start", "west"]


def load_chunks():
    with open(CHUNK_LIST) as f:
        chunks = json.load(f)
    by_video = defaultdict(list)
    for r in chunks:
        by_video[r["video_id"]].append(r)
    return by_video


def activity_score(pq_path):
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
    df = pq.read_table(pq_path).to_pandas()
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


def pick_chunk(by_video, vid, seq_frames=50):
    """在指定视频内挑 1 个活跃 chunk (与原逻辑一致: 随机抽 6 候选取最高分)."""
    cand = by_video.get(vid, [])
    cand = [c for c in cand if c["chunk_size"] >= seq_frames + 400]
    if not cand:
        return None
    random.shuffle(cand)
    sample = cand[:6]
    scored = [(activity_score(Path(c["chunk_dir"]) / "actions_raw.parquet"), c) for c in sample]
    scored.sort(key=lambda x: -x[0])
    return scored[0][1]


def extract_one(by_video, vid, seq_id, seq_frames=50):
    c = pick_chunk(by_video, vid, seq_frames)
    if c is None:
        print(f"  [skip] {vid}: 无满足 chunk_size>={seq_frames+400} 的 chunk")
        return None
    start, score = pick_active_window(Path(c["chunk_dir"]) / "actions_raw.parquet",
                                      win=seq_frames, step=50)
    pq_path = Path(c["chunk_dir"]) / "actions_raw.parquet"
    df = pq.read_table(pq_path).to_pandas()
    seg = df.iloc[start:start + seq_frames].reset_index(drop=True)
    jl = np.array([list(v) for v in seg["j_left"]], dtype=float)
    jr = np.array([list(v) for v in seg["j_right"]], dtype=float)
    rows = []
    for i in range(seq_frames):
        row = {b: int(seg.loc[i, b]) for b in BTN_COLS}
        row.update({
            "j_left_x": jl[i, 0], "j_left_y": jl[i, 1],
            "j_right_x": jr[i, 0], "j_right_y": jr[i, 1],
            "seq_id": seq_id, "frame_idx": i,
            "video_id": vid, "chunk_id": c["chunk_id"],
        })
        rows.append(row)
    print(f"  [TEST] seq {seq_id}: video={vid} chunk={c['chunk_id']} "
          f"frames[{start}:{start+seq_frames}] score={score:.1f}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vids", required=True,
                    help="存活 VOD 池 (逗号分隔, 顺序=优先级, 无需 v 前缀; 原测试集存活者放最前)")
    ap.add_argument("--n", type=int, default=4, help="抽取序列数 (默认 4)")
    args = ap.parse_args()

    vids = [v if v.startswith("v") else "v" + v for v in args.vids.split(",")]
    by_video = load_chunks()

    print(f"候选 VOD: {vids}")
    all_rows, seq_id = [], 0
    for vid in vids:
        if seq_id >= args.n:
            break
        rows = extract_one(by_video, vid, seq_id)
        if rows:
            all_rows.extend(rows)
            seq_id += 1

    if not all_rows:
        print("ERROR: 未能从候选池抽出任何序列")
        return

    test_df = pd.DataFrame(all_rows)
    # 与原测试集结构对齐
    col_order = BTN_COLS + ["j_left_x", "j_left_y", "j_right_x", "j_right_y",
                            "seq_id", "frame_idx", "video_id", "chunk_id"]
    test_df = test_df[col_order]

    # 与 M2 不重叠校验
    m2 = pd.read_parquet(WS / "m2_elden_ring_500frames.parquet")
    m2_keys = set(zip(m2["video_id"], m2["chunk_id"]))
    t_keys = set(zip(test_df["video_id"], test_df["chunk_id"]))
    overlap = m2_keys & t_keys
    if overlap:
        print(f"ERROR: 与 M2 重叠 {overlap}, 请换 VOD")
        return
    print("Check passed: 与 M2 无重叠")

    # 备份旧测试集
    if TEST_OUT.exists():
        bak = TEST_OUT.with_suffix(".parquet.bak")
        TEST_OUT.rename(bak)
        print(f"备份旧测试集 -> {bak.name}")

    test_df.to_parquet(TEST_OUT, index=False)
    print(f"Saved: {TEST_OUT} ({len(test_df)} frames, "
          f"{test_df['seq_id'].nunique()} sequences)")

    # 校验
    n_btns = test_df[BTN_COLS].sum(axis=1)
    print(f"  空按键帧占比: {(n_btns == 0).mean() * 100:.1f}%")
    print(f"  平均每帧按键数: {n_btns.mean():.2f}")
    print(f"  左摇杆活跃(|v|>0.1)占比: "
          f"{((test_df['j_left_x'].abs() > 0.1) | (test_df['j_left_y'].abs() > 0.1)).mean() * 100:.1f}%")
    print(f"  右摇杆活跃(|v|>0.1)占比: "
          f"{((test_df['j_right_x'].abs() > 0.1) | (test_df['j_right_y'].abs() > 0.1)).mean() * 100:.1f}%")
    print("\n各序列:")
    for s, g in test_df.groupby("seq_id"):
        print(f"  seq {s}: {g['video_id'].iloc[0]} chunk_{g['chunk_id'].iloc[0]} "
              f"按键率={(g[BTN_COLS].sum(axis=1) > 0).mean() * 100:.0f}%")


if __name__ == "__main__":
    main()
