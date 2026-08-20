# -*- coding: utf-8 -*-
"""收集 SHARD_0000 中所有 elden_ring 的 chunk 信息，并查看 parquet 结构。"""
import json
import time
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(r"C:\Users\HP\Desktop\general-game-agent\dataset\SHARD_0000")
OUT = Path(r"C:\Users\HP\Desktop\general-game-agent\workspace\elden_ring_chunks.json")

TARGET = "elden_ring"
records = []
t0 = time.time()
for meta_path in ROOT.rglob("metadata.json"):
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("game") == TARGET:
            records.append({
                "chunk_dir": str(meta_path.parent),
                "video_id": meta.get("original_video", {}).get("video_id"),
                "chunk_id": meta.get("chunk_id"),
                "chunk_size": meta.get("chunk_size", 0),
                "controller": meta.get("controller_type"),
            })
    except Exception:
        pass

elapsed = time.time() - t0
print(f"Found {len(records)} elden_ring chunks in {elapsed:.1f}s")

# 按视频分组统计
from collections import defaultdict
by_video = defaultdict(lambda: {"chunks": 0, "frames": 0})
for r in records:
    by_video[r["video_id"]]["chunks"] += 1
    by_video[r["video_id"]]["frames"] += r["chunk_size"]

print(f"Videos: {len(by_video)}")
print()
print(f"{'frames':>10} {'chunks':>6}  video_id")
for vid, s in sorted(by_video.items(), key=lambda kv: -kv[1]["frames"]):
    print(f'{s["frames"]:>10} {s["chunks"]:>6}  {vid}')

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(records, f, indent=1)
print(f"\nSaved chunk list to: {OUT}")

# 查看第一个 elden_ring chunk 的 parquet 结构
if records:
    sample = Path(records[0]["chunk_dir"]) / "actions_raw.parquet"
    print(f"\n=== Sample parquet: {sample} ===")
    tbl = pq.read_table(sample)
    print(f"Rows: {tbl.num_rows}")
    print(f"Columns ({tbl.num_columns}): {tbl.column_names}")
    df = tbl.to_pandas()
    print("\nFirst 3 rows:")
    print(df.head(3).to_string())
    print("\nDtypes:")
    print(df.dtypes.to_string())
