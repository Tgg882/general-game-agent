# -*- coding: utf-8 -*-
"""任务1：扫描 SHARD_0000 所有 chunk 的 metadata.json，统计每款游戏的帧数/视频数/chunk数。
输出：game_stats.csv（全量）+ 控制台 Top 30 摘要
"""
import json
import time
from collections import defaultdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent  # 仓库根目录
ROOT = PROJECT / "dataset" / "SHARD_0000"
OUT_CSV = PROJECT / "workspace" / "game_stats.csv"

# game -> {frames, chunks, videos:set, controller:set}
stats = defaultdict(lambda: {"frames": 0, "chunks": 0, "videos": set(), "controllers": set()})

t0 = time.time()
n_meta = 0
n_err = 0
for meta_path in ROOT.rglob("metadata.json"):
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        game = meta.get("game", "UNKNOWN")
        s = stats[game]
        s["frames"] += int(meta.get("chunk_size", 0))
        s["chunks"] += 1
        s["videos"].add(meta.get("original_video", {}).get("video_id", "?"))
        s["controllers"].add(meta.get("controller_type", "?"))
        n_meta += 1
    except Exception as e:
        n_err += 1
        if n_err <= 5:
            print(f"[WARN] {meta_path}: {e}")

elapsed = time.time() - t0
print(f"Scanned {n_meta} metadata files in {elapsed:.1f}s, errors={n_err}")
print(f"Distinct games: {len(stats)}")
print()

rows = sorted(stats.items(), key=lambda kv: -kv[1]["frames"])

# 写 CSV
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_CSV, "w", encoding="utf-8-sig") as f:
    f.write("game,frames,chunks,videos,controllers\n")
    for game, s in rows:
        controllers = "|".join(sorted(s["controllers"]))
        f.write(f'"{game}",{s["frames"]},{s["chunks"]},{len(s["videos"])},"{controllers}"\n')
print(f"Full stats saved to: {OUT_CSV}")
print()

print(f"{'frames':>10} {'chunks':>7} {'videos':>6}  game")
print("-" * 70)
for game, s in rows[:30]:
    print(f'{s["frames"]:>10} {s["chunks"]:>7} {len(s["videos"]):>6}  {game}')
