import json
from pathlib import Path
from collections import defaultdict

PROJECT = Path(__file__).resolve().parent.parent  # 仓库根目录
root = PROJECT / "dataset" / "SHARD_0000"
src = defaultdict(lambda: {"chunks": 0, "videos": set()})
for p in root.rglob("metadata.json"):
    m = json.load(open(p))
    if m.get("game") != "elden_ring":
        continue
    s = m.get("original_video", {}).get("source", "?")
    src[s]["chunks"] += 1
    src[s]["videos"].add(m.get("original_video", {}).get("video_id"))

for s, d in sorted(src.items(), key=lambda x: -x[1]["chunks"]):
    n_chunks = d["chunks"]
    n_videos = len(d["videos"])
    print(f"{s}: {n_chunks} chunks, {n_videos} videos")
    if s == "youtube":
        for v in sorted(d["videos"]):
            print(f"  youtube video: {v}")
