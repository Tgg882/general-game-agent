#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""download_test_frames.py - 下载测试集源视频帧 (Twitch VOD 精确切段 + 60fps 抽帧 + 对齐 parquet)

测试集 200 帧 = 4 序列 × 50 帧。每个序列对应一个 dataset chunk (actions_raw.parquet +
metadata.json)，chunk 的 original_video 指向 Twitch VOD 的绝对时间区间。

流程:
  yt-dlp --download-sections 精确切出 chunk 的 20s 区间 -> 本地 mp4
  -> ffmpeg 60fps 抽帧 -> 按 pick_active_window 窗口取 50 帧 -> 256x256 jpg

要点:
  - chunk 信息从 test parquet + metadata.json 动态读取 (不硬编码)
  - --download-sections 基于媒体清单精确切段, 规避 ffmpeg 对 HLS 输入 seek 的分片误差
  - 帧数不足的序列标记失败跳过, 不静默回退 (避免帧与标注错位)

用法:
  python scripts/download_test_frames.py
  python scripts/download_test_frames.py --cookies edge   # 若 Twitch 要求登录
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PROJECT = Path(__file__).resolve().parent.parent  # 仓库根目录
WS = PROJECT / "workspace"
SHARD = PROJECT / "dataset" / "SHARD_0000"
FRAMES_DIR = WS / "test_frames"
TMP_DIR = WS / "test_frames_tmp"
# 用当前解释器运行 yt_dlp（需在已安装 yt-dlp 的环境中执行）
PYTHON = sys.executable

def _parse_args():
    ap = argparse.ArgumentParser(description="下载数据集源视频帧 (支持 M3 测试集 / M2 500 帧)")
    ap.add_argument("--cookies", default=None, help="浏览器名 (edge/chrome), 仅在 Twitch 要求登录时使用")
    ap.add_argument("--frames", type=int, default=50, help="每序列帧数")
    ap.add_argument("--proxy", default=None, help="代理地址 (默认自动检测系统代理)")
    ap.add_argument("--only", default=None, help="只处理指定 seq (逗号分隔, 如 0,3)")
    ap.add_argument("--input", default="test_elden_ring_200frames.parquet",
                    help="数据 parquet (含 seq_id/video_id/chunk_id 列), 相对 workspace 或绝对路径")
    ap.add_argument("--out-dir", default="test_frames", help="帧输出目录 (workspace 下)")
    ap.add_argument("--tmp-dir", default="test_frames_tmp", help="临时 mp4/raw 目录 (workspace 下)")
    return ap.parse_args()

BTN_COLS = [
    "back", "dpad_down", "dpad_left", "dpad_right", "dpad_up",
    "east", "guide", "left_shoulder", "left_thumb", "left_trigger",
    "north", "right_shoulder", "right_thumb", "right_trigger",
    "south", "start", "west",
]


def pick_active_window(pq_path, win=50, step=50):
    """与 extract_frames_v2.py 完全一致的窗口选择 (确定性, 无随机)."""
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


def find_ffmpeg():
    p = shutil.which("ffmpeg")
    if p:
        return p
    # fallback: winget 安装路径 (新装后 PATH 未刷新的场景)
    base = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"
    if base.exists():
        hits = sorted(base.glob("Gyan.FFmpeg*/**/ffmpeg.exe"))
        if hits:
            return str(hits[0])
    print("ERROR: ffmpeg not found. Install with: winget install Gyan.FFmpeg")
    sys.exit(1)


def get_system_proxy():
    """读 Windows 系统代理 (HKCU Internet Settings), 返回如 http://127.0.0.1:7890 或 None."""
    try:
        import winreg
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                             r"Software\Microsoft\Windows\CurrentVersion\Internet Settings")
        enable, _ = winreg.QueryValueEx(key, "ProxyEnable")
        server, _ = winreg.QueryValueEx(key, "ProxyServer")
        winreg.CloseKey(key)
        if enable and server:
            return server if "://" in server else f"http://{server}"
    except Exception:
        pass
    return None


def run_with_retry(cmd, timeout, max_tries=10, interval=3, cwd=None):
    """运行命令并自动重试 (扛过 Clash 节点偶发抖动). 返回 subprocess.CompletedProcess."""
    last = None
    for attempt in range(1, max_tries + 1):
        if attempt > 1:
            print(f"  重试 {attempt}/{max_tries} ...")
            time.sleep(interval)
        try:
            last = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd)
            if last.returncode == 0:
                return last
        except subprocess.TimeoutExpired:
            last = None
    return last


def main():
    args = _parse_args()

    global FRAMES_DIR, TMP_DIR
    FRAMES_DIR = WS / args.out_dir
    TMP_DIR = WS / args.tmp_dir

    proxy = args.proxy or get_system_proxy()
    if proxy:
        print(f"使用代理: {proxy}")

    ffmpeg = find_ffmpeg()
    ffmpeg_dir = os.path.dirname(ffmpeg)
    print(f"ffmpeg: {ffmpeg}")

    # 读数据 parquet 的 chunk 信息 (动态, 不硬编码)
    input_path = args.input if os.path.isabs(args.input) else WS / args.input
    test_df = pd.read_parquet(input_path)
    chunks_info = test_df[["seq_id", "video_id", "chunk_id"]].drop_duplicates().sort_values("seq_id")
    if args.only:
        only = {int(s) for s in args.only.split(",")}
        chunks_info = chunks_info[chunks_info["seq_id"].isin(only)]
    print(f"数据: {args.input} -> {len(chunks_info)} 序列, {len(test_df)} 帧 (only={args.only})")

    # 检查是否所有 mp4 已就位 (手动下载场景: 跳过连通性测试与下载)
    all_mp4 = True
    for _, row in chunks_info.iterrows():
        mp4 = TMP_DIR / f"seq{int(row['seq_id']):02d}" / "chunk.mp4"
        if not mp4.exists() or mp4.stat().st_size == 0:
            all_mp4 = False
            break
    if all_mp4:
        print("所有序列 mp4 已就位, 跳过连通性测试与下载, 直接抽帧")
    else:
        # 连通性测试 (第一个视频)
        first = chunks_info.iloc[0]
        first_meta = json.load(open(
            SHARD / first["video_id"] / f"{first['video_id']}_chunk_{first['chunk_id']}" / "metadata.json"))
        test_url = first_meta["original_video"]["url"]
        print(f"\n=== 连通性测试: {test_url} ===")
        cmd = [PYTHON, "-m", "yt_dlp", "--no-warnings", "--skip-download", "--print", "%(title)s"]
        if proxy:
            cmd += ["--proxy", proxy]
        cmd.append(test_url)
        r = run_with_retry(cmd, timeout=90)
        if r is None or r.returncode != 0:
            err = r.stderr if r is not None else "timeout"
            print(f"ERROR: Twitch 不可访问: {err[:300]}")
            print("请确认 VPN/代理已开启且可访问 twitch.tv (节点可能抖动, 可稍后重试)")
            sys.exit(1)
        print(f"OK: Twitch 可访问 - {r.stdout.strip()[:80]}")

    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    try:
        from PIL import Image
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "Pillow", "-q"])
        from PIL import Image

    total_ok = 0
    failed = []
    for _, row in chunks_info.iterrows():
        seq_id = int(row["seq_id"])
        vid = row["video_id"]
        cid = row["chunk_id"]
        cdir = SHARD / vid / f"{vid}_chunk_{cid}"
        meta = json.load(open(cdir / "metadata.json"))
        ov = meta["original_video"]
        url = ov["url"]
        src = ov["source"]
        t0, t1 = float(ov["start_time"]), float(ov["end_time"])

        win_start, score = pick_active_window(cdir / "actions_raw.parquet")
        print(f"\n=== seq {seq_id}: {vid} chunk_{cid} ===")
        print(f"  source={src} url={url}")
        print(f"  chunk 绝对时间: {t0:.0f}s - {t1:.0f}s")
        print(f"  窗口起始: row {win_start} (score={score:.1f})")

        seg_dir = TMP_DIR / f"seq{seq_id:02d}"
        seg_dir.mkdir(parents=True, exist_ok=True)
        mp4 = seg_dir / "chunk.mp4"
        raw_pat = str(seg_dir / "raw_%06d.jpg")

        # 1) yt-dlp 精确切段下载 (绝对秒数, 基于媒体清单) — 已存在则跳过
        if mp4.exists() and mp4.stat().st_size > 0:
            print(f"  mp4 已存在, 跳过下载: {mp4.stat().st_size / 1e6:.1f} MB")
        else:
            # YouTube 的 best 格式(DASH 分离) 不兼容 --download-sections, 需显式选 bv*+ba 合并;
            # Twitch 用 best 即可
            if src == "youtube":
                dl_cmd = [PYTHON, "-m", "yt_dlp", "--no-warnings",
                          "-f", "bv*[height<=720]+ba/b[height<=720]", "--merge-output-format", "mp4",
                          "--download-sections", f"*{t0:.0f}-{t1:.0f}", "-o", str(mp4),
                          "--ffmpeg-location", ffmpeg_dir]
            else:
                dl_cmd = [PYTHON, "-m", "yt_dlp", "--no-warnings", "-f", "best",
                          "--download-sections", f"*{t0:.0f}-{t1:.0f}", "-o", str(mp4),
                          "--ffmpeg-location", ffmpeg_dir]
            if proxy:
                dl_cmd += ["--proxy", proxy]
            if args.cookies:
                dl_cmd += ["--cookies-from-browser", args.cookies]
            dl_cmd.append(url)
            print("  yt-dlp 切段下载 ...")
            if mp4.exists():
                mp4.unlink()
            r = run_with_retry(dl_cmd, timeout=600)
            if r is None or r.returncode != 0 or not mp4.exists():
                err = r.stderr if r is not None else "timeout"
                print(f"  ERROR yt-dlp: {err[-300:]}")
                failed.append(seq_id)
                continue
            print(f"  下载完成: {mp4.stat().st_size / 1e6:.1f} MB")

        # 2) ffmpeg 本地精确抽帧 (60fps) — 若 mp4 是全量 VOD (时长远超 chunk 区间), 按 t0 偏移
        ffprobe = os.path.join(ffmpeg_dir, "ffprobe.exe")
        mp4_dur = 0.0
        try:
            p = subprocess.run([ffprobe, "-v", "error", "-show_entries",
                                "format=duration", "-of",
                                "default=noprint_wrappers=1:nokey=1", str(mp4)],
                               capture_output=True, text=True, timeout=30)
            mp4_dur = float(p.stdout.strip())
        except Exception:
            pass
        seg_offset = t0 if mp4_dur > (t1 - t0) * 1.5 else 0.0
        ff_cmd = [ffmpeg, "-y"]
        if seg_offset:
            ff_cmd += ["-ss", f"{seg_offset:.3f}"]
        ff_cmd += ["-i", str(mp4), "-vf", "fps=60", "-q:v", "2", raw_pat]
        r = subprocess.run(ff_cmd, capture_output=True, text=True, timeout=300)
        raw = sorted(seg_dir.glob("raw_*.jpg"))
        note = f" (mp4 全量 {mp4_dur:.0f}s, 按 t0={seg_offset:.0f}s 偏移)" if seg_offset else ""
        print(f"  抽帧: {len(raw)} 张 (60fps){note}")
        if r.returncode != 0 or len(raw) < win_start + args.frames:
            print(f"  ERROR 帧数不足: 需 {win_start + args.frames}, 实得 {len(raw)}")
            if r.stderr:
                print(f"  stderr: {r.stderr[-200:]}")
            failed.append(seq_id)
            continue

        # 3) 取窗口内 50 帧, 缩放 256x256
        for fi in range(args.frames):
            img = Image.open(raw[win_start + fi]).convert("RGB")
            img = img.resize((256, 256), Image.BILINEAR)
            img.save(FRAMES_DIR / f"seq{seq_id:02d}_frame{fi:04d}.jpg", quality=95)
        total_ok += args.frames
        print(f"  OK: 保存 {args.frames} 帧 -> test_frames/seq{seq_id:02d}_frame*.jpg")

        # 清理临时帧 (保留 chunk.mp4 以便重跑抽帧)
        for f in seg_dir.glob("raw_*.jpg"):
            f.unlink()

    print(f"\n{'='*50}")
    print(f"成功保存: {total_ok} 帧; 失败序列: {failed if failed else '无'}")
    expected = len(chunks_info) * args.frames
    if total_ok == expected and not failed:
        print(f"ALL OK: {total_ok} 帧就绪, 可运行 m3_eval_v2.py")
    else:
        print(f"WARN: 预期 {expected} 帧 ({len(chunks_info)} 序列), 实际 {total_ok} 帧")
        sys.exit(2)


if __name__ == "__main__":
    main()
