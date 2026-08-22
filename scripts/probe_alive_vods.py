#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""probe_alive_vods.py - 批量探测 Elden Ring 候选 VOD 存活状态

测试集视频 (seq1/seq2) 被 Twitch 删除后, 从后备池重建测试集前,
先用本脚本确认哪些 VOD 仍然可访问。

通道: Twitch GraphQL (快) + yt-dlp (准), 任一通过即判存活。
用法:
  python scripts/probe_alive_vods.py                 # 探测全部候选
  python scripts/probe_alive_vods.py --proxy http://127.0.0.1:7890
  python scripts/probe_alive_vods.py --vids 2232417875,2251390311
"""
import argparse
import subprocess
import sys
import time

import requests

PY = sys.executable
CLIENT_ID = "kimne78kx3ncx6brgo4mv6wki5h1ko"

# 候选池: 测试集可用的非 M2 视频 (Elden Ring, twitch)
# TEST 原序列: v2220952459(v2220952459), v2122506189
# 后备池 (原排序 14-19): v2162110313, v2297468680, v2232417875,
#                        v2251390311, v2241243379, v2408179393
DEFAULT_VIDS = [
    "2220952459",   # TEST seq0 (曾成功, 复查)
    "2122506189",   # TEST seq3
    "2162110313",   # 后备
    "2297468680",   # 后备
    "2232417875",   # 后备 (已确认存活, 快速复核)
    "2251390311",   # 后备
    "2241243379",   # 后备
    "2408179393",   # 后备
]


def probe_graphql(vid, proxies, tries=3):
    """GraphQL 查询 video 是否存在."""
    q = {"query": f'query {{ video(id: "{vid}") {{ id title lengthSeconds }} }}'}
    last = None
    for _ in range(tries):
        try:
            r = requests.post(
                "https://gql.twitch.tv/gql", json=q, timeout=20,
                headers={"Client-ID": CLIENT_ID, "Content-Type": "application/json"},
                proxies=proxies)
            if r.status_code == 200:
                v = r.json().get("data", {}).get("video")
                if v:
                    return True, f"title={v['title'][:45]!r} len={v['lengthSeconds']}s"
                if r.json().get("errors"):
                    return False, "Video does not exist (GraphQL errors)"
                return False, "Video does not exist (data.video=null)"
        except Exception as e:
            last = f"EXC {str(e)[:70]}"
        time.sleep(2)
    return None, last


def probe_ytdlp(vid, proxy):
    """yt-dlp 探测 (慢但可信)."""
    cmd = [PY, "-m", "yt_dlp", "--no-warnings", "--skip-download",
           "--print", "%(title)s", "--socket-timeout", "30",
           f"https://www.twitch.tv/videos/{vid}"]
    if proxy:
        cmd += ["--proxy", proxy]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
        if r.returncode == 0 and r.stdout.strip():
            return True, r.stdout.strip()[:45]
        lines = [l for l in r.stderr.strip().splitlines() if l.strip()]
        err = lines[-1] if lines else f"rc={r.returncode}"
        if "does not exist" in err:
            return False, "Video does not exist"
        return None, err[:80]
    except Exception as e:
        return None, f"EXC {str(e)[:70]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proxy", default=None,
                    help="代理地址, 默认自动读系统代理")
    ap.add_argument("--vids", default=None,
                    help="逗号分隔的 VOD id (不带 v 前缀), 默认候选池")
    args = ap.parse_args()

    proxy = args.proxy
    if not proxy:
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                                 r"Software\Microsoft\Windows\CurrentVersion\Internet Settings")
            enable, _ = winreg.QueryValueEx(key, "ProxyEnable")
            server, _ = winreg.QueryValueEx(key, "ProxyServer")
            winreg.CloseKey(key)
            if enable and server:
                proxy = server if "://" in server else f"http://{server}"
        except Exception:
            pass
    proxies = {"http": proxy, "https": proxy} if proxy else None
    print(f"proxy: {proxy or '无'}")

    vids = args.vids.split(",") if args.vids else DEFAULT_VIDS
    print(f"探测 {len(vids)} 个 VOD ...\n")
    results = []
    for vid in vids:
        gql = probe_graphql(vid, proxies)
        ytd = probe_ytdlp(vid, proxy)
        if gql[0] is True or ytd[0] is True:
            verdict, info = "ALIVE", gql[1] if gql[0] else ytd[1]
        elif gql[0] is False or ytd[0] is False:
            verdict, info = "DEAD ", gql[1] if gql[0] is False else ytd[1]
        else:
            verdict, info = "UNKNOWN", f"GQL:{(gql[1] or '')[:40]} | yt-dlp:{(ytd[1] or '')[:40]}"
        results.append((verdict, vid, info))
        print(f"{verdict}  {vid}  {info}")
        time.sleep(2)

    print("\n===== 汇总 =====")
    alive = [v for v, _, _ in results if v == "ALIVE"]
    print(f"存活 {len(alive)}/{len(results)}: {alive}")
    if alive:
        print("\n可重建测试集 (从存活池选 4 段):")
        for v in alive:
            print(f"  https://www.twitch.tv/videos/{v}")


if __name__ == "__main__":
    main()
