#!/usr/bin/env python3
"""
Verify TokenLens ingest + Supabase cache fields after a cache test.

Usage:
  export TOKENLENS_API_KEY="tokenlens-secret-2026"
  export TOKENLENS_USER_ID="user_..."   # optional but recommended for Dashboard
  export SUPABASE_URL="https://....supabase.co"
  export SUPABASE_SECRET_KEY="sb_secret_..."

  python3 examples/check_ingest.py
  python3 examples/check_ingest.py --project cache-test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone

TRACK_ENDPOINT = "https://my-tokenlens.vercel.app/api/track"
SMART_QUOTES = "\"\"''\u201c\u201d\u2018\u2019"
EXPECTED_SAVED = 4167 * (0.8 * 0.9 / 1_000_000)  # Haiku 4.5, 4167 cache_read


def clean(name: str) -> str:
    return os.environ.get(name, "").strip().strip(SMART_QUOTES)


def load_dotenv_local() -> None:
    path = os.path.expanduser("~/Projects/tokenlens/.env.local")
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if key == "NEXT_PUBLIC_SUPABASE_URL" and not clean("SUPABASE_URL"):
                os.environ["SUPABASE_URL"] = value
            elif key == "SUPABASE_SECRET_KEY" and not clean("SUPABASE_SECRET_KEY"):
                os.environ["SUPABASE_SECRET_KEY"] = value
            elif key == "TOKENLENS_API_KEY" and not clean("TOKENLENS_API_KEY"):
                os.environ["TOKENLENS_API_KEY"] = value


def http_json(method: str, url: str, headers: dict, body: dict | None = None):
    data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=20) as resp:
        raw = resp.read().decode()
        return resp.status, json.loads(raw) if raw else {}


def check_track_api() -> bool:
    api_key = clean("TOKENLENS_API_KEY")
    if not api_key:
        print("❌ TOKENLENS_API_KEY 未设置 — SDK 测试不会上报到 TokenLens（静默跳过）")
        return False

    payload = {
        "project_name": "cache-check-probe",
        "model": "claude-haiku-4-5-20251001",
        "input_tokens": 3,
        "output_tokens": 20,
        "cache_read_tokens": 4167,
        "cache_write_tokens": 0,
        "cost": 0.0001,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_id": clean("TOKENLENS_USER_ID") or None,
    }
    if not payload["user_id"]:
        del payload["user_id"]

    try:
        status, data = http_json(
            "POST",
            TRACK_ENDPOINT,
            {
                "Content-Type": "application/json; charset=utf-8",
                "x-api-key": api_key,
            },
            payload,
        )
    except urllib.error.URLError as e:
        print(f"❌ 无法连接 {TRACK_ENDPOINT}: {e.reason}")
        return False
    except urllib.error.HTTPError as e:
        print(f"❌ /api/track 失败 HTTP {e.code}: {e.read().decode()}")
        return False

    event = data.get("event") or {}
    saved = event.get("cache_saved")
    read = event.get("cache_read_tokens")
    ok = status == 201 and read == 4167 and saved and abs(float(saved) - EXPECTED_SAVED) < 1e-6
    print(f"{'✅' if ok else '⚠️'} /api/track → status={status}, cache_read_tokens={read}, cache_saved={saved}")
    if not ok:
        print(f"   期望 cache_saved ≈ {EXPECTED_SAVED:.6f}")
    return ok


def check_supabase(project: str) -> bool:
    base = clean("SUPABASE_URL")
    key = clean("SUPABASE_SECRET_KEY")
    if not base or not key:
        print("⚠️ 未设置 SUPABASE_URL / SUPABASE_SECRET_KEY，跳过数据库检查")
        print("   可在 ~/Projects/tokenlens/.env.local 中读取，或手动 export")
        return False

    query = (
        f"{base}/rest/v1/usage_events"
        f"?select=created_at,project_name,model,cache_read_tokens,cache_write_tokens,cache_saved,user_id"
        f"&project_name=eq.{project}"
        f"&order=created_at.desc&limit=5"
    )
    try:
        status, rows = http_json(
            "GET",
            query,
            {"apikey": key, "Authorization": f"Bearer {key}"},
        )
    except urllib.error.URLError as e:
        print(f"❌ 无法连接 Supabase: {e.reason}")
        return False
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        if "cache_read_tokens" in body:
            print("❌ Supabase 缺少 cache_read_tokens 列 — 请在 SQL Editor 运行 migration 004")
        else:
            print(f"❌ Supabase 查询失败 HTTP {e.code}: {body}")
        return False

    if status != 200:
        print(f"❌ Supabase 意外状态 {status}")
        return False

    if not rows:
        print(f"❌ 数据库里没有 project={project!r} 的记录")
        print("   若刚跑过 test_cache.py，请先 export TOKENLENS_API_KEY 再重跑")
        return False

    print(f"✅ 最近 {len(rows)} 条 {project!r} 记录：")
    hit = False
    for row in rows:
        read = row.get("cache_read_tokens") or 0
        saved = float(row.get("cache_saved") or 0)
        print(
            f"   {row.get('created_at')} read={read} write={row.get('cache_write_tokens')} "
            f"saved={saved:.6f} user={row.get('user_id') or '(null)'}"
        )
        if read > 0:
            hit = True

    if not hit:
        print("⚠️ 有上报记录，但没有 cache_read_tokens > 0 的行")
        print("   常见原因：脚本退出太快，第二次上报线程被杀死 — 请升级 SDK v0.1.7+ 并重跑 test_cache.py")
    return hit


def main() -> None:
    load_dotenv_local()
    parser = argparse.ArgumentParser(description="Check TokenLens cache ingest pipeline")
    parser.add_argument("--project", default="cache-test")
    args = parser.parse_args()

    print("=== TokenLens 缓存上报检查 ===\n")
    print(f"TOKENLENS_API_KEY: {'已设置' if clean('TOKENLENS_API_KEY') else '未设置'}")
    print(f"TOKENLENS_USER_ID: {clean('TOKENLENS_USER_ID') or '(未设置，Dashboard 仍可见 user_id IS NULL 数据)'}\n")

    api_ok = check_track_api()
    print()
    db_ok = check_supabase(args.project)

    print()
    if api_ok and db_ok:
        print("结论：生产 /api/track 与 Supabase 缓存字段正常。刷新 Dashboard 应能看到 cache_savings。")
        return

    if not clean("TOKENLENS_API_KEY"):
        print("结论：请 export TOKENLENS_API_KEY 后重跑 test_cache.py，再执行本脚本。")
    elif not db_ok:
        print("结论：Anthropic 缓存已通，但 TokenLens 侧可能未入库或 migration 未跑。")
    sys.exit(1)


if __name__ == "__main__":
    main()
