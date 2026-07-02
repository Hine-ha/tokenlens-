#!/usr/bin/env python3
"""
TokenLens SDK end-to-end test.

Usage:
  export ANTHROPIC_API_KEY="sk-ant-..."
  export TOKENLENS_API_KEY="tokenlens-secret-2026"
  export TOKENLENS_USER_ID="user_..."

  python3 examples/test_track.py
  python3 examples/test_track.py --report-only   # skip Anthropic, test ingest only
  python3 examples/test_track.py --verbose
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime, timezone

REQUIRED = (
    ("ANTHROPIC_API_KEY", "Anthropic API key"),
    ("TOKENLENS_API_KEY", "TokenLens ingest key (x-api-key)"),
    ("TOKENLENS_USER_ID", "Clerk user_id from Dashboard"),
)

TRACK_ENDPOINT = "https://my-tokenlens.vercel.app/api/track"
DASHBOARD_URL = "https://my-tokenlens.vercel.app/dashboard"

SMART_QUOTES = "\"\"''\u201c\u201d\u2018\u2019"


def clean_env(name: str) -> str:
    raw = os.environ.get(name, "")
    value = raw.strip().strip(SMART_QUOTES)
    return value


def require_ascii(name: str, value: str) -> None:
    try:
        value.encode("ascii")
    except UnicodeEncodeError as e:
        print(
            f"{name} contains non-ASCII characters near index {e.start}. "
            f"Re-copy the value without Chinese punctuation or smart quotes.",
            file=sys.stderr,
        )
        sys.exit(1)


def check_env(*, need_anthropic: bool) -> None:
    keys = REQUIRED if need_anthropic else REQUIRED[1:]
    missing = [(key, label) for key, label in keys if not clean_env(key)]
    if missing:
        print("Missing environment variables:", file=sys.stderr)
        for key, label in missing:
            print(f'  export {key}="..."  # {label}', file=sys.stderr)
        sys.exit(1)

    for key, _ in keys:
        require_ascii(key, clean_env(key))


def report_only(project: str, user_id: str) -> None:
    import requests

    payload = {
        "project_name": project,
        "model": "claude-haiku-4-5-20251001",
        "input_tokens": 11,
        "output_tokens": 34,
        "cost": 0.000018,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_id": user_id,
    }
    api_key = clean_env("TOKENLENS_API_KEY")
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    response = requests.post(
        TRACK_ENDPOINT,
        data=body,
        headers={
            "x-api-key": api_key,
            "Content-Type": "application/json; charset=utf-8",
        },
        timeout=10,
    )
    print("Report-only status:", response.status_code)
    print("Response:", response.text)
    if response.status_code == 201:
        print(f"\nOpen Dashboard: {DASHBOARD_URL}")
    else:
        sys.exit(1)


def run_anthropic(project: str, user_id: str, model: str, verbose: bool) -> None:
    try:
        import anthropic
        from tokenlens import track
    except ImportError as e:
        print(f"Import failed: {e}", file=sys.stderr)
        print("Run: pip3 install -e . anthropic", file=sys.stderr)
        sys.exit(1)

    api_key = clean_env("ANTHROPIC_API_KEY")
    client = track(
        anthropic.Anthropic(api_key=api_key),
        project=project,
        user_id=user_id,
    )

    try:
        response = client.messages.create(
            model=model,
            max_tokens=32,
            messages=[{"role": "user", "content": "Reply with exactly: TokenLens OK"}],
        )
    except Exception as e:
        print(f"Anthropic call failed: {e}", file=sys.stderr)
        if verbose:
            traceback.print_exc()
        print(
            "\nTip: run `python3 examples/test_track.py --report-only` "
            "to verify TokenLens ingest without Anthropic.",
            file=sys.stderr,
        )
        sys.exit(1)

    text = ""
    for block in response.content:
        if hasattr(block, "text"):
            text += block.text

    usage = response.usage
    print("Anthropic reply:", text.strip() or "(empty)")
    print(f"Tokens: input={usage.input_tokens}, output={usage.output_tokens}")
    print("\nTrack event sent in background.")
    print(f"Open Dashboard: {DASHBOARD_URL}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Test TokenLens track() integration")
    parser.add_argument("--project", default="tokenlens-sdk-test")
    parser.add_argument("--model", default="claude-haiku-4-5-20251001")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Skip Anthropic; POST a sample event to /api/track",
    )
    parser.add_argument("--verbose", action="store_true", help="Print full traceback")
    args = parser.parse_args()

    check_env(need_anthropic=not args.report_only)
    user_id = clean_env("TOKENLENS_USER_ID")

    print(f"project: {args.project}")
    print(f"user_id: {user_id}")
    if args.report_only:
        print("mode: report-only\n")
        report_only(args.project, user_id)
        return

    print(f"model: {args.model}")
    print("mode: anthropic + track\n")
    run_anthropic(args.project, user_id, args.model, args.verbose)


if __name__ == "__main__":
    main()
