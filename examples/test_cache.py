#!/usr/bin/env python3
"""
Anthropic prompt cache test for TokenLens SDK.

Haiku 4.5 needs ~4096+ tokens in the cached prefix (use * 320+ for Chinese text).
Sonnet needs ~1024+ tokens (use * 120+).

Usage:
  pip3 install --force-reinstall "git+https://github.com/Hine-ha/tokenlens-.git#subdirectory=tokenlens"
  export ANTHROPIC_API_KEY="sk-ant-api03-..."
  python3 examples/test_cache.py
  python3 examples/test_cache.py --model claude-sonnet-4-6
"""

from __future__ import annotations

import argparse
import os
import sys

SMART_QUOTES = "\"\"''\u201c\u201d\u2018\u2019"


def clean_env(name: str) -> str:
    return os.environ.get(name, "").strip().strip(SMART_QUOTES)


def require_ascii(name: str, value: str) -> None:
    try:
        value.encode("ascii")
    except UnicodeEncodeError as e:
        print(
            f"{name} contains non-ASCII near index {e.start}. "
            f"Use a real sk-ant-api03-... key without Chinese placeholders.",
            file=sys.stderr,
        )
        sys.exit(1)


def min_repeat(model: str) -> int:
    name = model.lower()
    if "haiku-4-5" in name or "haiku-4.5" in name or "opus" in name:
        return 320
    return 120


def run(model: str, repeat: int) -> None:
    import anthropic
    from tokenlens import track

    api_key = clean_env("ANTHROPIC_API_KEY")
    require_ascii("ANTHROPIC_API_KEY", api_key)

    client = track(
        anthropic.Anthropic(api_key=api_key),
        project="cache-test",
        auto_cache=True,
    )

    chunk = "你是一个医学文献分析专家。"
    long_system = chunk * repeat
    print(f"model: {model}")
    print(f"system chars: {len(long_system)} (repeat={repeat})")
    print()

    def call(label: str):
        print(f"=== {label} ===")
        response = client.messages.create(
            model=model,
            max_tokens=50,
            system=long_system,
            messages=[{"role": "user", "content": "你好"}],
        )
        usage = response.usage
        creation = getattr(usage, "cache_creation_input_tokens", 0) or 0
        read = getattr(usage, "cache_read_input_tokens", 0) or 0
        print("input_tokens:", usage.input_tokens)
        print("cache_creation_input_tokens:", creation)
        print("cache_read_input_tokens:", read)
        print()
        return creation, read

    c1, r1 = call("第一次（写入缓存）")
    c2, r2 = call("第二次（命中缓存）")

    if c1 == 0 and r1 == 0:
        print(
            "第一次没有 cache_creation：system 可能仍不够长。"
            f" Haiku 4.5 请试 --repeat {min_repeat(model)} 或更大。",
            file=sys.stderr,
        )
        sys.exit(1)
    if r2 == 0:
        print(
            "第二次 cache_read 仍为 0：请确认已安装最新 SDK (v0.1.6+) 且两次 system 完全一致。",
            file=sys.stderr,
        )
        sys.exit(1)

    print("缓存测试通过：第二次已命中 cache_read。")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Anthropic prompt caching via TokenLens")
    parser.add_argument("--model", default="claude-haiku-4-5-20251001")
    parser.add_argument(
        "--repeat",
        type=int,
        default=None,
        help="Repeat count for system prompt chunk (default: model minimum)",
    )
    args = parser.parse_args()

    if not clean_env("ANTHROPIC_API_KEY"):
        print('export ANTHROPIC_API_KEY="sk-ant-api03-..."', file=sys.stderr)
        sys.exit(1)

    repeat = args.repeat if args.repeat is not None else min_repeat(args.model)
    run(args.model, repeat)


if __name__ == "__main__":
    main()
