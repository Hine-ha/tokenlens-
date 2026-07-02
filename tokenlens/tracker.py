import json
import os
import time
import requests
from datetime import datetime, timezone
from functools import wraps

ENDPOINT = "https://my-tokenlens.vercel.app/api/track"

PRICING = {
    # Anthropic
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0, "cache_read": 0.3, "cache_write": 3.75},
    "claude-haiku-4-5": {"input": 0.8, "output": 4.0, "cache_read": 0.08, "cache_write": 1.0},
    # OpenAI
    "gpt-4.1": {"input": 2.0, "output": 8.0},
    "gpt-4.1-mini": {"input": 0.4, "output": 1.6},
    "gpt-4.1-nano": {"input": 0.1, "output": 0.4},
    "gpt-4o": {"input": 2.5, "output": 10.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.6},
    # Other
    "deepseek-chat": {"input": 0.27, "output": 1.1},
}


def _calc_cost(
    model,
    input_tokens,
    output_tokens,
    cache_read_tokens=0,
    cache_write_tokens=0,
):
    p = PRICING.get(model, {"input": 0, "output": 0})
    cost = (
        input_tokens * p.get("input", 0)
        + output_tokens * p.get("output", 0)
        + cache_read_tokens * p.get("cache_read", 0)
        + cache_write_tokens * p.get("cache_write", 0)
    ) / 1_000_000
    return round(cost, 8)


def _resolve_user_id(explicit: str | None) -> str | None:
    if explicit and explicit.strip():
        return explicit.strip()
    env = os.environ.get("TOKENLENS_USER_ID")
    return env.strip() if env and env.strip() else None


def _send(payload):
    try:
        api_key = os.environ.get("TOKENLENS_API_KEY", "").strip()
        if not api_key:
            return
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        requests.post(
            ENDPOINT,
            data=body,
            headers={
                "x-api-key": api_key,
                "Content-Type": "application/json; charset=utf-8",
            },
            timeout=3,
        )
    except Exception:
        pass


def _detect_provider(client) -> str:
    if hasattr(client, "messages") and hasattr(client.messages, "create"):
        return "anthropic"
    if (
        hasattr(client, "chat")
        and hasattr(client.chat, "completions")
        and hasattr(client.chat.completions, "create")
    ):
        return "openai"
    raise TypeError(
        "Unsupported client. Pass anthropic.Anthropic() or openai.OpenAI()."
    )


def _extract_usage(response, provider: str):
    if not response or not hasattr(response, "usage"):
        return 0, 0, 0, 0

    usage = response.usage
    if provider == "anthropic":
        return (
            getattr(usage, "input_tokens", 0) or 0,
            getattr(usage, "output_tokens", 0) or 0,
            getattr(usage, "cache_read_input_tokens", 0) or 0,
            getattr(usage, "cache_creation_input_tokens", 0) or 0,
        )

    return (
        getattr(usage, "prompt_tokens", 0) or 0,
        getattr(usage, "completion_tokens", 0) or 0,
        0,
        0,
    )


def _wrap_create(original_create, *, provider, project, use_case, resolved_user_id):
    @wraps(original_create)
    def tracked_create(*args, **kwargs):
        start = time.time()
        success = True
        error_type = None
        response = None

        try:
            response = original_create(*args, **kwargs)
        except Exception as e:
            success = False
            error_type = type(e).__name__
            raise
        finally:
            latency_ms = int((time.time() - start) * 1000)
            model = kwargs.get("model", "unknown")
            input_tokens, output_tokens, cache_read_tokens, cache_write_tokens = (
                _extract_usage(response, provider)
            )
            cost = _calc_cost(
                model,
                input_tokens,
                output_tokens,
                cache_read_tokens,
                cache_write_tokens,
            )

            _send(
                {
                    "project_name": project,
                    "model": model,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cache_read_tokens": cache_read_tokens,
                    "cache_write_tokens": cache_write_tokens,
                    "cost": cost,
                    "success": success,
                    "error_type": error_type,
                    "latency_ms": latency_ms,
                    "use_case": use_case,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    **({"user_id": resolved_user_id} if resolved_user_id else {}),
                }
            )

        return response

    return tracked_create


def track(client, project: str, use_case: str = None, user_id: str = None):
    """
    用法（Anthropic）：
        import anthropic
        from tokenlens import track

        client = track(anthropic.Anthropic(), project="my-app")
        client.messages.create(...)

    用法（OpenAI）：
        import openai
        from tokenlens import track

        client = track(openai.OpenAI(), project="my-app")
        client.chat.completions.create(...)

    环境变量（推荐）：
        TOKENLENS_API_KEY  — ingest 密钥，未设置则不上报
        TOKENLENS_USER_ID  — Dashboard 复制的 user_id
    """
    resolved_user_id = _resolve_user_id(user_id)
    provider = _detect_provider(client)

    if provider == "anthropic":
        client.messages.create = _wrap_create(
            client.messages.create,
            provider=provider,
            project=project,
            use_case=use_case,
            resolved_user_id=resolved_user_id,
        )
    else:
        client.chat.completions.create = _wrap_create(
            client.chat.completions.create,
            provider=provider,
            project=project,
            use_case=use_case,
            resolved_user_id=resolved_user_id,
        )

    return client
