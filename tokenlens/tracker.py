import os
import time
import requests
from datetime import datetime, timezone
from functools import wraps

ENDPOINT = "https://my-tokenlens.vercel.app/api/track"

PRICING = {
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0, "cache_read": 0.3, "cache_write": 3.75},
    "claude-haiku-4-5":  {"input": 0.8, "output": 4.0,  "cache_read": 0.08, "cache_write": 1.0},
    "gpt-4.1":           {"input": 2.0, "output": 8.0},
    "deepseek-chat":     {"input": 0.27, "output": 1.1},
}

def _calc_cost(model, input_tokens, output_tokens,
               cache_read_tokens=0, cache_write_tokens=0):
    p = PRICING.get(model, {"input": 0, "output": 0})
    cost = (
        input_tokens       * p.get("input", 0) +
        output_tokens      * p.get("output", 0) +
        cache_read_tokens  * p.get("cache_read", 0) +
        cache_write_tokens * p.get("cache_write", 0)
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
        requests.post(
            ENDPOINT,
            json=payload,
            headers={"x-api-key": api_key},
            timeout=3,
        )
    except Exception:
        pass  # 不影响主业务


def track(client, project: str, use_case: str = None, user_id: str = None):
    """
    用法：
        import anthropic
        from tokenlens import track

        # 环境变量（推荐）：
        #   TOKENLENS_API_KEY   — 服务端 ingest 密钥，未设置则不上报
        #   TOKENLENS_USER_ID   — Dashboard 复制的 user_id，可替代 user_id 参数

        client = track(anthropic.Anthropic(), project="trial-reviewer")
        # 之后正常用 client，数据自动上报
    """
    resolved_user_id = _resolve_user_id(user_id)
    original_create = client.messages.create

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

            if response and hasattr(response, "usage"):
                u = response.usage
                input_tokens       = getattr(u, "input_tokens", 0)
                output_tokens      = getattr(u, "output_tokens", 0)
                cache_read_tokens  = getattr(u, "cache_read_input_tokens", 0)
                cache_write_tokens = getattr(u, "cache_creation_input_tokens", 0)
            else:
                input_tokens = output_tokens = cache_read_tokens = cache_write_tokens = 0

            cost = _calc_cost(model, input_tokens, output_tokens,
                              cache_read_tokens, cache_write_tokens)

            _send({
                "project_name":        project,
                "model":               model,
                "input_tokens":        input_tokens,
                "output_tokens":       output_tokens,
                "cache_read_tokens":   cache_read_tokens,
                "cache_write_tokens":  cache_write_tokens,
                "cost":                cost,
                "success":             success,
                "error_type":          error_type,
                "latency_ms":          latency_ms,
                "use_case":            use_case,
                "timestamp":           datetime.now(timezone.utc).isoformat(),
                **({"user_id": resolved_user_id} if resolved_user_id else {}),
            })

        return response

    client.messages.create = tracked_create
    return client
