import json
import os
import threading
import requests
from datetime import datetime, timezone

ENDPOINT = "https://my-tokenlens.vercel.app/api/track"

AUTO_CACHE_MIN_SYSTEM_LEN = 1024
CACHE_CONTROL = {"type": "ephemeral"}

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


def _calc_cost(model, input_tokens, output_tokens):
    p = PRICING.get(model, {"input": 0, "output": 0})
    cost = (
        input_tokens * p.get("input", 0) + output_tokens * p.get("output", 0)
    ) / 1_000_000
    return round(cost, 8)


def _resolve_user_id(explicit: str | None) -> str | None:
    if explicit and explicit.strip():
        return explicit.strip()
    env = os.environ.get("TOKENLENS_USER_ID")
    return env.strip() if env and env.strip() else None


def _send_sync(payload: dict) -> None:
    """POST usage to TokenLens only. Never touches Anthropic/OpenAI clients."""
    try:
        api_key = os.environ.get("TOKENLENS_API_KEY", "").strip()
        if not api_key:
            return
        api_key.encode("ascii")

        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        requests.post(
            ENDPOINT,
            data=body,
            headers={
                "x-api-key": api_key,
                "Content-Type": "application/json",
            },
            timeout=3,
        )
    except Exception:
        pass


def _send(payload: dict) -> None:
    threading.Thread(target=_send_sync, args=(payload,), daemon=True).start()


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


def _block_text_length(block) -> int:
    if isinstance(block, dict):
        if block.get("type") == "text":
            return len(block.get("text") or "")
        return len(str(block.get("text", "")))
    text = getattr(block, "text", None)
    if text is not None:
        return len(text)
    return 0


def _block_to_dict_with_cache(block) -> dict:
    if isinstance(block, dict):
        updated = dict(block)
    else:
        updated = {
            "type": getattr(block, "type", "text"),
            "text": getattr(block, "text", ""),
        }
    updated["cache_control"] = CACHE_CONTROL
    return updated


def _apply_auto_cache(kwargs: dict) -> dict:
    """Inject Anthropic prompt cache_control into system when enabled."""
    if "system" not in kwargs:
        return kwargs

    system = kwargs["system"]

    if isinstance(system, str):
        if len(system) <= AUTO_CACHE_MIN_SYSTEM_LEN:
            return kwargs
        updated = dict(kwargs)
        updated["system"] = [
            {
                "type": "text",
                "text": system,
                "cache_control": CACHE_CONTROL,
            }
        ]
        return updated

    if isinstance(system, list):
        best_idx = None
        best_len = -1
        for i, block in enumerate(system):
            length = _block_text_length(block)
            if length > best_len:
                best_len = length
                best_idx = i

        if best_idx is None:
            return kwargs

        block = system[best_idx]
        if isinstance(block, dict) and block.get("cache_control"):
            return kwargs

        updated = dict(kwargs)
        new_system = list(system)
        new_system[best_idx] = _block_to_dict_with_cache(block)
        updated["system"] = new_system
        return updated

    return kwargs


def _extract_usage(response, provider: str):
    if not response or not hasattr(response, "usage"):
        return 0, 0

    usage = response.usage
    if provider == "anthropic":
        return (
            getattr(usage, "input_tokens", 0) or 0,
            getattr(usage, "output_tokens", 0) or 0,
        )

    return (
        getattr(usage, "prompt_tokens", 0) or 0,
        getattr(usage, "completion_tokens", 0) or 0,
    )


def _build_track_payload(
    *,
    project: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cost: float,
    resolved_user_id: str | None,
) -> dict:
    """Fields for TokenLens /api/track only (not sent to LLM APIs)."""
    payload = {
        "project_name": project,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost": cost,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if resolved_user_id:
        payload["user_id"] = resolved_user_id
    return payload


def _wrap_create(
    original_create,
    *,
    provider,
    project,
    resolved_user_id,
    auto_cache=False,
):
    def tracked_create(*args, **kwargs):
        response = None
        call_kwargs = kwargs
        if auto_cache and provider == "anthropic":
            call_kwargs = _apply_auto_cache(kwargs)

        try:
            response = original_create(*args, **call_kwargs)
            return response
        finally:
            model = call_kwargs.get("model", kwargs.get("model", "unknown"))
            input_tokens, output_tokens = _extract_usage(response, provider)
            cost = _calc_cost(model, input_tokens, output_tokens)
            _send(
                _build_track_payload(
                    project=project,
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost=cost,
                    resolved_user_id=resolved_user_id,
                )
            )

    tracked_create.__name__ = getattr(original_create, "__name__", "create")
    tracked_create.__doc__ = getattr(original_create, "__doc__", None)
    return tracked_create


def track(
    client,
    project: str,
    use_case: str = None,
    user_id: str = None,
    auto_cache: bool = False,
):
    """
    Wrap Anthropic or OpenAI client for automatic usage tracking.

    Anthropic:
        client = track(anthropic.Anthropic(), project="my-app", auto_cache=True)
        client.messages.create(...)

    OpenAI:
        client = track(openai.OpenAI(), project="my-app")
        client.chat.completions.create(...)

    auto_cache: Anthropic only — inject cache_control on long system prompts.

    Env: TOKENLENS_API_KEY, TOKENLENS_USER_ID (optional)
    """
    resolved_user_id = _resolve_user_id(user_id)
    provider = _detect_provider(client)

    if provider == "anthropic":
        client.messages.create = _wrap_create(
            client.messages.create,
            provider=provider,
            project=project,
            resolved_user_id=resolved_user_id,
            auto_cache=auto_cache,
        )
    else:
        client.chat.completions.create = _wrap_create(
            client.chat.completions.create,
            provider=provider,
            project=project,
            resolved_user_id=resolved_user_id,
            auto_cache=False,
        )

    return client
