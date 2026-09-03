"""Per-host rates, read from the router rather than hardcoded."""

import json
import urllib.request

ENDPOINTS = "https://openrouter.ai/api/v1/models/{model}/endpoints"

_cache = {}


def endpoints(model, api_key):
    """[(provider, usd per input token, usd per output token)], cheapest first."""
    if model in _cache:
        return _cache[model]
    request = urllib.request.Request(
        ENDPOINTS.format(model=model), headers={"Authorization": f"Bearer {api_key}"})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            data = json.load(response)["data"]["endpoints"]
    except Exception:
        return []
    out = sorted(((e.get("provider_name"), float(e.get("pricing", {}).get("prompt", 0)),
                   float(e.get("pricing", {}).get("completion", 0))) for e in data),
                 key=lambda row: row[1])
    _cache[model] = out
    return out


def rates(model, api_key, provider=None):
    """(input, output) usd per token for a pinned host; (None, None) when unpinned."""
    for name, price_in, price_out in endpoints(model, api_key):
        if provider and name == provider:
            return price_in, price_out
    return None, None
