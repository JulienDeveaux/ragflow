#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#


import os
import tiktoken

from common.file_utils import get_project_base_directory

tiktoken_cache_dir = get_project_base_directory()
os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir
# encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")
encoder = tiktoken.get_encoding("cl100k_base")


def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    try:
        code_list = encoder.encode(string)
        return len(code_list)
    except Exception:
        return 0

def _coerce_token_details(value):
    """
    Normalize a ``prompt_tokens_details`` / ``completion_tokens_details`` payload
    (OpenAI / Mistral / DeepSeek expose ``cached_tokens``, ``reasoning_tokens``,
    ``audio_tokens``, ``accepted_prediction_tokens``, ...) into a plain dict of
    numeric values.

    Accepts either a Pydantic model (OpenAI SDK) or a plain dict. Returns ``{}``
    when no useful data is present so callers can ``.update()`` unconditionally.
    """
    if value is None:
        return {}
    if isinstance(value, dict):
        items = value.items()
    elif hasattr(value, "model_dump"):
        try:
            items = value.model_dump(exclude_none=True).items()
        except Exception:
            return {}
    else:
        # OpenAI SDK pre-v1 / arbitrary object: pull its public numeric attrs.
        try:
            items = ((k, getattr(value, k)) for k in dir(value)
                     if not k.startswith("_") and not callable(getattr(value, k, None)))
        except Exception:
            return {}
    out = {}
    for k, v in items:
        # Reject bool — Python's isinstance(True, int) gotcha would pollute traces.
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out[k] = v
    return out


def usage_dict_from_response(resp):
    """
    Extract per-segment token usage from an LLM provider response.

    Returns a dict shaped like the OpenAI usage block:
    ``{"prompt_tokens": int, "completion_tokens": int, "total_tokens": int}``.
    Missing segments default to 0 and ``total_tokens`` is back-filled from
    ``prompt_tokens + completion_tokens`` when the provider only ships the
    pair. Falls back to ``{"prompt_tokens": 0, "completion_tokens": total,
    "total_tokens": total}`` when only an aggregate count is available — that
    preserves billing accuracy at the price of losing the split for providers
    that don't expose it.

    When the provider also exposes ``prompt_tokens_details`` /
    ``completion_tokens_details`` (OpenAI ``gpt-4o`` cached_tokens, ``o1``
    reasoning_tokens, ...), those sub-dicts are propagated under the same
    keys so downstream consumers (Langfuse trace UI) can show the breakdown.

    Always returns a dict; callers can safely ``.get()`` each field.
    """
    zero = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if resp is None:
        return zero

    def _build(p, c, t):
        p = int(p or 0)
        c = int(c or 0)
        t = int(t or 0) or (p + c)
        return {"prompt_tokens": p, "completion_tokens": c, "total_tokens": t}

    def _attach_details(d, prompt_details, completion_details):
        pd = _coerce_token_details(prompt_details)
        cd = _coerce_token_details(completion_details)
        if pd:
            d["prompt_tokens_details"] = pd
        if cd:
            d["completion_tokens_details"] = cd
        return d

    # OpenAI-compatible: resp.usage.{prompt_tokens, completion_tokens, total_tokens}
    try:
        u = getattr(resp, "usage", None)
        if u is not None and any(
            hasattr(u, k) for k in ("prompt_tokens", "completion_tokens", "total_tokens")
        ):
            d = _build(
                getattr(u, "prompt_tokens", 0),
                getattr(u, "completion_tokens", 0),
                getattr(u, "total_tokens", 0),
            )
            if d["total_tokens"]:
                return _attach_details(
                    d,
                    getattr(u, "prompt_tokens_details", None),
                    getattr(u, "completion_tokens_details", None),
                )
    except Exception:
        pass

    # Anthropic / Bedrock style: resp.usage_metadata.{input_tokens, output_tokens, total_tokens}
    try:
        u = getattr(resp, "usage_metadata", None)
        if u is not None:
            d = _build(
                getattr(u, "input_tokens", 0),
                getattr(u, "output_tokens", 0),
                getattr(u, "total_tokens", 0),
            )
            if d["total_tokens"]:
                return d
    except Exception:
        pass

    # Dict response
    if isinstance(resp, dict):
        u = resp.get("usage") or {}
        d = _build(
            u.get("prompt_tokens") or u.get("input_tokens"),
            u.get("completion_tokens") or u.get("output_tokens"),
            u.get("total_tokens"),
        )
        if d["total_tokens"]:
            return _attach_details(
                d,
                u.get("prompt_tokens_details"),
                u.get("completion_tokens_details"),
            )
        meta = resp.get("meta") or {}
        tokens = meta.get("tokens") or {}
        d = _build(tokens.get("input_tokens"), tokens.get("output_tokens"), None)
        if d["total_tokens"]:
            return d

    # Last-resort: re-use the legacy aggregate extractor and assume it's completion.
    total = total_token_count_from_response(resp)
    if total:
        return {"prompt_tokens": 0, "completion_tokens": int(total), "total_tokens": int(total)}
    return zero


def total_token_count_from_response(resp):
    """
    Extract token count from LLM response in various formats.

    Handles None responses and different response structures from various LLM providers.
    Returns 0 if token count cannot be determined.
    """
    if resp is None:
        return 0

    try:
        if hasattr(resp, "usage") and hasattr(resp.usage, "total_tokens"):
            return resp.usage.total_tokens
    except Exception:
        pass

    try:
        if hasattr(resp, "usage_metadata") and hasattr(resp.usage_metadata, "total_tokens"):
            return resp.usage_metadata.total_tokens
    except Exception:
        pass

    try:
        if hasattr(resp, "meta") and hasattr(resp.meta, "billed_units") and hasattr(resp.meta.billed_units, "input_tokens"):
            return resp.meta.billed_units.input_tokens
    except Exception:
        pass

    if isinstance(resp, dict) and 'usage' in resp and 'total_tokens' in resp['usage']:
        try:
            return resp["usage"]["total_tokens"]
        except Exception:
            pass

    if isinstance(resp, dict) and 'usage' in resp and 'input_tokens' in resp['usage'] and 'output_tokens' in resp['usage']:
        try:
            return resp["usage"]["input_tokens"] + resp["usage"]["output_tokens"]
        except Exception:
            pass

    if isinstance(resp, dict) and 'meta' in resp and 'tokens' in resp['meta'] and 'input_tokens' in resp['meta']['tokens'] and 'output_tokens' in resp['meta']['tokens']:
        try:
            return resp["meta"]["tokens"]["input_tokens"] + resp["meta"]["tokens"]["output_tokens"]
        except Exception:
            pass
    return 0


def truncate(string: str, max_len: int) -> str:
    """Returns truncated text if the length of text exceed max_len."""
    return encoder.decode(encoder.encode(string)[:max_len])
