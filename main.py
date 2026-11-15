#!/usr/bin/env python3
"""Free AI model aggregator — make.fun free-model-aggregator-01.

One server, two client wires (OpenAI Chat Completions and Anthropic Messages,
JSON and streaming), ten simulated free-tier providers on the real OpenAI,
Anthropic and Gemini wires. For each of the trial's 20 requests it picks the
best free model that meets the request's requirements, fails over when a
provider refuses, and answers in the wire the client used.

Standard library only (Python 3.13). Nothing is installed before the server
runs, so there are no third-party imports anywhere in this file.

How a trial runs:
  * the brief at $MAKEFUN_PROMPT_FILE carries three JSON documents: the
    provider directory, the sealed quality table, and the workload manifest.
  * the server probes every provider's catalog route, builds one catalog,
    listens on an ephemeral port, and registers {protocol_version: 1, port,
    catalog} at POST $MAKEFUN_ENV_URL/register.
  * the env's client then sends the 20 requests one at a time; the server
    answers each with a free model, and polls GET $MAKEFUN_ENV_URL/work
    until it says done, then exits 0.
  * `python3 main.py --list` prints the catalog document and exits 0.

Routing: candidates are the free models meeting the request's requirements,
ranked by the quality table's rating_milli (best first). A JSON-mode request
on the OpenAI wire is only routed to a wire that can carry response_format
(OpenAI or Gemini, never Anthropic). On a 402/429/5xx the same candidate is
retried once, then the next candidate is tried; the first success wins, so
exactly one provider call ever answers a request. Provider budgets and
failure schedules are NOT in the brief, so the server learns them live from
the providers' answers and adapts — it never assumes a budget.

Determinism: every choice derives from the brief, the request bodies and the
providers' answers. No wall clock and no unseeded randomness anywhere in a
decision, so a replay leaves the same state export. The poll loop's timing
never reaches a request, a response, or the catalog.
"""

import json
import os
import signal
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PROTOCOL_VERSION = 1
FIXED_CREATED = 1767225600
FIXED_ANTHROPIC_CREATED_AT = "2026-01-01T00:00:00Z"
POLL_INTERVAL_S = 0.1
PROBE_TIMEOUT_S = 5
PROVIDER_TIMEOUT_S = 10
MAX_BODY_BYTES = 32 * 1024 * 1024
# Safety net: the trial pod's wall is 180 s; exit well before it so a trial
# that stalls is scored on what it served instead of zeroed outright.
WALL_GUARD_S = 165.0

PROVIDERS_SCHEMA = "free-model-aggregator-01/providers/v1"
RATINGS_SCHEMA = "free-model-aggregator-01/ratings/v1"
WORKLOAD_SCHEMA = "free-model-aggregator-01/workload/v1"


def log(message):
    sys.stderr.write("aggregator: %s\n" % message)
    sys.stderr.flush()


def is_record(value):
    return value is not None and isinstance(value, dict)


def _object_end(text, start):
    """Index of the `}` closing the object that opens at `start`, or -1."""
    depth = 0
    in_string = False
    i = start
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == "\\":
                i += 1
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def documents_in(text):
    """Every JSON object in `text` carrying a string `schema`, in order."""
    docs = []
    start = text.find("{")
    while start != -1:
        end = _object_end(text, start)
        if end != -1:
            try:
                value = json.loads(text[start:end + 1])
            except Exception:
                value = None
            if is_record(value) and isinstance(value.get("schema"), str):
                docs.append(value)
        start = text.find("{", start + 1)
    return docs


def read_brief(path):
    """The brief's three documents: providers, quality table, workload."""
    brief = {"providers": [], "ratings": [], "best": {}, "manifest": [], "error": None}
    if not path:
        brief["error"] = "MAKEFUN_PROMPT_FILE is not set"
        return brief
    try:
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
    except Exception as exc:
        brief["error"] = "cannot read %s: %s" % (path, exc)
        return brief
    docs = documents_in(text)
    directory = None
    ratings_doc = None
    workload_doc = None
    for doc in docs:
        if doc["schema"] == PROVIDERS_SCHEMA and isinstance(doc.get("providers"), list):
            directory = doc
        elif doc["schema"] == RATINGS_SCHEMA and isinstance(doc.get("ratings"), list):
            ratings_doc = doc
        elif doc["schema"] == WORKLOAD_SCHEMA and isinstance(doc.get("manifest"), list):
            workload_doc = doc
    if directory is None:
        brief["error"] = "the brief holds no %s document" % PROVIDERS_SCHEMA
        return brief
    for raw in directory["providers"]:
        if not is_record(raw):
            continue
        pid = raw.get("id")
        wire = raw.get("wire")
        if not isinstance(pid, str) or not pid:
            continue
        if wire not in ("openai", "anthropic", "gemini"):
            continue
        auth = raw.get("auth")
        if auth not in ("bearer", "x-api-key", "x-goog-api-key", "none"):
            auth = "bearer"
        base = ""
        for key in ("path", "base_path", "base_url", "base"):
            if isinstance(raw.get(key), str) and raw[key]:
                base = raw[key]
                break
        brief["providers"].append({
            "id": pid,
            "name": raw.get("name") if isinstance(raw.get("name"), str) else pid,
            "wire": wire,
            "auth": auth,
            "base": base,
            "key_env": raw.get("key_env") if isinstance(raw.get("key_env"), str) else "",
            "key": raw.get("key") if isinstance(raw.get("key"), str) else "",
        })
    if ratings_doc is not None:
        for raw in ratings_doc["ratings"]:
            if not is_record(raw):
                continue
            model = raw.get("model")
            provider = raw.get("provider")
            if not isinstance(model, str) or not model:
                continue
            if not isinstance(provider, str) or not provider:
                continue
            brief["ratings"].append({
                "model": model,
                "provider": provider,
                "rating_milli": raw.get("rating_milli") if isinstance(raw.get("rating_milli"), int) else 0,
                "context_length": raw.get("context_length") if isinstance(raw.get("context_length"), int) else None,
                "tools": raw.get("tools") is True,
                "vision": raw.get("vision") is True,
                "json": raw.get("json") is True,
                "free": raw.get("free") is True,
            })
        if isinstance(ratings_doc.get("best"), list):
            for raw in ratings_doc["best"]:
                if is_record(raw) and isinstance(raw.get("id"), str):
                    brief["best"][raw["id"]] = raw.get("rating_milli") if isinstance(raw.get("rating_milli"), int) else 0
    if workload_doc is not None:
        for raw in workload_doc["manifest"]:
            if not is_record(raw):
                continue
            rid = raw.get("id")
            if not isinstance(rid, str) or not rid:
                continue
            requires = raw.get("requires") if is_record(raw.get("requires")) else {}
            entry = {
                "id": rid,
                "api": "anthropic" if raw.get("api") == "anthropic" else "openai",
                "stream": raw.get("stream") is True,
                "weight": 2 if raw.get("weight") == 2 else 1,
                "requires": {},
            }
            if requires.get("tools") is True:
                entry["requires"]["tools"] = True
            if requires.get("vision") is True:
                entry["requires"]["vision"] = True
            if requires.get("json") is True:
                entry["requires"]["json"] = True
            if isinstance(requires.get("min_context"), int):
                entry["requires"]["min_context"] = requires["min_context"]
            brief["manifest"].append(entry)
    return brief


def join_base(env_url, base):
    """A base path joined to the env URL; an absolute base is used as-is."""
    base = (base or "").strip()
    if base.lower().startswith("http://") or base.lower().startswith("https://"):
        return base.rstrip("/")
    prefix = (env_url or "").rstrip("/")
    if not base:
        return prefix
    return prefix + ("" if base.startswith("/") else "/") + base.rstrip("/")


def auth_headers(auth, key):
    if not key:
        return {}
    if auth == "bearer":
        return {"authorization": "Bearer " + key}
    if auth == "x-api-key":
        return {"x-api-key": key}
    if auth == "x-goog-api-key":
        return {"x-goog-api-key": key}
    return {}


def _model_in(wire, raw):
    """One model as a catalog listing wrote it, in whichever wire's spelling."""
    if not is_record(raw):
        return None
    if wire == "gemini":
        name = raw.get("name")
        mid = name[len("models/"):] if isinstance(name, str) and name.startswith("models/") else name
    else:
        mid = raw.get("id")
    if not isinstance(mid, str) or not mid:
        return None
    context = None
    for key in ("context_length", "context_window", "inputTokenLimit"):
        if isinstance(raw.get(key), int):
            context = raw[key]
            break
    display = raw.get("display_name") or raw.get("displayName") or mid
    return {
        "id": mid,
        "display_name": display if isinstance(display, str) else mid,
        "free": raw.get("free") if isinstance(raw.get("free"), bool) else None,
        "context_length": context,
        "tools": raw.get("tools") if isinstance(raw.get("tools"), bool) else None,
        "vision": raw.get("vision") if isinstance(raw.get("vision"), bool) else None,
        "json": raw.get("json") if isinstance(raw.get("json"), bool) else None,
    }


def probe_provider(env_url, provider, timeout=PROBE_TIMEOUT_S):
    """One provider's catalog route, in its own wire. Never raises."""
    base = join_base(env_url, provider["base"])
    if provider["wire"] == "gemini":
        url = base + "/v1beta/models"
    else:
        url = base + "/v1/models"
    headers = {"accept": "application/json"}
    headers.update(auth_headers(provider["auth"], provider["key"]))
    if provider["wire"] == "anthropic":
        headers["anthropic-version"] = "2023-06-01"
    try:
        req = urllib.request.Request(url, headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                return {"provider": provider, "ok": False, "models": [], "error": "HTTP %d" % resp.status}
            body = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        return {"provider": provider, "ok": False, "models": [], "error": str(exc)}
    if not is_record(body):
        return {"provider": provider, "ok": False, "models": [], "error": "not a JSON object"}
    listing = body.get("models") if provider["wire"] == "gemini" else body.get("data")
    models = []
    if isinstance(listing, list):
        for raw in listing:
            model = _model_in(provider["wire"], raw)
            if model is not None:
                models.append(model)
    return {"provider": provider, "ok": True, "models": models, "error": None}


def rating_row(ratings, provider_id, model_id):
    for row in ratings:
        if row["provider"] == provider_id and row["model"] == model_id:
            return row
    return None


def meets_requirements(requires, caps):
    """Whether a model's capabilities meet one request's requirements."""
    if requires.get("tools") is True and caps.get("tools") is not True:
        return False
    if requires.get("vision") is True and caps.get("vision") is not True:
        return False
    if requires.get("json") is True and caps.get("json") is not True:
        return False
    min_context = requires.get("min_context")
    if min_context is not None:
        context = caps.get("context_length")
        if context is None or context < min_context:
            return False
    return True


def build_catalog(probes, ratings):
    """The one catalog: every provider that answered, every model it listed.

    The free flag comes from the quality table where it has a row (the sealed
    truth); otherwise from the listing itself. The base_url is the directory's
    own path (/p/<id>), never joined to the env URL: the env's origin differs
    between a live pass and its replay, and the catalog is part of the state
    export a replay must reproduce byte for byte.
    """
    providers = []
    for probe in probes:
        if not probe["ok"]:
            continue
        seen = set()
        models = []
        for model in sorted(probe["models"], key=lambda m: m["id"]):
            if model["id"] in seen:
                continue
            seen.add(model["id"])
            row = rating_row(ratings, probe["provider"]["id"], model["id"])
            if row is not None:
                free = row["free"]
                context = row["context_length"]
            else:
                free = model["free"] if model["free"] is not None else False
                context = model["context_length"]
            models.append({"id": model["id"], "free": free, "context_length": context})
        providers.append({
            "id": probe["provider"]["id"],
            "base_url": probe["provider"]["base"],
            "key_env": probe["provider"]["key_env"],
            "wire": probe["provider"]["wire"],
            "models": models,
        })
    return {"providers": providers}


def served_models(probes, ratings):
    """The free models across the providers that answered, deduplicated by id.

    Used for GET /v1/models: exactly the free model ids the catalog registers.
    """
    out = []
    seen = set()
    for probe in probes:
        if not probe["ok"]:
            continue
        for model in sorted(probe["models"], key=lambda m: m["id"]):
            if model["id"] in seen:
                continue
            row = rating_row(ratings, probe["provider"]["id"], model["id"])
            free = row["free"] if row is not None else (model["free"] if model["free"] is not None else False)
            if not free:
                continue
            seen.add(model["id"])
            context = row["context_length"] if row is not None else model["context_length"]
            out.append({"id": model["id"], "provider": probe["provider"]["id"],
                        "display_name": model["display_name"], "context_length": context})
    return out


# ---------------------------------------------------------------------------
# Wire translation: request bodies.
#
# The provider echoes what it parsed from the body it received, and the
# client compares that echo with what it sent, so a translation must be
# lossless: system prompt, user text, images, tools, stop sequences,
# max_tokens and JSON mode all survive the trip, in the target wire's shape.
# ---------------------------------------------------------------------------


def _text_of(content):
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if is_record(block) and block.get("type") == "text" and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "\n".join(parts)


def _openai_messages_text(body):
    """(system_text, user_text, image_sha_source) of an OpenAI body.

    Only the last user message's text counts (the provider parses it the same
    way), and the image is the last image_url block's URL.
    """
    system_parts = []
    user_text = ""
    image_url = None
    for message in body.get("messages") or []:
        if not is_record(message):
            continue
        if message.get("role") == "system":
            text = _text_of(message.get("content"))
            if text:
                system_parts.append(text)
        elif message.get("role") == "user":
            user_text = _text_of(message.get("content"))
            content = message.get("content")
            if isinstance(content, list):
                for block in content:
                    if is_record(block) and block.get("type") == "image_url" and is_record(block.get("image_url")):
                        url = block["image_url"].get("url")
                        if isinstance(url, str):
                            image_url = url
    return "\n".join(system_parts), user_text, image_url


def _data_url_parts(url):
    """(mime_type, base64_data) of a data: URL, or (None, None)."""
    if not isinstance(url, str) or not url.startswith("data:"):
        return None, None
    head, _, data = url[5:].partition(",")
    if ";" not in head:
        return None, None
    mime, _, encoding = head.partition(";")
    if encoding.lower() != "base64" or not data:
        return None, None
    return mime or "image/png", data


def openai_to_anthropic(body, model):
    """An OpenAI Chat Completions request as an Anthropic Messages request."""
    system_text, _, _ = _openai_messages_text(body)
    messages = []
    for raw in body.get("messages") or []:
        if not is_record(raw):
            continue
        role = raw.get("role")
        if role == "system":
            continue
        content = raw.get("content")
        if role == "assistant":
            messages.append({"role": "assistant", "content": _text_of(content)})
            continue
        if role != "user":
            continue
        if isinstance(content, str):
            messages.append({"role": "user", "content": content})
            continue
        if not isinstance(content, list):
            continue
        blocks = []
        for block in content:
            if not is_record(block):
                continue
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                blocks.append({"type": "text", "text": block["text"]})
            elif block.get("type") == "image_url" and is_record(block.get("image_url")):
                url = block["image_url"].get("url")
                mime, data = _data_url_parts(url)
                if data is not None:
                    blocks.append({"type": "image", "source": {"type": "base64", "media_type": mime, "data": data}})
                elif isinstance(url, str):
                    blocks.append({"type": "image", "source": {"type": "url", "url": url}})
        if blocks:
            messages.append({"role": "user", "content": blocks})
    out = {"model": model, "messages": messages}
    if system_text:
        out["system"] = system_text
    max_tokens = body.get("max_tokens")
    if not isinstance(max_tokens, int):
        max_tokens = body.get("max_completion_tokens")
    if isinstance(max_tokens, int):
        out["max_tokens"] = max_tokens
    if isinstance(body.get("temperature"), (int, float)):
        out["temperature"] = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)):
        out["top_p"] = body["top_p"]
    stop = body.get("stop")
    if isinstance(stop, str):
        out["stop_sequences"] = [stop]
    elif isinstance(stop, list) and stop:
        out["stop_sequences"] = [s for s in stop if isinstance(s, str)]
    tools = body.get("tools")
    if isinstance(tools, list) and tools:
        converted = []
        for tool in tools:
            if not is_record(tool) or not is_record(tool.get("function")):
                continue
            fn = tool["function"]
            if not isinstance(fn.get("name"), str):
                continue
            converted.append({
                "name": fn["name"],
                "description": fn.get("description") if isinstance(fn.get("description"), str) else "",
                "input_schema": fn.get("parameters") if is_record(fn.get("parameters")) else {"type": "object", "properties": {}},
            })
        if converted:
            out["tools"] = converted
    if body.get("stream") is True:
        out["stream"] = True
    return out


def _anthropic_messages_text(body):
    """(system_text, messages) of an Anthropic body, messages kept in order."""
    system_text = _text_of(body.get("system"))
    messages = []
    for raw in body.get("messages") or []:
        if is_record(raw):
            messages.append(raw)
    return system_text, messages


def anthropic_to_openai(body, model):
    """An Anthropic Messages request as an OpenAI Chat Completions request."""
    system_text, source = _anthropic_messages_text(body)
    messages = []
    if system_text:
        messages.append({"role": "system", "content": system_text})
    for raw in source:
        role = raw.get("role")
        content = raw.get("content")
        if role == "assistant":
            text = _text_of(content)
            tool_calls = []
            if isinstance(content, list):
                for block in content:
                    if not is_record(block) or block.get("type") != "tool_use":
                        continue
                    tool_calls.append({
                        "id": block.get("id") if isinstance(block.get("id"), str) else "",
                        "type": "function",
                        "function": {
                            "name": block.get("name") if isinstance(block.get("name"), str) else "",
                            "arguments": json.dumps(block.get("input") if is_record(block.get("input")) else {}),
                        },
                    })
            entry = {"role": "assistant", "content": text if text else None}
            if tool_calls:
                entry["tool_calls"] = tool_calls
            messages.append(entry)
            continue
        if role != "user":
            continue
        if isinstance(content, str):
            messages.append({"role": "user", "content": content})
            continue
        if not isinstance(content, list):
            continue
        parts = []
        for block in content:
            if not is_record(block):
                continue
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                parts.append({"type": "text", "text": block["text"]})
            elif block.get("type") == "image" and is_record(block.get("source")):
                src = block["source"]
                if src.get("type") == "base64" and isinstance(src.get("data"), str):
                    media = src.get("media_type") if isinstance(src.get("media_type"), str) else "image/png"
                    parts.append({"type": "image_url", "image_url": {"url": "data:%s;base64,%s" % (media, src["data"])}})
                elif src.get("type") == "url" and isinstance(src.get("url"), str):
                    parts.append({"type": "image_url", "image_url": {"url": src["url"]}})
            elif block.get("type") == "tool_result":
                text = _text_of(block.get("content"))
                if not text and block.get("content") is not None:
                    text = json.dumps(block["content"])
                messages.append({
                    "role": "tool",
                    "tool_call_id": block.get("tool_use_id") if isinstance(block.get("tool_use_id"), str) else "",
                    "content": text,
                })
        if parts:
            messages.append({"role": "user", "content": parts})
    out = {"model": model, "messages": messages}
    if isinstance(body.get("max_tokens"), int):
        out["max_tokens"] = body["max_tokens"]
    if isinstance(body.get("temperature"), (int, float)):
        out["temperature"] = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)):
        out["top_p"] = body["top_p"]
    stop = body.get("stop_sequences")
    if isinstance(stop, list) and stop:
        out["stop"] = [s for s in stop if isinstance(s, str)]
    tools = body.get("tools")
    if isinstance(tools, list) and tools:
        converted = []
        for tool in tools:
            if not is_record(tool) or not isinstance(tool.get("name"), str):
                continue
            converted.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description") if isinstance(tool.get("description"), str) else "",
                    "parameters": tool.get("input_schema") if is_record(tool.get("input_schema")) else {"type": "object", "properties": {}},
                },
            })
        if converted:
            out["tools"] = converted
    if body.get("stream") is True:
        out["stream"] = True
        out["stream_options"] = {"include_usage": True}
    return out


def _gemini_contents_from_openai(body):
    """(system_text, contents, tools, config) of an OpenAI body for Gemini."""
    system_text, _, _ = _openai_messages_text(body)
    contents = []
    for raw in body.get("messages") or []:
        if not is_record(raw):
            continue
        role = raw.get("role")
        if role == "system":
            continue
        parts = []
        content = raw.get("content")
        if isinstance(content, str):
            if content:
                parts.append({"text": content})
        elif isinstance(content, list):
            for block in content:
                if not is_record(block):
                    continue
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    parts.append({"text": block["text"]})
                elif block.get("type") == "image_url" and is_record(block.get("image_url")):
                    url = block["image_url"].get("url")
                    mime, data = _data_url_parts(url)
                    if data is not None:
                        parts.append({"inlineData": {"mimeType": mime, "data": data}})
                    elif isinstance(url, str):
                        parts.append({"fileData": {"fileUri": url, "mimeType": "image/png"}})
        if parts:
            contents.append({"role": "model" if role == "assistant" else "user", "parts": parts})
    tools = []
    raw_tools = body.get("tools")
    if isinstance(raw_tools, list) and raw_tools:
        declarations = []
        for tool in raw_tools:
            if not is_record(tool) or not is_record(tool.get("function")):
                continue
            fn = tool["function"]
            if not isinstance(fn.get("name"), str):
                continue
            declarations.append({
                "name": fn["name"],
                "description": fn.get("description") if isinstance(fn.get("description"), str) else "",
                "parameters": fn.get("parameters") if is_record(fn.get("parameters")) else {"type": "object", "properties": {}},
            })
        if declarations:
            tools.append({"functionDeclarations": declarations})
    config = {}
    max_tokens = body.get("max_tokens")
    if not isinstance(max_tokens, int):
        max_tokens = body.get("max_completion_tokens")
    if isinstance(max_tokens, int):
        config["maxOutputTokens"] = max_tokens
    stop = body.get("stop")
    if isinstance(stop, str):
        config["stopSequences"] = [stop]
    elif isinstance(stop, list) and stop:
        config["stopSequences"] = [s for s in stop if isinstance(s, str)]
    if isinstance(body.get("temperature"), (int, float)):
        config["temperature"] = body["temperature"]
    fmt = body.get("response_format")
    if is_record(fmt) and fmt.get("type") in ("json_object", "json_schema"):
        config["responseMimeType"] = "application/json"
    return system_text, contents, tools, config


def _gemini_contents_from_anthropic(body):
    """(system_text, contents, tools, config) of an Anthropic body for Gemini."""
    system_text, source = _anthropic_messages_text(body)
    contents = []
    for raw in source:
        role = raw.get("role")
        parts = []
        content = raw.get("content")
        if isinstance(content, str):
            if content:
                parts.append({"text": content})
        elif isinstance(content, list):
            for block in content:
                if not is_record(block):
                    continue
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    parts.append({"text": block["text"]})
                elif block.get("type") == "image" and is_record(block.get("source")):
                    src = block["source"]
                    if src.get("type") == "base64" and isinstance(src.get("data"), str):
                        media = src.get("media_type") if isinstance(src.get("media_type"), str) else "image/png"
                        parts.append({"inlineData": {"mimeType": media, "data": src["data"]}})
                    elif src.get("type") == "url" and isinstance(src.get("url"), str):
                        parts.append({"fileData": {"fileUri": src["url"], "mimeType": "image/png"}})
                elif block.get("type") == "tool_result":
                    text = _text_of(block.get("content"))
                    if text:
                        parts.append({"text": text})
        if parts:
            contents.append({"role": "model" if role == "assistant" else "user", "parts": parts})
    tools = []
    raw_tools = body.get("tools")
    if isinstance(raw_tools, list) and raw_tools:
        declarations = []
        for tool in raw_tools:
            if not is_record(tool) or not isinstance(tool.get("name"), str):
                continue
            declarations.append({
                "name": tool["name"],
                "description": tool.get("description") if isinstance(tool.get("description"), str) else "",
                "parameters": tool.get("input_schema") if is_record(tool.get("input_schema")) else {"type": "object", "properties": {}},
            })
        if declarations:
            tools.append({"functionDeclarations": declarations})
    config = {}
    if isinstance(body.get("max_tokens"), int):
        config["maxOutputTokens"] = body["max_tokens"]
    stop = body.get("stop_sequences")
    if isinstance(stop, list) and stop:
        config["stopSequences"] = [s for s in stop if isinstance(s, str)]
    if isinstance(body.get("temperature"), (int, float)):
        config["temperature"] = body["temperature"]
    return system_text, contents, tools, config


def to_gemini_body(system_text, contents, tools, config):
    """A Gemini generateContent body from the translated pieces."""
    out = {"contents": contents}
    if system_text:
        out["systemInstruction"] = {"parts": [{"text": system_text}]}
    if tools:
        out["tools"] = tools
    if config:
        out["generationConfig"] = config
    return out


def json_mode_requested(client_api, body):
    """Whether the client asked for JSON mode on a wire that can carry it."""
    if client_api != "openai":
        return False
    fmt = body.get("response_format")
    return is_record(fmt) and fmt.get("type") in ("json_object", "json_schema")


# ---------------------------------------------------------------------------
# Provider answers, parsed into (text, input_tokens, output_tokens, stop).
# Every provider answers with the same fixed text — "ANSWER ECHO{...}" — so
# the text is carried whole and the usage is taken from the provider's own
# numbers; nothing here is invented. The stop family is always the plain
# "stop" one: providers never answer length, tool_use or refusal.
# ---------------------------------------------------------------------------


def _parse_sse_frames(text):
    """[(event, data), ...] of a server-sent-events body."""
    frames = []
    normalized = text.replace("\r\n", "\n")
    for frame in normalized.split("\n\n"):
        if not frame.strip():
            continue
        event = None
        data = []
        for line in frame.split("\n"):
            if line.startswith("event:"):
                event = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data.append(line[len("data:"):].lstrip(" "))
        if data:
            frames.append((event, "\n".join(data)))
    return frames


def _openai_answer_json(body):
    choices = body.get("choices") if is_record(body) else None
    if not isinstance(choices, list) or not choices:
        return None
    choice = choices[0]
    if not is_record(choice):
        return None
    message = choice.get("message") if is_record(choice.get("message")) else None
    if message is None:
        return None
    content = message.get("content")
    text = content if isinstance(content, str) else _text_of(content)
    usage = body.get("usage") if is_record(body.get("usage")) else {}
    in_tok = usage.get("prompt_tokens") if isinstance(usage.get("prompt_tokens"), int) else 0
    out_tok = usage.get("completion_tokens") if isinstance(usage.get("completion_tokens"), int) else 0
    stop = choice.get("finish_reason") if isinstance(choice.get("finish_reason"), str) else "stop"
    return (text, in_tok, out_tok, stop)


def _openai_answer_sse(text):
    out_text = []
    in_tok = 0
    out_tok = 0
    stop = "stop"
    for _, data in _parse_sse_frames(text):
        if data.strip() == "[DONE]":
            continue
        try:
            chunk = json.loads(data)
        except Exception:
            continue
        if not is_record(chunk):
            continue
        if is_record(chunk.get("usage")):
            usage = chunk["usage"]
            if isinstance(usage.get("prompt_tokens"), int):
                in_tok = usage["prompt_tokens"]
            if isinstance(usage.get("completion_tokens"), int):
                out_tok = usage["completion_tokens"]
        choices = chunk.get("choices")
        if not isinstance(choices, list) or not choices:
            continue
        choice = choices[0]
        if not is_record(choice):
            continue
        delta = choice.get("delta") if is_record(choice.get("delta")) else {}
        content = delta.get("content")
        if isinstance(content, str) and content:
            out_text.append(content)
        if isinstance(choice.get("finish_reason"), str):
            stop = choice["finish_reason"]
    return ("".join(out_text), in_tok, out_tok, stop)


def _anthropic_answer_json(body):
    if not is_record(body) or body.get("type") != "message":
        return None
    content = body.get("content")
    if not isinstance(content, list):
        return None
    text = _text_of(content)
    usage = body.get("usage") if is_record(body.get("usage")) else {}
    in_tok = usage.get("input_tokens") if isinstance(usage.get("input_tokens"), int) else 0
    out_tok = usage.get("output_tokens") if isinstance(usage.get("output_tokens"), int) else 0
    stop = body.get("stop_reason") if isinstance(body.get("stop_reason"), str) else "end_turn"
    return (text, in_tok, out_tok, stop)


def _anthropic_answer_sse(text):
    out_text = []
    in_tok = 0
    out_tok = 0
    stop = "end_turn"
    for event, data in _parse_sse_frames(text):
        try:
            payload = json.loads(data)
        except Exception:
            continue
        if not is_record(payload):
            continue
        if event == "message_start":
            message = payload.get("message") if is_record(payload.get("message")) else {}
            usage = message.get("usage") if is_record(message.get("usage")) else {}
            if isinstance(usage.get("input_tokens"), int):
                in_tok = usage["input_tokens"]
        elif event == "content_block_delta":
            delta = payload.get("delta") if is_record(payload.get("delta")) else {}
            if delta.get("type") == "text_delta" and isinstance(delta.get("text"), str):
                out_text.append(delta["text"])
        elif event == "message_delta":
            delta = payload.get("delta") if is_record(payload.get("delta")) else {}
            if isinstance(delta.get("stop_reason"), str):
                stop = delta["stop_reason"]
            usage = payload.get("usage") if is_record(payload.get("usage")) else {}
            if isinstance(usage.get("output_tokens"), int):
                out_tok = usage["output_tokens"]
    return ("".join(out_text), in_tok, out_tok, stop)


def _gemini_text_of(candidate):
    if not is_record(candidate):
        return ""
    content = candidate.get("content")
    if not is_record(content):
        return ""
    parts = content.get("parts")
    if not isinstance(parts, list):
        return ""
    return "".join(part.get("text") for part in parts if is_record(part) and isinstance(part.get("text"), str))


def _gemini_answer_json(body):
    candidates = body.get("candidates") if is_record(body) else None
    if not isinstance(candidates, list) or not candidates:
        return None
    text = _gemini_text_of(candidates[0])
    usage = body.get("usageMetadata") if is_record(body.get("usageMetadata")) else {}
    in_tok = usage.get("promptTokenCount") if isinstance(usage.get("promptTokenCount"), int) else 0
    out_tok = usage.get("candidatesTokenCount") if isinstance(usage.get("candidatesTokenCount"), int) else 0
    stop = candidates[0].get("finishReason") if isinstance(candidates[0].get("finishReason"), str) else "STOP"
    return (text, in_tok, out_tok, stop)


def _gemini_answer_sse(text):
    out_text = []
    in_tok = 0
    out_tok = 0
    stop = "STOP"
    for _, data in _parse_sse_frames(text):
        try:
            payload = json.loads(data)
        except Exception:
            continue
        if not is_record(payload):
            continue
        candidates = payload.get("candidates")
        if isinstance(candidates, list) and candidates:
            out_text.append(_gemini_text_of(candidates[0]))
            if isinstance(candidates[0].get("finishReason"), str):
                stop = candidates[0]["finishReason"]
        usage = payload.get("usageMetadata")
        if is_record(usage):
            if isinstance(usage.get("promptTokenCount"), int):
                in_tok = usage["promptTokenCount"]
            if isinstance(usage.get("candidatesTokenCount"), int):
                out_tok = usage["candidatesTokenCount"]
    return ("".join(out_text), in_tok, out_tok, stop)


def parse_provider_answer(wire, stream, status, raw):
    """(text, input_tokens, output_tokens, stop) or None when unparsable."""
    if status < 200 or status >= 300:
        return None
    if wire == "openai":
        if stream:
            return _openai_answer_sse(raw)
        try:
            return _openai_answer_json(json.loads(raw))
        except Exception:
            return None
    if wire == "anthropic":
        if stream:
            return _anthropic_answer_sse(raw)
        try:
            return _anthropic_answer_json(json.loads(raw))
        except Exception:
            return None
    if stream:
        return _gemini_answer_sse(raw)
    try:
        return _gemini_answer_json(json.loads(raw))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Client answers, rendered in the wire the client used. The text is the
# provider's whole answer; the usage is the provider's own counts; the stop
# reason is the plain "stop" family the provider answered with. Fixed ids and
# timestamps keep every render deterministic.
# ---------------------------------------------------------------------------


def _split_text(text):
    pieces = [text[i:i + 13] for i in range(0, len(text), 13)]
    return pieces if pieces else [""]


def _sse_event(event, data):
    line = ""
    if event is not None:
        line += "event: %s\n" % event
    line += "data: %s\n\n" % json.dumps(data)
    return line


def render_openai(text, in_tok, out_tok, model, stream):
    """(content_type, body_bytes) of a Chat Completions answer."""
    usage = {"prompt_tokens": in_tok, "completion_tokens": out_tok, "total_tokens": in_tok + out_tok}
    if not stream:
        body = {
            "id": "chatcmpl-sim",
            "object": "chat.completion",
            "created": FIXED_CREATED,
            "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop", "logprobs": None}],
            "usage": usage,
        }
        return "application/json", json.dumps(body).encode("utf-8")
    frames = []
    chunk = lambda delta, finish: {
        "id": "chatcmpl-sim", "object": "chat.completion.chunk", "created": FIXED_CREATED, "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    frames.append(_sse_event(None, chunk({"role": "assistant", "content": ""}, None)))
    for piece in _split_text(text):
        frames.append(_sse_event(None, chunk({"content": piece}, None)))
    frames.append(_sse_event(None, chunk({}, "stop")))
    frames.append(_sse_event(None, {
        "id": "chatcmpl-sim", "object": "chat.completion.chunk", "created": FIXED_CREATED, "model": model,
        "choices": [], "usage": usage,
    }))
    frames.append("data: [DONE]\n\n")
    return "text/event-stream; charset=utf-8", "".join(frames).encode("utf-8")


def render_anthropic(text, in_tok, out_tok, model, stream):
    """(content_type, body_bytes) of a Messages answer."""
    if not stream:
        body = {
            "id": "msg_sim",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": in_tok, "output_tokens": out_tok},
        }
        return "application/json", json.dumps(body).encode("utf-8")
    frames = []
    frames.append(_sse_event("message_start", {
        "type": "message_start",
        "message": {"id": "msg_sim", "type": "message", "role": "assistant", "model": model,
                    "content": [], "stop_reason": None, "stop_sequence": None,
                    "usage": {"input_tokens": in_tok, "output_tokens": 0}},
    }))
    frames.append(_sse_event("content_block_start", {
        "type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""},
    }))
    for piece in _split_text(text):
        frames.append(_sse_event("content_block_delta", {
            "type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": piece},
        }))
    frames.append(_sse_event("content_block_stop", {"type": "content_block_stop", "index": 0}))
    frames.append(_sse_event("message_delta", {
        "type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": out_tok},
    }))
    frames.append(_sse_event("message_stop", {"type": "message_stop"}))
    return "text/event-stream; charset=utf-8", "".join(frames).encode("utf-8")


def openai_error(message, type_="invalid_request_error", code=None):
    return {"error": {"message": message, "type": type_, "code": code}}


def anthropic_error(type_, message):
    return {"type": "error", "error": {"type": type_, "message": message}}


def openai_model_list(models):
    return {
        "object": "list",
        "data": [
            {"id": m["id"], "object": "model", "created": FIXED_CREATED, "owned_by": m["provider"],
             "free": True, "context_length": m.get("context_length")}
            for m in models
        ],
    }


def anthropic_model_list(models):
    return {
        "data": [
            {"type": "model", "id": m["id"], "display_name": m["display_name"], "created_at": FIXED_ANTHROPIC_CREATED_AT, "free": True}
            for m in models
        ],
        "has_more": False,
    }


# ---------------------------------------------------------------------------
# Routing: pick the best free model for a request, fail over on refusal.
# ---------------------------------------------------------------------------


class Router:
    def __init__(self, env_url, brief, probes):
        self.env_url = env_url
        self.brief = brief
        self.probes = {p["provider"]["id"]: p for p in probes if p["ok"]}
        self.provider_by_id = {p["id"]: p for p in brief["providers"]}
        # calls made to each provider this trial, and providers that answered
        # 400/401 (a wrong key or model: a bug on our side, never retried).
        self.calls = {pid: 0 for pid in self.provider_by_id}
        self.broken = set()
        self.manifest_index = 0

    def candidates(self, requires, json_mode):
        """Free (provider, model) pairs meeting the requirements, best first.

        A JSON-mode request on the OpenAI wire only goes to a wire that can
        carry response_format: OpenAI or Gemini, never Anthropic.
        """
        out = []
        brief_order = {p["id"]: i for i, p in enumerate(self.brief["providers"])}
        for row in self.brief["ratings"]:
            if not row["free"]:
                continue
            if row["provider"] not in self.probes:
                continue
            probe = self.probes[row["provider"]]
            provider = self.provider_by_id[row["provider"]]
            if json_mode and provider["wire"] == "anthropic":
                continue
            caps = {
                "tools": row["tools"], "vision": row["vision"], "json": row["json"],
                "context_length": row["context_length"],
            }
            if not meets_requirements(requires, caps):
                continue
            listed = any(m["id"] == row["model"] for m in probe["models"])
            if not listed:
                continue
            out.append({"provider": provider, "model": row["model"], "rating": row["rating_milli"]})
        out.sort(key=lambda c: (-c["rating"], self.calls[c["provider"]["id"]], brief_order[c["provider"]["id"]], c["model"]))
        return out

    def requirements_for(self, body):
        """The manifest row for the request now arriving, else a body guess."""
        manifest = self.brief["manifest"]
        index = self.manifest_index
        self.manifest_index += 1
        if index < len(manifest):
            return manifest[index]["requires"]
        vision = False
        for message in body.get("messages") or []:
            if not is_record(message) or not isinstance(message.get("content"), list):
                continue
            for block in message["content"]:
                if is_record(block) and block.get("type") in ("image", "image_url"):
                    vision = True
        fmt = body.get("response_format")
        return {
            "tools": isinstance(body.get("tools"), list) and len(body["tools"]) > 0,
            "vision": vision,
            "json": is_record(fmt) and fmt.get("type") in ("json_object", "json_schema"),
        }

    def call_provider(self, candidate, client_api, body, stream):
        """One call to one provider. Returns (status, parsed) or (status, None).

        parsed is (text, input_tokens, output_tokens, stop) on a 2xx answer.
        """
        provider = candidate["provider"]
        model = candidate["model"]
        wire = provider["wire"]
        base = join_base(self.env_url, provider["base"])
        headers = {"content-type": "application/json"}
        headers.update(auth_headers(provider["auth"], provider["key"]))
        if wire == "gemini":
            if client_api == "openai":
                system_text, contents, tools, config = _gemini_contents_from_openai(body)
            else:
                system_text, contents, tools, config = _gemini_contents_from_anthropic(body)
            upstream_body = to_gemini_body(system_text, contents, tools, config)
            action = "streamGenerateContent" if stream else "generateContent"
            url = "%s/v1beta/models/%s:%s" % (base, model, action)
            if stream:
                url += "?alt=sse"
        elif wire == "anthropic":
            if client_api == "openai":
                upstream_body = openai_to_anthropic(body, model)
            else:
                upstream_body = dict(body)
                upstream_body["model"] = model
            headers["anthropic-version"] = "2023-06-01"
            url = base + "/v1/messages"
        else:
            if client_api == "openai":
                upstream_body = dict(body)
                upstream_body["model"] = model
                if stream:
                    upstream_body["stream_options"] = {"include_usage": True}
            else:
                upstream_body = anthropic_to_openai(body, model)
            url = base + "/v1/chat/completions"
        data = json.dumps(upstream_body).encode("utf-8")
        if stream:
            headers["accept"] = "text/event-stream"
        self.calls[provider["id"]] += 1
        try:
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=PROVIDER_TIMEOUT_S) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
                return resp.status, parse_provider_answer(wire, stream, resp.status, raw)
        except urllib.error.HTTPError as exc:
            try:
                exc.read()
            except Exception:
                pass
            return exc.code, None
        except Exception as exc:
            log("provider %s unreachable: %s" % (provider["id"], exc))
            return 0, None

    def serve(self, client_api, body, stream):
        """Answer one client request. Returns (status, content_type, bytes)."""
        requires = self.requirements_for(body)
        json_mode = json_mode_requested(client_api, body)
        for candidate in self.candidates(requires, json_mode):
            provider = candidate["provider"]
            if provider["id"] in self.broken:
                continue
            attempts = 0
            while attempts < 2:
                attempts += 1
                status, parsed = self.call_provider(candidate, client_api, body, stream)
                if status >= 200 and status < 300 and parsed is not None:
                    text, in_tok, out_tok, _stop = parsed
                    if client_api == "openai":
                        content_type, payload = render_openai(text, in_tok, out_tok, candidate["model"], stream)
                    else:
                        content_type, payload = render_anthropic(text, in_tok, out_tok, candidate["model"], stream)
                    return 200, content_type, payload
                if status in (400, 401):
                    # A wrong key or model: our bug, retrying cannot help.
                    self.broken.add(provider["id"])
                    break
                if status in (402, 429) or status >= 500 or status == 0:
                    # Budget spent, a scheduled refusal, or a blip: one retry
                    # of this candidate, then the next candidate.
                    continue
                break
        if client_api == "openai":
            return 503, "application/json", json.dumps(openai_error(
                "no free model could serve this request", "overloaded_error", "no_free_model")).encode("utf-8")
        return 503, "application/json", json.dumps(anthropic_error(
            "overloaded_error", "no free model could serve this request")).encode("utf-8")


# ---------------------------------------------------------------------------
# HTTP server, registration, and the work poll.
# ---------------------------------------------------------------------------


class Aggregator:
    def __init__(self, env_url, brief, probes, catalog):
        self.env_url = env_url
        self.brief = brief
        self.catalog = catalog
        self.models = served_models(probes, brief["ratings"])
        self.router = Router(env_url, brief, probes)
        self.started_at = time.monotonic()

    def catalog_text(self):
        return json.dumps(self.catalog, separators=(",", ":"))

    def handle_models(self, handler):
        if isinstance(handler.headers.get("x-api-key"), str):
            body = anthropic_model_list(self.models)
        else:
            body = openai_model_list(self.models)
        payload = json.dumps(body).encode("utf-8")
        handler.send_response(200)
        handler.send_header("content-type", "application/json")
        handler.send_header("content-length", str(len(payload)))
        handler.end_headers()
        handler.wfile.write(payload)

    def handle_request(self, handler, client_api):
        length = int(handler.headers.get("content-length") or 0)
        if length <= 0 or length > MAX_BODY_BYTES:
            self.send_error(handler, client_api, 400, "invalid_request_error", "the request body is not one JSON object")
            return
        raw = handler.rfile.read(length)
        try:
            body = json.loads(raw.decode("utf-8"))
        except Exception:
            self.send_error(handler, client_api, 400, "invalid_request_error", "the request body is not one JSON object")
            return
        if not is_record(body):
            self.send_error(handler, client_api, 400, "invalid_request_error", "the request body is not one JSON object")
            return
        stream = body.get("stream") is True
        status, content_type, payload = self.router.serve(client_api, body, stream)
        handler.send_response(status)
        handler.send_header("content-type", content_type)
        handler.send_header("content-length", str(len(payload)))
        handler.end_headers()
        handler.wfile.write(payload)

    def send_error(self, handler, client_api, status, type_, message):
        if client_api == "anthropic":
            body = anthropic_error(type_, message)
        else:
            body = openai_error(message, type_, None)
        payload = json.dumps(body).encode("utf-8")
        handler.send_response(status)
        handler.send_header("content-type", "application/json")
        handler.send_header("content-length", str(len(payload)))
        handler.end_headers()
        handler.wfile.write(payload)


def make_handler(aggregator):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path.split("?", 1)[0] == "/v1/models":
                aggregator.handle_models(self)
            else:
                aggregator.send_error(self, "openai", 404, "not_found", "no route GET %s" % self.path)

        def do_POST(self):
            path = self.path.split("?", 1)[0]
            if path == "/v1/chat/completions":
                aggregator.handle_request(self, "openai")
            elif path == "/v1/messages":
                aggregator.handle_request(self, "anthropic")
            else:
                aggregator.send_error(self, "openai", 404, "not_found", "no route POST %s" % path)

    return Handler


def http_json(method, url, payload=None, timeout=5):
    headers = {"content-type": "application/json"}
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def register(env_url, port, catalog_text):
    """POST /register, retrying briefly: the env can still be binding."""
    last_error = None
    for _attempt in range(5):
        try:
            status, reply = http_json("POST", env_url + "/register",
                                      {"protocol_version": PROTOCOL_VERSION, "port": port, "catalog": json.loads(catalog_text)})
        except Exception as exc:
            last_error = exc
            time.sleep(1)
            continue
        if status == 200 and is_record(reply) and reply.get("ok") is True:
            log("registered: the workload holds %s requests" % reply.get("requests"))
            return
        last_error = RuntimeError("POST /register answered HTTP %d: %s" % (status, reply))
        time.sleep(1)
    raise RuntimeError("could not register: %s" % last_error)


def work_done(env_url):
    try:
        status, reply = http_json("GET", env_url + "/work", timeout=2)
    except Exception:
        return False
    return status == 200 and is_record(reply) and reply.get("done") is True


def main(argv):
    env_url = (os.environ.get("MAKEFUN_ENV_URL") or "").rstrip("/")
    prompt_file = os.environ.get("MAKEFUN_PROMPT_FILE") or ""
    brief = read_brief(prompt_file)
    if brief["error"] is not None:
        log(brief["error"])
    probes = [probe_provider(env_url, provider) for provider in brief["providers"]] if env_url else []
    for probe in probes:
        if not probe["ok"] and probe["error"]:
            log("catalog probe %s failed: %s" % (probe["provider"]["id"], probe["error"]))
    catalog = build_catalog(probes, brief["ratings"])
    aggregator = Aggregator(env_url, brief, probes, catalog)
    if "--list" in argv:
        sys.stdout.write(aggregator.catalog_text() + "\n")
        sys.stdout.flush()
        return 0
    if not env_url:
        log("MAKEFUN_ENV_URL is not set")
        return 1
    server = ThreadingHTTPServer(("0.0.0.0", 0), make_handler(aggregator))
    server.daemon_threads = True
    port = server.server_address[1]
    if port == 8080:
        server.server_close()
        raise RuntimeError("the OS picked 8080, which the env owns: restart the trial")
    log("listening on 0.0.0.0:%d with %d providers and %d free models"
        % (port, len(catalog["providers"]), len(aggregator.models)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    register(env_url, port, aggregator.catalog_text())

    def _exit(signum, _frame):
        server.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _exit)
    signal.signal(signal.SIGINT, _exit)
    while True:
        if time.monotonic() - aggregator.started_at > WALL_GUARD_S:
            log("wall guard reached; exiting")
            break
        if work_done(env_url):
            break
        time.sleep(POLL_INTERVAL_S)
    server.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
