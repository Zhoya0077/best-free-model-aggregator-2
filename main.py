"""Deterministic free-model router for both Chat Completions and Messages."""

import copy
import json
import os
import sys
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


MAX_TOKENS = 8192
MAX_BODY = 8 * 1024 * 1024
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def encode(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def load_brief():
    path = os.environ.get("MAKEFUN_PROMPT_FILE")
    documents = {}
    if path:
        for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
            if line.strip().startswith("{"):
                try:
                    item = json.loads(line)
                except ValueError:
                    continue
                if isinstance(item, dict) and "schema" in item:
                    documents[item["schema"]] = item
    models = documents.get("free-model-aggregator-02/models/v1", {}).get("models", [])
    prompts = documents.get("free-model-aggregator-02/prompts/v1", {}).get("prompts", [])
    return [m for m in models if m.get("free") is True], prompts


def make_catalog(models):
    return {"providers": [{"id": "openrouter", "base_url": "https://openrouter.ai/api/v1",
                           "key_env": "OPENROUTER_API_KEY", "wire": "openai",
                           "models": [{"id": m["id"], "free": True,
                                       "context_length": m.get("context_length")}
                                      for m in models]}]}


def env_request(path, body=None):
    request = urllib.request.Request(
        os.environ["MAKEFUN_ENV_URL"].rstrip("/") + path,
        data=None if body is None else encode(body),
        headers={"Content-Type": "application/json"},
        method="GET" if body is None else "POST")
    with OPENER.open(request, timeout=10) as response:
        return json.load(response)


class RouterServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, models, prompts):
        super().__init__(("0.0.0.0", 0), Handler)
        self.models = models
        self.prompts = prompts
        self.catalog = make_catalog(models)
        self.prompt_index = 0
        self.request_lock = threading.Lock()

    def register(self):
        return env_request("/register", {"protocol_version": 1,
                                       "port": self.server_address[1],
                                       "catalog": self.catalog})


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def reply(self, status, value, headers=None, content_type="application/json"):
        raw = value if isinstance(value, bytes) else encode(value)
        self.send_response_only(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(raw)))
        for key, val in (headers or {}).items():
            self.send_header(key, val)
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path.split("?", 1)[0] != "/v1/models":
            return self.reply(404, {"error": {"message": "Unknown route"}})
        models = self.server.models
        if self.headers.get("x-api-key"):
            data = [{"id": m["id"], "type": "model", "display_name": m.get("display_name", m["id"]),
                     "created_at": "1970-01-01T00:00:00Z"} for m in models]
            result = {"data": data, "has_more": False,
                      "first_id": data[0]["id"] if data else None,
                      "last_id": data[-1]["id"] if data else None}
        else:
            result = {"object": "list", "data": [
                {"id": m["id"], "object": "model", "created": 0, "owned_by": "openrouter"}
                for m in models]}
        self.reply(200, result)

    def do_POST(self):
        if self.path not in ("/v1/chat/completions", "/v1/messages"):
            return self.reply(404, {"error": {"message": "Unknown route"}})
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= MAX_BODY:
                return self.reply(413, {"error": {"message": "Invalid request size"}})
            body = json.loads(self.rfile.read(size))
            if not isinstance(body, dict) or not isinstance(body.get("messages"), list):
                raise ValueError("messages must be an array")
        except (ValueError, TypeError) as error:
            return self.reply(400, {"error": {"type": "invalid_request_error", "message": str(error)}})
        with self.server.request_lock:
            index = self.server.prompt_index
            self.server.prompt_index += 1
            manifest = self.server.prompts[index] if index < len(self.server.prompts) else {}
            serve_completion(self, body, self.path == "/v1/messages", manifest)


def text_content(content):
    if isinstance(content, str):
        return content
    return "\n".join(block.get("text", "") for block in content or []
                     if block.get("type") == "text")


def translate_request(body, anthropic):
    if not anthropic:
        return copy.deepcopy(body)
    result = {key: copy.deepcopy(body[key]) for key in
              ("model", "max_tokens", "stream", "temperature", "top_p") if key in body}
    messages = []
    if body.get("system"):
        messages.append({"role": "system", "content": text_content(body["system"])})
    for message in body["messages"]:
        content = message.get("content", "")
        if isinstance(content, str):
            messages.append({"role": message["role"], "content": content})
            continue
        parts, calls = [], []

        def flush():
            if parts or calls:
                row = {"role": message["role"], "content": copy.deepcopy(parts) if parts else None}
                if calls:
                    row["tool_calls"] = copy.deepcopy(calls)
                messages.append(row)
                parts.clear()
                calls.clear()

        for block in content:
            kind = block.get("type")
            if kind == "text":
                parts.append({"type": "text", "text": block["text"]})
            elif kind == "image":
                source = block["source"]
                url = source.get("url") if source.get("type") == "url" else (
                    "data:" + source["media_type"] + ";base64," + source["data"])
                parts.append({"type": "image_url", "image_url": {"url": url}})
            elif kind == "tool_use":
                calls.append({"id": block["id"], "type": "function", "function": {
                    "name": block["name"], "arguments": encode(block.get("input", {})).decode()}})
            elif kind == "tool_result":
                flush()
                messages.append({"role": "tool", "tool_call_id": block["tool_use_id"],
                                 "content": text_content(block.get("content", ""))})
        flush()
    result["messages"] = messages
    if "stop_sequences" in body:
        result["stop"] = body["stop_sequences"]
    if body.get("tools"):
        result["tools"] = [{"type": "function", "function": {
            "name": tool["name"], "description": tool.get("description", ""),
            "parameters": tool.get("input_schema", {"type": "object"})}} for tool in body["tools"]]
    choice = body.get("tool_choice")
    if choice:
        kind = choice.get("type")
        result["tool_choice"] = ({"type": "function", "function": {"name": choice["name"]}}
                                 if kind == "tool" else {"any": "required"}.get(kind, kind))
        if "disable_parallel_tool_use" in choice:
            result["parallel_tool_calls"] = not choice["disable_parallel_tool_use"]
    return result


class UpstreamError(Exception):
    def __init__(self, status, code, body):
        super().__init__(str(code))
        self.status, self.code, self.body = status, str(code), body


def capped_request(body, model):
    payload = copy.deepcopy(body)
    payload["model"] = model
    cap = payload.pop("max_completion_tokens", payload.get("max_tokens", 1024))
    payload["max_tokens"] = min(MAX_TOKENS, max(1, int(cap)))
    if payload.get("stream"):
        payload["stream_options"] = {**payload.get("stream_options", {}), "include_usage": True}
    return payload


def sse_objects(response):
    lines = []
    for raw in response:
        line = raw.decode("utf-8").rstrip("\r\n")
        if line.startswith("data:"):
            lines.append(line[5:].lstrip(" "))
        elif not line and lines:
            data = "\n".join(lines)
            lines.clear()
            if data == "[DONE]":
                return
            yield json.loads(data)
    if lines and "\n".join(lines) != "[DONE]":
        yield json.loads("\n".join(lines))


def dial(payload):
    request = urllib.request.Request(
        os.environ["OPENAI_BASE_URL"].rstrip("/") + "/chat/completions",
        data=encode(payload), method="POST", headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + os.environ.get("OPENAI_API_KEY", ""),
            "Accept": "text/event-stream" if payload.get("stream") else "application/json"})
    try:
        with OPENER.open(request, timeout=20) as response:
            # Buffer complete upstream frames before committing client headers. This
            # keeps usage intact and makes failures before delivery recoverable.
            return list(sse_objects(response)) if payload.get("stream") else json.load(response)
    except urllib.error.HTTPError as error:
        try:
            raw = error.read()
            try:
                body = json.loads(raw)
            except ValueError:
                body = {"error": {"message": raw.decode("utf-8", "replace")[:500]}}
            detail = body.get("error", {}) if isinstance(body, dict) else {}
            if not isinstance(detail, dict):
                detail = {}
            raise UpstreamError(error.code, detail.get("code") or detail.get("type") or
                                "http_error", body) from None
        finally:
            error.close()


def eligible_models(models, body, manifest):
    required = dict(manifest.get("requires", {}))
    required["tools"] = bool(required.get("tools") or body.get("tools") or any(
        message.get("tool_calls") or message.get("role") == "tool"
        for message in body["messages"]))
    required["json"] = bool(required.get("json") or
                            body.get("response_format", {}).get("type") in ("json_object", "json_schema"))
    required["vision"] = bool(required.get("vision") or any(
        isinstance(message.get("content"), list) and any(
            block.get("type") in ("image_url", "image") for block in message["content"])
        for message in body["messages"]))
    required["streaming"] = bool(body.get("stream") or required.get("streaming"))
    # Conservative input estimate; the brief's explicit context requirement wins
    # whenever it is larger. All trial prompts are short instructions.
    needed = max(required.get("context_length", 0) or 0,
                 (len(encode(body["messages"])) + 2) // 3 +
                 capped_request(body, "auto")["max_tokens"])
    ranked = sorted(enumerate(models), key=lambda pair: (-pair[1].get("rating_milli", 0), pair[0]))
    result = []
    for _, model in ranked:
        if not model.get("free") or not model["id"].startswith("openrouter/") or not model["id"].endswith(":free"):
            continue
        if any(required.get(key) and not model.get(key) for key in ("tools", "vision", "json")):
            continue
        if required["streaming"] and model.get("streaming", model.get("stream", True)) is False:
            continue
        if (model.get("context_length") or 0) < needed:
            continue
        result.append(model)
    return result


def stop_reason(reason, has_tools=False):
    if has_tools or reason in ("tool_calls", "function_call"):
        return "tool_use"
    if reason == "length":
        return "max_tokens"
    return "end_turn"


def anthropic_message(response, model):
    choice = response["choices"][0]
    message = choice["message"]
    blocks = []
    if message.get("content"):
        blocks.append({"type": "text", "text": message["content"]})
    for call in message.get("tool_calls", []):
        function = call["function"]
        arguments = function.get("arguments", "{}")
        blocks.append({"type": "tool_use", "id": call["id"], "name": function["name"],
                       "input": json.loads(arguments) if isinstance(arguments, str) else arguments})
    usage = response.get("usage") or {}
    return {"id": response["id"], "type": "message", "role": "assistant", "model": model,
            "content": blocks, "stop_reason": stop_reason(choice.get("finish_reason"),
                                                           bool(message.get("tool_calls"))),
            "stop_sequence": None, "usage": {"input_tokens": usage.get("prompt_tokens", 0),
                                             "output_tokens": usage.get("completion_tokens", 0)}}


def json_answer(response, model, anthropic):
    if anthropic:
        return anthropic_message(response, model)
    result = copy.deepcopy(response)
    result["model"] = model
    result.setdefault("object", "chat.completion")
    result.setdefault("created", 0)
    return result


def event(name, value):
    return b"event: " + name.encode() + b"\ndata: " + encode(value) + b"\n\n"


def stream_answer(frames, model, anthropic):
    if not anthropic:
        output = []
        for frame in frames:
            chunk = copy.deepcopy(frame)
            chunk["model"] = model
            chunk.setdefault("object", "chat.completion.chunk")
            chunk.setdefault("created", 0)
            output.append(b"data: " + encode(chunk) + b"\n\n")
        return b"".join(output) + b"data: [DONE]\n\n"
    if not frames:
        raise ValueError("Empty upstream stream")
    usage, finish = {}, None
    # Collect tool metadata before emitting starts: providers can fragment names
    # and IDs across deltas, whereas Messages needs both at block start.
    blocks, order = {}, []
    for frame in frames:
        if frame.get("usage"):
            usage = frame["usage"]
        for choice in frame.get("choices", []):
            if choice.get("index", 0) != 0:
                continue
            finish = choice.get("finish_reason") or finish
            delta = choice.get("delta", {})
            if delta.get("content"):
                key = ("text", 0)
                if key not in blocks:
                    blocks[key] = {"parts": []}
                    order.append(key)
                blocks[key]["parts"].append(delta["content"])
            for call in delta.get("tool_calls", []):
                key = ("tool", call.get("index", 0))
                if key not in blocks:
                    blocks[key] = {"id": "", "name": "", "parts": []}
                    order.append(key)
                block = blocks[key]
                block["id"] += call.get("id", "")
                function = call.get("function", {})
                block["name"] += function.get("name", "")
                if "arguments" in function:
                    block["parts"].append(function["arguments"])
    start = {"id": frames[0]["id"], "type": "message", "role": "assistant", "model": model,
             "content": [], "stop_reason": None, "stop_sequence": None,
             "usage": {"input_tokens": usage.get("prompt_tokens", 0), "output_tokens": 0}}
    output = [event("message_start", {"type": "message_start", "message": start})]
    for index, key in enumerate(order):
        block = blocks[key]
        tool = key[0] == "tool"
        content = ({"type": "tool_use", "id": block["id"], "name": block["name"], "input": {}}
                   if tool else {"type": "text", "text": ""})
        output.append(event("content_block_start", {"type": "content_block_start", "index": index,
                                                     "content_block": content}))
        for part in block["parts"]:
            delta = ({"type": "input_json_delta", "partial_json": part}
                     if tool else {"type": "text_delta", "text": part})
            output.append(event("content_block_delta", {"type": "content_block_delta", "index": index,
                                                         "delta": delta}))
        output.append(event("content_block_stop", {"type": "content_block_stop", "index": index}))
    output.append(event("message_delta", {"type": "message_delta", "delta": {
        "stop_reason": stop_reason(finish, any(key[0] == "tool" for key in order)), "stop_sequence": None},
        "usage": {"output_tokens": usage.get("completion_tokens", 0)}}))
    output.append(event("message_stop", {"type": "message_stop"}))
    return b"".join(output)


def failure_headers(failures):
    if not failures:
        return {}
    return {"x-makefun-upstream-failures": encode(failures[:8]).decode("ascii")}


def serve_completion(handler, original, anthropic, manifest):
    failures = []
    try:
        body = translate_request(original, anthropic)
        candidates = eligible_models(handler.server.models, body, manifest)
        if not candidates:
            return handler.reply(400, {"error": {"type": "invalid_request_error",
                                                "message": "No free model meets the requirements"}})
        # At most 48 calls across the six prompts, below the trial's 60-call cap.
        for attempt in range(8):
            model = candidates[attempt % len(candidates)]["id"]
            try:
                response = dial(capped_request(body, model))
            except UpstreamError as error:
                failures.append({"status": error.status, "code": error.code[:120], "model": model})
                final = error.code in ("request_cap", "budget_exhausted", "model_not_allowed",
                                       "max_tokens_required")
                retryable = error.status in (408, 429) or 500 <= error.status <= 599
                if final or not retryable or attempt == 7:
                    value = error.body
                    if anthropic:
                        value = {"type": "error", "error": {"type": "api_error",
                                 "message": "Gateway refused the request: " + error.code}}
                    return handler.reply(error.status, value, failure_headers(failures))
                continue
            if body.get("stream"):
                answer = stream_answer(response, model, anthropic)
                return handler.reply(200, answer, failure_headers(failures), "text/event-stream")
            return handler.reply(200, json_answer(response, model, anthropic), failure_headers(failures))
    except (urllib.error.URLError, OSError, ValueError, KeyError, TypeError) as error:
        handler.reply(502, {"error": {"type": "api_error", "message":
                                     "Upstream exchange or translation failed: " + type(error).__name__}},
                      failure_headers(failures))


def main():
    models, prompts = load_brief()
    if "--list" in sys.argv or not os.environ.get("MAKEFUN_ENV_URL"):
        sys.stdout.buffer.write(encode(make_catalog(models)) + b"\n")
        return 0
    server = RouterServer(models, prompts)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        registration = server.register()
        if not registration.get("ok"):
            raise RuntimeError("Environment rejected registration")
        wait = threading.Event()
        while True:
            if env_request("/work").get("done") is True:
                break
            wait.wait(0.1)
    finally:
        server.shutdown()
        server.server_close()
        worker.join()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
