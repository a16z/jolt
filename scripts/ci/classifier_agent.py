#!/usr/bin/env python3
"""Run the gate's read-only coding agent without publishing anything to GitHub."""

from __future__ import annotations

import contextlib
import ctypes
import http.server
import json
import os
import re
from pathlib import Path
import shutil
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request

MODEL = "gpt-6.1-sol"
EFFORT = "xhigh"
MODEL_CONTEXT_WINDOW = 1_050_000
MAX_OUTPUT_TOKENS = 128_000
MAX_INPUT_TOKENS = MODEL_CONTEXT_WINDOW - MAX_OUTPUT_TOKENS
MAX_SPEND_USD = 50
WALL_SECONDS = 5 * 60 * 60
MAX_VERDICT_BYTES = 60_000
MAX_FAILURE_CHARS = 512

CLASSIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string", "minLength": 1, "maxLength": 12_000},
        "risks": {"type": "array", "minItems": 1, "maxItems": 5000, "items": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "minLength": 1, "maxLength": 4096},
                "level": {"type": "integer", "enum": [1, 2, 3, 4]},
                "reasoning": {"type": "string", "minLength": 1, "maxLength": 2000},
            },
            "required": ["path", "level", "reasoning"], "additionalProperties": False}},
    },
    "required": ["reasoning", "risks"], "additionalProperties": False,
}


class AgentFailure(Exception):
    """A failed or bounded attempt, never evidence for an approval."""


def failure_text(error: object, key: str) -> str:
    message = str(error)
    return (message.replace(key, "[REDACTED]") if key else message)[:MAX_FAILURE_CHARS]


def validate_verdict(answer: object, paths: set[str]) -> dict:
    def text(value: object, limit: int) -> bool:
        return isinstance(value, str) and 0 < len(value) <= limit and bool(value.strip()) \
            and all(ord(char) >= 32 or char in "\n\t" for char in value)

    if len(json.dumps(answer).encode()) > MAX_VERDICT_BYTES or not isinstance(answer, dict) \
            or set(answer) != {"reasoning", "risks"} \
            or not text(answer["reasoning"], 12_000) or not isinstance(answer["risks"], list) \
            or not 0 < len(answer["risks"]) <= 5000:
        raise AgentFailure("invalid classifier verdict schema")
    seen = set()
    for item in answer["risks"]:
        if not isinstance(item, dict) or set(item) != {"path", "level", "reasoning"} \
                or not text(item["path"], 4096) or item["path"] not in paths or item["path"] in seen \
                or type(item["level"]) is not int or item["level"] not in range(1, 5) \
                or not text(item["reasoning"], 2000):
            raise AgentFailure("invalid or duplicate per-path risk evidence")
        seen.add(item["path"])
    if not paths or seen != paths:
        raise AgentFailure("classifier did not account for every changed path")
    return answer


def request_cost(input_tokens: int, output_tokens: int, cached: int = 0, written: int = 0, *, long_context: bool = False) -> float:
    # https://developers.openai.com/api/docs/models/gpt-6.1-sol#pricing
    input_rate, output_rate = (4, 15) if long_context or input_tokens > 272_000 else (2, 10)
    fresh = input_tokens - cached - written
    return ((fresh + cached * .05 + written * 1.25) * input_rate + output_tokens * output_rate) / 1_000_000


class ModelBroker:
    """Keep the API key in a non-dumpable process; forward model traffic, not tool calls."""

    def __init__(self, key: str, request=None) -> None:
        if ctypes.CDLL(None, use_errno=True).prctl(4, 0, 0, 0, 0) != 0:
            raise AgentFailure("cannot make the credential broker non-dumpable")
        self.key = key
        self.started = time.monotonic()
        self.inflight = False
        self.request = request or self.upstream
        self.input_tokens = self.output_tokens = 0
        self.cost_usd = 0.0
        self.usage: list[dict] = []
        self.failure: str | None = None
        self.lock = threading.Lock()

    def upstream(self, endpoint: str, body: dict, headers: dict) -> tuple[int, bytes]:
        request = urllib.request.Request("https://api.openai.com" + endpoint,
            json.dumps(body).encode(), {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=max(1, WALL_SECONDS - (time.monotonic() - self.started))) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as error:
            with error:
                return error.code, error.read(64 * 1024)

    def request_failure(self, message: str, status: int, payload: bytes) -> AgentFailure:
        try:
            response = json.loads(payload)
        except ValueError:
            response = None
        error = response.get("error") if isinstance(response, dict) else None
        details = []
        for name in ("type", "code"):
            if isinstance(error, dict) and isinstance(error.get(name), str):
                value = json.dumps(failure_text(error[name], self.key))[:128]
                details.append(f"{name}={value}")
        suffix = ", " + ", ".join(details) if details else ""
        return AgentFailure(failure_text(f"{message} (HTTP {status}{suffix})", self.key))

    def forward(self, body: dict, headers: dict) -> tuple[int, bytes]:
        with self.lock:
            if self.failure:
                raise AgentFailure(self.failure)
            if body.get("model") != MODEL:
                raise AgentFailure("unexpected requested model")
            body["max_output_tokens"] = MAX_OUTPUT_TOKENS
            body["service_tier"] = "default"
            count_body = {key: value for key, value in body.items() if key in
                ("model", "input", "instructions", "previous_response_id", "tools", "tool_choice", "text", "reasoning")}
            status, counted = self.request("/v1/responses/input_tokens", count_body, {})
            if status != 200:
                raise self.request_failure("model token preflight failed", status, counted)
            count = json.loads(counted).get("input_tokens")
            output = body["max_output_tokens"]
            if type(count) is not int or count < 0:
                raise AgentFailure("invalid classifier input token count")
            if count > MAX_INPUT_TOKENS:
                raise AgentFailure("classifier per-request input cap")
            if self.cost_usd + request_cost(count, output, written=count, long_context=True) > MAX_SPEND_USD:
                raise AgentFailure("classifier spend cap")
            self.inflight = True
            status, payload = self.request("/v1/responses", body, {})
            if status != 200:
                raise self.request_failure("model request failed", status, payload)
            events = [json.loads(line[6:]) for line in payload.splitlines() if line.startswith(b"data: {")] \
                if body.get("stream") else [json.loads(payload)]
            completed = [event["response"] for event in events if event.get("type") in ("response.completed", "response.incomplete")]
            if not completed:
                raise AgentFailure("model response did not complete")
            response = completed[-1]
            usage = response.get("usage", {})
            if not all(type(usage.get(key)) is int for key in ("input_tokens", "output_tokens")):
                raise AgentFailure("model reported no token usage")
            if not re.fullmatch(re.escape(MODEL) + r"(-\d{4}-\d{2}-\d{2})?", response.get("model", "")):
                raise AgentFailure("unexpected served model")
            details = usage.get("input_tokens_details", {})
            cached = details.get("cached_tokens", 0)
            written = details.get("cache_write_tokens", 0)
            self.input_tokens += usage["input_tokens"]
            self.output_tokens += usage["output_tokens"]
            self.cost_usd += request_cost(usage["input_tokens"], usage["output_tokens"], cached, written)
            self.usage.append(usage)
            self.inflight = False
            if response.get("status") != "completed":
                raise AgentFailure("model response reached its output cap")
            return status, payload

    @contextlib.contextmanager
    def serve(self):
        broker = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_POST(self):
                try:
                    size = int(self.headers.get("Content-Length", "0"))
                    if self.path.partition("?")[0] != "/v1/responses" or not 0 < size <= 12_000_000:
                        raise AgentFailure("unsupported model request")
                    status, payload = broker.forward(json.loads(self.rfile.read(size)), {})
                except Exception as error:
                    broker.failure = failure_text(error, broker.key)
                    status = 429
                    payload = b'{"type":"error","error":{"type":"rate_limit_error","message":"classifier stopped"}}'
                self.send_response(status)
                self.send_header("Content-Type", "text/event-stream" if payload.startswith(b"event:") else "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.daemon_threads = True
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}"
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


def configure_codex(endpoint: str, directory: Path) -> None:
    ripgrep = directory / "ripgrep.conf"
    ripgrep.write_text("--no-ignore\n--hidden\n")
    rg = shutil.which("rg")
    if rg is None:
        raise AgentFailure("ripgrep is not installed")
    environment = {"PATH": f"{Path(rg).parent}:/usr/local/bin:/usr/bin:/bin", "HOME": str(directory / "agent-state"),
        "RIPGREP_CONFIG_PATH": str(ripgrep),
        "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null",
        "GIT_ATTR_NOSYSTEM": "1", "GIT_CONFIG_COUNT": "4",
        "GIT_CONFIG_KEY_0": "core.fsmonitor", "GIT_CONFIG_VALUE_0": "false",
        "GIT_CONFIG_KEY_1": "core.hooksPath", "GIT_CONFIG_VALUE_1": "/dev/null",
        "GIT_CONFIG_KEY_2": "diff.external", "GIT_CONFIG_VALUE_2": "/dev/null",
        "GIT_CONFIG_KEY_3": "core.attributesFile", "GIT_CONFIG_VALUE_3": "/dev/null"}
    toml_env = ",".join(f"{json.dumps(key)}={json.dumps(value)}" for key, value in environment.items())
    config = f'''
model = "{MODEL}"
model_reasoning_effort = "{EFFORT}"
model_context_window = {MODEL_CONTEXT_WINDOW}
model_provider = "gate"
approval_policy = "never"
allow_login_shell = false
project_doc_max_bytes = 0
web_search = "disabled"
model_instructions_file = {json.dumps(str(directory / "instructions.md"))}
[shell_environment_policy]
inherit = "none"
set = {{ {toml_env} }}
[model_providers.gate]
name = "Gate model broker"
base_url = "{endpoint}/v1"
wire_api = "responses"
request_max_retries = 0
stream_max_retries = 0
stream_idle_timeout_ms = {WALL_SECONDS * 1000}
[features]
shell_snapshot = false
hooks = false
plugins = false
apps = false
multi_agent = false
view_image = false
enable_request_compression = false
'''
    (directory / "agent-state/config.toml").write_text(config)


def run_agent(directory: Path, key: str, prompt: str | None = None) -> tuple[dict | None, dict]:
    plan = json.loads((directory / "plan.json").read_text())
    state = directory / "agent-state"
    state.mkdir()
    (state / "rules").mkdir()
    (state / "rules/gate.rules").write_text('prefix_rule(pattern=[["cargo", "rustc", "make", "cmake", "ninja"]], decision="forbidden", justification="Read source with git diff, rg and sed; never build PR code.")\n')
    broker = ModelBroker(key)
    start = time.monotonic()
    answer = None
    failure = None
    try:
        if not key:
            raise AgentFailure("OPENAI_API_KEY is not set")
        schema = state / "schema.json"
        schema.write_text(json.dumps(CLASSIFICATION_SCHEMA))
        with broker.serve() as endpoint:
            configure_codex(endpoint, directory)
            command = [shutil.which("codex") or "/missing-codex", "exec", "--sandbox", "read-only", "--strict-config",
                "--ephemeral", "--skip-git-repo-check", "--cd", str(directory / "workspace"),
                "--json", "--output-schema", str(schema),
                "--output-last-message", str(state / "answer.json"), "-"]
            with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  start_new_session=True, env={"PATH": os.environ["PATH"], "HOME": str(state), "CODEX_HOME": str(state),
                                       "OPENAI_API_KEY": "FAKE_CLASSIFIER_KEY"}) as process:
                try:
                    output, stderr = process.communicate((prompt or
                        "Classify change.json. Start with diffstat and path triage; read production "
                        "hunks and trace callers according to the trusted skill. Use git diff --no-index "
                        "--no-ext-diff --no-textconv -- base head, rg and sed ranges. Do not read the entire "
                        "diff or test files. Return only the structured verdict.").encode(), timeout=WALL_SECONDS)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.communicate()
                    raise AgentFailure("classifier wall-time cap") from None
            (directory / "agent-events.jsonl").write_bytes(output)
            (directory / "agent-stderr.txt").write_bytes(stderr)
            if broker.failure:
                raise AgentFailure(broker.failure)
            if process.returncode:
                raise AgentFailure(f"read-only agent exited {process.returncode}")
            answer = validate_verdict(json.loads((state / "answer.json").read_text()), set(plan["paths"]))
    except (AgentFailure, OSError, ValueError, subprocess.CalledProcessError) as error:
        failure = failure_text(error, key)
    receipt = {"wall_seconds": time.monotonic() - start, "input_tokens": broker.input_tokens,
               "output_tokens": broker.output_tokens, "cost_usd": broker.cost_usd,
               "requests": len(broker.usage), "usage": broker.usage, "usage_complete": not broker.inflight, "failure": failure}
    (directory / "result.json").write_text(json.dumps(answer))
    (directory / "receipt.json").write_text(json.dumps(receipt))
    return answer, receipt


if __name__ == "__main__":
    import sys
    _, receipt = run_agent(Path(sys.argv[1]).resolve(), os.environ.get("OPENAI_API_KEY", ""))
    print(json.dumps(receipt))
