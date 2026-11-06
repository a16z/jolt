#!/usr/bin/env python3
import json
import os
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parents[2]
(root / "target").mkdir(exist_ok=True)
env = {**os.environ, "JOLT_FUNC_NAME": "fib"}
with tempfile.TemporaryDirectory(prefix="guest-layout-cache-", dir=root / "target") as directory:
    target = Path(directory)
    elf = target / "riscv64imac-unknown-none-elf/ci/fibonacci-guest"
    command = ["jolt", "build", "-p", "fibonacci-guest", "--backtrace", "frame-pointers"]
    for stack, repeat in [(1 << 20, False), (2 << 20, False), (2 << 20, True)]:
        result = subprocess.run(
            [*command, "--stack-size", str(stack), "--",
             "--locked", "--profile", "ci", "--features", "guest",
             "--target-dir", str(target), "--message-format=json"],
            cwd=root, env=env, capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
        symbols = subprocess.run(["readelf", "--symbols", "--wide", str(elf)],
                                 check=True, capture_output=True, text=True).stdout
        bounds = {fields[-1]: int(fields[1], 16)
                  for line in symbols.splitlines()
                  if len(fields := line.split()) >= 8 and fields[-1] in
                  {"__stack_bottom", "__stack_top"}}
        span = bounds["__stack_top"] - bounds["__stack_bottom"]
        assert span == stack + 128, f"requested stack {stack} + 128-byte canary, got {span}"
        if repeat:
            artifacts = [item for line in result.stdout.splitlines()
                         if (item := json.loads(line)).get("reason") == "compiler-artifact"]
            assert artifacts and all(item["fresh"] for item in artifacts), result.stderr
            print(f"PASS unchanged layout: all {len(artifacts)} compiler artifacts fresh", flush=True)
        print(f"PASS stack={stack} repeat={repeat}", flush=True)
