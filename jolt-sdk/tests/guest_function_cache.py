#!/usr/bin/env python3
"""Switching JOLT_FUNC_NAME in a shared Cargo target dir must rebuild the guest ELF."""
import hashlib
import os
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[2]
target = root / "target/guest-function-cache"
elf = target / "riscv64imac-unknown-none-elf/ci/multi-function-guest"
command = ["jolt", "build", "-p", "multi-function-guest", "--", "--locked",
           "--profile", "ci", "--features", "guest", "--target-dir", str(target)]

hashes = []
for function in ["add", "mul", "add"]:
    env = {**os.environ, "JOLT_FUNC_NAME": function}
    subprocess.run(command, cwd=root, env=env, check=True)
    hashes.append(hashlib.sha256(elf.read_bytes()).hexdigest())
    print(f"{function}: {hashes[-1]}")

add, mul, add_again = hashes
assert add != mul, "switching the selector must rebuild the guest"
assert add == add_again, "reselecting add must reproduce its ELF"
