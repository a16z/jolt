#!/usr/bin/env python3
"""Check entry-point selection when guest builds share Cargo artifacts."""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-dir", type=Path, default=root / "target/guest-function-cache")
    args = parser.parse_args()
    target = args.target_dir.resolve()
    command = ["jolt", "build", "-p", "multi-function-guest", "--", "--locked",
               "--profile", "ci", "--features", "guest", "--target-dir", str(target)]
    elf = target / "riscv64imac-unknown-none-elf/ci/multi-function-guest"
    hashes = []
    for function in ["add", "mul", "add", None, "add"]:
        environment = os.environ.copy()
        environment.pop("JOLT_FUNC_NAME", None)
        if function is not None:
            environment["JOLT_FUNC_NAME"] = function
        result = subprocess.run(command, cwd=root, env=environment, capture_output=True, text=True)
        if function is None:
            if result.returncode == 0:
                raise AssertionError("unsetting the selector reused a cached entry point")
            # With both provable functions selected, the guest has duplicate entry points.
            if "defined multiple times" not in result.stderr:
                raise AssertionError(result.stderr)
            print("unset selector: duplicate entry points rejected")
            continue
        if result.returncode:
            raise RuntimeError(result.stderr)
        digest = hashlib.sha256(elf.read_bytes()).hexdigest()
        hashes.append(digest)
        print(f"{function}: {digest}")
    add, mul, add_again, add_after_unset = hashes
    assert add != mul, "switching the selector must produce the other function's ELF"
    assert add == add_again == add_after_unset, "reselecting add must recover its original ELF"


if __name__ == "__main__":
    main()
