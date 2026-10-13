#!/usr/bin/env python3
"""Compile SDK limb macro bounds through monomorphization; never execute them."""
import json
import os
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parents[2]
env = {**os.environ, "JOLT_FIELD_INLINE_MODULUS": "bn254"}
# The rustc consumers must match the Cargo-built library's instrumentation.
for key in ("RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "RUSTC_WRAPPER",
            "RUSTC_WORKSPACE_WRAPPER", "CARGO_BUILD_TARGET"):
    env.pop(key, None)
build = subprocess.run(
    ["cargo", "build", "--locked", "-p", "jolt-sdk", "--lib",
     "--no-default-features", "--features", "field-inline", "--message-format=json"],
    cwd=root, env=env, check=True, capture_output=True, text=True,
)
artifacts = [json.loads(line) for line in build.stdout.splitlines()]
rlib, = [Path(filename) for item in artifacts
         if item.get("reason") == "compiler-artifact"
         and item["target"]["name"] == "jolt_sdk"
         for filename in item["filenames"] if filename.endswith(".rlib")]

cases = [
    ("read_inferred", "let _: [u64; 1] = jolt_sdk::field_to_limbs!(0,);", None),
    ("read_explicit", "let _ = jolt_sdk::field_to_limbs!(16 - 1, 6,);", None),
    ("import", "jolt_sdk::field_from_limbs!(16 - 1, [0u64; 1],);", None),
    ("read_register", "let _ = jolt_sdk::field_to_limbs!(16, 1);",
     "field-inline field register index must be below 16"),
    ("import_register", "jolt_sdk::field_from_limbs!(16, [0u64]);",
     "field-inline field register index must be below 16"),
    ("read_zero", "let _ = jolt_sdk::field_to_limbs!(0, 0);",
     "field_to_limbs! needs at least one limb"),
    ("read_inferred_zero", "let _: [u64; 0] = jolt_sdk::field_to_limbs!(0);",
     "field_to_limbs! needs at least one limb"),
    ("import_zero", "jolt_sdk::field_from_limbs!(0, [0u64; 0]);",
     "field_from_limbs! needs at least one limb"),
]
with tempfile.TemporaryDirectory(prefix="jolt-limb-compile-") as directory:
    for name, body, error in cases:
        source = Path(directory) / f"{name}.rs"
        source.write_text(f"fn main() {{ {body} }}\n")
        result = subprocess.run(
            ["rustc", "--edition=2021", "--emit=obj", str(source),
             "--extern", f"jolt_sdk={rlib}", "-L", f"dependency={rlib.parent / 'deps'}",
             "-o", str(Path(directory) / f"{name}.o")],
            cwd=root, env=env, capture_output=True, text=True,
        )
        if error is None:
            assert result.returncode == 0, result.stderr
        else:
            assert result.returncode != 0, f"{name}: invalid conversion compiled"
            assert "E0080" in result.stderr and error in result.stderr, result.stderr
        print(f"PASS {name}")
