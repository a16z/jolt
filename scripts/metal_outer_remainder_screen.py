#!/usr/bin/env python3
"""Run the exact log-25 OuterRemainder candidate screen."""

from __future__ import annotations

import subprocess
import sys

try:
    from scripts import metal_outer_remainder_eval as evaluator
except ModuleNotFoundError:
    import metal_outer_remainder_eval as evaluator


SCHEMA = "outer_remainder_screen_v1"
SCHEMA_VERSION = 1
LOG_N = 25
PAIRS = 3
STORAGE_BYTES = 2_152_596_208
DENSE_STORAGE_BYTES = 2 * (1 << 30)
MAXIMUM_STORAGE_BUFFER_BYTES = 1 << 30


def configure() -> None:
    evaluator.SCHEMA = SCHEMA
    evaluator.SCHEMA_VERSION = SCHEMA_VERSION
    evaluator.LOG_N = LOG_N
    evaluator.PAIRS = PAIRS
    evaluator.ROUNDS = LOG_N + 1
    evaluator.STORAGE_BYTES = STORAGE_BYTES
    evaluator.DENSE_STORAGE_BYTES = DENSE_STORAGE_BYTES
    evaluator.REMAINING_SEQUENCE_STORAGE_BYTES = (
        STORAGE_BYTES - DENSE_STORAGE_BYTES
    )
    evaluator.MAXIMUM_STORAGE_BUFFER_BYTES = MAXIMUM_STORAGE_BUFFER_BYTES
    screen_path = "scripts/metal_outer_remainder_screen.py"
    if screen_path not in evaluator.SOURCE_PATHS:
        evaluator.SOURCE_PATHS = (*evaluator.SOURCE_PATHS, screen_path)


def main() -> int:
    configure()
    return evaluator.main()


if __name__ == "__main__":
    with evaluator.evaluator_lock({"direct_evaluator": SCHEMA}):
        try:
            raise SystemExit(main())
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            print(f"error: {error}", file=sys.stderr)
            raise SystemExit(2) from error
