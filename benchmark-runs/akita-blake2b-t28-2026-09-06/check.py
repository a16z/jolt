#!/usr/bin/env python3
"""Reuse the campaign's bounded check observer for this isolated benchmark."""
import sys
import time
import signal
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "akita-commit-occupancy-2026-09-06"))
import run_validation as validation

validation.ROOT = ROOT
validation.matrix.EVENTS = ROOT / "events.jsonl"
validation.record = lambda event, **fields: validation.matrix.record(dict(event=event, **fields))
WORKTREE = Path("/private/tmp/jolt-blake2b-t28-20260906")
TARGET = "/Users/mgeorghiades/worktrees/jolt/lever-c-jolt/target"
FEATURES = "prover-fixtures,metal,profiling"
validation.STEPS = {
    "prepare": (WORKTREE, TARGET, ["cargo", "run", "--release", "-p", "jolt-host",
        "--example", "blake2b_prepare"]),
    "build-candidate": (WORKTREE, TARGET, ["cargo", "build", "--release", "-p", "jolt-prover",
        "--example", "modular_benchmark", "--features", FEATURES]),
    "build-parent": (WORKTREE, TARGET, ["cargo", "build", "--release", "-p", "jolt-prover",
        "--example", "modular_benchmark", "--features", FEATURES]),
    "clippy": (WORKTREE, TARGET, ["cargo", "clippy", "--release", "-p", "jolt-prover",
        "--example", "modular_benchmark", "--features", FEATURES, "--", "-D", "warnings"]),
}
validation.STEPS["build-candidate-format"] = validation.STEPS["build-candidate"]
validation.STEPS["clippy-format"] = validation.STEPS["clippy"]

if __name__ == "__main__":
    def stop(_signum, _frame):
        raise KeyboardInterrupt("BLAKE2b check interrupted")
    signal.signal(signal.SIGTERM, stop)
    (ROOT / "runs").mkdir(exist_ok=True)
    deadline_index = sys.argv.index("--deadline-epoch") + 1
    sys.argv[deadline_index] = str(min(float(sys.argv[deadline_index]), time.time() + 720))
    raise SystemExit(validation.main())
