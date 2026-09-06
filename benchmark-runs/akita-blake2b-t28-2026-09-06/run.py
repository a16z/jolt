#!/usr/bin/env python3
"""The preregistered BLAKE2b observations, using the frozen observer."""
import json
from pathlib import Path
import re
import signal
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "akita-commit-occupancy-2026-09-06"))
from run_saturation import LOCK, matrix


def main():
    manifest = json.loads((ROOT / "manifest.json").read_text())
    if (ROOT / "started.json").exists():
        raise RuntimeError("pair already started; inspect immutable evidence before resuming")
    matrix.ROOT = ROOT
    matrix.EVENTS = ROOT / "events.jsonl"
    deadline = manifest["deadline_epoch"]
    sequence = manifest["sequence"]
    if sequence not in (["candidate"], ["candidate", "parent"]):
        raise RuntimeError("unexpected preregistered sequence")
    for filename, expected in manifest["frozen_files"].items():
        if matrix.digest(Path(filename)) != expected:
            raise RuntimeError("frozen fingerprint changed: " + filename)
    LOCK.mkdir()
    try:
        with (ROOT / "started.json").open("x") as output:
            json.dump(dict(epoch=time.time(), manifest_sha256=matrix.digest(ROOT / "manifest.json")), output)
        results = []
        for attempt, variant in enumerate(sequence, 1):
            if time.time() + 300 > deadline:
                raise RuntimeError("insufficient reserve for a cooled proof")
            selected = dict(manifest[variant], worktree=manifest["worktree"],
                cargo_target_dir=manifest["cargo_target_dir"])
            matrix.record(dict(event="blake2b_cell", variant=variant,
                binary_sha256=matrix.digest(Path(selected["binary"]))))
            result = matrix.observe(selected, "blake2b-chain", 28, "metal", attempt, 120, deadline)
            raw = (ROOT / result["raw"]).read_text()
            checks = re.findall(r"^BLAKE2B_CHAIN_CHECK iterations=(\d+) digest=([0-9a-f]+) value=true$", raw, re.M)
            if checks != [(str(manifest["iterations"]), manifest["expected_digest"])]:
                raise RuntimeError("BLAKE2b independent output or iteration identity mismatch")
            result["variant"] = variant
            results.append(result)
            for filename, expected in manifest["frozen_files"].items():
                if matrix.digest(Path(filename)) != expected:
                    raise RuntimeError("post-proof fingerprint changed: " + filename)
        if len({result["trace_len"] for result in results}) != 1:
            raise RuntimeError("parent/candidate trace lengths differ")
        candidate = next(result for result in results if result["variant"] == "candidate")
        parent = next((result for result in results if result["variant"] == "parent"), None)
        summary = dict(results=results,
            saving_s=parent["prove_s"]-candidate["prove_s"] if parent else None,
            candidate_m5_projected_mhz=1.13*candidate["padded_mhz"],
            projection_factor=1.13, observations_per_variant=1)
        with (ROOT / "result.json").open("x") as output:
            json.dump(summary, output, indent=2)
        matrix.record(dict(event="blake2b_pair_complete", **summary))
    except BaseException as error:
        matrix.record(dict(event="blake2b_pair_stopped", reason=str(error)))
        raise
    finally:
        LOCK.rmdir()


if __name__ == "__main__":
    def stop(_signum, _frame):
        raise KeyboardInterrupt("BLAKE2b pair interrupted")
    signal.signal(signal.SIGTERM, stop)
    main()
