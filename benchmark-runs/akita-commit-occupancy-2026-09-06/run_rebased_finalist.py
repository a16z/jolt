#!/usr/bin/env python3
"""One preregistered parent-stability amendment; all cohort gates are reused."""
import signal

import run_finalist as cohort

observe = cohort.matrix.observe


def observe_with_parent_stability(manifest, workload, scale, backend, attempt, cooldown, deadline):
    result = observe(manifest, workload, scale, backend, attempt, cooldown, deadline)
    if workload == "fibonacci" and attempt == 1:
        if abs(result["prove_s"] / 32.835085250 - 1) > 0.03:
            raise RuntimeError("additional parent calibration exceeds three-percent stability band")
        cohort.record("parent_calibration_pass", previous_s=32.835085250,
            current_s=result["prove_s"], discarded_candidate_observations=0)
    return result


if __name__ == "__main__":
    def stop(_signum, _frame):
        raise KeyboardInterrupt("rebased finalist controller interrupted")
    signal.signal(signal.SIGTERM, stop)
    cohort.DIRECTORY = cohort.ROOT / "finalist-rebased"
    cohort.matrix.observe = observe_with_parent_stability
    raise SystemExit(cohort.main() or 0)
