# Commit campaign checkpoint — 2026-09-06 09:24 UTC

Goal active.20% higher useful root-panel throughput is a milestone, not a cap;
it has NOT been achieved. No production optimization accepted or pushed.
Accepted Jolt554f62703/Akita d756e3a67 and parent binary remain unchanged.
Serial M4 Max, no subagents, no Xcode/admin installation, security frozen.

All processes terminal and machine lock released at this checkpoint.
Next: implement D23 staged radix26, preregistered in radix26.md. D22 passes
all arithmetic/output checks and saves1.077s boundary wall, but its8.97% GPU
reduction misses the fixed10% component gate. D23 amortizes digit decoding
at the explicit cost of25% more shared traffic and doubled tile barriers.
Same40persistent source words/thread,349matrix sweeps,16-input bound.
Epoch8 checkpoint09:45UTC or after an already-running fixed cohort.

Latest complete-panel verdicts (all exact output parity, zero swaps/watchdogs):

- D19 column-major: GPU improves1.72%, wall regresses0.253s. Reject.
- D20 explicit widened carry: GPU regresses5.82%. Reject.
- D21 one task/SIMD: GPU improves2.63%, wall saves0.315s. Reject below gate.
- D22 radix26: GPU improves8.97%, wall saves1.077s. Reject below gate.
- D16/D17 exact reuse preprocessing:683ms then289ms, both miss250ms gate.
  Reuse is parked despite the exact12.27% removable-update census.

Parent full-panel replay is now about12.0s, with stable cooled controls;
this is not a newly measured full-proof baseline. No production candidate
or finalist exists, and no new full-proof matrix has been launched.

Important process correction: quarter-second candidate screens showed drift;
continuous8-dispatch batches also failed as the GPU heated and downclocked.
D15 four separately cooled parent observations pass with1.9514% range/mean.
Use separate120s-cooled observations for subsequent panel ranking; do not
promote old~5% pairing/deferred-sign observations or rerun until lucky.

Artifacts and recovery:

- CONTRACT.md: goal/evaluator/resource/integration rules.
- radix26.md: next arithmetic mechanism, proof/cost/gates and claim-to-code map.
- analysis.md: every mechanism, cost model, preregistered gate and result.
- events.jsonl and runs/: append-only events/immutable raw evidence.
- runs/d2-capture/: verified T28 real selectors, selected-zero bitmap/metadata.
- run_saturation.py: guarded GPU diagnostics; run_capture.py: CPU censuses.
- run_panel.py: full-panel cooled ABBA, full192MiB output equality, immutable
  binary hashes. D21 predeclares first-pair futility; acceptance gates unchanged.
- Standalone binaries in bin/ are immutable per diagnostic.
- Isolated Akita /private/tmp/akita-kernel-campaign-20260906 has ONLY opt-in
  capture hook cbd3d5b6f atop d756; onehot.metal remains accepted/unmodified.
- Isolated Jolt /private/tmp/jolt-commit-occupancy-20260906 b2f89f9b9 has ONLY
  local dependency overrides and builds against that fork.

Use modern Python /opt/homebrew/opt/python@3.13/bin/python3.13.
Build diagnostic Objective-C++ with xcrun clang++ -O3 -std=c++17 -fobjc-arc
-Wall -Wextra -Werror -framework Foundation -framework Metal.
Machine lock: ../akita-10mhz-studies/scratch/machine.lock.
macmon: /private/tmp/akita-commit-macmon-build-20260906/release/macmon.

User edits remain untouched: crates/jolt-prover/Cargo.toml, its untracked
examples/onehot_census.rs, and untracked specs/akita-metal-m5-10mhz-pathway.md.
Production nextest/clippy/five-workload validation has not run for any finalist
because no finalist exists. Inherited fork preflight failures are documented
in the earlier ledger; do not claim a green tree or optimal kernel.
