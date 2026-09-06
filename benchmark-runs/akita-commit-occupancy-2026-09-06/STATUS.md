# Commit campaign checkpoint — 2026-09-06 09:36 UTC

Goal active.20% higher useful root-panel throughput is a milestone, not a cap;
the COMPONENT milestone now passes, but the full goal has NOT been achieved.
No production optimization accepted or pushed.
Accepted Jolt554f62703/Akita d756e3a67 and parent binary remain unchanged.
Serial M4 Max, no subagents, no Xcode/admin installation, security frozen.

All processes terminal and machine lock released at this checkpoint.
Next: isolated production integration of D23, preregistered in analysis.md.
D23 GPU11.994->9.570s,25.33% higher useful throughput,2.425s complete-wall
saving. Exact arithmetic/output checks pass, parent drift0.7%, no swaps or
watchdogs. Full proofs and transfer remain unverified. Preserve registered
shape alignment16 while using internal8-position/20KiB tiles; remove the
capture hook. No new mechanism before this finalist is evaluated.
Epoch8 transaction3 integration/targeted validation; checkpoint10:05UTC.

Latest complete-panel verdicts (all exact output parity, zero swaps/watchdogs):

- D19 column-major: GPU improves1.72%, wall regresses0.253s. Reject.
- D20 explicit widened carry: GPU regresses5.82%. Reject.
- D21 one task/SIMD: GPU improves2.63%, wall saves0.315s. Reject below gate.
- D22 radix26: GPU improves8.97%, wall saves1.077s. Reject below gate.
- D23 staged radix26: GPU improves20.21%, wall saves2.425s. Component pass.
- D16/D17 exact reuse preprocessing:683ms then289ms, both miss250ms gate.
  Reuse is parked despite the exact12.27% removable-update census.

Parent full-panel replay is now about12.0s, with stable cooled controls;
this is not a newly measured full-proof baseline. No production candidate
accepted optimization exists. D23 is a component finalist; no new full-proof
matrix has been launched.

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
