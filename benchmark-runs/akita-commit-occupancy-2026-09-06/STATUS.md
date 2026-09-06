# Commit campaign checkpoint — 2026-09-06 10:22 UTC

Goal active.20% higher useful root-panel throughput is a milestone, not a cap;
the COMPONENT milestone now passes, but the full goal has NOT been achieved.
No production optimization accepted or pushed.
Accepted Jolt554f62703/Akita d756e3a67 and parent binary remain unchanged.
Serial M4 Max, no subagents, no Xcode/admin installation, security frozen.

Rebased finalist proof controller now owns the machine lock. Do not start another
GPU run, build, or heavy CPU check while it runs. run_rebased_finalist.py adds
one parent-only stability condition to the unchanged twelve-proof sequence,
180s process/40minute epoch/120s cooling guards; deadline10:40:25UTC.
Next: finish/score two transfer pairs, then PCS feature-mode/clippy/fmt checks.
Live exec session57818, controller PID15605; no other jobs may contend.
Fibonacci ABBA passes: parent32.501577250/32.997158500s, candidate
30.615382291/30.392947500s. Means32.749367875->30.504164896s,2.245203s
saving (6.8557%),parent drift1.5133%,all four verified,zero swaps/watchdogs.
BTreeMap pair verifies27.610286->27.089531s,0.520755s saving. SHA2 pair
verifies29.383833->27.236926s,2.146907s saving. SHA3 parent cooling began
10:21:31UTC. Remaining four transfer proofs
retain original historical5% discrepancy guards and may stop for reassessment.
D23 GPU11.994->9.570s,25.33% higher useful throughput,2.425s complete-wall
saving. Exact arithmetic/output checks pass, parent drift0.7%, no swaps or
watchdogs. Fibonacci full-proof timing passes; transfer remains unverified. Preserve registered
shape alignment16 while using internal8-position/20KiB tiles; the capture
hook has been removed. No new mechanism before this finalist is evaluated.
Epoch8 completed: production Akita7878e5ba1,32/32Akita Metal parity tests,
329/329Jolt Metal tests, release build passes. Epoch9 is the reserved finalist
matrix stopped after one verified parent32.835085250s,9.25% faster than the
historical36.182954209s. No candidate proof ran. All identity/hash checks pass;
cause unproven. The one preregistered extra parent must agree within3% before
any candidate. That extra parent now verifies32.501577250s,1.016% from the
calibration control:PASS. All four new Fibonacci pair observations now pass. New artifacts:
finalist-rebased/; original epoch preserved.
Combined amended reserve13proofs/45minutes measurement+cooling execution.

Latest complete-panel verdicts (all exact output parity, zero swaps/watchdogs):

- D19 column-major: GPU improves1.72%, wall regresses0.253s. Reject.
- D20 explicit widened carry: GPU regresses5.82%. Reject.
- D21 one task/SIMD: GPU improves2.63%, wall saves0.315s. Reject below gate.
- D22 radix26: GPU improves8.97%, wall saves1.077s. Reject below gate.
- D23 staged radix26: GPU improves20.21%, wall saves2.425s. Component pass.
- D16/D17 exact reuse preprocessing:683ms then289ms, both miss250ms gate.
  Reuse is parked despite the exact12.27% removable-update census.

Parent full-panel replay is now about12.0s, with stable cooled controls;
this is not a newly measured full-proof baseline. No production optimization
is accepted. D23's frozen full-proof matrix is now in flight.

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
- Isolated Akita /private/tmp/akita-kernel-campaign-20260906 is7878e5ba1;
  only D128 shader/resource/shape-naming changes versus d756, no capture hook.
- Isolated Jolt /private/tmp/jolt-commit-occupancy-20260906 b2f89f9b9 has ONLY
  local dependency overrides and builds against that fork.

Use modern Python /opt/homebrew/opt/python@3.13/bin/python3.13.
Build diagnostic Objective-C++ with xcrun clang++ -O3 -std=c++17 -fobjc-arc
-Wall -Wextra -Werror -framework Foundation -framework Metal.
Machine lock: ../akita-10mhz-studies/scratch/machine.lock.
macmon: /private/tmp/akita-commit-macmon-build-20260906/release/macmon.

User edits remain untouched: crates/jolt-prover/Cargo.toml, its untracked
examples/onehot_census.rs, and untracked specs/akita-metal-m5-10mhz-pathway.md.
Akita Metal32/32 and Jolt Metal329/329 pass. PCS feature-mode and clippy gates
remain; the five-workload validation is in flight. Inherited fork preflight
failures remain documented; do not claim a green tree or optimal kernel.
