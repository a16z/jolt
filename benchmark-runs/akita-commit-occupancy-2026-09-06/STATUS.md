# Commit campaign — accepted local result, 2026-09-06 10:50 UTC

Registered performance/security/validation requirements are met. D23 staged
radix26 is accepted as the local campaign parent, Akita7878e5ba1; isolated
Jolt b2f89f9b9 supplies dependency-path integration only. No push or default
production-pin change. User edits are untouched. No process or lock remains.
Goal closure follows the durable acceptance checkpoint.

- Full-panel GPU11.993870 -> 9.570013s:25.3276% higher useful throughput,
  20.2091% less GPU time,2.425245s complete-boundary saving.
- Verified Fibonacci ABBA32.749368 -> 30.504165s:2.245203s saving,
  6.8557% wall reduction; same-session parent drift1.5133%.
- All five workload comparisons improve. Measured M4 mean padded MHz
  8.415475741 -> 8.909192209,+5.866768%. Four transfers use single pairs.
- Frozen1.13M5 projection10.067387MHz:thin0.674%margin,not measured M5
  and not a robust margin guarantee.
- All12scored proofs verify,zero swaps/watchdogs,max RSS86.934586GiB.
- Akita Metal32/32,Jolt Metal329/329,PCS23/23 in each serial/parallel
  mode; all three fork clippy modes and both Jolt modes pass. Final fmt,
  scoped-diff and frozen source/binary/guest fingerprints pass.

Arithmetic, protocol, field, parameters, catalogs, transcript and verifier
are unchanged. See radix26.md for the signed bound and acceptance.md for
mechanism, all gates, results and limitations. Inherited preflight failures
and unavailable occupancy/ISA data remain explicit; no optimality claim,
independent security audit or new constant-time guarantee.

Immutable runnable artifact:
bin/modular_benchmark_radix26_7878e5ba1
SHA216737e3d08ab90b6bbdbdfce6858ae733862c6d1922f8ed65ab2ec9f2907e20.
Source:/private/tmp/akita-kernel-campaign-20260906
Jolt:/private/tmp/jolt-commit-occupancy-20260906
Default Jolt554f62703/Akita d756e3a67 production selection remains unchanged;
later main Jolt commits contain only local campaign evidence/tools.

Recovery:read CONTRACT.md,acceptance.md and the last events.jsonl entry.
The original finalist/ parent-drift stop is preserved. finalist-rebased/
contains the complete fresh comparison; its result.json is the unmodified
timing-time snapshot, not the final acceptance decision. No candidate proof
was selectively retried. The extra calibration preceded candidate timings.

Do not relaunch rejected mechanisms without new causal data. No untested
preregistered candidate remains; a12-position staged tile is only an
unpriced future idea, not an expected additive gain. Any further search
needs a new bounded epoch from the accepted local parent.20% was not a
ceiling:the accepted result reaches25.3% useful panel throughput.

Serial M4 Max,no subagents,no Xcode/admin installation,security frozen.
Modern Python:/opt/homebrew/opt/python@3.13/bin/python3.13.
Machine lock:../akita-10mhz-studies/scratch/machine.lock.
Preserved user edits:crates/jolt-prover/Cargo.toml,examples/onehot_census.rs,
specs/akita-metal-m5-10mhz-pathway.md and the separate Akita user's changes.
