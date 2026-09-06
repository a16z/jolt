# D23 production validation

Candidate Akita7878e5ba1 atop accepted d756e3a67; isolated Jolt b2f89f9b9
has only dependency-path overrides atop accepted554f62703. No accepted
branch/pin or user worktree changes. No protocol/security parameter change.

2026-09-06 09:42UTC: `cargo nextest run -p akita-metal --lib --test-threads 1
--cargo-quiet`, dev profile, parallel defaults, exclusive machine lock,
CARGO_TARGET_DIR=/Users/mgeorghiades/worktrees/akita-feat-metal/target.
Nextest run12afc97a-a335-48fe-a553-d46fe5ac2394:32/32 pass,0skipped,19.295s
test execution after compilation. All four D128 independent CPU/Metal cases,
shape rejection and D512 parity pass. Output is in the tool execution record
(sessions42912); no separate raw file was captured for this one check.
`cargo fmt --all` and `git diff --check` pass. RTK is unavailable, so direct
nextest is used. Subsequent checks use run_validation.py and immutable logs.

Remaining: release build, Jolt Metal suite, affected PCS fp128 E2E/commitment
contract/protocol-soundness targets under both serial/parallel feature graphs,
both Jolt clippy modes, all three fork clippy configurations, final fmt,
same-session full proofs and five-workload transfer. No acceptance claim.

09:43UTC: first release-build controller failed before spawning Cargo because
the supplied modern-Python PATH omitted /opt/homebrew/opt/rustup/bin. No
compilation or GPU work ran; preserve finalist-build.out and failure events.
Resolve Cargo explicitly and add rustup to child PATH. The one corrected
launch uses finalist-build-resolved-cargo.out; candidate source unchanged.

09:48UTC: corrected frozen release build passes in254.51s, peak sampled
family RSS11.38GB; rawSHA1e600f74f35ef45875280c59fb936feecf81cae365ac90379152b2dfbd72d3ac.
Candidate binary bin/modular_benchmark_radix26_7878e5ba1 SHA256
216737e3d08ab90b6bbdbdfce6858ae733862c6d1922f8ed65ab2ec9f2907e20.
All five AOT guest SHA256 values match the original matrix manifest. Production
normalizer and final reducer bodies exactly match the independently validated
diagnostic after removing comments/whitespace; no diagnostic kernel is shipped.
Jolt serial Metal nextest started09:49UTC, not yet complete at this entry.
