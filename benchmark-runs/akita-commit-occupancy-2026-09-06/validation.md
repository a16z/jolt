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

09:50UTC: Jolt Metal nextest329/329 pass,0skipped,9.659s test execution,
52.925s total; rawSHA338b5243398807dbf74295415ca452935190b3adf4daec96fd7ba84edc81a4ed.
The twelve-proof finalist epoch starts after this gate; its frozen manifest,
contract and immutable per-proof logs live in finalist/.

The original proof epoch stopped after one verified parent32.835085250s on
historical drift; no candidate was run. The preregistered additional parent
in finalist-rebased/ verifies32.501577250s,1.016% from the calibration control,
passing its3% band. RawSHA86fe93de73c79f399432e27906eb9930a147fab9ad1a183207564437c7818835.
All13-proof amended-budget records remain visible; the original parent is
calibration only. The candidate comparison uses the new ABBA controls.

10:11UTC Fibonacci ABBA passes:

| Order | Variant | prove_s | Raw SHA256 |
|---|---|---:|---|
|1|Parent|32.501577250|86fe93de73c79f399432e27906eb9930a147fab9ad1a183207564437c7818835|
|2|Candidate|30.615382291|94256c0c5453b15d19c9a8d40de9d548a0e59a19b6573e19385584bf5d82904d|
|3|Candidate|30.392947500|912219dff619d244f474f5e6a4ae73d978c5161e1565ce9b2ba0e68b1fff262a|
|4|Parent|32.997158500|342718713997c6a910016ecc7ecd948fe6166bccb96fa8ec9353cb6c1ee1bf62|

Means32.749367875->30.504164896s, saving2.245203s (6.8557%wall,
7.3603%throughput). Parent drift1.5133%, below3%; all four exact proof markers,
trace_len201327593, T28, zero swaps/watchdogs. RSS~83.7GiB in both variants.
The historical36.183s parent is not used in these gains. Four workload
transfers and remaining PCS/clippy/fmt gates are still pending; NOT accepted.

10:16UTC BTreeMap one-pair transfer:27.610285583->27.089530625s,
saving0.520755s (1.886%wall). Both verify with trace_len177115820,T28,
zero swaps/watchdogs and~81.6GiB RSS. This is a single descriptive pair,
not a confidence interval. Smaller than Fibonacci; do not extrapolate the
Fibonacci saving uniformly. Raw parent/candidate SHA256:
327517ef697788f4adc9da5bea342cca43099c03b26ef86184c31d837933695a
6a9faf4a8cb52df7c534a988e2c27ec0406e1606e7999482db83ca293f8328d8.

Pre-launch PCS command audit: catalog-gen is owned by akita-planner, not
akita-pcs. The package-scoped nextest commands now spell it
akita-planner/catalog-gen, preserving the intended CI feature graph. This
was corrected from Cargo.toml inspection before any PCS invocation; no
failed test/run is hidden. Proof evaluator and candidate sources unchanged.

10:21UTC SHA2-chain one-pair transfer:29.383833042->27.236926333s,
saving2.146907s (7.306%wall). Both verify with trace_len151102206,T28,
zero swaps/watchdogs and~81.2GiB RSS. Raw parent/candidate SHA256:
73be8b98028615f30b2a645b332da9e5833a0fac62ebd4a2e968fd543d1f5a82
801a2a3b6363a883e6f083e7905b7e36877cdfe2397cdf5fb6e8fb74d968d9d4.
All first three workload comparisons improve; the five-workload mean is
still pending SHA3/Collatz. No historical-baseline gain is substituted.

10:27UTC SHA3-chain one-pair transfer:42.081180541->37.023138833s,
saving5.058042s (12.020%wall). Both verify with trace_len184846378,T28,
zero swaps/watchdogs; reported RSS85.83/85.95GiB. Parent is3.428% above
its historical reference, within the frozen5% band. This larger single-pair
effect is descriptive, not independently replicated or precisely attributed
in full to a stage-level saving. No selective retry or control exclusion.
Raw parent/candidate SHA256:
d606b43232c7a1073f0bcadaa3a1f417943e0da6d0b6a39233b3ad072585e85a
4b1db840aeaa66dd336664c7395c8e70c7571795e50ee68b5b53d198718d624a.
Only Collatz remains in the timed matrix; correctness/lint gates still follow.
