# Four-workload handoff integration

Production integration: Jolt `63ef92c12`, Akita
`7878e5ba14b0a9f7cc89ddbc1331a543efa11db7`. The fork branch was fast-forwarded
and published first so the immutable Git pin is fetchable. No force push,
parameter/catalog/transcript/verifier change or diagnostic capture hook.

The three Akita production-file hashes match the accepted campaign exactly:

- onehot.metal: `77140d9cd84be8f9addb8eb3e5d87266f7fd2b42ab39b6a743c0aa975f8b8ada`
- runtime.rs: `ac19f91365015bd12c6f09da57700c0fc59c40cf93d8fb27cdc22b16f65a7d46`
- packed_onehot_fp128_d128_rank3.rs: `54abeca1d910ed77591c4696ad61c96932be4da16e430b307108e92316c75bfe`

Reuse the accepted kernel's Akita Metal 32/32, Jolt Metal 329/329, PCS 23/23
in both feature modes and three fork clippy gates. Their raw evidence and
limitations remain in ../akita-commit-occupancy-2026-09-06/acceptance.md and
validation.md. These suites are reused, not claimed as fresh integration runs.

Fresh checks:

| Check | Result |
|---|---|
| Locked release build against the remote Git pin | Pass, 266.30 s |
| Jolt clippy --all --features host -q --all-targets | Pass, 12.17 s |
| Jolt clippy --all --features host,zk -q --all-targets | Pass, 4.11 s |
| Focused release Metal/profiling example clippy | Pass, 10.20 s |
| Final formatted locked release build | Pass, 209.33 s |
| Clean-checkout cargo fmt --all --check | Pass |
| Python script test suite | 29/29 pass, including 5 new matrix-contract tests |
| Scoped diff checks | Pass |
| Four-workload T24 smoke through handoff runner | 4/4 verified, zero swaps/watchdogs |

Cargo checks use -D warnings; exact commands, exit codes, raw hashes and
resource samples are in events.jsonl and runs/. The first build controller
failed in its temporary network wrapper while invoking the RSS sampler
(KeyError: env); its child was terminated. The corrected invocation uses
Cargo's --config net.offline=false instead of changing subprocess behavior.
Both logs/events are preserved; no prover failure was involved.

The initial main-worktree fmt check found one formatting issue in the user's
untracked onehot_census.rs, as well as the new example's println layout.
Only the latter was formatted. A clean checkout of the committed integration
passes the complete fmt check. User census/pathway files and the fork's
uncommitted tile-benchmark files remain untouched and uncommitted.

The BLAKE2b workload support is promoted without its one-off sizing example
or extra jolt-host development dependencies. A single Workload::program
method owns guest selection for both preparation and proof execution.
The portable runner lives in scripts/; historical campaign scripts remain
archival evidence, not dependencies of Andrew's workflow.

## Completed smoke, 17:03:43 UTC

The clean integration checkout is `/private/tmp/jolt-andrew-handoff-20260906`.
Binary SHA256: `8d4aff5ef7ab6850cc021c9ac39db0f10fcbcd7ea414cf843b0a35be87c61585`.
All four AOT preparations completed in about nine seconds, and every guest
hash matches the earlier campaign. Each proof followed a full 120-second
cooldown; no concurrent compilation or proof ran. Machine: M4 Max, 40 GPU
cores, 16 CPU cores, 128 GiB, macOS 26.6.2. Full identity: smoke-t24/manifest.json.

| Workload | T24 prove seconds | Actual rows | Peak RSS GiB |
|---|---:|---:|---:|
| Fibonacci | 8.970361125 | 12583871 | 8.39899 |
| SHA2-chain | 9.110998000 | 9444142 | 9.08986 |
| BTreeMap | 7.904195125 | 9650940 | 9.29550 |
| BLAKE2b-chain, 5000 hashes | 8.789301666 | 10433678 | 9.11061 |

All padded lengths are 16777216. Measured T24 arithmetic mean: 1.9357781226
MHz; optional projected M5 mean: 2.1874292785 MHz. These are T24 results, not
new T28 measurements. The earlier first-three T24 observations were 9.2719,
9.4607 and 8.1738 seconds respectively; no obvious regression required a retry.
No proof observation was repeated or omitted. BLAKE2b matches the independent
Python digest. Independent post-run parsing verifies all raw hashes, exactly
one proof/timing marker per cell, trace scales, zero swaps and the reported
arithmetic mean. The process exited zero and released both run locks.

The full T24–T28 sweep remains Andrew's independent measurement. Existing
four-workload T28 observations average 9.3269 MHz measured on M4; 10.5394 MHz
is the separate 1.13x M5 projection, not an M4 pass.

Could not verify here: Andrew's independent 20-cell sweep, actual M5 speeds,
physical occupancy/register/spill measurements, or a new independent security
audit. The inherited Akita preflight line-cap, recursive_commit error-owner,
unused-dependency and unavailable typos limitations are not resolved by this
handoff and remain disclosed in the accepted campaign audit.
