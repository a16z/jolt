# Spec: Production Fuzz Infrastructure

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | Jolt maintainers               |
| Created     | 2026-07-23                     |
| Status      | proposed                       |
| PR          |                                |

## Summary

Jolt currently has eight independent `cargo-fuzz` workspaces containing 22 fuzz
targets. The targets cover useful cryptographic and algebraic properties, but the
infrastructure around them is incomplete: CI runs only a subset, toolchains and
dependencies are not uniformly pinned, every run starts without durable corpus
state, and there is no standard regression, minimization, coverage, or crash
handling workflow.

This feature establishes a reliable base for defensive fuzzing across the crates.
It provides one repository-level interface that discovers every fuzz workspace,
reproducible builds, checked-in bootstrap inputs, durable coverage-guided corpora,
and PR, daily, and weekly CI tiers. It does not attempt to redesign individual
fuzz targets. Target-specific improvements should be independently reviewable
follow-up PRs built on this base.

## Intent

### Goal

Make every crate fuzz target reproducible, stateful, discoverable, and useful in
both local development and tiered CI without requiring maintainers to keep
duplicated target lists in sync.

### Invariants

- The checked-in fuzz manifests are the source of truth for workspace and target
  discovery. Adding or removing a `[[bin]]` entry and its colocated policy must
  not require editing a target matrix outside that manifest.
- Each target's focus and PR, daily, and weekly mutation budgets are declared in
  its fuzz manifest. The repository runner and CI workflow must not contain a
  second target-to-budget map.
- Every fuzz workspace has an exact nightly toolchain, an exact `cargo-fuzz`
  version in automation, and a committed lockfile.
- Every target has at least one small checked-in seed that reaches its input
  parser and satisfies any minimum input-length gate.
- Fuzz state has three distinct lifecycles:
  - `seeds/<target>/` contains small, reviewed bootstrap inputs committed to Git.
  - `regressions/<target>/` contains minimized, reviewed reproducers for fixed
    bugs and is committed to Git.
  - `corpus/<target>/` contains mutable coverage-guided state, is ignored by Git,
    and is persisted by trusted scheduled CI runs.
- Checked-in seeds and regressions are replayed before new mutation work starts.
  A known failure must fail deterministically without waiting for a fuzz budget.
- Pull-request code may consume the latest trusted corpus but may not update the
  shared corpus state. Only trusted default-branch scheduled runs may publish new
  corpus caches.
- One broken target must not prevent the runner from attempting the remaining
  selected targets. The final command must still return a failure.
- Crashes, timeouts, out-of-memory failures, and sanitizer findings are failing
  outcomes. Their reproducer paths and the exact local replay command must be
  available to maintainers.
- CI must not grant write permissions, expose repository secrets to fuzzed code,
  or treat public workflow artifacts as a private vulnerability-reporting
  channel.
- This infrastructure must not alter Jolt runtime, proving, or verification
  behavior.

### Non-Goals

- Adding, deleting, or substantially redesigning individual fuzz harnesses.
- Claiming complete coverage of the verifier, prover, or cryptographic attack
  surface.
- Enrolling Jolt in OSS-Fuzz or deploying ClusterFuzzLite. Those remain useful
  follow-ups once this base is reliable.
- Introducing a coverage percentage gate before a stable baseline exists.
- Automatically classifying, fixing, publishing, or disclosing fuzz findings.
- Running untrusted pull-request fuzzing with credentials or allowing it to
  poison the persistent corpus.
- Replacing deterministic unit, integration, property, or `jolt-eval` tests.

## Evaluation

### Acceptance Criteria

- [ ] A repository command discovers all current fuzz workspaces and all targets
      from their manifests. At the time of this spec, that is eight workspaces
      and 22 targets.
- [ ] The inventory can be emitted as human-readable text, JSON, and a GitHub
      Actions matrix, including each target's focus and profile budgets.
- [ ] Configuration validation fails for a missing target source, duplicate
      target name, unpinned toolchain, missing lockfile, or target without a
      checked-in seed.
- [ ] Configuration validation fails for missing, non-positive, or invalid
      per-target profile budgets and reports each workspace's profile runtime.
- [ ] All fuzz workspaces use the same exact nightly toolchain and declare the
      components needed for sanitizer builds and coverage.
- [ ] All fuzz workspaces have committed lockfiles that resolve with `--locked`.
- [ ] A single local command can build, replay, fuzz, minimize, or collect
      coverage for all workspaces, one workspace, or one target.
- [ ] Checked-in seeds and regressions are replayed exactly once each.
- [ ] A smoke build and regression replay succeed for `jolt-field`.
- [ ] PR CI validates discovery, builds and replays every target, restores but
      does not save corpus state, and performs a bounded fuzz run.
- [ ] Daily CI restores the latest trusted corpus, fuzzes every target with a
      target-specific larger budget, and saves updated corpus state only after a
      successful run.
- [ ] Weekly CI performs the daily behavior with a larger budget, minimizes the
      corpus, and produces per-workspace coverage output.
- [ ] Fuzz logs retain per-target execution count, executions per second,
      coverage, feature count, corpus size, peak RSS, elapsed time, and exit
      status so budget changes can be evaluated rather than guessed.
- [ ] CI preserves crash artifacts for maintainer triage with short, explicit
      retention and documents that artifacts in a public repository are not an
      embargoed reporting mechanism.
- [ ] The operator documentation explains installation, local commands, state
      ownership, reproducer promotion, CI tiers, and adding a target.
- [ ] Automated tests cover discovery, configuration errors, deterministic input
      ordering, workspace selection, and generated CI inventory.

### Testing Strategy

The repository runner has unit tests that use temporary fuzz manifests and file
trees. The tests must exercise both successful discovery and failures that could
silently remove a target from CI. They also cover profile-budget parsing, invalid
budgets, deterministic target ordering, workspace runtime totals, and explicit
local duration overrides.

The implementation is validated in increasing-cost layers:

1. Run the runner's unit tests and static configuration check.
2. Resolve every fuzz workspace with its committed lockfile.
3. Build and replay all `jolt-field` targets locally under the pinned nightly
   toolchain and AddressSanitizer.
4. Let the workflow exercise the complete eight-workspace matrix in CI.

The primary Jolt `muldiv` checks are not required for infrastructure-only changes
because no production crate source is modified. If implementation changes leak
outside fuzz workspaces, the standard host and ZK clippy and `muldiv` validation
requirements apply.

### Performance

This feature has no production runtime or proof-size impact. Fuzz jobs run per
workspace in parallel and targets within a workspace run sequentially so that
build artifacts are reused without creating excessive runner contention.

Uniform wall-clock budgets are a poor fit for these targets. Parser fuzzers can
execute hundreds of thousands of inputs per second and often reach their shallow
coverage frontier quickly. Protocol targets may perform group operations,
construct proofs, or intentionally drive a verifier through several valid rounds
before mutation begins. Soundness-affected paths also deserve a minimum budget
even when their raw coverage growth is slower than a parser target.

The initial policy therefore assigns time per target. These values are priors to
be calibrated from CI results, not claims that the current harnesses are already
optimal.

| Workspace | Target | PR | Daily | Weekly | Focus |
|-----------|--------|----|-------|--------|-------|
| `jolt-crypto` | `deser_group` | 15s | 5m | 15m | Defensive parsing |
| `jolt-crypto` | `group_arith` | 30s | 10m | 25m | Algebraic correctness |
| `jolt-crypto` | `pedersen_commit` | 60s | 25m | 45m | Commitment correctness |
| `jolt-dory` | `deser_commitment` | 15s | 5m | 15m | Defensive parsing |
| `jolt-dory` | `verify_tampered` | 20s | 5m | 15m | Soundness; provisional pending a structured seed |
| `jolt-eval` | `field_mul_scalar` | 30s | 10m | 25m | Optimized/reference equivalence |
| `jolt-eval` | `split_eq_bind_high_low` | 45s | 15m | 30m | Optimized/reference equivalence |
| `jolt-eval` | `split_eq_bind_low_high` | 45s | 15m | 30m | Optimized/reference equivalence |
| `jolt-eval` | `transcript_consistency_blake2b` | 30s | 10m | 15m | Prover/verifier consistency |
| `jolt-eval` | `transcript_consistency_keccak` | 30s | 10m | 15m | Prover/verifier consistency |
| `jolt-eval` | `transcript_consistency_poseidon` | 30s | 10m | 15m | Prover/verifier consistency |
| `jolt-field` | `from_bytes` | 15s | 5m | 15m | Canonical decoding |
| `jolt-field` | `field_arith` | 30s | 10m | 25m | Algebraic correctness |
| `jolt-field` | `wide_accumulator_fmadd` | 30s | 10m | 25m | Optimized/reference equivalence |
| `jolt-field` | `wide_accumulator_merge` | 30s | 10m | 25m | Optimized/reference equivalence |
| `jolt-hyperkzg` | `commit_open_verify` | 60s | 25m | 45m | Positive protocol correctness |
| `jolt-hyperkzg` | `tampered_proof` | 60s | 25m | 45m | Soundness |
| `jolt-hyperkzg` | `wrong_eval` | 30s | 10m | 20m | Soundness; fixed underlying proof |
| `jolt-poly` | `dense_poly_ops` | 45s | 15m | 45m | Optimized/reference equivalence |
| `jolt-sumcheck` | `sumcheck_verifier` | 30s | 10m | 25m | Defensive verification |
| `jolt-sumcheck` | `valid_prefix_proof` | 90s | 30m | 90m | Deep soundness path |
| `jolt-transcript` | `transcript_no_panic` | 15s | 5m | 15m | Defensive transcript handling |

With all eight workspaces running in parallel and targets within each workspace
running sequentially, the policy has the following expected mutation cost:

| Profile | Aggregate target time | Longest workspace |
|---------|-----------------------|-------------------|
| PR and default-branch push | 13m 5s | 3m 30s (`jolt-eval`) |
| Daily | 4h 35m | 70m (`jolt-eval`) |
| Weekly | 10h 20m | 130m (`jolt-eval`) |

The weekly `jolt-eval` total leaves little headroom in a 150-minute job for a
cold build, replay, corpus minimization, and coverage. The workflow must either
use at least a 180-minute timeout or split that workspace's weekly targets across
multiple jobs while retaining build-cache reuse.

Every target also has a per-input timeout, RSS limit, and maximum input length so
one pathological input cannot consume its complete campaign. A target may
override the common input limits when its encoding or algorithm requires it.

#### Allocation principles

For a sound target and a fixed corpus, more mutation time gives the fuzzer more
opportunities to find a counterexample. It does not fix an input model that
cannot reach the intended path, a decoder that rejects almost every mutation, a
weak oracle, or deterministic setup rebuilt on every iteration. Extra time also
has diminishing value after a target's coverage and semantic depth plateau.

Every soundness-relevant target keeps a baseline budget even after a plateau.
Additional compute is allocated in this order:

1. Targets that can cause false verifier acceptance or a prover/verifier
   disagreement.
2. Targets that reach deep protocol state with valid prefixes or structured
   mutations.
3. Targets still gaining edge coverage, libFuzzer features, or protocol-specific
   semantic depth.
4. Defensive parser and no-panic targets that have reached a stable frontier.

Raw executions per second are not an allocation objective. A target that returns
before parsing or verification can report high throughput while doing little
useful work. Coverage is also an operational signal rather than a quality score:
soundness depends on the oracle and the states reached, not the percentage alone.

#### Known harness constraints

Two current targets should not receive longer campaigns until their reachability
is improved:

- `jolt-dory/verify_tampered` only calls the verifier after arbitrary bytes
  deserialize into a complex `DoryProof`, while its bootstrap seed is an empty
  JSON array. A follow-up target should construct or seed a structurally valid
  proof and apply field-, round-, and byte-level mutations. After that change,
  its initial allocation becomes 25 minutes daily and 60–90 minutes weekly.
- `jolt-poly/dense_poly_ops` cannot exercise its largest intended dimensions
  under the common 4096-byte input cap. Seven variables require 4320 bytes, and
  its current `% 8` expression never produces eight variables. The target should
  use a larger target-specific limit or expand a compact fuzzer-controlled seed
  into coefficients with a deterministic RNG.

Cryptographic targets should place input-independent fixtures in `OnceLock` or an
equivalent one-time initializer. Improving a target from one hundred to one
thousand valid iterations per second usually beats multiplying its wall-clock
budget by ten.

#### Budget calibration

The initial allocations are recalibrated from a controlled experiment rather
than from one CI run:

1. Snapshot the same starting corpus for every trial.
2. Run at least three independent 30-minute trials per target with different
   libFuzzer seeds on the same runner class.
3. Record execution count, executions per second, edge coverage, feature count,
   corpus size, peak RSS, and any available semantic metric at 5, 10, and 30
   minutes.
4. Classify the target as growing, slow-growing, or plateaued. Keep a baseline
   for plateaued soundness targets and move discretionary time toward targets
   that still reach new high-value states.
5. Re-run the calibration after a material harness, mutator, input-limit, or
   oracle change.

The target-aware policy is useful if it reaches more distinct soundness-relevant
states per CPU-hour than the uniform policy. If longer protocol campaigns only
repeat the same paths while reduced parser budgets lose meaningful coverage, the
allocation must be revised.

#### Soundness-focused target roadmap

The following follow-up targets are outside the infrastructure-only scope of this
feature, but they define where additional fuzzing capacity should go. For
soundness, false acceptance takes priority over false rejection; panic resistance
remains a separate availability and hardening objective.

| Proposed target | Daily | Weekly | Required harness shape |
|-----------------|-------|--------|------------------------|
| Full verifier over structured mutations of a valid `JoltProof` | 30m | 2h | Construct fixtures once; mutate one proof field, opening, or transcript element at a time |
| BlindFold input/output claim and constraint equivalence | 20m | 60m | Evaluate the cleartext claim and corresponding constraint against the same generated accumulator |
| Standard/ZK claim reconstruction with trusted and untrusted advice | 20m | 60m | Compare full standard-mode claims with decomposed public and advice contributions |
| Sumcheck prover/verifier round-trip plus targeted corruption | 30m | 90m | Generate small valid instances, then mutate claims, rounds, and transcript messages separately |
| RISC-V tracer differential execution, sharded by ISA family | 30m per shard | 2h | Compare short bounded instruction sequences with an independent reference model |
| RAM and register read-write relations over small traces | 30m | 2h | Compare optimized proof relations with direct state simulation |
| Proof and transcript canonical serialization | 10m | 30m | Cover round-trip, truncation, trailing bytes, malformed lengths, and noncanonical encodings |
| ELF and bytecode preprocessing | 10m | 30m | Exercise attacker-controlled parsing with strict size and allocation limits |

## Design

### Architecture

`scripts/fuzz.py` is the repository-level control plane. It discovers
`crates/*/fuzz/Cargo.toml` plus top-level `*/fuzz/Cargo.toml` manifests (for
crates that have not yet moved under `crates/`), validates the associated
files, and invokes `cargo fuzz` without embedding Jolt target names in the
script.

```
fuzz Cargo.toml files
          |
          v
  discovery + validation
          |
          +---- inventory (table / JSON / CI matrix)
          |
          +---- build / replay / run / cmin / coverage
                              |
              +---------------+---------------+
              |               |               |
        checked-in seeds   regressions    mutable corpus
                                              |
                                  trusted scheduled cache
```

The runner exposes the same operations locally and in CI. This keeps CI as an
orchestrator rather than a second implementation of fuzz behavior and gives
maintainers and AI-assisted workflows deterministic commands and
machine-readable inventory to inspect.

Target policy is stored next to the corresponding `[[bin]]` entry without
duplicating target names in the runner:

```toml
[package.metadata.jolt-fuzz.targets.valid_prefix_proof]
focus = "soundness"
pr-seconds = 90
daily-seconds = 1800
weekly-seconds = 5400
```

The runner validates that every `[[bin]]` has one matching policy entry, that
policy entries do not name nonexistent targets, and that `focus` is one of
`soundness`, `correctness`, or `defensive`. The `run` command's `--profile`
option uses the declared durations. An explicit `run --seconds N` remains
available as a local override for all selected targets.

Each fuzz invocation uses:

- AddressSanitizer through `cargo-fuzz` defaults;
- a bounded maximum input length;
- a per-input timeout and RSS limit;
- libFuzzer final statistics and value profiling;
- a target-specific artifact directory.

#### CI tiers

The workflow has one discovery job and a dynamically generated per-workspace
matrix. It selects a profile from the triggering event:

- Pull requests and pushes use the PR profile.
- One scheduled event uses the daily profile.
- A separate scheduled event uses the weekly profile.
- Manual dispatch permits an explicit profile for maintainer testing.

All profiles validate and replay the immutable state before mutation. The
workspace job invokes each target with its manifest-declared profile duration and
continues after a target failure so the remaining targets are attempted.
Scheduled profiles restore the newest per-workspace corpus cache. Successful
daily and weekly runs save a new immutable cache key; later runs restore it
through a stable prefix. PR jobs are restore-only. A failed run never replaces
trusted state.

Weekly jobs minimize the mutable corpus before saving it and upload coverage
output separately from failure artifacts. Cache persistence is an efficiency
mechanism, not the only copy of a regression: any important reproducer is
minimized, reviewed, and committed under `regressions/`.

#### Failure handling

The immediate output of a failure must identify the workspace, target, artifact,
and replay command. CI retains raw reproducers briefly for maintainers. Because
GitHub Actions artifacts on a public repository are not a private disclosure
channel, suspected security findings are moved to the project's private
maintainer process before analysis or discussion. Once fixed, a minimized
non-sensitive reproducer is committed as a regression test whenever practical.

### Alternatives Considered

- **One root fuzz crate:** This would centralize configuration but couples
  unrelated dependency and feature graphs and makes isolated builds harder. A
  discovery layer preserves the existing workspace boundaries.
- **A hand-maintained CI matrix:** Simpler initially, but it already drifted from
  the manifests. Manifest-derived inventory makes omission a testable error.
- **Commit the complete evolving corpus:** Durable but creates rapid repository
  growth and noisy reviews. Only reviewed seeds and regressions belong in Git;
  mutable coverage state belongs in a cache or external corpus store.
- **Allow PR jobs to update the corpus:** This improves continuity but permits
  contributor-controlled code and inputs to poison trusted state. PR jobs are
  deliberately restore-only.
- **Start with OSS-Fuzz or ClusterFuzzLite:** Both can add scale, corpus
  management, and better private reporting. Adopting them before local target
  discovery, replay, and reproducibility work would preserve the current drift
  behind another service.
- **Gate on line coverage immediately:** A single percentage is a weak proxy for
  fuzz quality and would be arbitrary without historical data. Weekly coverage
  is collected first so later gates can be evidence-based and target-specific.

## Documentation

Add a root `FUZZING.md` operator guide. No Jolt book changes are required because
this feature changes maintainer tooling rather than the zkVM's user-facing API or
protocol.

The guide documents:

- prerequisites and pinned versions;
- inventory, validation, build, replay, run, minimization, and coverage commands;
- seed, regression, mutable corpus, and artifact lifecycles;
- PR, daily, and weekly profiles, target-specific budgets, and the tuning
  procedure;
- interpreting execution rate, coverage, feature, corpus, semantic-depth, and
  resource metrics;
- secure crash triage and reproducer promotion;
- the checklist for adding a fuzz target.

## Execution

### Repository runner and reproducibility

- Add `scripts/fuzz.py` and focused unit tests.
- Pin every fuzz workspace to one exact nightly with `llvm-tools-preview` and
  `rust-src`.
- Generate and commit every fuzz workspace lockfile.
- Declare each target's focus and profile budgets in its fuzz manifest and
  include them in human-readable, JSON, and CI inventory output.
- Make target synchronization deterministic and verify that generated manifests
  do not drift.

### Stateful inputs

- Add one reviewed bootstrap seed per target.
- Standardize ignored `artifacts/`, `corpus/`, `coverage/`, and `target/`
  directories.
- Reserve checked-in `regressions/<target>/` directories for triaged failures.

### CI and operations

- Replace the partial hard-coded fuzz workflow with dynamic discovery.
- Implement target-aware PR, daily, weekly, and manual profiles with an explicit
  local duration override.
- Restore corpus state in every fuzzing tier, but save it only from trusted
  successful daily and weekly runs.
- Preserve per-target final statistics and use the calibration procedure before
  materially changing budgets.
- Upload failure artifacts and weekly coverage with explicit short retention.
- Add `FUZZING.md`.

### Commit structure

Keep the change reviewable with conventional commits:

1. `docs(fuzz): specify production fuzz infrastructure`
2. `build(fuzz): make fuzz workspaces reproducible`
3. `test(fuzz): add persistent seed and regression layout`
4. `ci(fuzz): add pull request daily and weekly tiers`
5. `docs(fuzz): document production fuzz operations`

## References

- [cargo-fuzz](https://github.com/rust-fuzz/cargo-fuzz)
- [libFuzzer documentation](https://llvm.org/docs/LibFuzzer.html)
- [ClusterFuzzLite](https://google.github.io/clusterfuzzlite/)
- [OSS-Fuzz Rust integration](https://google.github.io/oss-fuzz/getting-started/new-project-guide/rust-lang/)
