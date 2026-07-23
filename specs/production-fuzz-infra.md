# Spec: Production Fuzz Infrastructure

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | Jolt maintainers               |
| Created     | 2026-07-23                     |
| Status      | proposed                       |
| PR          |                                |

## Summary

Jolt currently has 13 independent `cargo-fuzz` workspaces containing 38 fuzz
targets. The targets cover cryptographic, algebraic, parser, verifier, R1CS,
opening, and tracer properties through manifest-discovered workspaces with
checked-in seeds, lockfiles, and per-profile budgets.

This feature is delivered in two phases.

**Phase 1 (base infrastructure, complete)** establishes a reliable base for
defensive fuzzing across the crates. It provides one repository-level interface
that discovers every fuzz workspace, reproducible builds, checked-in bootstrap
inputs, durable coverage-guided corpora, and PR, daily, and weekly CI tiers.

**Phase 2 (harness quality and coverage, this document's second half)** acts on
the base. The initial audit covered 22 targets, repaired broken builds, retired
or merged wasteful targets, sharpened weak oracles, and added structure-aware
targets for previously uncovered crates. The current suite now includes
transparent and ZK verifier proof-tamper targets, BlindFold expression mapping
coverage, symbolic claim lowering coverage in `jolt-r1cs`, and homomorphic Dory
batch-opening coverage in `jolt-openings`. The audit findings, verdicts, and
new-target scope are in the [Coverage Program](#coverage-program) section.

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

- For **Phase 1**: adding, deleting, or substantially redesigning individual
  fuzz harnesses. Phase 2 lifts this restriction deliberately and scopes the
  harness work below; it remains out of scope for the base-infrastructure
  commits.
- Claiming complete coverage of the verifier, prover, or cryptographic attack
  surface, even after Phase 2. The audit prioritizes soundness-relevant surface
  per unit of engineering effort, not exhaustiveness.
- Rewriting the proving system to make objects cheaper to construct for
  fuzzing. Phase 2 works with the existing constructors and test helpers, using
  fixture-once patterns where honest construction is expensive.
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
      from their manifests. At the time of this spec, that is 13 workspaces and
      38 targets.
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
4. Let the workflow exercise the complete 13-workspace matrix in CI.

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

The policy assigns time per target. These values are priors to be calibrated
from CI results, not claims that the current harnesses are already optimal. The
manifest inventory is the source of truth; this workspace-level snapshot records
the cost of the current 38-target suite.

| Workspace | Targets | PR | Daily | Weekly |
|-----------|---------|----|-------|--------|
| `jolt-crypto` | 4 | 2m15s | 50m | 110m |
| `jolt-dory` | 3 | 1m35s | 35m | 75m |
| `jolt-eval` | 4 | 2m45s | 55m | 115m |
| `jolt-field` | 6 | 2m30s | 50m | 130m |
| `jolt-hyperkzg` | 2 | 2m30s | 60m | 110m |
| `jolt-lookup-tables` | 1 | 30s | 15m | 30m |
| `jolt-openings` | 1 | 30s | 20m | 60m |
| `jolt-poly` | 3 | 1m45s | 35m | 95m |
| `jolt-r1cs` | 1 | 30s | 20m | 60m |
| `jolt-sumcheck` | 4 | 3m30s | 75m | 175m |
| `jolt-transcript` | 2 | 30s | 10m | 30m |
| `jolt-verifier` | 5 | 3m30s | 90m | 300m |
| `tracer` | 2 | 1m | 30m | 90m |

With all 13 workspaces running in parallel and targets within each workspace
running sequentially, the policy has the following expected mutation cost:

| Profile | Aggregate target time | Longest workspace |
|---------|-----------------------|-------------------|
| PR and default-branch push | 23m 20s | 3m 30s (`jolt-sumcheck`, `jolt-verifier`) |
| Daily | 9h 5m | 90m (`jolt-verifier`) |
| Weekly | 23h | 300m (`jolt-verifier`) |

The weekly `jolt-verifier` total requires a timeout well above five hours for a
cold build, replay, corpus minimization, and coverage. The workflow can instead
split that workspace's weekly targets across multiple jobs while retaining
build-cache reuse.

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

For soundness, false acceptance takes priority over false rejection; panic
resistance remains a separate availability and hardening objective. The highest
value verifier and dependency gaps now have checked-in targets:

| Target | Daily | Weekly | Harness shape |
|--------|-------|--------|---------------|
| `jolt-verifier/proof_tamper_must_reject` | 30m | 2h | Transparent verifier fixtures; mutate one proof, public input, preprocessing, claim, or opening field at a time |
| `jolt-verifier/zk_proof_tamper_must_reject` | 30m | 90m | ZK verifier fixtures; mutate committed-sumcheck, Dory ZK opening, and BlindFold proof components |
| `jolt-verifier/blindfold_expression_mapping_equivalence` | 20m | 60m | Generate Jolt claim expressions and assert verifier-side BlindFold remapping preserves their value |
| `jolt-r1cs/claim_lowering_equivalence` | 20m | 60m | Generate symbolic claim expressions, lower them to R1CS, assert the honest witness passes and a corrupted output fails |
| `jolt-openings/dory_homomorphic_batch_must_reject` | 20m | 60m | Generate honest transparent/ZK homomorphic Dory batches, then mutate statements, source lists, commitments, and transcripts |

Remaining follow-ups should focus on surfaces not covered by those targets:

| Proposed target | Daily | Weekly | Required harness shape |
|-----------------|-------|--------|------------------------|
| Standard/ZK claim reconstruction with trusted and untrusted advice | 20m | 60m | Compare full standard-mode claims with decomposed public and advice contributions |
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

## Coverage Program

Phase 1 made the harnesses reproducible, discoverable, and scheduled. It did not
judge whether they fuzz anything worthwhile. This phase does. The initial audit
read all 22 then-existing targets against the API of the crate each one tests
and classified every target as keep, improve, merge, or delete, then scoped
structure-aware targets for the gaps. The current manifest inventory contains 38
targets across 13 workspaces.

### Structure-aware harness doctrine

The recurring weakness across the existing targets is black-box byte
manipulation: handing raw fuzzer bytes to a decoder for a complex structure and
hoping mutation stumbles into a valid object. For a multilinear proof, a Dory
proof, or a group element with a subgroup check, the probability that random or
mutated bytes deserialize into a valid instance is effectively zero, so the
fuzzer spends its entire budget in the decoder and never reaches the verifier or
the arithmetic. Coverage-guided mutation also gets no useful signal when the
whole input collapses to an RNG seed.

Phase 2 targets follow these principles:

- **Construct valid objects through the crate's API, then mutate one field.**
  Build an honest commitment, proof, sumcheck transcript, or instruction with
  the real constructors; let the fuzzer choose which single field to corrupt and
  how. The interesting code (verify, fold, pairing check, discharge) is reached
  on every iteration, and mutation feedback is meaningful.
- **Prefer differential and must-reject oracles over no-panic.** A no-panic
  oracle passes for any function without an assertion. Where a reference or a
  second implementation exists (naive MLE, `num-bigint` reduction, the two RISC-V
  decoders, materialize-vs-evaluate for lookup tables, cleartext-vs-compressed
  sumcheck), assert equality. For soundness targets, assert the verifier rejects
  the tampered object — and, where the verifier returns a claim, that the
  returned claim is actually wrong (a discharge check), not merely that it did
  not panic.
- **Hoist expensive fixtures into `OnceLock`.** Setup that does not depend on
  the fuzz input — SRS generation, an honest proof to be tampered, generator
  vectors — must be built once per process, not per iteration. Several existing
  targets re-prove or re-sample the same instance every execution, which is the
  single largest throughput loss in the current suite.
- **Keep the fuzzer in control of the semantically interesting dimension.**
  Derive polynomials, points, scalars, and instruction words directly from fuzz
  bytes (via width-exact windows or `arbitrary`), not from an RNG the fuzzer only
  seeds. Bias field inputs toward algebraically special values (0, 1, r−1,
  near-modulus) with a value-class selector byte.
- **Seed from real objects.** Check in seeds that are genuine serialized valid
  instances (and, for `jolt-eval`, exports of the invariants' existing
  `seed_corpus()`), so mutation explores the neighborhood of valid structure
  instead of pure garbage.

### Audit summary and blocking repairs

The audit found three issues that must be fixed before any coverage work,
because they mean part of the current suite is inert:

- **R1 — two workspaces do not compile.** `crates/jolt-dory/fuzz/Cargo.toml` and
  `crates/jolt-hyperkzg/fuzz/Cargo.toml` lack the `[patch.crates-io]` block that
  redirects `ark-*` to the `a16z/arkworks-algebra` fork (only
  `crates/jolt-crypto/fuzz/Cargo.toml` has it). Their builds fail on
  `jolt-transcript` because `light-poseidon` pulls unpatched `ark-ff`.
- **R2 — API bitrot in five targets.** `CommitmentScheme::commit`/`open` now
  return `Result` (`crates/jolt-openings/src/schemes.rs:60`,`65`) but
  `verify_tampered`, `commit_open_verify`, `tampered_proof`, and `wrong_eval`
  destructure the old tuple and pass the result straight into `verify`.
  `Fr::from_bytes` was removed from `jolt-field`; `tampered_proof.rs:51,63,72`
  and `wrong_eval.rs:38` still call it. These five targets cannot build even
  once R1 is fixed.
- **R3 — `check` does not detect R1/R2.** `scripts/fuzz.py check` validates
  toolchain, lockfile, and seeds and runs `cargo metadata`, which never
  compiles a target, so the fast discovery gate passed while five targets were
  broken. `check` (or a dedicated CI step) must run `cargo fuzz build` — or at
  least `cargo check` — for every selected target so this class of rot fails at
  the gate rather than in the timed run.

Repairing R1/R2 is mechanical and belongs in the first Phase 2 commit; R3 is the
guard that keeps it from recurring.

### Verdicts on existing targets

Focus column uses the manifest categories. "Improve" keeps the target and
strengthens it; "merge"/"delete" frees budget.

| Workspace | Target | Verdict | Action |
|-----------|--------|---------|--------|
| jolt-field | `field_arith` | keep | Optionally verify the computed product via distributivity (currently computed, never asserted). |
| jolt-field | `wide_accumulator_fmadd` | keep | Strong differential oracle against naive Σ aᵢ·bᵢ. No change. |
| jolt-field | `wide_accumulator_merge` | keep | Strong consistency oracle. Optionally exercise `add` (covered by neither target). |
| jolt-field | `from_bytes` | improve | Replace the tautological round-trip with a `num-bigint` reference reduction; assert canonical encoding (`< r`). |
| jolt-poly | `dense_poly_ops` | improve | Fix `data[0] % 8` (never yields 8 vars; folds 0→1); replace the **dead 67-byte seed** (needs ≥96 B to reach `evaluate`); `num_vars=7` needs 4320 B > the 4096 cap — shrink scalar windows or lower the cap dependency; add a naive-MLE reference leg. |
| jolt-sumcheck | `valid_prefix_proof` | keep | Best target in the suite: dual completeness/no-panic oracle at full Fiat-Shamir depth. Optionally add a `CenteredIntegerDomain` selector. |
| jolt-sumcheck | `sumcheck_verifier` | improve | Largely subsumed by `valid_prefix_proof` at `valid_rounds=0`. Repoint at the unfuzzed production wire path `verify_compressed` (c₁-recovery, `CompressedPolynomialTooShort`). |
| jolt-transcript | `transcript_no_panic` | improve | No-panic on a surface with no panic conditions. Rebuild as an op-sequence over twin instances asserting identical challenges/`state()` plus one-byte-perturbation divergence. |
| jolt-eval | `field_mul_scalar` | keep | Strong optimized-vs-reference small-scalar Montgomery oracle, near-zero cost. |
| jolt-eval | `split_eq_bind_low_high` | improve | Strong round-by-round oracle but O(2ⁿ)/round at `num_vars=16` starves the fuzzer. Cap vars at ~10–12 (keep one large case in `seed_corpus`) or check `merge` at first/last/one random round. |
| jolt-eval | `split_eq_bind_high_low` | improve | Same cost fix; distinct code path justifies keeping separate (or merge the pair with a direction bit to free a slot). |
| jolt-eval | `transcript_consistency_poseidon` | keep | The only jolt-owned sponge (Circom-compatible BN254); highest-risk of the three. |
| jolt-eval | `transcript_consistency_blake2b` | merge | Blake2b512 is an upstream spongefish instantiation over the shared generic layer. Merge into one `transcript_consistency` target selecting the sponge from an input byte. |
| jolt-eval | `transcript_consistency_keccak` | merge/delete | Duplicate of blake2b modulo an upstream permutation; ~1/3 of the transcript budget for near-zero marginal coverage. |
| jolt-crypto | `deser_group` | improve | No-panic on decoders whose accept paths (G2 subgroup, GT r-torsion) are unreachable from random bytes. Add a decode→re-encode→decode canonical-form oracle and check in real serialized seeds. |
| jolt-crypto | `group_arith` | improve | Fuzzes an arkworks pass-through (`scalar_mul`/`msm` delegate to arkworks); the hand-written GLV and batch-addition code has zero coverage. Repoint at a GLV-vs-`scalar_mul` differential (see new targets). |
| jolt-crypto | `pedersen_commit` | improve | Rebuilds the generator setup every iteration (hoist to `OnceLock`); pins vector length to capacity (vary it to hit the prefix path); add homomorphism and single-position negative oracles. |
| jolt-dory | `deser_commitment` | keep* | Correct defensive target for the `MAX_SERIALIZED_PROOF_ROUNDS` OOM guard, **after R1**. Add a real serialized proof/commitment seed. |
| jolt-dory | `verify_tampered` | improve/rewrite | Byte-decodes a full `DoryProof`, so ~0% of inputs reach `verify` — degenerates into a slow `deser_commitment`. Rewrite: `OnceLock` honest proof, mutate structured `ArkDoryProof` fields, assert reject (plus R1/R2). |
| jolt-hyperkzg | `tampered_proof` | improve | Strong must-reject oracle but re-proves a deterministic instance every iteration (hoist to `OnceLock`) and only single-field tampers. Extend the mutation menu and absorb `wrong_eval` (plus R1/R2). |
| jolt-hyperkzg | `wrong_eval` | delete/merge | One-dimensional: every wrong scalar dies in the same branch; re-proves a deterministic instance to vary it. Fold into `tampered_proof` as one tamper class (plus R2). |
| jolt-hyperkzg | `commit_open_verify` | delete/rebuild | Input collapses to (u64 seed, len mod 4); rebuilds the SRS every iteration; redundant with unit tests. Rebuild as `fuzzed_poly_completeness` (fuzzer-controlled evaluations over a `OnceLock` SRS) or delete (plus R1/R2). |

Cross-cutting: the checked-in `jolt-eval/fuzz/seeds/*` files are placeholder
ASCII, not exports of the rich `seed_corpus()` the invariants already define. An
encoder from `seed_corpus()` to `Unstructured`-decodable bytes should replace
them; the same real-seed principle applies to the crypto and PCS deserialization
targets.

### New targets in existing workspaces

Net-new harnesses (the rewrites above are folded into their existing targets).
All construct valid objects via the API and mutate structurally.

| Workspace | Target | Harness shape | Oracle | Focus |
|-----------|--------|---------------|--------|-------|
| jolt-crypto | `glv_differential` | `OnceLock` base points; fuzzer scalars via `from_le_bytes_mod_order`; run `glv::fixed_base_vector_msm_g1` and `glv_four_scalar_mul` | equals plain `scalar_mul` | correctness |
| jolt-crypto | `batch_addition_differential` | bounded index sets over a `OnceLock` base set; `batch_g1_additions_multi` vs naive fold-add; include duplicates/empty/repeats | equals reference | correctness |
| jolt-field | `signed_accumulator_diff` | sequences of (small signed scalar, Fr) through `Fr{Signed,SmallScalar}Accumulator` and their `Naive*` counterparts | `reduce()` equality | correctness |
| jolt-field | `canonical_decode_boundary` | 32-byte encodings biased around the modulus (`<r`, `=r`, `r+small`, `2²⁵⁶−1`); decode then re-encode | re-encoding canonical and equal to `num-bigint` `mod r` | defensive |
| jolt-poly | `eq_poly_diff` | random point (≤10 vars); `EqPolynomial::evaluations` vs `eq_index_msb` per index vs split-eq reconstruction | entry-wise equality across implementations | correctness |
| jolt-poly | `compressed_roundtrip` | random `UnivariatePoly` → `CompressedPoly` (drop c₁) → recover via hint and `evaluate_with_hint` | recovered eval equals uncompressed | soundness |
| jolt-sumcheck | `honest_then_corrupt` | honest prove over a small MLE product, then one fuzzer corruption (flip coeff, perturb sum, drop/dup round, inflate degree) | verifier rejects, or returned claim differs from true evaluation (discharge) | soundness |
| jolt-sumcheck | `clear_vs_compressed_diff` | same rounds fed to `verify` and `verify_compressed` with matching labels | identical accept/reject and `EvaluationClaim` | soundness |
| jolt-transcript | `encoding_injectivity` | op sequences that concatenate to the same bytes with different boundaries (`append("ab")` vs `append("a"),append("bc")`; label/length-prefix variants) | `state()` differs whenever the structured ops differ | soundness |
| jolt-dory | `zk_mode_confusion` | `OnceLock` SRS; fuzzer evals; `commit_zk`/`open_zk`/`verify_zk` accept, then assert non-ZK `verify` rejects the ZK proof and vice versa | completeness + mode-confusion must-reject | soundness |
| jolt-eval | `compact_poly_bind_equivalence` | small-scalar coeff vectors + challenges; `CompactPolynomial::bind` (both orders) vs field-promoted `DensePolynomial` bind, per round | equals reference | correctness |
| jolt-eval | `transcript_narg_robustness` | honest NARG + fuzzer (offset, xor) mutations replayed through `verifier_transcript` | no panic/unbounded alloc; a mutation to absorbed bytes changes a later challenge | soundness |
| jolt-eval | `legacy_transcript_digest_compat` | shared op sequence through `LegacyBlake2bTranscript` and legacy `Blake2bTranscript` | identical challenge streams (proof-compat boundary) | correctness |

### New fuzz workspaces (ranked)

Five uncovered crates warrant workspaces, ordered by soundness value per unit of
engineering effort. Each entry names its best 1–2 targets; all use the
fixture-once, mutate-structurally pattern.

1. **jolt-verifier** — the system's must-reject surface. Implemented targets:
   `proof_tamper_must_reject` over transparent `muldiv`, advice, and committed
   program fixtures; `zk_proof_tamper_must_reject` over ZK `muldiv` and
   committed-program fixtures; `blindfold_expression_mapping_equivalence`;
   `proof_deserialize_no_panic`; and `validate_inputs_no_panic`. Fixtures are
   generated by the ignored `generate_fuzz_fixture` nextest target in both
   transparent and ZK feature modes.
2. **tracer (+ jolt-program)** — two independently maintained RISC-V decoders,
   `tracer::Instruction::decode` and `jolt_program::decode_instruction`, over the
   same ISA. `decode_differential`: fuzz a u32 word (and a compressed halfword via
   `uncompress_rv64_instruction`), decode with both, assert accept/reject parity
   and matching kind/operands. A divergence means the emulator executes one
   instruction while bytecode preprocessing proves another — direct soundness
   impact, microsecond iterations. Add `single_instruction_execute_no_panic`
   (x0 stays 0, registers canonically sign-extended).
3. **jolt-lookup-tables** — the cheapest pure soundness oracle in the repo.
   `mle_materialize_equivalence`: fuzz (table-kind byte, u128 index),
   assert `F::from_u64(materialize_entry(i)) == evaluate_mle(bits(i))` across the
   ~40 tables; `prefix_suffix_combine_equivalence`: assert `combine(prefixes,
   suffixes) == materialize_entry(i)` — the decomposition the sumcheck uses.
   ~128 field mults per iteration; the `pub(crate)` test helpers need a
   `test-utils` feature or ~20 lines re-implemented.
4. **jolt-program (+ common)** — the only genuine attacker-controlled byte parser
   in the uncovered set. `decode_elf_no_panic`: raw bytes → `decode_elf`, seeded
   with a minimal valid RV64 ELF, no panic/OOM and structural invariants on
   success. Add `bytecode_preprocess_pc_mapping` (PC map injective/round-trips)
   and `memory_layout_invariants` (`MemoryLayout::new` under `catch_unwind`,
   regions disjoint and ordered).
5. **jolt-r1cs** — operationalizes the repo's most-emphasized invariant
   (claim computation equals its R1CS constraint). Implemented
   `claim_lowering_equivalence`: fuzz a small `Expr` plus source values,
   evaluate directly, lower through `assert_claim_expr_eq`, assign the witness,
   assert the lowered constraint accepts, then corrupt only the claimed output
   and assert rejection. Microseconds per iteration, no PCS.
6. **jolt-openings** — covers the verifier dependency that batches final Dory
   openings. Implemented `dory_homomorphic_batch_must_reject`: construct honest
   transparent and ZK same-point batches, then mutate claimed evaluations,
   commitments, transcript labels, point agreement, witness dimensions, and
   prover source counts.

Deferred: **jolt-blindfold** (valuable, but its prover harness is trapped in
`tests/support` and jolt-verifier ZK-mode tampering already exercises
`BlindFoldProof` end-to-end) and a **jolt-openings prefix-packing layout**
target, which is worthwhile but distinct from homomorphic Dory batching.
**jolt-witness**, **jolt-claims** (covered via jolt-r1cs), and
**jolt-prover-legacy** (no standalone attacker surface; heaviest build) are out
of scope. These realize the earlier
[soundness-focused target roadmap](#soundness-focused-target-roadmap) with
concrete harness shapes.

### Budgets, evaluation, and commits for this phase

Budgets follow the existing manifest policy: soundness targets get the largest
tiers, differential/correctness targets a middle tier (they are cheap but their
oracle is strong), and defensive/no-panic targets the smallest. New targets are
added with conservative priors and recalibrated by the same procedure as Phase 1
once the weekly runs produce data. Repurposed and merged targets return their
freed budget to the pool; merging the three `transcript_consistency_*` targets
into one, and folding `wrong_eval` into `tampered_proof`, roughly offsets the new
targets' cost.

Phase 2 acceptance criteria:

- [x] Every fuzz target builds under the pinned nightly and ASan; `check
      --compile` and a CI build step run `cargo check`/`cargo fuzz build` per
      workspace and fail on a build error (closes R1/R2/R3).
- [x] No target's default seed early-returns before the code under test; seeds
      are validated by replay under a real fuzz build (macOS uses
      `--sanitizer none`; see FUZZING.md), and the crypto/PCS and verifier
      fixtures are genuine serialized objects generated by ignored tests.
- [x] Every soundness target uses a must-reject or discharge oracle, not
      no-panic; every correctness target asserts equality against a reference or
      second implementation.
- [x] Deterministic, input-independent fixtures are built once per process
      (`OnceLock`) or embedded via `include_bytes!` for the verifier bundle.
- [x] The three `transcript_consistency_*` targets are one target; `wrong_eval`
      is folded into `tampered_proof`; `group_arith` is replaced by
      `glv_differential`/`batch_addition_differential`; `sumcheck_verifier`
      exercises `verify_compressed`.
- [x] The top-ranked new workspaces (jolt-verifier, tracer,
      jolt-lookup-tables, jolt-r1cs, and jolt-openings) exist with their targets,
      seeds, budgets, and lockfiles, and pass discovery and `check`.

Deferred to follow-up commits: the jolt-eval structure-aware trio
(`compact_poly_bind_equivalence`, `transcript_narg_robustness`,
`legacy_transcript_digest_compat`), the jolt-lookup-tables
`prefix_suffix_combine_equivalence` target (needs the phased sparse-dense
binding machinery exposed via a `test-utils` feature), the tracer
`single_instruction_execute_no_panic` target (needs a bounded CPU execution
sandbox), and the jolt-program workspace.

Commit structure (independently reviewable, on top of the Phase 1 commits):

1. `fix(fuzz): repair crypto and PCS fuzz builds` — R1 patch blocks, R2 API
   updates, and the `check` compile step (R3).
2. `test(fuzz): strengthen field, poly, and sumcheck oracles` — the improve
   verdicts and dead-seed fixes in the cheap workspaces.
3. `test(fuzz): consolidate transcript and PCS targets` — the merges/deletions
   and `OnceLock` fixture hoists.
4. `test(fuzz): add structure-aware targets to existing workspaces` — the
   net-new targets table.
5. `test(fuzz): add jolt-verifier fuzz workspace` — ranked new workspace 1.
6. `test(fuzz): add tracer decode-differential workspace` — ranked 2.
7. `test(fuzz): add jolt-lookup-tables fuzz workspace` — ranked 3.
8. `test(fuzz): add jolt-r1cs and jolt-openings fuzz workspaces` — claim
   lowering and homomorphic-batch dependency coverage.

Commits 3 and 4 landed together (`test(fuzz): consolidate targets and add
structure-aware coverage`) because the consolidation and net-new targets share
per-workspace manifests. Later workspaces (`jolt-program`) and the jolt-eval
trio follow the same one-change-per-commit pattern.

## References

- [cargo-fuzz](https://github.com/rust-fuzz/cargo-fuzz)
- [libFuzzer documentation](https://llvm.org/docs/LibFuzzer.html)
- [ClusterFuzzLite](https://google.github.io/clusterfuzzlite/)
- [OSS-Fuzz Rust integration](https://google.github.io/oss-fuzz/getting-started/new-project-guide/rust-lang/)
