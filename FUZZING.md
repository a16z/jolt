# Fuzzing

Operator guide for Jolt's fuzz infrastructure. `scripts/fuzz.py` is the single
entry point: it discovers every cargo-fuzz workspace from the checked-in
manifests and drives `cargo fuzz` for local development and CI. Design
rationale lives in [`specs/production-fuzz-infra.md`](specs/production-fuzz-infra.md).

## Prerequisites

| Tool | Pinned version |
|------|----------------|
| Rust toolchain | `nightly-2026-07-20` with `llvm-tools-preview` and `rust-src` (installed automatically via each workspace's `rust-toolchain.toml`) |
| cargo-fuzz | `0.13.2` |
| Python | 3.10+ (standard library only) |

```bash
cargo install cargo-fuzz --version 0.13.2 --locked
```

The runner refuses to fuzz with a mismatched `cargo-fuzz` version, an unpinned
toolchain, or a missing lockfile, so drift fails loudly instead of producing
irreproducible runs.

## Layout and discovery

Fuzz workspaces live next to the crate they test and are discovered from
`crates/*/fuzz/Cargo.toml` plus top-level `*/fuzz/Cargo.toml` (for crates that
have not yet moved under `crates/`). The manifests are the single source of
truth: adding or removing a `[[bin]]` entry is all that is required — there is
no second target list to keep in sync, in the script or in CI.

```bash
python3 scripts/fuzz.py inventory                    # human-readable table
python3 scripts/fuzz.py inventory --format json      # machine-readable
python3 scripts/fuzz.py inventory --format github    # CI matrix
```

## Local commands

Every command operates on all workspaces by default, or is narrowed with
`--workspace <crate>` and (except `inventory`/`check`) `--target <name>`.

```bash
python3 scripts/fuzz.py check                        # validate configuration + report profile runtimes
python3 scripts/fuzz.py check --resolve              # + lockfiles resolve with --locked
python3 scripts/fuzz.py check --compile              # + every target type-checks (catches API bitrot)
python3 scripts/fuzz.py build                        # build fuzz targets
python3 scripts/fuzz.py replay                       # run each seed/regression exactly once
python3 scripts/fuzz.py run --profile pr             # fuzz with each target's manifest budget
python3 scripts/fuzz.py run --seconds 60             # override every selected target's budget
python3 scripts/fuzz.py cmin                         # minimize mutable corpora
python3 scripts/fuzz.py coverage                     # coverage data from all corpora
```

`run` takes exactly one of `--profile <pr|daily|weekly>` (each target fuzzes
for its manifest-declared budget) or `--seconds N` (one explicit duration for
every selected target). It applies per-input safety limits so one pathological
input cannot consume a whole campaign: `--max-len 4096`, `--timeout 30`
(seconds per input), and `--rss-limit-mb 4096` by default. Failing outcomes —
crashes, timeouts, out-of-memory, sanitizer findings — land in the workspace's
`fuzz/artifacts/<target>/` directory, and the failing workspace/target and
exit status are listed at the end of the run. One broken target does not stop
the remaining selected targets; the command still exits non-zero.

Target builds use AddressSanitizer by default. On macOS the workspaces whose
dependency tree includes the patched arkworks fork (`jolt-crypto`, `jolt-dory`,
`jolt-hyperkzg`) fail to **link** under ASan: the fork's `ark-ff` enables its
`allocative` feature by default, which pulls in `ctor`, and the macOS linker
rejects `ctor`'s static initializer in sanitized builds (`ld: initializer
pointer has no target`). CI fuzzes on Linux and passes
`--target-triple x86_64-unknown-linux-gnu` so sanitizer builds do not use a
static musl target. For local work on macOS, pass `--sanitizer none`:

```bash
python3 scripts/fuzz.py --workspace jolt-dory --sanitizer none replay
```

Single-artifact triage commands (both require `--workspace` and `--target`):

```bash
python3 scripts/fuzz.py --workspace jolt-field --target from_bytes \
    reproduce fuzz/artifacts/from_bytes/crash-<hash>
python3 scripts/fuzz.py --workspace jolt-field --target from_bytes \
    tmin fuzz/artifacts/from_bytes/crash-<hash>
```

## Fuzz state

Each state directory has a distinct lifecycle and owner:

| Directory | Committed to Git | Purpose |
|-----------|------------------|---------|
| `fuzz/seeds/<target>/` | yes | Small, reviewed bootstrap inputs. Every target must have at least one seed that reaches its parser and satisfies any minimum-length gate; `check` enforces this. |
| `fuzz/regressions/<target>/` | yes | Minimized, reviewed reproducers for fixed bugs. Replayed deterministically before any mutation work. |
| `fuzz/corpus/<target>/` | no (gitignored) | Mutable coverage-guided state. Persisted only by trusted scheduled CI runs; safe to delete locally. |
| `fuzz/artifacts/<target>/` | no (gitignored) | Raw failure outputs from libFuzzer. Triage material, never committed as-is. |
| `fuzz/coverage/` | no (gitignored) | Coverage output from `coverage` runs. |
| `fuzz/fixtures/` | yes | Reviewed serialized objects the target embeds via `include_bytes!` (e.g. an honest proof to tamper). Regenerated by an ignored, host-gated test, not by fuzzing. |

Some targets need a real object that is too expensive to build per iteration
(an honest proof, a serialized valid instance). These are generated once by an
ignored test and committed under `fuzz/fixtures/` or `fuzz/seeds/`:

```bash
# jolt-verifier tamper fixture (needs the host/prover build):
cargo nextest run -p jolt-verifier --features prover-fixtures \
    --test generate_fuzz_fixture --run-ignored ignored-only --cargo-quiet
cargo nextest run -p jolt-verifier --features prover-fixtures,zk \
    --test generate_fuzz_fixture --run-ignored ignored-only --cargo-quiet
# crypto/PCS real serialized seeds:
cargo nextest run --manifest-path crates/jolt-crypto/fuzz/Cargo.toml \
    --run-ignored ignored-only --cargo-quiet
cargo nextest run --manifest-path crates/jolt-dory/fuzz/Cargo.toml \
    --run-ignored ignored-only --cargo-quiet
```

`replay` runs every checked-in seed and regression exactly once, so a known
failure fails deterministically without waiting for a fuzz budget. `run`
merges seeds and regressions into the mutable corpus as additional corpus
directories.

## Target budgets

Every target declares its focus and per-profile mutation budgets next to its
`[[bin]]` entry, so the runner and CI never carry a second target-to-budget
map:

```toml
[package.metadata.jolt-fuzz.targets.valid_prefix_proof]
focus = "soundness"
pr-seconds = 90
daily-seconds = 1800
weekly-seconds = 5400
```

`focus` must be `soundness`, `correctness`, or `defensive`. Validation fails
for a target without a policy, a policy naming a nonexistent target, and a
missing, non-integer, or non-positive budget. `check` and `inventory` report
each workspace's total runtime per profile; soundness-focused targets get the
largest budgets, and hot parser/no-panic targets the smallest.

The current allocations are priors, not measurements. Before materially
changing a budget, run the calibration procedure in the spec (same starting
corpus, at least three independent 30-minute trials per target, metrics
recorded at 5/10/30 minutes) and reallocate toward targets that still reach
new high-value states. A plateaued soundness target keeps a baseline budget;
a plateaued defensive target is a candidate for reduction.

Targets that require non-default crate features declare them in the same policy
block. For example, `jolt-verifier/zk_proof_tamper_must_reject` is compiled and
run with `cargo-features = ["zk"]`, so the runner builds the ZK verifier path
without forcing every verifier fuzz target into that feature mode.

## Interpreting fuzz output

Each `run` invocation prints libFuzzer status lines, final statistics
(`-print_final_stats=1`), and a runner `[stats]` line per target with exit
status, focus, budget, and elapsed time. The signals to read:

- **exec/s** — throughput. High values on a protocol target often mean inputs
  are rejected before reaching the interesting path, not that fuzzing is going
  well. A drop after a harness change usually matters more than the absolute
  number.
- **cov / ft** — edge coverage and libFuzzer features. Growth means the fuzzer
  is still finding new behavior; a plateau means more time is buying little.
  Coverage is an operational signal, not a quality score — soundness depends
  on the oracle and the states reached.
- **corp** — corpus size and bytes. Steady growth with flat coverage suggests
  redundant inputs; weekly `cmin` keeps this in check.
- **peak RSS** — memory headroom against the `-rss_limit_mb` limit.
- **semantic depth** — target-specific (e.g. verifier rounds reached, valid
  proofs mutated). Where a harness can report it, prefer it over raw coverage
  when deciding budgets.

## CI tiers

`.github/workflows/fuzz-crates.yml` has one discovery job that validates the
configuration and emits the workspace matrix, then one job per workspace.
Workspaces fuzz in parallel; targets within a workspace run sequentially to
reuse build artifacts. The profile is selected from the triggering event:

Budgets come from each target's manifest policy (see [Target
budgets](#target-budgets)); the workflow passes only the profile name.

| Profile | Trigger | Aggregate mutation time | Corpus cache | Extras |
|---------|---------|-------------------------|--------------|--------|
| `pr` | pull request / push to main | 23m 20s (longest workspace 3m 30s) | restore only | |
| `daily` | cron `17 4 * * *` | 9h 5m (longest workspace 90m) | restore + save on success | |
| `weekly` | cron `43 3 * * 0` | 23h (longest workspace 300m) | restore + save on success | `cmin` + coverage upload |
| manual | `workflow_dispatch` | per chosen profile | per chosen profile | |

All profiles validate the configuration and replay seeds and regressions
before mutation. Pull-request jobs may consume the latest trusted corpus but
never publish new corpus state — only successful scheduled runs on the default
branch save a new cache, so contributor-controlled code and inputs cannot
poison trusted state. A failed run never replaces the previous cache.

The workflow runs with read-only permissions and exposes no secrets to fuzzed
code. Failure artifacts are retained for 7 days, weekly coverage output for
14 days. The corpus cache is an efficiency mechanism, not the only copy of
anything important: any reproducer worth keeping is minimized, reviewed, and
committed under `regressions/`.

## Crash triage and reproducer promotion

GitHub Actions artifacts on a public repository are **not** a private
disclosure channel. If a finding looks security-relevant — a soundness break,
a verifier accepting an invalid proof, memory corruption — move it to the
private maintainer process in [SECURITY.md](SECURITY.md) before analysis or
discussion in public issues or PRs.

For a non-sensitive failure:

1. Download the `fuzz-failure-<workspace>-<run id>` artifact, or reproduce
   locally with the `reproduce` command printed in the job log.
2. Minimize it: `python3 scripts/fuzz.py --workspace <ws> --target <t> tmin <artifact>`.
3. Fix the bug.
4. Commit the minimized input as `fuzz/regressions/<target>/<short-name>` in
   the same PR as the fix. Every later run replays it deterministically.

## Adding a fuzz target

1. Write `fuzz_targets/<name>.rs` in the crate's fuzz workspace and add the
   matching `[[bin]]` entry to `fuzz/Cargo.toml`. For `jolt-eval`, annotate
   the invariant and run `./jolt-eval/sync_targets.sh` instead — it generates
   both.
2. Add a `[package.metadata.jolt-fuzz.targets.<name>]` policy next to the
   `[[bin]]` entry with a `focus` and `pr-seconds`/`daily-seconds`/
   `weekly-seconds` budgets (see [Target budgets](#target-budgets)).
3. Add at least one small seed under `fuzz/seeds/<name>/` that reaches the
   input parser (and meets any minimum-length gate in the harness).
4. `python3 scripts/fuzz.py --workspace <crate> check --resolve` — commit the
   `Cargo.lock` update if dependencies changed.
5. `python3 scripts/fuzz.py --workspace <crate> --target <name> replay`, then
   a short `run --seconds 30` to confirm the harness executes.

No workflow or script changes are needed — discovery picks the target up from
the manifest, and the configuration check fails if the seed is missing.

For a new fuzz **workspace**, mirror an existing one: `fuzz/Cargo.toml` with
`[package.metadata] cargo-fuzz = true`, the pinned `rust-toolchain.toml`, a
committed `Cargo.lock`, and the standard `.gitignore` entries (`artifacts/`,
`corpus/`, `coverage/`, `target/`).
