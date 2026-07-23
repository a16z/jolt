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
python3 scripts/fuzz.py check                        # validate configuration
python3 scripts/fuzz.py check --resolve              # + lockfiles resolve with --locked
python3 scripts/fuzz.py build                        # build fuzz targets
python3 scripts/fuzz.py replay                       # run each seed/regression exactly once
python3 scripts/fuzz.py run --seconds 60             # coverage-guided fuzzing
python3 scripts/fuzz.py cmin                         # minimize mutable corpora
python3 scripts/fuzz.py coverage                     # coverage data from all corpora
```

`run` applies per-input safety limits so one pathological input cannot consume
a whole job: `--max-len 4096`, `--timeout 30` (seconds per input), and
`--rss-limit-mb 4096` by default. Failing outcomes — crashes, timeouts,
out-of-memory, sanitizer findings — land in the workspace's
`fuzz/artifacts/<target>/` directory, and the failing workspace/target and
exit status are listed at the end of the run. One broken target does not stop
the remaining selected targets; the command still exits non-zero.

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

`replay` runs every checked-in seed and regression exactly once, so a known
failure fails deterministically without waiting for a fuzz budget. `run`
merges seeds and regressions into the mutable corpus as additional corpus
directories.

## CI tiers

`.github/workflows/fuzz-crates.yml` has one discovery job that validates the
configuration and emits the workspace matrix, then one job per workspace.
Workspaces fuzz in parallel; targets within a workspace run sequentially to
reuse build artifacts. The profile is selected from the triggering event:

| Profile | Trigger | Per-target budget | Corpus cache | Extras |
|---------|---------|-------------------|--------------|--------|
| `pr` | pull request / push to main | 30 s | restore only | |
| `daily` | cron `17 4 * * *` | 10 min | restore + save on success | |
| `weekly` | cron `43 3 * * 0` | 15 min | restore + save on success | `cmin` + coverage upload |
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
2. Add at least one small seed under `fuzz/seeds/<name>/` that reaches the
   input parser (and meets any minimum-length gate in the harness).
3. `python3 scripts/fuzz.py --workspace <crate> check --resolve` — commit the
   `Cargo.lock` update if dependencies changed.
4. `python3 scripts/fuzz.py --workspace <crate> --target <name> replay`, then
   a short `run` to confirm the harness executes.

No workflow or script changes are needed — discovery picks the target up from
the manifest, and the configuration check fails if the seed is missing.

For a new fuzz **workspace**, mirror an existing one: `fuzz/Cargo.toml` with
`[package.metadata] cargo-fuzz = true`, the pinned `rust-toolchain.toml`, a
committed `Cargo.lock`, and the standard `.gitignore` entries (`artifacts/`,
`corpus/`, `coverage/`, `target/`).
