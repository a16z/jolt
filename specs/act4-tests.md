# Spec: ACT4 Architectural Tests

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | @moodlezoup                    |
| Created     | 2026-04-22                     |
| Status      | implemented                    |
| PR          |                                |

## Summary

The RISC-V International Architectural Test SIG has deprecated RISCOF in favor of the ACT4 framework (`https://github.com/riscv/riscv-arch-test`). ACT4 simplifies the testing framework and increases test coverage. Jolt has a pre-existing integration with RISCOF under `tests/arch-tests/` with a CI workflow at `.github/workflows/arch-tests.yml`. This spec replaces that integration with ACT4.

## Intent

### Goal

Replace Jolt's RISCOF-based architectural-test integration with an ACT4-based integration: re-point the `third-party/riscv-arch-test` submodule to `riscv/riscv-arch-test`, drop Spike as the reference model, and narrow ISA coverage to RV64IMAC.

### Invariants

- Every ACT4 test in the RV64IMAC suite passes, or is explicitly listed in `tests/arch-tests/skip.txt` with a `#`-prefixed comment explaining why.
- `jolt-emu` test outcomes are derived from the self-checking ELFs' baked-in expected signatures (generated at ELF-build time by Sail). No runtime reference model is invoked.
- `make arch-tests-64imac` and `.github/workflows/arch-tests.yml` are the canonical entry points for running the suite locally and in CI.
- The upstream ACT4 submodule is used unmodified. No companion patch directory or patch-application step.

No changes to entries in `jolt-eval/src/invariant/` are required — existing invariants (`split_eq_bind_*`, `soundness`) concern polynomial binding and full prover+verifier soundness, neither of which is touched by this tracer-level test-infrastructure migration. No new `jolt-eval` invariants are warranted: ACT4 is a multi-hundred-test suite whose per-test checks don't map naturally onto the single-binary, `Arbitrary`-input, red-team model that `jolt-eval::Invariant` targets.

### Non-Goals

- **Privileged-mode tests** (Machine/Supervisor/User CSRs, traps, virtual memory) are explicitly out of scope. Any ACT4 tests targeting privileged modes go in `tests/arch-tests/skip.txt`.
- **No tracer semantics changes.** `jolt-emu`'s CPU/MMU/ELF-handling code is reused as-is. A narrow exception is permitted in `tracer/src/main.rs` / `run_test`: propagating the HTIF `endcode` as the process exit status so the shell runner can detect pass/fail reliably (see Design → Architecture).
- **No architectural testing through the full prover+verifier pipeline.** Coverage remains at the tracer/emulator level.
- **No Spike or RISCOF fallback.** Both are removed, not retained as alternates.
- **No F/D (floating-point) coverage** — RV64IMAC only. The old Makefile targets for `32im`, `32imac`, `32gc`, `64im`, `64gc` are removed.

## Evaluation

### Acceptance Criteria

- [ ] `third-party/riscv-arch-test` submodule points to `riscv/riscv-arch-test`.
- [ ] `scripts/bootstrap` no longer installs `riscof`, `riscv-ctg`, `riscv-isac`, or the Spike binary.
- [ ] `tests/arch-tests/` contains only ACT4-relevant configuration; RISCOF plugins (`riscof_jolt.py`, `riscof_spike.py`), the `spike/` directory, all `jolt-*.ini` files, `jolt_platform.yaml`, and `jolt_isa_*.yaml` are deleted.
- [ ] `make arch-tests-64imac` runs the full ACT4 RV64IMAC suite end-to-end against `jolt-emu` and exits 0 on success.
- [ ] The old Makefile targets `arch-tests-32im`, `arch-tests-32imac`, `arch-tests-32gc`, `arch-tests-64im`, `arch-tests-64gc` are removed.
- [ ] `.github/workflows/arch-tests.yml` runs `make arch-tests-64imac` on Ubuntu 24.04 and succeeds.
- [ ] `tests/arch-tests/skip.txt` exists, with one test name per line and `#`-prefixed reason comments for each entry (or a comment above each group of entries).
- [ ] The shell runner reads `skip.txt` and does not execute listed tests; every other generated ELF is run through `jolt-emu` and must exit 0.
- [ ] `scripts/apply-patches` and the `patches/` directory are removed; the ACT4 submodule is consumed unmodified.
- [ ] `jolt-emu` propagates the HTIF `endcode` as its process exit status when running a test ELF (exits 0 on pass, non-zero on fail). The current behavior of unconditionally exiting 0 is replaced.
- [ ] A deliberate-failure smoke test exists, confirming the harness surfaces failures rather than silently passing.
- [ ] `tests/arch-tests/README.md` exists and documents how to run the suite locally.

### Testing Strategy

- **Primary end-to-end gate**: `make arch-tests-64imac`. No additional `cargo nextest` coverage is added — this migration does not introduce new emulator flags or output-format changes that would require unit-level tests.
- **Existing tests**: must keep passing. This change does not touch anything under `cargo nextest`.
- **ZK feature axis**: not applicable. Architectural testing is tracer-only and does not interact with `--features zk`.
- **Failure-mode coverage**: one deliberate-failure test case (e.g., an intentionally-broken ELF or an assertion that flips `RVMODEL_HALT_FAIL`) confirms that the harness reports failure rather than silently passing when signatures diverge.

### Performance

No existing `jolt-eval/src/objective/` entries are expected to move. The code-quality objectives (`lloc`, `cognitive_complexity_avg`, `halstead_bugs`) are measured over `jolt-core/src/` and are unaffected by changes to `tracer/`, `tests/arch-tests/`, `Makefile`, `scripts/bootstrap`, or CI. The Criterion performance objectives (`bind_parallel_*`, `naive_sort_time`, `prover_time_*`) are unrelated to architectural testing.

No new `jolt-eval` objectives are warranted: the only relevant measure is CI wall-clock, which is neither a recurring optimization target nor measured by Criterion. The sole constraint is that the arch-tests CI job must not time out.

## Design

### Architecture

**Submodule**
- Re-point `third-party/riscv-arch-test` from `riscv-non-isa/riscv-arch-test.git` to `riscv/riscv-arch-test.git`.

**`tests/arch-tests/jolt/`** (replaces the RISCOF plugin tree)

ACT4 is configured declaratively — no per-DUT Python plugin. The new tree contains:
- `test_config.yaml` — top-level ACT4 config (compiler/objdump/Sail paths, linker-script path, DUT include path, pointer to UDB config).
- `jolt-rv64imac.yaml` — UDB config declaring RV64IMAC extensions and parameters.
- `rvmodel_macros.h` — assembly macros required by ACT4. `RVMODEL_HALT_PASS` / `RVMODEL_HALT_FAIL` map onto `jolt-emu`'s existing HTIF termination primitive (`tohost` write with device `0x00` and payload LSB set; see `tracer/src/emulator/mod.rs::run_test`).
- `link.ld` — adapted from the existing `tests/arch-tests/jolt/env/link.ld`, updated for ACT4's required sections (`.text.init`, `.text.rvtest`, `.data`, `.text.rvmodel`) and Jolt's memory base.
- `sail.json`, `rvtest_config.svh`, `rvtest_config.h` — small companion files required by ACT4.

**`tests/arch-tests/skip.txt`** — plain-text skip list. One test name per line; blank lines ignored; lines starting with `#` are comments. Used by the shell runner to filter which generated ELFs are executed — skipped tests are not run at all (no "unexpected-pass" detection). Replaces the prior `patches/`-based approach of rewriting upstream test sources to mask privileged-mode requirements.

**Deleted** — `tests/arch-tests/jolt/riscof_jolt.py`, the entire `tests/arch-tests/spike/` directory, `tests/arch-tests/jolt/jolt_platform.yaml`, all `tests/arch-tests/jolt/jolt_isa_*.yaml`, every non-64imac `.ini` file (as well as `jolt-64imac.ini` in its current RISCOF shape), `scripts/apply-patches`, and the `patches/` directory.

**Build/run flow**
1. Driver: `CONFIG_FILES=tests/arch-tests/jolt/test_config.yaml make -j$(nproc)` inside the ACT4 submodule. At generate-time, ACT4 uses `sail_riscv_sim` to compute expected signatures and bakes them into each self-checking ELF at `$WORKDIR/<name>/elfs/*.elf`.
2. Runner: a shell script (e.g., `tests/arch-tests/run.sh`) iterates over the generated ELFs, skips any whose basename appears in `tests/arch-tests/skip.txt`, runs each remaining ELF through `target/debug/jolt-emu`, and treats the process exit status as the pass/fail signal (0 = pass, non-zero = fail). No `--signature` comparison is needed — unlike RISCOF, ACT4 signatures are baked into the ELF and compared in-test.
3. `jolt-emu` change: in `tracer/src/main.rs`, after `run_test` returns, exit with the HTIF `endcode` instead of always exiting 0. This requires surfacing the `endcode` from `Emulator::run_test` (currently only logged via `tracing`). Scope is minimal — no changes to CPU, MMU, or decoding.
4. `Makefile`: a single target `arch-tests-64imac` that (a) builds `jolt-emu` in debug, (b) invokes the ACT4 Make driver, (c) runs the shell runner. Old multi-ISA Makefile targets are removed.

**Bootstrap** (`scripts/bootstrap`)
- Remove: `pip install riscof`, `riscv-ctg`, `riscv-isac`, and the Spike tarball download.
- Add: `sail_riscv_sim` (needed only at ELF-generate time). Default to the existing Jolt-release-tarball pattern used for Spike; implementer should consult ACT4 upstream for an alternate source if preferred.
- Keep: `riscv64-unknown-elf-gcc`/binutils toolchain and submodule init.

**CI** (`.github/workflows/arch-tests.yml`)
- Single job: `make arch-tests-64imac` on Ubuntu 24.04.
- Paths filter retains the same semantics (`tracer/**`, `tests/arch-tests/**`, `third-party/riscv-arch-test/**`, `Makefile`, `Cargo.toml`, `Cargo.lock`).

**1:1 mapping**

| RISCOF (current)                              | ACT4 (new)                                                                |
|-----------------------------------------------|---------------------------------------------------------------------------|
| `riscof_jolt.py` plugin                       | Deleted — no per-DUT Python                                               |
| `jolt_isa_64imac.yaml`                        | `jolt-rv64imac.yaml` (UDB format)                                         |
| `jolt_platform.yaml`                          | Folded into UDB + `test_config.yaml`                                      |
| `jolt-64imac.ini` (pairs DUT with Spike)      | `test_config.yaml` (no DUT pairing; Sail is the reference at build-time)  |
| —                                             | `rvmodel_macros.h` (maps HALT_PASS/FAIL onto jolt-emu's HTIF semantics)   |
| —                                             | `link.ld` with ACT4's required sections                                   |
| `riscof run --config ...ini --suite ...`      | `CONFIG_FILES=... make -jN` in the submodule + shell runner               |
| `jolt-emu --signature <file>` signature dump  | Unused — signatures are baked into the ELF                                |
| Per-test signature comparison by RISCOF       | Per-test self-check via `RVMODEL_HALT_PASS/FAIL` (tohost)                 |

### Alternatives Considered

1. **Keep RISCOF frozen on the old submodule.** Rejected — upstream deprecation means new coverage and bug fixes would be lost, and the existing suite would rot.
2. **Write a bespoke in-repo runner that fetches only the ACT4 ELF sources and skips the ACT4 Make/Sail toolflow.** Rejected — forks upstream and imposes ongoing maintenance for every ACT4 internal change.
3. **Dual-track (RISCOF and ACT4 side-by-side).** Rejected — doubles CI time and doubles the config surface area with no compensating benefit, since ACT4 coverage is a superset of RISCOF's.
4. **Chosen: full replacement.** Single integration, upstream-aligned, smallest surface area.

## Documentation

- No Jolt book (`book/`) changes. The book is guest/prover-focused and does not document arch-tests today.
- Add `tests/arch-tests/README.md` documenting how to run the suite locally (prerequisites, Makefile target, how to interpret pass/fail output, how to add allowlist entries).
- No `CONTRIBUTING.md` change required.

## Execution

Implementer's call. Intent + Evaluation + Design provide sufficient direction; sequencing, build-flag choices (e.g., keeping `jolt-emu` in debug for arch-tests, as indicated by the existing Makefile comment), and the order of bring-up vs. failure-mode smoke test are left to the implementer.

## References

- [`riscv/riscv-arch-test`](https://github.com/riscv/riscv-arch-test) — ACT4 framework (target)
- [`riscv-non-isa/riscv-arch-test`](https://github.com/riscv-non-isa/riscv-arch-test) — prior RISCOF suite (historical context)
- `tracer/src/emulator/mod.rs` — existing `tohost` / HTIF handling reused for `RVMODEL_HALT_PASS/FAIL`
