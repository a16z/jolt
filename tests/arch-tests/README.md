# Architectural Tests (ACT4)

Jolt's tracer/emulator is verified against the RISC-V International
Architectural Test SIG's ACT4 suite (`riscv/riscv-arch-test`), restricted to
RV64IMAC, unprivileged-only.

The canonical entry point is `make arch-tests-64imac`. CI runs the same
target via `.github/workflows/arch-tests.yml`.

## Prerequisites

One-time setup (Linux):

```sh
make bootstrap
```

This installs:

- `riscv-none-elf-gcc` (GCC 15+) and binutils via xpack-dev-tools —
  assembles and links the ACT4 test sources. ACT4 enforces GCC 15+ at
  build time; Ubuntu 24.04's apt-provided `gcc-riscv64-unknown-elf` (13.2)
  is too old.
- `sail_riscv_sim` — reference model, invoked by ACT4 at ELF-build time to
  compute and bake expected signatures into each self-checking ELF.
- `mise` — tool manager that provisions the Ruby and uv versions ACT4
  declares in its `.mise.toml`.

No runtime reference model is used: each generated ELF compares its observed
signature against a baked-in expected signature and terminates via
`RVMODEL_HALT_PASS`/`RVMODEL_HALT_FAIL`, which map onto Jolt's existing HTIF
tohost mechanism.

## Running locally

```sh
make arch-tests-64imac
```

Steps performed:

1. Builds `jolt-emu` in debug (`cargo build -p tracer --bin jolt-emu`).
   Debug is used deliberately; arch tests are correctness-sensitive and a
   release build's optimizations are unnecessary for the suite's runtime.
2. Invokes the ACT4 Make driver under `third-party/riscv-arch-test/` with
   `CONFIG_FILES=tests/arch-tests/jolt/test_config.yaml`. ACT4 uses Sail to
   compute expected signatures and emits self-checking ELFs under the
   submodule's build directory.
3. Runs `tests/arch-tests/run.sh`, which:
   - Iterates over every generated `.elf`.
   - Skips any whose basename appears in `tests/arch-tests/skip.txt`.
   - Runs each remaining ELF through `target/debug/jolt-emu`.
   - Treats the process exit status as the pass/fail signal (0 = pass,
     non-zero = fail). `jolt-emu` propagates the HTIF endcode from
     `tohost` as its exit status.

A non-zero exit from the runner fails the Make target and, in CI, fails
the job.

## Skip list

`tests/arch-tests/skip.txt` contains one test basename per line, with
`#`-prefixed comments explaining why each entry (or group) is skipped.
The only legitimate reason is that the test targets privileged state (M/S/U
CSRs, traps, PMP, virtual memory) — those are out of scope per the ACT4
migration spec (`specs/act4-tests.md` → Non-Goals).

If a non-privileged test starts failing, do **not** add it to `skip.txt`.
Fix the tracer, the DUT config, or open an issue.

## Adding entries to the skip list

1. Run `make arch-tests-64imac` and confirm the failing test is a
   privileged-mode test (check the ACT4 test source under the submodule).
2. Add its basename to `skip.txt`, grouped with similar entries under a
   `#`-prefixed comment explaining the category.

## Deliberate-failure smoke test

`make arch-tests-smoke` assembles `tests/arch-tests/smoke/fail.S` — which
writes a non-zero HTIF endcode — and runs it through the same harness. The
target confirms that `jolt-emu` returns non-zero and that the harness
surfaces the failure rather than silently passing. This catches regressions
in the pass/fail plumbing that would otherwise make the full suite
vacuously green.

## Interpreting output

The runner prints one line per test:

```
PASS  add-01
FAIL  misaligned-ldst-01 (exit 3)
SKIP  misa-01
```

followed by a summary (`N passed, M failed, K skipped (of T)`). If any test
fails, the failing names and their exit codes are listed again at the end
and the runner exits non-zero.

## Architecture mapping (ACT4 ↔ Jolt)

| ACT4 component                                   | Jolt file                                                |
|---                                               |---                                                        |
| DUT (Device Under Test)                          | `target/debug/jolt-emu` (the tracer binary)              |
| Top-level test config                            | `tests/arch-tests/jolt/test_config.yaml`                 |
| UDB config (ISA + extensions + memory)           | `tests/arch-tests/jolt/jolt-rv64imac.yaml`               |
| Required assembly macros (HALT_PASS/FAIL, etc.)  | `tests/arch-tests/jolt/rvmodel_macros.h`                 |
| Linker script                                    | `tests/arch-tests/jolt/link.ld`                          |
| Reference model config                           | `tests/arch-tests/jolt/sail.json`                        |
| DUT C / SV config headers                        | `tests/arch-tests/jolt/rvtest_config.{h,svh}`            |
| Test harness                                     | `tests/arch-tests/run.sh`                                |
| Skip list                                        | `tests/arch-tests/skip.txt`                              |

## Relationship to the prior RISCOF integration

This replaces the previous RISCOF-based suite. The old per-DUT Python plugin
(`riscof_jolt.py`), the Spike reference model, the multi-ISA `.ini` configs,
and the `patches/` directory are all gone. The ACT4 submodule is consumed
unmodified — there is no `scripts/apply-patches` step. See
`specs/act4-tests.md` for the full migration rationale.
