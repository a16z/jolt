# Lane URS (wave 8) — recurring gate-flake hygiene

**Verdict: URS hypothesis FALSIFIED. Real mechanism: guest-ELF artifact
replace/read race in `/tmp/jolt-guest-targets`. Fixed, 3× suite green.**
Commit `2954f0ce1` on `lane/metal-w8-urs` (base dc49d402f; orthogonal to
wave-8's 71ee2c57e merge — touches only jolt-prover-legacy host).

## Falsification receipt (URS disk-cache race)

- All `~/Library/Caches/dory/dory_N.urs` files pre-existed every repro run
  and stayed **byte-untouched** (mtime+inode constant, diffed before/after
  each of 4 full-suite runs) — yet run 4 flaked anyway. No URS was
  generated, loaded-stale, or overwritten during any failing window.
- Code-side: both setup paths (jolt-dory `scheme.rs:107`, legacy
  `commitment_scheme.rs:139`) already hold the `dory.lock` advisory lock
  across the whole load-or-generate-save; files are size-keyed
  (`dory_{max_log_n}.urs`) and nothing ever rewrites an existing file. The
  interleaving the comments fear (commit under URS_A, later load URS_B) has
  no writer left to produce it.
- The flake is **not a byte-diff mismatch at all** — it's a panic. Nobody
  had captured the failure output before; the "byte-diff test fails fast"
  reading came from the test's suite membership, not its assert.

## Actual mechanism (receipt)

Reproduced on run 4 of 4 (matches ~1-in-2..4 gate rate; wave-8 battery's
clean first pass is the same coin landing heads):

```
FAIL [1.070s] jolt-prover::byte_diff muldiv::prover_matches_legacy_on_muldiv
panicked at crates/jolt-prover-legacy/src/host/program.rs:280:
could not open elf file: "/tmp/jolt-guest-targets/muldiv-guest-/
riscv64imac-unknown-none-elf/release/muldiv-guest"
```

Interleaving:

```
test A (muldiv)                      test B (committed_muldiv, same guest)
--------------------------------     --------------------------------------
jolt build  → artifact linked
elf_path.exists()  ✓
                                     jolt build → fingerprint DIRTY (always)
                                       cargo uplift: unlink(artifact)   ← gap
File::open(artifact) → ENOENT ✗
                                       hard_link(deps copy, artifact)
```

- **Every** `jolt build` is fingerprint-dirty: a warm, no-change rebuild
  swaps the artifact inode (measured: 13752854 → 13758874, same
  mtime/size) — cargo re-uplifts (unlink + hardlink) on each invocation.
- 5 parallel test processes share `muldiv-guest-`'s target dir (muldiv +
  committed_muldiv×4); 4 share the advice guest. Any reader landing in a
  sibling's unlink window dies at ~1–5 s (the guest-build phase).
- Explains every observed trait: fast failure, different victim per gate
  (wave 5 advice_committed, wave 7 committed_muldiv_many_chunks, now plain
  muldiv), passes isolated and on rerun, always in the guest-heavy suite.

## Fix (`2954f0ce1`, 2 files, +56/−10, jolt-prover-legacy host only)

1. `lock_guest_target()` — exclusive advisory lock on
   `<guest_target_dir>.lock` (same best-effort shape as `urs_lock`), held
   in `build_with_features` across the cargo build, the exists check, and
   a new ELF read.
2. ELF bytes cached in `Program` at build time (`elf_contents`,
   `elf_compute_advice_contents`); `get_elf_contents`, `trace`,
   `trace_to_file` (and `decode` via `jolt_program`) consume the build-time
   copy instead of re-opening the shared path. Path fallback kept for
   externally assigned `elf` paths.

No proof-byte change (byte-diff suite is the oracle, 3× green). No kill
switch: pure hygiene; only behavior change is that a Program re-reads its
own build-time bytes rather than whatever a sibling last linked — which is
the point. Single-process cost: one flock + one in-memory ELF copy (~KB–MB).

## Evidence

| check | result |
|---|---|
| pre-fix: full suite ×4 | runs 1–3 green, run 4 FAIL (the panic above); URS files untouched all 4 runs |
| pre-fix: warm rebuild inode | artifact inode swapped on no-change rebuild (uplift receipt) |
| post-fix: `prover-fixtures` ×3 consecutive | 20/20, 20/20, 20/20 (175.3 s / 173.6 s / 143.0 s) |
| metal suites (`-p jolt-kernels -p jolt-dory -p jolt-eval --features jolt-kernels/metal,jolt-eval/metal`) | 406/406 |
| clippy `--features host` and `host,zk`, `-D warnings` | clean |

Note for the orchestrator: `git commit` hangs forever in agent sessions on
this box — `commit.gpgsign=true` (ssh) with no ssh-agent; hooks pass, then
git blocks on signing. Campaign history is unsigned; committed with
`--no-gpg-sign`. Worktree `.worktrees/metal-w8-urs` ready for merge +
cleanup (`wt remove metal-w8-urs` after).
