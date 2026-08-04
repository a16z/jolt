# Metal W2: isolated-kernel harness

## Result

- `jolt-eval` accepts runtime keys shaped `callgrind:<bench>:instructions`.
- `measure-objectives` and optimizer measurement dispatch run the named iai-callgrind target and reject missing/malformed/schema-drifted output.
- `benches/callgrind/eq_evals.rs` is the CPU single-kernel template.
- `benches/metal/metal_fr_bind.rs` is the Metal template: setup outside timing, shared GPU lock, one synchronous command buffer per Criterion sample.
- `sync_targets.sh` regenerates plain Criterion, Metal, and callgrind targets without losing paths or the Metal feature gate.

## Exact usage

### Add a deterministic CPU instruction-count objective

1. Copy `jolt-eval/benches/callgrind/eq_evals.rs` to `jolt-eval/benches/callgrind/<bench>.rs`; replace setup and the benchmark body.
2. Regenerate target entries:

   ```bash
   ./jolt-eval/sync_targets.sh
   ```

3. With Valgrind and workspace-matching `iai-callgrind-runner` installed, measure the parameterized objective:

   ```bash
   /usr/bin/lockf -k /tmp/jolt-metal-wave2-cargo.lock cargo run -q --message-format=short -p jolt-eval --bin measure-objectives -- \
     --objective callgrind:<bench>:instructions
   ```

### Add a Metal wall-time objective

1. Copy `jolt-eval/benches/metal/metal_fr_bind.rs` to `jolt-eval/benches/metal/<bench>.rs`.
2. Replace `KernelId`, params, buffers, work size, and the post-setup warm dispatch; retain `gpu_lock()` and synchronous `run_once()` inside `bencher.iter`.
3. Regenerate entries with `./jolt-eval/sync_targets.sh`.
4. Run only that kernel:

   ```bash
   /usr/bin/lockf -k /tmp/jolt-metal-wave2-cargo.lock cargo bench -p jolt-eval --features metal --bench <bench> -- --quick
   ```

Callgrind measures host `Ir`, not Metal device instructions; use the Criterion scaffold for GPU kernels.

## Integrated from `origin/main` / `fa303e27f`

- Runtime callgrind key parser and type-safe `OptimizationObjective::Callgrind` dispatch.
- iai-callgrind schema-v6 JSON parsing, new-run metric selection, multi-case summation, and loud errors.
- `measure-objectives` and optimizer measurement paths.
- Callgrind target synchronization and `eq_evals` example.

## Deliberately omitted

- Parameterized telemetry, modular-prover `summary.json`, profiling feature plumbing, span taxonomy, heap snapshots, and curated modular objective functions: these require the newer modular profile pipeline and would replace campaign profiling behavior.
- Direct `iai-callgrind-runner` schema-oracle dev dependency: avoided its expanded dependency surface; schema version and representative fresh/baselined fixtures remain tested.
- All unrelated `fa303e27f` prover, kernel, verifier, profiling, CI, and documentation changes; no rebase.

The root workspace manifest, `crates/jolt-kernels`, and `vendor/dory-pcs` are unchanged. The vendored `dory-pcs:0.4.0` patch and existing Metal workspace dependencies remain active. `Cargo.lock` changes only add existing workspace packages to `jolt-eval`'s dependency list.

## Verification

- `/usr/bin/lockf -k /tmp/jolt-metal-wave2-cargo.lock cargo fmt -q --message-format=short` — pass.
- `/usr/bin/lockf -k /tmp/jolt-metal-wave2-cargo.lock cargo check -p jolt-eval --all-targets --features metal --message-format=short` — pass; three pre-existing workspace `default-features` warnings.
- `/usr/bin/lockf -k /tmp/jolt-metal-wave2-cargo.lock cargo nextest run -p jolt-eval --cargo-quiet -E 'test(/callgrind/)'` — 4 passed, 95 skipped.
- `/usr/bin/lockf -k /tmp/jolt-metal-wave2-cargo.lock cargo run -q --message-format=short -p jolt-eval --bin measure-objectives -- --help` — callgrind grammar exposed.
- `/usr/bin/lockf -k /tmp/jolt-metal-wave2-cargo.lock cargo clippy -q --message-format=short -p jolt-eval --all-targets --features metal` — pass.
- `/usr/bin/lockf -k /tmp/jolt-metal-wave2-cargo.lock cargo bench -p jolt-eval --features metal --bench metal_fr_bind -- --quick` — pass; 2^20 `FrBind` 254.56–256.25 µs, 4.0920–4.1191 Gelem/s.
- `sh -n jolt-eval/sync_targets.sh` and `git diff --check` — pass.

No end-to-end prover benchmark ran. Callgrind execution was not run because Valgrind is unavailable on this host; its parser, target, and CLI compile paths were verified.
