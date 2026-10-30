# Deferred Akita recursion trace

This experiment generates two different SHA2-chain execution proofs with a requested
power-of-two padded trace. A RISC-V recursion guest verifies both direct Akita
proofs and publishes every deferred setup query and answer in its public output.
The harness measures execution without proving the outer execution.

Akita is pinned to `9531295760583b12b876a45a81ed40dabefa4b76` on
`codex/external-setup-accumulation-checkpoint` in the workspace manifest and lockfile.
Regenerate inputs when that pin changes.
The experiment generates one exact direct scalar catalog for the inner trace shape
and adapts the grouped opening through Jolt's normal preprocessing. The supplied
directory provides the dense auxiliary catalogs.

From the repository root, with the Jolt guest toolchain installed, run:

```bash
cargo install --path . --locked
bash examples/akita-recursion/benchmark.sh
```

The script builds the host, generates two `2^26` direct inner proofs, and executes
both full-check and deferred recursion guests. Each run gets a fresh directory
under `benchmark-runs/akita-recursion/` with the proof pair, generated catalogs,
cycle report, logs, embedded setup material, guest ELFs, Jolt commit and lockfile.
Pass a directory to choose where to save a new run:

```bash
bash examples/akita-recursion/benchmark.sh benchmark-runs/my-recursion-run
```

To run the individual steps:

```bash
cargo build --locked --profile ci --target-dir target \
  -p jolt-akita-recursion --features recursion-host

RUST_LOG=info RAYON_NUM_THREADS=8 target/ci/akita-recursion generate \
  --trace-log 26 --schedules crates/jolt-akita/schedules \
  --output benchmark-runs/akita-recursion/pair.bin

RUST_LOG=info RAYON_NUM_THREADS=1 target/ci/akita-recursion trace \
  --pair benchmark-runs/akita-recursion/pair.bin \
  --report benchmark-runs/akita-recursion/cycles.json
```

`generate` requires both actual padded traces to equal `2^26`; it fails if the
iteration estimate misses that interval. `--iterations` selects an explicit chain
length. Both proofs are verified natively and must have no setup-offloaded edge.
For a small smoke run, use `--trace-log 14 --iterations 6`.
To benchmark an existing pair again, rerun only `trace` and choose a fresh report
filename. Keep the matching Akita pin: proof artifacts are tied to that version.

`trace` executes the same pair through `verify_before` (`SetupMode::Check`) and
`verify_deferred` (`SetupMode::Defer`). It retains no recursive trace rows, checks
the guest panic flag, and checks the exact deferred public output. The JSON
includes total cycles, the canonical padded outer trace length, real/virtual cycle
counts for each marker, and an exclusive breakdown that sums to total execution:

- pair decoding;
- loading the program-bound Akita key and terminal cache;
- shared Akita setup-domain preparation;
- inner 0's Akita PCS opening and other Jolt verifier work;
- inner 1's Akita PCS opening and other Jolt verifier work;
- batch assembly overhead;
- setup-claim export;
- input and claim cleanup;
- remaining runtime and I/O.

`innerN_jolt` includes `innerN_akita`. The exclusive table subtracts that child
before reporting the remaining Jolt verifier work. The Akita marker covers Jolt's
native PCS adapter, including commitment decoding, transcript bridging, and the
Akita verifier. Markers use expanded Jolt trace rows, with real RV64IMAC and
virtual instructions also reported separately.

Before building either guest, `trace` exports the native seed-derived verifier
key and selected row's terminal transform into `<report-stem>.material/` and
embeds them through `JOLT_AKITA_RECURSION_MATERIAL`. Both guests use the same
material. These are trusted setup constants fixed by the guest ELF, not advice
accepted from proof input. Catalog decoding and installation are measured in
`install_prepared_setup`; the guest does not expand the setup seed or compute
terminal transforms. Keep the material directory with the report to reproduce
the guest build.

The `before` baseline uses the same two direct inner proofs and checks setup in
every fold. Comparing against Akita main with setup-offloaded inner proofs requires
a separate checkout pinned to main and freshly generated offloaded proofs.
Compare per-proof `innerN_jolt` or `innerN_akita` scopes when discussing verifier
costs, and the complete guest totals when discussing the outer trace.

For later composition, an inherited batch's setup domain must equal the prefix
committed by the selected outer offloaded fold. Re-embed the same direct-proof
queries and answers into a larger prefix of the same public stream with:

```bash
target/ci/akita-recursion rebind \
  --pair benchmark-runs/akita-recursion/pair.bin --setup-log <outer-prefix-log2> \
  --output benchmark-runs/akita-recursion/rebound-pair.bin
```

Rerun `trace` on the rebound pair. Contribution support outside the original
domain is zero. The logical setup domain may exceed the verifier's locally
materialized matrix; the verifier retains its ordinary direct/terminal prefix.
Native advice generation materializes its required domain on the host. The report
records both lengths. A smaller outer prefix requires another schedule.

Deferred replay alone is conditional. A composing application must bind the
complete public output using `InheritedSetupClaims::from_public_output`, attach it
with `AkitaProverSetup::with_inherited_setup`, and check the same collection using
`InheritedSetupClaims::accumulator` in the outer opening. Acceptance also requires
an empty pending collection after outer verification. Setup answers remain advice
until that discharge. This trace harness does not claim to perform that outer proof.
