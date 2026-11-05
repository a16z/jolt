# Deferred Akita recursion trace

This experiment generates two different SHA2-chain execution proofs with a requested
power-of-two padded trace. A RISC-V recursion guest verifies both direct Akita
proofs and publishes every deferred setup query and answer in its public output.
The harness measures execution without proving the outer execution.

Akita is pinned to `e1f722d5633af4575f0cb6aa57d93d69c9c66024` from
remote `main` in the workspace manifest and lockfile.
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
every fold. Compare setup-offloaded inner proofs by generating a fresh pair with
`--offload` using the same pinned checkout.
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
an explicit `AkitaVerificationContext` accumulator in the outer opening.
Acceptance also requires an empty pending collection after outer verification. Setup answers remain advice
until that discharge. This trace harness does not claim to perform that outer proof.

## Fresh proof replay

Like `examples/recursion`, this example generates its inputs with the current
checkout before executing a verifier guest. Proofs and verifier preprocessing
are version-specific; regenerate the pair after changing Jolt or Akita.
There is no checked-in proof bundle.

The archived accumulated-proof measurements and original fixture are available
at [Jolt commit 88c38e0e](https://github.com/a16z/jolt/tree/88c38e0e4ae0d671e3fb8e08abf138da4c5e5520/examples/akita-recursion).
They require that checkout and its Akita pin; they are not current-code
performance claims. The current harness generates inner proofs and executes
verifier guests. It does not generate an accumulated outer proof.

The local smoke command lints both host packages with `recursion-host`, generates
a small pair with `--trace-log 14 --iterations 6`, checks native Check/Defer
replay and rejection of false setup advice and altered public output, and builds
all three guest crates with the guest toolchain. Like `examples/recursion`, it has
no dedicated CI job. Run it after installing the current CLI:

```bash
bash examples/akita-recursion/smoke.sh
```

To generate committed CPU inner proofs, select direct setup checking by default
or add `--offload`:

```bash
cargo run --profile ci -p jolt-akita-recursion --features recursion-host -- \
  generate --trace-log 26 --offload --schedules crates/jolt-akita/schedules \
  --output benchmark-runs/new-pair.bin
cargo run --profile ci -p jolt-akita-recursion --features recursion-host -- \
  rebind --pair benchmark-runs/new-pair.bin --setup-log 28 \
  --output benchmark-runs/new-pair-domain28.bin
cargo run --profile ci -p jolt-akita-external --features akita --bin outer-verifier-audit -- \
  --prepare-pair benchmark-runs/new-pair-domain28.bin benchmark-runs/pair-material
JOLT_AKITA_EXTERNAL_MATERIAL="$PWD/benchmark-runs/pair-material" \
JOLT_FUNC_NAME=verify_advice jolt build -p jolt-akita-advice-guest \
  --mode std --backtrace off --stack-size 33554432 --heap-size 1073741824 \
  --memory-size 2147483648 -- --release --features guest
```

The `advice-guest` is the separate new outer layout with fixed verifier
identity. Its output is A. An enclosing prover must read A from actual guest
output, validate the full canonical statement and supply it through
`InheritedSetupClaims::from_public_output`; the public output retains A.

The `external-guest` accepts an application-supplied accumulated outer proof and
exports B for an enclosing setup-offloaded opening. The local smoke command
compiles it; execution requires that separately generated outer proof. Final
verification uses Check and requires no pending claims.

## Verification context

Ordinary `jolt_verifier::verify` keeps the complete Check policy. Call
`jolt_verifier::verify_with_context` for setup accumulation, supplying an
`AkitaVerificationContext` with the selected `SetupMode` and an owned
`SetupClaimAccumulator`. The accumulator borrows the caller's answer tape;
reuse the context across inner proofs, then extract and discharge all pending
claims. Defer without an accumulator is rejected.

Jolt passes this backend-owned context directly to `PCS::verify_batch`.
There is no replaceable opening-verification callback. The context is runtime
state and is never serialized with verifier preprocessing or proof bytes.
Setup-domain provisioning uses the same prover setup and verifier types as
ordinary verification. Covered domains reuse their prepared state. Enlarging
a domain before commitment retains the existing seed and prefix commitments,
rebuilds only the one-hot backend, and preserves the dense backend and caches.

Other PCS implementations declare `type VerificationContext<'a> = ()` when
no per-verification state is needed. Manual native group-batch calls now pass
a mutable context as their final argument; the separate homomorphic batch API
is unchanged.
