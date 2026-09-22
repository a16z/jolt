# Recursion verifier guest

Generate an inner Fibonacci proof, then execute its verifier as a RISC-V guest:

```bash
export RAYON_NUM_THREADS=1 RUST_MIN_STACK=268435456 CARGO_BUILD_JOBS=1
cargo run --release -p recursion --features akita,field-inline -- \
  generate --example fibonacci --proofs 1 --workdir /tmp/jolt-recursion
cargo run --release -p recursion --features akita,field-inline -- \
  trace --example fibonacci --workdir /tmp/jolt-recursion
```

Guest build outputs are isolated under `<workdir>/guest-targets`; temporary trace
files also use the work directory. Generated guest sources still live in the
checkout, so concurrent runs require separate checkouts as well as work directories.

`trace` executes the verifier and reports its output; it does not prove that
execution. Successful verification returns `Recursion output (trace-only): 1`.
The `"verification"` cycle count excludes preprocessing and proof decoding;
`trace length` includes the complete guest execution. For an Akita/FR outer proof,
use the modular `outer` command below; the packed `verify` command does not prove
the outer execution. The non-Akita `verify` path uses the legacy Dory prover.

For the Dory verifier, omit `akita` from the feature list. Regenerate the inner
proof when changing the commitment backend, protocol, or schedule catalogs.
Keep the same proof input when comparing algebra-preserving guest optimizations.

Add `--embed` to `trace` or `verify` to bake the verifier setup into the guest
image. Proofs remain runtime inputs. Prepared binary catalogs, expanded matrices,
and NTT caches are trusted setup data: embedding binds them to the guest image;
in input mode the caller must authenticate the setup.

For instruction-level attribution:

```bash
JOLT_BACKTRACE=1 JOLT_PC_PROFILE=/tmp/recursion-pc.txt \
  cargo run --release -p recursion --features akita,field-inline -- \
  trace --example fibonacci --workdir /tmp/jolt-recursion
python3 scripts/guest_pc_profile.py report /tmp/recursion-pc.txt \
  /tmp/jolt-recursion/guest-targets/recursion-guest-verify/riscv64imac-zero-linux-musl/release/recursion-guest \
  --top 25
```

Use the ELF produced by that trace build when interpreting its PC profile.

### Modular q128 outer proof and preflight

With `akita,field-inline,ntt-inline`, `outer` consumes an existing FR guest ELF;
no guest build or generated-source mutation occurs in this command. The older
`generate`, `verify`, and `trace` commands still generate guest sources in this
checkout: their work directories do not make concurrent runners safe. Use an
exclusive checkout for those commands.

```sh
RUST_MIN_STACK=268435456 RAYON_NUM_THREADS=1 cargo run --release -p recursion \
  --features akita,field-inline,ntt-inline -- outer \
  --elf /path/to/recursion-guest \
  --embedded-stream /path/to/fibonacci-guest_proofs.bin \
  --max-trace-length 67108864 \
  --max-untrusted-advice-size 0 --max-trusted-advice-size 0 \
  --workdir /path/to/preflight --preflight
```

The ELF must embed the setup from that stream. Memory options must match the
ELF's guest configuration; defaults match the embedded Fibonacci recursion
profile (16,000,000 input bytes, 4096 output bytes, 128 MiB heap, 32 MiB stack).
Reserved advice capacities must be supplied explicitly, even with no actual
advice; they change the compiled I/O addresses. The ELF loader does not recover
these capacities. Validate the arguments against the guest macro or its generated
`memory_config_*` host function; preflight records the complete configuration.
For another guest, `--input` accepts its already-serialized entry-point input
instead of `--embedded-stream`. Neither option regenerates an inner proof.

The output work directory must not already exist, including for preflight. Use
a different new directory for each attempt.

Preflight streams cycles from the lazy emulator directly into one pre-reserved
modular row vector; it does not materialize a complete `Vec<Cycle>`. It derives
padded proof geometry and counts FR
rows, records row-vector capacity and an FR allocation estimate, and provisions
the grouped schedule through the production catalog API. It writes
`preflight.json` and drops the trace before returning. Catalog provisioning does
not measure PCS setup, witness, or sumcheck memory; it is not a memory-fit claim.
The required `--max-trace-length` option has no default reservation and bounds the number of rows collected as well as proof
geometry. Storage for that many rows is reserved before execution, so the row
vector does not grow by doubling. This is not a total process memory or wall-time
limit: emulator/decode state, per-tick cycle scratch, FR allocations, final-memory
extraction overlap and later prover buffers remain additional. The preflight
figures describe surviving allocations, not peak RSS. Large runs still require
an external process memory and time guard.

Omitting `--preflight` continues through the canonical FR setup,
`TraceBackend::with_field_inline`, the optimized modular q128 Akita prover, and
the full verifier. After verification accepts, the proof and public device are
written under `.proof-incomplete`, then a single directory rename publishes them
as `accepted-proof/outer-proof.bin` and `accepted-proof/outer-device.bin`. Consumers
must use the `accepted-proof` directory and successful process exit; incomplete
files are not a result. This provides atomic pair visibility on the same filesystem,
not crash-durable publication. The row vector transfers ownership into the witness
without cloning it. Large proving runs require a separately reviewed
resource budget; the trace-only commands remain available.
