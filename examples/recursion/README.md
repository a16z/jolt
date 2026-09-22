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
  --workdir /path/to/preflight --preflight
```

The ELF must embed the setup from that stream. Memory options must match the
ELF's guest configuration; defaults match the embedded Fibonacci recursion
profile (16,000,000 input bytes, 4096 output bytes, 128 MiB heap, 32 MiB stack).
For another guest, `--input` accepts its already-serialized entry-point input
instead of `--embedded-stream`. Neither option regenerates an inner proof.

Preflight owns one modular trace, derives its padded proof geometry, counts FR
rows, records row-vector capacity and an FR allocation estimate, and provisions
the grouped schedule through the production catalog API. It writes
`preflight.json` and drops the trace before returning. Catalog provisioning does
not measure PCS setup, witness, or sumcheck memory; it is not a memory-fit claim.
The maximum trace option is a proof-geometry admission limit, not an execution
watchdog. Run resource limits outside the process.

Omitting `--preflight` continues through the canonical FR setup,
`TraceBackend::with_field_inline`, the optimized modular q128 Akita prover, and
the full verifier. Only after verification accepts are `outer-proof.bin` and
`outer-device.bin` written. The trace vector transfers ownership into the witness
without a full-vector clone. Large proving runs require a separately reviewed
resource budget; the trace-only commands remain available.
