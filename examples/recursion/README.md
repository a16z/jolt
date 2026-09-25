# Recursion verifier guest

Generate an inner Fibonacci proof, then execute its verifier as a RISC-V guest:

```bash
export RAYON_NUM_THREADS=1 RUST_MIN_STACK=268435456 CARGO_BUILD_JOBS=1
cargo run --release -p recursion --features akita,field-inline -- \
  generate --example fibonacci --proofs 1 --workdir /tmp/jolt-recursion
cargo run --release -p recursion --features akita,field-inline -- \
  trace --example fibonacci --workdir /tmp/jolt-recursion
```

`trace` executes the verifier and reports its output; it does not prove that
execution. It stores no trace rows and requires a non-panicking guest with
`Recursion output (trace-only): 1`. Use `--disk` only when the trace
artifact is needed.
The `"verification"` cycle count excludes preprocessing and proof decoding;
`trace length` includes the complete guest execution. This Akita path supports
trace execution only; its outer `verify` operation is not implemented.

For the Dory verifier, omit `akita` from the feature list. Its `verify` command
also proves and verifies the outer execution. Regenerate the inner
proof when changing the commitment backend, protocol, or schedule catalogs.
Keep the same proof input when comparing algebra-preserving guest optimizations.

Add `--embed` to `trace` (or Dory `verify`) to bake the verifier setup into the guest
image. Proofs remain runtime inputs. Prepared binary catalogs, expanded matrices,
and NTT caches are trusted setup data: embedding binds them to the guest image;
in input mode the caller must authenticate the entire prepared setup, including
the expanded matrix, NTT cache, and catalog bytes. The preprocessing digest alone
does not authenticate these detached prepared objects.

For instruction-level attribution, use the explicit disk trace (the execute-only
path does not collect the PC histogram):

```bash
JOLT_BACKTRACE=1 JOLT_PC_PROFILE=/tmp/recursion-pc.txt \
  cargo run --release -p recursion --features akita,field-inline -- \
  trace --disk --example fibonacci --workdir /tmp/jolt-recursion
python3 scripts/guest_pc_profile.py report /tmp/recursion-pc.txt \
  /tmp/jolt-guest-targets/recursion-guest-verify/riscv64imac-zero-linux-musl/release/recursion-guest \
  --top 25
```

Use the ELF produced by that trace build when interpreting its PC profile.

### Retained guest acceptance check

For an Akita build, the example can retain the actual verifier ELF and framed
input before execution. Use a fresh output directory for each artifact; the
prepared form currently supports full, non-embedded input only.

```sh
recursion generate --example fibonacci --proofs 1 --workdir proofs
recursion prepare-guest --example fibonacci --workdir proofs --output accepted
# Inspect and hash accepted/guest.elf and accepted/execution.bin here.
recursion execute-prepared --directory accepted --expect accept
recursion tamper-opening --example fibonacci --workdir proofs --output invalid-proofs
recursion prepare-guest --example fibonacci --workdir invalid-proofs --output rejected
cmp accepted/guest.elf rejected/guest.elf
recursion execute-prepared --directory rejected --expect reject
```

`execute-prepared` uses the retained ELF without rebuilding and rejects a
mismatched host feature profile. Acceptance requires output `1`; rejection
requires output `0`. Both require successful output decoding and no guest panic.
`tamper-opening` changes one typed Stage1 opening by one and reserializes the
proof through the normal record owner; the resulting stream must still decode.
These commands test guest verification, not an Akita outer recursive proof.
