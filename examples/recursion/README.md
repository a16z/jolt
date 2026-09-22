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
files also use the work directory. Use a distinct work directory for each run.

`trace` executes the verifier and reports its output; it does not prove that
execution. Successful verification returns `Recursion output (trace-only): 1`.
The `"verification"` cycle count excludes preprocessing and proof decoding;
`trace length` includes the complete guest execution. Use `verify` instead of
`trace` to also prove and verify the outer execution.

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
