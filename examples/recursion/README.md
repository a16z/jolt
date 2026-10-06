# Recursion verifier guest

Generate an inner Fibonacci proof, then execute its verifier as a RISC-V guest:

```bash
export RAYON_NUM_THREADS=1 RUST_MIN_STACK=268435456 CARGO_BUILD_JOBS=1
cargo run --release -p recursion --features akita,field-inline,ntt-inline -- \
  generate --example fibonacci --proofs 1 --workdir /tmp/jolt-recursion
cargo run --release -p recursion --features akita,field-inline,ntt-inline -- \
  trace --example fibonacci --workdir /tmp/jolt-recursion
```

`trace` executes the verifier and reports its output; it does not prove that
execution. It stores no trace rows and requires a non-panicking guest with
`Recursion output (trace-only): 1`. Use `--disk` only when the trace
artifact is needed.
The `"verification"` cycle count excludes preprocessing and proof decoding;
`trace length` includes the complete guest execution. This Akita path supports
trace execution only; its outer `verify` operation is not implemented.

For the Dory verifier, build with `--features field-inline`; its `verify`
command also proves and verifies the outer execution.

Add `--embed` to `trace` (or Dory `verify`) to bake the verifier setup into the
guest image; proofs remain runtime inputs. The setup is trusted data, so in
input mode the caller must authenticate it (see the trust model in
[`specs/recursion-guest.md`](../../specs/recursion-guest.md)). The spec's row
counts are for embedded mode.

### Acceptance and rejection

`trace` requires the guest's verdict to match `--expect` (default `accept`); a
guest panic is neither verdict. `tamper-opening` changes one typed Stage1
opening by one and reserializes the proof, so the stream still decodes and the
guest must reject it:

```sh
recursion generate --example fibonacci --workdir proofs
recursion trace --embed --example fibonacci --workdir proofs
recursion tamper-opening --example fibonacci --workdir proofs --output tampered
recursion trace --embed --expect reject --example fibonacci --workdir tampered
```

These commands test guest verification, not an Akita outer recursive proof.
