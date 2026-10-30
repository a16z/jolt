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
guest image; proofs remain runtime inputs. In input mode the host
bakes a BLAKE2b-512 digest of the complete setup section into the guest image.
The guest authenticates the setup record and every detached key, catalog, and
terminal-cache body before decoding or trusting them; a missing pin or mismatch
returns `0`. Consumers must pin the intended guest image, including this digest
(see [`specs/recursion-guest.md`](../../specs/recursion-guest.md)). The spec's row
counts are for embedded mode.

### Acceptance and rejection

The recursive statement is: under the setup bound by this guest image, the
entire runtime input contains a nonempty batch of valid inner proof/device
pairs. The count must be at least one and no bytes may follow the last padded
record. Malformed records return `0`; acceptance returns `1`. The output is
still one `u32` and does not commit separately to the count or inner devices.
A consumer checking a particular computation must bind the runtime input and
inspect the intended count and devices; checking the verdict alone establishes
only that some nonempty batch verifies under the image's setup.

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

`tamper --target sumcheck|claim|output|panic|empty` writes a fixture with one
Stage1 round coefficient changed, one Stage1 opening claim changed, one device
output byte flipped, the panic bit flipped, or a zero-proof batch. Run each
through `trace --embed --expect reject`. Generate with `--proofs 2` and trace
normally to test batch acceptance. Input-mode traces authenticate the setup
against the digest baked into the newly built guest.

With `--features field-inline`, `check-field-kernels` runs a separate compiled
guest under the same tracer. Native software folds supply the expected results
for deterministic full-width values and zero, one, and minus one. It covers
row counts 1 through 10 and vector lengths 0 through 11, including every
five-row block tail and four-element tail, product arities 0 through 4, nested
signed-sum iterators, and zero/nonzero inverses. The Akita build requires the
Fp128 row kernels to dispatch; the Dory build checks the BN254 arithmetic
wrappers. CI runs both. The `kernel-tests` guest feature is used only for this
entry point.
