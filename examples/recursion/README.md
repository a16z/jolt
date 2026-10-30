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
postcard-decoded record slice contains a nonempty batch of valid inner
proof/device pairs. The count must be at least one and no bytes may follow the
last padded record within that slice. The entry point also requires every byte
from the end of the postcard envelope to the end of the input capacity to be
zero. The runtime exposes a zero-padded region, not the submitted input length,
so an explicit zero suffix is indistinguishable from unused capacity and is
allowed. A nonzero transport suffix, malformed envelope, or malformed record
returns `0`; acceptance returns `1`. The output is
still one `u32` and does not commit separately to the count or inner devices.
A consumer checking a particular computation must bind the runtime input and
inspect the intended count and devices; checking the verdict alone establishes
only that some nonempty batch verifies under the image's setup.

`trace` requires the guest's verdict to match `--expect` (default `accept`); a
guest panic is neither verdict. `tamper-opening` requires the Akita build and
flips one bit inside the first PCS opening proof body, preserving its envelope
and length prefixes. The stream still decodes and the guest must reject it:

```sh
recursion generate --example fibonacci --workdir proofs
recursion trace --embed --example fibonacci --workdir proofs
recursion tamper-opening --example fibonacci --workdir proofs --output tampered
recursion trace --embed --expect reject --example fibonacci --workdir tampered
```

`check-input-framing --embed --example fibonacci --workdir proofs` builds one
verifier image, checks the honest input, then requires verdict `0` without a
panic for a ten-byte `0xff` length prefix and a length exceeding the input
region. It also rejects a nonzero suffix immediately after the envelope or at
the last capacity byte, and accepts explicit zero padding. Recursion opts into `#[jolt::provable(input_error = 0)]`; decoding errors
and nonzero public-input padding skip the verifier body and serialize that
return value. Other guests retain
the default decoding behavior.

These commands test guest verification, not an Akita outer recursive proof.

`tamper --target sumcheck|claim|output|panic|empty` writes a fixture with one
Stage1 round coefficient changed, one Stage1 opening claim changed, one device
output byte flipped, the panic bit flipped, or a zero-proof batch. Run each
through `trace --embed --expect reject`. Generate with `--proofs 2` and trace
normally to test batch acceptance; `tamper --target claim --proof-index 1`
then changes only the second proof, and the batch must return `0`.
The Dory lane also checks framing and rejects a changed claim. Input-mode
traces authenticate the setup
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
