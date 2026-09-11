# Montgomery NTT inline

`forward_ntt64` computes a twisted, 64-point forward DIF NTT with signed
32-bit Montgomery coefficients (`R = 2^32`) and bit-reversed output. The
caller supplies the modulus, Montgomery inverse, twist powers, and stage
twiddles. See the function's API contract for their bounds and layout.

The inline keeps all 64 coefficients in virtual registers across the six
butterfly stages. It expands to 4,456 rows of existing proved integer
instructions, including register resets. It adds no advice or new proof
constraints. Arithmetic has explicit wrapping semantics even outside the
NTT parameter domain; supplying valid roots remains the caller's job.

RISC-V calls require 8-byte alignment for paired memory accesses. The safe
API copies to aligned stack buffers when an array lacks that alignment;
non-RISC-V targets always use the portable implementation. A host that
executes the inline must link this crate with its `host` feature.

Validation:

```sh
cargo nextest run -p jolt-inlines-ntt -p jolt-inlines-fixtures --features host --cargo-quiet
RAYON_NUM_THREADS=1 RUST_MIN_STACK=268435456 CARGO_BUILD_JOBS=1 cargo run --release -p ntt-example
```

The unit tests compare against a direct DFT, cover signed boundary inputs,
and check wrapping arithmetic. The example proves a guest execution and
checks its output against an independently computed DFT checksum.

Akita recursion opts in with the `ntt-inline` feature on the `recursion`
package. This forwards to the companion Akita algebra crate and registers
the inline on the host. The companion dependency currently resolves through
the Jolt workspace's local patch, pending publication of this crate.
