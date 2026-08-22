# Fp128 HOL Light proofs

These proofs cover AArch64 addition and subtraction for
`Prime128OffsetA7F7`. They prove the exact instruction words imported from the
standalone objects. Production Rust includes the same instruction body files.

Start with the Book chapter
[Formal verification of field kernels](../../book/src/how/formal-verification/field-kernels.md).
It explains the arithmetic, theorem shape, byte connection, and trust
boundary.

The final theorems cover callable functions, including `ret` and the AArch64
procedure call convention. They assume canonical inputs and prove the
canonical result modulo `0xffffffffffffffffffffffff00005809`.

The public witness check proves that optimized calls through
`Prime128OffsetA7F7` contain the same instruction words. Jolt does not inspect
an Akita executable. Akita must perform that final binary check at its pinned
Jolt revision.

## Requirements

You need an AArch64 host, `llvm-objdump`, an OCaml environment that can build
HOL Light proofs, and local checkouts of HOL Light and `s2n-bignum`.

CI pins these revisions.

* HOL Light commit `433477862bb90b328a593e012e09390e99b2439b`
* `s2n-bignum` commit `ac31a43db30953037abd1b64b540e65cf31f4c67`

## Run every check

```sh
HOL_LIGHT_DIR=/path/to/hol-light \
S2N_BIGNUM_DIR=/path/to/s2n-bignum \
  ./proofs/hol-light/check.sh
```

The script uses a fresh Cargo target directory. It requires one fresh addition
object and one fresh subtraction object. It builds the optimized public
witness and checks every instruction word. It then builds and runs both HOL
Light subroutine theorems. Temporary files are removed when the script exits.

Packed SIMD operations and multiplication are outside the present proof scope.
