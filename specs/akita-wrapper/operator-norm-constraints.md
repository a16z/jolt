# D64 exact candidate norm constraints

This component proves acceptance of one dense D64 mixed-shell polynomial. It
constrains exactly 31 coefficients of magnitude one, 11 of magnitude two and 22
zeros, then all 32 native fixed-point upper inequalities. It does not prove that
the polynomial came from transcript bytes, position selection, the correct retry,
or the first accepted attempt. It is not a complete sampler or wrapper.

## Source and ownership

The source contract is Akita `28fc72021c120e7bc4e0102e07844f25299051eb`,
`akita-challenges/src/sampler/op_norm.rs`, `decide_parts`, together with the
production `D64_SELECTIVE_L2_OP_NORM_TABLE` initialization. The independently
checked capacity packet is `sampling-capacity-review.md` in the original research
workspace. That review establishes the actual table error is **1**; the policy
ceiling 4 is not the error used by the native predicate.

A separate native API change exposes the existing production table by a fallible
getter and read-only slices/accessors. Fields and table construction remain
private. The native strict predicate's existing body is unchanged. There is no
second trigonometric table builder and no hand-copied table in Jolt. The constrained
consumer checks D64 shell counts, threshold 18, fractional bits 48, actual error
1, table lengths 2048 and each entry's absolute bound 2^48 in release builds.
Entries have native layout `position * 32 + frequency`, including the native
odd-frequency order. These compile-time constants belong to the circuit identity;
the eventual verification key must authenticate that identity.

`jolt-r1cs::bn254_bits::BitVar` exposes the existing Boolean allocation, and
`integer_bn254::SignedVar::absolute_value` adds bounded absolute value with its
first caller here. Protocol-specific shell and frequency composition remain in
`jolt-akita::r1cs::operator_norm`. No extra foundational crate is introduced.

## Constraint argument

For each coefficient c, Boolean u,d,s satisfy `u*d=0` and
`(u+2*d)*(1-2*s)=c`. The two aggregate equalities sum(u)=31 and sum(d)=11
establish the exact shell, hence L1=53. These small integer sums cannot wrap in
BN254. Signed field representatives are intentional; no positivity convention is
assumed for c. Zero has two equivalent sign assignments, which do not change c.

For each native frequency k, linear combinations of the same coefficient handles
form R and I. Their bounded signed handles have bound `B=53*2^48`; equality to the
linear combinations is an integer equality because the shell bounds the latter
and the difference has magnitude at most 2B, less than the BN254 scalar modulus.
Both real and imaginary accumulators use the production table slices.

Absolute value allocates an unsigned magnitude m in [0,B] and a Boolean s, then
constrains `m*(1-2*s)=x`. Existing exact unsigned range constraints include both
value and complement decompositions; they do not rely on honest assignment.
Since 2B<r, this equality fixes m=|x| without modular aliases. This generic helper
also supports the existing signed type's largest allowed bound: 2B<2^128<r.

With radius rho=53, each frequency constrains

`U=R^2+I^2+2*rho*(|R|+|I|)+2*rho^2`

and equality to a signed variable bounded by `T=18^2*2^96`. Signed comparison is
sufficient here: U is nonnegative, U<2^110, and U+T<2^111<r, so a negative signed
representative cannot satisfy the equality. Thus the exact interval constraint
is equivalent to 0<=U<=T. Squares, absolute values, ranges, bindings and comparison
ranges are all included. All 32 inequalities are emitted. Native lower-bound
early rejection changes execution only: it cannot accept a candidate that fails
an upper inequality. Native inconclusive intervals reject, matching this AND.

Every parameter entering these inequalities is fixed and release checked before
constraint construction; this is not an unchecked generic norm API. All products
and native witness calculations fit u128/i128 for this profile.

## Witness and interface boundary

`D64ShellVar::bind` consumes 64 existing coefficient handles and optional dense
witness values. Values only generate auxiliaries; shell and frequency equations
bind the actual input handles. `None` emits the same matrices as an accepted known
assignment. Known rejected assignments return typed errors; construction is not
transactional, so callers must discard a builder after error. Handles must remain
in their allocating builder. Index membership is checked, but matching indices
from a different builder do not establish provenance. The outer statement must
fix the constant-one column.

Shell/sign/range witnesses and the norm auxiliaries are private only if the
future outer SNARK hides its witness. Circuit shape is independent of accepted
witness values. This component makes no standalone zero-knowledge claim. Binding
the dense coefficients to sparse positions, unbiased sign/magnitude selection,
stream consumption and retry acceptance remain separate obligations.

## Verification and reproducibility

Permanent tests compare a historical accepted polynomial and two rejected shell
polynomials against the live native predicate. One rejected example fails after
frequency zero. Known/unknown matrices agree. Completed witnesses are modified at
input coefficients, shell count bits, absolute-value signs and the final comparison
variable. Synthetic accumulator tests distinguish both signs immediately around
the exact upper boundary; a forged honest-generation hint cannot satisfy the
actual comparison equation for an out-of-bound accumulator.

The intentional `operator_norm_r1cs_cost` example counts the entire component with
64 input handles, shell constraints and all frequencies. This is a matrix census,
not a timing or complete-wrapper performance measurement.

The native API is published at `e24f3b6f54bfcfb378a9ad65a0609e1bdd1ccdc1`
in `https://github.com/markosg04/akita.git` (draft PR #2). All direct Akita
workspace dependencies and the checked-in lock use that revision. The three API
source files are byte-identical to reviewed candidate `6b17d7158`; the public
revision additionally includes the fork CI fixes. Integration validation below
uses the public Git graph without the original local API overlays.

The measured complete component has **22,370 rows, 21,985 variables (including
ONE), and 107,012 matrix nonzeros**. Inputs and every shell, range, absolute-value,
frequency binding and comparison row are included. No retry multiplier is applied.

The original component validation passed 69/69 Jolt library tests in both default
and minimal configurations, plus 38/38 native challenge tests, scoped clippy,
formatting and Taplo. Its local-overlay evidence remains archived under
`/private/tmp/norm-r1cs-*` and `/private/tmp/norm-native-*`.

## Published dependency integration

The focused integration is based on `153a0351a` (reviewed sparse Blake stream),
with only the reviewed norm component cherry-picked. Additive module/manifest
conflicts retain both independent components. No reviewed constraint semantics
change. The public native API revision supersedes the original publication
limitation; complete sampler/stream-to-coefficient binding remains absent.

Public graph validation commands use `CARGO_INCREMENTAL=0` and
`CARGO_TARGET_DIR=../crypto-r1cs/target`, with no `.git/validation.toml` or other
native path overlay:

```sh
cargo metadata --format-version 1
cargo nextest run --locked -p jolt-r1cs -p jolt-akita --lib --features jolt-akita/r1cs -j 1 --cargo-quiet --status-level fail --final-status-level fail
cargo nextest run --locked -p jolt-r1cs -p jolt-akita --lib --no-default-features --features jolt-akita/r1cs -j 1 --cargo-quiet --status-level fail --final-status-level fail
cargo clippy --locked -p jolt-r1cs -p jolt-akita --lib --tests --features jolt-akita/r1cs -q -- -D warnings
cargo clippy --locked -p jolt-r1cs -p jolt-akita --lib --tests --example operator_norm_r1cs_cost --no-default-features --features jolt-akita/r1cs -q -- -D warnings
cargo fmt --check -q
taplo format --check Cargo.toml crates/jolt-akita/Cargo.toml
```

Public graph evidence and raw logs are `/private/tmp/norm-public-*`. These scoped
checks do not replace full workspace or end-to-end proving validation.

Public integration results: default and minimal suites each passed **69/69**,
respectively runs `fa193c4f-9ec7-4311-b848-1a398493a88b` and
`ba554dc0-e921-48a1-927c-c33f64f4a04c`. Both clippy commands exited zero, as did
formatting and Taplo. Test/clippy `.log` and `.exit` records are retained under
`/private/tmp/norm-public-*`. `/private/tmp/norm-public-graph.txt` records all
14 native Git identities and the three reviewed API byte comparisons; full
metadata is `/private/tmp/norm-public-metadata.json`. Native public fetch needed
network-enabled execution; tests and clippy ran without an API overlay.
The four norm implementation/example files are byte-identical to reviewed
`ec410d81111eb74550a17e656afc3179742f7fac`; the earlier component count therefore
still describes these unchanged matrices. No new timing or whole-sampler claim
is made by the dependency integration.
