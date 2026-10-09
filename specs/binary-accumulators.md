# Spec: Deferred Reduction for the Binary Fields

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | Quang Dao, Claude              |
| Created     | 2026-10-09                     |
| Status      | proposed                       |
| PR          |                                |

## Summary

The prover's inner loops have the shape `acc += a * b`, and every field-generic kernel reaches them through `WithAccumulator`. The binary fields of `specs/binary-field.md` currently use `NaiveAccumulator`, which reduces after every product. In characteristic 2 an unreduced product is a polynomial over $\mathbb F_2$ and addition is XOR, which never carries, so any number of unreduced products can be summed in a fixed-width accumulator and reduced once. This spec gives `F64`, `F128` and `F192` such accumulators. It is step 2 of the roadmap in `specs/binary-field.md`.

## Intent

### Goal

Replace `NaiveAccumulator` by an XOR accumulator of unreduced carry-less products in the `WithAccumulator` impls of `F64`, `F128` and `F192`, with no change to any value a caller can observe.

The accumulator of each field holds the unreduced product polynomial:

| Field | Accumulator state | One `fmadd` | `reduce` |
|---|---|---|---|
| `F64` | 127-bit polynomial, one `u128` | one 64-bit carry-less multiply, XOR | one reduction by $x^{64}+x^4+x^3+x+1$ |
| `F128` | 255-bit polynomial, two `u128` | the carry-less multiplies of one 128-bit product, XOR | one reduction by $x^{128}+x^7+x^2+x+1$ |
| `F192` | three unreduced `F64` coefficients, three `u128` | the carry-less multiplies of one product over `F64`, folded by $y^3 = y+1$ and $y^4 = y^2+y$, XOR | three `F64` reductions |

For `F192` the reduction by $y^3+y+1$ is applied before accumulation and the `F64` reductions after. The two commute: the first only XORs coefficients, and `F64` reduction is $\mathbb F_2$-linear. The kernel multiply already orders them this way.

### Invariants

1. **Exactness without a bound.** For every sequence of `add`, `fmadd`, `fmadd_*` and `merge` calls, of any length, `reduce` returns the field element that `NaiveAccumulator` returns for the same sequence. There is no headroom limit and no count to track: XOR does not carry, and the state width is fixed by the degree of one product.
2. **Integer scalars are parity.** `fmadd_u8`, `fmadd_u64`, `fmadd_u128`, `fmadd_i64`, `fmadd_i128`, `fmadd_signed_u64`, `fmadd_s256` and `fmadd_bool` add `a` when the scalar is odd (for `fmadd_bool`, true) and do nothing otherwise, with no multiplication. The sign is ignored, since $-1 = 1$. This is what the trait defaults compute through `from_u64` and its siblings; the overrides only remove the multiply.
3. **One accumulator type per field.** `Accumulator`, `SmallScalarAccumulator` and `SignedProductAccumulator` of a field are the same type. The three exist to give prime fields differently shaped integer slots, and a binary field has one shape.
4. **`Mul` is unchanged.** Field multiplication and squaring return the same values as before, on the portable path and on each kernel path. On a kernel path the multiply is the composition of the unreduced product and the reduction that the accumulator uses, so the two cannot drift apart. The portable multiplies of `binary/portable.rs`, which reduce while they shift, are left as they are: they are the oracle of the kernel differential test and stay independent of the code under test.
5. **Both arithmetic paths.** The accumulators work on every target: with a carry-less-multiply kernel where `binary/mod.rs` selects one, and with a portable unreduced multiply (shift and XOR into the double-width word) elsewhere. The two paths produce the same accumulator state bit for bit, not only the same reduced value. The reductions are plain shifts and XORs with no architecture dependence and are shared by both paths.
6. **`Unreduced` and `MulBaseUnreduced` are not implemented.** Their contract is stated in terms of integer slots (`Self × u64` products, signed `i32` lanes, a documented headroom, `SUM_IS_EXACT`), none of which describes a polynomial over $\mathbb F_2$. Their only callers are concrete-field code paths for `Fp128` in `jolt-akita`. The module documentation of `unreduced.rs` says so, in one sentence, so that the absence reads as a decision.
7. **No other change.** The spine traits, `NaiveAccumulator`, the other backends, and `F8` are untouched. `F8` keeps `NaiveAccumulator`: its multiplication is not on a hot path.

### Non-Goals

- A deferred `F192 × F64` product. `Accumulator::fmadd` takes two elements of the same field, and no caller in this repository accumulates extension-times-base products over a binary field. When one exists, it gets a method on the same three-word state.
- Vectorised accumulation (`vpclmulqdq`, packed inputs). That is step 4 of the roadmap.
- Changing any kernel in `jolt-kernels` or `jolt-sumcheck`. They are generic over `WithAccumulator` and pick the new accumulators up without a source change.
- An accumulator for `F8`.

## Evaluation

### Acceptance Criteria

- [ ] `<F64 as WithAccumulator>::Accumulator`, `SmallScalarAccumulator` and `SignedProductAccumulator` are one type, and likewise for `F128` and `F192`; none is `NaiveAccumulator`. Checked by a compile-time type-equality assertion in the tests.
- [ ] Differential test against `NaiveAccumulator<F>` for each of the three fields: a seeded random script of at least 10,000 operations drawn from `add`, `fmadd`, every `fmadd_*` variant and `merge` of an independently built partial accumulator, applied to both; the reduced results agree after every 97th operation and at the end.
- [ ] Fixed edge cases for each field, compared with the value computed by `Mul` and `Add`: the empty accumulator reduces to zero; one `fmadd` of two all-ones raw words; `fmadd(a, b)` twice reduces to zero; `fmadd(a, 0)` and `fmadd(0, b)`; `fmadd` of the two operands whose product has the highest possible degree.
- [ ] No headroom: $2^{20}$ `fmadd` calls with all-ones operands for `F64`, reduced once, equal the expected value (zero, since the count is even), and $2^{20}+1$ calls equal one product.
- [ ] Scalar parity, for each field and a nonzero `a`: `fmadd_u64(a, 2)`, `fmadd_i64(a, i64::MIN)`, `fmadd_u128(a, 1 << 100)`, `fmadd_bool(a, false)` leave the accumulator at zero; `fmadd_u64(a, 3)`, `fmadd_i64(a, -1)`, `fmadd_i128(a, -1)`, `fmadd_signed_u64(a, 5, false)`, `fmadd_s256` with a negative odd magnitude and with an odd magnitude whose upper limbs are nonzero, and `fmadd_bool(a, true)` each reduce to `a`.
- [ ] `merge` is associative and commutative on three independently built accumulators, compared after `reduce`.
- [ ] Path agreement, in the in-crate test module, for the three fields on the existing differential inputs: the portable unreduced product reduces to the portable multiply; and, on kernel targets, the kernel and portable unreduced products are equal word for word.
- [ ] The existing vector and contract tests for `F64`, `F128`, `F192` pass unchanged, which covers invariant 4.
- [ ] `benches/binary_kernels.rs` gains, per field, a group that times a 1024-term `fmadd` loop followed by one `reduce` for the new accumulator and for `NaiveAccumulator`.
- [ ] `cargo clippy -p jolt-field --all-targets -- -D warnings` passes with `--no-default-features --features binary`, `--no-default-features --features binary,allocative` and `--features solinas,binary,allocative`; `cargo nextest run -p jolt-field --no-default-features --features binary --cargo-quiet` passes; `cargo nextest run -p jolt-sumcheck --cargo-quiet` passes (its binary-field tests drive the accumulators through the generic prover); `cargo fmt --check` passes.
- [ ] The four binary configurations of `.github/workflows/field-portability.yml` run the new tests without a workflow change.

### Testing Strategy

A new `crates/jolt-field/tests/binary_accumulators.rs`, gated on `binary`. Ground truth is `NaiveAccumulator`, which is built from `Mul` and `Add` alone, and those are pinned by the frozen vectors of `tests/binary_vectors.rs`. The accumulator under test shares only the unreduced multiply and the reduction with `Mul`, and the path-agreement test covers those.

### Performance

Informational, not a merge gate. On a kernel target, the PR records the time per term of the 1024-term loop for both accumulators and the three fields. What is saved per term is the reduction, a chain of about ten shifts and XORs for `F64` and `F128` and three such chains for `F192`, against one, three and six carry-less multiplies for the product. The size of the gain is therefore an empirical question, and the spec does not predict it. If a field shows no gain, the PR says so and keeps the accumulator, because invariant 2 removes multiplies from the scalar paths in any case.

No `jolt-eval` objective moves: no shipped configuration proves over a binary field.

## Design

### Architecture

Each multiply in `binary/kernels.rs` already consists of an unreduced product followed by `reduce64` or `reduce128`. The products are given names, the multiplies become their composition with the reduction, and the accumulators call the two separately. For `F192` the unreduced product is the three words that the kernel passes to `reduce64` today. `binary/portable.rs` gains unreduced products with the same signatures, written as shift and XOR into the double-width word; its existing multiplies are not rewritten in terms of them. `reduce64` and `reduce128` move to where both paths can use them.

The accumulator types live in `binary/accumulator.rs`. They are `Copy`, hold raw words, and are not exported from the crate root: callers name them only through `WithAccumulator`.

`add(value)` XORs the element into the low part of the state, which is its own unreduced representation. `merge` XORs two states. `reduce` runs the reduction half once.

The reason one type serves all three associated types is invariant 2. For a prime field, `SmallScalarAccumulator` exists because a `field × u64` product fits a narrower integer slot than a `field × field` product. Here a `u64` scalar acts as its low bit, so the small-scalar path is a conditional XOR into the same state.

### Why `Unreduced` does not apply

`Unreduced` (`crates/jolt-field/src/unreduced.rs`) describes three integer accumulators and the number of terms each can absorb before a slot overflows. A binary field needs none of that vocabulary: its unreduced product has one shape, its sum is exact for any number of terms, and "scale by a small integer" is a parity test. Implementing the trait would mean choosing associated types to satisfy a signature (`Wide: From<Self>` scaled by `i32`, `SmallProduct` for `Self × u64`) that no caller would use. `WithAccumulator` is the surface the generic kernels consume, and it expresses everything a binary field can defer.

### Alternatives Considered

1. **Keep `NaiveAccumulator`.** Correct, and the simplest. Rejected because the accumulator is the hot-loop interface of every kernel, and deferring reduction is free of the headroom analysis that makes it delicate for prime fields.
2. **Implement `Unreduced` as well.** See above.
3. **Separate small-scalar accumulator types.** They would be the same state with the same methods.
4. **Accumulate reduced elements and override only the scalar variants.** Removes the multiplies of invariant 2 but keeps a reduction per `fmadd`.

## Documentation

`binary/mod.rs` module docs gain a paragraph on deferred reduction. `unreduced.rs` gains the sentence of invariant 6. No book change.

## Execution

Name the kernel unreduced products and add the portable ones, with the existing tests green; add `binary/accumulator.rs` and switch the three `WithAccumulator` impls; add the tests and the bench group. One commit.

## References

- `specs/binary-field.md`, roadmap step 2.
- `crates/jolt-field/src/algebra.rs`, `Accumulator` and `WithAccumulator`.
- `crates/jolt-field/src/fp128_accumulators.rs`, the prime-field accumulators this mirrors in role.
