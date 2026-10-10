# Spec: Deferred Reduction for the Binary Fields

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | Quang Dao, Claude              |
| Created     | 2026-10-09                     |
| Status      | proposed                       |
| PR          |                                |

## Summary

The prover's inner loops have the shape `acc += a * b`, and the optimized kernels reach them through `WithAccumulator`. The binary fields of `specs/binary-field.md` currently use `NaiveAccumulator`, which reduces after every product. In characteristic 2 an unreduced product is a polynomial over $\mathbb F_2$ and addition is XOR, which never carries, so any number of unreduced products can be summed in a fixed-width accumulator and reduced once. This spec gives `F64`, `F128` and `F192` such accumulators. It is the accumulator part of step 2 of the roadmap in `specs/binary-field.md`. It also settles the contract question that step raised, by leaving the binary fields outside `Unreduced` and `MulBaseUnreduced`, and it defers a specialised extension-times-base product until something consumes one.

## Intent

### Goal

Replace `NaiveAccumulator` by an XOR accumulator of unreduced carry-less products in the `WithAccumulator` impls of `F64`, `F128` and `F192`, with no change to any value a caller can observe.

The accumulator of each field holds the unreduced product polynomial:

| Field | Accumulator state | One `fmadd` | `reduce` |
|---|---|---|---|
| `F64` | 127-bit polynomial, one 128-bit word | one 64-bit carry-less multiply, XOR | one reduction by $x^{64}+x^4+x^3+x+1$ |
| `F128` | 255-bit polynomial. Portable: two `u128` `[low, high]` for `low + x^128·high`. Kernel: three 128-bit words `[t0, t1, t2]` for `t0 + x^64·t1 + x^128·t2` | the carry-less multiplies of one 128-bit product, XOR | one reduction by $x^{128}+x^7+x^2+x+1$ |
| `F192` | three unreduced `F64` coefficients `[C0, C1, C2]` in ascending degree of $y$, three 128-bit words | the carry-less multiplies of one product over `F64`, folded by $y^3 = y+1$ and $y^4 = y^2+y$, XOR | three `F64` reductions |

On a kernel path a 128-bit word is a vector register, the one the carry-less multiply writes its result to, so that a product and the running state never leave that register file inside a loop; on the portable path it is a `u128`. The `F128` kernel state keeps the middle word of the product whole: one more word of state, in exchange for not splitting that word across `low` and `high` on every product. `F64` on AArch64 is a measured exception and keeps a scalar `u128` state, because the scalar XOR recurrence was the faster one there.

For `F192` the reduction by $y^3+y+1$ is applied before accumulation and the `F64` reductions after. The two commute: the first only XORs coefficients, and `F64` reduction is $\mathbb F_2$-linear. The kernel multiply already orders them this way.

### Invariants

1. **Exactness without a bound.** For every sequence of `add`, `fmadd`, `fmadd_*` and `merge` calls, of any length, `reduce` returns the field element that `NaiveAccumulator` returns for the same sequence. There is no headroom limit and no count to track: XOR does not carry, and the state width is fixed by the degree of one product.
2. **Integer scalars are parity.** `fmadd_u8`, `fmadd_u64`, `fmadd_u128`, `fmadd_i64`, `fmadd_i128`, `fmadd_signed_u64`, `fmadd_s256` and `fmadd_bool` add `a` when the scalar is odd (for `fmadd_bool`, true) and do nothing otherwise, with no multiplication. The sign is ignored, since $-1 = 1$. This is what the trait defaults compute through `from_u64` and its siblings; the overrides only remove the multiply.
3. **One accumulator type per field.** `Accumulator`, `SmallScalarAccumulator` and `SignedProductAccumulator` of a field are the same type. The three exist to give prime fields differently shaped integer slots, and a binary field has one shape.
4. **`Mul` is unchanged.** Field multiplication and squaring return the same values as before, on the portable path and on each kernel path. On a kernel path the multiply is the composition of the unreduced product and the reduction that the accumulator uses, so the two cannot drift apart. The portable multiplies of `binary/portable.rs`, which reduce while they shift, are left as they are: they are the oracle of the kernel differential test and stay independent of the code under test.
5. **Both arithmetic paths.** The accumulators work on every target: with a carry-less-multiply kernel where `binary/mod.rs` selects one, and with a portable unreduced multiply (shift and XOR into the double-width word) elsewhere. After the same calls the two paths hold the same polynomial, and so reduce to the same element; they need not hold it in the same words. Every representation is closed under XOR and its reduction is $\mathbb F_2$-linear, which is all that invariant 1 uses. The portable reductions are shifts and XORs. A kernel reduces by carry-less multiplication with the low word of the modulus, or by the same shifts on vector lanes where that measured faster.
6. **`Unreduced` and `MulBaseUnreduced` are not implemented.** Their documented semantics are integer slots (`Self × u64` products, signed `i32` lanes, a headroom, `SUM_IS_EXACT`), and nothing consumes them for a binary field: the field-generic sum-check, polynomial and kernel layers require neither trait, and the current consumers of `Unreduced` are the `Fp128` accumulators in `jolt-field` and concrete `Fp128` commitment paths in `jolt-akita`. The module documentation of `unreduced.rs` says, in one sentence, that binary fields defer reduction through `WithAccumulator` instead.
7. **No other change.** The spine traits, `NaiveAccumulator`, the other backends, and `F8` are untouched. `F8` keeps `NaiveAccumulator`: its multiplication is not on a hot path.

### Non-Goals

- A specialised deferred `F192 × F64` product. `Accumulator::fmadd` takes two elements of the same field, and no caller in this repository accumulates extension-times-base products over a binary field. `ExtField::mul_base` and `fmadd` of a lifted base element remain available; the cheaper three-multiply path gets a method on the same three-word state when it has a caller.
- Vectorised accumulation (`vpclmulqdq`, packed inputs). That is step 4 of the roadmap.
- Changing any consumer. Code generic over `WithAccumulator`, which today is the optimized tier of `jolt-kernels`, picks the new accumulators up without a source change.
- An accumulator for `F8`.

## Evaluation

### Acceptance Criteria

- [ ] `<F64 as WithAccumulator>::Accumulator`, `SmallScalarAccumulator` and `SignedProductAccumulator` are one type, and likewise for `F128` and `F192`. The bound helper in `tests/binary_contract.rs`, which today requires `NaiveAccumulator`, is changed to require `JoltField` and equality of the three associated types; nothing else in that file's existing assertions changes. A unit test inside the backend asserts that each associated type is the intended concrete accumulator.
- [ ] Frozen results, for each field, written out as raw words. The expected values come from the existing externally generated product fixtures of `tests/binary_vectors.rs` and from the moduli, never from calling `Mul` in the test:
  - `fmadd` of a fixture's operands reduces to the fixture's product, for several asymmetric fixtures per field.
  - A mixed sequence of several fixture `fmadd`s, reduced `add`s, and the `merge` of a separately built partial state reduces to one frozen word, equal to the XOR of the fixture products and addends.
  - For `F128`: `fmadd(from_raw(1 << 127), from_raw(2))` reduces to raw `0x87`, and a following `add(from_raw(1))` gives `0x86`; `fmadd(from_raw(1 << 127), from_raw(1))` reduces to raw `1 << 127`; `fmadd` of `from_raw(1 << 127)` with itself reduces to `0xc0000000000000000000000000001067`. These separate the low and high words, the placement of `add`, and the second reduction fold.
  - For `F64`: the same three shapes with `1 << 63`, expected `0x1b`, `0x1a`, `1 << 63`, and top square `0xc00000000000005a`.
  - For `F192`: fixtures whose operands are supported on a single coefficient each, for every pair of coefficient positions, so that a swapped or misfolded coefficient changes the result.
- [ ] Algebraic properties, each on states that also carry a nonzero frozen result so that an accumulator that discards everything fails: the empty accumulator reduces to zero; `fmadd(a, b)` twice cancels; `fmadd(a, 0)` and `fmadd(0, b)` are no-ops; `merge` is commutative and associative on three independently built states; splitting one operation sequence at any of several points and merging the halves gives the result of the unsplit sequence.
- [ ] Persistence across many terms: after one `add` of a nonzero sentinel, $2^{16}$ identical `fmadd` calls reduce to the sentinel and one more reduces to the sentinel plus the frozen product. (A finite count does not prove invariant 1; the fixed state width does. The test guards against a counter or periodic reset.)
- [ ] Scalar parity: the existing signed-accumulator test of `tests/binary_contract.rs`, which retargets to the new types through the associated type, is extended with the `u8`, `u64`, `u128`, `i64` and `bool` variants, including `i64::MIN` and `1 << 100`, each applied to an empty state and to a state already holding a nonzero value.
- [ ] Path agreement, in the in-crate test module. On every target: the portable unreduced product, reduced, equals the portable interleaved multiply, for the three fields on the existing differential inputs. On kernel targets in addition: the kernel unreduced product, read back as a polynomial, equals the portable one word for word, and every product formulation and every reduction in the kernel module agrees with the portable multiply, including the ones the target does not select. The portable comparison is compiled under `cfg(test)` on all targets, with only the kernel half behind the hardware predicate, and the existing test name `binary::tests::kernel_matches_portable` is kept because the portability workflow checks for it.
- [ ] The existing vector tests pass unchanged, and the existing contract tests pass with the one change named above. This covers invariant 4.
- [ ] `benches/binary_kernels.rs` gains, per field, a group that times a 1024-term `fmadd` loop followed by one `reduce`, for the new accumulator and for `NaiveAccumulator`, over the same pre-generated varying operand pairs, with generation outside the timed region and inputs and output passed through `black_box`.
- [ ] `cargo clippy -p jolt-field --all-targets -- -D warnings` passes with `--no-default-features --features binary`, `--no-default-features --features binary,allocative` and `--features solinas,binary,allocative`; `cargo nextest run -p jolt-field --no-default-features --features binary --cargo-quiet` passes; `cargo nextest run -p jolt-sumcheck --cargo-quiet` passes as a regression check (its tests use `Add` and `Mul`, not the accumulators); `cargo fmt --check` passes.
- [ ] The four binary configurations of `.github/workflows/field-portability.yml` run the new tests without a workflow change.

### Testing Strategy

A new `crates/jolt-field/tests/binary_accumulators.rs`, gated on `binary`, plus the two edits to `tests/binary_contract.rs` and the in-crate path test. Ground truth is the externally generated product fixtures, values that follow from the moduli by hand, and algebraic laws of accumulation. The portable interleaved multiply is a second maintained arithmetic path, independent of the unreduced products and reductions under test, and the path-agreement test compares against it.

`NaiveAccumulator` is not a permanent oracle here: comparing the replaced accumulator with its replacement is a transition check, and its `fmadd` calls the `Mul` that shares code with the accumulator on kernel targets. A randomised comparison against it is run once during implementation and reported in the PR; it is not committed.

### Performance

Informational, not a merge gate. On a kernel target, the PR records the time per term of the 1024-term loop for both accumulators and the three fields, with the architecture and target features. The CI benchmark comment compares a benchmark with itself across base and head and lists new benchmarks without timings, so these numbers are recorded by hand. What is saved per term is the reduction, a chain of about ten shifts and XORs for `F64` and `F128` and three such chains for `F192`, against one, three and six carry-less multiplies for the product. The size of the gain is therefore an empirical question, and the spec does not predict it. If a field shows no gain, the PR says so and keeps the accumulator, because invariant 2 removes multiplies from the scalar paths in any case.

No `jolt-eval` objective moves: no shipped configuration proves over a binary field.

## Design

### Architecture

Each multiply in `binary/kernels.rs` is an unreduced product followed by a reduction. The products are named, the multiplies are their composition with the reduction, and the accumulators call the two separately. For `F192` the unreduced product is the three words that the kernel reduces. `binary/portable.rs` has unreduced products and reductions with the same names, written as shift and XOR into the double-width word; its interleaved multiplies are not rewritten in terms of them.

The kernels are written once, over a 128-bit word type that each file of `binary/arch/` defines for its architecture. That type owns the lane conventions and every `unsafe` intrinsic call. A product, its reduction and the accumulator state are values of it. Where two formulations of the same map exist (Karatsuba or schoolbook for the `F128` product, multiplication or shifts for a reduction, vector or scalar state for `F64`), the architecture file selects one by a constant, from paired measurement on that architecture, and the differential test runs both.

On x86_64 a word is built from a `u128` in one of two ways. An accumulator operand is loaded as 16 bytes. A multiplication operand is assembled from its two halves by one `punpcklqdq`, written with `asm!` so that the compiler cannot merge the two lane moves into a 16-byte load. Such a load, placed behind two 8-byte stores of the same value, cannot be forwarded from the store buffer, and that stall doubled the time of a single multiplication.

The accumulator types live in `binary/accumulator.rs`. They are `pub`, as an associated type of a public trait impl must be, with private raw-word state; they derive `Default`, `Clone` and `Copy` and are not re-exported from the crate root, so callers name them only through `WithAccumulator`. The specialised squarings keep their two- and three-multiply schedules and are not routed through the full product.

`add(value)` XORs the reduced element into the state as its own unreduced representation: into the low 64 bits of the word for `F64`, into the lowest word for `F128`, and coefficient by coefficient into the low 64 bits of `C0`, `C1`, `C2` for `F192`. `merge` XORs two states. `reduce` runs the reduction half once.

The reason one type serves all three associated types is invariant 2. For a prime field, `SmallScalarAccumulator` exists because a `field × u64` product fits a narrower integer slot than a `field × field` product. Here a `u64` scalar acts as its low bit, so the small-scalar path is a conditional XOR into the same state.

### Why `Unreduced` does not apply

`Unreduced` (`crates/jolt-field/src/unreduced.rs`) describes three integer accumulators and the number of terms each can absorb before a slot overflows. A binary field needs none of that vocabulary: its unreduced product has one shape, its sum is exact for any number of terms, and "scale by a small integer" is a parity test. The signatures could be satisfied, but only by choosing associated types (`Wide: From<Self>` scaled by `i32`, `SmallProduct` for `Self × u64`) for semantics the trait documents in integer terms and that no caller would use. `WithAccumulator` is the surface the generic kernels consume, and it expresses everything a binary field can defer.

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
