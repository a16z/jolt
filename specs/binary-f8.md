# Spec: `F8` and Its Embeddings into the Binary Fields

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | Quang Dao, Claude              |
| Created     | 2026-10-09                     |
| Status      | proposed                       |
| PR          |                                |

## Summary

A sum-check over a binary field cannot use consecutive integers as evaluation points: `specs/binary-sumcheck.md` rejects `CenteredIntegerDomain` in characteristic 2 for that reason. The binary-field systems surveyed there (Binius64, flock) take their univariate-skip domain to be an $\mathbb F_2$-subspace of $\mathrm{GF}(2^8)$, embedded into the large field. This spec adds the 8-bit field `F8` to the `binary` backend of `jolt-field` and its embeddings into `F64`, `F128` and `F192`, which is the subfield part of step 3 of the roadmap in `specs/binary-field.md`. `From<F8>` is the conversion contract for that subfield; a general capability for packed-bit embeddings stays deferred until it has a caller. The domain itself, and the sum-check round that uses it, are a later spec.

## Intent

### Goal

Add `F8` $= \mathbb F_2[x]/(x^8+x^4+x^3+x+1)$ to `jolt-field`'s `binary` backend as a `JoltField`, together with `From<F8>` for `F64`, `F128` and `F192`, each a homomorphism of fields.

`F8` is represented by one `u8`, bit $i$ the coefficient of $x^i$, with `F8::from_raw(u8)` and `F8::to_raw`, as for `F64`. The modulus is the AES polynomial.

### Invariants

1. **Field axioms.** `F8` is a field of order 256: addition is XOR, multiplication is polynomial multiplication reduced by $x^8+x^4+x^3+x+1$, and `inverse` returns `None` exactly for zero.
2. **Spine.** `F8` implements the spine exactly as `F64` does in `specs/binary-field.md`, invariants 3 to 9, 11 and 13 there, with width 8 in place of 64: integer maps are parity; `two_inv` and `half` panic; `NUM_BYTES = 1` and `MODULUS_BITS = 9`; `to_u128_checked` returns the raw byte, `from_u128_checked` rejects values above `0xff`, `from_u128_reduced` truncates; `from_bytes_le_reduced` reads the first byte and maps an empty input to zero; challenges are `from_bytes_le_reduced`; `random` reads exactly one byte; serde through `impl_serde_bytes!` with width 1; `Display` is two lowercase hexadecimal digits; `WithAccumulator` uses `NaiveAccumulator` for all three associated types; `Allocative` is derived under the feature.
3. **Embeddings are homomorphisms.** For `E` in `F64`, `F128`, `F192` and all `a, b: F8`: `E::from(a + b) == E::from(a) + E::from(b)`, `E::from(a * b) == E::from(a) * E::from(b)`, `E::from(F8::one()) == E::one()`, and `E::from` is injective.
4. **Which embedding.** A homomorphism `F8 -> E` is determined by the image $\beta_E$ of $x$, which must be a root in `E` of $x^8+x^4+x^3+x+1$; there are eight. For `F64` and `F128`, $\beta_E$ is the root whose raw word is smallest as an unsigned integer:

   | `E` | $\beta_E$ (raw) |
   |---|---|
   | `F64` | `0x033ce8beddc8a656` |
   | `F128` | `0x053d8555a9979a1ca13fe8ac5560ce0d` |

   `E::from(a)` is $\sum_i a_i \beta_E^i$ over the bits $a_i$ of `a.to_raw()`.
5. **`F192` factors through `F64`.** `F192::from(a) == F192::lift_base(F64::from(a))` for every `a`.
6. **No other change.** `F64`, `F128`, `F192`, the spine traits, and the other backends are not modified, apart from the three `From` impls. `F8` gets no carry-less-multiply kernel and no architecture-specific code: it has one arithmetic path on every target.

### Non-Goals

- `ExtField<F8>` for `F64` or `F128`. It would require a basis of the large field over `F8` and base-coefficient extraction, and nothing needs them.
- A subspace or evaluation-domain type, additive NTT, Lagrange or subspace-polynomial evaluation. Those belong with the sum-check round that consumes them.
- An embedding of `F64` into `F128`. The two are used as alternative fields, not as a tower.
- Agreement with another project's embedding table. Akita's `akita-algebra` at `3d69096ead` defines no 8-bit field, so there is no coordinate to match; invariant 4 fixes the choice by a rule instead.
- A capability trait for "field with an embedded `F8`". The bound `F: Field + From<F8>` expresses it, as Binius64's `F: BinaryField + From<B8>` does.

## Evaluation

### Acceptance Criteria

- [ ] `F8` is defined in `crates/jolt-field/src/binary/f8.rs` and re-exported from the crate root under the `binary` feature, beside `F64`, `F128`, `F192`.
- [ ] Compile-time bound assertions: `F8: JoltField`, with and without `allocative`; `F64: From<F8>`, `F128: From<F8>`, `F192: From<F8>`.
- [ ] Multiplication against FIPS 197: `{57} * {83} == {c1}` and `{57} * {13} == {fe}` (section 4.2), and `{53}.inverse() == {ca}` (the pair used in the S-box literature). `inverse` of zero is `None`.
- [ ] Exhaustive field checks over all 256 elements: `a * a.inverse() == 1` for nonzero `a`; `a.square() == a * a`; $a^{256} = a$. Exhaustive over all 65,536 pairs: commutativity, and distributivity against a third fixed nonzero element.
- [ ] `x` has multiplicative order exactly 51 ($x^{51} = 1$, $x^{17} \ne 1$, $x^{3} \ne 1$) and `x + 1` (raw `0x03`) has order exactly 255 (its powers 85, 51 and 15 are not one). (The AES polynomial is irreducible and not primitive.)
- [ ] A contract test for the `F8` spine, covering the following and no more: `NUM_BYTES == 1` and `MODULUS_BITS == 9`; `from_u64(2)`, `pow2(1)`, `one().mul_pow_2(1)` are zero and `from_u64(3)`, `from_i64(-1)` are one; `two_inv` and `half` panic; the accumulator's signed `fmadd` variants act by parity; frozen `to_bytes_le` and bincode bytes for one element, and `to_bytes_le` panics on a buffer of length 0 or 2; `from_bytes_le_reduced` on an empty and a two-byte input, and both challenge constructors agree with it; `from_bytes_le_checked` on lengths 0, 1, 2; `from_u128_checked(0x100)` is `None`, `from_u128_reduced(0x1ab)` is `0xab`, and `to_u128_checked` returns the raw byte; `num_bits` of zero, one and `0x80`; `Display` of zero and one is `00` and `01`; `random` consumes one byte, checked with a counting RNG. Inherited default methods need no test of their own.
- [ ] Embedding, for each of `F64` and `F128`:
  - `E::from(F8::from_raw(2))` equals the constant of invariant 4, written out in the test.
  - That value is a root: $\beta^8+\beta^4+\beta^3+\beta+1 = 0$ in `E`.
  - Minimality: the raw words of $\beta^{2^i}$ for $i = 0..8$ are eight distinct values, and $\beta$ is the smallest. (The roots of an irreducible polynomial are exactly the Frobenius conjugates of one root, so this checks the rule without a root-finding routine.)
  - Homomorphism, exhaustively: `E::from(a * b) == E::from(a) * E::from(b)` and `E::from(a + b) == E::from(a) + E::from(b)` over all 65,536 pairs, and `E::from(F8::one()) == E::one()`.
  - Injectivity: the 256 images are distinct.
- [ ] `F192::from(a) == F192::lift_base(F64::from(a))` for all 256 elements. With the `F64` checks this determines every `F192` image, and `lift_base` is already tested as a homomorphism, so no pair check is repeated for `F192`.
- [ ] Two frozen images for each of `F64` and `F128`, written out in the test: `F64::from(F8::from_raw(0x53)) == 0xff054c3f7cef0cca`, `F64::from(F8::from_raw(0xff)) == 0x5c8346787364a654`, `F128::from(F8::from_raw(0x53)) == 0xde77167b8539a7970d972a0b4c6fa967`, `F128::from(F8::from_raw(0xff)) == 0xfae6e08c31e89f90017eb6dcd4f33a26`.
- [ ] One embedding table each for `F64` and `F128`, computed at compile time from the eight basis images or from $\beta$ alone; no 256-entry literal is checked in. `F192::from` reads the `F64` table and lifts the result; it has no table of its own. No embedding performs a field multiplication at run time.
- [ ] `cargo clippy -p jolt-field --all-targets -- -D warnings` passes with `--no-default-features --features binary`, `--no-default-features --features binary,allocative` and `--features solinas,binary,allocative`; `cargo nextest run -p jolt-field --no-default-features --features binary --cargo-quiet` passes; `cargo fmt --check` passes. The four binary configurations of `.github/workflows/field-portability.yml` run the new tests without a workflow change, since they run the whole binary suite.
- [ ] `crates/jolt-field/src/lib.rs` and `binary/mod.rs` module docs list `F8` and state that `From<F8>` is the field embedding of invariant 4.

### Testing Strategy

A new `crates/jolt-field/tests/binary_f8.rs`, gated on the `binary` feature. Ground truth is FIPS 197 for multiplication, exhaustive algebraic properties for the field (the one-element distributivity check is a smoke test; the embedding checks into fields already tested carry the rest), and for the embeddings the root equation, the minimality rule and the exhaustive homomorphism check. The frozen images guard against a silent change of root. Existing tests pass unchanged.

### Performance

None measured. `F8` arithmetic is not on a hot path: its elements are domain points and table indices. The embedding is one table read.

## Design

### Architecture

`F8` is a fourth type in `binary/`, written like `F64`: a newtype over its word, portable shift-and-XOR multiplication reduced by `0x1b`, operators stamped by `impl_ring_ops!`, inversion by the shared `inverse` helper (exponentiation to $2^8 - 2$).

`GF(2^8)` is a subfield of `GF(2^n)` exactly when 8 divides `n`, which holds for 64 and 128, and for 192 through `F64`. The embedding is $\mathbb F_2$-linear, so it is determined by the images $\beta^0, \ldots, \beta^7$ of the monomial basis, and the image of a byte is the XOR of the basis images selected by its bits. The 256-entry tables for `F64` and `F128` are built in a `const` context from those eight words, which needs only XOR; the eight words are themselves either literals derived from $\beta$ or computed by a `const fn` portable multiply. The implementer chooses; the tests pin the result either way.

The rule "smallest raw word" exists only to make the choice reproducible. Any of the eight roots gives a valid embedding, and they differ by a power of the Frobenius automorphism of `F8`. Nothing in Jolt or Akita depends on which one is used, provided every party uses the same one, and a rule that a test can check is preferable to an unexplained constant.

### Alternatives Considered

1. **A tower representation**, where `F8` is literally the low byte of the larger field (as in the Fan–Paar towers of Binius). Rejected: `F64` and `F128` are fixed in the polynomial bases that Akita uses and that carry-less multiply serves; a tower would change their representation.
2. **Matching Binius64's Rijndael-to-GHASH table** for `F128`. It would give byte-identical domain points with Binius64. Rejected: there is no interoperation requirement with it, it would cover `F128` only, and `F64` would need a separate rule anyway.
3. **`ExtField<F8>`.** See Non-Goals.
4. **Log and antilog tables for `F8` multiplication.** Faster than shift-and-XOR, but `F8` multiplication is not hot, and one path is simpler to audit.

## Documentation

Module docs as listed in the acceptance criteria. No book change.

## Execution

`src/binary/f8.rs` for the type and its spine; the embeddings in the same file or in `binary/embed.rs`; `tests/binary_f8.rs`. One commit.

## References

- `specs/binary-field.md`, roadmap step 3; `specs/binary-sumcheck.md`, Prior Art.
- FIPS 197, section 4.2 (multiplication in $\mathrm{GF}(2^8)$).
- Binius64 `4428e759`: `crates/field/src/fields/ghash.rs` (`From<Rijndael8b> for Ghash128b`).
- leanMultisig `c7b1daa5b1fdd61cfc000a54ec30b22acb174a9b`: `crates/primitives/src/field/phi8_tower.rs`.
