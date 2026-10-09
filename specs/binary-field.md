# Spec: Binary-Field Backend for `jolt-field`

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | Quang Dao, Claude              |
| Created     | 2026-10-09                     |
| Status      | proposed                       |
| PR          |                                |

## Summary

Akita is gaining a commitment and opening for tables of bits, following LaBinius (ePrint 2026/2103). Its evaluation claims live in a binary field: Akita's `dev` branch defines the host fields `BinaryField128` and `BinaryField192` for them. For Jolt to produce and consume such claims, its sum-check, polynomial and transcript crates must run over a field of characteristic 2. Those crates are generic over `JoltField`, and no type in the workspace has characteristic 2. This spec adds a third backend to `jolt-field`, behind a `binary` feature, with binary fields that implement `JoltField` and match Akita's representations. The contract layer and the existing backends do not change.

## Intent

### Goal

Add a `binary` backend module to `jolt-field` with three element types that implement `JoltField` and agree in value with Akita's binary host fields.

| Type | Field | Representation | Akita counterpart |
|---|---|---|---|
| `F64` | $\mathbb F_2[x]/(x^{64}+x^4+x^3+x+1)$ | one `u64`, bit $i$ is the coefficient of $x^i$ | the base field of `BinaryField192` |
| `F192` | `F64`$[y]/(y^3+y+1)$ | `[F64; 3]`, index $i$ is the coefficient of $y^i$ | `BinaryField192` |
| `F128` | $\mathbb F_2[x]/(x^{128}+x^7+x^2+x+1)$ | one `u128`, bit $i$ is the coefficient of $x^i$ | `BinaryField128` |

`F192` implements `ExtField<F64>` with `DEGREE = 3`. `F128` is not an extension of `F64` in this representation and implements no `ExtField`.

Each type exposes its representation through one constructor and one accessor, which are the coordinates of the Akita contract: `F64::from_raw(u64)` and `to_raw`, `F128::from_raw(u128)` and `to_raw`, and for `F192` the `ExtField<F64>` methods `from_base_fn` and `base_coefficient`. A raw word is a vector of $\mathbb F_2$-coefficients, not an integer; `Ring::from_u64` is a different map (invariant 3).

### Invariants

1. **Field axioms.** Each type is a field of the stated order: addition is XOR of representations, multiplication is polynomial multiplication reduced by the stated modulus, `inverse` returns `None` exactly for zero and otherwise `a * a.inverse() == 1`.
2. **Agreement with Akita.** Multiplication, squaring and inversion of `F128` and `F192` equal those of Akita's `BinaryField128` and `BinaryField192` (`crates/akita-algebra/src/binary/host.rs` on `dev` at `3d69096ead`), and `F64` equals the base field of `BinaryField192`. This is agreement of values and coordinates: Akita's `BinaryField128` is two low-degree-first `u64` words, so an `F128` is `lo | (hi << 64)`, and `BinaryField192` is three `u64` coefficients in the order of `F192`'s. Conversion copies words and performs no field arithmetic. No layout, transmute or borrowing compatibility is promised.
3. **`Ring` integer maps are the ring homomorphism.** `from_u64(v)`, `from_i64`, `from_u128`, `from_i128` return `one()` if `v` is odd and `zero()` otherwise. Hence `from_u64(2) == zero()`, `pow2(k) == zero()` for `k > 0`, and the inherited `mul_pow_2(k)` is zero for `1 <= k <= 255` and panics above, as for every backend. The accumulator defaults `fmadd_i128`, `fmadd_signed_u64` and `fmadd_s256` fold limbs with `mul_pow_2(64)` and therefore reduce a signed scalar to the parity of its low limb, which is the ring map; negation is the identity. `jolt-sumcheck`'s `BooleanHypercube::round_sum_coefficients` relies on `from_u64(2)` being the ring image of 2.
4. **`two_inv` and `half` panic.** No correct caller reaches them in characteristic 2, and they keep the default `expect("field has characteristic two")`. Generic code that calls them (`prove_batch` in `jolt-sumcheck`) cannot be used over this backend until it has a characteristic-2 path; that is a separate spec.
5. **Canonical encoding.** `NUM_BYTES` is 8, 24 and 16; `MODULUS_BITS` is 65, 193 and 129 (bit length of the order). `to_bytes_le` writes the representation little-endian (`F192`: the three `F64` coefficients in index order) and panics unless the buffer has exactly `NUM_BYTES` bytes, as the `solinas` extensions do. Every byte string of length `NUM_BYTES` is canonical, so `from_bytes_le_checked` fails only on a wrong length. `from_bytes_le_reduced` reads the first `NUM_BYTES` bytes, zero-padding a short input and ignoring the rest; it is a truncation, not a reduction of an integer.
6. **`u128` conversions.** These follow the existing contract of `CanonicalEncoding`, under which an extension field reports its constant coefficient only when all higher coefficients are zero. `F64` and `F128`: `to_u128_checked` returns the raw word, always `Some`; `from_u128_checked` is its inverse (`F64` returns `None` above 64 bits); `from_u128_reduced` truncates to the field's width. `F192`: `to_u128_checked` returns the raw word of the constant coefficient when the $y$ and $y^2$ coefficients are zero and `None` otherwise; `from_u128_checked` and `from_u128_reduced` produce a constant, rejecting respectively truncating above 64 bits. In every case the `u128` is a coefficient vector, not an integer. `num_bits` is the bit length of the whole representation (for `F192`, of the 192-bit string with the $y^2$ coefficient highest).
7. **Challenges.** `from_challenge_bytes` and `from_scalar_challenge_bytes` are `from_bytes_le_reduced`. A challenge is uniform on the field only if the transcript supplies at least `NUM_BYTES` bytes; with Jolt's 16-byte squeeze that holds for `F64` and `F128` and not for `F192`. A transcript that squeezes 24 bytes is out of scope here.
8. **Sampling.** `random` reads exactly `NUM_BYTES` bytes from the RNG and decodes them; no rejection is needed.
9. **Serde.** Wire serialization uses `impl_serde_bytes!` with width `NUM_BYTES`, as the other backends do.
10. **Relative Frobenius.** `F192::frobenius_pow(k)` is exponentiation by $(2^{64})^k$. On coefficients $(c_0,c_1,c_2)$ one application gives $(c_0,\,c_2,\,c_1+c_2)$, since $y^{2^{64}}=y^2$; it fixes `F64` and has order 3.
11. **Display.** Lowercase hexadecimal of the representation, highest coefficient first. It is for diagnostics and is not a wire format.
12. **Allocative.** Under the `allocative` feature the three types derive `Allocative`, which `JoltField` then requires.

No `jolt-eval` invariant changes: the backend has no caller yet.

### Non-Goals

- Carry-less multiply intrinsics, packed backends and deferred-reduction accumulators. `WithAccumulator` uses `NaiveAccumulator` for all three associated types, and a portable multiply is sufficient here. Performance kernels are a separate change, measured against this reference.
- Changes to the existing spine traits or to the `bn254` and `solinas` backends, and changes to `jolt-sumcheck` or `jolt-transcript`. Splitting the integer embedding out of `Ring` (see Alternatives) is not part of this spec.
- A dependency on Akita from `jolt-field`, and conversions to Akita's types. Those belong to `jolt-akita`, which owns that dependency.
- Akita's 162-bit field and any other binary field without a consumer in Jolt.
- A capability trait for characteristic 2 (a basis, the embedding of packed bits). It is added with its first caller; this spec's public surface is limited to the three types, whose external contract is value agreement with Akita.

## Evaluation

### Acceptance Criteria

- [ ] `jolt-field` gains a `binary` feature, off by default, that enables `mod binary` and re-exports `F64`, `F128`, `F192`. The backend references no other backend and contains no `unsafe`.
- [ ] `cargo clippy -p jolt-field --all-targets -- -D warnings` passes with each of `--no-default-features --features binary`, `--no-default-features --features binary,allocative` and `--features solinas,binary,allocative`.
- [ ] Compile-time bound assertions: `F64`, `F128`, `F192: JoltField` (checked with and without `allocative`) and `F192: ExtField<F64>`.
- [ ] Fixed vectors for multiplication, squaring and inversion, at least 16 per type and operation and including zero, one, the top basis element and all-ones, match values generated from Akita at `3d69096ead`. Akita exposes only `Add` and `Mul` on these types, so the recipe is: product by `Mul`; square as `a * a`; inverse of a nonzero `a` as $a^{2^n-2}$ by square-and-multiply; `F64` values by embedding `[a, 0, 0]` in `BinaryField192` and reading coefficient 0. The commit and this recipe are recorded next to the vectors. `inverse` of zero is `None`.
- [ ] Algebraic properties on seeded random inputs: associativity, commutativity, distributivity, `a + a == 0`, `a.square() == a * a`, `a * a.inverse() == 1` for nonzero `a`, and $a^{2^n}=a$ with $n$ = 64, 128, 192.
- [ ] `from_u64(2)`, `pow2(1)` and `one().mul_pow_2(1)` are zero; `from_u64(3) == one()`; `from_i64(-1) == one()`. One accumulator test pins the parity of signed scalars: odd and even low limbs, a magnitude with only a high limb set, both signs, and `i128::MIN`.
- [ ] Frozen encodings: for one asymmetric element per type, the expected `to_bytes_le` bytes and bincode bytes are written out in the test. `from_bytes_le_reduced` zero-pads a short input and ignores bytes past `NUM_BYTES`; `from_bytes_le_checked` returns `None` for every length below `NUM_BYTES`, for `NUM_BYTES + 1` and for a longer input, and round-trips at `NUM_BYTES`.
- [ ] `u128` conversions at their boundaries: `F64::from_u128_checked(1 << 64)` is `None`; `F192` with a nonzero $y$ coefficient gives `to_u128_checked() == None`; `num_bits` of zero, one and the top basis element.
- [ ] `random` consumes exactly `NUM_BYTES` bytes, checked with a counting RNG.
- [ ] `ExtField<F64>` contract for `F192`: `lift_base` and `mul_base` against coefficientwise definitions, `from_base_fn` called once per index in ascending order, `base_coefficient` out of range panics, `frobenius_pow(1)` maps $y$ to $y^2$ and fixes `F64`, and `frobenius_pow(3)` is the identity.
- [ ] `cargo fmt --check` passes.

### Testing Strategy

New tests only, in `crates/jolt-field/tests/binary_*.rs`, each gated on the `binary` feature so that the feature-off build still compiles, run with `cargo nextest run -p jolt-field --features binary --cargo-quiet`. Existing `jolt-field` tests must pass unchanged with and without the feature. There is no host or ZK mode distinction at this layer. The permanent ground truth is the frozen Akita vectors, the frozen encodings and the algebraic properties. A bitwise multiply written from the definition is compared with the production multiply on 10,000 seeded pairs per type once, as validation recorded in the PR; it is not kept in the suite, which may not hold a second implementation as its oracle.

### Performance

None claimed. This is the reference arithmetic. No existing `jolt-eval` objective moves.

## Design

### Architecture

`JoltField` is a blanket bundle of `Field`, `CanonicalEncoding`, `WithAccumulator`, serde and `MaybeAllocative` (`crates/jolt-field/src/algebra.rs:518–532`). None of the supertraits asserts a prime order, and `CanonicalEncoding` already documents extension fields, so the binary fields implement the existing spine and the protocol crates need no second trait hierarchy. What the spine cannot express, the embedding of packed bits against a basis, needs a capability contract beside `PseudoMersenne` and `ExtField`; it is added with its first caller.

The backend lives in `jolt-field` because that is the crate's stated architecture: a contract layer at the root and feature-gated backend modules that never reference each other (`crates/jolt-field/src/lib.rs`, "Architecture: contracts and backends"). `bn254` and `solinas` are the two existing backends; `binary` is the third. The feature is off by default, so no existing build compiles it, and the crate's byte-compatibility invariants, which are stated per backend, are untouched.

The representations are Akita's, so that `jolt-akita` can pass elements across the boundary without arithmetic.

The places where the trait surface assumes odd characteristic, and what this backend does about each:

| Method | Behaviour here | Consequence for generic callers |
|---|---|---|
| `two_inv`, `half` | panic (default) | `prove_batch` in `jolt-sumcheck` needs a characteristic-2 path |
| `from_u64` and the other integer maps, and the signed `fmadd_*` defaults built on them | parity | correct for `BooleanHypercube`; coefficient words enter through `from_raw` |
| `pow2`, `mul_pow_2` | zero for a positive exponent (`mul_pow_2` keeps its bound of 255) | batch padding by powers of two degenerates and needs the same path |
| `to_u128_checked`, `from_u128_*` | coefficient-word maps | wrong for any caller treating them as integer conversions |
| `from_challenge_bytes` | truncating decode | `F192` challenges need a 24-byte squeeze |

### Alternatives Considered

- **A separate crate.** Rejected: `jolt-field` already separates contracts from backends by feature, and a second crate would split a later capability contract from the spine for no build-time gain. Plonky3 uses one crate per field, but its `p3-field` holds only traits; `jolt-field` holds its backends, as Binius64's `field` crate does.
- **Using Akita's field types directly.** Rejected: it would put a git dependency under `jolt-field`, which is in the verifier closure, and Akita's host fields expose only word conversions, `Add` and `Mul`, not Jolt's spine.
- **Making `from_u64` the coefficient-word embedding.** Rejected: `Ring::from_u64` is used as the ring map by existing generic code, and a word embedding there would make `from_u64(2)` nonzero and silently break round-sum reconstruction.
- **Reshaping the spine first, as Plonky3 does.** Plonky3's base trait is `PrimeCharacteristicRing`, where integers enter through the prime subfield and integer-valued semantics sit on `PrimeField`. `Ring::from_u64`, `pow2`, `mul_pow_2`, `two_inv` and `half` are where Jolt's spine assumes odd characteristic. Moving them to a prime-only trait is the cleaner shape, but it touches every generic caller in the workspace; it is deferred until a binary-field consumer shows which callers need which semantics.

## Documentation

None. The backend has no user-facing surface yet.

## Execution

`src/binary/{mod,f64,f128,f192}.rs`. `F64` multiply: shift-and-XOR with reduction by `0x1B`. `F128` multiply: the same over `u128` with reduction by `0x87`. `F192`: schoolbook over `F64` with $y^3=y+1$, $y^4=y^2+y$. Inversion by exponentiation to $2^n-2$. Stamp operator impls with one `impl_ring_ops!` per type (it includes `impl_group_ops!`) and serde with `impl_serde_bytes!`. Where clippy's `suspicious_arithmetic_impl` fires on XOR inside `Add`, use a narrow `#[expect]` with a reason. Test vectors are generated outside the repository from Akita and checked in as constants.

## References

- LaBinius, ePrint 2026/2103.
- Akita, `crates/akita-algebra/src/binary/host.rs` on `dev` at `3d69096ead`: `BinaryField128` and `BinaryField192`.
- `specs/consolidate-field-traits.md`, `specs/jolt-field-rebuild.md`: the trait spine this backend implements.
- `specs/verifier-closure-lints.md`: the lint set the backend must satisfy.
