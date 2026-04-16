# Spec: ECDSA Inputs Sanitation

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | @0xAndoroid                    |
| Created     | 2026-04-02                     |
| Status      | implemented                    |
| PR          | #1402                          |

## Summary

`ecdsa_verify()` in both the secp256k1 and P-256 inline SDKs accepts typed inputs but performs no validation on them, trusting that callers used checked constructors. PR #1391 added documentation noting this caller responsibility, but this is a security footgun: callers can bypass validation via `_unchecked` constructors or direct struct construction and obtain proofs over incorrect ECDSA results. This spec moves input validation into `ecdsa_verify()` itself and restricts the `_unchecked` constructors to crate-internal use, eliminating the footgun.

## Intent

### Goal

Make `ecdsa_verify()` self-contained and safe by validating all inputs internally — scalar field range, point field range, on-curve, and not-infinity checks — returning an error on malformed inputs, and restricting `_unchecked` constructors to `pub(crate)` visibility.

### Invariants

1. `ecdsa_verify()` returns `Err` if any scalar (z, r, s) has limbs outside `[0, n)` (scalar field modulus)
2. `ecdsa_verify()` returns `Err` if any point coordinate (q.x, q.y) has limbs outside `[0, p)` (base field modulus)
3. `ecdsa_verify()` returns `Err` if q is not on the curve (`y² != x³ + ax + b`)
4. `ecdsa_verify()` returns `Err` if q is the point at infinity
5. `ecdsa_verify()` returns `Err` if r or s is zero
6. All checks are guest-side (part of the execution trace), not host-only assertions
7. `from_u64_arr_unchecked()` is not accessible from outside the crate

### Non-Goals

1. Validating inputs for non-ECDSA operations (standalone field arithmetic, point addition)
2. Signature malleability protection (enforcing low-s normalization)
3. Supporting curves beyond secp256k1 and P-256

## Evaluation

### Acceptance Criteria

- [ ] `ecdsa_verify` returns `Err` when q is not on-curve (both secp256k1 and P-256)
- [ ] `ecdsa_verify` returns `Err` when q is point at infinity
- [ ] `ecdsa_verify` returns `Err` when r or s is zero
- [ ] `ecdsa_verify` returns `Err` when scalar/coordinate limbs are out of field range
- [ ] `ecdsa_verify` returns `Ok(())` for valid signatures (existing inline tests pass)
- [ ] Checks are proved (guest-side, part of the execution trace)
- [ ] `from_u64_arr_unchecked` is `pub(crate)` or private — not accessible from external crates

### Testing Strategy

**Existing tests that must pass:**
- `jolt-inlines/secp256k1/src/tests.rs` — all existing ECDSA and field op tests
- `jolt-inlines/p256/src/tests.rs` — all existing ECDSA and field op tests
- Example guest programs (`secp256k1-ecdsa-verify`, `p256-ecdsa-verify`)

**New tests needed:**
- Unit tests for each rejection case: out-of-range scalar, out-of-range coordinate, off-curve point, point at infinity, zero r, zero s
- Tests for both secp256k1 and P-256

**Feature coverage:** `--features host` only. No `zk` feature coverage needed.

### Performance

The on-curve check adds ~3 field operations (1 square, 1 square, 1 mul, 1 add, 1 compare) per `ecdsa_verify` call. This is negligible relative to the two GLV scalar multiplications in the verification itself. No benchmark target beyond "no regression beyond the expected ~3 field ops overhead."

## Design

### Architecture

Changes are contained within the `jolt-inlines` workspace:

**`jolt-inlines/secp256k1/src/sdk.rs`:**
- Add validation at the top of `ecdsa_verify()`: scalar range checks via `is_fr_non_canonical()`, coordinate range checks via `is_fq_non_canonical()`, on-curve check via `is_on_curve()`, infinity check, r/s non-zero check
- Change `from_u64_arr_unchecked()` visibility from `pub` to `pub(crate)` on `Secp256k1Fr`, `Secp256k1Fq`, and `Secp256k1Point`

**`jolt-inlines/p256/src/sdk.rs`:**
- Same changes for P-256 types and `ecdsa_verify()`

**`jolt-inlines/sdk/src/ec.rs`:**
- If `from_u64_arr_unchecked` exists on `AffinePoint`, restrict its visibility

No new types, traits, or modules needed.

### Alternatives Considered

1. **Documentation-only (PR #1391 status quo):** Rejected — relying on callers to validate is a security footgun. Callers may not notice the requirement or implement checks sub-optimally.
2. **Remove `_unchecked` constructors entirely:** Rejected — they are used internally for values already proven valid by the inline proof system. Removing them would add redundant re-validation on every intermediate field operation result.

## Documentation

No Jolt book changes needed. The `ecdsa_verify` API signature is unchanged; it simply becomes safer by validating internally. The book's existing usage examples remain correct.

## Execution

The implementer should derive the approach from the intent and evaluation sections above.

## References

- [PR #1391](https://github.com/a16z/jolt/pull/1391) — Added documentation noting caller responsibility for input validation
