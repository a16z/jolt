# Zeromorph review 1 — `002eaf717`

## Findings

1. **BLOCKER — `crates/jolt-zeromorph/src/lib.rs:312` — verification performs a runtime G2 scalar multiplication.**

   `setup.inner.g2().scalar_mul(&x)` computes `[x]₂` online. The target EVM has no G2 scalar-multiplication precompile, so this verifier cannot implement the selected on-chain design.

   The checked equation is internally sound:

   ```text
   H(X) = (X - x) Q(X)
   C_H  = [H(β)]₁
   π    = [β² Q(β)]₁

   e(C_H, [β²]₂) · e(-π, [β - x]₂) = 1
   ```

   **Fix:** move `x` to G1 and submit three static-G2 pairs in one pairing-precompile call:

   ```text
   e(C_H, [β²]₂) · e(-π, [β]₂) · e(xπ, [1]₂) = 1.
   ```

   This needs one G1 scalar multiplication and the already-stored `[1]₂`, `[β]₂`, `[β²]₂`; `[β³]₂` is unused by Zeromorph.

2. **MAJOR — `crates/jolt-zeromorph/src/lib.rs:247` — the proof consumes `[β^N]₁`, contrary to the requested `i < 2^ell` SRS bound.**

   `witness` has `N - 1` coefficients and is committed with shift 2, so the MSM range is `[β²]₁ .. [β^N]₁`. This is consistent with the wrapped HyperKZG setup, which publishes `N + 1` G1 powers, and the `[β²]₂` equation correctly checks `deg H <= N - 1` under that larger SRS. It does not meet the stated SRS contract.

   At maximum arity, Kohrita–Towa §6 uses powers `[β^0]₁ .. [β^(N-1)]₁` and shift 1. Merely changing this crate to shift 1 is unsound while `[β^N]₁` remains public under the same trapdoor: an adversary could then open a degree-`N` `H` with the available shifted witness.

   **Fix:** either (a) remove `[β^N]₁` from the shared public SRS and use shift 1 with the matching `[β]₂` equation, including the existing HyperKZG degree-check consumers, or (b) make the `N + 1`-power SRS an accepted protocol requirement. Option (a) matches the requested bound.

3. **MINOR — `crates/jolt-zeromorph/tests/zeromorph.rs:109` — the tamper test does not isolate the degree/evaluation equation.**

   Mutating a quotient commitment changes an item absorbed before `y`; mutating the lifted commitment changes an item absorbed before `x`. Both cases therefore derive different `y`, `x`, `z`, and `rho` values and eventually fail the final pairing. The test would still pass if a quotient were omitted from `combined_identity_commitment`, because transcript divergence alone rejects the reused proof. Honest round trips also cannot catch a wrong lift exponent copied into both prover and verifier.

   **Fix:** add independent tests for (a) the quotient identity at `ell = 1, 2, 3`, including a constant table, (b) lifted coefficients against `sum_k y^k X^(N-2^k) q_k`, and (c) quotient/lifted mutations under a scripted transcript returning fixed challenges, so the final pairing equation is the only changing check.

## Protocol audit

- **Embedding/order:** pinned by `commitment_order_matches_hyperkzg`: table index `i` becomes coefficient degree `i`, and the expected Horner commitment equals HyperKZG's commitment.
- **Quotients:** `multilinear_quotients` eliminates Jolt coordinates high-to-low, records `high - low`, folds at the corresponding coordinate, then reverses the rows into paper order `q_0 .. q_(ell-1)`. Thus `q_0` has one coefficient and `q_(ell-1)` has `N/2`; a constant table yields all-zero rows.
- **Degree batch:** `lifted_degree_quotient` plus the outer `N/2` shift commits to `q_hat = sum_k y^k X^(N-2^k) q_hat_k`. `C_qhat` is present. The final shifted witness and pairing enforce `deg H <= N - 1`; random `y` then binds every separate `deg q_k < 2^k` bound.
- **Transcript:** all quotient commitments precede `y`; every lifted batch commitment precedes `x`; `z` and multi-point `rho` follow those commitments; the witness follows all challenges. Single-point mode draws an unused `rho`, harmless but unnecessary.
- **Closed forms:** `powers[k] = x^(2^k)` and `suffix[k] = product_(i=k)^(ell-1)(1 + x^(2^i)) = Phi_(ell-k)(x^(2^k))`. For `ell = 2`, the code gives shifts `(x^3, x^2)` and factors `(x(1+x^2)-u_0(1+x)(1+x^2), x^2-u_1(1+x^2))`. For `ell = 3`, it gives shifts `(x^7, x^6, x^4)` and factors `(x(1+x^2)(1+x^4)-u_0 Phi_3(x), x^2(1+x^4)-u_1(1+x^2)(1+x^4), x^4-u_2(1+x^4))`.
- **Three points:** `H_multi = sum_j rho^j (zeta_j + z Z_j)` is fixed before the shared witness; `rho` is sampled after every per-point quotient/lift commitment. A false component can cancel only at roots of the resulting polynomial in `rho`. The wire count is `3 ell + 3 + 1 = 3(ell+1)+1` G1.
- **Trait/style:** `CommitmentScheme` arity, dense-table conversion, error mapping, and additive combination match the existing HyperKZG path. Production code is 723 lines, has no `#[allow]`, uses imported nominal names, and keeps the Phi/shift factors in `identity_scalars`.
- **Checks:** `cargo clippy -p jolt-zeromorph --all-targets -q -- -D warnings` passed. `cargo nextest run -p jolt-zeromorph --cargo-quiet` passed 5/5.

VERDICT: 1 blockers, 1 majors, 1 minors
