# Zeromorph review 1 — `002eaf717`

## Findings

1. **BLOCKER — `crates/jolt-zeromorph/src/lib.rs:220` — the Fiat–Shamir challenges do not bind the opening statement.**

   `open_multi` and `verify_multi` absorb the point count and quotient commitments, but never the
   commitment, point coordinates, or claimed evaluations. A prover can therefore choose the
   statement after learning `x`. A standalone diagnostic against the committed tree used identity
   quotient/lift commitments, set

   ```text
   v = f_hat(x) / Phi_n(x),
   point = (0, ..., 0),
   ```

   committed to `X^2 (f_hat(X) - v Phi_n(x)) / (X - x)`, and obtained
   `Ok(())` although `v != f(point)`. This is the public `CommitmentScheme` path used by the crate's
   tests: both sides start with only `Transcript::new(...)`, so no caller-owned prefix repairs it.

   **Fix:** absorb a canonical statement encoding—commitment, arity/claim count, every point, and
   every claimed evaluation—before the first proof message/challenge. Avoid recomputing the
   commitment during `open`: return it as `OpeningHint`, implement `combine_hints`, and require the
   combined hint in the trait path, or change the opening API to accept the commitment. Add the
   identity-quotient forgery above as a rejection test.

2. **BLOCKER — `crates/jolt-zeromorph/src/lib.rs:312` — verification performs a runtime G2 scalar multiplication.**

   `setup.inner.g2().scalar_mul(&x)` computes `[x]2` online. The EVM has no G2
   scalar-multiplication precompile, so this verifier cannot run under the stated on-chain model.
   The native two-pair equation is algebraically correct:

   ```text
   e(C_H, [beta^2]2) * e(-pi, [beta - x]2) = 1.
   ```

   **Fix:** move `x` to G1 and submit three fixed-G2 pairs:

   ```text
   e(C_H, [beta^2]2) * e(-pi, [beta]2) * e(x pi, [1]2) = 1.
   ```

   Cost: one G1 scalar multiplication and one extra pairing-precompile pair; no G2 arithmetic.

3. **MINOR — `crates/jolt-zeromorph/tests/zeromorph.rs:92` — the tests miss the decisive equation and boundary cases.**

   Quotient/lift mutations also change every later challenge, so rejection does not prove those
   elements enter the final equation. No test mutates `opening_proof`; arity one, a zero/constant
   table, exact SRS-end use, and the 2,048-byte three-point shape at `ell = 20` are absent. The
   order test pins `evaluations[i] -> beta^i`, but not the point-coordinate/bit reversal.

   **Fix:** add the adaptive-statement forgery test; fixed-challenge algebra tests for the quotient,
   lift, and final pairing; an `opening_proof` mutation; `ell = 1` zero/constant cases; an assertion
   that the setup has exactly `N + 1` G1 powers; and a deterministic `ell = 2` Boolean-point test
   pinning Jolt's high-to-low point order.

## Formula audit

- **Univariatisation and quotient identity:** table entry `i` is coefficient `X^i`. Jolt's first
  point coordinate is the high bit, so `multilinear_quotients` eliminates coordinates front to
  back and reverses the resulting rows into paper order `q_0, ..., q_(ell-1)`. The rows have lengths
  `1, 2, ..., 2^(ell-1)` and satisfy `f - v = sum_k (X_k - u_k) q_k`.
- **Degree batch:** after all `C_k`, challenge `y` fixes
  `q_hat = sum_k y^k X^(N-2^k) U_k(q_k)`. The separate lifted commitment is present before `x`.
  `z` combines the degree and evaluation identities; the shifted witness commits to `X^2 Q`.
- **Closed forms:** `powers[k] = x^(2^k)` and
  `suffix[k] = Phi_(ell-k)(x^(2^k))`. The implementation's factor is exactly
  `x^(2^k) Phi_(ell-k-1)(x^(2^(k+1))) - u_k Phi_(ell-k)(x^(2^k))`; its shift is
  `x^(N-2^k)`.
- **Final equation and SRS:** the shared HyperKZG SRS contains powers `beta^0 .. beta^N`, so the
  paper's maximum-degree parameter is extended by one and the degree-enforcing shift is two. The
  journal states this deviation and the implementation uses the exact terminal power `beta^N`.
  The check `e(C_H,[beta^2]2) = e(pi,[beta-x]2)` is therefore complete and degree-sound. Merely
  changing the shift to one while retaining `beta^N` would lose the degree bound.

## Transcript and multi-point audit

- Prover-message order is correct: all quotient commitments precede `y`; all lifted commitments
  precede `x`, `z`, and `rho`; the final witness follows every challenge. Consecutive transcript
  squeezes advance the sponge. Finding 1 is the missing public statement, not a message-order bug.
- `commitment_order_matches_hyperkzg` independently checks
  `C = [sum_i evaluations[i] beta^i]1` and equality with HyperKZG. It pins coefficient order; the
  coordinate-to-bit mapping lacks a local fixed-vector test.
- The three-point path is a random linear combination of three Zeromorph final identities. It keeps
  three independent quotient/lift commitment families, shares `y`, `x`, and `z`, samples `rho`
  after every family is fixed, then emits one KZG witness. It is not a native three-point KZG
  opening. With statements fixed first, the shared challenges introduce no binding shortcut: a
  false component survives the degree-`t-1` polynomial in `rho` except with probability at most
  `(t-1)/|Fr|`. The current statement omission still breaks binding as finding 1 demonstrates.
- At `ell = 20`, three points contain `3(20+1)+1 = 64` compressed G1 payloads: **2,048 bytes**.
  This count is honest for raw group payload. The current postcard encoding is 2,114 bytes: one
  byte-string length prefix per G1 plus two vector-length prefixes.

## Edge cases, trait, and EVM counts

- Public setup rejects `ell = 0`, matching the paper's positive-arity domain. `ell = 1`, identity
  commitments, zero polynomials, and constant polynomials are algebraically supported but untested.
- Setup creates exactly `N + 1` G1 powers. The longest prover MSM is the shifted witness range
  `[2, N]`; no off-by-one access was found. A too-short prover SRS cannot be built through the
  public constructors.
- `setup`, `verifier_setup`, dense commitment, opening shape checks, error mapping, and additive
  commitment combination match `jolt_openings::CommitmentScheme`; the missing statement binding
  makes its opening semantics unsound until finding 1 is fixed.
- `verify_multi` performs one G1 MSM with `t(ell+1)+2` bases, one G1 negation, one variable G2
  scalar multiplication, one G2 subtraction, and two pairings. Counts: 23 MSM bases for one point
  at `ell = 20`; 65 for three points. The fixed-G2 rewrite in finding 2 uses three pairings and one
  extra G1 scalar multiplication.

## Checks

- `cargo clippy -p jolt-zeromorph --all-targets -q --message-format=short -- -D warnings`: passed.
- `cargo nextest run -p jolt-zeromorph --cargo-quiet`: 5 passed.
- Production source: 723 lines; no `#[allow]`, dead mode, qualified imported nominal, or source file
  over the requested size threshold found.

VERDICT: 2 blockers, 0 majors, 1 minor
