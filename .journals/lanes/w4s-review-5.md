# W4-S review #5

Scope: committed tree `6317adf51`, reviewed read-only in detached worktree
`/Volumes/Dev/worktrees/jolt/w4s-review5` against `86da3b7f1`. Review #4's three findings,
the stream/Spartan and HyperKZG verifier paths, the scoped tests, and both W4-S journals were
checked.

## Findings

1. **BLOCKER — `crates/jolt-wrapper/src/spartan.rs:322`: the verifier accepts seven alternate
   encodings of every `Challenge125` word.** `Fr::from_challenge_bytes` masks the top three bits of
   the little-endian word, while `unpack_challenges` calls it without checking that those bits are
   zero. Starting with the packer's canonical word `b`, changing `b[15]` by any nonzero mask in
   `0xe0` produces different proof bytes and the same public field value, so the Spartan transcript
   and verification result are unchanged. The proof therefore carries a canonical decoder preimage,
   not the unrecoverable raw squeeze claimed by both journals. `Scalar128` is injective because every
   128-bit integer is below the BN254 scalar modulus. **Fix:** after decoding, require
   `decoder.pack(value)? == bytes` (or reject `bytes[15] & 0xe0 != 0` for `Challenge125`), add a
   top-bit tamper rejection, and call the wire value a canonical decoder preimage rather than a raw
   squeeze.

2. **MAJOR — `crates/jolt-wrapper/src/stream/protocol.rs:373` and
   `crates/jolt-hyperkzg/src/scheme.rs:290`: `VerifierCost` still counts Fr work with detached
   formulas, and the published totals are already low.** `observe_clear_stage` runs after generic
   verification and uses `2 * members`; it does not observe `BatchPrelude::new`'s
   `mul_pow_2` multiplication. `single_member_output` performs another unobserved multiplication.
   HyperKZG executes `r*r`, `(2*r)`, `4(ell-1)` reconstruction multiplications, and `5ell`
   consistency multiplications: `9ell-2`, versus the recorded `9ell-3`. The five-factor G fixture is
   low by seven literal Rust Fr multiplications (`6,149`/`6,146`, not `6,142`/`6,139`); even if an
   EVM implementation folds away the five multiply-by-one scales, two operations remain missing.
   No test asserts the counter. **Fix:** route Fr arithmetic through observer-aware operations, or
   restore the `modeled`/estimate label and stop claiming every executed multiplication is counted.
   The pairing and EC callbacks cover the executed group operations, and the counting transcript
   covers each Keccak initialization/append/squeeze. N4 supports the rounded 7.7k, 114.7k, 183.4k,
   and 100-gas terms; 20 gas per Fr multiplication is an estimate from
   `.journals/plan-evm-verifier.md`, not an N4 measurement.

## Verified fixes and regression audit

- Decoder kinds are verifier-known through `SpartanPublicInputStatement`; the proof carries only
  16-byte words. Proving uses the corresponding `PublicChallenge` kind, and verification calls the
  production `Fr::from_challenge_bytes` / `Fr::from_scalar_challenge_bytes` decoder. The cached
  Fibonacci `2^18` fixture's native-verifier outputs cover both kinds and round-trip. The full
  Spartan test rejects a low-bit word change. Finding 1 is the remaining byte-uniqueness failure.
- For fold level `i` and coordinate `x = x_{ell-i}`, the checked identity is
  `P_{i+1}(r^2) = (1-x)(P_i(r)+P_i(-r))/2 + x(P_i(r)-P_i(-r))/(2r)`.
  The verifier starts with proof-supplied `P_0(r^2)`, reconstructs the later row entries, and passes
  the reconstructed row to the cubic KZG batch. Thus each reconstructed value is bound to its fold
  commitment. The scoped stream tamper test changes `P_0(r^2)` and rejects.
- Review #4's co-pointing inventory remains intact: Stage B leaves the same single packed claim at
  `(r_A, s_slot)`, and the commitment weights use verifier-derived `s_group`. Statement-owned rows,
  columns, packing, terms, stage encoding, public decoder kinds, and opening-point dimensions fix
  every proof shape used by the verifier.
- `git diff --check` passed. Scoped source maximum: 816 lines. No added `#[allow]`, unsafe code, or
  unrelated shape abstraction found.

## Verification

- `cargo clippy -p jolt-wrapper -p jolt-hyperkzg --all-targets -q --message-format=short -- -D warnings`: passed.
- `cargo nextest run -p jolt-wrapper -p jolt-hyperkzg --cargo-quiet`: 45 passed, 1 skipped.
- `cargo nextest run -p jolt-wrapper --features prover-fixtures --cargo-quiet -E 'test(fibonacci_2_18_relation)' --no-capture`: cached fixture loaded; 1 passed, 26 skipped.

VERDICT: 1 blocker, 1 major, 0 minors
