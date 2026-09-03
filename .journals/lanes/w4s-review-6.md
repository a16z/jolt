# W4-S review #6

Scope: committed tree `fab285197`, reviewed read-only in detached worktree
`/Volumes/Dev/worktrees/jolt/w4s-review6` against `0eee0134b`. Review #5's two findings,
the stream/Spartan and HyperKZG verifier paths, the scoped tests, and the two W4-S journals were
checked.

## Findings

1. **MAJOR — `crates/jolt-wrapper/src/stream/protocol.rs:483`: the N4 gas total still gives every
   verifier inversion zero cost.** `VerifierCost::fr_mul` now exactly counts explicit source-level
   field multiplications, with the audited convention that `inverse()` counts as zero
   multiplications. The accepted compressed path nevertheless executes five inversions: the Stage-A
   batching coefficient here, `two_r` at `crates/jolt-hyperkzg/src/scheme.rs:256`, and three
   interpolation denominators at `crates/jolt-hyperkzg/src/kzg.rs:250`. The arkworks CPU backend uses
   binary extended-GCD rather than field `Mul`, so 3,072 is correct under that narrow convention;
   an EVM verifier must still pay for inversion. `.journals/plan-evm-verifier.md:48` budgets one
   batched modexp plus about ten multiplications (about 5,000 gas), but
   `crates/jolt-wrapper/tests/stream_timing.rs:35-41` and both updated W4-S gas tables omit it while
   listing only unrelated exclusions. KZG-committed Stage A is worse: its interpolation performs
   three inversions per committed round before the four final-opening inversions. **Fix:** observe
   inversions separately, price their intended batched EVM implementation, and update the totals;
   alternatively label the current totals as excluding inversion and add the plan's inversion line.

## Verified fixes and regression audit

- `Fr::from_challenge_bytes` reads a 16-byte little-endian integer and masks bits 125–127 in its
  high limb. `bytes[15] & 0xe0` is therefore the exact pre-decoder canonicality check. It runs before
  the decoder. The baseline proof accepts its canonical word; the test sets each of the seven
  nonzero high-bit patterns and every alias rejects.
- `Scalar128` interprets all 16 bytes as a big-endian integer. Every such integer is below the
  BN254 scalar modulus, so the map is injective and needs no extra canonicality check.
- Proving deterministically recovers the `Challenge125` post-mask word by multiplying the recorded
  field value by `2^128`, emits it little-endian, and round-trips it through the production decoder.
  `Scalar128` emits the checked `u128` big-endian. The real recording-transcript fixture covers both
  kinds.
- Every explicit `Fr` multiplication reachable from `verify_stream_with_cost` now crosses an
  observer call; no bypassing `*`, `square`, or `pow` remains. The synthetic total is independently
  structured: 90 for compressed A, 120 for 24 five-factor tensor terms, 2,658 for 120 six-round
  column members, 7 for the three-bit group table, and 197 for the 15-variable HyperKZG opening.
  The test does not derive its expected value from `VerifierCost`.
- Observer plumbing preserves the prior algebra and transcript schedule. Group weights retain the
  big-endian eq-table order; BDFG degree/shift checks are unchanged; the Gemini fold still binds the
  proof-supplied `P_0(r^2)` and reconstructs later entries before the cubic KZG batch. Review #4/#5
  co-pointing and single-opening arguments remain intact.
- `git diff --check` passed. Scoped source maximum: 889 lines. No added `#[allow]`, unsafe code, or
  unrelated shape abstraction found.

## Verification

- `cargo clippy -p jolt-wrapper -p jolt-hyperkzg --all-targets -q --message-format=short -- -D warnings`:
  passed.
- `cargo nextest run -p jolt-wrapper -p jolt-hyperkzg --cargo-quiet`: 45 passed, 1 skipped.

VERDICT: 0 blockers, 1 majors, 0 minors
