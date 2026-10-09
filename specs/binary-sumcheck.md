# Spec: Sum-Check over Binary Fields

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | Quang Dao, Claude              |
| Created     | 2026-10-09                     |
| Status      | proposed                       |
| PR          |                                |

## Summary

`specs/binary-field.md` adds fields of characteristic 2 to `jolt-field` so that Jolt can produce and consume the evaluation claims of Akita's bit-table commitment, which follows LaBinius (ePrint 2026/2103). `jolt-sumcheck` is generic over `Field`, yet two parts of it assume that 2 is invertible. The batched prover pads a shorter member by extending its summand constantly over the batch's extra variables, which multiplies its claim by a power of two and then halves it once per inactive round; in characteristic 2 the padded claim is zero and `Field::two_inv` panics. The centered integer domain of the univariate-skip round maps consecutive integers into the field, where they collide. This spec gives the batched prover a padding rule that is valid in characteristic 2, makes the integer domain reject a field in which its points are not distinct, and adds a test suite that runs the engine over the binary fields. Behaviour in odd characteristic does not change.

## Intent

### Goal

`BatchPrelude` and `prove_batch` produce a sound batched sum-check over a field of characteristic 2, including batches whose members have different round counts, and every other entry point of `jolt-sumcheck` either works in characteristic 2 or fails with a typed error.

Key abstractions:

- **Padding rule.** How a member with fewer rounds than the batch is extended over the extra variables. It fixes three things: the factor applied to the member's input claim in the combined claim, the round polynomial the member contributes in a round outside its window, and the factor relating the member's final claim to its own polynomial at its opening point. `crates/jolt-sumcheck/src/batch.rs` owns the rule; `prove_batch` consumes it.
- **Constant extension** (odd characteristic, today's rule). The summand does not depend on the extra variables. Claim factor `2^(max_num_vars − rounds)`, inactive round polynomial the constant `claim / 2`, output factor 1.
- **Zero extension** (characteristic 2, new). The summand is multiplied by `∏ (1 − x_j)` over the variables `j` outside the member's window, so it is supported on the slice where those variables are 0. Claim factor 1, inactive round polynomial `claim · (1 − X)`, output factor `∏ (1 − r_j)` over the challenges outside the window.

### Invariants

Write `n = max_num_vars`, and for member `i` write `a_i` for its input claim, `c_i` for its batching coefficient, `W_i = [offset_i, offset_i + rounds_i)` for its window, and `g_i` for the polynomial in `rounds_i` variables whose sum over the Boolean hypercube is `a_i`. Write `r` for the vector of the batch's `n` challenges and `r_{W_i}` for its restriction to the window.

1. **Rule selection.** The rule is constant extension when `F::from_u64(2).inverse()` is `Some`, and zero extension when it is `None`. The selection is made in one place in `batch.rs`. No trait in `jolt-field` changes.
2. **Odd characteristic is unchanged.** For every field in which 2 is invertible, `BatchPrelude::claimed_sum`, every round polynomial, every transcript byte, and every field of `ProvedBatch` are identical to those produced before this change. No existing test, fixture or expected value is edited.
3. **Zero extension, combined claim.** In characteristic 2, `claimed_sum = Σ_i c_i · a_i`.
4. **Zero extension, inactive rounds.** In characteristic 2, in a batch round `j ∉ W_i`, member `i` contributes `c_i · m · (1 − X)` to the batched round polynomial, where `m` is its running claim, and its running claim becomes `m · (1 − r_j)`. A member enters its window with running claim `a_i · ∏ (1 − r_j)` over the rounds `j < offset_i`; a member with `offset_i = 0` enters with `a_i`, at no padded scale.
5. **Zero extension, output.** In characteristic 2, `ProvedBatch::member_claims[i] = g_i(r_{W_i}) · ∏_{j ∉ W_i} (1 − r_j)` for an honest member, and `ProvedBatch::final_claim = Σ_i c_i · member_claims[i]`.
6. **Output factor has one owner.** `BatchPrelude::member_output_scale(member, challenges)` returns the factor by which `member_claims[member]` exceeds `g(r_W)`: `F::one()` under constant extension and `∏_{j ∉ W} (1 − r_j)` under zero extension. It returns a typed error when `member` is out of range or `challenges.len() != max_num_vars`. A verifier of a characteristic-2 batch computes its expected final claim as `Σ_i c_i · member_output_scale(i, r) · g_i(r_{W_i})`.
7. **No panic in characteristic 2.** `BatchPrelude::try_new` and `prove_batch` call neither `Field::two_inv` nor `Field::half`, and `Ring::mul_pow_2` only under constant extension.
8. **Integer domains must be distinct in the field.** `CenteredIntegerDomain::round_sum_coefficients` returns `SumcheckError::IntegerDomainNotDistinct { domain_size }` when two of the domain's `domain_size` consecutive integers have the same image in `F`, that is, when `F::from_u64(k)` is zero for some `1 ≤ k < domain_size`. `SumcheckDomainSpec::CenteredInteger`, `check_round_sum`, `prove_uniskip_clear` and `prove_uniskip_committed` inherit the error. In characteristic 2 this rejects every `domain_size ≥ 3`. For BN254 `Fr` and the Solinas fields nothing changes for any domain size the repository uses.
9. **Boolean rounds need no change.** `BooleanHypercube::round_sum_coefficients`, `CompressedPoly`, `SumcheckVerifier::verify` and `verify_compressed` are correct in characteristic 2 as written: the round check is `2·c_0 + c_1 + … + c_d = c_1 + … + c_d`, and the omitted linear coefficient is recovered as `h − c_0 − c_0 − c_2 − …`. Their code is not edited; the new tests cover them.
10. **Error enum is append-only.** `IntegerDomainNotDistinct` is appended after the existing unconditional variants of `SumcheckError`; no variant is renamed, reordered or removed.

### Non-Goals

1. Univariate skip in characteristic 2. It needs an evaluation domain inside the field (an `F_2`-subspace of a subfield) and the subfield embeddings of step 3 of the `specs/binary-field.md` roadmap.
2. Round-polynomial construction from evaluations. `UnivariatePoly::from_evals`, `from_evals_toom`, `from_evals_and_hint`, `interpolate_over_integers`, the Gruen helpers in `jolt-poly/src/split_eq.rs` and `jolt-poly/src/lagrange.rs` evaluate at consecutive integers and stay as they are. A characteristic-2 member returns its round polynomial in coefficient form.
3. The committed (zero-knowledge) recorder, the `r1cs` lowering, and the fuzz target over binary fields.
4. `jolt-kernels`, `jolt-claims`, `jolt-verifier` and `jolt-prover`. Their relations and kernels are written for odd characteristic; in particular the generated batch drivers in `jolt-verifier` keep the constant-extension formulas.
5. Any change to the wire format of proofs or to transcript labels.
6. A choice between padding rules by the caller. The rule is a function of the field.

## Evaluation

### Acceptance Criteria

- [ ] `crates/jolt-sumcheck/Cargo.toml` enables `jolt-field`'s `binary` feature for dev-dependencies only; the crate's normal dependency features are unchanged.
- [ ] `cargo fmt -q --check`, `cargo clippy -p jolt-sumcheck -q --all-targets -- -D warnings`, and the same with `--features committed,r1cs`, pass.
- [ ] `cargo nextest run -p jolt-sumcheck --cargo-quiet` and the same with `--features committed,r1cs` pass, with no edit to any existing test or expected value (invariant 2). `git diff` over `crates/jolt-sumcheck/src/tests.rs`, `src/round_scheduler_tests.rs` and the existing files under `crates/jolt-sumcheck/tests/` is empty.
- [ ] A new integration test `crates/jolt-sumcheck/tests/binary_fields.rs` contains the tests below, each run over `F128` and over `F64`. Honest members are dense tables; a member of degree `d` is a product of `d` dense multilinear tables, and its round polynomial is assembled in coefficient form by multiplying the per-point linear factors, with no interpolation.
  - [ ] **Single member.** For degrees 1, 2 and 3 and 4 variables, `prove_batch` followed by `verify_compressed_boolean` accepts; the claimed sum is the direct sum of the product table over the hypercube; the final claim equals the product of the tables' multilinear extensions at the challenge point, each computed by `jolt-poly`'s dense evaluation; prover and verifier transcript states agree.
  - [ ] **Mixed lengths, tail-aligned.** A 4-round member batched with a 2-round member at `offset = 2`. `claimed_sum` equals `Σ c_i · a_i` (invariant 3); verification accepts; `member_claims[1]` equals the multilinear extension, at all four challenges, of the 16-entry table obtained by placing the short member's 4 entries where the two extra variables are 0 and zeros elsewhere (invariant 5 against an explicit table); `final_claim = Σ c_i · member_claims[i]`.
  - [ ] **Mixed lengths, head-aligned.** The same with the short member at `offset = 0`, and with a third member at `offset = 1`, `rounds = 2`, so that one member has inactive rounds on both sides of its window.
  - [ ] **Output factor.** In each mixed-length test, `member_claims[i] == member_output_scale(i, r) · g_i(r_{W_i})`. Over BN254 `Fr`, `member_output_scale` returns one for a tail-aligned and for a head-aligned short member. Out-of-range `member` and a wrong challenge count each return the typed error.
  - [ ] **Rejection.** With an honest prover transcript: a claimed sum changed by one makes `prove_batch` return `RoundCheckFailed` at round 0; a proof with one stored coefficient changed verifies to a final claim different from `Σ c_i · member_claims[i]`; a proof with one round removed returns `WrongNumberOfRounds`.
  - [ ] **Uncompressed path.** One single-member test verifies the full-coefficient proof through `SumcheckVerifier::verify` with `BooleanHypercube` (invariant 9).
- [ ] `CenteredIntegerDomain::new(3).round_sum_coefficients(2)` over `F128` returns `IntegerDomainNotDistinct { domain_size: 3 }`; `new(2)` succeeds; `prove_uniskip_clear` over `F128` with `domain_size = 3` returns the same error and leaves the transcript state unchanged. Over BN254 `Fr`, `new(3)` succeeds (covered by the existing tests).
- [ ] The crate docs (`src/lib.rs`), the module docs of `batch.rs` and `prover.rs`, and the docs of `BatchMember`, `BatchPrelude`, `ProvedBatch`, `ProveRounds::prove_round` and `CenteredIntegerDomain` state both padding rules and which is used when.
- [ ] No diff outside `crates/jolt-sumcheck/`, `Cargo.lock` and `specs/binary-sumcheck.md`.

### Testing Strategy

The ground truth for the characteristic-2 tests is independent of the engine: hypercube sums are computed by direct summation of tables, expected outputs by `jolt-poly`'s multilinear evaluation of explicit tables (the zero-extended table for a padded member), and acceptance by the verifier in the same crate, which shares no padding code with the prover. The odd-characteristic guarantee is carried by the existing suite, which pins transcript states and proof bytes and is not edited; the verifier fixtures and `jolt-prover` suites in CI exercise the unchanged path end to end.

### Performance

`prove_batch` already computed `two_inv` once per call; it now computes one inverse of 2 through the rule selection, once per call, and `BatchPrelude::try_new` computes one more. Per round and per member the arithmetic in odd characteristic is the same multiplication by `two_inv` as before. `CenteredIntegerDomain::round_sum_coefficients` gains `domain_size − 1` integer conversions and zero tests per call; it is called once per univariate-skip round. No benchmark is added.

## Design

### Architecture

`batch.rs` gains a crate-private padding rule, an enum with one variant per rule, the constant-extension variant carrying `two_inv`. It is constructed from the field (invariant 1) and exposes exactly the three quantities of the rule: the claim factor for an exponent, the inactive round polynomial's two coefficients for a running claim, and the running claim after a challenge. `BatchPrelude::try_new` uses the first; `prove_batch` uses all three in place of its direct `two_inv` and `mul_pow_2` calls; `BatchPrelude::member_output_scale` is the public reading of the rule for verifiers (invariant 6).

Under zero extension the inactive member's contribution has degree 1, so it writes to the first two slots of the batched coefficient vector. A batch with rounds already has `max_degree ≥ 1`, so the slots exist.

Why zero extension is sound: for member `i`, `Σ_{x ∈ {0,1}^n} g_i(x_{W_i}) · ∏_{j ∉ W_i} (1 − x_j) = a_i`, because the product is the indicator of the single assignment `x_j = 0` of the extra variables. The batch is then an ordinary sum-check of `Σ_i c_i · ĝ_i` over `n` variables with claim `Σ_i c_i · a_i`, and `ĝ_i(r) = g_i(r_{W_i}) · ∏_{j ∉ W_i} (1 − r_j)`. The rule would be valid in odd characteristic too; it is not used there because that would change every existing transcript.

`CenteredIntegerDomain::round_sum_coefficients` performs the distinctness test before mapping the integer power sums into the field.

`member_output_scale` has no caller in the workspace's verifier, which is written for odd characteristic. Its contract is the one stated in invariant 6, and its first consumer is the verifier of the binary-field opening claims that motivate this work. Without it a characteristic-2 verifier would have to restate the padding rule.

### Alternatives Considered

1. **Require equal round counts in characteristic 2.** Smallest change, and no new method. Rejected: Jolt's stages batch relations over different numbers of variables as a matter of course, so every characteristic-2 caller would re-implement padding inside its members, each with its own convention.
2. **Zero extension in every characteristic.** One rule instead of two. Rejected: it changes the round polynomials and transcripts of every existing proof, and the head-aligned kernels emit at the constant-extension scale.
3. **A characteristic marker on `Field`** (an associated constant or a fallible `two_inv`). Rejected for now: `from_u64(2).inverse()` already answers the question through the existing contract, and a new trait item would touch every backend for one caller.
4. **Let the caller pick the rule.** Rejected: constant extension is unsound to request in characteristic 2 and there is no reason to pick zero extension elsewhere, so the parameter would have exactly one valid value per field.

## Documentation

Doc comments as listed in the acceptance criteria. No book change.

## Execution

One commit for the padding rule and `member_output_scale`, one for the integer-domain check, one for the test suite; or a single commit if the split is awkward.

## References

- `specs/binary-field.md`
- LaBinius, ePrint 2026/2103
- `crates/jolt-sumcheck/src/batch.rs`, `crates/jolt-sumcheck/src/prover.rs`, `crates/jolt-sumcheck/src/domain.rs`
