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
2. **Odd characteristic is unchanged.** For every field in which 2 is invertible, `BatchPrelude::claimed_sum`, every round polynomial, every transcript byte, every field of `ProvedBatch`, the claim passed to each member, and the order and kind of every error are identical to those produced before this change. In particular a member still receives its claim at the constant-extension scale, and its returned polynomial is folded as returned: the precommitted-reduction kernels in `jolt-kernels` build polynomials whose coefficients encode that scale. No existing test, fixture or expected value is edited.
3. **Zero extension, combined claim.** In characteristic 2, `claimed_sum = Σ_i c_i · a_i`.
4. **Zero extension, engine state.** In characteristic 2 the engine keeps two values per member: its *native claim* `m_i`, initially `a_i`, and its *padding factor* `p_i`, initially one. Neither is ever recovered from the other or from their product by division; either can be zero.
   - In a batch round `j ∉ W_i` the member contributes `c_i · p_i · m_i · (1 − X)` to the batched round polynomial, and then `p_i ← p_i · (1 − r_j)`. Its native claim does not change.
   - In a batch round `j ∈ W_i` the engine passes `m_i` to `ProveRounds::prove_round` as `previous_claim`, receives the member's native round polynomial `s`, contributes `c_i · p_i · s(X)`, and then sets `m_i ← s(r_j)`. The factor does not change.
   A member therefore never sees a padded claim and emits round polynomials at no padded scale, wherever its window sits. The scaling by `p_i` inside the window is what makes a window with inactive rounds before it correct: a member whose native claim is zero cannot learn `p_i` from the claim it is handed.
5. **Zero extension, output.** In characteristic 2, `ProvedBatch::member_claims[i] = p_i · m_i` at the end of the loop, which for an honest member is `g_i(r_{W_i}) · ∏_{j ∉ W_i} (1 − r_j)`, and `ProvedBatch::final_claim = Σ_i c_i · member_claims[i]`.
6. **Output factor has one owner.** `BatchPrelude::member_output_scale(member, challenges)` returns the multiplier `λ` with `member_claims[member] = λ · g(r_W)` for an honest member: `F::one()` under constant extension and `∏_{j ∉ W} (1 − r_j)` under zero extension, where `challenges[j]` is the challenge of batch round `j`. It validates the prelude's dimensions as `prove_batch` does and returns a typed error when they are invalid, when `member` is out of range, or when `challenges.len() != max_num_vars`. A verifier of a characteristic-2 batch computes its expected final claim as `Σ_i c_i · member_output_scale(i, r) · g_i(r_{W_i})` at the point `r` returned by its own round verification.
7. **No panic in characteristic 2.** `BatchPrelude::try_new` and `prove_batch` call neither `Field::two_inv` nor `Field::half`, and `Ring::mul_pow_2` only under constant extension. Dimension validation, including the rejection of a padding exponent above 255, is unchanged and applies under both rules.
8. **Integer domains must be distinct in the field.** `CenteredIntegerDomain::round_sum_coefficients` proceeds in this order: validate the domain size with the existing size validation (`InvalidIntegerDomain` on failure, as today); return `SumcheckError::IntegerDomainNotDistinct { domain_size }` if `F::from_u64(k).is_zero()` for some `1 ≤ k < domain_size`, which is exactly the condition for two of the domain's consecutive integers to have the same image in `F`; then compute the integer power sums (`InvalidIntegerDomain` on overflow, as today) and map them into the field. `SumcheckDomainSpec::CenteredInteger` and `check_round_sum` inherit the error, and so do `prove_uniskip_clear` and `prove_uniskip_committed` when the round polynomial passes their earlier degree check; none of them touches the transcript first. In characteristic 2 this rejects every `domain_size ≥ 3`; size 2 is `{0, 1}` and stays valid. For BN254 `Fr` and the Solinas fields nothing changes for any domain size the repository uses.
9. **Boolean rounds need no change.** `BooleanHypercube::round_sum_coefficients`, `CompressedPoly`, `SumcheckVerifier::verify` and `verify_compressed` are correct in characteristic 2 as written: the round check is `2·c_0 + c_1 + … + c_d = c_1 + … + c_d`, and the omitted linear coefficient is recovered as `h − c_0 − c_0 − c_2 − …`. Their code is not edited; the new tests cover them.
10. **Error enum is append-only.** `IntegerDomainNotDistinct` is appended after the existing unconditional variants of `SumcheckError`; no variant is renamed, reordered or removed.

### Non-Goals

1. Univariate skip in characteristic 2. It needs an evaluation domain inside the field (an `F_2`-subspace of a subfield) and the subfield embeddings of step 3 of the `specs/binary-field.md` roadmap.
2. Round-polynomial construction from evaluations. `UnivariatePoly::from_evals`, `from_evals_toom`, `from_evals_and_hint`, `interpolate_over_integers`, the Gruen helpers in `jolt-poly/src/split_eq.rs` and `jolt-poly/src/lagrange.rs` evaluate at consecutive integers and stay as they are. A characteristic-2 member returns its round polynomial in coefficient form.
3. The committed (zero-knowledge) recorder, the `r1cs` lowering, and the fuzz target over binary fields.
4. `jolt-kernels`, `jolt-claims`, `jolt-verifier` and `jolt-prover`. Their relations, kernels and generated batch drivers are written for odd characteristic and remain unsupported over binary fields: the generated final fold in `jolt-verifier-derive` applies no output factor.
5. Any change to the wire format of proofs or to transcript labels.
6. A choice between padding rules by the caller. The rule is a function of the field.

## Evaluation

### Acceptance Criteria

- [ ] `crates/jolt-sumcheck/Cargo.toml` enables `jolt-field`'s `binary` feature for dev-dependencies only; the crate's normal dependency features are unchanged.
- [ ] `cargo fmt -q --check`, `cargo clippy -p jolt-sumcheck -q --all-targets -- -D warnings`, and the same with `--features committed,r1cs`, pass.
- [ ] `cargo nextest run -p jolt-sumcheck --cargo-quiet` and the same with `--features committed,r1cs` pass, with no edit to any existing test or expected value (invariant 2). `git diff` over `crates/jolt-sumcheck/src/tests.rs`, `src/round_scheduler_tests.rs` and the existing files under `crates/jolt-sumcheck/tests/` is empty.
- [ ] A new integration test `crates/jolt-sumcheck/tests/binary_fields.rs` contains the tests below, each run over `F128` and over `F64` (`F64` is functional coverage, not a security level). Table entries are field elements with many nonzero coefficient bits, built with `from_raw` or seeded sampling; the integer maps give only parity. Honest members are dense tables; a member of degree `d` is a product of `d` dense multilinear tables, and its round polynomial is assembled in coefficient form by multiplying the per-point linear factors, with no interpolation. The variable order is `jolt-poly`'s: a member's local round 0 binds the most significant index bit, so in a batch of `n` rounds the challenge `r_j` corresponds to bit `n − 1 − j` of a full-size table index. The recorder is `ClearSumcheckRecorder::<F, ()>`. Where transcript states are compared, the verifier side mirrors the recorder's `finish` appends.
  - [ ] **Single member.** For degrees 1, 2 and 3 and 4 variables, `prove_batch` followed by `verify_compressed_boolean` accepts; the claimed sum is the direct sum of the product table over the hypercube; the final claim equals the product of the tables' multilinear extensions at the challenge point, each computed by `jolt-poly`'s dense evaluation; prover and verifier transcript states agree.
  - [ ] **Mixed lengths, degree 1.** A 4-round member batched with 2-round members of degree 1 at `offset = 2` (tail), `offset = 0` (head) and `offset = 1` (inactive rounds on both sides). `claimed_sum` equals `Σ c_i · a_i` (invariant 3); verification accepts; each short member's `member_claims` entry equals the multilinear extension, at all four challenges, of the 16-entry table that holds the member's 4 entries at indices `{0,1,2,3}`, `{0,4,8,12}` and `{0,2,4,6}` respectively and zero elsewhere (invariant 5 against an explicit table); `final_claim = Σ c_i · member_claims[i]`.
  - [ ] **Mixed lengths, degree 2.** A 4-round degree-2 member batched with a 2-round degree-2 member at `offset = 2` and one at `offset = 1`. Each short member's entry equals the product of its two factors' multilinear extensions at `r_W`, times `∏_{j ∉ W} (1 − r_j)` taken once.
  - [ ] **Zero native claim behind leading padding.** A 2-round batch with a full-window member and a 1-round member at `offset = 1` whose table is `[u, u]` for a nonzero `u`, so that its input claim is zero and its polynomial is the nonzero constant `u`. `member_claims[1] == u · (1 − r_0)` and verification accepts (invariant 4).
  - [ ] **Zero padding factor.** With a test transcript or scheduler-independent means that forces an inactive round's challenge to one, the padded member's contribution and final claim are zero and nothing panics. If the stock transcripts cannot force a challenge, this is tested through `member_output_scale` alone with a challenge vector containing one.
  - [ ] **Output factor.** In each mixed-length test, `member_claims[i] == member_output_scale(i, r) · g_i(r_{W_i})`. Over BN254 `Fr`, `member_output_scale` returns one for a tail-aligned and for a head-aligned short member. Out-of-range `member` and a wrong challenge count each return a typed error.
  - [ ] **Rejection.** (a) A prelude whose `claimed_sum` field is changed by one, with honest members and member claims, makes `prove_batch` return `RoundCheckFailed` at round 0. (b) An honest compressed proof with one stored coefficient changed is replayed by the verifier from the pre-round transcript state, giving `(r', v')`; then `v' ≠ Σ_i c_i · member_output_scale(i, r') · g_i(r'_{W_i})`, with the right-hand side evaluated from the original tables at `r'`. The fixture is deterministic. (c) A proof with one round removed returns `WrongNumberOfRounds`.
  - [ ] **Uncompressed path.** One test builds an honest full-coefficient proof by running the rounds of a single dense member against a transcript with `LabeledRoundPoly` absorption, and verifies it through `SumcheckVerifier::verify` with `BooleanHypercube` (invariant 9). Decompressing a compressed proof is not a valid fixture, since the two forms absorb different bytes.
- [ ] `CenteredIntegerDomain::new(3).round_sum_coefficients(2)` over `F128` returns `IntegerDomainNotDistinct { domain_size: 3 }`; `new(4).round_sum_coefficients(127)` returns the same error, not the overflow error; `new(2)` succeeds; `new(0)` still returns `InvalidIntegerDomain`; `prove_uniskip_clear` over `F128` with `domain_size = 3` returns `IntegerDomainNotDistinct` and leaves the transcript state unchanged. Over BN254 `Fr`, `new(3)` succeeds (covered by the existing tests).
- [ ] **Odd-characteristic byte identity, recorded in the PR.** Before and after the change, the serialized proof bytes and final transcript state of the existing tail-aligned and head-aligned batch twin fixtures in `src/tests.rs` are printed by a temporary probe and compared; the PR description records that they are equal. The probe is not committed.
- [ ] The padding rules are documented once, on `BatchPrelude`, including the engine state of invariant 4, the temporal order of `challenges`, that the factor can be zero, and the errors of `member_output_scale`. The module docs of `batch.rs` and `prover.rs` and the docs of `BatchMember`, `ProvedBatch` and `ProveRounds::prove_round` are corrected where they describe halving as unconditional and otherwise link there. `CenteredIntegerDomain`'s docs state the distinctness requirement. The crate docs say which entry points support characteristic 2.
- [ ] No diff outside `crates/jolt-sumcheck/`, `Cargo.lock` and `specs/binary-sumcheck.md`.

### Testing Strategy

The ground truth for the characteristic-2 tests is independent of the engine: hypercube sums are computed by direct summation of tables, expected outputs by `jolt-poly`'s multilinear evaluation of explicit tables (the zero-extended table for a padded member), and acceptance by the verifier in the same crate, which shares no padding code with the prover. For odd characteristic, the existing suite compares the current prover with the current verifier, so it shows consistency rather than byte identity with the previous code. Byte identity is shown by the one-time before/after comparison above, and by the repository's CI gates that this change must keep green: the standard and ZK clippy runs, and the `jolt-verifier` fixture and `jolt-prover` suites, whose fixtures were produced by the previous code.

### Performance

`prove_batch` computed `Field::two_inv` once per call. The rule selection instead calls the general `inverse` on 2, once in `prove_batch`, once in `BatchPrelude::try_new` and once per `member_output_scale` call. On the Solinas fields and their extensions `two_inv` is a cheap specialised constant and `inverse` is an exponentiation or a small linear solve, so this adds a bounded cost per batch, of the order of a few hundred multiplications, against a round loop that does work proportional to the tables. It is not measured here. Per round and per member the arithmetic in odd characteristic is the same multiplication by the inverse of 2 as before. `CenteredIntegerDomain::round_sum_coefficients` gains `domain_size − 1` integer conversions and zero tests per call; it is called once per univariate-skip round. No benchmark is added.

## Design

### Architecture

`batch.rs` gains a crate-private padding rule, an enum with one variant per rule, the constant-extension variant carrying `two_inv`. It is constructed from the field (invariant 1). `BatchPrelude::try_new` asks it for the claim factor of an exponent. `prove_batch` branches on it: the constant-extension arm is today's loop unchanged, with one running claim per member; the zero-extension arm keeps the native claim and the padding factor of invariant 4 and scales each active member's polynomial by its factor when folding. `BatchPrelude::member_output_scale` is the public reading of the rule for verifiers (invariant 6). The crate denies `indexing_slicing` and `panic_in_result_fn`, so the two coefficient slots of an inactive contribution are reached by checked access.

Under zero extension the inactive member's contribution has degree 1, so it writes to the first two slots of the batched coefficient vector. A batch with rounds already has `max_degree ≥ 1`, so the slots exist.

Why zero extension is sound: for member `i`, `Σ_{x ∈ {0,1}^n} g_i(x_{W_i}) · ∏_{j ∉ W_i} (1 − x_j) = a_i`, because the product is the indicator of the single assignment `x_j = 0` of the extra variables. The batch is then an ordinary sum-check of `Σ_i c_i · ĝ_i` over `n` variables with claim `Σ_i c_i · a_i`, and `ĝ_i(r) = g_i(r_{W_i}) · ∏_{j ∉ W_i} (1 − r_j)`. The rule would be valid in odd characteristic too; it is not used there because that would change every existing transcript.

`CenteredIntegerDomain::round_sum_coefficients` performs the distinctness test before mapping the integer power sums into the field.

`member_output_scale` has no caller in the workspace's verifier, which is written for odd characteristic. It is added under the repository's allowance for a documented external contract: the contract is invariant 6, written out in its rustdoc, and its consumer is the future verifier of the binary-field opening claims that motivate this work. Without it a characteristic-2 verifier would have to restate the padding rule.

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
