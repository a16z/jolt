# Spec: Whiteout masking layer and BlindFold on committed pads (M6b, M6c)

| Field     | Value                                                             |
| --------- | ----------------------------------------------------------------- |
| Author(s) | @markosg04                                                        |
| Created   | 2026-10-19                                                        |
| Status    | reviewed; implementation in progress                              |
| Parent    | `specs/shared-primitive-layer.md`, `specs/whiteout-interfaces.md` |

## Summary

Today's ZK proof commits every round polynomial and claim row with Pedersen,
and BlindFold proves an R1CS with one variable block per round. M6b adds
`jolt-zk::masking`: every private field message is sent as m̂ = m + p with p
committed before the first challenge (Whiteout §3.2, §4.1). M6c makes
BlindFold the first `ZkBackend`, proving the residual relation over (pads,
residual witness) with the existing Nova fold. M6b and M6c land as one stack.
Clear proofs, clear fixtures and Akita fixtures are byte-identical.

## Pad accounting

Masked messages are exactly the `PrivateChannel` messages clear mode sends
(interfaces §2), at their clear positions, plus one auxiliary block:

- every wire coefficient of compressed rounds (degree d: d values) and all
  coefficients of the two uni-skip rounds;
- every `Sent` output-claim cell, the uni-skip output claims, and the stage-4
  `Staged` cells at their clear position before the stage-4 batch;
- the auxiliary block: one slot per symbolic intermediate product.

Static aliases take no slot. The stage-6b booleanity `bytecode_ra` route is
decided by a runtime point equality (`Stage6bOutputPoints::claim_routes`), so
ZK mode never aliases those cells: each gets its own slot, and when the
runtime route says `Alias` the residual adds the equality row
(cell − source = 0). Clear routing is untouched.

Derived claims (batch input claims) go through `bind_derived` and are not
sent. Stage 8 follows today's ZK path (`verify_zk`, `ZkEvaluationClaim`):
the joint evaluation stays in Dory's hiding evaluation commitment, Dory's
cut point (parent §7).

`pad_layout` in `jolt-verifier` is the one derivation. It runs the masked
stage verifiers over the seeded prefix (header and commitments) followed by
zero bytes, on a throwaway transcript, with an unbounded counting view: their
own receives count the base slots, and the residual lowering in shape mode
(below) counts the auxiliary slots. No value it reads changes either count, so
the schedule's only owner is the masked verifier. Both roles call it; the
prover computes it once, commits the pads on it, and its relation replay
reuses it (`replay_stages`). The final `cursor == num_pads` check is a
release check in `BlindFold::verify` (and `CommittedPads::finish` on the
prover side): a proof that leaves scheduled pads unused is rejected.
Soundness rests on the lowering emitting a row for every clear check.

Pads are base-field coordinates: an extension element at slot `s` takes
`s..s+E::DEGREE` in `ExtField` coefficient order. Jolt/Dory has `E = F`.

## Masked transcript

Preamble and trace commitments are unchanged. A new `PADS` site follows
`absorb_public_commitments`, before stage 1's first challenge: ⌈P/L⌉
Pedersen row commitments, each with its own independent blind (a shared
blind would make C_i − C_j an unblinded commitment to p_i − p_j), sent and so
absorbed. L is `VC::capacity(setup)`, lower-bounded by
`MAX_BLINDFOLD_GENERATORS`. Stages 1–7 carry m̂ in place of each private
value, with the clear challenges. Then stage 8, the checkpoint, the masked
auxiliary block, and the BlindFold tail.

Only `ZkBackend::commit_pads` builds `CommittedPads` (pads, cursor, pad
opening data), and only `receive_pad_commitment` builds
`CommittedPadsView<PadCommitment>` (no blind). The prover's masking
`SumcheckRecorder` owns `&mut CommittedPads`, and the prover's other private
sends (uni-skip, stage 4's staged openings, the auxiliary block) go through
`CommittedPads::send`, which returns the masked values it sent. There is no
prover-side `PrivateChannel`: its methods
are generic over the message field, which a pad vector of one field cannot
serve. The verifier threads `&mut CommittedPadsView` through the ZK drivers
only; its `MaskedVerifier` is the masked `PrivateChannel`. The recorder type,
not a flag, decides the routes a proof sends on and which hidden-value checks
are skipped. Clear `verify` signatures stay concrete; only the masked leaf
readers are generic over `C: PrivateChannel` with `C::Error:
From<TranscriptError>`.

## Residual rows for Jolt's stages

`jolt-verifier/src/stages/zk/` implements `ResidualRows`. Checks are
evaluated over affine forms in the pads; zero outputs become rows. A row
whose pad coefficients cancel but whose constant is nonzero is kept (it is
unsatisfiable), never dropped.

- **Compressed rounds:** weights from `SumcheckDomain::round_sum_coefficients`;
  c₁ is solved from the running claim; v_i = Σ c_j r_iʲ stays affine. No rows.
- **Uni-skip:** two rows each: Σ_D f = input claim, and
  (v̂ − p_v) − Σ_j (ĉ_j − p_j) r₀ʲ = 0 (the clear check of `uniskip.rs`).
- **Claims:** the batched input claim is affine; v_n = Σ_k β_k · output_k is
  one row per stage.
- **Products:** `jolt-claims` expressions become factored: Σ coeff · ∏ affine
  factors. `output_expression` is derived from it by expansion, so each
  relation has one formula and clear bytes are unchanged. Spartan outer,
  instruction read-RAF and the Spartan product remainder are ported; monomial
  relations factor trivially. A term with factors f₁…f_d becomes a chain:
  a factor that is not a single masked slot gets an auxiliary slot and a
  defining affine row; then d−1 `ProductRow`s with x a masked slot and z a new
  auxiliary slot. Slots are allocated per symbolic factor and term, never
  skipped because a coefficient happens to be zero.
- **Stage 8** belongs to the back end's PCS integration in `jolt-prover` /
  `jolt-verifier`, not to the stage-agnostic lowering: for Dory,
  y − Σ γ_i scale_i (m̂_i − p_i) = 0 with y = `ResidualVar::Witness(0)`.

The lowering is generic over its value domain: symbolic shape (layout),
affine forms (verifier rows), decoded values (prover auxiliary values). The
prover never restates it. **A1:** the auxiliary block comes after stage 8.
This is sound because auxiliary slots form an acyclic chain, each defined by
exactly one product row or defining affine row whose inputs are earlier
slots, and no base-protocol challenge depends on them.

## BlindFold lowering

`AffineRow` → A = Σa_k x_k + b·u, B = u, C = 0. `ProductRow` →
A = u·m̂_x − p_x, B = y, C = z, where the constants of y and z also multiply
u (homogenized for the folded u ≠ 1 instance). W = [pad rows | dedicated y
and ρ rows | padding]. The relaxed instance's pad-row commitments are the
transcript's `PADS` commitments, never recommitted. The random instance is
regenerated for this layout: fresh uniform rows and blinds for every pad row
and for the y/ρ rows. `BlindFold::Witness = (y, ρ)`;
`ResidualCommitment` is Dory's hiding evaluation commitment.

Whiteout will be `Whiteout<R: ResponseProver<F>>` with R reached through
`Setup`, since `ResponseProver` is not dyn-compatible.

## Deleted

- `jolt-sumcheck`: `committed.rs`, `r1cs.rs`, the `committed`/`r1cs`
  features and optional deps, `CommittedSumcheckRecorder`,
  `prove_uniskip_committed`, `verify_committed`'s commitment path, the
  committed consistency/claim types, committed cases in `tests.rs`,
  `tests/committed.rs`.
- `jolt-claims::r1cs` and the `claim_lowering_equivalence` fuzz target.
- `jolt-zk`: `statements.rs`, `builder.rs`, `assignment.rs`, the stage
  layout in `r1cs.rs`, the old `BlindFoldDimensions` rows and
  `FoldingCommitments` fields, and the pipeline/real-prover tests with their
  support module.
- `jolt-verifier`: `stages/zk/{inputs,outputs}.rs`, the commitment ZK
  branches of `stage{1..7}`, `CommittedClaimLayout` and its helpers,
  `CheckedInputs::committed_row_len`, derive's `committed_claim_layout`.
- `jolt-prover`: `ZkStageWitnesses`, `STAGE_DOMAINS`, the ZK fields of
  `ProvedUniskipMode`, the committed engine twin.
- `specs/jolt-prover-blindfold.md`'s per-round design and the book page.

Kept: `ClaimRecorder`, since the `impl_stage_prover!` driver is generic over
the recorder and the recorder type decides which claim routes the batch
sends (the clear routes, or `ClaimRoutes::masked`); and
`tests/engine_twins.rs`, with its clear and uni-skip twins and a masked twin
(`MaskingRecorder` against the generated `verify_masked`) in place of the
committed one. The `claim_lowering_equivalence` fuzz target is replaced by a
property test that the shape, values and rows domains of `lower_factored`
agree on random factored formulas.

Regenerated: ZK fixtures (renamed, since fixtures are cached by name), the
golden-shape test, the FS inventory, profiling spans.

## Security sketch

**Soundness.** PADS binding is computational (DLog): it fixes p before the
first challenge, so each decoded m = m̂ − p precedes the next challenge
(§4.1). BlindFold's extractor gives W satisfying the R1CS whose pad segment
opens `PADS`, so it equals p. The curve analogue of Lemma
`wo:lem:residual-correct` (products are native gates: no constant-term lift,
no scalarity rows) decodes an accepting clear transcript with the same
challenges, prefix-preserving. The y row and Dory's ZK opening bind the
final evaluation; clear Jolt soundness finishes. Two tests back the
lowering: a structural check that every pad has a nonzero coefficient in
some row (jolt-verifier), and a prover-side probe that shifts each decoded
base value of the muldiv proof, recomputes the auxiliary block as a cheater
would, and asserts a row breaks (jolt-prover `zk_e2e`, with a two-value
test binding the stage-6b alias cell to its source).

**Zero knowledge** (Theorem `wo:thm:zk`, fewer hybrids, no retries):
per-row-blinded Pedersen hides p; one-time uniform pads make m̂ uniform; the
folded witness W″ = W + rW′ is uniform; the folded opening reveals
y + r·y′ and ρ + r·ρ′ only; Dory's and BlindFold's simulators share y_com.
Pre-existing leakage is unchanged: the `ProofHeader` fields, and preprocessing
commitments are ZK only relative to themselves.

## Cost (muldiv, T = 2¹⁰)

| | clear | per-round Pedersen | masked |
|---|---|---|---|
| proof bytes | 97,663 | 65,599 | 113,951 |
| bytes before the BlindFold tail | — | 44,959 | 102,143 |
| BlindFold tail bytes | — | 20,640 | 11,808 |
| BlindFold prover (replay, auxiliary block, fold, tail) | — | 262–272 ms | VERIFY_PROVER |
| verifier | — | VERIFY_BASE | VERIFY_MASKED |

The layout is 2,073 pads: 1,965 base slots and 108 auxiliary slots. The
relation has 16 affine rows and 106 product rows. Masked messages are the
clear proof's field messages, so the masked proof is the clear proof plus
16,288 B (+17%): the pad commitment (65 rows, 2,080 B), the auxiliary block
(3,456 B), the BlindFold tail (about 8,350 B) and Dory's hiding opening
(about 2,370 B). The per-round design was smaller at this size because it
committed round polynomials in rows instead of sending them; the masked
proof grows with the round and claim count like the clear proof does.

VERIFY_NOTE

Open size reductions, not implemented:

- A public per-round degree schedule, so rounds send only the coefficients
  their member degrees need: about 33.5 KB.
- Taking the y row commitment to be Dory's hiding evaluation commitment
  instead of a separate row: about 2.3 KB.
- Carrying the auxiliary products as BlindFold witness instead of masked
  messages: about 3.3 KB.

## Open questions

- Pinning L independently of the setup: `MAX_BLINDFOLD_GENERATORS`' uni-skip
  rationale disappears, since rounds no longer occupy witness rows.
- Deferred: `ZkError::FoldOverflow` (W2); extension-to-base row expansion
  (W3; the masking channel already implements extension methods with
  per-coefficient pads); `operand_pads` (W4, from the same shape routine, with
  a release check that every `ProductRow.x.pad ∈ operand_pads`).
