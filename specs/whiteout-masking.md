# Spec: Whiteout masking layer and BlindFold on committed pads

| Field     | Value        |
| --------- | ------------ |
| Author(s) | @markosg04   |
| Created   | 2026-10-19   |
| Status    | design note  |

## Summary

Today's ZK proof commits every round polynomial and claim row with Pedersen,
and BlindFold proves an R1CS with one variable block per round. This note adds
`jolt-zk::masking`: every private field message is sent as m̂ = m + p with p
committed before the first challenge (Whiteout §3.2, §4.1). It also makes
BlindFold the first `ZkBackend`, proving the residual relation over (pads,
residual witness) with the existing Nova fold. BlindFold stays curve-based
(Pedersen and Dory, as today); only its input changes, from per-round
commitments to pads committed up front. The lattice path is Akita's.
Clear proofs, clear fixtures and Akita fixtures are byte-identical.

## Pad accounting

Masked messages are exactly the `PrivateChannel` messages clear mode sends,
at their clear positions, plus one auxiliary block:

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

Batch input claims are neither sent nor absorbed:
`MaskingRecorder::absorb_input_claims` absorbs nothing. Every masked input
claim is a function of values already absorbed as pad-masked sends (earlier
`finish` outputs, the uni-skip output claims, the stage-6a address claims),
challenges and public data. The stage-4 staged openings that the RAM value
check subtracts (advice and program image) are sent through
`send_private`/`receive_private` before the stage-4 proof of work and γ
draw. The clear recorder, by contrast, absorbs each input claim with
`append_private`.

Stage 8 follows today's ZK path (`verify_zk`, `ZkEvaluationClaim`):
the joint evaluation stays in Dory's hiding evaluation commitment, Dory's
cut point.

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
`CommittedPads::send`, which returns the masked values it sent. For
protocols written once over both roles' channels (`prove_single`, Akita's
masked levels), `CommittedPads::channel` is the prover's masked
`PrivateChannel`, `MaskedProver`, over the same masking. A pad vector has
one field: the prover masks base-field coordinates, and an extension element
takes one pad per base coordinate (`CommittedPadsView::take_ext` on the
verifier side). The verifier threads `&mut CommittedPadsView` through the ZK drivers
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
  Eq-factored rounds (`RoundFormat::EqFactored`, Akita's stage 1) solve q₀
  from the weights `[1, τ_i, τ_i, …]` the same way; the running claim is the
  normalized q_i(r_i).
- **Uni-skip:** two rows each: Σ_D f = input claim, and
  (v̂ − p_v) − Σ_j (ĉ_j − p_j) r₀ʲ = 0 (the clear check of `uniskip.rs`).
- **Claims:** the batched input claim is affine; v_n = Σ_k β_k · output_k is
  one row per stage.
- **Products:** `jolt-claims` expressions become factored: Σ coeff · ∏ affine
  factors. `output_expression` is derived from it by expansion, so each
  relation has one formula and clear bytes are unchanged. Spartan outer,
  instruction read-RAF and the Spartan product remainder are ported; monomial
  relations factor trivially. A term with factors f₁…f_d becomes a chain.
  Under the lattice back end's operand rule, a factor that is not a single
  masked slot first gets an auxiliary slot and a defining affine row;
  BlindFold takes it as an affine operand. Then come d−2 `ProductRow`s, each
  with x the operand and z a new auxiliary slot, and the last product stays
  pending as a pair of the stage's check row (below). Slots are allocated per symbolic factor and term, never
  skipped because a coefficient happens to be zero.
- **Stage 8** belongs to the back end's PCS integration in `jolt-prover` /
  `jolt-verifier`, not to the stage-agnostic lowering: for Dory,
  y − Σ γ_i scale_i (m̂_i − p_i) = 0 with y = `ResidualVar::Witness(0)`.

The lowering is generic over its value domain, `jolt_transcript::ClaimDomain`
(shared with Akita's stage checks, whose clear verifier uses its
`ClearDomain`): symbolic shape (layout), affine forms (verifier rows), decoded
values (prover auxiliary values). Its checks are `check_zero` (an affine row)
and `products_into` (a product row Σ_k x_k·y_k = z whose z is an existing
value, so no auxiliary slot; `product_into` is its one-pair case). Each
stage's check, running claim against the batched output, is one
`products_into` row whose pairs are the output terms' last products, or an
affine row when the outputs are affine; alias equalities go through
`check_zero`. The prover never restates it. The auxiliary block comes after stage 8.
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

## Deleted by the switch to masked ZK

- `jolt-claims::r1cs` and its `claim_lowering_equivalence` fuzz workspace.
- `jolt-zk`: `statements.rs`, `builder.rs`, `assignment.rs`, the stage
  layout in `r1cs.rs`, the old `BlindFoldDimensions` rows and
  `FoldingCommitments` fields, and the pipeline/real-prover tests with their
  support module.
- `jolt-verifier`: `stages/zk/{inputs,outputs,committed}.rs`,
  `stages/zk/blindfold`, the commitment ZK branches of `stage{1..7}`,
  `CommittedClaimLayout` and its helpers, `CheckedInputs::committed_row_len`,
  the derive's `committed_claim_layout`.
- `jolt-prover`: `ZkStageWitnesses`, `STAGE_DOMAINS`, the ZK fields of
  `ProvedUniskipMode`, the committed engine twin.
- `specs/jolt-prover-blindfold.md` (the per-round design); the book's
  BlindFold chapter is rewritten for masking.

Not deleted yet: `jolt-sumcheck`'s `committed` and `r1cs` modules. No
protocol path uses them after the switch, but `SumcheckProof::Committed` is
the only field of `JoltProof` that names the vector-commitment parameter, so
removing it means removing that parameter from `JoltProof` and every
signature that carries it; that is a separate change.

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
| prover ZK work: pad commitment (incl. layout) + BlindFold tail (replay, auxiliary block, fold, tail) | — | 393–402 ms (tail only) | 85–96 + 308–313 ms |
| full prove, including guest preparation | — | 1,770–1,797 ms | 1,660–1,688 ms |
| verifier | — | 300 ms | 319 ms |

The layout is 2,073 pads: 1,965 base slots and 108 auxiliary slots. The
relation has 16 affine rows and 106 product rows. Masked messages are the
clear proof's field messages, so the masked proof is the clear proof plus
16,288 B (+17%): the pad commitment (65 rows, 2,080 B), the auxiliary block
(3,456 B), the BlindFold tail (about 8,350 B) and Dory's hiding opening
(about 2,370 B). The per-round design was smaller at this size because it
committed round polynomials in rows instead of sending them; the masked
proof grows with the round and claim count like the clear proof does.

Times are medians of repeated runs in the test profile (opt-level 3) on
one machine, optimized prover backend. The verifier's extra 19 ms is
mostly `pad_layout`, which replays stages 1–7 a second time (about 46 ms);
the masked stages themselves are cheaper than the per-round Pedersen
checks. The prover derives the layout once and reuses it in its relation
replay.

Rounds follow a public per-round degree schedule
(`jolt_sumcheck::DegreeSchedule`): each batch round is sent at the largest
degree its active members declare for that round
(`SymbolicSumcheck::round_degree`), so stage 5's 128 instruction read-RAF
address rounds go at degree 2 instead of 10. The table above predates it.
With it, muldiv proofs are 64,895 B clear (−32,768) and 79,071 B masked
(−34,880: the masked coefficients plus fewer pad-commitment rows), and the
layout is 1,049 pads. Stage 2's RAM read-write address rounds (3
coefficients) and stage 6a's bytecode address phase (20 coefficients) still
send zero high coefficients on muldiv; declaring their phase degrees would
save about 0.7 KB more.

Open size reductions, not implemented:

- Taking the y row commitment to be Dory's hiding evaluation commitment
  instead of a separate row: about 2.3 KB.
- Carrying the auxiliary products as BlindFold witness instead of masked
  messages: about 3.3 KB.

## Multi-pair product rows and merged products

Muldiv's layout has 108 auxiliary slots: 106 product slots and 2 defining
slots. The count does not depend on T, and it grows with the number of
one-hot chunks. The paper's count of 32 (`06_evaluation.tex:83-88`) rests on
two assumptions:

- the last product of every term lands in its stage check without a slot;
- a product that repeats across terms is computed once.

Three changes realize both assumptions. They land as separate commits, in
this order.

**1. Merged products (jolt-claims).** The bytecode read-RAF cycle output is
Σ_t c_t · ra₀ra₁ra₂, where each c_t is public (the γ-weighted stage values,
the two RAFs and the entry). Its factored form becomes
`product([Σ_t c_t, ra₀, ra₁, ra₂])`, and `output_expression` is defined as the
expansion of that form, following the `outer_remainder` pattern. One formula
keeps one owner. Clear proofs are unchanged, because the expansion has the
same value. The lowering takes 2 slots for this output instead of 16.

**2. Multi-pair rows.** A product row becomes

    ProductRow { first: ProductPair, rest: Vec<ProductPair>, z }
    ProductPair { x: ProductOperand, y: AffineRow }

with the check Σ_k x_k·y_k = z over `first` and `rest`. A row with one pair
is the old row. The type makes a row without a pair unrepresentable; such a
row would lift to its −Z alone, so no back end needs a check.

- *ClaimDomain op.* `jolt_transcript::ClaimDomain` gains
  `products_into(first, rest, target)`, the check Σ x_k·y_k = target with no
  new slot. `product_into` becomes a provided method,
  `products_into((x, y), Vec::new(), target)`. The change is additive: Akita only calls these methods.
  `ClearDomain` and `ValuesDomain` compare, `RowsDomain` emits one
  `ProductRow`, and `OperandShapeDomain` records each pair's operand pad.
- *Lowering.* `lower_factored` returns `ProductSum { pairs, affine }`. A
  term with h ≥ 2 hidden factors takes h − 2 product slots, plus a defining
  slot when it has no single-opening factor. Its last product stays pending as
  the pair (x, scale·y). `add` and `scale` act on both parts.
  `ProductSum::into_value` materializes the pending pairs through `product`.
  Input claims use it; on Jolt they are affine, so it takes no slot.
- *Stage check.* Each stage emits one row,
  `products_into(β-scaled output pairs, running − output.affine)`, through a
  single `JoltResidual::check_stages`, which calls `ProductSum::check_equals`
  (an affine row when no pair is pending). The pad layout (shape domain) and
  the rows both call it, so the operand pads and the rows have one owner.
- *BlindFold.* A row with n pairs lowers to n R1CS constraints:
  x_k·y_k = w_k for k < n, and x_n·y_n = z − Σ w_k. The w_k are witness-only
  variables. They occupy extra auxiliary witness rows after the y and ρ rows,
  so they are committed, randomized and folded with those rows. Their number
  follows from the relation's shape. The prover computes the w_k from the
  decoded values.
- *Whiteout Σ.* The lift flattens pairs into product entries, each tagged
  with its row. Entry (i, k) carries (x̂_ik, P_x, Y_ik). Only the row's first
  entry carries Z_i; the other entries carry zero. Every entry is weighted by
  its row's μ_i, so Q_μ = Σ_i μ_i(Σ_k X_ik Y_ik − u Z_i) + u(λ + g). Labels
  count rows. The product sumcheck runs over entries, with
  U_e = μ_{row(e)}[X_e](α) and W_e = [Y_e](α). `first_violation` sums the
  pairs of each row.

**3. Affine operands (BlindFold only).** An R1CS A-row accepts any linear
combination, so BlindFold needs no defining slot.

- `OperandRule` is `MaskedSlot` for the lattice back end and `Affine` for
  the curve back end. BlindFold states its rule as the inherent const
  `BlindFold::OPERANDS`, which jolt-verifier reads; the lattice path keeps
  the default until the Jolt integration gives it a caller.
- `ProductPair.x` becomes `ProductOperand { Masked(MaskedOperand),
  Affine(AffineRow) }`.
- The three lowering domains implement `LoweringDomain`, which states the
  rule. Their constructors default to `MaskedSlot`, so Akita's uses are
  unchanged; `with_operands` selects a rule.
- `lower_factored` reads the rule from its domain, so one lowering has one
  rule. `JoltResidual` carries the rule from `ResidualInputs.operands` into
  the layout, the values and the rows.
- Under `Affine`, rows emit `Affine` only for an operand that is not a
  single slot, and the shape records no operand pads.
- The lattice lift rejects `Affine` with a release check, where it reads
  each pair's operand pad.

**Σ-statement impact.** The public surface does not change.
`ResponseStatement::{layout, challenge, rho, v_image,
check_witness}` keep their signatures and meaning. `ResponseLayout` is
unchanged, since θ still has 63 coefficients. With multi-pair rows,
`row_shape().products` and the lengths of `product_x` and `product_y` now
count product entries (pairs), not rows. `U_e` carries its row's μ, and Z
appears once per row inside `weigh`'s adjoint. `rho().len()` is ℓ for the
smaller row count. A `ResponseProver` that treats products as an opaque list
of pairs needs no change. The edits are internal to `lift.rs`:
`LiftedProduct` gains a row index, `product_weights` holds one weight per
entry, and `first_violation` groups entries by row. `CheckingResponseProver`
is unchanged.

**Security.** *Soundness.* By Lemma `wo:lem:ct-lift`, a constant operand
gives ct(X·Y) = x·ct(Y). The lift of Σ_k X_k Y_k − uZ is therefore exact pair
by pair and linear in the sum: ct(Σ_k X_k Y_k − Z) = Σ_k x_k y_k − z. The
aggregation is unchanged: Q_μ is Σ over rows of μ_i·row_i, with distinct
multilinear labels, so a nonzero row survives with probability at least
1 − ℓ/|F|. The product sumcheck proves Σ_e U_e W_e for any entry list.
Several entries sharing one μ does not affect its soundness. Scalarity rows
still pin every pair's operand pad. The decoded relation is the clear
verifier's own stage check, Σ_m β_m output_m = running claim, which also
checks only the sum. Clear Jolt soundness applies to the decoded transcript
as before. In particular, moving value between two pairs of one row changes
an opening or a slot that another row pins: a later stage's input claim,
stage 8's evaluation row, or a slot's defining row. *Zero knowledge.* The
shape (rows, pairs, labels) is public and fixed by the schedule. BlindFold's
w_k are hidden by the same row blinds and folding as the y and ρ rows.
Whiteout's response proof is ZK in e for any entry count. Fewer masked
slots means fewer one-time pads, never a revealed value.

**Bytes.** Clear proofs and Akita fixtures are byte-identical: the clear
muldiv fixture regenerated before and after the three changes has the same
SHA-1. ZK proofs, the pad count, the auxiliary block and BlindFold's
auxiliary row count change. Measured on muldiv at T = 2¹⁰ (the golden test
pins the shape):

| | pads | auxiliary | rows (affine / product / pairs) | ZK proof | BlindFold tail |
|---|---|---|---|---|---|
| before | 1,049 | 108 | 16 / 106 / 106 | 79,071 B | 7,264 B |
| 1. merged products | 1,035 | 94 | 16 / 92 / 92 | 78,623 B | 7,264 B |
| 2. multi-pair rows | 969 | 28 | 10 / 32 / 92 | 76,511 B | 7,328 B |
| 3. affine operands | 967 | 26 | 8 / 32 / 92 | 76,447 B | 7,328 B |

Each auxiliary slot is one 32-byte masked value plus one pad, and every 32
pads take one 32-byte pad-row commitment. Change 2 adds 60 BlindFold
intermediates, which take two witness rows; the tail grows by 64 B. In all,
the masked proof shrinks by 2,624 B (3.3%). Stages 2–6 each land their
output products on one row, with 4, 4, 5, 2 and 50 pairs. The 26 remaining
slots are the inner products of terms with three or more hidden factors:
the instruction read-RAF RA chain (7), the bytecode RA product (1), the
instruction RA virtualization chunks (8 × 2) and one four-factor stage-6
term (2).

## Open questions

- Pinning L independently of the setup: `MAX_BLINDFOLD_GENERATORS`' uni-skip
  rationale disappears, since rounds no longer occupy witness rows.
- Deferred: `ZkError::FoldOverflow` (Akita's one-shot fold responses);
  extension-to-base row expansion (Akita's masked levels; the masking channel
  already implements extension methods with per-coefficient pads);
  `operand_pads` (`specs/whiteout-sigma.md`, from the same shape routine, with
  a release check that every `ProductRow.x.pad ∈ operand_pads`).
