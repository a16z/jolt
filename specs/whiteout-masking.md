# Spec: Whiteout masking layer and BlindFold on committed pads (M6b, M6c)

| Field     | Value                                                             |
| --------- | ----------------------------------------------------------------- |
| Author(s) | @markosg04                                                        |
| Created   | 2026-10-19                                                        |
| Status    | design note, awaiting review                                      |
| Parent    | `specs/shared-primitive-layer.md`, `specs/whiteout-interfaces.md` |

## Summary

Today's ZK proof commits every round polynomial and claim row with Pedersen,
and BlindFold proves an R1CS with one variable block per round. M6b adds
`jolt-zk::masking`: every private field message is sent as m̂ = m + p with p
committed before the first challenge (Whiteout §3.2, §4.1). M6c makes
BlindFold the first `ZkBackend`, proving the residual relation over (pads,
residual witness) with the existing Nova fold. Clear proofs are unchanged.

## Pad accounting

Masked messages are exactly the `PrivateChannel` messages clear mode sends
(interfaces §2), so the masked transcript has the clear shape plus one
auxiliary block:

- every wire coefficient of compressed rounds (degree d: d values) and all
  coefficients of the two uni-skip rounds;
- every `Sent` and `Staged` output-claim cell and the uni-skip output claims
  (aliases take no slot);
- the auxiliary block: one slot per intermediate product (below).

Derived claims (batch input claims, stage 8's scaled values) go through
`bind_derived` and are not sent. Stage 8's joint evaluation stays in Dory's
hiding evaluation commitment: Dory's cut point (parent §7).

`jolt_verifier::pad_layout(&CheckedInputs, &JoltFormulaDimensions)` is the one
derivation, built from the per-stage shape tables (relation objects from
`new(dimensions)`, batch statements, `CommittedClaimLayout`) that the stage
verifiers and the residual lowering also consume. Both roles call it and
check `cursor == num_pads` at the end with a typed error, so a derivation
bug costs completeness, never soundness.

Pads are base-field coordinates: an extension element at slot `s` takes
`s..s+E::DEGREE` in `ExtField` coefficient order. Jolt/Dory has `E = F`.

## Masked transcript

Preamble and trace commitments are unchanged. A new `PADS` site follows
`absorb_public_commitments`, before stage 1's first challenge: ⌈P/L⌉
Pedersen row commitments, L the BlindFold witness row length (a power of two
≤ the VC capacity). Stages 1–7 carry m̂ in place of each private value, with
the clear challenges. Stage 8 (Dory ZK opening), the checkpoint, the masked
auxiliary block, then the BlindFold tail.

Only `ZkBackend::commit_pads` builds `CommittedPads`, and only
`receive_pad_commitment` builds `CommittedPadsView`. The masking channels
borrow a transcript and one of these: the prover sends m̂ and advances the
cursor; the verifier fills m̂ and returns `PadSlots`. ZK stage verifiers
record `(m̂, slots)` and check nothing hidden.

## Residual rows for Jolt's stages

`jolt-verifier/src/stages/zk/` implements `ResidualRows`, replacing the
`BlindFoldStatement` construction. Checks are evaluated over affine forms in
the pads; only zero outputs become rows.

- **Compressed rounds:** weights from `SumcheckDomain::round_sum_coefficients`;
  c₁ is solved from the running claim, and v_i = Σ c_j r_iʲ stays affine. No
  rows.
- **Uni-skip:** one explicit row each (power sums · coefficients = input claim).
- **Claims** (`jolt-claims` expressions): the batched input claim is affine;
  v_n = Σ_k β_k · output_k is one row per stage.
- **Products:** a monomial with d ≥ 2 hidden factors becomes d−1 `ProductRow`s
  whose x is a masked slot (an opening or the previous auxiliary slot) and
  whose z is a new auxiliary slot. Sources: RA virtualization (∏ of chunk
  evaluations), booleanity, read-write terms, Spartan's τ·Az·Bz.
- **Stage 8:** y − Σ γ_i scale_i (m̂_i − p_i) = 0, y = `ResidualVar::Witness(0)`.

The prover computes auxiliary values after its verifier replay through stage
8, with the same lowering over decoded m = m̂ − p, and sends them masked.

## BlindFold lowering

`AffineRow` → A = Σa_k x_k + b·u, B = u, C = 0. `ProductRow` →
A = u·m̂_x − p_x, B = y, C = z; m̂ and challenges are matrix constants.
W = [pad rows | dedicated y and ρ rows | padding]. The relaxed instance's
pad-row commitments are the transcript's `PADS` commitments, never
recommitted (this replaces reusing round commitments). Fold, outer/inner
sumchecks, and the eval-commitment check are unchanged.
`BlindFold::Witness = (y, ρ)`.

## Deleted

- `jolt-sumcheck`: `committed.rs`, `r1cs.rs`, the `committed`/`r1cs`
  features, `CommittedSumcheckRecorder`, `prove_uniskip_committed`,
  `verify_committed`, the committed consistency/claim types, `tests/committed.rs`,
  and the optional `jolt-crypto`/`jolt-r1cs`/`rand_core` dependencies.
- `jolt-claims::r1cs` (only `jolt-zk` used it) and its fuzz target.
- `jolt-zk`: `statements.rs`, `builder.rs`, `assignment.rs`, the stage layout
  in `r1cs.rs`.
- `jolt-prover`: `ZkStageWitnesses`, `STAGE_DOMAINS`, `ClaimRecorder`, the ZK
  fields of `ProvedUniskipMode`. Commitments in `Stage*ZkOutput` and
  `jolt-verifier-derive`'s ZK drivers become masked records.

Regenerated: ZK fixtures, the golden-shape test, the FS inventory, and the
committed-recorder profiling spans (a taxonomy change).

## Security sketch

**Soundness.** Pedersen binding fixes p before the first challenge, so every
decoded m = m̂ − p precedes the challenge after it (§4.1). BlindFold's
extractor gives W satisfying the R1CS whose pad segment opens `PADS`, so it
equals p. The curve analogue of Lemma `wo:lem:residual-correct` (products are
native gates: no constant-term lift, no scalarity rows) decodes an accepting
clear transcript with the same challenges, prefix-preserving. The y row and
Dory's ZK opening bind the final evaluation; clear Jolt soundness finishes.

**Zero knowledge** (Theorem `wo:thm:zk`, fewer hybrids, no retries):
perfectly hiding Pedersen makes `PADS` and the folded commitments
independent of p; one-time uniform pads make m̂ uniform; Dory's opening and
the NovaBlindFold tail are simulated by their own simulators.

## Cost (muldiv, T = 2¹⁰, estimated from today's golden shape)

Today's per-stage ZK messages are 243 G1 (227 rounds + 16 claim rows), about
7.8 KB. Masking sends ~600 coefficients and ≤512 claims (~35 KB of field
elements, about the clear proof's message bytes) plus ~35 pad rows (1.1 KB):
**ZK proofs grow by about 25–30 KB**, scaling with rounds (log T). Monomial
lowering of τ·Az·Bz gives ~n² products (n ≈ 30–40): today's 37 auxiliary rows
(~1.1k values) would become ~35 KB of slots; factored lowering cuts that to a
handful (A2). The tail shrinks: the witness drops from 280 mostly padding
rows to ~35–70 dense rows, with far fewer constraints and error rows. Prover
commitment work stays ~P scalar multiplications; masking adds P additions.
M6c records exact numbers from the regenerated fixture.

## Ambiguity register

- **A1 Auxiliary timing.** The paper sends a product "before any challenge
  that uses it"; here all go in one block after stage 8. No base challenge
  uses them and product rows determine them from earlier slots, so decoding
  stays prefix-preserving, and the prover can compute them from its replay.
- **A2 Factored lowering.** `Expr` is monomial. Proposed: a `jolt-claims`
  sum-of-products-of-affine-factors form, starting from the existing
  `output_factor_expressions`; an affine x factor gets an auxiliary slot and a
  defining row (§3.2). Fallback, auxiliary values as BlindFold-only witness,
  forks the lowering per back end (I7).
- **A3 Interface fixes.** `CommittedPadsView` is parameterized by the pad
  commitment, not `Blind`; `verify` takes the view and the residual
  commitment (Dory's hiding evaluation commitment). M6c updates interfaces §5.
- **A4** `ZkError::FoldOverflow` has no producer before W2; add it there.
- **A5** Extension row expansion and `PadLayout::operand_pads` have no M6c
  caller (`E = F`, no scalarity rows). Defer to W3/W4, or derive
  `operand_pads` from `products[].x`. Needs an owner ruling against M6b's
  acceptance criteria.
- **A6** Uni-skip output claims are derivable, but clear bytes send them, so
  masked mode does too (two slots, two rows).
- **A7** L depends on the VC capacity (32 or 64 by feature).
