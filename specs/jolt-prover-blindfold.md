# Spec: BlindFold ZK in the Modular Prover

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | Andrew Tretyakov               |
| Created     | 2026-07-22                     |
| Status      | proposed                       |
| PR          |                                |

## Summary

The modular prover (`crates/jolt-prover` + `crates/jolt-kernels`) proves only transparent
proofs today, while the branch already carries every other piece of the ZK story: the
verifier's compile-time BlindFold path (`crates/jolt-verifier/src/stages/zk/`), a complete
generic BlindFold prover with **zero callers** (`crates/jolt-blindfold/src/prove.rs`), a
fully implemented `CommittedSumcheckRecorder` behind the recorder seam the stage drivers
were designed around, committed uni-skip proving (`prove_uniskip_committed`), and hiding
PCS opening machinery (`ZkOpeningScheme` / `ZkBatchOpeningScheme`, implemented by Dory
unconditionally). Only the legacy prover can currently produce a ZK proof, using its own
parallel BlindFold implementation. This PR wires the existing pieces together: a
compile-time `zk` feature on `jolt-prover` that swaps the recorder, retains the committed
witnesses, and closes the proof with a BlindFold tail — producing `JoltProofClaims::Zk`
proofs that the existing `jolt-verifier` ZK path accepts unmodified. It is the ZK-wiring
follow-up that `specs/prover-stage-drivers.md` §Non-Goals deferred ("PR C is a recorder
swap plus front work").

## Intent

### Goal

Produce BlindFold ZK proofs from the modular prover behind a compile-time `zk` cargo
feature — recorder-typed stage proving, prover-side BlindFold protocol construction that is
the *verifier's own code path*, and witness assembly from recorder-retained secrets — with
zero clear-mode changes and zero runtime mode flags in prover code.

Key abstractions introduced or modified:

- **`zk` feature on `jolt-prover`** (`zk = ["jolt-verifier/zk", "jolt-kernels/zk",
  "dep:jolt-blindfold", "dep:rand_core"]`): the compile-time mode axis, mirroring the
  verifier's `SELECTED_ZK_CONFIG`. One compiled prover produces exactly one protocol; the
  proof self-describes it via `JoltProtocolConfig::for_zk(cfg!(feature = "zk"))`, and the
  co-compiled verifier's `validate_proof_config` rejects the other kind fail-closed. No
  `zk_mode` field, no runtime branch: mode lives in the recorder type (stages), in
  `#[cfg]`-gated retention fields (stage outputs), and in `#[cfg]`-alternate arms at the
  three protocol-mandated divergence points (uni-skip, stage-0 commit finish, stage-8
  opening).
- **Mode-selected recorder** (`jolt-prover/src/recorder.rs`, new): one constructor
  `stage_recorder(...)` whose *return type* is `#[cfg]`-selected —
  `CommittedSumcheckRecorder<'_, F, VC, OsRng>` under `zk`,
  `ClearSumcheckRecorder<F, VC::Output>` otherwise. Both have `Commitment = VC::Output`,
  so `Proved<F, S, VC::Output>` and every downstream type are mode-independent. The eight
  stage call sites replace their hardcoded `ClearSumcheckRecorder::new()` with this one
  call and change nothing else — the generated `StageProver::prove` drivers are already
  recorder-generic and stay untouched.
- **Committed-witness retention** (stage outputs): each `StageNProverOutput` gains a
  `#[cfg(feature = "zk")] pub committed_witness: CommittedSumcheckWitness<F>` field
  (stages 1 and 2 additionally a `uniskip_witness`), populated from
  `RecordedSumcheck::committed_witness` / `ProvedUniskipCommitted::witness` — the round
  coefficients, output-claim rows, and their Pedersen blinds that BlindFold later opens.
  `Stage8ProverOutput` gains `#[cfg(feature = "zk")] pub joint_evaluation: F` and
  `pub evaluation_blind: F` (the value and blind inside Dory's `y_com`). In clear builds
  these fields do not exist; in ZK builds a recorder returning no witness is an
  `InvariantViolation`.
- **`ZkBatchOpening` return** (`jolt-openings`): `ZkBatchOpeningScheme::prove_batch_zk`
  additionally returns the joint evaluation it already computes
  (`Σ γⁱ · evalᵢ`) — the scalar the BlindFold witness must place in the dedicated
  final-opening evaluation row. Return shape becomes a named struct
  `ZkBatchOpening { proof, hiding_commitment, blind, joint_evaluation }` (replacing the
  3-tuple; the legacy analog is `OpeningProofData { joint_claim, y_blinding }`).
- **Prover-side protocol construction by replay** (`jolt-prover/src/blindfold.rs`, new,
  `#[cfg(feature = "zk")]`): after stage 8 the prover assembles a **shell proof**
  `JoltProof<PCS, VC, ()>` — every wire field real, `claims: JoltProofClaims::Zk
  { blindfold_proof: () }` — and replays it through the verifier's **existing public
  verification surface**: `validate_and_seed_transcript`, `build_formula_dimensions`,
  `stage1::verify` … `stage8::verify`, then the verifier's own
  `stages::zk::blindfold::build(BlindFoldInputs { .. })` — the exact call sequence
  `verify` runs before its BlindFold tail, and the same surface the verifier's own
  zk-audit test harness already consumes externally. This yields the `BlindFoldProtocol`
  and a transcript positioned exactly where the verifier's will be, and doubles as a
  free self-check that the just-produced proof shell verifies. Every stage `verify` is
  already generic over `ZkProof`, which is what makes the replay type-check with the
  unit placeholder. The spine spelled out prover-side is a deliberate structural cost:
  jolt-verifier gains **zero code** for the prover's benefit (its "prover-free"
  invariant extends to prover-serving promotions), and any spine change surfaces as a
  compile- or replay-failure here, never as silent drift — the protocol the prover
  proves against is still literally the verifier's own lowering. A `debug_assert`
  compares the replay transcript state against the prover's forward transcript at the
  stage-8 boundary.
- **Witness assembly** (`jolt-blindfold`, new prover-side API):
  `BlindFoldProtocol::assign_witness(&self, stage_domains: &[SumcheckDomainSpec],
  stage_witnesses: &[&CommittedSumcheckWitness<F>], eval_outputs: &[F], eval_blindings:
  &[F], rng) -> Result<AssignedBlindFoldWitness<F>, ProverError<F>>` produces the full
  row matrix and per-row blinds that `jolt_blindfold::prove` consumes — from the
  protocol's **public parts alone** (layout, constraint matrices, consistency,
  dimensions), so the verifier's lowering needs to expose nothing beyond the
  `BlindFoldProtocol` its `build` already returns:
  1. Assign layout variables from prover data: per stage, round coefficients from the
     retained witness; the claim chain *derived*, not supplied — the input claim as the
     domain round-sum of round 0 (`SumcheckDomainSpec::round_sum_coefficients` over the
     caller-supplied stage domain, a protocol constant the prover's own sumchecks ran
     over), each `claim_out` as the round polynomial evaluated at the consistency
     challenge; output-claim row values; final-opening evaluation/blinding scalars.
  2. Solve the remaining private values — the product auxiliaries of the claim-expression
     lowering — directly from the constraint matrices: `R1csBuilder::multiply` emits the
     canonical `A · B = 1·v_fresh` shape with `v` allocated after its operands, so one
     forward pass over the constraints in emission order assigns each product from its
     already-known operands. Equality constraints (round sums, claim bindings, final
     openings) have a zero C side and are skipped — they are checks, not definitions.
     Unassigned slots after the pass are the layout's zero padding.
  3. Slice per `protocol.dimensions` into coefficient rows / output-claim rows /
     auxiliary rows / zero padding, exactly as `WitnessRowLayout` prescribes; blinds =
     retained round blinds ∥ retained output-claim blinds ∥ fresh random per auxiliary
     row ∥ zero for padding.
  4. Shape validation throughout: per-stage round counts and degrees against the
     protocol's consistency (`degree + 1 == coefficients.len()`), row counts against
     the layout; `debug_assert` full R1CS satisfaction (`check_witness`).
- **Stage-8 ZK arm** (`jolt-prover/src/stages/stage8.rs`): the clear arm is unchanged;
  the `#[cfg(feature = "zk")]` arm builds the same batch entries and opening point, then
  calls `HomomorphicBatch::<PCS>::prove_batch_zk` — which squeezes the gamma powers
  *without* absorbing claim scalars, opens the joint polynomial in hiding mode, and
  absorbs `ZkEvaluationClaim` — mirroring the verifier's stage-8 ZK arm label-for-label.
- **Uni-skip arms** (stages 1–2 fronts): `prove_uniskip_clear` ↔
  `prove_uniskip_committed` selected by `#[cfg]`. The committed variant already exists,
  retains its `CommittedSumcheckWitness`, and keeps the output claim prover-internal
  (never absorbed); the stage forwards the claim downstream exactly as today.
- **ZK commit finish** (`jolt-kernels`, new `zk` feature): the reference witness-commit
  kernel switches `StreamingCommitment::finish_with_hint` /
  `finish_one_hot_column_major_chunks` to the `ZkStreamingCommitment` `finish_zk_*`
  variants, so every witness commitment (including untrusted advice) is hiding, as in
  legacy (`maybe_blind_commitment`). Kernels stay transcript-free; nothing else in the
  crate is mode-dependent.
- **BlindFold tail** (`jolt-prover/src/blindfold.rs` + `prover.rs` assembly): replay →
  `blindfold::build` → `protocol.assign_witness` (stage domains + witnesses in protocol
  stage order: stage-1 uni-skip, stage-1 batch, stage-2 uni-skip, stage-2 batch, stages
  3, 4, 5, 6a, 6b, 7; `eval_outputs = [joint_evaluation]`, `eval_blindings =
  [evaluation_blind]`) → `transcript.append(&Label(b"BlindFold"))` →
  `jolt_blindfold::prove` with `preprocessing.verifier.vc_setup` — the first real caller
  of the crate's prover — → final `JoltProof<PCS, VC>` reassembled from the shell with
  `claims: JoltProofClaims::Zk { blindfold_proof }`.
- **Randomness**: `rand_core::OsRng`, held internally (legacy precedent:
  `rand::thread_rng()` inside `prove`). `prove`'s public signature is identical in both
  modes; blinds are fresh per commitment. Where the where-clause needs ZK bounds
  (`PCS: ZkOpeningScheme<HidingCommitment = VC::Output, Blind = F>`,
  `VC::Output: Copy + HomomorphicCommitment<F>`, the `RingAccumulator` bound), they are
  added unconditionally, matching the verifier's `verify`, which already demands
  `ZkOpeningScheme` on the clear path.

What stays mode-independent (deliberately): every generated stage driver, `begin_batch`,
kernels' round compute, opening-point derivation, the value-carrying `clear_output`
carriers between stages (the prover always knows the values; in ZK they simply never reach
the transcript or the wire), and stage-4's transcript-silent advice evaluation staging.
The committed recorder makes `absorb_input_claims` a no-op and replaces the output-claim
flush with row commitments; that recorder substitution *is* the transcript-discipline
change, everywhere except the three arms listed above.

### Invariants

1. **Compile-time mode, prover side.** No runtime ZK flag exists in `jolt-prover` or
   `jolt-kernels`: no mode field, no `is_zk`/`zk_mode` boolean, no branch on proof
   content. The only appearance of the mode as a value is
   `JoltProtocolConfig::for_zk(cfg!(feature = "zk"))` at proof assembly and the same
   constant handed to the verifier's shared `validate_inputs_from_parts` — both
   compile-time constants. (The verifier's `CheckedInputs.zk` remains that crate's
   affair; it const-folds against `SELECTED_ZK_CONFIG`.)
2. **Clear-mode wire freeze.** Without the `zk` feature, proof bytes, transcript bytes,
   and public API are byte-identical to pre-PR state: the byte-diff harness passes 16/16
   with zero fixture regeneration. The `#[cfg(not(feature = "zk"))]` build of every
   touched stage differs from today only by the recorder-constructor call (returning the
   same `ClearSumcheckRecorder`) and by absent retention fields.
3. **ZK wire and Fiat-Shamir compatibility with the existing verifier.** The modular
   prover's ZK proofs verify against today's `jolt-verifier` ZK path *unmodified*: same
   `JoltProof` shape (`SumcheckProof::Committed` in all ten stage slots,
   `JoltProofClaims::Zk`), same transcript discipline — input claims never absorbed,
   `b"sumcheck_commitment"` per round, `b"output_claims_coms"` rows, uni-skip committed
   absorbs, stage-8 gamma squeeze without `rlc_claims`, `ZkEvaluationClaim`, then
   `Label(b"BlindFold")` and the BlindFold sub-transcript. This discipline is not
   re-implemented: it is inherited from `CommittedSumcheckRecorder`,
   `prove_uniskip_committed`, and `prove_batch_zk`, whose twin tests already pin it
   against the generated `verify_zk` drivers.
4. **Prover/verifier protocol identity by construction — with a prover-free verifier.**
   The `BlindFoldProtocol` (R1CS matrices, layout, dimensions, baked publics/challenges)
   the prover proves against is produced by executing the verifier's own public stage
   functions and `blindfold::build` on the assembled proof — the same code the verifier
   runs — never by a prover-side reconstruction. A claim-formula change that updates the
   verifier's BlindFold lowering is therefore picked up by the prover automatically;
   only a change to *witness content* (what the recorder retains) can desynchronize, and
   R1CS satisfaction fails loudly at proof time. Dually, `jolt-verifier` gains no code
   for any of this: the net diff to `crates/jolt-verifier` from this PR is **zero** —
   the replay and the witness assembly consume only its existing public surface.
5. **Commitment/witness binding.** Every Pedersen commitment on the wire (round rows,
   output-claim rows) opens with the retained witness handed to BlindFold — guaranteed by
   using the recorder's own retained `CommittedSumcheckWitness` verbatim, never
   recomputed values. `assign_witness` validates shapes against the statement and
   `debug_assert`s full R1CS satisfaction.
6. **Driver mode-agnosticism preserved.** No `prove_clear`/`prove_zk` split appears
   anywhere: stages differ only in recorder construction, retention, and the three
   protocol-mandated arms. `StageProver`, the consumer macros, and `jolt-verifier`'s
   prover-free invariant (prover-stage-drivers §Invariants 3) are untouched.

`jolt-eval` plan: no existing invariant entries change. A `zk_completeness_modular`
invariant (modular ZK proof accepted by the compiled-`zk` verifier) is the natural
follow-up via `/new-invariant` once the harness grows a `zk` matrix; the byte-diff
invariant (`legacy_proof_byte_equality`) is explicitly clear-mode-only and stays primary.

### Non-Goals

- **No legacy changes.** `jolt-prover-legacy`'s own BlindFold implementation stays as-is
  and stays the fixture source for the verifier's ZK test suites. Deleting it in favor of
  the modular ZK prover is a later PR, after the modular path has soaked.
- **No byte-parity with legacy ZK proofs.** ZK proofs contain fresh randomness (Pedersen
  blinds, Nova random instance, hiding commitments); byte-diff against legacy is neither
  possible nor meaningful. The correctness bar is acceptance by the existing verifier
  (plus the legacy ZK e2e staying green).
- **No protocol changes.** Transcript labels, absorb order, R1CS shape, and the
  `JoltProof` wire type are frozen at what the verifier already implements. Zero
  verifier changes of any kind.
- **No akita interaction.** `zk` and `akita` are mutually exclusive by the verifier's
  existing `compile_error!`; the prover inherits the exclusion through
  `jolt-verifier/zk` and adds nothing.
- **No new backends and no GPU work.** The `BlindFoldRowCommitter` acceleration seam
  stays at its CPU default (`DirectBlindFoldRowCommitter`); the reference commit kernel's
  ZK finish is the only kernels change.
- **No prover-side streaming or memory work for BlindFold.** The BlindFold witness is
  ~hundreds of 32-wide rows; assembly is negligible next to witness generation.

## Evaluation

### Acceptance Criteria

- [ ] Clear-mode byte-diff harness untouched: `cargo nextest run -p jolt-prover
      --cargo-quiet --features prover-fixtures` — all 16 tests pass with zero fixture
      regeneration and zero expectation edits.
- [ ] New ZK e2e (`crates/jolt-prover/tests/zk_e2e.rs`, `cfg(all(feature =
      "prover-fixtures", feature = "zk"))`): the modular prover proves `muldiv-guest` in
      ZK mode — plain and committed-program variants — and
      `jolt_verifier::verify::<Fr, DoryScheme, Pedersen<Bn254G1>, LegacyBlake2bTranscript>`
      accepts both; a tampered `blindfold_proof.random_u` is rejected. Runs under
      `cargo nextest run -p jolt-prover --cargo-quiet --features prover-fixtures,zk`.
- [ ] Legacy ZK e2e stays green: `cargo nextest run -p jolt-prover-legacy muldiv
      --cargo-quiet --features host,zk`.
- [ ] `cargo nextest run -p jolt-kernels --cargo-quiet` and `cargo nextest run -p
      jolt-verifier --cargo-quiet` (and `--features akita`) green; the verifier's zk
      fixture suites (completeness, tampering) stay green untouched.
- [ ] `cargo clippy --all --features host -q --all-targets -- -D warnings`, the same with
      `host,zk`, and `cargo fmt -q` clean. (The `host,zk` invocation compiles the new
      prover ZK path workspace-wide — it is the ZK build gate.)
- [ ] No runtime mode flag in prover code:
      `grep -rnE "zk_mode|zk: bool|is_zk" crates/jolt-prover/src crates/jolt-kernels/src`
      → no hits; every mode divergence is `#[cfg(feature = "zk")]`.
- [ ] `jolt_blindfold::prove` has a production caller (the tail), and
      `BlindFoldProtocolBuilder::build` / `blindfold::build` still exist with unchanged
      signatures (verification untouched).
- [ ] `git diff <base> -- crates/jolt-verifier` is empty: the verifier crate gains no
      code (and loses none) for the BlindFold prover work.
- [ ] jolt-sumcheck engine twins (clear, committed, uni-skip) pass unchanged; the
      `prove_type_checks_with_committed_recorder` compile witness is superseded by real
      wiring and removed or kept as-is, but not duplicated.

### Testing Strategy

- **Primary new gate:** the ZK e2e above — full pipeline (ZK commits, committed recorder
  through all eight stage drivers, committed uni-skips, hiding stage-8 opening, BlindFold
  tail) against the unmodified verifier. Committed-program variant covers the extra
  precommitted stages (6b/7 reductions) and their output-claim rows. Both run on a
  128 MiB-stack thread like the verifier's ZK suites.
- **Witness assembly units** (jolt-blindfold): `assign_witness` against the existing
  test-harness pipeline — same generated committed stages, prove with the real
  `jolt_blindfold::prove` on the assembled rows/blinds and verify with
  `BlindFoldProtocol::verify`; shape-mismatch rejection cases. The harness's independent
  reference prover is deliberately kept (it cross-validates `prove.rs`).
- **Replay identity:** the ZK e2e `debug_assert`s (via the tail) that the replay
  transcript equals the prover's forward transcript at the stage-8 boundary — any
  recorder/driver drift fails here before BlindFold math starts.
- **Existing suites as regression floor:** byte-diff 16/16 (clear), jolt-verifier zk
  completeness/tampering (legacy-generated fixtures), jolt-kernels, engine twins. Both
  clippy matrices per repo policy.

### Performance

Clear mode: zero cost — the ZK path does not exist in the binary. ZK mode adds, relative
to the clear modular prover: per-round Pedersen commits (MSM of ≤ 33 points per round,
thousands of rounds — noise next to witness commitment), one verifier replay (~one
`verify` execution, milliseconds), BlindFold witness assembly (one R1CS assignment pass,
~10⁴ variables), and the BlindFold prove itself (row commits + two small sumchecks +
Hyrax openings over a ~256-row grid) — the same asymptotics as the legacy ZK prover. No
`jolt-eval` objective moves; no new benchmarks. The `muldiv` ZK e2e wall-clock is the
smoke check that nothing is accidentally quadratic.

## Design

### Architecture

Mode is a type-level axis threaded through existing seams; no new crate and no new
inter-crate edges except `jolt-prover → jolt-blindfold` (optional, `zk`):

```
jolt-sumcheck     SumcheckRecorder: Clear ↔ Committed; prove_uniskip_{clear,committed}   [unchanged]
jolt-crypto       Pedersen VectorCommitment                                              [unchanged]
jolt-verifier     (consumed through its existing public surface only)                    [ZERO diff]
jolt-blindfold    + BlindFoldProtocol::assign_witness (matrix-driven solver)
                  prove.rs gains its first caller                                        [prover-side API]
jolt-openings     prove_batch_zk returns ZkBatchOpening (adds joint_evaluation)          [one struct]
jolt-kernels      + zk feature: reference commit finish_zk_*                             [transcript-free]
jolt-prover       + zk feature: recorder.rs (mode-typed constructor),
                  retention fields, uniskip/stage0/stage8 cfg arms,
                  blindfold.rs (shell → replay spine → build → assign → prove)           [this PR's core]
```

ZK-mode proof flow (clear flow identical up to the recorder type):

```
stage 0     commit_zk-finished witness commitments; validate(zk = cfg!)         [hiding commits]
stages 1–7  begin_batch → prove_batch → CommittedSumcheckRecorder               [rounds committed,
            uni-skip: prove_uniskip_committed                                    witnesses retained]
stage 8     prove_batch_zk → (Dory proof, y_com, blind, joint_eval)             [claims hidden]
shell       JoltProof<PCS, VC, ()> with claims = Zk { blindfold_proof: () }
replay      validate_and_seed_transcript + stage1..8::verify on the shell      [existing pub surface;
            (prover-side spine, the zk-audit pattern)                           self-check + FS position]
construct   blindfold::build(BlindFoldInputs { shell, stage outputs })          [verifier's own lowering]
assign      protocol.assign_witness(domains, stage witnesses, [joint_eval], [blind])
prove       Label(b"BlindFold"); jolt_blindfold::prove(vc_setup, …)
final       shell with claims = Zk { blindfold_proof }
```

The ten protocol stages and their witness sources, in the order `blindfold::build`
inserts them (domains supplied alongside: the two uni-skips over their centered integer
domains, every batch over the Boolean hypercube) (= `add_stage1 … add_stage7`, uni-skip before its remainder batch):

| # | BlindFold stage        | Witness source                                   |
|---|------------------------|--------------------------------------------------|
| 1 | stage-1 uni-skip       | `ProvedUniskipCommitted::witness` (stage 1)      |
| 2 | stage-1 outer remainder| `Stage1ProverOutput::committed_witness`          |
| 3 | stage-2 uni-skip       | `ProvedUniskipCommitted::witness` (stage 2)      |
| 4 | stage-2 batch          | `Stage2ProverOutput::committed_witness`          |
| 5–8 | stages 3, 4, 5, 6a   | respective `committed_witness`                   |
| 9 | stage-6b batch         | `Stage6bProverOutput::committed_witness`         |
| 10| stage-7 batch          | `Stage7ProverOutput::committed_witness`          |

Order bugs cannot survive: `assign_witness` checks every stage's round count and per-round
`degree + 1` against the statement's consistency, and any residual mismatch fails the
R1CS satisfaction debug-check and then BlindFold's own internal round-claim self-checks.

**Why replay instead of constructing stage ZK outputs inline.** The verifier's
`StageNZkOutput`s (consistency, batching coefficients, derived points, stage-8 joint
commitment) are byproducts of the generated `verify_zk` drivers. The prover *could*
assemble them from its own stage data — it holds every ingredient — but that is a
hand-maintained mirror of eight stages of verifier logic, exactly the drift class this
branch exists to kill. The replay costs one verifier execution (milliseconds against a
prover that spends seconds committing the witness), reuses only `pub` verifier API,
validates the proof shell for free, and makes invariant 4 structural. The zk-audit
harness already proved the pattern — the prover's `replay_stages` is its production
twin, kept prover-side so the verifier crate stays untouched.

**Why the spine is spelled out prover-side.** Promoting it into a shared
`jolt_verifier::verify_stages` (and exposing the lowering's statement via a
`build_construction`) was implemented first and rejected in review: the branch's
"prover-free jolt-verifier" invariant covers prover-serving additions too, and the
verifier must not grow API whose only consumer is the prover. The prover-side copy is
the same ~80 lines the zk-audit harness already maintains against the same public
surface, and it cannot drift silently — the stage functions' signatures and the
`BlindFoldInputs` shape are the compiler's problem, and the transcript debug-assert
plus the final verification are the runtime net.

**Why the shell uses `ZkProof = ()`.** The BlindFold proof does not exist until after the
replay, but `validate_proof_consistency` (correctly) requires the claims variant to match
the compiled mode before any stage runs. Every stage `verify` and
`validate_and_seed_transcript` is already generic over `ZkProof`, so
`JoltProofClaims::Zk { blindfold_proof: () }` type-checks, costs nothing, touches no
transcript byte (claims are never absorbed in ZK), and cannot be confused with a real
proof. The final proof is reassembled from the same moved parts with the real
`BlindFoldProof`.

**Witness assembly correctness argument.** Claim expressions lower to sums of products
over openings, with challenges and publics folded into linear-combination constants;
`R1csBuilder::multiply` is the only allocator of non-layout private variables, and every
product constraint has the `A · B = 1·v_fresh` shape with `v` allocated after its
operands. A forward pass over the constraint matrices in emission order therefore
assigns exactly the auxiliary witness — no statement, no sources, no solver state beyond
the partial witness. The claim chain is derived from the committed coefficients
themselves (domain round-sum for the input, Horner at the Fiat-Shamir challenge for each
link), so a prover whose sumcheck was honest produces a satisfying assignment by
construction, and one whose retained data drifted from the wire fails `check_witness` in
debug and the BlindFold sumcheck self-checks in release. The zero-fill of untouched
slots is sound because the only unconstrained variables are the layout's zero padding —
anything else left unassigned fails the satisfaction check.

### Alternatives Considered

- **Runtime mode flag on the prover** (a `zk: bool` in `ProverConfig`, recorder behind
  `dyn`/enum): rejected outright — violates the stated hard constraint, contradicts the
  verifier's compile-time `SELECTED_ZK_CONFIG` design ("one compiled verifier runs
  exactly one protocol"), and would put a dead cleartext path in ZK binaries. The
  recorder seam was built so the type system carries the mode.
- **Porting the legacy BlindFold prover** (`subprotocols/blindfold/` `StageConfig` /
  `BakedPublicInputs` / `VerifierR1CSBuilder`): rejected. It is a second, prover-owned
  encoding of the verifier R1CS — the exact dual-maintenance the shared
  `jolt-blindfold` crate exists to end — and its accumulator-coupled witness types don't
  fit the accumulator-free modular stack. The generic `prove.rs` is complete, tested by
  an independent reference prover, and shaped for exactly the inputs the recorder
  retains.
- **Per-stage inline `StageNZkOutput` construction instead of replay**: rejected above
  (§Architecture); summary — eight stages of mirrored verifier logic, no free
  self-check, drift by silence instead of drift by failure.
- **A `BlindFoldProtocol::from_committed_proofs(...)` prover constructor**
  (`specs/jolt-prover-model-crate.md` §BlindFold sketches this name): subsumed. The
  protocol is Jolt-shaped by the verifier's `blindfold::build`; a second Jolt-aware
  constructor inside the generic crate would duplicate the lowering.
- **`verify_stages` promotion + `blindfold::build_construction` in jolt-verifier**
  (implemented first, reworked on review): rejected — prover-serving verifier API
  violates the branch's prover-free invariant even when behavior-neutral. The spine
  lives prover-side against the existing public surface; the statement never needs
  exposing because the witness assembly reads the public protocol directly.
- **Statement-based witness assembly** (re-run `allocate_layout` + `append` with claim
  sources, letting `multiply` evaluate auxiliaries eagerly): rejected with the above —
  it needs the statement and baked sources, which only a verifier-side
  `build_construction` could surface. The matrix-driven product solver gets the same
  auxiliary witness from `BlindFoldProtocol.r1cs` alone, and is *less* machinery: no
  second builder pass, no source table, no layout re-allocation.
- **Witness assembly in `jolt-prover` instead of `jolt-blindfold`**: rejected — the
  assembly is generic over the protocol, needs layout/dimension slicing that shouldn't
  leak, and in `jolt-blindfold` it is reusable by the crate's own tests and any future
  non-Jolt user.
- **RNG as a `prove` parameter** (`#[cfg(feature = "zk")] rng: &mut R`): rejected —
  cfg-dependent public signatures poison every caller (SDK, tests) with the same cfg;
  legacy's internal-rng precedent works, and proof reproducibility is not a requirement
  (ZK proofs are randomized by definition; the statistical-independence suite *wants*
  fresh entropy).
- **Byte-diff-style ZK parity harness against legacy**: rejected as a goal — would
  require lock-stepping two RNG streams through structurally different provers for a
  property (byte equality) that ZK proofs intentionally do not have. Acceptance by the
  shared verifier is the meaningful equivalence.
- **Skipping the hiding witness commitments** (transparent Dory commits + BlindFold
  only): rejected — legacy blinds the tier-2 commitment in ZK mode
  (`maybe_blind_commitment`), the ZK finish paths exist precisely for this, and
  transparent commitments of secret-dependent polynomials leak. Parity with the audited
  legacy discipline wins.

## Documentation

Module docs on the new seams (`jolt-prover/src/blindfold.rs` header walking the tail
flow, `recorder.rs` on the mode axis, `assign_witness` on the assign-before-append
ordering), plus this spec. The book's prover chapters describe the legacy prover and gain
nothing yet; the modular prover's book treatment is deferred until it replaces legacy
(tracked by the clean-slate plan, not this PR). `CLAUDE.md`'s ZK table stays accurate —
it describes the legacy crate; the modular equivalents land in the respective crate docs.

## Execution

Commit-sized, gate-checked steps; each leaves the workspace green under both clippy
matrices:

1. **jolt-openings**: `ZkBatchOpening` struct; `prove_batch_zk` returns it (one impl,
   one trait touch; verifier side untouched).
2. **jolt-blindfold**: `BlindFoldProtocol::assign_witness` (matrix-driven product
   solver) + `AssignedBlindFoldWitness` with shape errors under `ProverError`;
   prove/verify roundtrip tests over builder-built protocols.
3. **jolt-verifier**: untouched. (Confirm with `git diff <base> -- crates/jolt-verifier`
   staying empty; run the full jolt-verifier suite including akita and the legacy-fixture
   zk suites as regression floor.)
4. **jolt-kernels**: `zk` feature; `finish_zk_*` switch in the reference commit kernel
   (bound handling per whatever the mock-PCS test surface allows — cfg the bound only if
   a transparent-only test PCS exists).
5. **jolt-prover, mode plumbing**: `zk` feature; `recorder.rs`; per-stage recorder call
   swap + `#[cfg]` retention fields; uni-skip arms; stage-0 `cfg!` zk flag; stage-8 ZK
   arm. Byte-diff after this step is the no-drift check for the clear build.
6. **jolt-prover, tail**: `blindfold.rs` (shell; the replay spine spelled against the
   verifier's existing public stage functions; `blindfold::build`; `assign_witness`;
   prove); `prover.rs` final assembly under cfg; `for_zk(cfg!)`.
7. **Tests**: `not(feature = "zk")` guard on the byte-diff modules (they compare against
   clear legacy proofs and cannot run in a zk-compiled workspace); `zk_e2e.rs` (plain +
   committed muldiv + tamper rejection, 128 MiB stack); full gate sweep, including the
   legacy `muldiv --features host,zk` target.

Risk watch-list for the implementer: the stage-6b curated output order and stage-7
precommitted address phases are already canonical in `curate_opening_values` — the
recorder commits exactly those values, so no ZK-specific ordering code should ever be
written; if the replay's `Stage8ZkOutput.constraint_coefficients` and the prover's
`prove_batch_zk` gamma stream ever disagree, the transcript debug-assert fires first —
look at absorb order around `ZkEvaluationClaim` before suspecting BlindFold.

## References

- `specs/prover-stage-drivers.md` — the recorder-generic driver design this PR completes
  ("PR C"); its invariants carry over.
- `specs/clean-slate-prover.md`, `specs/jolt-prover-model-crate.md` §BlindFold — the
  modular-prover plan and the (superseded) `from_committed_proofs` sketch.
- `crates/jolt-blindfold/src/prove.rs` — the generic BlindFold prover (currently
  caller-free); `tests/support/mod.rs::protocol_backed_witness` — the witness-assembly
  prototype.
- `crates/jolt-verifier/src/stages/zk/` — the lowering (`blindfold::build`) and the ZK
  stage drivers; `tests/support/zk_audit.rs` — the replay prototype.
- `crates/jolt-prover-legacy/src/zkvm/prover.rs::prove_blindfold` and
  `subprotocols/blindfold/` — the legacy reference for transcript discipline and witness
  content.
- BlindFold: Nova-style folding of the sumcheck-verifier R1CS (see module docs in
  `jolt-blindfold`); Hyrax row openings per Wahby et al.
