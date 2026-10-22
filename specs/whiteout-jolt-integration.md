# Spec: Whiteout ZK on Jolt + Akita (W6, RP5)

| Field   | Value |
| ------- | ----- |
| Author  | @markosg04 |
| Created | 2026-10-21 |
| Status  | draft |
| Parent  | `specs/shared-primitive-layer.md` (W6, RP5); Akita `specs/whiteout-akita.md` §7 |
| Base    | Quang's J-stack: J7 `d4e949c43` (#2110), J8 `ed90d31d7` (#2111), Akita #254 `e9d85d1f7` |

Citations to the J-stack are `J8:path:line` at `ed90d31d7`. Citations to Akita are `A254:` at `e9d85d1f7`.

## Decision

The `zk` feature means "zero knowledge", and the commitment axis picks the back end:

- `zk` without `akita` keeps BlindFold over committed pads (M6c).
- `zk` with `akita` selects Whiteout.

`ZkConfig` gains a third variant, `Whiteout`, appended at the end. The pad commitment t_p is absorbed after the commitments and before the first challenge. Stage 8 calls a new masked group opening in `jolt-zk`. It takes Jolt's masked claims as affine forms, borrows Jolt's `CommittedPads`, and stops at Akita's cut. Jolt then sends both auxiliary blocks and runs the Σ-protocol and Akita's response proof, all in the same spongefish argument string. A fold overflow or exhausted Σ attempts trigger a bounded rerun of the whole proof with fresh randomness, inside `jolt-prover`.

## Requirements ledger

| ID | Item | Kind |
|---|---|---|
| R1 | `akita`+`zk` compiles. Today both crates reject it (`J8:crates/jolt-verifier/src/config.rs:20-24`, `J8:crates/jolt-prover/src/lib.rs:36-37`). | requirement |
| R2 | Clear Dory, clear Akita and Dory BlindFold proofs stay byte-identical. | invariant |
| R3 | Every witness-dependent field message goes through `PrivateChannel` (I7). | invariant |
| R4 | One pad schedule, counted by the masked verifier itself. | invariant |
| R5 | A relation's expressions have one owner, in jolt-claims. Stage-8 affine claims are derived from those expressions, not restated. | invariant |
| R6 | Every per-proof commitment is hiding. | requirement |
| R7 | A clear verifier rejects a ZK proof, and the reverse, both at `JoltStatement` and at Akita's descriptor bit (S9). | requirement |
| A1 | RP4 delivers Akita §7's list unchanged. | assumption |
| A2 | M6b, M6c and W4 are re-landed on the J-stack before W6. | assumption |

## Mechanism

### 1. Modes and features

- `ZkConfig { Transparent, BlindFold, Whiteout }` (`J8:…/config.rs:27`). The type is serialized, so variants are append-only. `SELECTED_ZK_CONFIG` (`:76-79`) becomes a `match` on (`zk`, `akita`). `JoltStatement`'s `zk` word (`J8:crates/jolt-verifier/src/verifier.rs:319-325`) encodes `Whiteout` as 2.
- `CheckedInputs.zk: bool` (`verifier.rs:447`) and `for_zk(bool)` (`config.rs:52`) become `ZkConfig`. A bool cannot name the back end.
- Whiteout over Dory, and BlindFold over Akita, are not built. BlindFold needs a curve field, and Whiteout's pad commitment needs the Akita ring.
- The Akita config is `ZkCommitmentConfig<RecursiveCommitmentConfig<OneHot>>`.

### 2. Transcript order (both roles)

| # | Message | Owner | Absorbed as |
|---|---|---|---|
| 0 | `JOLT_SESSION`, `JoltStatement` (`zk = 2`) | jolt-verifier | instance |
| 1 | Akita commitments made by `commit_hiding` (metadata and rows, J8 stage 0) | jolt-verifier | public |
| 2 | label `PADS`, then t_p (raw; compressed after W5c) | `ZkBackend::commit_pads` | NARG |
| 3 | stages 1–7, with m̂ in place of every private value | stage verifiers | masked |
| 4 | stage-8 reduction challenges; claims become `MaskedOpeningClaims` | jolt stage 8 | — |
| 5 | Akita descriptor (zk bit), digests, points; evaluations through `bind_derived_base`; levels 0..=`cut_level` masked; u_a | `MaskedGroupOpening` | mixed |
| 6 | auxiliary block: Jolt's slots, then Akita's | `CommittedPads::send` | masked |
| 7 | ρ, σ, t_D, n | `Whiteout::prove` | NARG |
| 8 | response proof | `AkitaResponseProver` | NARG |
| 9 | `check_eof` | verifier | — |

Rows 0, 1, 3 and 4 are J8's flow (`verifier.rs:213-293`). Rows 2 and 5–8 are Akita §7 items 2 and 5–8. Akita's item 1, "preamble, hiding commitment", is J8's rows 0 and 1. Akita's item 4 is row 4. No challenge separates rows 1 and 2, so the paper's step 1 holds (`04_protocol.tex:514-516`). Row 9 replaces J8's `PCS::check_eof` after stage 8 (`verifier.rs:293`), because the argument string now continues to row 8.

### 3. Masked group opening and pad ownership

`GroupBatchOpening` (`J8:crates/jolt-openings/src/schemes.rs:118`) takes clear `GroupOpeningClaim`s, and Akita absorbs their evaluations publicly (`A254:crates/akita-verifier/src/fold/root.rs:77-90`). jolt-openings does not depend on jolt-zk. The masked variant therefore lives in `jolt_zk::whiteout`:

```rust
pub struct MaskedGroupClaim<F, C> { pub commitment: C, pub point: Vec<F>, pub evaluations: Vec<AffineRow<F>> }

pub trait MaskedGroupOpening: GroupBatchOpening + HidingCommitment {
    type ResidualOpening; // RP4's `ResidualOpening`, W4's `Witness`
    type MaskedPrefix: ResidualRows<Self::Field>;
    fn prove_batch_masked<O>(setup: &Self::ProverSetup,
        groups: Vec<(MaskedGroupClaim<Self::Field, Self::Output>, Self::OpeningHint)>,
        pads: &mut CommittedPads<Self::Field, O>, transcript: &mut ProverState<Self::Sponge>,
    ) -> Result<(Self::ResidualOpening, Vec<Self::Field>), ZkError>; // aux values, slot order
    fn verify_batch_masked<V: PrivateChannel>(setup: &Self::VerifierSetup,
        groups: &[MaskedGroupClaim<Self::Field, Self::Output>], pads: &mut V,
        transcript: &mut VerifierState<'_, Self::Sponge>,
    ) -> Result<(Self::MaskedPrefix, Self::PendingEof), ZkError>;
}
```

- **Pads.** jolt-prover owns `CommittedPads` from `commit_pads` until `ZkBackend::prove` consumes it. Akita borrows it for levels 0..=`cut_level` only. The cursor is shared, so Akita's slots follow Jolt's base slots.
- **Verifier.** The verifier passes its masked channel, or the counting view (§4), as `V`.
- **Adapter.** jolt-akita's impl forwards to RP4's ZK opening prover and masked verifier. Under the shared spec's no-logic-in-the-adapter rule, it is a thin forwarder. Its body runs `shape_guard` (`J8:crates/jolt-akita/src/native_batching.rs:375`) against the ZK family's catalog rows.
- **Prover replay.** The prover replays the shell through the masked verifier to build `ResidualRelation`, as J7's `replay_stages` does. It keeps J7's 32-byte probe (`J8:crates/jolt-prover/src/blindfold.rs:128-141`), now taken at the cut, before row 6.

### 4. Pad layout (`CountingGrinding`)

`pad_layout` runs the masked verifier over zeroed messages with `CommittedPadsView::counting()`. It is Akita-excluded today (`whiteout/dev:crates/jolt-verifier/src/verifier.rs:389`). On the J-stack, zeroed messages fail Jolt's own grind check (`J8:crates/jolt-transcript/src/verifier.rs:91-100`), not only Akita's. W6 therefore adds `ShapeTranscript` to jolt-transcript:

- It is a sealed `FieldTranscript` whose `check_grind` returns `Ok` and whose challenges come from a throwaway sponge.
- It does not implement `VerifierInit`, so `verify` cannot be instantiated with it.

Akita's `CountingGrinding` wraps it. A single replay counts, in order:

1. Jolt's base slots;
2. Akita's prefix slots;
3. Jolt's auxiliary slots, then Akita's.

It also produces the operand list.

### 5. Wire type, Σ and response

- **Jolt stage messages.** They keep the J-stack's placement, with m̂ substituted.
- **Whiteout messages.** Rows 2 and 5–8 are NARG prover messages. The `ZkBackend` and `ResponseProver` APIs write to the transcript, and J8 already puts Akita in the argument string.
- **Proof type.** `JoltProof<PCS, VC, ()>` with `JoltProofClaims::Zk` (`J8:crates/jolt-verifier/src/proof.rs:247`). The Whiteout tail has no serde payload. A non-empty `backend_proof` is rejected, as in J8.

### 6. Rerun semantics

`jolt-prover`'s ZK entry treats exactly two errors as retryable: `ZkError::FoldOverflow` (mapped from `AkitaError::FoldOverflow`) and Σ's `AttemptsExhausted` (`specs/whiteout-sigma.md` §2.6).

- **What a rerun repeats.** It restarts at stage 0, so `commit_hiding` produces new commitments. It draws new pads, g, r_p and Σ coins from the same `CryptoRng` stream, with no reseed and no counter.
- **What it reuses.** Only the trace and the witness.
- **Bound.** At most 3 attempts, then the error is returned.
- **Disposal.** An aborted attempt's transcript and messages are dropped and never serialized.

### 7. Masked-site census (HT1 across Jolt and Akita)

A checked-in file, `zk-site-classes`, maps every entry of `absorb-sites.inventory` (J8 FS inventory) to exactly one class:

- statement;
- commitment;
- `PADS`;
- masked slot;
- challenge-derived public;
- Σ;
- response.

The test does two things:

- It fails when an inventory entry has no class, so a new clear absorb cannot land unreviewed.
- It runs a ZK muldiv verify under `ScriptedTranscript` and checks that every public absorb is either a masked-channel event or a site in a non-masked class.

The positive control routes one Jolt claim and one Akita stage-1 value clear, and the test must flag both.

### 8. Statement binding under ZK

#254 states that Akita's soundness "does not depend on what the caller absorbed", because the root absorbs the claimed values. Under ZK the values go through `bind_derived_base`, which absorbs nothing. The argument is replaced by this one:

- Each claim is an affine form y = Σ aᵢ(m̂ᵢ − pᵢ) + b.
- Its m̂ᵢ were absorbed in rows 3–5 before any challenge that depends on y.
- The coefficients aᵢ and b are functions of the statement, the commitments and earlier challenges.
- p is fixed by t_p, absorbed in row 2 before the first challenge.

So a second claim vector that verifies under the same challenges must either change an absorbed message, which changes the challenges, or open t_p to a different p. The second breaks A_p's binding, which the Σ-extractor relies on (Lemma `wo:lem:residual-correct`). The commitment digests, points and descriptor stay absorbed, and the zk bit binds the mode on both sides (R7).

The caller contract of #254 gains one item: in ZK mode, Akita's root may run only after t_p and every masked slot its claims reference. `MaskedGroupClaim` enforces this by type, because the claims are rows over pad slots.

## Paper deltas

- **Bounded rerun instead of abort.** The paper aborts on overflow (`04_protocol.tex:100-104`; `05_security.tex:70`). §6 reruns at most twice. Argument: a rerun is a fresh, independent execution, and only the accepted run is output. The output distribution therefore equals the real prover's conditioned on no abort, which is within 3·δ_fold of the H1 prover's distribution (`05_security.tex:344-350`). Soundness is unaffected, because reruns only discard executions (`05_security.tex:111`).

## Tradeoffs

- **A separate `whiteout` feature.** It would allow invalid pairs (Whiteout with Dory, BlindFold with Akita) that need compile errors. Rejected; one axis picks the back end.
- **Masked claims as a mode flag on `GroupBatchOpening`.** This would give clear and masked claims one entry point. It would make jolt-openings depend on jolt-zk and let a clear value reach a ZK opening. Rejected, by R3.
- **The caller reruns.** The Akita note says "the caller reruns". A caller could reuse an RNG seed, so the loop stays inside `jolt-prover`.

## Verification

- **V1, end to end.** `cargo nextest run -p jolt-prover --features akita,zk,prover-fixtures`: a ZK muldiv proof verifies. Two RNG seeds share no pad, m̂, commitment row, Σ nonce or t_p (HT3).
- **V2, clear byte identity.**
  - Clear Dory, clear Akita and Dory BlindFold verifier fixtures pass unchanged.
  - `protocol_wire_format_is_explicit` (`config.rs:120`) keeps its bytes and adds a `Whiteout` case.
- **V3, fixtures.**
  - A new `whiteout_muldiv` verifier fixture, cached by name.
  - Re-blessed `absorb-sites`/`challenge-sites` inventories.
- **V4, HT tests.**
  - HT1: the census in §7.
  - Coverage: every pad has a nonzero coefficient.
  - Counting equality: `pad_layout` equals the honest cursor, on zero bytes and on random bytes.
  - MaskedSlot lift: build the muldiv relation under `OperandRule::MaskedSlot` and lift it through `PadPlacement`/`LiftedRelation`. Under BlindFold's `Affine` rule the shape records no operand pads, so a `pad_layout` that skipped `JoltResidual::check_stages` would go undetected; only this lift catches a pair operand missing from `operand_pads`.
  - HT3: a forced `FoldOverflow` via `ZkOpeningSchedule::for_tests` reuses nothing from the aborted attempt.
  - HT4: one consistent-cheat case per site class, including stage-8 claims.
- **V5, tamper suite.** Each case rejects:
  - one m̂ per stage;
  - t_p;
  - one hiding-commitment row;
  - an Akita masked-level value;
  - an auxiliary value from each block;
  - t_D and n;
  - a response-proof byte;
  - an appended NARG byte;
  - `zk` word 1 against a Whiteout build, and the reverse;
  - a flipped Akita descriptor zk bit;
  - a non-empty `backend_proof`.
- **V6, builds and cost.**
  - Clippy passes for `host,akita,zk`.
  - The `akita,zk` CI lane runs.
  - Proof bytes and prover time are recorded against the paper's model.

Unverified: whether the ZK planner finds a schedule at muldiv's T (Akita A26). If it does not, V1 uses `for_tests`.

## PR slicing and dependencies

| PR | Content | Depends on |
|---|---|---|
| W6a | `ZkConfig::Whiteout`, typed `CheckedInputs.zk`, statement word; Whiteout not yet selectable | J7, J8 merged |
| W6b | `HidingCommitment`/`ZkOpening` split; Dory implements both, jolt-akita `HidingCommitment`; shape guard admits ZK rows | W6a, RP4 (W5d), Akita `main`→`dev` sync containing #253/#254 |
| W6c | `ShapeTranscript`; `MaskedGroupOpening` and jolt-akita impl; `FoldOverflow` mapping | W6b, Quang's sign-off on a new sealed impl |
| W6d | Whiteout path in prover and verifier, rerun loop, compile-error removal, V1–V6 | W6c, M6b/M6c/W4 on the J-stack |

Jolt pins Akita `dev` at W6b.

## Ambiguity register and open questions

- **Q1, resolved by Akita W2b** (`akita:specs/whiteout-akita.md` §3.5). Jolt's per-proof auxiliary groups, untrusted advice and `FieldInc` (`J8:crates/jolt-prover/src/akita/stage8.rs:1-4`), enter Akita as precommitted groups. Akita admission now requires every precommitted group at a masked level to be hiding, so W6 commits them (and trusted advice or a committed program, when a ZK row consumes them) with `GroupContext::explicit` on a hiding profile from a ZK row of the group's shape (R6), instead of rejecting the proof.
- **Q2 (Quang).** Do Jolt's stage messages move to the NARG? If yes, `pad_layout`'s zero-filled replay applies unchanged. If no, it needs a zero-valued, shape-correct `JoltProof`, built from `validate_proof_consistency`'s expected shapes so that shape keeps one owner.
- **Q3 (Quang).** Does he accept `ShapeTranscript` as a third sealed implementation?
- **Q4.** `ZkBackend`'s `ProverTranscript<H>` signatures (`whiteout/dev:crates/jolt-zk/src/masking/backend.rs:113`) must port to `ProverState`. This is an M6b re-landing item, not W6.
- **Q5 (paper authors).** Is the bounded rerun delta acceptable as argued?
