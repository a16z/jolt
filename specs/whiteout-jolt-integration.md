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

The argument also needs the residual to be bound. The cut row equates Akita's masked witness evaluation with the residual witness w_k, committed in u_a. A cheater who can open u_a to a second w_k moves the cut evaluation and can balance Akita's root equation against shifted claims, whatever t_p binds. So u_a's binding of w_k is a premise of §8 alongside A_p's: the residual's `A_a` and `[B_a | B′_a]` must be Module-SIS hard at the Σ collision radius (Akita A41, resolved by W5e: `ZkOpeningSchedule::admit` refuses a residual below its Σ-binding rank, `akita:specs/whiteout-akita.md` §5.5.2), and so must `A_p` (Akita A40, resolved in W6e by `jolt_akita::whiteout::sigma_matrix_rank`). Both are priced by Akita's Σ-binding table at jolt-zk's `binding_collision_radius()`, one radius for both.

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
  - HT3: a forced `FoldOverflow` reuses nothing from the aborted attempt. *As built:* a `prover-fixtures` fault seam (`jolt_prover::akita::whiteout_faults`) aborts the first attempt after its masked levels, not `ZkOpeningSchedule::for_tests`; it forces the abort rather than producing a real overflow.
  - HT4: one consistent-cheat case per site class, including stage-8 claims. *As built:* the statement class is a live attack (`fs_attacks::whiteout`); the other classes rest on Σ special soundness (`specs/whiteout-sigma.md`), with byte tampers per class (`soundness::whiteout`) and the compensating-shift probe for the stage-8 claims. No prover-side cheat construction.
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
    Recorded in `specs/whiteout-v6-costs.md`. On the measured shapes (T = 2^12–2^20) a proof grows by 28.8–31.0 KB, of which raw u_a is 19.5 KB (D29). Σ is 160 B, as predicted. Steady prover overhead is +1.2–2.5 s, and the verifier takes 185–250 ms against the paper's +0.8 ms. At V6 a fresh process paid 12–16 s of one-time planning to prove, and 12–14 s to verify, because the Whiteout catalog and response plan were `serde(skip)`; O1 persists them (D32).
  - W6f (N5): a Whiteout verifier plans its setup's whole catalog, every admitted combination of precommitted groups (up to four trace rows plus one dense row per group shape), on first use, and builds an Akita prover setup twice (once to derive the verifier setup, once lazily for the prover). Recorded, not optimized; shipping the rows as artifacts or deriving the verifier setup without a prover setup removes both. *Planning resolved by O1* (D32): the catalog is planned at setup and persisted with the verifier setup (`WhiteoutPlans`); exported plans, response schedules included, enter a new setup only at construction (`AkitaSetupParams::with_whiteout_plans`, `preprocess_full_from_plans`), and are re-audited and re-admitted on load. The double prover setup remains (about 0.02 s).

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

- **Q1, resolved by Akita W2b** (`akita:specs/whiteout-akita.md` §3.5). Jolt's per-proof auxiliary groups, untrusted advice and `FieldInc` (`J8:crates/jolt-prover/src/akita/stage8.rs:1-4`), enter Akita as precommitted groups. Akita admission now requires every precommitted group at a masked level to be hiding, so W6 commits them (and trusted advice or a committed program, when a ZK row consumes them) with `GroupContext::explicit` on a hiding profile from a ZK row of the group's shape (R6), instead of rejecting the proof. Implemented in W6f (prototype deltas, *Precommitted groups*; `specs/whiteout-paper-deltas.md` D31).
- **Q2 (Quang).** Do Jolt's stage messages move to the NARG? If yes, `pad_layout`'s zero-filled replay applies unchanged. If no, it needs a zero-valued, shape-correct `JoltProof`, built from `validate_proof_consistency`'s expected shapes so that shape keeps one owner.
- **Q3 (Quang).** Does he accept `ShapeTranscript` as a third sealed implementation?
- **Q4.** `ZkBackend`'s `ProverTranscript<H>` signatures (`whiteout/dev:crates/jolt-zk/src/masking/backend.rs:113`) must port to `ProverState`. This is an M6b re-landing item, not W6.
- **A44 (Akita, follow-up from W6f), resolved.** A trusted party ships (advice or program, C, r) to the prover, which rebuilds its hint, and an auditor holding (data, r) checks C (D31, *Trust root*; prototype deltas, *Precommitted groups*). Akita: `CommitmentHandle::hiding_randomness`, `HidingRandomness`, `CpuBackend::recommit_hiding` (`akita:specs/whiteout-akita.md` A44).
- **Q5 (paper authors).** Is the bounded rerun delta acceptable as argued?

## Prototype deltas (W6 on `whiteout/dev`)

The prototype runs on `whiteout/dev` at `9589dd045` (Akita dev `9dd27e997`), which has neither J7/J8 nor W5d. This section maps each J-stack citation above to its equivalent here and records each decision. It is the port map back to the J-stack. Paths are relative to the jolt root unless prefixed `akita:`.

| Spec item | J-stack | Here | Decision |
|---|---|---|---|
| §1 mode enum | `J8:…/config.rs:27` | `ZkConfig` at `crates/jolt-verifier/src/config.rs:29`, serialized in `JoltProof.protocol` | Same: `Whiteout` appended. |
| §1 statement word | `JoltStatement.zk` (`J8:verifier.rs:319-325`) | byte 0 of `JoltProtocolConfig::transcript_bytes`, absorbed by `absorb_public_preamble` (`verifier.rs:748`) | `Whiteout` → 2 in that byte. |
| §1 selection | `SELECTED_ZK_CONFIG` (`J8:config.rs:76-79`) | same name, `config.rs:111-115` | `match` on (`zk`, `akita`). |
| §1 compile errors | `J8:config.rs:20-24`, `J8:jolt-prover/src/lib.rs:36-37` | `config.rs:21-26`, `crates/jolt-prover/src/lib.rs:36-37` | Lifted by the first W6d commit (masked stages under Akita). The tail is wired, and production setups prove Whiteout since W6e prices `A_p` (A40, below) after W5e (A41). |
| §1 typed mode | `CheckedInputs.zk: bool`, `for_zk(bool)` | same (`verifier.rs:729`, `config.rs:55`) | Same change; stage code asks `ZkConfig::is_masked()`. |
| §1 Akita config | `ZkCommitmentConfig<RecursiveCommitmentConfig<OneHot>>` | Akita dev has no `ZkCommitmentConfig` (W5d); jolt-akita's presets are `JoltOneHotK16`/`K256`… (`crates/jolt-akita/src/configs.rs`) | W6b wraps jolt-akita's own recursive presets in W5d's `ZkCommitmentConfig` once merged. |
| §2 transport | spongefish `ProverState`/`VerifierState`, NARG | jolt-transcript `ProverTranscript`/`VerifierTranscript`; the proof *is* the argument string (`JoltProof { protocol, narg }`, `proof.rs:35`); Akita and `ZkBackend` already write to the same type | Rows 0–9 all go to one argument string. Q4 does not arise: `ZkBackend` already takes `ProverTranscript<H>`. |
| §2 rows 0, 1 | `J8:verifier.rs:213-293` | `seed_transcript` (`PREAMBLE`, `COMMITMENTS` sites) | Unchanged. |
| §2 row 2 | `PADS` | `PADS` site in `verify_stages_on` (`verifier.rs:566`), BlindFold only | Same position, Whiteout's pad commitment. |
| §2 row 9 | `check_eof` | `VerifierTranscript::finish` (`verifier.rs:116, 667`) | Unchanged. |
| §3 opening trait | `GroupBatchOpening` (`J8:jolt-openings/src/schemes.rs:118`) | `CommitmentScheme::{prove_batch, verify_batch}` (`crates/jolt-openings/src/schemes.rs:131-155`) | `MaskedGroupOpening` in `jolt_zk::whiteout` as specified, but not a subtrait of a group-opening trait (none exists): it names its own setup and commitment types, and the verifier bounds `PCS: CommitmentScheme + MaskedGroupOpening<…>`. The verifier is generic over `PCS`, so a trait is needed on that side. |
| §3 hiding commit | `HidingCommitment`/`ZkOpening` split (W6b) | `ZkOpeningScheme::commit_zk` (Dory, `schemes.rs:339`); the Akita prover is generic over `PCS` but already bounds jolt-akita's own `TraceOneHotCommitment` | **No split.** Hiding commits are prover-only, and Akita's stage 0 already bounds jolt-akita traits; W6b adds the hiding entry point there. Dory and `ZkOpeningScheme` are untouched (R2 by construction). |
| §3 probe | `J8:jolt-prover/src/blindfold.rs:128-141` | `MaskedStages.checkpoint` (`verifier.rs:180`), compared in `crates/jolt-prover/src/blindfold.rs:116-120` | Same, taken after Akita's cut, before the auxiliary block. |
| §4 `pad_layout` | Akita-excluded at `whiteout/dev:verifier.rs:389` | same function, `verifier.rs:390` | Extended: stages 1–7, Jolt's stage-8 reductions, then Akita's `count_masked_prefix` (`akita:crates/akita-verifier/src/fold/masked_prefix.rs:90`, `CountingGrinding`), then the shapes. |
| §4 `ShapeTranscript` | needed because J8's stages grind (`J8:jolt-transcript/src/verifier.rs:91-100`) | Jolt's stages never grind here (no `check_grind` in jolt-verifier); only Akita grinds, and `CountingGrinding` skips it | **Not built.** Q3 does not arise. The zero-filled replay stays. |
| §4 Q2 | stage messages in NARG? | They already are. | Zero-filled replay applies unchanged. |
| §5 proof type | `JoltProof<PCS, VC, ()>`, `JoltProofClaims::Zk`, `backend_proof` | `JoltProof { protocol, narg }`; no claims or backend field | V5's "non-empty `backend_proof`" case is covered by "appended NARG byte" (`finish`). |
| §6 rerun owner | `jolt-prover` ZK entry | `jolt_prover::akita::prove`, the Whiteout build's (`crates/jolt-prover/src/akita/whiteout.rs`) | Loop there: `WHITEOUT_ATTEMPTS = 3`, retrying on `ZkError::is_rerunnable` (fold overflow, `AttemptsExhausted`); `KernelContext::restart` keeps the witness and drops the session. |
| §6 RNG | one `CryptoRng` stream | BlindFold draws from `OsRng` (`blindfold.rs:72`); Akita's backend draws hiding randomness from its own OS stream (`akita:crates/akita-cpu-backend/src/opaque/backend.rs:134`; `with_hiding_rng` is test-support) | Both OS streams. Each draw is fresh, so "no reseed, no counter" holds. HT3 compares two runs' values, not seeds. A seeded Akita hiding RNG per commit would make the test reproducible; that is an Akita request (W6b), not a blocker. |
| §7 census | `absorb-sites.inventory` | `crates/jolt-verifier/tests/fs_inventory/absorb-sites.inventory`, checked by `fs_obligations` | Landed: `zk-site-classes` sits beside it (never blessed; an eighth class, `clear-only`, holds absorbs only a clear or non-Whiteout protocol runs). The runtime census is `crates/jolt-verifier/tests/soundness/whiteout.rs`: it classifies each message and absorb of a Whiteout muldiv verify by (region, site), with Akita's rows by `akita:specs/whiteout-akita.md` §4.2; its positive control and mutation check run on the event log. |
| §8 binding | `bind_derived_base` | same (Akita M3b fixup is on dev) | Plus one item from our base: stage 8's prefix-packing reduction absorbs the claim evaluations in the clear build. Under Whiteout it absorbs only the layout, digest and point (`PrefixPackedLayout::select` with `PackedEvaluations::Masked`, `crates/jolt-openings/src/prefix.rs`) and carries the evaluations as affine forms; the evaluations are derived from masked messages already absorbed (D25's argument, D31). |

**Prerequisites our base lacks** (not in the J-stack spec, because there M6b/M6c/W4 are re-landed first):

- The masked stage verifiers and `JoltResidual` are Dory-only (`#[cfg(not(feature = "akita"))]`, `crates/jolt-verifier/src/stages/zk/mod.rs:14`). Akita's relation set (fused increments, the Akita-only stage-6b/7 relations) has no residual lowering. W6d must lower them; this is the largest jolt-side item.
- Stage 8 on Akita takes clear stage outputs (`stages/stage8/akita.rs:218`). W6d writes its leaf and packing claims once over affine forms.
- jolt-zk is not in the root `[patch."https://github.com/a16z/jolt"]`, so Akita's `zk` features would resolve a second jolt-zk. W6b adds it.

**Masked stages under Akita (first W6d commit).**

- *Owner of the Akita lowering.* The Akita build changes stage 5 (canonical instruction addresses), 6a/6b (the lattice read-RAF and booleanity relations, no increment reduction, no advice cycle phases) and 7 (the digit-zero reduction). Under `akita`, `JoltResidual` lowers stages 5–7 from `MaskedMember`s the masked stage verifiers record (`ConcreteSumcheck::masked_member`, generated `masked_members`). Each member's formulas are its symbolic relation's, and every public value comes from the member's own `derive_input_term`/`derive_output_term` and drawn challenges, the clear verifier's owners. The bytecode read-RAF cycle relation gained `derive_output_term` (its publics, formerly inline in `expected_output`, are one shared helper), and the cycle wrapper dispatches `masked_member` per variant like `expected_output`. Whiteout builds the bytecode table fold in masked mode for these publics; BlindFold does not (its residual derives its own).
- *Inherited debt.* The homomorphic residual computes stages 5–7's publics by hand, beside the verifier's owners, and both builds lower stages 1–4 by hand (the relations are the same in both builds). Migrate them to `MaskedMember` publics at port time; R2 kept them untouched here.
- *Operand rule.* Whiteout lowers with `OperandRule::MaskedSlot` (`verifier.rs`, `WHITEOUT_OPERANDS`).
- *Pads.* `ModePads` under Akita is `CommittedPads<F, WhiteoutPadOpening<F>>`, built by `TraceOneHotCommitment::whiteout_commit_pads` (jolt-zk `commit_lattice_pads`). The `prover-fixtures` seam `prove_masked_stages` and `CommittedPads::uncommitted`'s jolt caller are gone.
- *Committed programs.* The committed-program reductions' publics (6b `FinalScale`/`OutputWeight` by `derive_output_term`; stage 7's address phases from `Stage7References`, which Whiteout fills from the masked stage 4/6b carriers) take the same owner path.
- *Test.* Superseded by `crates/jolt-prover/tests/whiteout_e2e.rs` (below).

**Whiteout tail (W6b–W6d).**

- *Seam.* `jolt_zk::whiteout::MaskedGroupOpening<F>` is the verifier's seam (`shape`, `receive_pads`, `verify_masked`, `count_masked`, `prefix_shape`, `append_prefix_rows`, `verify_tail`); `AkitaScheme` implements it (`crates/jolt-akita/src/whiteout/opening.rs`). The prover's half is three methods on jolt-akita's `TraceOneHotCommitment` (`whiteout_commit_pads`, `whiteout_prove_masked`, `whiteout_prove_tail`), since the Akita prover already bounds that trait. jolt-verifier's `whiteout` module is the verifier (`whiteout_stages`, `whiteout_prefix`, `whiteout_relation`, `verify`, `pad_layout`); the prover replays exactly those functions.
- *Aux block order: Akita's slots, then Jolt's.* §2 row 6 and §4 put Jolt's first. Akita's `MaskedPrefix` lowers its rows from the start of the layout's auxiliary range and has no first-slot offset, while `JoltResidual::rows_from` takes one, so Akita's part leads and no Akita change is needed. Both roles derive the split from the same `prefix_shape` count.
- *Two replays in the prover.* Stage 8's forms exist only in the verifier's residual, so the prover replays stages 1–7 once after stage 7 (forms, Jolt's auxiliary values), and once more over the whole string after the auxiliary block (Akita's prefix, the 32-byte checkpoint, the relation, and `remaining() == 0`), plus `pad_layout`'s counting replay. `VerifierTranscript` cannot resume on appended bytes. The J-stack has the same need.
- *Stage 8 absorbs no evaluation (§8, gap 2).* Every group's claims go to Akita as affine forms (`masked_group_claims`, `crates/jolt-verifier/src/stages/stage8/akita.rs`), built by the clear `leaf_claims`' leaf geometry (`leaves`) and the clear packing geometry (`PrefixPackedObjectPlan::packed_leaves`). One generic routine (`whiteout_groups`) runs over the residual's forms in the verifier and over their decodings in the prover (`clear_group_claims`), so both roles draw the same packing selectors on their own transcripts.
- *Precommitted groups (W6f, D31).* `whiteout_opening_plan` derives, from the statement, the trace group and each precommitted object in the clear stage 8's canonical order: untrusted advice, trusted advice (their word arity from the memory layout, `CommitmentMatrixShape::advice_from_max_bytes`), then `FieldInc` or the two committed-program objects. Each object's commitment metadata is checked against its plan first. The plan fixes the masked opening's shape (`MaskedOpeningShape`: the precommitted groups' arities and coefficient ranges, then the trace group), which `MaskedGroupOpening::shape` takes. jolt-akita plans (or loads from the setup's persisted `WhiteoutPlans`, D32), per setup and on first use in both roles, a `WhiteoutCatalog`: one zero-knowledge trace row per list of precommitted groups the setup admits (`GroupedScheduleParams::whiteout_admitted`: each combination of the optional advice groups, then the mandatory ones), keyed on the precommitted groups' hiding profiles. Each profile is the root profile of the unchunked zero-knowledge row of its group's shape (`ZkCommitmentConfig<JoltDenseBounded>` or `<JoltDenseFull>`), so it carries the masked root's B randomness (Akita §3.5, R6). The catalog's one backend commits every hiding group (`TraceOneHotCommitment::whiteout_commit_precommitted`; `commit_full_width_object` routes `FieldInc` there) and proves the opening: Akita cannot import a hiding commitment into another backend. Its setup's matrix covers the widest group (a committed program's bytecode can be wider than the trace). Trusted advice and committed programs are committed hiding at preprocessing and the prover keeps the randomness; D31 gives the argument that a commitment made before the pads, and reused across proofs, stays hiding. *Trust root (D31, A44).* The trust root of a trusted-advice or committed-program commitment C is the holder of its hiding randomness r. The committing process exports (C, r) (`AdviceObject::hiding_opening`, `DirectProgramObjects::hiding_openings`, as a `HidingOpening` holding Akita's `HidingRandomness`, whose wire form is `to_bytes`/`from_bytes`). A prover in another process rebuilds its hint with `reopen_trusted_advice` or `preprocess_committed_reopened`, through `TraceOneHotCommitment::whiteout_recommit_precommitted` and Akita's `CpuBackend::recommit_hiding`; the rebuild fails closed unless (data, r) reproduce exactly C, so the same calls are the auditor's check of C. Without r, C is not recomputable from the advice or program, and the verifier accepts it on the strength of whoever holds r. r is as secret as the data (D31).
- *ZK rows planned at setup (Akita A41, D32).* A fresh setup runs `find_zk_schedule` for the exact `(num_vars, num_polys)` at construction (`WhiteoutSetup::new`) and persists the rows with the setup; a setup built from persisted plans loads and validates them instead. No row shape is hard-coded; W5e re-priced the ZK families (A41), so the planned rows admit the residual at the Σ radius.
- *Σ ranks (Akita A40, W6e).* `A_p` is priced at the Σ collision radius by `jolt_akita::whiteout::sigma_matrix_rank`: Akita's `sigma_binding_min_rank` at jolt-zk's `binding_collision_radius()`, the radius the residual is admitted at. `A_D` commits range-checked radix-16 digits and needs Module-SIS only at the digit-difference bound (`05_security.tex:31-33`), so `digit_matrix_rank` prices it as Akita prices a digit commitment, through Akita's stable `akita_schedules::digit_commitment_min_rank` (A43: the planner's `projected_collision_role_price` for the D-role key at the Σ digits' base, then `min_secure_rank`). The production gate and the test-only ranks are gone. `crates/jolt-akita/src/whiteout/tests.rs` pins frozen table values: muldiv's `A_p` (d 64, width 898) at rank 18, in the Σ-binding bracket 179 < width ≤ 2463 (rank 17 at 179 and below), and `A_D` (Open, d 64, ℓ∞ 15) at rank 1 for width 4736, within the rank-1 width 93,935. The rank and the randomness columns of `A_p` are iterated to a fixed point, as Akita's planner prices a hiding commitment. The iteration only climbs (both the rank and `r_p`'s columns grow with the width), so it cannot oscillate; `whiteout_pad_rank_steps_at_the_sigma_binding_brackets` pins it at the brackets: at d 64 `r_p` takes 578 columns at rank 18, so no pad commitment lands below rank 18 (the empty layout climbs from rank 16 to 18 at width 610), 115 operand pads stay at rank 18 (width 2450), and 116 would reach 2466 > 2463 at rank 18, so the rank steps to 19 (610 columns, width 2498). The residual's admission at its Σ-binding rank is Akita's test (`admission_prices_the_residual_at_the_sigma_radius`).
- *Scope.* One single-chunk K=16 trace group beside the precommitted groups its setup admits (`opening_shape`); other shapes reject with `UnsupportedSchedule`.
- *Prover sampler in verifier builds (W6g, resolved).* jolt-akita's `zk` is the verifier half and enables jolt-zk's `whiteout` only; `zk-prover` (implies `zk`, adds `jolt-zk/whiteout-prover` and `rand_core`) gates the prover half: `TraceOneHotCommitment::{whiteout_commit_pads, whiteout_prove_masked, whiteout_prove_tail}`, `WhiteoutTail::{commit_pads, prove}`, `WhiteoutRow::prove_masked`, `ProverGroup`, `WhiteoutPads` (its opening type, jolt-zk's `WhiteoutPadOpening`, is prover-only), `WhiteoutResidual`, and `whiteout::tests`. jolt-prover's `zk` selects `jolt-akita?/zk-prover`; jolt-verifier has no normal dependency on jolt-akita, and its `akita`/`zk` features reach jolt-akita only through the fixture prover (a dev-dependency). Check: `cargo tree -p jolt-verifier --features akita,zk -e features,normal | grep -E "whiteout-prover|jolt-dit|sample"` is empty (with default edges the dev-dependency fixture prover legitimately pulls the sampler). Akita split the same way (A46): akita-pcs's and akita-cpu-backend's `zk` compile no sampler, and their `zk-prover` adds `jolt-zk/whiteout-prover`, jolt-crypto's `sample` and the OS-drawn hiding randomness; jolt-akita's `zk-prover` forwards `akita-pcs/zk-prover`. So `cargo tree -p jolt-akita --features zk -e features,normal | grep -E "whiteout-prover|jolt-dit|sample"` is empty too (that split also exposed the last items above, which akita-pcs's unconditional `whiteout-prover` had let the verifier half compile), and I6's "the verifier never compiles the prover sampler" holds for an Akita verifier. CI lints both halves (`cargo clippy -p jolt-akita --features zk / zk-prover`) and fails if either check above matches (`Whiteout verifier graph is sampler-free`).
- *Unmasked passthrough (Akita A42, resolved).* jolt-transcript's `Channel` has an explicit public send, `exchange_public`/`exchange_public_all`, which Akita uses for the messages public in both modes (the D image, the next-witness payload and u_a, Stage 3's claim and evaluation, `PublicGrindingSumcheck`'s rounds). jolt-zk's `MaskedProver` and `MaskedVerifier` refuse `exchange`/`exchange_all` with `TranscriptError::ClearMessageOnMaskedChannel`, so a private value sent clear on a masked channel is a typed error, not a clear message (`jolt-zk`'s `masked_channels_refuse_clear_messages`). Clear transcripts send the same bytes through either method. HT1's census still classifies every unmasked message of a Whiteout proof.
- *Test.* `crates/jolt-prover/tests/whiteout_e2e.rs` (lane `-p jolt-prover --features akita,zk,prover-fixtures --test whiteout_e2e`): an honest muldiv proof verifies; bit flips across the string, truncation and an appended byte reject; a second proof differs (HT3). Since W6f, an advice proof (untrusted and trusted) and a committed-program proof verify, reject byte tampers, and reject a trusted-advice commitment of other advice, a recommitted trusted advice, a missing one, and verifier preprocessing holding a recommitted program; both also verify against verifier preprocessing after a serde round trip, which loads the persisted catalog plan (and plans the response schedule, which preprocessing without exported plans does not carry). Since O1, `whiteout_plans_persist_and_are_consumed` checks that serialized preprocessing is a function of the setup, that a prover and verifier built from exported plans load them, and that plans of another format are refused at the prover's first use. With `field-inline` (lane `-p jolt-prover --features akita,zk,field-inline,prover-fixtures --test whiteout_e2e`), a field-ops proof whose increments leave the `u64` envelope, beside untrusted advice, verifies and rejects byte tampers. jolt-verifier's `whiteout_muldiv_case` fixture (and, since W6f, `whiteout_advice_case` and `whiteout_committed_case`, whose census pins Stage 8's absorbs to each packed object's layout words, digest and point and the commitments region's public absorbs to two per statement-fixed commitment, and whose per-class tampers cover the new regions) (lanes `-p jolt-verifier --features akita,zk,prover-fixtures,logging` and, with FieldInc in every case and no committed program, `akita,zk,field-inline,prover-fixtures,logging -E 'test(census)|binary(fs_attacks)'`) carries the census, the message sweep, per-class tampers, statement and mode tampers, the HT4 statement attack (`fs_attacks`) and the shift probe. *Compensating shifts* (jolt-verifier `soundness::whiteout::shifts`): on muldiv's 56 final-opening claims, Jolt's rows see one combination; Akita's rows apply exactly one functional, whose coefficients are the EvaluationBatch challenges; and a shift in that functional's kernel passes every row under the drawn challenges. On the advice case (W6f) the advice groups' claims join the same single functional, each through its packing weight; Jolt's rows see two combinations there, since the advice openings also feed stage 4's RAM value check. The binding is Schwartz–Zippel over challenges drawn after t_p and the masked values (D25), plus u_a's binding of w_k (§8). jolt-akita's `whiteout::tests` (lane `-p jolt-akita --features zk-prover -E test(whiteout)`) opens a muldiv-shaped group alone. jolt never enables Akita's disk-persistence cache, so every run is a no-disk-cache run.

**Slicing on this base.** W6a is jolt-only and lands now. W6b–W6d need W5d (`ZkCommitmentConfig`, `IS_ZK`, the published ZK entry points, the ZK family artifacts, `ResidualOpening.public_segments`/`blocks`).
