# Spec: Whiteout zero knowledge on Jolt over Akita

| Field   | Value |
| ------- | ----- |
| Author  | @markosg04 |
| Status  | implemented by this PR series (jolt-akita opening, then the Jolt prover and verifier) |
| Related | `specs/whiteout-masking.md` (masking front end), `specs/whiteout-sigma.md` (Σ-protocol), Akita `specs/whiteout-akita.md` (masked levels, residual, response proof) |

## Decision

The `zk` feature means zero knowledge, and the commitment axis picks the back end:

- `zk` without `akita` is BlindFold over committed pads (Dory).
- `zk` with `akita` is Whiteout.

`ZkConfig::Whiteout` is the third, appended variant. The lattice pad commitment t_p is absorbed after the commitments and before the first challenge. Stage 8 opens every committed group through a masked group opening in `jolt-zk` (`MaskedGroupOpening`). The opening takes Jolt's stage-8 claims as affine forms over the pads and stops at Akita's cut level. Jolt then sends both auxiliary blocks, runs the Σ-protocol and Akita's response proof, all in the one spongefish argument string. A fold overflow or exhausted Σ attempts rerun the whole proof with fresh randomness, at most three times.

Whiteout over Dory and BlindFold over Akita are not built: BlindFold needs a curve field, and Whiteout's pad commitment needs Akita's ring.

## Invariants

- Clear Dory, clear Akita and Dory BlindFold proofs are unchanged by this integration.
- Every witness-dependent field message goes through the private channel: masked as m̂ = m + p in ZK mode, clear otherwise.
- One pad schedule, counted by the masked verifier itself (§4).
- A relation's expressions have one owner, in jolt-claims. Stage 8's affine claims are derived from them.
- Every commitment the verifier reads in a Whiteout proof is hiding, including the per-proof precommitted groups and the statement-fixed trusted advice and committed program (§5).
- A clear verifier rejects a ZK proof, and the reverse, at `JoltStatement`'s zk word and at Akita's descriptor zk bit.

## 1. Modes and features

- `JoltStatement`'s first word is the zk config: 0 transparent, 1 BlindFold, 2 Whiteout. `SELECTED_ZK_CONFIG` is a `match` on (`zk`, `akita`).
- jolt-akita's `zk` is the verifier half: the zero-knowledge configurations and the masked opening's verification, with jolt-zk's `whiteout` only. Its `zk-prover` adds the hiding commitments, pads and response prover (jolt-zk's `whiteout-prover`, akita-pcs's `zk-prover`). jolt-prover's `zk` with `akita` selects `zk-prover`. jolt-verifier reaches jolt-akita only through its fixture prover, a dev-dependency.
- So a Whiteout verifier's normal-edge graph compiles no prover sampler (jolt-zk's `whiteout-prover`, `jolt-dit`, jolt-crypto's `sample`). `cargo tree -p jolt-verifier --features akita,zk -e features,normal` and the same for `jolt-akita --features zk` show it; the Whiteout CI lanes run both checks.

## 2. Transcript order (both roles)

| # | Message | Owner | Carrier |
|---|---|---|---|
| 0 | `JOLT_SESSION`, `JoltStatement` (zk word 2) | jolt-verifier | instance |
| 1 | hiding Akita commitments: the trace group and every precommitted group | jolt-verifier | public absorb |
| 2 | t_p | `WhiteoutTail::commit_pads` / `receive_pads` | NARG |
| 3 | stages 1–7, m̂ in place of every private value | stage verifiers | masked tape |
| 4 | stage-8 reduction challenges; the claims become affine forms | jolt stage 8 | — |
| 5 | Akita's descriptor (zk bit), digests and points; levels 0..=cut masked; u_a | `MaskedGroupOpening::verify_masked` | public / Akita's private proof channel |
| 6 | auxiliary block: Akita's slots, then Jolt's | `CommittedPads::send` | masked tape |
| 7 | ρ, σ, t_D, nonce | Σ-protocol | NARG |
| 8 | response proof | Akita's response prover | NARG |
| 9 | every pad and the whole tape used, then `check_eof` | verifier | — |

No challenge separates rows 1 and 2, so t_p is fixed before any challenge.

Masked stage messages travel on the proof's masked tape (`JoltProofClaims::Zk`), each absorbed with `append_private` at the clear message's position and encoding. Masked sumcheck rounds have a fixed width at the scheduled degree, with no count word. Derived claims bind nothing. Akita's masked levels exchange m̂ through Akita's private proof channel, and messages public in both modes through its public exchange. The verifier's masked state is `jolt_zk::masking::MaskedSession`; the prover's is `CommittedPads` and the tape.

## 3. Masked group opening and pads

jolt-openings does not depend on jolt-zk, so the masked seam lives in `jolt_zk::whiteout`: `MaskedGroupClaim` (commitment, point, one `AffineRow` per member), `MaskedOpeningShape` (the precommitted groups' arities and coefficient ranges, then the trace group), and the verifier trait `MaskedGroupOpening` (`shape`, `receive_pads`, `verify_masked`, `count_masked`, `prefix_shape`, `append_prefix_rows`, `verify_tail`). jolt-akita implements it for `AkitaScheme`.

- **Pads.** jolt-prover owns `CommittedPads` from `commit_pads` until the tail consumes it. Akita borrows it for its masked levels only. The cursor is shared, so Akita's slots follow Jolt's base slots.
- **Shape guard.** Each group's commitment metadata is checked against its planned row before any claim is formed.
- **Prover replay.** The prover replays its own proof shell through the masked verifier to build the residual relation, and checks that the replay's transcript equals its own before the tail.

## 4. Pad layout

`pad_layout` replays the masked verifier on a shape-correct shell, over zero messages (`MaskedReader::zeros`) and `ShapeTranscript`. `ShapeTranscript` prices every proof of work at zero and implements neither `VerifierInit` nor `ProverInit`, so it cannot drive `verify`. One replay counts, in order: Jolt's base slots, Akita's prefix slots (`count_masked`), the auxiliary slots (Akita's, then Jolt's), and the operand list.

Akita's replay reads its prover messages from a zero argument string whose length the shape does not fix. The replay starts with one buffer and doubles it while the replay runs out of bytes (`ZkError::TranscriptRejected`), up to 16 MiB. Any other error returns at once.

## 5. Precommitted groups

Jolt's per-proof auxiliary groups (untrusted advice, `FieldInc` under field-inline, `FusedInc` always) and its statement-fixed objects (trusted advice, the committed program's bytecode and program image) enter Akita as precommitted groups. Akita admits a precommitted group at a masked level only if it is hiding, so each is committed with a hiding profile: the root profile of the unchunked zero-knowledge row of the group's shape (`ZkCommitmentConfig<JoltDenseBounded>` or `<JoltDenseFull>`), which carries the masked root's B randomness (Akita `specs/whiteout-akita.md` §3.5).

- `whiteout_opening_plan` derives, from the statement, the trace group and each precommitted object in the clear stage 8's canonical order, and checks each object's commitment metadata against its plan.
- jolt-akita plans, per setup, a `WhiteoutCatalog`: one zero-knowledge trace row per list of precommitted groups the setup admits (each combination of the optional advice groups, then the mandatory ones). The plans are persisted with the verifier setup (`WhiteoutPlans`) and re-audited when loaded.
- The catalog's one backend commits every hiding group and proves the opening: Akita cannot import a hiding commitment into another backend. Its matrix covers the widest group.
- The verifier is prepared at load, as clear Akita's is. It derives one verifier matrix from Akita's setup seed, sized for the catalog rows, the residual matrices and every response row, and builds the residual matrices, the pad stream (A_p and its L row), the V stream (A_D) and every response verifier then. Response rows come from the persisted response set, else from the packaged response catalog (`schedules/jolt-fp128-response-digits.aks`, ν = 18..27). A response length without a row is rejected: the verify path never plans and never builds a prover setup or backend.
- The setup declares how many coefficients of the pad stream and of the V stream it supports. Each proof's geometry is computed from its shape and checked against that capacity before any matrix is read, by the prover (`commit_pads`) and the verifier (`receive_pads`); a shortfall is `WhiteoutCapacityError`. Preprocessing sets the default capacity by replaying `pad_layout` (§4) on shape-correct shells, with zero commitments, for every RAM size the setup admits, with and without untrusted advice. Capacity bounds resources and is not part of the transcript. A decoded setup checks its plans' version and bounds its capacity before it derives anything, and the prepared streams travel in `PreparedVerifier`'s trusted payload.
- Trusted advice and committed programs are committed hiding at preprocessing, and the prover keeps the randomness. A commitment made before the pads and reused across proofs stays hiding: its randomness is never revealed, and each proof's masked opening adds fresh masks.

**Trust root.** The trust root of a trusted-advice or committed-program commitment C is the holder of its hiding randomness r. Without r, C cannot be recomputed from the data, so the verifier accepts C on the strength of whoever holds r, and r is as secret as the data.

## 6. Rerun semantics

The ZK prover treats exactly two errors as rerunnable: a fold overflow and exhausted Σ attempts (`specs/whiteout-sigma.md`). Both are independent of the witness.

- A rerun restarts at stage 0, so the commitments are new. It draws new pads, masks and Σ coins from the same RNG stream.
- It reuses only the trace and the witness.
- At most three attempts; then the error is returned.
- An aborted attempt's transcript and messages are dropped and never serialized.

## 7. Masked-site census

`crates/jolt-verifier/tests/fs_inventory/zk-site-classes` maps every absorb site of the Fiat-Shamir inventory to one class: statement, commitment, pads, masked slot, challenge-derived public, Σ, response, or clear-only. The census test fails on an unclassified site, so a new clear absorb cannot land unreviewed. It then verifies a Whiteout proof under `ScriptedTranscript` and checks that every public absorb is a masked-channel event or a site of a non-masked class. A positive control routes one Jolt claim and one Akita stage-1 value clear, and the test must flag both.

## 8. Statement binding under zero knowledge

In clear mode Akita's root absorbs the claimed evaluations. Under zero knowledge the values are bound with `bind_derived_base`, which absorbs nothing, so the clear argument does not apply. It is replaced by this one:

- Each claim is an affine form y = Σ aᵢ(m̂ᵢ − pᵢ) + b.
- Its m̂ᵢ were absorbed (rows 3–5) before any challenge that depends on y.
- The coefficients aᵢ and b are functions of the statement, the commitments and earlier challenges.
- p is fixed by t_p, absorbed in row 2 before the first challenge.

So a second claim vector that verifies under the same challenges must change an absorbed message, which changes the challenges, or open t_p to a different p, which breaks `A_p`'s binding (the Σ extractor relies on it). The commitment digests, points and descriptor stay absorbed, and the zk word and descriptor bit bind the mode on both sides.

The residual must be bound too. The cut row equates Akita's masked witness evaluation with the residual witness w_k, committed in u_a. A prover that could open u_a to a second w_k could move the cut evaluation and balance Akita's root equation against shifted claims, whatever t_p binds. So u_a's binding is a premise alongside `A_p`'s: the residual's `A_a` and `[B_a | B′_a]` must be Module-SIS hard at the Σ collision radius (Akita's `ZkOpeningSchedule::admit` refuses a residual below its Σ-binding rank), and so must `A_p` (`jolt_akita::whiteout::sigma_matrix_rank`). Both are priced by Akita's Σ-binding table at jolt-zk's `binding_collision_radius()`. `A_D` commits range-checked radix-16 digits and needs Module-SIS only at the digit-difference bound, so `digit_matrix_rank` prices it as Akita prices a digit commitment.

The caller contract of Akita's opening gains one item: in zero-knowledge mode, Akita's root may run only after t_p and every masked slot its claims reference. The type does not enforce it: `MaskedGroupClaim`'s fields are public. The stage-8 call order does. `jolt_verifier::whiteout::verify` receives t_p before stage 1, runs the masked stages, and only then forms the claims (`whiteout_stages`) and hands them to `verify_masked` (`whiteout_prefix`); the prover follows the same order.

On muldiv's final-opening claims, Jolt's rows see one combination and Akita's rows apply exactly one functional, whose coefficients are the evaluation-batch challenges. A shift in that functional's kernel passes every row under the drawn challenges; it is excluded by Schwartz–Zippel over challenges drawn after t_p and the masked values, plus u_a's binding of w_k. The verifier suite's shift probe exercises this.

## Verification

- `jolt-prover --features akita,zk,prover-fixtures --test whiteout_e2e`: honest muldiv, advice (untrusted and trusted) and committed-program proofs verify; bit flips, truncation and an appended byte reject; a second proof of the same statement differs; a substituted, missing or recommitted trusted-advice commitment rejects; serialized verifier preprocessing verifies after a round trip.
- `jolt-verifier --features akita,zk,prover-fixtures,logging`: the Whiteout verifier fixtures carry the census, the message sweep, per-class tampers, statement and mode tampers, the statement attack in `fs_attacks`, and the shift probe.
- `jolt-akita --features zk-prover -E 'test(whiteout)'`: a muldiv-shaped group opened alone; frozen Σ-binding ranks at the bracket edges.

## Not verified

- The Σ-protocol's constant-time claim (`specs/whiteout-sigma.md`).
- x86_64 runs.
