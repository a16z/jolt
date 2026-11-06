# Spec: Upstreaming Whiteout (U4)

| Field | Value |
| --- | --- |
| Author(s) | @markosg04 |
| Created | 2026-10-29 |
| Status | draft for discussion; no PR opens before the user signs off §6 |
| Parent | `specs/shared-primitive-layer.md` (RP0–RP5, tickets M*/W*) |
| Siblings | `specs/whiteout-jolt-integration.md` (W6), `specs/whiteout-paper-deltas.md` (D1–D35), Akita `specs/whiteout-akita.md` (RP4, S1–S13, A24–A47, Q1–Q8) |

**Pins.** Sizes were measured at jolt `whiteout/dev` @ `c84edca00` and Akita `whiteout/dev` @ `f0de039e2`. Since then U1–U5 have landed: jolt `4b9b54494`, Akita `9657c074e`. Both branches now contain `origin/main` as of 2026-10-29: jolt `50919db1c`, Akita `038db2d13`. That merge (U3) shifts the sizes by a few percent. Upstream state was read with `gh` on 2026-10-29.

## Decision

Whiteout goes upstream as **18 jolt PRs and 14 Akita PRs**. They land in the review-point order of `shared-primitive-layer.md`: RP1 (jolt), then RP2 plus seams (Akita `main`), then RP3 (jolt), then RP4 (Akita `dev`), then RP5 (jolt). Three changes to that plan follow from upstream's current state:

1. **The transcript base moved.** Both integration branches are built on jolt #2000 and akita #185. Both PRs **closed unmerged** on 2026-10-21; they were not merged, contrary to the ticket brief. Quang's spongefish-0.8 stack superseded them:
   - jolt: J7 #2110, J6 #2108, J8 #2111, J4 #2113;
   - Akita: #253 → #255 → #260 → #254.

   Before any extraction, a convergence ticket **T0** must port both integration branches onto that stack's heads. Every jolt PR from J5 on is rewritten against `ProverState`/`VerifierState`.
2. **RP3 does not wait on RP2.** M1d deletes the `jolt-field` shim and is the only RP3 ticket that needs Akita's RP2. It moves to a trailing cleanup PR (J18). The masking, BlindFold and Σ work (J9–J14) then follows RP1 directly.
3. **The Akita seams go to `main` early and are carried in `dev` at once.** On `main`, ~7.2k source lines of Whiteout edits sit in files that already exist (measured below). The maintainer gate on `main` is the slowest step, so the three seam PRs open together with RP2. Their code is carried into `dev` under `docs/branches.md` "Carry a seam while its `main` pull request is in review".

ZK over Akita stays unselectable until the last PR in each repo:
- Akita: A13 publishes `ZkCommitmentConfig` and the ZK family.
- jolt: J16 deletes the `zk`+`akita` `compile_error!`.

## 1. Prerequisite stacks

| Stack | Upstream state (2026-10-29) | Blocks | What blocks it |
| --- | --- | --- | --- |
| Transcript, jolt: J2 #2097, J3 #2099 (+J5 #2106) | merged | — | — |
| Transcript, jolt: J7 #2110 (base `main`, +8.2k/−7.2k) | open, review required | J5 onward (every PR that touches sumcheck/transcript) | review; J6 #2108 stacked on it has CHANGES_REQUESTED |
| Transcript, jolt: J8 #2111 (Akita opening inside Jolt's sponge), J4 #2113 (Poseidon) | open, on J7 | J15 (Akita inside Jolt's argument string) | J7 |
| Transcript, Akita: #253 (v8, on `main`) → #255 → #260 → #254 (v9) | all open | RP2 (A1–A4), W2's #255 port (§2.5 of the Akita spec, D10) | #253 pins jolt to unmerged J6 (`9470792dc`); it re-pins after J6/J7 land |
| jolt #2000 / akita #185 (NARG) | **closed, superseded** | — | integration branches still sit on them → T0 |
| Shared primitives: spec PR jolt #2003 | closed (Aerie leak); branch `spec/shared-primitives` @ RP0–RP5 text | RP0 gate: LZ agreement that Whiteout's building blocks live in `jolt-crypto`/`jolt-zk` | **Resolved by owner ruling (2026-11-04):** assume Akita adopts the jolt crates and deletes its copies (A1/A2). akita#120 still has no comments; that is no longer a blocker. |
| Recursion: akita #279 (re-land #189/#190 on jolt `main`) | open, review requested from RadNi, mtbadakhshan | jolt #1911 | LZ review |
| Recursion: jolt #1911 (+6.5k/−1.5k) | open, review required | nothing in Whiteout. It conflicts with J1 (it touches `jolt-field` and `jolt-akita`) | akita #279 |
| jolt #2163 (field/inline prereqs) | merged 2026-10-28 | — | not yet on `whiteout/dev`; U3 brings it in |
| Akita `main` since the integration base: #232 (Module-SIS → NIST cat. 3), #212, #261, #262, #213 | merged | ZK parameter pricing (D3, D14, D35) | U3 merge. **The ZK schedules must be re-planned under #232's estimator** (risk R7) |

Jolt `main` pins Akita `main` @ `f594bfc0d`. The `a4978d59f` pin is the integration branch's (the #185 head) and lacks `zk-prover`. The local `[patch."…akita.git"]` path block in the root `Cargo.toml` (tagged `[pin]`) is never extracted.

## 2. Proposed PR series

**How sizes were measured.** Two methods:
- **Path set.** `git diff -M --numstat` over the listed paths. The jolt base is `aa5a8d697` (the #2000 head merged with `main` at the merge-base), so the superseded transcript work is excluded. The Akita base is `87b4ca046` (the integration branch after its `origin/dev` merge).
- **Churn.** Summed over the ticket-tagged commits. It overstates the size when a fixup rewrites earlier lines.

"src" excludes `tests/`, `fuzz/`, `benches/`, `examples/` and `tests.rs`. It *includes* inline `#[cfg(test)]` modules.

**Risk classes.**
- **P**: protocol or soundness.
- **CT**: constant time on secrets.
- **W**: wire format (proof, setup or descriptor bytes change).
- **M**: mechanical.

### 2.1 jolt (`a16z/jolt`, target `main`)

| # | PR (tickets) | Scope | Size src / test | Depends on | D# · Akita spec | Risk | Tests landing with it |
|---|---|---|---|---|---|---|---|
| J1 | Rename `jolt-field` → `jolt-algebra`, keep a `jolt-field` re-export crate (M1a) | `crates/jolt-field`→`crates/jolt-algebra`, workspace `Cargo.toml`, CI names | churn +359/−315, 26 files | #1911 landed or rebased (R6) | — | M | existing field suites renamed; one `jolt-algebra` identity check |
| J2 | Copy Akita's NTT/CRT into `jolt-algebra` behind `solinas` (M1b-a) | `crates/jolt-algebra/src/ntt/**` | path 7,934 / 1,555 | J1 | — | M (copy). The SIMD paths carry CT relevance | NEON/AVX-512/IFMA NTT tests (`field-portability.yml` lanes) |
| J3 | Copy the ring, FFT, checked and narrowing arithmetic (M1b-b) | `jolt-algebra/src/{ring/**,fft.rs,checked.rs,narrowing.rs,modulus.rs,accum.rs}`, `benches/{crt_ntt_ops,fft_smooth}.rs` | path 6,787 / 1,872 | J2 | — | M | ring and decomposition tests, CRT bench |
| J4 | Copy the sparse challenges, op-norm rejection and public-matrix derivation into `jolt-crypto` behind `lattice` (M5b) | `jolt-crypto/src/challenge/**`, `lattice/{matrix,view,mod}.rs` | path ≈4,480 / 0 (churn +5,247/−1,035) | J3 | — | M | the moved unit tests; truncated-prefix test |
| J5 | Single-instance sumcheck driver; `PrivateChannel` and `ClaimDomain` in `jolt-transcript` (M2a, M3a) | `jolt-sumcheck/src/{single,schedule,error,round_proof,domain,prover,lib}.rs`, `jolt-transcript/src/{private,channel,fork}.rs` | path 1,192 / 768 | **T0, J7 #2110** | D7 · Akita §4.4, §7 | P (private routing must be byte-identical) | sumcheck/transcript suites; clear proofs byte-identical (verifier fixtures) |
| J6 | Public per-round degree schedule for batched sumchecks (DS) | `jolt-sumcheck/src/schedule.rs` users, `jolt-verifier` stages, fuzz bundles | churn +496/−152 / +325/−71 | J5 | D6 | **W** for clear Jolt (fuzz fixture bundles regenerate) | regenerated verifier fixtures and fuzz seeds |
| J7 | Constant-time discrete Gaussian, rejection rule, DIT pool (M5a, A11) | `jolt-crypto/src/sample/**`, `crates/jolt-dit`, `jolt-crypto/examples/sample_dudect.rs` | path 1,864 / 244 | J1 | D1, D15, D16 | **CT** | sampler statistics tests; dudect harness as a documented tool (R10) |
| J8 | Delete `jolt-akita`'s placeholder `ZkOpeningScheme` impls (M4) | `jolt-akita/src`, two tests | churn +22/−246 / +8/−231 | — | — | M | — |
| J9 | Rename `jolt-blindfold` → `jolt-zk` (M6a) | crate rename, imports | churn +41/−39 / +71/−71 | J8 | — | M | renamed BlindFold suites |
| J10 | Masking: pad schedule, masking recorders, factored claims (M6b) | `jolt-zk/src/masking/**`, `jolt-sumcheck/src/{masked,recorder}.rs`, `jolt-claims/src/factored.rs`, `jolt-prover/src/recorder.rs` | path 2,586 / 1,098 | J5, J9 | D11, D12, D26 | P | `masking/tests.rs`; claims fuzz; pad coverage |
| J11 | Jolt residual-relation rows per stage (M6b/M6c) | `jolt-verifier/src/stages/zk/residual/**` | path 4,048 / 0 (`mod.rs` alone is 1,989) | J10 | D12, D26 | P | exercised by J12/J16 suites (no standalone suite today; see R11) |
| J12 | BlindFold over committed pads; per-round Pedersen removed (M6c) | `jolt-zk/src/blindfold/**`, `jolt-verifier/src/stages/zk/{blindfold,mod,inputs,outputs}`, `jolt-prover/src/blindfold.rs` | path +1,276 / −~4.4k net of rename noise (churn +3,406/−5,845) | J11 | D8 | **W** (Dory ZK proof) + P | regenerated Dory-ZK verifier fixtures; `zk_proof_tamper_must_reject` seeds; `blindfold_backend.rs` |
| J13 | Ajtai map, layout, hiding randomness; the constant-term lift, segments, parameters; M6d's moved blocks (M6d, W4 verifier half) | `jolt-crypto/src/lattice/**`, `jolt-zk/src/whiteout/{lift,linear,ring,params,segments,labels,error,mod,placement,challenge}.rs`, `jolt-zk/src/secret.rs` | path 4,069 / 831 | J4, J10 | D14 · Akita §6 (M6d side) | P, CT (Ajtai on secrets) | `whiteout_params.rs`, `whiteout_ring.rs`, lattice tests |
| J14 | Σ-protocol, response statement, `ResponseProver` trait, pad commitment, compression (W4 prover half, W5a/W5c jolt side) | `jolt-zk/src/whiteout/{sigma,sample,dit,weights,response,committed_pads,pad_commitment,compression,opening}.rs`, test support, `examples/sigma_*` | path 4,040 / 1,930 | J7, J13 | D1, D2, D9, D13, D14, D17, D34 | **P, CT** | `whiteout/tests.rs`, zeroize, sample tests, Σ dudect (tool) |
| J15 | `ZkConfig::Whiteout` (appended), statement word, `HidingCommitment`/`ZkOpening` split, `jolt-akita` `HidingCommitment`, **pin Akita `dev`** (W6a, W6b) | `jolt-verifier/src/{config,proof}.rs`, `jolt-akita/src/{scheme,adapters}.rs`, `jolt-akita/src/whiteout/**` (part), `Cargo.toml` | churn +2,264/−321 / +232/−274 | J14, **A13**, J8 #2111 | D25 | **W** (statement word, enum variant) | `protocol_wire_format_is_explicit` + Whiteout case; clear fixtures unchanged |
| J16 | Masked group opening, Whiteout prover/verifier path, bounded rerun, `compile_error!` removed, CI lane (W6c, W6d, W6e, W6g) | `jolt-verifier/src/{whiteout.rs,stages/stage8/akita.rs,verifier.rs}`, `jolt-prover/src/akita/**`, `jolt-akita/src/whiteout/{mod,opening}.rs`, `.github/workflows/{rust,fs-soundness}.yml` | churn +2,379/−961 / +1,960/−152 (W6 path total 5,393 / 3,488 across J15–J17) | J15 | D13, D23, D25, D29→O4 | **P, W** | `whiteout_e2e`; `whiteout_muldiv` fixture; `soundness::whiteout`, `fs_attacks::whiteout`; HT1 census; tamper sweep V5 |
| J17 | Precommitted groups opened masked; persisted plans (W6f, O1, O4/O5 jolt side) | `jolt-akita/src/whiteout/**`, `jolt-prover/src/akita/{preprocessing,witness,spine}.rs`, `jolt-openings/src` | churn +2,521/−648 / +958/−178 | J16 | D24, D31, D32, D34 | P, **W** (verifier setup carries `WhiteoutPlans`) | W6f suites; plan re-audit-on-load tests |
| J18 | Cleanup and perf: delete the `jolt-field` shim (M1d); O0/O3/O3b/V6/M1/M2/S1/S2 | 635 files (import sweep) + `jolt-zk`, `jolt-verifier` hot paths | M1d churn +890/−928; perf ≈ +1.6k/−0.6k | A1 (Akita off `jolt-field`), J17 | — | M; perf needs benchmark evidence | profile summaries; no new permanent tests |

**Over the 5k target.** Two PRs exceed it, and neither should be split further:
- J2 and J3 together are 14.7k lines of copied code. They are reviewable only as provenance-checked copies, against `akita-algebra` at a stated rev with lint-driven edits listed (R5).
- J11 is 4.0k lines, and one 2k-line `mod.rs` holds the stage-generic row builder.

J16 sits at ~2.4k lines plus 2k of tests, but its review weight is the soundness argument, not its size.

### 2.2 Akita (`LayerZero-Labs/akita`)

The RP2 seam sizes are churn on `a4978d59f..a0e3087ad`. The RP4 sizes are churn and path sets on `87b4ca046..HEAD`.

| # | PR (tickets) | Base | Scope | Size src / test | Depends on | D# · Akita spec | Risk | Tests landing with it |
|---|---|---|---|---|---|---|---|---|
| A1 | Switch to `jolt-algebra`; delete Akita's ring/NTT/FFT copies, keep thin re-exports (M1c) | `main` | `akita-algebra/**`, `akita-cpu-backend/src/{opaque,sources,commitment,arithmetic}` imports | churn +1,478/−15,341 / +406/−3,754 | J3 on jolt `main`, #253 | — | M; byte-identical | existing suites; proof bytes unchanged (C0 digests if A8's test lands first; R9) |
| A2 | Switch to `jolt-crypto` challenges and matrix derivation; delete `akita-challenges`' copies (M5c) | `main` | `akita-challenges/**`, `akita-cpu-backend`, `akita-types/src/proof` | churn +507/−4,178 / +95/−181 | J4, A1 | — | M | as A1 |
| A3 | Port Akita's driver onto `jolt-sumcheck` (M2b) | `main` | `akita-sumcheck/**`, `akita-cpu-backend/src/opaque`, `akita-prover/src/protocol`, `akita-verifier/src/stages` | churn +2,332/−2,530 / +170/−183 | J5, A1 | D7 · §4.4 | P (driver swap); byte-identical | transcript-hardening suites; byte identity |
| A4 | Route partial-evaluation messages through `PrivateChannel` (M3b) | `main` | `akita-types/src/stage{1,2,3}_sites.rs`, `akita-verifier/src/fold`, `akita-prover/src/protocol` | churn +213/−118 / 0 | A3 | §4.2, §8 | M; byte-identical | event-log fixtures (A29) |
| A5 | Seams S1, S11, S13: one-probe fold responses, `failure_target` (for W2) | `main`, carried into `dev` | `akita-planner/src/schedule_params/**`, `akita-params/src/{sis,transcript_grinding*}`, `akita-prover/src/protocol/fold_grind.rs` | ≤ ~2.0k: Whiteout's edits to these existing `main` files total 1,385 + 300 + 165 + … | A4 | §1, §8 | P (shared code), byte-identical | golden digests |
| A6 | Seams S2, S3, S12: `randomness_columns`, empty randomness tail, nonzero-randomness paths (for W1) | `main`, carried | `akita-cpu-backend/src/{commitment,opaque,sources}`, `akita-params/src/layout`, `akita-types/src/setup_contribution` | ≤ ~1.2k (existing-file edits) | A4 | §1, §3.3 | P, byte-identical | golden digests |
| A7 | Seams S6, S9, S10: `SuffixEnd::Cut`, descriptor `ProtocolFeatureSet` from config, `zk_schedule_family_name` (for W3) | `main`, carried | `akita-verifier/src/{stages,fold}`, `akita-prover/src/protocol`, `akita-types/src/{instance_descriptor,transcript_grinding}`, `akita-config/src/lib.rs` | ≤ ~2.3k (existing-file edits) | A4 | §1 (S9 descriptor bit) | P; **W** only for configs that select ZK | descriptor accept/reject tests |
| A8 | Delete `akita-zk-prover`/`-verifier` (M6e); golden proof digests (C0) | `dev` | two crates, `scripts/check-crate-deps.sh`, `akita-pcs/tests/golden_proof_digests.rs` | +20/−216 / +273/−134 | J13 on jolt `main` (lift moved) | §6 | M | C0 golden digests; dependency check |
| A9 | One-shot fold responses; every precommitted group at a masked level is hiding (W2, W2b) | `dev` | `akita-planner`, `akita-params/src/sis`, `akita-types/src/transcript_grinding`, `akita-verifier/src/masked.rs` | churn +2,594/−1,230 / +2,290/−891 (path 2,678 / 363) | A5, A8 | D3, D10, D18, D24 · §2, §3.5 | **P** | W2-a…d; `mixed_d_rejections.rs` |
| A10 | Hiding randomness per slice, `commit_hiding` (W1) | `dev` | `akita-cpu-backend/src/{commitment,opaque,hiding.rs}`, layout, setup contribution | churn +2,145/−411 / +774/−70 (path 1,980 / 342) | A6, A8 | D4 · §3 | **P, CT** (A24, kernels unaudited) | W1-a…d; setup-contribution tests |
| A11 | Masked levels and the cut; `ClaimDomain` stage checks; `CountingGrinding`; `ResidualOpening`; `cm(u_a)` at the cut (W3, W3c, O4) | `dev` | `akita-verifier/src/{masked,claims}.rs`, `fold/**`, `stages/**`, `akita-prover/src/protocol/prove/**`, `akita-types/src/{stage_checks*,transcript_grinding/replay*}` | churn +3,644/−875 / +1,428/−70 (path 3,672 / 306) | A7, A9, A10 | D5, D7, D18, D29 · §4, §4.8.1 | **P** | W3-a…g; `masked_prefix.rs` |
| A12 | `AkitaResponseProver`, compressed images, CT compression (W5a, W5c, A45, A46) | `dev` | `akita-pcs/src/zk/**`, `akita-params/src/{zk.rs,compression*}`, `akita-cpu-backend/src/{kernels,arithmetic}` | churn +2,574/−579 / +1,340/−119 (path 3,994 / 984) | J14 on jolt `main`, A8 | D9, D20, D27, D34 · §5.1–5.4 | **P, CT** | W5-a, W5-b; `zero_knowledge.rs`; compression dudect (tool) |
| A13 | **Publish ZK**: `IS_ZK`, ZK family artifacts, residual over `R_128`, Σ-radius pricing, CI `test-zk` (W5d, W5e, O5, O6) | `dev` | `artifacts/**`, `akita-planner/src/emit`, `akita-types/src/instance_descriptor`, `akita-pcs/tests/**`, `.github/workflows/ci.yml`, `akita-sis-estimator` | churn +2,528/−451 / +3,126/−1,062 (path 705 / 4,775) | A11, A12 | D19, D28, D30, D35 · §5.5.1–5.5.3 | **P, W** (new schedule artifacts, descriptor bit live) | HT1, coverage, HT3–HT5, W5-e; `zk_end_to_end.rs`; two-witness distinguisher |
| A14 | Perf: parallel planner, root order, memory (O2, O2b, O3, M1, M2, W5-perf) | `dev` | `akita-planner/**`, `akita-pcs/src/zk` | churn +2,492/−1,331 / +226/−51 | A13 | — | M (with evidence); O2b's root-order invariant is P-adjacent | planner cost guard (#262's CI), benchmark record |

Every Akita PR is under 5k source lines. Akita's spec plan W5b (σ_N offload) is dropped (D20, shelved). No Akita PR before A13 makes ZK selectable: until then, `ZkCommitmentConfig` is `#[cfg(feature = "test-support")] pub`.

### 2.3 Order and critical path

```
J7 #2110 ─┬─ T0 (port both integration branches) ─ J1─J2─J3─J4 ─┐
akita #253→#255→#260→#254 ─┘                       J5─J6, J7, J8 ├─ A1─A2─A3─A4 ─ A5,A6,A7 (main; carried in dev)
                                                                   └─ J9─J10─J11─J12, J13─J14 ─ A8 ─ A9,A10 ─ A11 ─ A12 ─ A13 ─ J15─J16─J17 ─ J18, A14
```

**Critical path:** J7/#253 stack → T0 → J1–J5 → A1–A4 (LZ approval on `main`) → A5–A7 (LZ approval) → A9–A11 → A12 (needs J14) → A13 → J15 → J16 → J17.

The two LZ `main` gates sit on this path. Everything else is internal or `dev` review. Jolt RP3 (J9–J14) runs in parallel with the LZ gates.

## 3. Feature gating and rollout

These features exist on the integration branches and must reach upstream unchanged:

| Repo / crate | Feature | Contract |
|---|---|---|
| `jolt-algebra` | `solinas` (existing) | gates the copied ring, NTT and FFT (J2, J3) |
| `jolt-crypto` | `lattice`, `sample`, `sample-diagnostics` | `lattice` is verifier-safe. `sample` is prover-only. Neither is default |
| `jolt-zk` | `whiteout`, `whiteout-prover`, `whiteout-test-support`, `test-support`, `whiteout-diagnostics` | invariant I6: a verifier graph never reaches `whiteout-prover`, `jolt-dit` or `jolt-crypto/sample` |
| `jolt-akita` | `zk`, `zk-prover` | `jolt-verifier/akita`+`zk` → `zk`. `jolt-prover/zk` → `zk-prover` |
| `jolt-prover`, `jolt-verifier` | `zk`, `akita` (existing) | **`compile_error!` on `zk`+`akita` stays on `main` until J16.** J15 appends `ZkConfig::Whiteout`, but no build can produce it before J16 |
| `jolt-prover`, `jolt-verifier`, `jolt-sdk` | `whiteout` | implies `zk` and `akita`. From J16, `zk`+`akita` without it is a `compile_error!` (`jolt-verifier/src/config.rs`) while the Σ-protocol's `N·y` constant-time residual is open |
| Akita `akita-params`/`types`/`prover`/`verifier`/`pcs`/`planner`/`config` | `zk` | compiles the modules. Selects nothing (branches.md rules 2–3) |
| Akita `akita-pcs`, `akita-cpu-backend` | `zk-prover`, `test-support`, `zk-diagnostics` | samplers (A46). `ZkCommitmentConfig` stays behind `test-support` until A13 |

Whiteout's jolt gate is the explicit `whiteout` feature: `zk`+`akita` selects the protocol, and the `compile_error!` refuses that selection unless `whiteout` is also on, until the `N·y` constant-time residual is closed. A13's config type is the Akita gate.

Akita's C0 golden digests (A8) are the mechanical proof of branches.md rule 1 for the seams and for J15's pin of Akita `dev`. On the jolt side, J15 must show that the clear-Akita verifier fixtures are byte-identical across the pin bump.

**CI lanes to add** (all already run on the integration branches):

- **jolt, `rust.yml`:**
  - clippy `jolt-zk` with each of `whiteout`, `whiteout-prover` and `whiteout-diagnostics` (J13/J14);
  - clippy `jolt-verifier --features jolt-zk/whiteout` (additivity);
  - the I6 sampler-free `cargo tree` check (J14);
  - clippy `jolt-akita` with `zk` and with `zk-prover` (J15);
  - clippy Whiteout `jolt-verifier`/`jolt-prover --features whiteout[,prover-fixtures,logging]` (J16);
  - nextest `jolt-prover --features whiteout,prover-fixtures --test whiteout_e2e`, `jolt-verifier --features whiteout,prover-fixtures,logging`, and `jolt-akita --features zk-prover -E 'test(whiteout)'` (J16);
  - `jolt-zk --features whiteout-prover` (J14).
- **jolt, `fs-soundness.yml`:** J16 replaces "Probe Akita ZK support", which asserts that the two features are mutually exclusive, with "Build Akita ZK (Whiteout) verifier" under `whiteout`, and keeps a probe that `zk`+`akita` without `whiteout` fails to compile.
- **jolt, `CLAUDE.md`:** J16 updates the clippy matrix and the "`akita` is mutually exclusive with `zk`" text.
- **Akita, `ci.yml`:**
  - the golden-digest test and the removal of the `akita-zk-*` dependency checks (A8);
  - a feature check that `akita-verifier` resolves `jolt-zk` without `whiteout-prover` (A8);
  - `test-zk` and `whiteout-response-no-setup-cache` (A13).

**Rollback.** Before J16, each PR reverts on its own, because nothing selects Whiteout. From J16 on, a revert of J16 restores the `compile_error!` and leaves J15's appended enum variant in place, which is harmless.

## 4. Verification per gate

- **RP1 (J1–J8).** Clear Dory, clear Akita and Dory-ZK fixtures are byte-identical, except at J6, whose wire change regenerates the fixtures in the same PR. The pinned Akita builds against the new crates.
- **RP2 + seams (A1–A7).** C0 golden digests are unchanged. Every PR records a proof-byte comparison against `main` (branches.md enforcement gap).
- **RP3 (J9–J14).** Dory-ZK fixtures regenerate once, at J12. Masking and Σ suites pass.
- **RP4 (A8–A13).** ZK is selectable only after A13. HT1, HT3 and HT5 pass.
- **RP5 (J15–J17).** V1–V6 of `whiteout-jolt-integration.md`. The cost record is refreshed against `whiteout-v6-costs.md`.

## 5. Review asks

**Paper authors.** Sign-off on D1–D35, grouped by topic. D33 has no entry in the log (R12).

1. **Σ-protocol and sampling (J7, J14):**
   - D1: the CT attempt is outside the paper's model;
   - D2: near-zero rejection by growing σ; please confirm α, M and T_att;
   - D13: oracle input (σ, t_D, n) and the transcript order;
   - D14: closed-form widths, 7 digits, R_256 rounding;
   - D16: the 2^−149 precision term;
   - D17 and A14: ⊥ after T_att is a rerunnable error;
   - A7 (d_A = d_B), A9 (the box envelope), A8 (L).
2. **Fold responses (A9):**
   - D3: one-shot sizing at 2^−136 in total;
   - D22/Q7: the 1/p clause holds only for the whole challenge tuple;
   - D10: #255's public retry, Lemmas 2.5a/2.5b;
   - Q8: is L2 forbidden before the cut, or should it become a hidden range proof?
   - Q6: which Q_tot does §6.2 assume?
3. **Hiding (A10, J17):**
   - D4: one sizing owner, the open alphabet;
   - D24: every precommitted group at a masked level is hiding;
   - D31: the trust root of statement-fixed groups is the holder of r;
   - Q1: pads instead of LHL randomness below the cut.
4. **Cut and residual (A11, A13):**
   - D5: k = 3 only;
   - D18: structural constraints at the masked levels;
   - D28: the public coordinates of w_k;
   - D29 → O4: `cm(u_a)` at the cut;
   - D35: R_128 for the residual;
   - D19: the d = 512 challenge shape at level 1 is unexplained.
5. **Masking and transcript (J5, J6, J10, J11, J16):**
   - D6: the degree schedule;
   - D7: the stage-2 row has no constant term;
   - D11: uni-skip rounds sent in full;
   - D12: auxiliary slots;
   - D26: multi-pair product rows;
   - D25: the rewritten statement binding (#254);
   - D23 / W6-Q5: a bounded rerun (≤ 3) where the paper aborts.
6. **Soundness ledger and evaluation:**
   - D27: the response proof totals about 2^−119.7, below the 128-bit headline. It was accepted for now on 2026-10-22, and needs an author decision before A13;
   - D30: the distinguisher methodology;
   - D15: the measured sampler cost is 40× the paper's;
   - D9 and D34: the chunked compression chains;
   - D32: persisted plans;
   - D20: no response offload.
7. **Back end (J12):** D8, BlindFold over committed pads as the curve back end.

**Akita maintainers (LZ).**

1. In akita#120: agree that Whiteout's building blocks live in `jolt-crypto`/`jolt-zk`, and that Akita depends on `jolt-algebra`, `jolt-crypto`, `jolt-sumcheck` and `jolt-zk`. This is the RP0 gate; the issue has no comments.
2. Review RP2 (A1–A4) on `main`. These PRs delete ~19k lines of Akita code in favour of jolt crates. Ask for a reviewer commitment and an expected turnaround.
3. Accept seams S1–S13 on `main` as behavior-preserving, with C0 digest evidence. Alternatively, tell us which seams they would rather see restructured.
4. Confirm that jolt `main` may pin an Akita `dev` revision, under branches.md rule 1 plus C0.
5. Name an owner for A37 (`akita-setup` does not re-derive the cached public matrix) and A24 (the commitment kernels are not CT-audited).
6. A34: should the response sumchecks join the 128-bit grinding policy?
7. After #232: confirm the ZK families are priced with the cat-3 estimator.
8. State the transcript-stack order: Whiteout needs #254 (embedded transcript) and #255 (one sparse challenge message per fold group) on `main` before A9.
9. Long term: does ZK stay on `dev`, or is it a `dev-protocol` candidate for `main`?

## 6. Decisions and risks for discussion

| ID | Item | Recommendation |
|---|---|---|
| R1 | **The transcript base is closed.** The integration branches carry #2000/#185 and the J-stack is still open. Porting to it rewrites the M2a, M3a, M6b, W4 and W6 transcript code. Q2–Q4 of `whiteout-jolt-integration.md` (stage messages in the NARG, the `ShapeTranscript` sealed impl, `ZkBackend` on `ProverState`) are open with Quang. | Run T0 now, onto the J7/J8 and #254 heads. Do not wait for them to merge. Freeze feature work on the integration branches during T0. Settle Q2–Q4 with Quang before T0 starts. **Resolved by T0 (2026-11-04):** see §7. |
| R2 | **LZ `main` gates (RP2 + seams).** ~7.2k source lines of Whiteout edits sit in files that exist on Akita `main`. | Open A1–A7 together, carry A5–A7 into `dev` at once, and keep the seam diffs to the declared seams. Move non-seam logic into new `zk` modules before opening. |
| R3 | **Jolt `main` pins Akita `dev`** (J15), so clear Akita on jolt then also builds from a `dev` revision. | Accept, on rule 1 + C0. The alternative, waiting for ZK on Akita `main`, has no date. |
| R4 | **J12 changes the Dory BlindFold wire** and deletes per-round Pedersen commitments. That breaks every existing `zk` proof. | Land it. Keeping the committed recorders beside the masking recorders would put two owners on one claim path (the CLAUDE.md invariant). Bump the protocol version and regenerate the fixtures in J12. Needs an a16z owner's explicit OK. |
| R5 | **Copied code over budget** (J2, J3: 14.7k lines; J4: 4.5k). | Accept. Each PR names its Akita source rev and ships a script that diffs the moved files against it, so reviewers read only the listed lint-driven edits. |
| R6 | **#1911 vs J1's rename.** Both touch `jolt-field` and `jolt-akita`. | Land #1911 (after akita #279) before J1. If #1911 slips past RP1's start, J1 rebases over it, not the reverse. |
| R7 | **Akita #232 raised SIS to category 3** after the ZK parameters were priced. | Done in U3, reviewed. All schedules were regenerated. Clear `.aks` files are byte-identical to `main`. Every ZK row still fits ν ≤ 27, the priced search never offloads into the cut level, and the residual stays on `R_128`. The `SisBoundV1` tag moved 4 → 5 (append-only). The chunk bound grew from 884 to 15,115 coefficients, so u_a takes one chunk. Re-run the V6 costs before A13. |
| R15 | **The Σ-binding table is still at 128-bit SIS**, while clear and digit commitments are now at 169. D27 (≈2^−119.7) does not depend on SIS and remains the overall floor. Moving Σ to 169 raises A_p's rank at d = 128 from 9 to 11, which costs proof bytes and verify time. Since #232 the estimator knows only the 169-bit policy, so the checked-in 128-bit table cannot be regenerated (U3 review N1). | **Decide.** Option (a): keep 128 for Σ, consistent with the D27 floor, and restore a 128-bit Σ-only estimator policy so the table regenerates. Option (b): move Σ to 169 for one uniform SIS level, and measure the cost first. Recommend (a) unless LZ or the authors want a uniform cat-3 claim; ask LZ item 7. |
| R16 | **Main's FusedInc commitment (#2145).** Under `zk` it would have been committed in the clear. U3 made it a hiding precommitted group, opened masked (reviewed; D24/D31 updated in U5). | Carry it in J16/J17. Ask the authors to confirm D24's scope covers it. |
| R17 | **U3 kept Whiteout's transcript bytes** and did not take main's spongefish 0.8, challenge codecs or transcript proof-of-work (#2095/#2097/#2099). The reviewer found these unused by any `main` consumer, so dropping them is safe for now. The dev branches therefore diverge from `main` exactly where T0 must converge. | Fold into T0 (R1). T0 lands Whiteout on main's transcript, not the reverse. **Resolved by T0 (2026-11-04):** both branches now run on main's transcript; see §7. |
| R18 | **Format split.** Whiteout's binary verifier view now has magic `AKSCVFY2` (U5): postcard always writes each row's `zk` field. Views from `main` and Whiteout fail closed on each other. | Carry it in A13 as an explicit wire change. |
| R8 | **DS (J6) changes clear Jolt bytes** for a Whiteout-motivated reason. | Land it early as its own wire PR, so the fixture churn is separate from the Whiteout PRs. |
| R9 | **The C0 golden digests land in `dev` (A8) after RP2 needs them.** | Move the C0 test into a `main` PR ahead of A1, so RP2's byte-identity claim is mechanical. |
| R10 | **Diagnostics in PRs:** dudect, distinguisher and timing examples; the `*-diagnostics` features. | Keep `sample_dudect`, `sigma_dudect`, `compression_dudect` and `zk_distinguisher` as documented tools, each with its command in a doc comment, because CT is a review risk. Defer to U2's audit for the rest. |
| R11 | **J11 has no suite of its own.** | Land J11 together with J12 as a stacked pair, or add a row-shape test against the reference tier. Recommend the stacked pair. |
| R12 | **Paper-log gaps.** D33 is missing. D27 is below the 128-bit headline. D22 and D23 are open. | Treat author sign-off on D3, D22, D23 and D27 as a gate on A13 and J16 only. Everything before those PRs is unselectable and can land first. |
| R13 | **Campaign perf commits** (O*, M1, M2, S1, S2). | Fold the ones that change parameters or wire (O1/D32, O4, O5, O6) into their feature PRs (J17, A11, A13). Ship pure perf as J18 and A14, each with a benchmark record. |
| R14 | **Constant time.** U1 removed the secret-indexed digit lookup. U1b made the IFMA i16 scalar fallback branch-free. Still open: `Ifma52Params::reconstruct` branches on output data; A24's kernels are unaudited; nothing has run on x86_64. On M4, the scalar i16 NTT runs faster on zero operands (a hardware effect, 5–14 t-stat even with DIT). The prover's NEON path does not show it, but x86's scalar path at D < 64 (the response's D = 16/8 levels) is unmeasured. | Get a named CT reviewer for J7, J13, J14, A10 and A12. Run dudect on an x86_64 host (with and without AVX2) before A13. List A24 and the x86 result as known limitations in A13's PR text. |

## 7. T0: integration branches on main's transcript (2026-11-04)

**What landed.** jolt `wo/t0` merges origin/main (beebf45d8) and #2111
(190bdea9d); Akita `wo/akita-t0` merges origin/main (b05545e69, protocol v8)
and origin/dev. Whiteout's own transcript code is gone (Channel, sites, Fork,
nonce, NARG codecs, akita #185's channel); its needs sit on main's:
private routing through wave-1 `PrivateChannel` (clear) and a masked tape in
the proof (Jolt) or Akita's `PrivateProofChannel` (v8 NARG); `ClaimDomain` in
wave-1 form; pad commitments absorbed after the commitments; D25's binding
argument unchanged in structure (D36). Clear Jolt proofs are main's bytes;
clear Akita proofs and schedule artifacts reproduce Akita's golden table.
R1 and R17 are closed. J5's scope shrinks to what wave 1 does not carry
(below); J10–J17 now extract against main's types.

**Where dev keeps a richer form than wave 1** (each is a Whiteout addition,
lands with its first caller):

| Crate | Wave 1 | dev adds |
|---|---|---|
| jolt-transcript | `PrivateChannel`, `ClaimDomain` | `PadSlots`; `ShapeTranscript` (sealed, Q3) |
| jolt-sumcheck | single driver, degree schedule | `masked` (fixed-width masked rounds, `MaskedRoundReader`), `compressed_wire`, `padded_coefficients`, public `check_uniskip_round`, `SumcheckError::PrivateChannel` |
| jolt-crypto | sparse challenges, op-norm rejection, public matrix | Whiteout samplers (indexed XOF, constant-time, position, signed sparse), `*_ct` helpers, the Ajtai/hiding-randomness lattice modules |
| jolt-algebra | ring, NTT, FFT, checked arithmetic | constant-time lifts (`lift_ct`, `to_ring_ct`, `assign_centered_i32_ct`, `digits_to_field_ct`, `centered_mixed_radix_ct`) |
| jolt-zk | (not in wave 1) | masking, BlindFold over pads, Whiteout |

jolt-sumcheck still carries main's `committed`/`r1cs` modules, which no
Whiteout path uses; J12 deletes them with the per-round Pedersen path.

## Not verified

- T0's size. No port onto #2110/#254 was attempted.
- Whether J5–J17 cherry-pick cleanly. The sizes come from path sets and ticket churn, not from extraction dry runs.
- How the seam lines (A5–A7) split from the `dev`-only lines within the ~7.2k existing-file edits. The A5–A7 figures are upper bounds per crate group.
- Whether J1–J4 conflict with `main` after U3, including #2163 and #2165/#2168 in `jolt-akita`.
- Whether akita#120 has an agreement recorded elsewhere, such as Slack or a call.
- The figures in D1–D35 themselves. No tests were run for this note.
