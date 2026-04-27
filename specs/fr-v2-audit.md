# FR Coprocessor v2 — Soundness Audit

15-agent parallel audit of `feat/fr-coprocessor-v2` on refactor-crates, run 2026-04-27.

13 of 15 reports in at the time this file was written. Two outstanding:
- Audit 11 redux (two-pass advice; one came back, second still running)
- Audit 1 redux (FR opcode decode; one came back, second still running)

## How to use this file

1. Pick a finding from the table.
2. **Manually re-verify** by reading the cited file:line. Mark `STATUS` as `verified`, `false-positive`, or `partial`.
3. If verified: write the fix or open a follow-up. If false-positive: cross out, note why.
4. Update the table and move on.

Running list of "verified" / "false-positive" decisions lives at the bottom.

---

## Critical findings (14)

### C1 — Spartan group split excludes FR R1CS rows 19-31
- **Source:** Audits 8, 13 (double-confirmed).
- **File:** `crates/jolt-compiler/examples/jolt_core_module.rs:1318-1319` (prover) and `:1643-1644` (verifier).
- **Claim:** `group0_indices = [1,2,3,4,5,6,11,14,17,18]; group1_indices = [0,7,8,9,10,12,13,15,16]`. Total = rows 0-18. FR rows 19-31 (FAdd/Sub/Mul/Inv/AssertEq/Mov/SLL64/128/192) are absent from both groups → uniskip Lagrange kernel evaluates them at coefficient 0 → prover can plant arbitrary `V_FIELD_RS1/RS2/RD_VALUE` undetected.
- **Status:** ✅ **VERIFIED + FIXED.** Confirmed by reading `params.rs:49` (`UNISKIP_DOMAIN_SIZE=16` correctly accounted for 32 rows) but the index lists hardcoded only 19. Fix: extended both groups to 16 entries each (`group0 += [19,20,21,24,27,28]`, `group1 += [22,23,25,26,29,30,31]`). Verifier mirror at `:1643-1644` updated to match. Stale comments at `params.rs:150-151` (`// 9, // 10 → // 15, // 16`) and `jolt_core_module.rs:1296-1306, 1363, 1365` updated. **Perf:** zero regression — kernel was already sized for 16+16; the lists just under-filled the slots. Regression verified: 41/41 equivalence tests pass, 161/161 witness+host+r1cs+compiler tests pass.

### C2 — Verifier under-squeezes Stage 3 challenges (2 missing)
- **Source:** Audit 15.
- **File:** `crates/jolt-compiler/examples/jolt_core_module.rs:6788` (off by 11 lines from the audit's claim, found nearby).
- **Claim:** Verifier hard-codes `num_s3_challenges = 3 + 3 + log_t`. Prover squeezes 4 gammas (line 2746: shift, inst_input, registers, fr_cr_gamma) and 4 batching coeffs (line 2910). FS transcript diverges from prover after Stage 3.
- **Status:** ✅ **VERIFIED + FIXED.** Counted prover squeezes: 4 gammas (lines 2731, 2740, 2749, 2757) + 4 batching coeffs (line 2921 `for i in 0..4`) + log_t round = `8 + log_t`. Verifier had `3 + 3 + log_t = 6 + log_t`. Off by 2 (one gamma + one batching coeff). The walk-back at line 6789 (`s3_ch_base = ch.decls.len() - num_s3_challenges`) compounds: starts 2 indices too late and squeezes 2 fewer iterations → wrong indices AND wrong count. Fix: changed to `4 + 4 + params.log_t`. **Perf:** zero — verifier op count goes up by 2, both O(1) Squeeze ops. Test gate: 41/41 equivalence + 161/161 unit pass; clippy clean.
- **Note:** Same caveat as C5 — the muldiv test suite doesn't exercise the verifier through full schedule (only commit_skip_alignment), so the missing squeezes weren't caught by regression. Bug surfaces only when the complete self-verify path runs (currently blocked by C4 / Stage 5 verifier ops).

### C3 — Verifier under-squeezes Stage 4 challenges (4 missing)
- **Source:** Audit 15.
- **File:** `crates/jolt-compiler/examples/jolt_core_module.rs:6749-6776`.
- **Claim:** Verifier squeezes only `gamma_registers_rw`, `gamma_ram_val_check`. Prover (lines 3382, 3563-3582) squeezes `ch_gamma_fr_rw` + `ch_batch0/1/2` — 3 more. Same FS divergence.
- **Status:** ✅ **VERIFIED + FIXED.** Counted prover stage-4 FS squeezes: 3 gammas (registers_rw 3382, fr_rw 3393, ram_val_check 3407) + 3 batching (batch0/1/2 at 3574-3582) = **6**. Verifier had `num_s4_challenges = 4` and emitted only 2 explicit Squeezes (gamma_reg_rw, gamma_ram_vc). Comment "(Batching ... handled by VerifySumcheck when added.)" was stale — prover squeezes batching coeffs explicitly via `Op::Squeeze`. Fix: bumped count to 6, added 4 missing Squeeze ops (`gamma_fr_rw` between gamma_reg_rw and the domain separator; 3 batching coeffs after gamma_ram_vc), order mirrors prover exactly. **Perf:** zero — verifier op stream gains 4 O(1) Squeeze ops. Test gate: 41/41 equivalence + clippy clean.
- **Note:** Same caveat as C2/C5 — silent on muldiv because the full stage-4 verify path isn't exercised yet (blocked on C4 Stage 5 ops + complete VerifySumcheck wiring).

### C4 — Stage 5 has zero verifier ops; FR commitments never Dory-verified
- **Source:** Audit 15.
- **File:** `crates/jolt-compiler/examples/jolt_core_module.rs:1191, 1211`.
- **Claim:** Schedule emits only stages 1-4 (`num_stages: 4`). No `build_verifier_stage5_ops`. Stage 4 verifier also lacks `AbsorbCommitment`/`VerifySumcheck`/`RecordEvals` for FR. `FieldRegInc` and `FieldRegRa(d)` commitments are absorbed into the transcript at Stage 1 but never opened or Dory-verified.
- **Status:** ⚠️ **VERIFIED + STUBBED, NOT FIXED.** Confirmed Stage 5 verifier was absent.
  - **Origin:** Pre-existing on refactor-crates. Traced via `git log -S`:
    - `1ed0d7d57 stage 4 done` (markosg04, 2026-04-08): introduced Stage 3/4 verifier stubs (Squeeze-only) + `num_stages: 4`.
    - `819a0d529 stage 5 done` (markosg04, 2026-04-10): added Stage 5 **prover** (1544 LOC). Never wrote Stage 5 verifier or bumped `num_stages`.
    - FR work added FR instances to existing Stage 4 + Stage 5 prover batches without extending the (already-incomplete) verifier — that gap was inherited, not introduced.
  - **What I did:** Added a Stage 5 stub mirroring the existing Stage 3/4 pattern (BeginStage + Squeeze loop for `2 + 4 + LOG_K_INSTRUCTION + log_t` challenges). Bumped `num_stages: 4 → 5`. Per Markos (2026-04-27 slack): **transcript IS in sync** despite stubs skipping explicit `AbsorbRoundPoly` calls — `VerifierOp::Squeeze` reads-and-absorbs the prover's round-poly bytes implicitly from the proof stream then hashes to derive the challenge. Stubs just skip the `s(0)+s(1)=prev_claim` checks and claim propagation (the soundness part), not transcript absorption. So C2/C3/C5/C4-stub correctly maintain alignment; the "not sound" gap is what Markos is now wiring (VerifySumcheck/CheckOutput on stages 3/4/5).
  - **What's left (out of scope for FR audit cleanup):** Build out full `VerifySumcheck` + `RecordEvals` + `AbsorbEvals` + `CheckOutput` for Stages 3, 4, 5 across all instances (including FR instances in Stages 3/4/5). When this happens, the FR-side opening claims need to be included in Stage 4's `AbsorbEvals` (`FieldRegRa` chunks, `FieldRegInc`) and Stage 5's `AbsorbEvals` (`FieldRegInc` + `FieldRegWa` gather). Pinging Markos as the original author of the verifier-side scaffolding.
- **Perf:** zero — stub is `n` Squeeze ops, O(1) each.
- **Test gate:** 41/41 equivalence + clippy clean.

### C5 — Verifier `commit_pairs` skips FieldRegInc + FieldRegRa
- **Source:** Audit 15-redux.
- **File:** `crates/jolt-compiler/examples/jolt_core_module.rs:1592-1601`.
- **Claim:** Verifier iterates `[rd_inc, ram_inc] ⨁ instruction_ra ⨁ ram_ra ⨁ bytecode_ra`. Prover (lines 1227, 1231) pushes `field_reg_inc` and `field_reg_ra`. Prover absorbs FR commitments into Fiat-Shamir; verifier doesn't → as soon as a non-empty FR program runs, every challenge diverges. Empty-FR programs (muldiv) accidentally verify; FR-bearing programs silently break.
- **Status:** ✅ **VERIFIED + FIXED.** Confirmed by reading 1598-1605 (verifier order) vs 1225-1231 (prover order); FieldRegInc / FieldRegRa missing from verifier. Fix: extended `commit_pairs` builder to mirror prover's exact order (`rd_inc, ram_inc, field_reg_inc, instruction_ra, ram_ra, bytecode_ra, field_reg_ra`). Updated stale `Vec::with_capacity(25)` → `params.num_committed + 2`. Updated stage1 verifier doc comment. **Perf:** verifier-side only — adds 2 absorption entries (one per FR poly type) at O(log_t) Dory work each. Negligible. Prover unchanged. Test gate: 41/41 equivalence + 161/161 witness/host/r1cs/compiler tests pass; clippy clean.
- **Note:** the existing `modular_self_verify_commit_skip_alignment` test passes both before and after the fix because it inspects `proof.commitments` shape (prover side), not the verifier's absorption sequence. The hole was purely verifier-side and silent on muldiv until something downstream tried to consume FR commitments.

### C6 — FieldRegRa booleanity missing in Stage 6
- **Source:** Audit 6.
- **File:** `crates/jolt-compiler/examples/jolt_core_module.rs:4860, 5452, 6106`.
- **Claim:** Stage 6 Booleanity sumcheck `ra_poly_ids` iterates only `InstructionRa`, `BytecodeRa`, `RamRa`. `FieldRegRa(d)` is omitted. `total_d = instruction_d + bytecode_d + ram_d` — no `field_reg_d` term. Attacker commits non-Boolean FieldRegRa with no constraint catching it.
- **Status:** ✅ **VERIFIED + FIXED.** Confirmed three sites needed extension:
  - **Line 4860** (`total_d` for booleanity gamma_sq powers): added `+ params.field_reg_d` so gamma_sq powers cover all FR RA dims.
  - **Line 5452** (`ra_poly_ids` for booleanity kernel inputs): chained `(0..params.field_reg_d).map(PolynomialId::FieldRegRa)` after RamRa.
  - **Line 6106** (`bool_ra_poly_ids` for AbsorbEvals flush): same chain extension. MUST match the kernel input order — the AbsorbEvals at line 6121 pairs with the kernel binding via `AliasEval`.
  - Booleanity now checks `ra² = ra` per cell across all RA dimensions including FieldRegRa, so a malicious prover can no longer commit a non-Boolean FR write indicator.
  - Hamming weight extension to FR is **separately tracked** (the audit's `total_d` at line 6206 still uses the integer-only definition; FR has different per-cycle Hamming semantics — at most 1 per cycle vs always 1, so the integer Hamming sumcheck doesn't directly apply). Not part of C6 scope; would be its own finding/work item.
- **Perf:** for FR-active programs, +`field_reg_d` (= 1) gamma squeezes, +1 BooleanityG poly evaluation, +2 booleanity kernel terms (ra² and -ra). For non-FR programs (muldiv), FieldRegRa is all-zero so booleanity is trivially satisfied; cost is the same +1 evaluation but on a zero polynomial. <1% prover overhead either way.
- **Test gate:** 74/74 (jolt-equivalence + jolt-witness) pass; clippy clean.

### C7 — Virtual `field_reg_wa` not cross-bound to committed FieldRegInc
- **Source:** Audit 6.
- **File:** `crates/jolt-compiler/examples/jolt_core_module.rs:4252-4260` (Stage 5 ValEvaluation EqGather); `crates/jolt-witness/src/derived.rs::field_reg_wa, frd_gather_index`; `crates/jolt-witness/src/field_reg.rs::FrCycleBytecode`; `crates/jolt-host/src/tracer_cycle.rs::fr_meta`.
- **Claim:** Stage 4 FR RW + Stage 5 ValEvaluation kernels consume virtual `field_reg_wa` and `frd_gather_index`. Both materialized from `events.write_slot` (uncommitted prover input) instead of bytecode. Compare integer Registers Twist: `rd_gather_index` materializes from `RegisterAccessData.rd_indices` which is bytecode-bound via Spartan. FR's analog had no committed source.
- **Status:** ✅ **VERIFIED + FIXED.** Confirmed root cause: `FrCycleBytecode` was missing `frd` and `writes_frd` fields, so the materializers had no bytecode-side data to source from and fell back to events. Fix:
  - **`FrCycleBytecode`** extended with `frd: u8` and `writes_frd: bool` fields, documented as the cryptographic anchor for the FR write-slot indicator.
  - **`fr_meta()`** in `tracer_cycle.rs` now populates these for every FR variant: FieldOp/FieldMov/FieldSLL64/128/192 set `writes_frd=true` with `frd = operands.rd`; FieldAssertEq sets `writes_frd=false` (no register write).
  - **`field_reg_wa`** in `derived.rs` switched from `snaps.write_slot` to `cfg.bytecode[c].frd / writes_frd`. Identical structure to the existing bytecode-sourced `field_reg_ra_rs1/2` materializers.
  - **`frd_gather_index`** same switch — Stage 5 FieldRegValEvaluation's `EqGather` now evaluates over a verifier-recomputable, bytecode-committed source.
  - **`fr_snapshots`** helper deleted (no remaining callers — both wa and frd_gather_index now source from bytecode directly).
  - Test fixtures in `field_reg.rs`, `fr_derived_materializers.rs`, and `fr_r1cs_integration.rs` updated to populate the new fields. The `frd_gather_index_marks_writes` test was rewritten to assert the new contract: gather index reflects bytecode, not events.
  - Soundness now mirrors integer Registers Twist: FR write slot is committed via Spartan/bytecode preprocessing, the prover can no longer desync `field_reg_wa` from the cycle's actual FR write.
- **Perf:** zero. Same arithmetic, same materializer shape, just sourced from a different (already-attached) field of the input config. `fr_snapshots` deletion saves a per-call replay.
- **Test gate:** 33/33 (jolt-witness) + 41/41 (jolt-equivalence) + 22/22 (tracer FR + jolt-witness FR cross-section) pass; clippy on all touched crates clean.

### C8 — FINV(0) is unsatisfiable in R1CS (~~completeness + soundness bug~~ DX issue)
- **Source:** Audit 4.
- **File:** `crates/jolt-r1cs/src/constraints/rv64.rs:418-437`; tracer `tracer/src/instruction/field_op.rs:120`; SDK `jolt-inlines/bn254-fr/src/sdk.rs:102-115`.
- **Claim:** Tracer defines FINV(0) = 0 via `a.inverse().unwrap_or(Fr::from(0))`. R1CS row 26 demands `rs1 · rd = 1`; unsatisfiable for rs1=0.
- **Reclassification:** Per team review (mzhu, Andrew), this is **neither a soundness nor a completeness gap** under the operative contract ("SDK is the boundary"). Soundness preserved (constraint unsatisfiable, no malicious prover can satisfy it). Completeness preserved within the domain (SDK never emits FINV(0)). Inline-asm callers are explicitly `unsafe` territory — same as bypassing any other safety boundary. Not a CRITICAL audit finding; it's a developer-experience / fail-fast issue.
- **Status:** ✅ **VERIFIED + FIXED.** Two-layer guard:
  - **SDK** (`jolt-inlines/bn254-fr/src/sdk.rs`): renamed `Fr::inv() -> Self` to `Fr::inverse() -> Option<Self>`, mirroring `ark_bn254::Fr::inverse()`. Zero operand short-circuits to `None` before any FieldOp is emitted. Internal `unary_op` impls (host + compute_advice) tightened from `unwrap_or(0)` to `expect("FINV(0) reached unary_op — SDK contract violated")` for defense in depth.
  - **Tracer** (`tracer/src/instruction/field_op.rs:120`): replaced silent `unwrap_or(Fr::from(0))` with a fail-fast panic carrying the offending PC and a pointer to the SDK API. Inline-asm callers get an actionable error at the right layer instead of a cryptic R1CS satisfaction failure downstream.
  - **R1CS:** unchanged. `rs1 · rd = 1` stays.
  - **Regression test:** added `finv_of_zero_panics` in `tracer/src/instruction/field_op.rs` (`#[should_panic]` covers the new path).
- **Perf:** zero. SDK gains one comparison per inverse call. Tracer panic path is unreachable for SDK callers. R1CS unchanged.
- **Test gate:** 100/100 (tracer + jolt-witness + jolt-host + jolt-equivalence) pass; clippy clean.

### C9 — Replay state-vs-event check is `debug_assert!`; release accepts forged events
- **Source:** Audits 3, 9.
- **File:** `crates/jolt-witness/src/field_reg.rs:117`.
- **Claim:** `debug_assert_eq!(state[slot], ev.old, …)` is the only check that the bytecode/event/instruction-word agree about the slot value. Stripped in release. A malicious host can emit mismatched (bytecode.frs1, event.slot) pairs and produce an accepting proof in release builds.
- **Status:** ✅ **VERIFIED + DX-fixed (NOT a soundness issue).** Re-analysis post-C7: Stage 5 FieldRegValEvaluation cryptographically enforces `events.slot == bytecode.frd` and consistent state evolution. The witness `val` (computed from events: `current[ev.slot] = ev.new`) must match the Stage-5 formula `val(k, t) = Σ inc(j)·eq_gather(k, bytecode.frd[j])·LT(t, j)`, which sources `frd[j]` from committed bytecode after C7. Mismatched events → witness val and formula val diverge → Stage 5 sumcheck rejects. So C9 is a **host-side bug-catcher**, not a cryptographic soundness check; a malicious prover can't exploit a release-mode debug strip because the proof would fail downstream anyway.
- **Fix applied** (DX hardening, not soundness): promoted `debug_assert_eq!` to `assert_eq!`. Trade-off: one extra comparison per FR event in release builds; benefit: host-side bugs (or malicious witnesses) fail loud at replay rather than producing a cryptic deep-prover sumcheck error.
- **Test gate:** 195/195 pass; clippy clean. Existing `stale_event_old_state_is_rejected` test still validates.

### C10 — Adversarial event stream not validated (out-of-order, dup, OOB cycles dropped)
- **Source:** Audit 9.
- **File:** `crates/jolt-witness/src/field_reg.rs:82-149`.
- **Claim:** Three release-mode classes silently accepted: out-of-order, duplicates, OOB cycles.
- **Status:** ✅ **VERIFIED + DX-fixed (NOT a soundness issue).** Re-analysis: malformed event streams produce a witness whose val/inc evolution diverges from the Stage 5 formula's bytecode-anchored expectation → sumcheck rejects. Same shape as C9. Cryptographic soundness is preserved by Stage 5 ValEvaluation; the lax replay was a host-side latent bug that surfaced only as cryptic downstream errors.
- **Fix applied** (DX hardening): added explicit up-front validation before replay:
  - Sortedness: `events[i].cycle < events[i+1].cycle` for all i (catches out-of-order AND duplicates in one check).
  - Range: `events.last().cycle < trace_len` (catches OOB).
  - Both as `assert!` (always-on), with descriptive panic messages.
  - Three new `#[should_panic]` regression tests cover each class.
- **Test gate:** 33/33 (jolt-witness) + full equivalence pass; 195/195 across (witness + tracer + equivalence).

### C11 — FieldRegInc committed all-zero in every production path
- **Source:** Audits 14, 9, 7 (triple-confirmed).
- **File:** `crates/jolt-host/src/extract.rs:107-115` (deliberately leaves `dense[FIELD_REG_INC] = 0`); only one repo-wide overwrite at `crates/jolt-equivalence/tests/poseidon2_sdk_e2e.rs`.
- **Claim:** No production code overwrites `FieldRegInc`. Stage 5 sumcheck claim `field_reg_val(r_addr_fr, r_cycle_fr) = Σ inc(j)·eq_gather·LT(j, r_cycle_fr)` only satisfied if trace performs zero FR writes → FR Twist gives no guarantee.
- **Status:** ✅ **VERIFIED + FIXED (latent footgun closed).** Confirmed by repo-wide grep: only the e2e smoke test inserted `FieldRegInc`. Production callers (`jolt-zkvm/tests/muldiv_e2e.rs`, `jolt-bench/src/stacks/modular.rs`, `jolt-equivalence/tests/uniskip_azbz.rs`, etc.) never insert. They were latently safe today only because none of them run an FR-active program; the moment a future caller does, FieldRegInc would be all-zero while FieldRegRa is non-zero — silent FR Twist soundness gap.
  - **Fix:** Added public helper `jolt_witness::field_reg_inc_polynomial<F: Field>(events, trace_length) -> Vec<F>` that computes `new − old` per cycle as **field-element subtraction (mod p)**, NOT `sub_limbs` (which is mod 2^256 and aliases on underflow per Audit 7's H1).
  - Re-exported from `jolt-witness/src/lib.rs`.
  - Updated `extract.rs::cycle_input` doc comment to reference the helper by name and explicitly document the contract: "FR-active program → caller MUST `polys.insert(FieldRegInc, field_reg_inc_polynomial(events, T))` after `polys.finish()`."
  - Replaced the inline `build_field_reg_inc` in `poseidon2_sdk_e2e.rs` with the canonical helper.
  - Future FR-using production callers now have one place to look and one function to call. Forgetting still produces a soundness gap, but the API is documented and the helper is the obvious answer (no inline reimplementation).
- **Perf:** zero. Helper is the same arithmetic the test was already doing inline.
- **Test gate:** 74/74 (jolt-equivalence + jolt-witness) pass; clippy on jolt-witness/jolt-host clean. (Pre-existing clippy lint in `poseidon2_cycle_count.rs:11-12` doc-list indentation is unrelated.)

### C12 — `num_constraints_padded` mismatch (32 vs 64) between params and key
- **Source:** Audit 13.
- **File:** `crates/jolt-compiler/src/params.rs:41, 162` (`NUM_R1CS_CONSTRAINTS = 32`, `num_constraints_padded = 32`); `crates/jolt-r1cs/src/key.rs:49` (`matrices.num_constraints.next_power_of_two() = 35.next_power_of_two() = 64`).
- **Claim:** Prover/verifier disagree on row stride.
- **Status:** ⚠️ **VERIFIED REAL, FIX DEFERRED.** Confirmed end-to-end: `R1csSource::compute_matvec` (provider.rs:54-72) writes Az with `k_pad = key.num_constraints_padded = 64` per cycle (35 rows filled, 29 zero). `Op::RegroupConstraints` handler delegates to `backend.regroup_constraints` which reads `buf[c * old_stride + src_idx]` — at `old_stride = 32` (from `params.num_constraints_padded`), this reads cycle (c+1)/2's data shifted into cycle c's regrouped slot for half the cycles. Garbage output for the regrouped Az/Bz on every odd c.
  - **Why no test currently catches it:** muldiv test paths either (a) check pre-Spartan witness/R1CS satisfaction (bypassing RegroupConstraints), (b) run jolt-core's prover (not modular), or (c) run modular self-verify only up through commit-skip alignment with explicit tolerance for downstream errors past Stage 2. The Stage 1 `CheckOutput` IS in the schedule but not actually exercised because earlier failures (or the test's swallow-error handling) intercept first.
  - **Fix when ready:** change `params.num_constraints_padded` to `NUM_CONSTRAINTS_PER_CYCLE.next_power_of_two() = 64` (matching key). One-line param change, but requires end-to-end self-verify validation to confirm no other latent assumptions on the value. Defer until Markos's verifier patch lands and a real e2e Stage-1 output check can be exercised.
- **Perf impact when fixed:** none expected — kernels and uniskip already operate on the regrouped 16-stride layout. `params.num_constraints_padded` only feeds `Op::RegroupConstraints`'s `old_stride`, which is just an indexing parameter.
- **Test gate:** N/A until fix lands. Existing tests pass either way (bug is masked).

### C13 — FR product (FMUL) not in `ProductConstraintLabel`; FMUL unconstrained
- **Source:** Audit 13.
- **File:** `crates/jolt-r1cs/src/constraints/rv64.rs:411-413`; `crates/jolt-compiler/src/params.rs:60`.
- **Claim:** `rv64.rs:411-413` introduces a 4th use of `V_PRODUCT` (FieldMul). `NUM_PRODUCT_CONSTRAINTS = 3` is wrong; FMUL is "in unsampled FR group → unconstrained."
- **Status:** ❌ **FALSE POSITIVE.** Re-reading rv64.rs:411-413:
  ```rust
  // 23: IsFieldMul · (Product − FieldRdValue) = 0
  a_rows.push(row::<F>(&[(V_FLAG_IS_FIELD_MUL, 1)]));
  b_rows.push(row::<F>(&[(V_PRODUCT, 1), (V_FIELD_RD_VALUE, -1)]));
  c_rows.push(empty()); // <-- empty C-row → eq constraint, not product
  ```
  Row 23 is an **eq constraint** that *uses* `V_PRODUCT` as input, not a new product row. `V_PRODUCT` itself is defined by row 32 (the actual product row): `V_PRODUCT = LeftInstructionInput · RightInstructionInput`. Rows 21+22 force `LeftInstructionInput = V_FIELD_RS1_VALUE` and `RightInstructionInput = V_FIELD_RS2_VALUE` on FMUL cycles, so `V_PRODUCT = rs1 · rs2` automatically; row 23 then asserts `V_PRODUCT = rd_value`. **`NUM_PRODUCT_CONSTRAINTS = 3` is correct** — FMUL doesn't add a 4th product row.
  
  The audit's secondary claim (row 23 is in the "unsampled FR group → unconstrained") was true *before* C1 but is **already fixed by C1** — row 23 is in `group0_indices` after the C1 patch.
- **Fix:** none required.

### C14 — jolt-core legacy R1CSConstraintLabel has 19 variants — doesn't know FR rows exist
- **Source:** Audit 13.
- **File:** `jolt-core/src/zkvm/r1cs/constraints.rs:155-175,228`.
- **Claim:** Enumerates only 19 variants; `NUM_R1CS_CONSTRAINTS = R1CSConstraintLabel::COUNT = 19`. Legacy jolt-core prover/verifier path doesn't know FR rows exist.
- **Status:** ✅ **VERIFIED, WON'T FIX (out of scope).** Confirmed: jolt-core's enum is integer-RV-only. This is **by design** — refactor-crates is in the process of deprecating jolt-core in favor of the modular stack (jolt-zkvm/jolt-r1cs/jolt-compiler), which is where FR lives. Anyone routing an FR program through jolt-core would panic upstream (Audit 1 confirmed jolt-core's `Flags` and `InstructionLookup` impls panic on FR `Cycle` variants). The 19-variant enum is consistent with that direction; extending it would just add dead code as jolt-core winds down.
- **Fix:** none. Documented for future implementers as "jolt-core does not support FR — use the modular stack."

---

## High findings (selected)

### H1 — `FieldAssertEq::execute` is `debug_assert_eq!` (release no-op)
- **Source:** Audits 1, 2, 11. **Likely false-positive at the proof layer.**
- **File:** `tracer/src/instruction/field_assert_eq.rs:79-82`.
- **Claim:** Release strips the equality check.
- **Mitigation already in place:** R1CS row 27 (`crates/jolt-r1cs/src/constraints/rv64.rs:439-445`) enforces `IsFieldAssertEq · (V_FIELD_RS1_VALUE − V_FIELD_RS2_VALUE) = 0` independently. Audit 5 confirmed this is sufficient. **Slow-fail vs fast-fail issue, not soundness break — assuming C1, C6, C7 are fixed so V_FIELD_RS1/2/RD_VALUE are actually bound.**
- **Status:** unverified — re-verify after C1/C6/C7 are addressed.

### H2 — FR opcode decode: bit-4 of register fields unconstrained, tracer panics in release on slot ≥ 16
- **Source:** Audits 1, 3.
- **File:** `tracer/src/instruction/field_*.rs` execute methods (debug_assert! for `frd/frs1/frs2 < field_regs.len()`).
- **Claim:** FormatR fields are 5-bit (0..31) but `field_regs.len() = 16`. In release, bit-4 set in any reg field → release-build slice-index panic (DOS, not soundness). Decode does not constrain bit-4.
- Witness layer masks `& 0xF` everywhere; tracer doesn't. They DISAGREE on bit-4 semantics.
- **Status:** unverified.

### H3 — FieldOp MASK omits funct3; collides with sibling FR opcodes under generic `(word & MASK) == MATCH` walks
- **Source:** Audit 1.
- **File:** `tracer/src/instruction/field_op.rs:58`.
- **Claim:** MASK = `0xfe00007f` excludes funct3, while siblings use `0xfe00_707f`. Hand-ordered dispatch in `tracer/src/instruction/mod.rs:1112-1117` saves prod, but registry/fuzz/property-test paths double-match on funct3 ∈ {6, 7}.
- **Status:** unverified.

### H4 — `From<NormalizedInstruction> for FieldOp` defaults funct3 → FMUL
- **Source:** Audit 1.
- **File:** `tracer/src/instruction/field_op.rs:218-228`.
- **Claim:** Round-trip through `NormalizedInstruction` silently coerces FADD/FSUB/FINV to FMUL. Acknowledged in code, not gated.
- **Status:** unverified — verify whether bytecode preprocessing ever round-trips through `NormalizedInstruction`.

### H5 — `Program::build_with_features` short-circuits regardless of feature flag
- **Source:** Audit 11. **Confirmed in our session — I tripped this earlier.**
- **File:** `crates/jolt-host/src/program.rs:124-127`.
- **Claim:** If a caller does `program.trace(...)` (or `decode()`) before `trace_two_pass_advice`, the compute_advice ELF never gets built → Pass 2 panics on advice read.
- **Status:** verified (during the e2e test session — I worked around it by reordering).

### H6 — `IsRdNotZero` falsely set on FR ops (vacuous today, future hazard)
- **Source:** Audit 12.
- **File:** `crates/jolt-host/src/tracer_cycle.rs:454`.
- **Claim:** `IsRdNotZero = (norm.operands.rd != 0)`. For FR variants, `operands.rd` is `frd` (FR slot 0..31), not integer rd. Routes phantom integer-register write (Inc=0) through RdInc / Registers Twist. Currently vacuous; happenstance soundness.
- **Status:** unverified.

### H7 — `NUM_INSTRUCTION_FLAGS` mismatch (flags.rs=7 vs params.rs=6)
- **Source:** Audit 12.
- **File:** `crates/jolt-instructions/src/flags.rs:93`; `crates/jolt-compiler/src/params.rs:56`.
- **Claim:** Compiler-side consumers under-allocate and silently drop `IsRdNotZero`.
- **Status:** unverified.

### H8 — FR bytecode/lookup table flag stale comments / counts
- **Source:** Audit 12.
- **File:** `crates/jolt-witness/src/bytecode_raf.rs:20`.
- **Claim:** Doc says `NUM_CIRCUIT_FLAGS = 14`; actual is 23. `Vec<bool>` so doesn't break compile, but consumers trusting the doc misread.
- **Status:** unverified — doc only.

### H9 — `replay_field_regs` produces inc as mod 2^256, not mod p
- **Source:** Audit 7.
- **File:** `crates/jolt-witness/src/field_reg.rs:122` (uses `sub_limbs(ev.new, ev.old)`).
- **Claim:** If anything ever wires `FrCycleData.inc` into `FieldRegInc`, Stage 5 identity breaks on any limb underflow. Currently `inc` is computed but the e2e test computes the delta via `limbs_to_field(new) − limbs_to_field(old)` directly, so this is dormant. Recommend deleting `sub_limbs` and the `inc` field.
- **Status:** unverified.

### H10 — FR one-hot K dimension is `k_chunk` (256), not `K_FR` (16)
- **Source:** Audit 14.
- **File:** `crates/jolt-witness/src/polynomials.rs:75`.
- **Claim:** `OneHotBuffer.k = config.k_chunk` for every source. With `log_k_chunk=8` (typical for log_t ≥ 25), `k_chunk=256`. FR write slots span `0..16`. `field_reg_d=1` so chunk extraction is `(value >> 0) & 0xFF` — values stay <16, no truncation, but `expand_one_hot` produces a `T·256` buffer (16× too large) and Dory commits at the wrong K. Fix: thread per-source `k` into `OneHotBuffer`.
- **Status:** unverified.

### H11 — Adversarial advice tape: `cpu.handle_advice_write` discards `Result<(), Trap>`
- **Source:** Audit 11.
- **File:** `tracer/src/instruction/virtual_host_io.rs:25`.
- **Claim:** MMU load fault during advice write silently truncates the tape → Pass 2 desync.
- **Status:** unverified.

### H12 — SDK `encode_r` masks funct7 with `& 0x7F` silently truncates typos
- **Source:** Audit 10.
- **File:** `jolt-inlines/bn254-fr/src/encode.rs:48`.
- **Claim:** `(funct7 & 0x7F) << 25` truncates a stray high bit. No compile error.
- **Status:** unverified — defensive; current call sites are `const` so safe today.

### H13 — SDK clobbers FR slots 1-5 with no save/restore
- **Source:** Audit 10.
- **File:** `jolt-inlines/bn254-fr/src/sdk.rs:122-153, 219-238`.
- **Claim:** Every binary_op clobbers slots 1-5. Inline asm declares no clobber for FReg slots (invisible to LLVM). Footgun the moment a second FR consumer appears.
- **Status:** unverified — single-consumer today.

### H14 — `Fr::from_limbs` accepts non-canonical limbs (≥ p) without reduction
- **Source:** Audit 10.
- **File:** `jolt-inlines/bn254-fr/src/sdk.rs:64-68`.
- **Claim:** Doc says caller asserts value < p; nothing enforces. Host path silently reduces; hardware path may treat bits literally → witness/claim divergence on edge inputs.
- **Status:** unverified.

---

## Medium / Low (compact)

- **M1** — No mutual-exclusivity / booleanity R1CS row for FR flags (`crates/jolt-r1cs/src/constraints/rv64.rs:371-479`). Mutex relies on `static_circuit_flags` macro. Audits 4, 12.
- **M2** — `FieldRegEvent` duplicated (tracer cpu.rs:169 vs witness field_reg.rs:67), positional copy. No `From` impl. Audit 3, 9.
- **M3** — Three-impl SDK arithmetic duplicated (host / compute_advice / tracer); FINV(0)→0 happens to match, no test pinning. Audit 10.
- **M4** — FieldAssertEq R1CS row 27 doesn't constrain `V_FIELD_RD_VALUE` (only RS1−RS2=0). Footgun for future row reads. Audits 9, 12.
- **M5** — `IsLastInSequence` only fires on JALR; FR variants can't terminate virtual sequences. Latent. Audit 12.
- **M6** — RaRs1/RaRs2/Wa share same Stage 4 phase-2 randomness binding without booleanity backing. Audit 6.
- **M7** — `silent fall-through when guest lacks compute_advice feature (`build_with_features` returns OK without setting `elf_compute_advice`). Audit 11.
- **L1** — Docstring drift on FieldSLL128/192 limb position (`field_sll{128,192}.rs:6`). Cosmetic. Audits 1, 5.
- **L2** — `JOLT_ADVICE_WRITE_CALL_ID` u32 vs u64 type mismatch. Works only because tracer reads as u32. Audit 10.
- **L3** — `K_FR=16` constant duplicated in 4 places (cpu.rs, derived.rs, config.rs, field_reg.rs). Audit 3.
- **L4** — Stale comments referencing 19 R1CS rows / 14 circuit flags in `jolt_core_module.rs` and `bytecode_raf.rs`. Audits 12, 13.
- **L5** — Stale `Vec::with_capacity(25)` for `commit_pairs`. Audit 15-redux.
- **L6** — Sentinel `255` in PolynomialId doc vs `u64::MAX` in derived.rs. Audit 7.

---

## Bottom line

The FR Twist was structurally mirrored from Registers Twist but **the verifier-side support and Spartan integration were not wired up**:
- Outer Spartan uniskip doesn't sample FR rows (C1)
- Stage 3/4 verifier doesn't squeeze FR challenges (C2, C3)
- Stage 5 verifier ops don't exist (C4)
- Verifier commit_pairs skips FR polys (C5)
- Stage 6 Booleanity doesn't cover FieldRegRa (C6)
- Virtual `field_reg_wa` not bound to committed Inc (C7)
- `num_constraints_padded` disagrees between params and key (C12)
- FR product rows count out of date (C13)

Plus several runtime soundness checks live only in `debug_assert!` (C9, C10, H2).

**Net effect:** the FR coprocessor proof currently provides zero soundness guarantee. A malicious prover can produce a passing proof for arbitrary FR computations.

**This is a "wiring isn't done" situation, not a "subtle algebra mistake" — most fixes are mechanical.** The hard part is C7 (cross-binding wa to Inc), which may need a Stage 6 Hamming/Booleanity extension.

---

## Verification & decision log

| ID | First check | Status | Notes |
|---|---|---|---|
| C1 | jolt_core_module.rs:1318 | ✅ verified+fixed | extended group_indices to 16+16; perf-neutral; muldiv + full equivalence suite pass |
| C2 | jolt_core_module.rs:6788 | ✅ verified+fixed | num_s3_challenges 3+3 → 4+4; perf-neutral |
| C3 | jolt_core_module.rs:6749 | ✅ verified+fixed | num_s4_challenges 4 → 6; added 4 missing Squeeze ops; perf-neutral |
| C4 | jolt_core_module.rs:1191 | ⚠️ verified, stubbed only | pre-existing scaffolding gap; full verifier rebuild deferred to Markos |
| C5 | jolt_core_module.rs:1592 | ✅ verified+fixed | extended verifier commit_pairs to mirror prover; muldiv self-verify still passes |
| C6 | jolt_core_module.rs:4860/5452/6106 | ✅ verified+fixed | chained FieldRegRa(d) into booleanity ra_poly_ids + bumped local total_d |
| C7 | derived.rs / fr_meta / FrCycleBytecode | ✅ verified+fixed | added frd/writes_frd to bytecode; switched wa + frd_gather to bytecode-sourced |
| C8 | rv64.rs:418 / sdk.rs / field_op.rs | ✅ verified+fixed (downgraded MEDIUM) | SDK Option<Fr> + tracer panic on FINV(0); not soundness/completeness per team review |
| C9 | field_reg.rs:117 | ✅ verified, DX-fixed | not soundness — debug_assert promoted to assert; cryptographic enforcement is in Stage 5 ValEval |
| C10 | field_reg.rs:82 | ✅ verified, DX-fixed | not soundness — added sortedness/range asserts + 3 should_panic regression tests |
| C11 | extract.rs:107 | ✅ verified+fixed | added `field_reg_inc_polynomial` helper in jolt-witness; documented the contract |
| C12 | params.rs:162, key.rs:49 | ⚠️ verified, deferred | real stride mismatch (32 vs 64); fix needs e2e self-verify, blocked on Markos's verifier patch |
| C13 | rv64.rs:411 | ❌ false positive | row 23 is eq, not product; FMUL reuses V_PRODUCT; NUM_PRODUCT_CONSTRAINTS=3 is correct |
| C14 | jolt-core/zkvm/r1cs/constraints.rs:155 | ✅ won't-fix (by design) | jolt-core deprecated in favor of modular stack; FR isn't routed through jolt-core |
