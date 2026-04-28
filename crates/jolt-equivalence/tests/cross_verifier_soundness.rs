//! Cross-verifier soundness suite (spec §4.5 / §4.6).
//!
//! Quantifies the gap between the modular verifier (`V_mod`) and the
//! reference jolt-core verifier (`V_core`) on the muldiv fixture.
//!
//! Test inventory:
//!
//! - `honest_acceptance_equivalence` — S1: V_mod accepts honest proof; S5'/S6
//!    cannot disagree on an honest run.
//! - `known_gap_registry_consistency` — KGC: every entry in `KNOWN_GAPS`
//!    still produces `(V_core = Reject, V_mod = Accept)` on the fixture.
//! - `t1_round_poly_coefficients` — T1: per-stage round-poly coefficient
//!    tampers; both verifiers must reject for stages 1–2 (already wired)
//!    and the registered known-gap entry covers stages 3–7.
//! - `t8_round_poly_degree` — T8: round-poly degree tampers (truncate /
//!    extend), same coverage as T1.
//! - `taxonomy_covers_in_scope_constraints` — narrow S7: every
//!    `Constraint` produced by the in-scope schedule is covered by at
//!    least one tamper in `TAMPER_COVERAGE`.

#![allow(clippy::print_stderr)]

use jolt_equivalence::cross_verifier::categories::Constraint;
use jolt_equivalence::cross_verifier::tamper::DegreeKind;
use jolt_equivalence::cross_verifier::{
    fixture, run_tampered, ConstraintTemplate, ExpectedResult, KnownGap, TamperKind,
    TamperLocation, TamperMutation, TamperPoint, KNOWN_GAPS, TAMPER_COVERAGE,
};

/// S1 — V_mod accepts the honest fixture proof. V_core also accepts the
/// converted form (sanity-check: the conversion preserves honest
/// acceptance).
#[test]
fn honest_acceptance_equivalence() {
    let f = fixture();
    let modular = jolt_verifier::verify(f.modular_verifying_key, &f.modular_proof, &f.io_hash);
    assert!(
        modular.is_ok(),
        "V_mod must accept honest proof: {modular:?}"
    );

    let core_proof =
        jolt_equivalence::cross_verifier::modular_to_core(&f.modular_proof, &f.core_scaffold);
    let core = jolt_equivalence::cross_verifier::fixture::verify_with_core(core_proof, f);
    assert!(
        core.is_ok(),
        "V_core must accept converted honest proof: {core:?}"
    );
}

/// KGC — for every registered known gap, verify the gap reproduces:
/// V_core rejects, V_mod accepts. If an entry no longer reproduces,
/// either the gap closed (remove the entry) or the harness regressed
/// (fix the harness).
#[test]
fn known_gap_registry_consistency() {
    let f = fixture();
    let mut still_present = 0usize;
    let mut closed_but_unremoved: Vec<&KnownGap> = Vec::new();

    for gap in KNOWN_GAPS {
        let tamper = canonical_tamper_for(gap);
        let Some(result) = run_tampered(f, &tamper) else {
            // Vacuous: tamper couldn't be applied (e.g. no evals to
            // tamper). Treat as gap-closed-trivially.
            closed_but_unremoved.push(gap);
            continue;
        };
        // Gap "reproduces" iff modular fails to reject a tamper that
        // should be rejected. Two manifestations:
        //   (a) CoreRejectsModularAccepts — core catches it, modular
        //       doesn't. The classic cross-equivalence gap (T1/T8).
        //   (b) BothAccept — neither catches it, but modular ought to
        //       (for tamper kinds where core's view is filtered by
        //       opening-claim substitution, e.g. T2 eval tampers).
        let reproduces = result.core_rejects_modular_accepts()
            || (result.both_accept() && modular_must_reject(gap.kind));
        if reproduces {
            still_present += 1;
        } else {
            eprintln!(
                "[KGC] gap (stage={}, kind={:?}) no longer reproduces: \
                 core={:?} modular={:?}",
                gap.stage, gap.kind, result.core, result.modular
            );
            closed_but_unremoved.push(gap);
        }
    }

    eprintln!(
        "KGC report: {still_present}/{} gaps reproduce; {} closed-but-unremoved",
        KNOWN_GAPS.len(),
        closed_but_unremoved.len(),
    );

    assert!(
        closed_but_unremoved.is_empty(),
        "{} known-gap entries no longer reproduce — remove them from KNOWN_GAPS:\n{:#?}",
        closed_but_unremoved.len(),
        closed_but_unremoved,
    );
}

/// T1 — round polynomial coefficient tampers across stages 1–7.
/// Stages 1–2: both verifiers must reject (BothReject).
/// Stages 3–7: registered known gaps cover these — V_core rejects, V_mod
/// accepts (until parity work lands).
#[test]
fn t1_round_poly_coefficients() {
    let f = fixture();
    let mut total = 0usize;
    let mut rejected_by_both = 0usize;
    let mut registered_gap = 0usize;
    let mut failures: Vec<String> = Vec::new();

    for stage in 1..=7 {
        // Tamper the middle round's coefficient 0 (constant term) — the
        // most reliably non-trivial coefficient that propagates to the
        // sumcheck round verifier's `s(0) + s(1)` check.
        let tamper = round_poly_tamper(
            stage,
            /*round*/ middle_round_for(stage, f),
            /*coeff*/ 0,
        );
        total += 1;
        let Some(result) = run_tampered(f, &tamper) else {
            failures.push(format!(
                "T1 stage {stage}: tamper vacuous (no round poly at chosen index?)"
            ));
            continue;
        };
        if result.both_reject() {
            rejected_by_both += 1;
            continue;
        }
        if result.core_rejects_modular_accepts()
            && jolt_equivalence::cross_verifier::is_registered(stage, TamperKind::T1RoundPolyCoeff)
        {
            registered_gap += 1;
            continue;
        }
        failures.push(format!(
            "T1 stage {stage}: outcome={} — core={:?}, modular={:?}",
            result.outcome_label(),
            result.core,
            result.modular,
        ));
    }

    eprintln!(
        "T1 report: total={total}, both_reject={rejected_by_both}, \
         registered_gap={registered_gap}, failures={}",
        failures.len(),
    );

    assert!(
        failures.is_empty(),
        "T1 unexpected outcomes:\n{}",
        failures.join("\n"),
    );
}

/// T8 — round polynomial degree tampers (truncate or extend) across
/// stages 1–7. Same expected-outcome shape as T1.
#[test]
fn t8_round_poly_degree() {
    let f = fixture();
    let mut total = 0usize;
    let mut rejected_by_both = 0usize;
    let mut registered_gap = 0usize;
    let mut failures: Vec<String> = Vec::new();

    for stage in 1..=7 {
        let tamper = round_poly_degree_tamper(
            stage,
            /*round*/ middle_round_for(stage, f),
            DegreeKind::Truncate,
        );
        total += 1;
        let Some(result) = run_tampered(f, &tamper) else {
            failures.push(format!("T8 stage {stage}: vacuous"));
            continue;
        };
        if result.both_reject() {
            rejected_by_both += 1;
            continue;
        }
        if result.core_rejects_modular_accepts()
            && jolt_equivalence::cross_verifier::is_registered(stage, TamperKind::T8RoundPolyDegree)
        {
            registered_gap += 1;
            continue;
        }
        failures.push(format!(
            "T8 stage {stage}: outcome={}",
            result.outcome_label(),
        ));
    }

    eprintln!(
        "T8 report: total={total}, both_reject={rejected_by_both}, \
         registered_gap={registered_gap}, failures={}",
        failures.len(),
    );

    assert!(
        failures.is_empty(),
        "T8 unexpected outcomes:\n{}",
        failures.join("\n"),
    );
}

/// T2 — evaluation tampers across stages 1–7. For each stage that
/// records evals (`stage_proofs[stage-1].evals` non-empty), tamper the
/// first eval. The pass condition is "modular rejects" — soundness is
/// a modular-side property and cross-equivalence is bypassed for eval
/// tampers because `modular_to_core` substitutes core's honest
/// opening_claims (the conversion strips the tamper from core's view).
///
/// Modular rejects via two paths today:
/// - With CheckOutput on the affected stage, `final_eval ≠
///   output_check(evals, point)` (direct).
/// - Without CheckOutput, the tampered value gets absorbed by
///   `AbsorbEvals`, diverging the verifier's transcript from the
///   prover's. The next stage's sumcheck verification then fails
///   on a mismatched first-round expected sum.
///
/// Stage 7 has no downstream sumcheck, so eval tampers slip through
/// until either CheckOutput or CollectOpeningClaim is wired
/// (registered gap).
#[test]
fn t2_eval_tampers() {
    let f = fixture();
    let mut total = 0usize;
    let mut rejected_by_modular = 0usize;
    let mut registered_gap = 0usize;
    let mut vacuous = 0usize;
    let mut failures: Vec<String> = Vec::new();

    for stage in 1..=7 {
        let tamper = eval_tamper(stage, /*idx*/ 0);
        total += 1;
        let Some(result) = run_tampered(f, &tamper) else {
            vacuous += 1;
            continue;
        };
        // Pass condition: modular caught the tamper. Outcome could be
        // BothReject (CheckOutput stages) or CoreAcceptsModularRejects
        // (transcript-divergence stages — core's view is filtered by
        // opening-claim substitution).
        if result.modular.is_err() {
            rejected_by_modular += 1;
            continue;
        }
        // Stage where modular accepts → must be registered as known gap.
        if jolt_equivalence::cross_verifier::is_registered(stage, TamperKind::T2Eval) {
            registered_gap += 1;
            continue;
        }
        failures.push(format!(
            "T2 stage {stage}: outcome={} — core={:?}, modular={:?}",
            result.outcome_label(),
            result.core,
            result.modular,
        ));
    }

    eprintln!(
        "T2 report: total={total}, modular_rejected={rejected_by_modular}, \
         registered_gap={registered_gap}, vacuous={vacuous}, \
         failures={}",
        failures.len(),
    );

    assert!(
        failures.is_empty(),
        "T2 unexpected outcomes:\n{}",
        failures.join("\n"),
    );
}

/// T10 — domain-separator tag tampers. Mutates one `VerifierOp::AbsorbEvals`
/// in the schedule from `DomainSeparator::OpeningClaim` to a different
/// tag, then runs the modular verifier with the modified key. The
/// verifier's transcript diverges from what the prover absorbed, and
/// downstream sumcheck verification fails.
///
/// Tampers schedule (not proof) since transcript tags live in the
/// verifier's compiled schedule, not in proof bytes. We clone the
/// verifying key, mutate one op, and call verify with the modified
/// key.
#[test]
fn t10_domain_separator_tag_tampers() {
    use jolt_compiler::{DomainSeparator, VerifierOp};
    let f = fixture();

    // Clone the verifying key so we can mutate the schedule.
    let mut tampered_key = f.modular_verifying_key.clone();

    // Find the first AbsorbEvals op with DomainSeparator::OpeningClaim
    // and replace its tag with SumcheckClaim. This shifts the
    // domain-separator label absorbed at that position; the prover's
    // honest run absorbed `b"opening_claim"`, the tampered verifier
    // absorbs `b"sumcheck_claim"` instead, diverging the transcripts.
    let mut mutated = false;
    for op in &mut tampered_key.schedule.ops {
        if let VerifierOp::AbsorbEvals { tag, .. } = op {
            if matches!(tag, DomainSeparator::OpeningClaim) {
                *tag = DomainSeparator::SumcheckClaim;
                mutated = true;
                break;
            }
        }
    }
    assert!(
        mutated,
        "T10 tamper vacuous: no AbsorbEvals with OpeningClaim tag in schedule"
    );

    let modular = jolt_verifier::verify(&tampered_key, &f.modular_proof, &f.io_hash);
    eprintln!("T10 report: modular={modular:?}");
    assert!(
        modular.is_err(),
        "T10 must be caught by modular verifier; got {modular:?}",
    );
}

/// T7 — cross-stage eval tampers. Specifically targets evals at stage
/// S whose values flow into stage S+1's `input_claim` formula via
/// `ClaimFactor::Eval` or `ClaimFactor::StagedEval`. T2/T9 cover idx
/// 0 / last-idx; T7 picks a MIDDLE index per stage to ensure complete
/// coverage of evals that propagate downstream.
///
/// Pass condition: modular rejects. Stages 1-6 reject via the next
/// stage's sumcheck verification (combined_claim shifts because the
/// recorded eval differs); stage 7 rejects via stage-8 PCS opening
/// verification (the eval drives the AliasEval-aliased PCS claim).
#[test]
fn t7_cross_stage_eval_tampers() {
    let f = fixture();
    let mut total = 0usize;
    let mut rejected_by_modular = 0usize;
    let mut vacuous = 0usize;
    let mut failures: Vec<String> = Vec::new();

    for stage in 1..=7 {
        let stage_idx = stage - 1;
        let n = f
            .modular_proof
            .stage_proofs
            .get(stage_idx)
            .map(|sp| sp.evals.len())
            .unwrap_or(0);
        if n < 2 {
            // Need at least one eval that's neither idx 0 nor last —
            // T2 / T9 cover those.
            vacuous += 1;
            continue;
        }
        let mid = n / 2;
        let tamper = TamperPoint {
            kind: TamperKind::T7CrossStage,
            label: "T7_CrossStage",
            ..eval_tamper(stage, mid)
        };
        total += 1;
        let Some(result) = run_tampered(f, &tamper) else {
            vacuous += 1;
            continue;
        };
        if result.modular.is_err() {
            rejected_by_modular += 1;
            continue;
        }
        if jolt_equivalence::cross_verifier::is_registered(stage, TamperKind::T7CrossStage) {
            continue;
        }
        failures.push(format!(
            "T7 stage {stage} idx {mid}: outcome={} — core={:?}, modular={:?}",
            result.outcome_label(),
            result.core,
            result.modular,
        ));
    }

    eprintln!(
        "T7 report: total={total}, modular_rejected={rejected_by_modular}, \
         vacuous={vacuous}, failures={}",
        failures.len(),
    );

    assert!(
        failures.is_empty(),
        "T7 unexpected outcomes:\n{}",
        failures.join("\n"),
    );
}

/// T4 — opening-proof structural tamper. Drop the last entry from
/// `proof.opening_proofs`. The verifier's stage-8 length check
/// (`reduced.len() != opening_proofs.len()`) rejects.
///
/// This test only runs the modular verifier (the cross-conversion to
/// core asserts `opening_proofs.len() == 1` and would panic on a
/// truncated proof — the runner's dual-verify path isn't applicable).
#[test]
fn t4_opening_proof_truncation() {
    let f = fixture();
    let mut proof = f.modular_proof.clone();
    let _ = proof.opening_proofs.pop();
    let modular = jolt_verifier::verify(f.modular_verifying_key, &proof, &f.io_hash);
    eprintln!("T4 report: modular={modular:?}");
    assert!(
        modular.is_err(),
        "T4 must be caught by modular verifier; got {modular:?}",
    );
}

/// T3 — commitment-swap tampers. Swap commitments[idx] with
/// commitments[idx+1] in place. Now the commitment associated with
/// `poly[idx]` is the wrong group element; the modular verifier's
/// stage-8 PCS opening verification rejects.
#[test]
fn t3_commitment_swaps() {
    let f = fixture();
    let mut total = 0usize;
    let mut rejected_by_modular = 0usize;
    let mut vacuous = 0usize;
    let mut failures: Vec<String> = Vec::new();

    // For each non-zero slot idx > 0, swap with slot 0 (RdInc — a
    // dense poly with non-trivial commitment). Adjacent swaps within
    // the InstructionRa block can be invisible if both polys are
    // all-zero for the workload (common in muldiv); this against-slot-0
    // pattern catches every Some slot with a structurally-different
    // commitment.
    let n = f.modular_proof.commitments.len();
    for idx in 1..n {
        let tamper = TamperPoint {
            stage: 0,
            location: TamperLocation::Commitment { idx },
            mutate: TamperMutation::AddOne, // unused for Commitment
            witnesses: vec![Constraint::CommitSlot(idx)],
            expected: ExpectedResult::BothReject,
            kind: TamperKind::T3CommitmentSwap,
            label: "T3_CommitmentSwap",
        };
        total += 1;
        let Some(result) = run_tampered(f, &tamper) else {
            vacuous += 1;
            continue;
        };
        if result.modular.is_err() {
            rejected_by_modular += 1;
            continue;
        }
        if jolt_equivalence::cross_verifier::is_registered(0, TamperKind::T3CommitmentSwap) {
            continue;
        }
        failures.push(format!(
            "T3 idx {idx}: outcome={} — core={:?}, modular={:?}",
            result.outcome_label(),
            result.core,
            result.modular,
        ));
    }

    eprintln!(
        "T3 report: total={total}, modular_rejected={rejected_by_modular}, \
         vacuous={vacuous}, failures={}",
        failures.len(),
    );

    assert!(
        failures.is_empty(),
        "T3 unexpected outcomes:\n{}",
        failures.join("\n"),
    );
}

/// T5 — commit-slot None ↔ Some tampers. Either zeroing an honest
/// commitment (SomeToNone) or substituting a different slot's
/// commitment (NoneToSome) breaks the stage-8 PCS verification.
#[test]
fn t5_commit_slot_tampers() {
    let f = fixture();
    let mut total = 0usize;
    let mut rejected_by_modular = 0usize;
    let mut vacuous = 0usize;
    let mut failures: Vec<String> = Vec::new();

    // Apply SomeToNone to every Some slot.
    for (idx, slot) in f.modular_proof.commitments.iter().enumerate() {
        if slot.is_none() {
            continue;
        }
        let tamper = TamperPoint {
            stage: 0,
            location: TamperLocation::CommitSlot {
                idx,
                op: jolt_equivalence::cross_verifier::tamper::CommitSlotOp::SomeToNone,
            },
            mutate: TamperMutation::AddOne, // unused
            witnesses: vec![Constraint::CommitSlot(idx)],
            expected: ExpectedResult::BothReject,
            kind: TamperKind::T5CommitSlotNoneSome,
            label: "T5_CommitSlot",
        };
        total += 1;
        let Some(result) = run_tampered(f, &tamper) else {
            vacuous += 1;
            continue;
        };
        if result.modular.is_err() {
            rejected_by_modular += 1;
            continue;
        }
        if jolt_equivalence::cross_verifier::is_registered(0, TamperKind::T5CommitSlotNoneSome) {
            continue;
        }
        failures.push(format!(
            "T5 idx {idx}: outcome={} — core={:?}, modular={:?}",
            result.outcome_label(),
            result.core,
            result.modular,
        ));
    }

    eprintln!(
        "T5 report: total={total}, modular_rejected={rejected_by_modular}, \
         vacuous={vacuous}, failures={}",
        failures.len(),
    );

    assert!(
        failures.is_empty(),
        "T5 unexpected outcomes:\n{}",
        failures.join("\n"),
    );
}

/// T9 — batch-claim tampers. Tamper an eval at a non-first index in
/// `stage_proofs[stage].evals` (T2 covers idx 0; T9 explores other
/// positions to surface gaps where the first eval happens to be
/// trivially boolean). Each tampered eval flows into either an
/// AbsorbEvals (transcript divergence) or a downstream input_claim,
/// either way modular must reject.
#[test]
fn t9_batch_claim_tampers() {
    let f = fixture();
    let mut total = 0usize;
    let mut rejected_by_modular = 0usize;
    let mut vacuous = 0usize;
    let mut failures: Vec<String> = Vec::new();

    // Tamper eval at the LAST recorded index of each stage with non-empty
    // evals. The last index of each stage typically holds a poly that
    // flows into a downstream claim, exposing different code paths than
    // T2's idx=0 tamper.
    for stage in 1..=7 {
        let stage_idx = stage - 1;
        let last_eval_idx = f.modular_proof.stage_proofs.get(stage_idx).and_then(|sp| {
            if sp.evals.is_empty() {
                None
            } else {
                Some(sp.evals.len() - 1)
            }
        });
        let Some(idx) = last_eval_idx else {
            vacuous += 1;
            continue;
        };
        let tamper = eval_tamper(stage, idx);
        // Reuse the eval_tamper helper but flag as T9.
        let tamper = TamperPoint {
            kind: TamperKind::T9BatchClaim,
            label: "T9_BatchClaim",
            ..tamper
        };
        total += 1;
        let Some(result) = run_tampered(f, &tamper) else {
            vacuous += 1;
            continue;
        };
        if result.modular.is_err() {
            rejected_by_modular += 1;
            continue;
        }
        if jolt_equivalence::cross_verifier::is_registered(stage, TamperKind::T9BatchClaim) {
            continue;
        }
        failures.push(format!(
            "T9 stage {stage} idx {idx}: outcome={} — core={:?}, modular={:?}",
            result.outcome_label(),
            result.core,
            result.modular,
        ));
    }

    eprintln!(
        "T9 report: total={total}, modular_rejected={rejected_by_modular}, \
         vacuous={vacuous}, failures={}",
        failures.len(),
    );

    assert!(
        failures.is_empty(),
        "T9 unexpected outcomes:\n{}",
        failures.join("\n"),
    );
}

/// T6 — config-field tampers. Mutating `proof.config.{trace_length,
/// ram_k}` changes preamble absorption; both verifiers replay the
/// modified config, transcript diverges from prover's, downstream
/// sumcheck fails.
#[test]
fn t6_config_field_tampers() {
    let f = fixture();
    let mut total = 0usize;
    let mut rejected_by_modular = 0usize;
    let mut vacuous = 0usize;
    let mut failures: Vec<String> = Vec::new();

    let cases: Vec<(&'static str, TamperPoint)> = vec![
        (
            "trace_length_doubled",
            TamperPoint {
                stage: 0,
                location: TamperLocation::Config(
                    jolt_equivalence::cross_verifier::tamper::ConfigField::TraceLength,
                ),
                mutate: TamperMutation::SetUsize(f.modular_proof.config.trace_length * 2),
                witnesses: vec![Constraint::Preamble],
                expected: ExpectedResult::BothReject,
                kind: TamperKind::T6ConfigField,
                label: "T6_TraceLength",
            },
        ),
        (
            "ram_k_plus_one",
            TamperPoint {
                stage: 0,
                location: TamperLocation::Config(
                    jolt_equivalence::cross_verifier::tamper::ConfigField::RamK,
                ),
                mutate: TamperMutation::SetUsize(f.modular_proof.config.ram_k + 1),
                witnesses: vec![Constraint::Preamble],
                expected: ExpectedResult::BothReject,
                kind: TamperKind::T6ConfigField,
                label: "T6_RamK",
            },
        ),
    ];

    for (name, tamper) in &cases {
        total += 1;
        let Some(result) = run_tampered(f, tamper) else {
            vacuous += 1;
            continue;
        };
        if result.modular.is_err() {
            rejected_by_modular += 1;
            continue;
        }
        if jolt_equivalence::cross_verifier::is_registered(tamper.stage, TamperKind::T6ConfigField)
        {
            continue;
        }
        failures.push(format!(
            "T6 case {name}: outcome={} — core={:?}, modular={:?}",
            result.outcome_label(),
            result.core,
            result.modular,
        ));
    }

    eprintln!(
        "T6 report: total={total}, modular_rejected={rejected_by_modular}, \
         vacuous={vacuous}, failures={}",
        failures.len(),
    );

    assert!(
        failures.is_empty(),
        "T6 unexpected outcomes:\n{}",
        failures.join("\n"),
    );
}

/// T11 — public-IO tampers. Mutating the proof's public inputs/outputs
/// changes the preamble absorption, which both verifiers replay; the
/// transcript diverges immediately and downstream sumcheck verification
/// fails.
#[test]
fn t11_public_io_tampers() {
    let f = fixture();
    let mut total = 0usize;
    let mut rejected_by_modular = 0usize;
    let mut vacuous = 0usize;
    let mut failures: Vec<String> = Vec::new();

    let cases: Vec<(&'static str, TamperPoint)> = vec![
        (
            "input_byte[0]_set_0xFF",
            TamperPoint {
                stage: 0,
                location: TamperLocation::Io(
                    jolt_equivalence::cross_verifier::tamper::IoField::InputByte(0),
                ),
                mutate: TamperMutation::SetByte(0xFF),
                witnesses: vec![Constraint::Preamble],
                expected: ExpectedResult::BothReject,
                kind: TamperKind::T11PublicIo,
                label: "T11_InputByte",
            },
        ),
        (
            "panic_flag_set_true",
            TamperPoint {
                stage: 0,
                location: TamperLocation::Io(
                    jolt_equivalence::cross_verifier::tamper::IoField::PanicFlag,
                ),
                mutate: TamperMutation::SetBool(true),
                witnesses: vec![Constraint::Preamble],
                expected: ExpectedResult::BothReject,
                kind: TamperKind::T11PublicIo,
                label: "T11_Panic",
            },
        ),
    ];

    for (name, tamper) in &cases {
        total += 1;
        let Some(result) = run_tampered(f, tamper) else {
            vacuous += 1;
            continue;
        };
        if result.modular.is_err() {
            rejected_by_modular += 1;
            continue;
        }
        if jolt_equivalence::cross_verifier::is_registered(tamper.stage, TamperKind::T11PublicIo) {
            continue;
        }
        failures.push(format!(
            "T11 case {name}: outcome={} — core={:?}, modular={:?}",
            result.outcome_label(),
            result.core,
            result.modular,
        ));
    }

    eprintln!(
        "T11 report: total={total}, modular_rejected={rejected_by_modular}, \
         vacuous={vacuous}, failures={}",
        failures.len(),
    );

    assert!(
        failures.is_empty(),
        "T11 unexpected outcomes:\n{}",
        failures.join("\n"),
    );
}

/// Narrow S7 — every constraint produced by the in-scope schedule is
/// covered by at least one tamper kind in `TAMPER_COVERAGE`.
///
/// "In-scope" here means RoundPoly per stage 1–7 — the only constraint
/// type whose witnessing taxonomy is implemented in this commit. The
/// constraint enumeration grows as further tamper kinds (T2, T3, …)
/// land in subsequent commits.
#[test]
fn taxonomy_covers_in_scope_constraints() {
    let in_scope: Vec<Constraint> = (1..=7)
        .flat_map(|stage| (0..3).map(move |round| Constraint::RoundPoly { stage, round }))
        .collect();

    let mut uncovered: Vec<Constraint> = Vec::new();
    for c in &in_scope {
        let covered = TAMPER_COVERAGE
            .iter()
            .any(|(_, templates)| templates.iter().any(|t: &ConstraintTemplate| t.matches(c)));
        if !covered {
            uncovered.push(*c);
        }
    }

    assert!(
        uncovered.is_empty(),
        "S7 violation — constraints uncovered by taxonomy: {uncovered:#?}",
    );
}

// ── Helpers ──────────────────────────────────────────────────────────

fn round_poly_tamper(stage: usize, round: usize, coeff: usize) -> TamperPoint {
    TamperPoint {
        stage,
        location: TamperLocation::RoundPolyCoeff { round, coeff },
        mutate: TamperMutation::AddOne,
        witnesses: vec![Constraint::RoundPoly { stage, round }],
        expected: if jolt_equivalence::cross_verifier::is_registered(
            stage,
            TamperKind::T1RoundPolyCoeff,
        ) {
            ExpectedResult::CoreRejectsModularAccepts
        } else {
            ExpectedResult::BothReject
        },
        kind: TamperKind::T1RoundPolyCoeff,
        label: "T1_RoundPolyCoeff",
    }
}

/// Return true if the modular verifier MUST reject this tamper kind
/// regardless of core's outcome. Used by KGC to recognize "BothAccept"
/// as a gap manifestation for tamper kinds where the test
/// infrastructure (proof conversion) bypasses core's view of the
/// tamper.
fn modular_must_reject(kind: TamperKind) -> bool {
    matches!(kind, TamperKind::T2Eval)
}

fn eval_tamper(stage: usize, idx: usize) -> TamperPoint {
    TamperPoint {
        stage,
        location: TamperLocation::Eval { idx },
        mutate: TamperMutation::AddOne,
        witnesses: vec![Constraint::EvalConsistency {
            stage,
            eval_idx: idx,
        }],
        expected: if jolt_equivalence::cross_verifier::is_registered(stage, TamperKind::T2Eval) {
            ExpectedResult::CoreRejectsModularAccepts
        } else {
            ExpectedResult::BothReject
        },
        kind: TamperKind::T2Eval,
        label: "T2_Eval",
    }
}

fn round_poly_degree_tamper(stage: usize, round: usize, kind: DegreeKind) -> TamperPoint {
    TamperPoint {
        stage,
        location: TamperLocation::RoundPolyDegree { round, kind },
        mutate: TamperMutation::AddOne, // unused for degree tamper
        witnesses: vec![Constraint::RoundPoly { stage, round }],
        expected: if jolt_equivalence::cross_verifier::is_registered(
            stage,
            TamperKind::T8RoundPolyDegree,
        ) {
            ExpectedResult::CoreRejectsModularAccepts
        } else {
            ExpectedResult::BothReject
        },
        kind: TamperKind::T8RoundPolyDegree,
        label: "T8_RoundPolyDegree",
    }
}

fn canonical_tamper_for(gap: &KnownGap) -> TamperPoint {
    match gap.kind {
        TamperKind::T1RoundPolyCoeff => {
            round_poly_tamper(gap.stage, /*round*/ 1, /*coeff*/ 0)
        }
        TamperKind::T8RoundPolyDegree => {
            round_poly_degree_tamper(gap.stage, /*round*/ 1, DegreeKind::Truncate)
        }
        TamperKind::T2Eval => eval_tamper(gap.stage, /*idx*/ 0),
        // Other tamper kinds: not implemented in this commit — produce a
        // round-poly tamper as a placeholder. The runner will treat the
        // outcome via the registered exemption.
        _ => round_poly_tamper(gap.stage, 1, 0),
    }
}

/// Pick a "middle" round index for the given stage on the cached
/// fixture. Falls back to round 1 if the stage has too few rounds.
fn middle_round_for(stage: usize, f: &jolt_equivalence::cross_verifier::HonestFixture) -> usize {
    let stage_idx = stage.saturating_sub(1);
    let n = f
        .modular_proof
        .stage_proofs
        .get(stage_idx)
        .map(|sp| sp.round_polys.round_polynomials.len())
        .unwrap_or(0);
    if n >= 4 {
        n / 2
    } else if n >= 2 {
        1
    } else {
        0
    }
}
