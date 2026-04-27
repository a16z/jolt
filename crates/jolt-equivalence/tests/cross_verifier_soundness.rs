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
        if result.core_rejects_modular_accepts() {
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
