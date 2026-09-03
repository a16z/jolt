//! End to end over a real Dory opening: witness columns, both stage-A
//! members driven with random challenges, the verifier's closed forms
//! against the prover's public columns, the term list against the native
//! final claim, the digit link against the wire values, and tamper rejection.

#![expect(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::print_stdout,
    reason = "test"
)]

mod common;

use std::time::Instant;

use ark_bn254::Fr as ArkFr;
use ark_ff::UniformRand;
use jolt_field::{Fr, Ring, Zero};
use jolt_sumcheck::prover::ProveRounds;
use jolt_wrapper::limb_table::columns::{operand_columns, Columns};
use jolt_wrapper::limb_table::digit_link::{link_term, LinkMember};
use jolt_wrapper::limb_table::dory::{FlattenedCheck, Wire, WireValues};
use jolt_wrapper::limb_table::export::{free_column, pin_columns, ClaimedColumns};
use jolt_wrapper::limb_table::layout::LOG_ROWS;
use jolt_wrapper::limb_table::lookup::{
    omega_column, omega_eval, public_evals, LookupColumns, PublicColumns,
};
use jolt_wrapper::limb_table::relation::{
    eq_tau_column, Challenges, Col, LookupConstants, RowRelation, RowSumcheck, SLOTS,
};
use jolt_wrapper::limb_table::schedule::{build, Layout};
use jolt_wrapper::limb_table::terms::evaluate_terms;
use jolt_wrapper::limb_table::wiring::{copy_kernel_table, fingerprint_columns};
use jolt_wrapper::stream::VerifierCost;
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

fn fr(rng: &mut ChaCha20Rng) -> Fr {
    Fr::from(ArkFr::rand(rng))
}

fn challenges(rng: &mut ChaCha20Rng) -> Challenges {
    Challenges {
        tau: (0..LOG_ROWS).map(|_| fr(rng)).collect(),
        xi: fr(rng),
        alpha: fr(rng),
        gamma: fr(rng),
        lambda: fr(rng),
        beta: fr(rng),
        fp_root: fr(rng),
        fp_combine: fr(rng),
        lambda_lookup: fr(rng),
        copy_root: fr(rng),
        constancy_root: fr(rng),
    }
}

/// Drives a member with the challenges of `rng`: returns the little-endian
/// point and the final claim; every round check `s(0) + s(1) = claim` must hold.
fn drive(member: &mut dyn ProveRounds<Fr>, input: Fr, rng: &mut ChaCha20Rng) -> (Vec<Fr>, Fr) {
    let mut claim = input;
    let mut point = Vec::with_capacity(LOG_ROWS);
    let mut bind = None;
    for round in 0..member.num_rounds() {
        let poly = member.prove_round(bind, round, claim).unwrap();
        assert_eq!(
            poly.evaluate(Fr::zero()) + poly.evaluate(Fr::from_u64(1)),
            claim,
            "round {round} check"
        );
        let r = fr(rng);
        claim = poly.evaluate(r);
        point.push(r);
        bind = Some(r);
    }
    member.finish_rounds(bind.unwrap()).unwrap();
    (point, claim)
}

struct Witness {
    layout: Layout,
    check: FlattenedCheck,
    values: WireValues,
    columns: Columns,
}

fn witness(seed: u64) -> Witness {
    let opening = common::synthetic_opening(8, 5, seed);
    let sigma = opening.statement.challenges.beta.len();
    let n = opening.witness.commitments.len();
    let check = FlattenedCheck::derive(sigma, n);
    let values = WireValues::derive(&opening.statement, sigma, n);
    let layout = build(&check, &values, &opening.setup, &check.wires());
    let coords = opening.witness.coordinates_in(&layout.input_order);
    let v = layout.program.evaluate(&coords).expect("evaluate");
    layout.program.check_pins(&v).expect("pins");
    let columns = Columns::generate(&layout.program, &v, LOG_ROWS);
    Witness {
        layout,
        check,
        values,
        columns,
    }
}

/// Every prover column (claimed then public) for the row member.
fn matrix(
    w: &Witness,
    relation: &RowRelation,
    public: &PublicColumns,
    digits: &[Vec<u8>; 5],
) -> Vec<Vec<Fr>> {
    let ch = &relation.challenges;
    let z_xi = w.columns.xi_values(ch.xi);
    let operands = operand_columns(&w.layout.program, &z_xi, SLOTS);
    let fingerprints = fingerprint_columns(&w.layout.table_reads, &z_xi, relation);
    let lookup = LookupColumns::new(
        public,
        &operands,
        &fingerprints.0,
        &fingerprints.1,
        relation,
    );
    let (helpers, mult) = w.columns.logup_columns(ch.alpha, digits);
    let claimed = ClaimedColumns::assemble(
        &w.columns,
        public,
        operands,
        helpers,
        mult.into_iter()
            .map(|m| Fr::from_u64(u64::from(m)))
            .collect(),
        PublicColumns::inverse_table(ch.alpha),
        lookup,
        fingerprints,
        pin_columns(&w.layout),
        free_column(&w.layout),
    );
    let eq_tau = eq_tau_column(&ch.tau);
    let copy = copy_kernel_table(
        &w.layout.program,
        &public.kinds,
        &w.layout.table_reads,
        &eq_tau,
        relation,
    );
    let constancy = public.constancy_weights(&eq_tau);
    let (small, id) = PublicColumns::small_and_id();
    let mut columns = claimed.columns;
    columns.extend([
        eq_tau,
        copy,
        public.sel.clone(),
        public.is_gt.clone(),
        public.is_g1.clone(),
        public.is_g2.clone(),
        public.s0.clone(),
        public.coord.clone(),
        constancy,
        small,
        id,
    ]);
    assert_eq!(columns.len(), Col::WIDTH);
    columns
}

fn tau_le(ch: &Challenges) -> Vec<Fr> {
    ch.tau.iter().rev().copied().collect()
}

#[test]
fn every_constraint_vanishes_on_an_honest_witness() {
    let w = witness(0xE2E);
    let mut rng = ChaCha20Rng::seed_from_u64(3);
    let ch = challenges(&mut rng);
    let relation = RowRelation::new(
        ch,
        LookupConstants {
            one_row: w.layout.one_cell * 16,
        },
    );
    let public = PublicColumns::new(&w.layout);
    let columns = matrix(&w, &relation, &public, &public.digits.clone());
    let rows = 1usize << LOG_ROWS;
    let mut row_values = vec![Fr::zero(); Col::WIDTH];
    let mut sums: Vec<(&str, Fr)> = Vec::new();
    let mut first_bad: Vec<(&str, usize)> = Vec::new();
    for x in 0..rows {
        for (slot, column) in row_values.iter_mut().zip(&columns) {
            *slot = column[x];
        }
        for (i, (name, value)) in relation
            .constraint_values(&row_values)
            .into_iter()
            .enumerate()
        {
            if !value.is_zero() && !first_bad.iter().any(|(n, _)| *n == name) {
                first_bad.push((name, x));
            }
            if sums.len() <= i {
                sums.push((name, Fr::zero()));
            }
            let _ = i;
        }
        for (name, value) in relation.linear_values(&row_values) {
            match sums.iter_mut().find(|(n, _)| *n == name) {
                Some((_, acc)) => *acc += value,
                None => sums.push((name, value)),
            }
        }
    }
    for (name, row) in &first_bad {
        println!(
            "constraint {name} first violated at row {row} (cell {} c {})",
            row / 16,
            row % 16
        );
    }
    for (name, sum) in &sums {
        if !sum.is_zero() {
            println!("identity {name} does not sum to zero");
        }
    }
    assert!(
        first_bad.is_empty(),
        "row-local constraints violated: {first_bad:?}"
    );
    assert!(
        sums.iter().all(|(_, s)| s.is_zero()),
        "linear identities violated"
    );
}

#[test]
fn members_verify_and_terms_match_on_a_real_opening() {
    let start = Instant::now();
    let w = witness(0xE2E);
    println!("layout + witness {:?}", start.elapsed());
    let mut rng = ChaCha20Rng::seed_from_u64(7);
    let ch = challenges(&mut rng);
    let rho = fr(&mut rng);
    let relation = RowRelation::new(
        ch.clone(),
        LookupConstants {
            one_row: w.layout.one_cell * 16,
        },
    );
    let public = PublicColumns::new(&w.layout);
    let digits = public.digits.clone();
    let start = Instant::now();
    let columns = matrix(&w, &relation, &public, &digits);
    println!("columns {:?}", start.elapsed());

    // Row member.
    let start = Instant::now();
    let mut row = RowSumcheck::new(&relation, &columns);
    assert_eq!(row.input_claim(), Fr::zero(), "honest witness sums to zero");
    let mut driver = ChaCha20Rng::seed_from_u64(99);
    let (r_le, final_claim) = drive(&mut row, Fr::zero(), &mut driver);
    println!("row member {:?}", start.elapsed());
    let claims = row.claims();
    let prover_public = row.public_evals();

    // Verifier closed forms equal the prover's public columns at the point.
    let mut cost = VerifierCost::default();
    let evals = public_evals(&w.layout, &relation, &tau_le(&ch), &r_le, &mut cost);
    assert_eq!(evals, prover_public, "public multilinears at r");
    let terms = relation.terms(&evals);
    let max_degree = terms.iter().map(|t| t.degree()).max().unwrap();
    println!(
        "terms {} max degree {} verifier fr_mul {}",
        terms.len(),
        max_degree,
        cost.fr_mul
    );
    assert_eq!(
        evaluate_terms(&terms, &claims),
        final_claim,
        "terms vs final claim"
    );
    assert!(max_degree <= 5);

    // Digit link at the same point.
    let mut link = LinkMember::new(omega_column(&w.layout, rho), &public.digit_values);
    let input = link.input_claim();
    let mut expected = Fr::zero();
    let mut power = Fr::from_u64(1);
    for wire in w.check.wires() {
        expected += power * Fr::from(w.values.get(&Wire::Named(wire)));
        power *= rho;
    }
    expected += power;
    assert_eq!(input, expected, "digit link input = Σ ρ^k s_k + ρ^K");
    let mut driver = ChaCha20Rng::seed_from_u64(99);
    let (r_link, link_claim) = drive(&mut link, input, &mut driver);
    assert_eq!(r_link, r_le);
    let (digit, omega_final) = link.final_values();
    assert_eq!(digit, claims[Col::D], "digit value claim");
    let mut link_cost = VerifierCost::default();
    let omega = omega_eval(&w.layout, rho, &r_le, &mut link_cost);
    assert_eq!(omega, omega_final, "ω̃(r)");
    assert_eq!(link_term(omega).evaluate(&claims), link_claim, "link term");
    println!("digit link verifier fr_mul {}", link_cost.fr_mul);
}

/// A cheating prover (round checks forced) is caught by the term check.
fn rejects(
    mut columns: Vec<Vec<Fr>>,
    relation: &RowRelation,
    layout: &Layout,
    tamper: impl FnOnce(&mut Vec<Vec<Fr>>),
) {
    tamper(&mut columns);
    let mut row = RowSumcheck::new(relation, &columns);
    row.cheat = true;
    let mut driver = ChaCha20Rng::seed_from_u64(5);
    let (r_le, final_claim) = drive(&mut row, Fr::zero(), &mut driver);
    let claims = row.claims();
    let mut cost = VerifierCost::default();
    let evals = public_evals(
        layout,
        relation,
        &tau_le(&relation.challenges),
        &r_le,
        &mut cost,
    );
    let terms = relation.terms(&evals);
    assert_ne!(
        evaluate_terms(&terms, &claims),
        final_claim,
        "tamper must be rejected"
    );
}

#[test]
fn tampered_witnesses_are_rejected() {
    let w = witness(0xE2E);
    let mut rng = ChaCha20Rng::seed_from_u64(11);
    let ch = challenges(&mut rng);
    let relation = RowRelation::new(
        ch,
        LookupConstants {
            one_row: w.layout.one_cell * 16,
        },
    );
    let public = PublicColumns::new(&w.layout);
    let columns = matrix(&w, &relation, &public, &public.digits.clone());
    let op = w.layout.digit_ops[100];
    let row = op.first_row as usize;
    // A chunk of a compute row (breaks the limb identity and the copies).
    rejects(columns.clone(), &relation, &w.layout, |c| {
        c[Col::CHUNKS + 3][row] += Fr::from_u64(1);
    });
    // A chunk pushed past the 16-bit range table (breaks the range LogUp).
    rejects(columns.clone(), &relation, &w.layout, |c| {
        c[Col::CHUNKS + 5][row] += Fr::from_u64(1 << 16);
    });
    // A wrong digit on one op row (breaks the lookup or the constancy).
    rejects(columns.clone(), &relation, &w.layout, |c| {
        c[Col::E0][row] = Fr::from_u64(1) - c[Col::E0][row];
    });
    // A copied operand that does not match its source.
    rejects(columns.clone(), &relation, &w.layout, |c| {
        c[Col::X + 3][row] += Fr::from_u64(1);
    });
    // A looked-up operand replaced by another value.
    rejects(columns, &relation, &w.layout, |c| {
        c[Col::Y + 3][row] += Fr::from_u64(1);
    });
}
