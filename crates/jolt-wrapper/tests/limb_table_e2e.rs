//! End to end over a real Dory opening: witness columns, both stage-A
//! members driven with random challenges, the verifier's closed forms
//! against the prover's public columns, the term list against the native
//! final claim, the digit link against the wire values, and tamper rejection.

#![expect(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::print_stdout,
    reason = "test"
)]

mod common;

use std::time::Instant;

use common::Opening;

use ark_bn254::{Fq, Fq12, Fr as ArkFr, G1Affine, G1Projective};
use ark_ec::{AffineRepr, CurveGroup};
use ark_ff::{BigInteger, PrimeField, UniformRand};
use jolt_dory::DoryProof;
use jolt_field::{Field, Fr, Ring, Zero};
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::prover::ProveRounds;
use jolt_sumcheck::SumcheckError;
use jolt_wrapper::limb_table::columns::{
    q_biguint, range_group, Columns, CANON_CHUNKS, GROUP_SIZE, Z_CHUNKS,
};
use jolt_wrapper::limb_table::digit_link::{link_terms, LinkMember};
use jolt_wrapper::limb_table::digits::WINDOW_BOUND;
use jolt_wrapper::limb_table::dory::{
    DoryWitnessInputs, ElementKind, FlattenedCheck, Wire, WireValues,
};
use jolt_wrapper::limb_table::layout::LOG_ROWS;
use jolt_wrapper::limb_table::lookup::{
    digit_bits, link_evals, link_weights, public_evals, PublicColumns,
};
use jolt_wrapper::limb_table::relation::{
    Challenges, Col, LookupConstants, RowRelation, RowSumcheck,
};
use jolt_wrapper::limb_table::schedule::{build, Layout, WINDOW_ROW_BASE};
use jolt_wrapper::limb_table::stream::{link_input_claim, StreamBuilder, StreamWitness};
use jolt_wrapper::limb_table::terms::evaluate_terms;
use jolt_wrapper::stream::VerifierCost;
use num_bigint::{BigInt, BigUint, Sign};
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

/// Drives a member with the challenges of `rng`: returns the point in binding
/// order (big-endian: the members bind the most significant row bit first)
/// and the final claim; every round check `s(0) + s(1) = claim` must hold.
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
    opening: Opening,
}

fn witness(seed: u64) -> Witness {
    witness_with(seed, |_, _| {})
}

/// A witness whose byte-linked input coordinates are altered by `mutate`
/// (pins are not checked: a dishonest input reaches the verifier's checks).
fn witness_with(seed: u64, mutate: impl Fn(&Layout, &mut Vec<Fq>)) -> Witness {
    witness_at(8, 5, seed, mutate)
}

/// [`witness_with`] over a `2^num_vars` opening split into `n` commitments.
fn witness_at(
    num_vars: usize,
    n: usize,
    seed: u64,
    mutate: impl Fn(&Layout, &mut Vec<Fq>),
) -> Witness {
    let opening = common::synthetic_opening(num_vars, n, seed);
    let sigma = opening.statement.challenges.beta.len();
    let n = opening.witness.commitments.len();
    let check = FlattenedCheck::derive(sigma, n);
    let values = WireValues::derive(&opening.statement, sigma, n, common::offset_challenge());
    let layout = build(&check, &values, &opening.setup, &check.wires());
    let mut coords = opening.witness.coordinates_in(&layout.input_order);
    let honest = coords.clone();
    mutate(&layout, &mut coords);
    let v = layout.program.evaluate(&coords).expect("evaluate");
    if coords == honest {
        layout.program.check_pins(&v).expect("pins");
    }
    let columns = Columns::generate(&layout.program, &v, LOG_ROWS);
    Witness {
        layout,
        check,
        values,
        columns,
        opening,
    }
}

/// The stage point in little-endian order (the verifier's kernels).
fn little_endian(point: &[Fr]) -> Vec<Fr> {
    point.iter().rev().copied().collect()
}

/// The prover's witness for `ch` through the staged stream builder (the
/// production path: every phase fed exactly the challenges drawn before it).
fn staged(w: &Witness, ch: &Challenges, packing: usize, group_offset: usize) -> StreamWitness {
    let mut builder = StreamBuilder::new(&w.layout, &w.columns, packing);
    let _ = builder.phase_1b();
    let _ = builder.phase_2a(ch.xi, ch.alpha);
    let _ = builder.phase_2b(ch.fp_root);
    let _ = builder.phase_2c(ch.beta, ch.fp_combine, ch.copy_root, Vec::new());
    builder.finish(
        ch.tau.clone(),
        ch.gamma,
        ch.lambda,
        ch.lambda_lookup,
        ch.constancy_root,
        group_offset,
    )
}

/// Every prover column (claimed then public) for the row member.
fn matrix(w: &Witness, relation: &RowRelation) -> Vec<Vec<Fr>> {
    let witness = staged(w, &relation.challenges, 4, 0);
    (0..Col::WIDTH)
        .map(|column| witness.matrix.field_column(column))
        .collect()
}

/// R's scalar-link claim `Σ_k W_k(ρ)·s_k` over the named wires of `w`.
fn r_link_claim(w: &Witness, rho: Fr) -> Fr {
    let weights = link_weights(&w.layout, rho);
    w.check
        .wires()
        .iter()
        .zip(&weights)
        .map(|(wire, weight)| *weight * Fr::from(w.values.get(&Wire::Named(wire.clone()))))
        .sum()
}

/// The first eight committed chunk columns of `w` (the digit link's window values).
fn window_chunks(w: &Witness) -> Vec<Vec<Fr>> {
    (0..8)
        .map(|j| {
            w.columns
                .chunk_column(j)
                .into_iter()
                .map(|c| Fr::from_u64(u64::from(c)))
                .collect()
        })
        .collect()
}

/// The signed radix-16 recoding of an integer (`digits[i]` weighs `16^i`),
/// the reviewer's alias construction.
fn recode(value: &BigInt) -> [i64; 64] {
    let sixteen = BigInt::from(16);
    let mut value = value.clone();
    let mut digits = [0i64; 64];
    for digit in &mut digits {
        let residue = ((&value % &sixteen) + &sixteen) % &sixteen;
        let residue = i64::try_from(residue).unwrap();
        *digit = if residue >= 8 { residue - 16 } else { residue };
        value = (value - BigInt::from(*digit)) / &sixteen;
    }
    assert!(value.is_zero(), "64 signed digits");
    digits
}

fn fr_from_bigint(value: &BigInt) -> Fr {
    let modulus = BigInt::from_biguint(
        Sign::Plus,
        BigUint::from_bytes_le(&ArkFr::MODULUS.to_bytes_le()),
    );
    let reduced = ((value % &modulus) + &modulus) % &modulus;
    Fr::from(ArkFr::from(reduced.to_biguint().unwrap()))
}

/// A small field element as an integer (chunk values).
fn fr_u64(value: Fr) -> u64 {
    let limbs = ArkFr::from(value).into_bigint().0;
    assert!(limbs[1..].iter().all(|l| *l == 0));
    limbs[0]
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
    let columns = matrix(&w, &relation);
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
    members_accept(&w);
}

/// The member protocol at the fibonacci profile (σ = 11, 42 commitments),
/// the opening the 2^18 layout is sized for, with the tamper suite's
/// representative rejections.
#[test]
fn members_verify_and_tampers_are_rejected_at_the_fibonacci_profile() {
    let start = Instant::now();
    let w = witness_at(22, 42, 0xF1B, |_, _| {});
    println!(
        "σ=11 layout + witness {:?} ({} rows used)",
        start.elapsed(),
        w.layout.used_rows()
    );
    assert_eq!(w.opening.statement.challenges.beta.len(), 11);
    assert_eq!(w.opening.witness.commitments.len(), 42);
    members_accept(&w);
    let mut rng = ChaCha20Rng::seed_from_u64(13);
    let relation = RowRelation::new(
        challenges(&mut rng),
        LookupConstants {
            one_row: w.layout.one_cell * 16,
        },
    );
    let columns = matrix(&w, &relation);
    let row = w.layout.digit_ops[100].first_row as usize;
    rejects(columns.clone(), &relation, &w.layout, |c| {
        c[Col::CHUNKS + 3][row] += Fr::from_u64(1);
    });
    rejects(columns, &relation, &w.layout, |c| {
        c[Col::E0][row] = Fr::from_u64(1) - c[Col::E0][row];
    });
}

/// Independent oracle: the production Dory verifier and the table's pins
/// agree on proofs with one replaced element (a G1 message, a GT message).
#[test]
fn production_verifier_and_pins_agree_on_tampered_proofs() {
    let w = witness(0xE2E);
    let honest = &w.opening.witness.proof;
    assert!(w.opening.verifier.accepts(&DoryProof(honest.clone())));
    let mut g1 = honest.clone();
    g1.vmv_message.e1.0 += G1Projective::from(G1Affine::generator());
    let mut gt = honest.clone();
    gt.first_messages[0].d1_left.0 .0 *= Fq12::from(2u64);
    for tampered in [g1, gt] {
        assert!(!w.opening.verifier.accepts(&DoryProof(tampered.clone())));
        let witness = DoryWitnessInputs {
            commitments: w.opening.witness.commitments.clone(),
            proof: tampered,
        };
        let coords = witness.coordinates_in(&w.layout.input_order);
        let v = w.layout.program.evaluate(&coords).expect("evaluate");
        assert!(w.layout.program.check_pins(&v).is_err(), "pins must fail");
    }
}

/// Both members over `w` with random challenges: the verifier's closed forms
/// against the prover's public columns, the term list against the final
/// claim, and the digit link against the wire values.
fn members_accept(w: &Witness) {
    let mut rng = ChaCha20Rng::seed_from_u64(7);
    let ch = challenges(&mut rng);
    let rho = fr(&mut rng);
    let relation = RowRelation::new(
        ch.clone(),
        LookupConstants {
            one_row: w.layout.one_cell * 16,
        },
    );
    let start = Instant::now();
    let columns = matrix(w, &relation);
    println!("columns {:?}", start.elapsed());

    // Row member.
    let start = Instant::now();
    let mut row = RowSumcheck::new(&relation, &columns);
    assert_eq!(row.input_claim(), Fr::zero(), "honest witness sums to zero");
    let mut driver = ChaCha20Rng::seed_from_u64(99);
    let (point, final_claim) = drive(&mut row, Fr::zero(), &mut driver);
    let r_le = little_endian(&point);
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
    assert_eq!(max_degree, RowRelation::max_factors());

    // Digit link at the same point.
    let mut link = LinkMember::new(
        &w.layout,
        rho,
        &columns[Col::D],
        &columns[Col::CHUNKS..Col::CHUNKS + 8],
    );
    let input = link.input_claim();
    let theta = Fr::from(w.values.get(&Wire::Offset));
    assert_eq!(
        input,
        link_input_claim(r_link_claim(w, rho), rho, theta, &w.layout),
        "digit link input = Σ_k W_k(ρ)·s_k + W_K(ρ) + W_(K+1)(ρ)·θ + window constant"
    );
    let mut driver = ChaCha20Rng::seed_from_u64(99);
    let (r_link, link_claim) = drive(&mut link, input, &mut driver);
    assert_eq!(r_link, point);
    let finals = link.final_values();
    assert_eq!(finals.digit, claims[Col::D], "digit value claim");
    let mut link_cost = VerifierCost::default();
    let evals = link_evals(&w.layout, rho, &r_le, &mut link_cost);
    assert_eq!(evals, finals.evals, "ω̃, κ̃, κ̃' at r");
    assert_eq!(
        evaluate_terms(&link_terms(&evals), &claims),
        link_claim,
        "link terms"
    );
    println!("digit link verifier fr_mul {}", link_cost.fr_mul);
}

/// Review #1 blocker 4: a raw GT input outside the norm-one torus (conjugation
/// is not its inverse) fails the norm-one pin at the verifier.
#[test]
fn non_norm_one_gt_input_is_rejected() {
    use jolt_wrapper::limb_table::tower::fq12_coords;
    let w = witness_with(0xE2E, |layout, coords| {
        let mut offset = 0;
        for element in &layout.input_order {
            let width = element.kind().coords();
            if element.kind() == ElementKind::Gt {
                let mut rng = ChaCha20Rng::seed_from_u64(0xBAD);
                let random = Fq12::rand(&mut rng);
                coords[offset..offset + 12].copy_from_slice(&fq12_coords(&random));
                return;
            }
            offset += width;
        }
        unreachable!("no GT input");
    });
    let mut rng = ChaCha20Rng::seed_from_u64(11);
    let relation = RowRelation::new(
        challenges(&mut rng),
        LookupConstants {
            one_row: w.layout.one_cell * 16,
        },
    );
    let columns = matrix(&w, &relation);
    rejects(columns, &relation, &w.layout, |_| {});
}

/// A cheating prover: every round polynomial is adjusted so the round check
/// `s(0) + s(1) = claim` passes, so a tampered witness can only be caught by
/// the verifier's final relation check.
struct Cheating<'a, P: ProveRounds<Fr>>(&'a mut P);

impl<P: ProveRounds<Fr>> ProveRounds<Fr> for Cheating<'_, P> {
    fn num_rounds(&self) -> usize {
        self.0.num_rounds()
    }

    fn prove_round(
        &mut self,
        bind: Option<Fr>,
        round: usize,
        previous_claim: Fr,
    ) -> Result<UnivariatePoly<Fr>, SumcheckError<Fr>> {
        let mut coefficients = self
            .0
            .prove_round(bind, round, previous_claim)?
            .into_coefficients();
        let tail: Fr = coefficients[1..].iter().fold(Fr::zero(), |acc, c| acc + *c);
        coefficients[0] = (previous_claim - tail) * Field::inverse(&Fr::from_u64(2)).unwrap();
        Ok(UnivariatePoly::new(coefficients))
    }

    fn finish_rounds(&mut self, bind: Fr) -> Result<(), SumcheckError<Fr>> {
        self.0.finish_rounds(bind)
    }
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
    let mut driver = ChaCha20Rng::seed_from_u64(5);
    let (point, final_claim) = drive(&mut Cheating(&mut row), Fr::zero(), &mut driver);
    let claims = row.claims();
    let mut cost = VerifierCost::default();
    let evals = public_evals(
        layout,
        relation,
        &tau_le(&relation.challenges),
        &little_endian(&point),
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
    let columns = matrix(&w, &relation);
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
    rejects(columns.clone(), &relation, &w.layout, |c| {
        c[Col::Y + 3][row] += Fr::from_u64(1);
    });
    let data_row = w.layout.program.input_rows[0] as usize;
    // Review #1 blocker 3: a byte-linked input replaced by `x + q` / `x + 2q`
    // (the same Fq value, another integer encoding) with the best canonicality
    // witness the prover can offer.
    for multiple in [1u64, 2] {
        rejects(columns.clone(), &relation, &w.layout, |c| {
            let x: BigUint = (0..Z_CHUNKS).fold(BigUint::zero(), |acc, i| {
                acc + (BigUint::from(fr_u64(c[Col::CHUNKS + i][data_row])) << (16 * i))
            });
            let alias = x + q_biguint() * multiple;
            for i in 0..Z_CHUNKS {
                let chunk: u64 = ((&alias >> (16 * i)) & BigUint::from(0xffffu32))
                    .try_into()
                    .unwrap();
                c[Col::CHUNKS + i][data_row] = Fr::from_u64(chunk);
            }
            // `d = (q_hi − 1) − alias_hi` is negative: the closest 16-bit witness.
            for i in 0..CANON_CHUNKS {
                c[Col::CHUNKS + Z_CHUNKS + i][data_row] = Fr::zero();
            }
        });
    }
    // Review #1 blocker 1: an out-of-range chunk whose multiplicity and inverse
    // extend the range table above `2^16` (the table is gated to its rows).
    let alpha = relation.challenges.alpha;
    rejects(columns, &relation, &w.layout, |c| {
        let chunk = Z_CHUNKS;
        let invalid = 1usize << 16;
        c[Col::CHUNKS + chunk][data_row] = Fr::from_u64(invalid as u64);
        let group = chunk / GROUP_SIZE;
        let denominator = range_group(group).fold(Fr::from_u64(1), |acc, i| {
            acc * (alpha - c[Col::CHUNKS + i][data_row])
        });
        c[Col::HELPERS + group][data_row] = Field::inverse(&denominator).unwrap();
        c[Col::MULT][0] -= Fr::from_u64(1);
        c[Col::MULT][invalid] += Fr::from_u64(1);
        c[Col::INV][invalid] = Field::inverse(&(alpha - Fr::from_u64(invalid as u64))).unwrap();
    });
}

/// Coordinates of the first byte-linked input of `kind` in the witness order.
fn input_offset(layout: &Layout, kind: ElementKind) -> usize {
    let mut offset = 0;
    for element in &layout.input_order {
        if element.kind() == kind {
            return offset;
        }
        offset += element.kind().coords();
    }
    unreachable!("no input of that kind")
}

/// Milestone 2 negative: a point of the twist outside G2 (on the curve, in
/// the cofactor torsion) as a proof-derived Miller `Q` fails the guarded
/// ψ-chain's negation pins at the verifier.
#[test]
fn twist_point_outside_g2_is_rejected() {
    use ark_bn254::{Fq2, G2Affine};
    use jolt_wrapper::limb_table::dory::InputElement;
    let w = witness_with(0xE2E, |layout, coords| {
        let mut rng = ChaCha20Rng::seed_from_u64(0x7015);
        let outside = loop {
            let x = Fq2::rand(&mut rng);
            if let Some(p) = G2Affine::get_point_from_x_unchecked(x, true) {
                if !p.is_in_correct_subgroup_assuming_on_curve() {
                    break p;
                }
            }
        };
        let mut offset = 0;
        for element in &layout.input_order {
            if *element == InputElement::FinalE2 {
                coords[offset..offset + 4].copy_from_slice(&[
                    outside.x.c0,
                    outside.x.c1,
                    outside.y.c0,
                    outside.y.c1,
                ]);
                return;
            }
            offset += element.kind().coords();
        }
        unreachable!("E2_fin is an input");
    });
    let mut rng = ChaCha20Rng::seed_from_u64(11);
    let relation = RowRelation::new(
        challenges(&mut rng),
        LookupConstants {
            one_row: w.layout.one_cell * 16,
        },
    );
    let columns = matrix(&w, &relation);
    rejects(columns, &relation, &w.layout, |_| {});
}

/// The two small prime factors called out in the BN254 twist cofactor cannot
/// ride into the proof-derived Miller input.
#[test]
fn exact_small_torsion_pairing_inputs_are_rejected() {
    use ark_bn254::{g2::Config as G2Config, Fq2, G2Affine};
    use ark_ec::CurveConfig;
    use jolt_wrapper::limb_table::dory::InputElement;

    let cofactor = G2Config::COFACTOR
        .iter()
        .rev()
        .fold(BigUint::zero(), |acc, limb| (acc << 64) + limb);
    let subgroup_order = BigUint::from_bytes_le(&ArkFr::MODULUS.to_bytes_le());
    for factor in [10_069u64, 5_864_401] {
        let mut rng = ChaCha20Rng::seed_from_u64(factor);
        let quotient = &subgroup_order * (&cofactor / factor);
        let torsion = loop {
            let x = Fq2::rand(&mut rng);
            let Some(point) = G2Affine::get_point_from_x_unchecked(x, true) else {
                continue;
            };
            let point = point.mul_bigint(quotient.to_u64_digits()).into_affine();
            if !point.is_zero() {
                assert!(point.mul_bigint([factor]).is_zero());
                break point;
            }
        };

        let opening = common::synthetic_opening(8, 5, 0xE2E);
        let sigma = opening.statement.challenges.beta.len();
        let n = opening.witness.commitments.len();
        let check = FlattenedCheck::derive(sigma, n);
        let values = WireValues::derive(&opening.statement, sigma, n, common::offset_challenge());
        let layout = build(&check, &values, &opening.setup, &check.wires());
        let mut coords = opening.witness.coordinates_in(&layout.input_order);
        let mut offset = 0;
        for element in &layout.input_order {
            if *element == InputElement::FinalE2 {
                coords[offset..offset + 4].copy_from_slice(&[
                    torsion.x.c0,
                    torsion.x.c1,
                    torsion.y.c0,
                    torsion.y.c1,
                ]);
                break;
            }
            offset += element.kind().coords();
        }
        rejected(&witness_of(opening, check, values, layout, &coords));
    }
}

/// The witness of `coords` for `layout` (pins unchecked: a dishonest input
/// reaches the verifier's checks).
fn witness_of(
    opening: Opening,
    check: FlattenedCheck,
    values: WireValues,
    layout: Layout,
    coords: &[Fq],
) -> Witness {
    let v = layout.program.evaluate(coords).expect("evaluate");
    let columns = Columns::generate(&layout.program, &v, LOG_ROWS);
    Witness {
        layout,
        check,
        values,
        columns,
        opening,
    }
}

/// The term check rejects `w` as committed.
fn rejected(w: &Witness) {
    let mut rng = ChaCha20Rng::seed_from_u64(11);
    let relation = RowRelation::new(
        challenges(&mut rng),
        LookupConstants {
            one_row: w.layout.one_cell * 16,
        },
    );
    let columns = matrix(w, &relation);
    rejects(columns, &relation, &w.layout, |_| {});
}

/// Decision (A) negative: the point `P = −(d − 8)⁻¹·G` that made an affine
/// add exceptional against the former fixed offsets (`R = G`, `Z0 = 2G`)
/// meets θ-randomized offsets: no add of the layout degenerates (the witness
/// evaluates) and the verifier rejects the substituted input.
#[test]
fn point_crafted_for_fixed_offsets_is_rejected() {
    use ark_bn254::{Fr as Scalar, G1Affine};
    use ark_ec::{AffineRepr, CurveGroup};
    use ark_ff::Field as ArkField;
    use jolt_wrapper::limb_table::digits::{digit_value, digits};
    let w = witness_with(0xE2E, |layout, coords| {
        let offset = input_offset(layout, ElementKind::G1);
        let wire = &layout.check.g1_acc.bases[0].1;
        // The first processed window with a nonzero digit (the attack's add).
        let d = digits(layout_scalar(layout, wire))
            .iter()
            .rev()
            .map(|j| digit_value(*j))
            .find(|d| *d != 0)
            .unwrap();
        let inverse = Scalar::from(d.unsigned_abs() as u64).inverse().unwrap();
        let scalar = if d < 0 { inverse } else { -inverse };
        let crafted = (G1Affine::generator() * scalar).into_affine();
        coords[offset] = crafted.x;
        coords[offset + 1] = crafted.y;
    });
    let mut rng = ChaCha20Rng::seed_from_u64(11);
    let relation = RowRelation::new(
        challenges(&mut rng),
        LookupConstants {
            one_row: w.layout.one_cell * 16,
        },
    );
    let columns = matrix(&w, &relation);
    rejects(columns, &relation, &w.layout, |_| {});
}

/// A zero MSM (`A1 = FinalE1 + d·Γ1_0 = 0`, `FinalE1` prover-chosen) makes the
/// chain's last add `acc + entry = 0` for every `θ`: the slope pin has no
/// witness and the verifier rejects, for `θ = 1, 2` as for any other.
#[test]
fn zero_msm_output_is_rejected_for_every_offset_challenge() {
    use jolt_wrapper::limb_table::dory::InputElement;

    for theta in [ArkFr::from(1u64), ArkFr::from(2u64)] {
        let opening = common::synthetic_opening(8, 5, 0xE2E);
        let sigma = opening.statement.challenges.beta.len();
        let n = opening.witness.commitments.len();
        let check = FlattenedCheck::derive(sigma, n);
        let values = WireValues::derive(&opening.statement, sigma, n, theta);
        let layout = build(&check, &values, &opening.setup, &check.wires());
        let mut coords = opening.witness.coordinates_in(&layout.input_order);
        let mut offset = 0;
        for element in &layout.input_order {
            if *element == InputElement::FinalE1 {
                let d = values.get(&check.g1_a1.bases[1].1);
                let zero_sum = (-opening.setup.g1_0.mul_bigint(d.into_bigint())).into_affine();
                coords[offset] = zero_sum.x;
                coords[offset + 1] = zero_sum.y;
                break;
            }
            offset += element.kind().coords();
        }
        rejected(&witness_of(opening, check, values, layout, &coords));
    }
}

/// The scalar of `wire` in the fixture (the digits the layout committed).
fn layout_scalar(layout: &Layout, wire: &Wire) -> ArkFr {
    let _ = layout;
    let opening = common::synthetic_opening(8, 5, 0xE2E);
    let sigma = opening.statement.challenges.beta.len();
    let n = opening.witness.commitments.len();
    WireValues::derive(&opening.statement, sigma, n, common::offset_challenge()).get(wire)
}

/// Milestone 2 sign flags: the committed flag of every byte-linked point is
/// arkworks' canonical sign (`y > −y`, Fq2 lexicographic on `(c1, c0)`), and
/// flipping it — the other decompression of the same `x` — is rejected.
#[test]
fn sign_flags_match_arkworks_and_flips_are_rejected() {
    let w = witness(0xE2E);
    let opening = common::synthetic_opening(8, 5, 0xE2E);
    assert!(!w.layout.sign_rows.is_empty());
    let mut g1_row = None;
    let mut g2_row = None;
    for (element, row) in &w.layout.sign_rows {
        let expected = match element.kind() {
            ElementKind::G1 => {
                let y = opening.witness.g1(*element).y;
                let _ = g1_row.get_or_insert(*row);
                y > -y
            }
            ElementKind::G2 => {
                let y = opening.witness.g2(*element).y;
                let _ = g2_row.get_or_insert(*row);
                y > -y
            }
            ElementKind::Gt => unreachable!("GT elements have no sign"),
        };
        assert_eq!(w.columns.flags[*row as usize] == 1, expected, "{element:?}");
    }
    let mut rng = ChaCha20Rng::seed_from_u64(11);
    let relation = RowRelation::new(
        challenges(&mut rng),
        LookupConstants {
            one_row: w.layout.one_cell * 16,
        },
    );
    let columns = matrix(&w, &relation);
    for row in [g1_row.unwrap(), g2_row.unwrap()] {
        rejects(columns.clone(), &relation, &w.layout, |c| {
            c[Col::FLAG][row as usize] = Fr::from_u64(1) - c[Col::FLAG][row as usize];
        });
    }
}

/// Review #1 blocker 2: the committed phases are contiguous, cover every
/// committed column, and each column's phase is the one whose range holds it.
#[test]
fn phases_commit_values_before_their_challenges() {
    use jolt_wrapper::limb_table::export::{columns, phases};
    let phases = phases();
    let mut next = 0;
    for spec in &phases {
        assert_eq!(spec.columns.start, next, "{:?} is contiguous", spec.phase);
        assert!(!spec.challenges_before.is_empty());
        next = spec.columns.end;
    }
    assert_eq!(next, Col::COMMITTED);
    for column in columns() {
        let range = column.first..column.first + column.count;
        match phases
            .iter()
            .find(|p| p.columns.start <= range.start && range.end <= p.columns.end)
        {
            Some(spec) => assert_eq!(spec.phase, column.phase, "{}", column.name),
            None => assert!(
                range.start >= Col::COMMITTED,
                "{} is a VK column",
                column.name
            ),
        }
    }
}

/// Review #1 blocker 2: the operand collision that keeps a selected row's
/// fingerprint needs `fp_root`; `Y` is committed (phase 2a) before `fp_root`
/// (phase 2b), so the prover can only guess it, and a delta for a guessed root
/// is rejected.
#[test]
fn selected_operand_collision_for_a_guessed_fingerprint_root_is_rejected() {
    use jolt_wrapper::limb_table::wiring::ReadKind;
    let w = witness(0xE2E);
    let mut rng = ChaCha20Rng::seed_from_u64(11);
    let relation = RowRelation::new(
        challenges(&mut rng),
        LookupConstants {
            one_row: w.layout.one_cell * 16,
        },
    );
    let columns = matrix(&w, &relation);
    let guess = fr(&mut ChaCha20Rng::seed_from_u64(0x6E55));
    let weight = |s: usize| (0..s).fold(Fr::from_u64(1), |acc, _| acc * guess);
    let op = w
        .layout
        .digit_ops
        .iter()
        .find(|op| op.kind == ReadKind::Gt)
        .unwrap();
    let row = op.first_row as usize;
    let (a, b, c) = (0, 1, 2);
    let f = [weight(a), weight(b), weight(c)];
    let x = [
        columns[Col::X + a][row],
        columns[Col::X + b][row],
        columns[Col::X + c][row],
    ];
    let delta = [
        f[1] * x[2] - f[2] * x[1],
        f[2] * x[0] - f[0] * x[2],
        f[0] * x[1] - f[1] * x[0],
    ];
    assert!(delta.iter().any(|d| !d.is_zero()));
    rejects(columns, &relation, &w.layout, |cols| {
        for (slot, change) in [a, b, c].into_iter().zip(delta) {
            cols[Col::Y + slot][row] += change;
        }
    });
}

include!("limb_table_e2e/stream.rs");
