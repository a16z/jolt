#![expect(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::print_stdout,
    reason = "manual size and timing gate"
)]

use std::time::Instant;

use bincode::config::standard;
use bincode::serde::encode_to_vec;
use jolt_crypto::Bn254;
use jolt_field::{Field, Fr, One, Ring, Zero};
use jolt_hyperkzg::{HyperKZGScheme, HyperKZGVerifierSetup};
use jolt_poly::{EqPolynomial, MultilinearPoly};
use jolt_wrapper::carry::CarryProver;
use jolt_wrapper::stream::{
    combine_packed_phases, commit_packed, commitment_prefix_challenges, prove_assembly,
    verify_assembly_with_cost, AffineForm, AssemblyMemberStatement, AssemblyStatement, Column,
    ColumnId, CommitmentPhase, StageMember, StageMemberSpec, StageProof, Term, TermContext,
    TermExporter, TermObserver, VerifierCost, WrapperProof,
};

const LOG_ROWS: usize = 18;
const ROWS: usize = 1 << LOG_ROWS;
const K: usize = 16;
const TERM_COUNT: usize = 600;
const IO_BYTES: usize = 608;

struct RepeatedTerms {
    source_point: Vec<Fr>,
    coefficient_scale: Fr,
}

impl TermExporter for RepeatedTerms {
    fn terms(&self, context: &TermContext<'_>) -> Vec<Term> {
        let coefficient = context.batching_coefficients[0]
            * EqPolynomial::<Fr>::mle(&self.source_point, context.row_point)
            * self.coefficient_scale;
        repeated_terms(coefficient)
    }

    fn terms_observed(
        &self,
        context: &TermContext<'_>,
        observer: &mut dyn TermObserver,
    ) -> Vec<Term> {
        let eq = eq_observed(&self.source_point, context.row_point, observer);
        let coefficient = observer.fr_mul(context.batching_coefficients[0], eq);
        let coefficient = observer.fr_mul(coefficient, self.coefficient_scale);
        repeated_terms(coefficient)
    }
}

fn repeated_terms(coefficient: Fr) -> Vec<Term> {
    (0..TERM_COUNT)
        .map(|_| Term {
            coefficient,
            factors: vec![
                AffineForm {
                    constant: Fr::zero(),
                    weights: vec![(ColumnId { group: 0, slot: 0 }, Fr::one())],
                },
                AffineForm {
                    constant: Fr::one(),
                    weights: Vec::new(),
                },
                AffineForm {
                    constant: Fr::one(),
                    weights: Vec::new(),
                },
                AffineForm {
                    constant: Fr::one(),
                    weights: Vec::new(),
                },
                AffineForm {
                    constant: Fr::one(),
                    weights: Vec::new(),
                },
            ],
        })
        .collect()
}

fn eq_observed(left: &[Fr], right: &[Fr], observer: &mut dyn TermObserver) -> Fr {
    left.iter()
        .zip(right)
        .fold(Fr::one(), |result, (&left, &right)| {
            let both = observer.fr_mul(left, right);
            let neither = observer.fr_mul(Fr::one() - left, Fr::one() - right);
            observer.fr_mul(result, both + neither)
        })
}

#[test]
#[ignore = "manual 2^18 term-compression gate"]
fn term_compression_gate() {
    let uptime = std::process::Command::new("uptime")
        .output()
        .expect("uptime")
        .stdout;
    let columns = (0..2 * K)
        .map(|column| {
            Column::Bits(
                (0..ROWS)
                    .map(|row| (mix(row as u64 ^ ((column as u64) << 32)) & 1) as u8)
                    .collect(),
            )
        })
        .collect::<Vec<_>>();
    let first = match &columns[0] {
        Column::Bits(values) => values
            .iter()
            .map(|&value| Fr::from_u64(u64::from(value)))
            .collect::<Vec<_>>(),
        Column::U16(_) | Column::Fr(_) => unreachable!(),
    };
    let source_point = (0..LOG_ROWS)
        .map(|index| Fr::from_u64(index as u64 + 3))
        .collect::<Vec<_>>();
    let input_claim = first.as_slice().evaluate(&source_point);

    let started = Instant::now();
    let setup = HyperKZGScheme::<Bn254>::setup_from_secret(
        Fr::from_u64(97),
        ROWS * K,
        Bn254::g1_generator(),
        Bn254::g2_generator(),
    );
    let setup_ms = started.elapsed().as_millis();
    let started = Instant::now();
    let wire = commit_packed(&columns[..K], K, &setup).expect("wire commitments");
    let wire_ms = started.elapsed().as_millis();
    let phase_challenges =
        commitment_prefix_challenges(&[19; 32], &[Fr::from_u64(23)], &[(&wire.commitments, 2)]);
    let started = Instant::now();
    let helpers = commit_packed(&columns[K..], K, &setup).expect("helper commitments");
    let helper_ms = started.elapsed().as_millis();
    let packed = combine_packed_phases(vec![wire, helpers]).expect("combine phases");
    let statement = AssemblyStatement {
        key_digest: [19; 32],
        public_inputs: vec![Fr::from_u64(23)],
        rows: ROWS,
        column_count: 2 * K,
        k: K,
        members: vec![AssemblyMemberStatement {
            input_claim,
            spec: StageMemberSpec {
                rounds: LOG_ROWS,
                degree: 5,
                offset: 0,
            },
        }],
        commitment_phases: vec![
            CommitmentPhase {
                group_count: 1,
                challenge_count: 2,
            },
            CommitmentPhase {
                group_count: 1,
                challenge_count: 0,
            },
        ],
    };
    let exporter = RepeatedTerms {
        source_point: source_point.clone(),
        coefficient_scale: Fr::from_u64(TERM_COUNT as u64)
            .inverse()
            .expect("nonzero term count"),
    };
    let exporters = [&exporter as &dyn TermExporter];
    let mut carry = CarryProver::new(&first, &source_point, input_claim).expect("carry");
    let mut members = [StageMember {
        prover: &mut carry,
        input_claim,
        degree: 5,
        offset: 0,
    }];
    let started = Instant::now();
    let proof = prove_assembly(&packed, &statement, &mut members, &exporters, &setup)
        .expect("assembly proof");
    let prove_ms = started.elapsed().as_millis();
    let verifier_setup = HyperKZGVerifierSetup::from(&setup);
    let started = Instant::now();
    let (_, cost) = verify_assembly_with_cost(&proof, &statement, &exporters, &verifier_setup)
        .expect("assembly verification");
    let verify_ms = started.elapsed().as_millis();

    let stage_a = committed_stage_bytes(&proof.stages[0]);
    let term_stage = committed_stage_bytes(&proof.stages[1]);
    let term_evaluations = proof.term_evaluations.len() * 32;
    let stage_b = clear_stage_bytes(&proof.stages[2]);
    let commitments = proof.commitments.len() * 32;
    let shared_round_opening = 96 * usize::from(proof.round_opening.is_some());
    let reduced_claim = proof.reduced_claims.len() * 32;
    let opening = proof.payload_bytes()
        - stage_a
        - term_stage
        - term_evaluations
        - stage_b
        - commitments
        - shared_round_opening
        - reduced_claim;
    let bincode = encode_to_vec(&proof, standard()).expect("serialize").len();
    assert_eq!(proof.bincode_bytes(), bincode);
    println!("uptime={}", String::from_utf8_lossy(&uptime).trim());
    println!(
        "rows={ROWS} k={K} T={TERM_COUNT} phase_challenges={} setup={setup_ms}ms wire_commit={wire_ms}ms helper_commit={helper_ms}ms prove={prove_ms}ms verify={verify_ms}ms",
        phase_challenges.len()
    );
    println!(
        "bytes commitments={commitments} stage_a={stage_a} term_stage={term_stage} shared_round_opening={shared_round_opening} ell={term_evaluations} stage_b={stage_b} reduced={reduced_claim} opening={opening} proof={} io={IO_BYTES} total_with_io={} bincode={bincode}",
        proof.payload_bytes(),
        proof.payload_bytes() + IO_BYTES,
    );
    println!("cost={cost:?} gas={}", estimated_gas(cost, &proof));
}

fn committed_stage_bytes(stage: &StageProof) -> usize {
    let committed = stage.committed_rounds.as_ref().expect("committed stage");
    32 * (committed.round_commitments.len() + committed.round_claims.len() + 1)
}

fn clear_stage_bytes(stage: &StageProof) -> usize {
    32 * stage
        .round_polynomials
        .round_polynomials
        .iter()
        .map(|round| round.coeffs_except_linear_term().len())
        .sum::<usize>()
}

fn estimated_gas(cost: VerifierCost, proof: &WrapperProof) -> usize {
    let proof_g1 = proof.commitments.len()
        + proof
            .stages
            .iter()
            .filter_map(|stage| stage.committed_rounds.as_ref())
            .map(|stage| stage.round_commitments.len() + 3 * usize::from(stage.opening.is_some()))
            .sum::<usize>()
        + 3 * usize::from(proof.round_opening.is_some())
        + proof.opening.com.len()
        + 1;
    let calldata = proof.payload_bytes() + 32 * proof_g1 + IO_BYTES;
    21_000
        + 16 * calldata
        + 7_700 * cost.ec_mul
        + 20 * cost.fr_mul
        + batched_inversion_gas(cost.fr_inv)
        + 100 * cost.keccak
        + 2 * 114_700
        + 183_400
}

fn batched_inversion_gas(inversions: usize) -> usize {
    if inversions == 0 {
        return 0;
    }
    let multiplication_complexity = 32usize.div_ceil(8).pow(2);
    let iteration_count = 253;
    let modexp = (multiplication_complexity * iteration_count / 3).max(200);
    modexp + 3 * (inversions - 1) * 20
}

fn mix(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}
