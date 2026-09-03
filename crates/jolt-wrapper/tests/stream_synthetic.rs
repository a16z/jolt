#![expect(
    clippy::indexing_slicing,
    clippy::expect_used,
    reason = "test fixtures use dimensions fixed by local constructors"
)]

use bincode::config::standard;
use bincode::serde::encode_to_vec;
use jolt_crypto::Bn254;
use jolt_field::{Fr, Ring, Zero};
use jolt_hyperkzg::{HyperKZGScheme, HyperKZGVerifierSetup};
use jolt_poly::{BindingOrder, CompressedPoly, Polynomial, UnivariatePoly};
use jolt_sumcheck::prover::ProveRounds;
use jolt_sumcheck::SumcheckError;
use jolt_wrapper::stream::{
    commit_packed, prove_stream, verify_stream, verify_stream_with_cost, Column, PackedPolynomial,
    StageAEncoding, TensorStreamStatement, TensorTerm,
};

struct RowRelation {
    columns: Vec<Polynomial<Fr>>,
    terms: Vec<TensorTerm>,
    rounds: usize,
    claim: Fr,
}

impl RowRelation {
    fn new(columns: Vec<Vec<Fr>>, terms: Vec<TensorTerm>) -> Self {
        let rows = columns[0].len();
        let claim = (0..rows)
            .map(|row| {
                terms
                    .iter()
                    .map(|term| {
                        term.columns
                            .iter()
                            .fold(term.coefficient, |value, &column| {
                                value * columns[column][row]
                            })
                    })
                    .sum::<Fr>()
            })
            .sum();
        Self {
            columns: columns.into_iter().map(Polynomial::new).collect(),
            terms,
            rounds: rows.trailing_zeros() as usize,
            claim,
        }
    }

    fn bind(&mut self, challenge: Fr) {
        for column in &mut self.columns {
            column.bind_with_order(challenge, BindingOrder::HighToLow);
        }
    }
}

impl ProveRounds<Fr> for RowRelation {
    fn num_rounds(&self) -> usize {
        self.rounds
    }

    fn prove_round(
        &mut self,
        bind: Option<Fr>,
        round: usize,
        previous_claim: Fr,
    ) -> Result<UnivariatePoly<Fr>, SumcheckError<Fr>> {
        if let Some(challenge) = bind {
            self.bind(challenge);
        }
        let half = self.columns[0].len() / 2;
        let evaluations: Vec<Fr> = (0..=5)
            .map(|x| {
                let x = Fr::from_u64(x);
                (0..half)
                    .map(|row| {
                        self.terms
                            .iter()
                            .map(|term| {
                                term.columns
                                    .iter()
                                    .fold(term.coefficient, |value, &column| {
                                        value * self.columns[column].sumcheck_round_eval(row, x)
                                    })
                            })
                            .sum::<Fr>()
                    })
                    .sum()
            })
            .collect();
        if evaluations[0] + evaluations[1] != previous_claim {
            return Err(SumcheckError::RoundCheckFailed {
                round,
                expected: previous_claim,
                actual: evaluations[0] + evaluations[1],
            });
        }
        Ok(UnivariatePoly::from_evals(&evaluations))
    }

    fn finish_rounds(&mut self, bind: Fr) -> Result<(), SumcheckError<Fr>> {
        self.bind(bind);
        Ok(())
    }
}

fn columns(rows: usize) -> Vec<Column> {
    let mut columns = Vec::with_capacity(60);
    for column in 0..40 {
        columns.push(Column::Bits(
            (0..rows)
                .map(|row| ((row.wrapping_mul(13) + column * 7) >> (column % 9)) as u8 & 1)
                .collect(),
        ));
    }
    for column in 0..20 {
        columns.push(Column::U16(
            (0..rows)
                .map(|row| row.wrapping_mul(31).wrapping_add(column * 101) as u16)
                .collect(),
        ));
    }
    columns
}

fn dense_columns(columns: &[Column]) -> Vec<Vec<Fr>> {
    columns
        .iter()
        .map(|column| match column {
            Column::Bits(values) => values
                .iter()
                .map(|&value| Fr::from_u64(u64::from(value)))
                .collect(),
            Column::U16(values) => values
                .iter()
                .map(|&value| Fr::from_u64(u64::from(value)))
                .collect(),
            Column::Fr(values) => values.clone(),
        })
        .collect()
}

fn terms() -> Vec<TensorTerm> {
    (0..24)
        .map(|term| TensorTerm {
            coefficient: Fr::from_u64(term as u64 + 1),
            columns: (0..5).map(|factor| (term * 7 + factor * 11) % 60).collect(),
        })
        .collect()
}

#[test]
fn synthetic_stream_round_trip_and_tampers() {
    let rows = 1 << 12;
    let fixture = columns(rows);
    let setup = HyperKZGScheme::<Bn254>::setup_from_secret(
        Fr::from_u64(11),
        rows * 8,
        Bn254::g1_generator(),
        Bn254::g2_generator(),
    );
    let verifier_setup = HyperKZGVerifierSetup::from(&setup);
    let packed = commit_packed(&fixture, 8, &setup).expect("commit columns");
    assert!(packed.polynomials[..5]
        .iter()
        .all(|polynomial| matches!(polynomial, PackedPolynomial::Bits(_))));
    assert!(packed.polynomials[5..]
        .iter()
        .all(|polynomial| matches!(polynomial, PackedPolynomial::U16(_))));
    let relation_terms = terms();
    let mut row_relation = RowRelation::new(dense_columns(&fixture), relation_terms.clone());
    let row_input = row_relation.claim;
    let statement = TensorStreamStatement {
        key_digest: [17; 32],
        rows,
        column_count: fixture.len(),
        k: 8,
        row_input_claim: row_input,
        row_degree: 5,
        stage_a_encoding: StageAEncoding::Compressed,
        terms: relation_terms,
    };
    let proof = prove_stream(&packed, &statement, &mut row_relation, &setup).expect("prove stream");
    let verified = verify_stream(&proof, &statement, &verifier_setup).expect("verify stream");
    assert_eq!(verified.len(), 2);
    let (_, cost) = verify_stream_with_cost(&proof, &statement, &verifier_setup)
        .expect("count verifier operations");
    let hand_traced_fr_mul = 90 + 120 + 2_658 + 7 + 197;
    assert_eq!(cost.fr_mul, hand_traced_fr_mul);
    assert_eq!(cost.fr_inv, 5);
    assert_eq!(
        proof.bincode_bytes(),
        encode_to_vec(&proof, standard())
            .expect("serialize proof")
            .len()
    );

    let mut wrong_digest = statement.clone();
    wrong_digest.key_digest[0] ^= 1;
    assert!(verify_stream(&proof, &wrong_digest, &verifier_setup).is_err());

    let mut stage_claim_tamper = proof.clone();
    stage_claim_tamper.stage_claims[0][0] += Fr::from_u64(1);
    assert!(verify_stream(&stage_claim_tamper, &statement, &verifier_setup).is_err());

    let mut changed_fixture = fixture.clone();
    if let Column::Bits(values) = &mut changed_fixture[0] {
        values[0] ^= 1;
    }
    let changed = commit_packed(&changed_fixture, 8, &setup).expect("changed commitment");
    let mut column_tamper = proof.clone();
    column_tamper.commitments[0] = changed.commitments[0];
    assert!(verify_stream(&column_tamper, &statement, &verifier_setup).is_err());

    let mut swapped_fixture = fixture.clone();
    swapped_fixture.swap(0, 1);
    let swapped = commit_packed(&swapped_fixture, 8, &setup).expect("swapped commitment");
    let mut swapped_columns = proof.clone();
    swapped_columns.commitments = swapped.commitments;
    assert!(verify_stream(&swapped_columns, &statement, &verifier_setup).is_err());

    let mut output_tamper = proof.clone();
    let last_row_round = output_tamper.stages[0]
        .round_polynomials
        .round_polynomials
        .last_mut()
        .expect("row round");
    let mut coefficients = last_row_round.coeffs_except_linear_term().to_vec();
    coefficients[0] += Fr::from_u64(1);
    *last_row_round = CompressedPoly::new(coefficients);
    assert!(verify_stream(&output_tamper, &statement, &verifier_setup).is_err());

    let mut degree_tamper = proof.clone();
    let round = &mut degree_tamper.stages[0].round_polynomials.round_polynomials[0];
    let mut coefficients = round.coeffs_except_linear_term().to_vec();
    coefficients.push(Fr::from_u64(1));
    *round = CompressedPoly::new(coefficients);
    assert!(verify_stream(&degree_tamper, &statement, &verifier_setup).is_err());

    let mut truncated_rounds = proof.clone();
    let _ = truncated_rounds.stages[0]
        .round_polynomials
        .round_polynomials
        .pop();
    assert!(verify_stream(&truncated_rounds, &statement, &verifier_setup).is_err());

    let mut extended_rounds = proof.clone();
    let extra_round = extended_rounds.stages[0]
        .round_polynomials
        .round_polynomials[0]
        .clone();
    extended_rounds.stages[0]
        .round_polynomials
        .round_polynomials
        .push(extra_round);
    assert!(verify_stream(&extended_rounds, &statement, &verifier_setup).is_err());

    let mut group_point_tamper = proof.clone();
    let round = &mut group_point_tamper.stages[1]
        .round_polynomials
        .round_polynomials[0];
    let mut coefficients = round.coeffs_except_linear_term().to_vec();
    coefficients[0] += Fr::from_u64(1);
    *round = CompressedPoly::new(coefficients);
    assert!(verify_stream(&group_point_tamper, &statement, &verifier_setup).is_err());

    let mut polynomial_index_tamper = proof.clone();
    polynomial_index_tamper.commitments.swap(0, 1);
    assert!(verify_stream(&polynomial_index_tamper, &statement, &verifier_setup).is_err());

    let mut tensor_tamper = statement.clone();
    tensor_tamper.terms[0].columns[0] = 1;
    assert!(verify_stream(&proof, &tensor_tamper, &verifier_setup).is_err());

    let mut claim_tamper = proof.clone();
    claim_tamper.reduced_claims[0] += Fr::from_u64(1);
    assert!(verify_stream(&claim_tamper, &statement, &verifier_setup).is_err());

    let mut swapped_claims = proof.clone();
    swapped_claims.stage_claims[0].swap(0, 1);
    assert!(verify_stream(&swapped_claims, &statement, &verifier_setup).is_err());

    let mut extra_commitment = proof.clone();
    extra_commitment.commitments.push(proof.commitments[0]);
    assert!(verify_stream(&extra_commitment, &statement, &verifier_setup).is_err());

    let mut opening_v = proof.clone();
    opening_v.opening.v[0][0] += Fr::from_u64(1);
    assert!(verify_stream(&opening_v, &statement, &verifier_setup).is_err());

    let mut opening_com = proof.clone();
    opening_com.opening.com.swap(0, 1);
    assert!(verify_stream(&opening_com, &statement, &verifier_setup).is_err());

    let mut opening_w = proof.clone();
    opening_w.opening.w = opening_w.opening.com[0];
    assert!(verify_stream(&opening_w, &statement, &verifier_setup).is_err());

    let mut opening_r_squared = proof.clone();
    opening_r_squared.opening.p0_at_r_squared += Fr::from_u64(1);
    assert!(verify_stream(&opening_r_squared, &statement, &verifier_setup).is_err());
}

#[test]
fn committed_stage_a_round_trip_and_tampers() {
    let rows = 1 << 8;
    let fixture = columns(rows);
    let setup = HyperKZGScheme::<Bn254>::setup_from_secret(
        Fr::from_u64(31),
        rows * 8,
        Bn254::g1_generator(),
        Bn254::g2_generator(),
    );
    let verifier_setup = HyperKZGVerifierSetup::from(&setup);
    let packed = commit_packed(&fixture, 8, &setup).expect("commit columns");
    let relation_terms = terms();
    let mut row_relation = RowRelation::new(dense_columns(&fixture), relation_terms.clone());
    let statement = TensorStreamStatement {
        key_digest: [37; 32],
        rows,
        column_count: fixture.len(),
        k: 8,
        row_input_claim: row_relation.claim,
        row_degree: 5,
        stage_a_encoding: StageAEncoding::KzgCommitted,
        terms: relation_terms,
    };
    let proof = prove_stream(&packed, &statement, &mut row_relation, &setup).expect("prove stream");
    let verified = verify_stream(&proof, &statement, &verifier_setup).expect("verify stream");
    assert_eq!(verified.len(), 2);
    assert_eq!(
        proof.bincode_bytes(),
        encode_to_vec(&proof, standard())
            .expect("serialize proof")
            .len()
    );

    let mut commitment_tamper = proof.clone();
    commitment_tamper.stages[0]
        .committed_rounds
        .as_mut()
        .expect("committed stage")
        .round_commitments[0] += Bn254::g1_generator();
    assert!(verify_stream(&commitment_tamper, &statement, &verifier_setup).is_err());

    let mut next_claim_tamper = proof.clone();
    next_claim_tamper.stages[0]
        .committed_rounds
        .as_mut()
        .expect("committed stage")
        .round_evaluations[0][1] += Fr::from_u64(1);
    assert!(verify_stream(&next_claim_tamper, &statement, &verifier_setup).is_err());

    let mut opening_tamper = proof.clone();
    let committed = opening_tamper.stages[0]
        .committed_rounds
        .as_mut()
        .expect("committed stage");
    committed.opening.evaluation_witness = committed.opening.shifted_commitment;
    assert!(verify_stream(&opening_tamper, &statement, &verifier_setup).is_err());

    let mut degree_tamper = proof.clone();
    degree_tamper.stages[0]
        .committed_rounds
        .as_mut()
        .expect("committed stage")
        .opening
        .shifted_commitment += Bn254::g1_generator();
    assert!(verify_stream(&degree_tamper, &statement, &verifier_setup).is_err());

    let mut tensor_rlc_tamper = proof.clone();
    tensor_rlc_tamper.stage_claims[0][0] += Fr::from_u64(1);
    assert!(verify_stream(&tensor_rlc_tamper, &statement, &verifier_setup).is_err());
}

#[test]
fn non_power_of_two_group_padding() {
    let rows = 8;
    let setup = HyperKZGScheme::<Bn254>::setup_from_secret(
        Fr::from_u64(19),
        rows * 8,
        Bn254::g1_generator(),
        Bn254::g2_generator(),
    );
    for column_count in [33, 237] {
        let mut fixture: Vec<Column> = (0..column_count - 1)
            .map(|column| Column::Bits((0..rows).map(|row| ((row + column) & 1) as u8).collect()))
            .collect();
        fixture.push(Column::U16((0..rows).map(|row| row as u16 + 1).collect()));
        let packed = commit_packed(&fixture, 8, &setup).expect("commit padded groups");
        assert_eq!(packed.layout.group_count, column_count.div_ceil(8));
        assert!(packed.layout.padded_group_count.is_power_of_two());
        assert_eq!(
            packed.layout.padded_column_count,
            packed.layout.padded_group_count * 8
        );
        let values = packed
            .column_evaluations(&[Fr::zero(); 3])
            .expect("column evaluations");
        assert!(values[column_count..].iter().all(Fr::is_zero));

        let missing_group = packed.layout.padded_group_count - 1;
        let mut column_point = boolean_point(missing_group, packed.layout.group_vars());
        column_point.extend(boolean_point(0, packed.layout.slot_vars()));
        let weights = packed
            .layout
            .group_weights(&column_point)
            .expect("group weights");
        assert!(weights.iter().all(Fr::is_zero));
    }
}

fn boolean_point(index: usize, variables: usize) -> Vec<Fr> {
    (0..variables)
        .map(|bit| Fr::from_u64(((index >> (variables - bit - 1)) & 1) as u64))
        .collect()
}
