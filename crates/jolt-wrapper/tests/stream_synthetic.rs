#![expect(
    clippy::indexing_slicing,
    clippy::expect_used,
    reason = "test fixtures use dimensions fixed by local constructors"
)]

use jolt_crypto::Bn254;
use jolt_field::{Fr, Ring, Zero};
use jolt_hyperkzg::{HyperKZGScheme, HyperKZGVerifierSetup};
use jolt_poly::{BindingOrder, CompressedPoly, Polynomial, UnivariatePoly};
use jolt_sumcheck::prover::ProveRounds;
use jolt_sumcheck::SumcheckError;
use jolt_wrapper::stream::{
    commit_packed, new_stream_transcript, prove_reduced_opening, prove_stage, verify_stream,
    Column, ColumnBatching, ReductionClaim, ReductionClaimRef, StageClaims, StageMember,
    StageMemberSpec, TensorTerm,
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

    fn finals(&self) -> Vec<Fr> {
        self.columns
            .iter()
            .map(|column| column.evals()[0])
            .collect()
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
    let relation_terms = terms();
    let mut transcript = new_stream_transcript(&packed.commitments);

    let mut row_relation = RowRelation::new(dense_columns(&fixture), relation_terms.clone());
    let row_input = row_relation.claim;
    let mut row_members = [StageMember {
        prover: &mut row_relation,
        input_claim: row_input,
        degree: 5,
        offset: 0,
    }];
    let (row_proof, row_result) =
        prove_stage(&mut row_members, &mut transcript).expect("row stage");
    let mut column_values = row_relation.finals();
    column_values.resize(64, Fr::zero());
    let mut column_batch =
        ColumnBatching::new(column_values, relation_terms.clone()).expect("column stage");
    let column_input = column_batch.input_claim();
    assert_eq!(row_result.output_claims, [column_input]);
    let mut column_members = [StageMember {
        prover: &mut column_batch,
        input_claim: column_input,
        degree: 2,
        offset: 0,
    }];
    let (column_proof, column_result) =
        prove_stage(&mut column_members, &mut transcript).expect("column stage");
    let column_expected = ColumnBatching::expected_final(
        64,
        &relation_terms,
        &column_result.point,
        &column_batch.final_evaluations(),
    )
    .expect("column final claim");
    assert_eq!(column_result.output_claims, [column_expected]);

    let column_vars = 64usize.trailing_zeros() as usize;
    let group_vars = packed.commitments.len().trailing_zeros() as usize;
    let slot_vars = packed.k.trailing_zeros() as usize;
    assert_eq!(column_vars, group_vars + slot_vars);
    let mut claims = Vec::new();
    for column_point in column_result.point.chunks_exact(column_vars) {
        let slot_point = &column_point[group_vars..];
        let packed_point = packed
            .point(&row_result.point, slot_point)
            .expect("packed point");
        for (polynomial, evaluations) in packed.evaluations.iter().enumerate() {
            claims.push(ReductionClaim {
                polynomial,
                point: packed_point.clone(),
                value: Polynomial::new(evaluations.clone()).evaluate(&packed_point),
            });
        }
    }
    let claim_refs: Vec<ReductionClaimRef> = claims
        .iter()
        .map(|claim| ReductionClaimRef {
            polynomial: claim.polynomial,
            point: claim.point.clone(),
        })
        .collect();
    let proof = prove_reduced_opening(
        &packed,
        vec![row_proof, column_proof],
        claims,
        &setup,
        &mut transcript,
    )
    .expect("reduction and opening");
    let shapes = vec![
        vec![StageMemberSpec {
            rounds: 12,
            degree: 5,
            offset: 0,
        }],
        vec![StageMemberSpec {
            rounds: 5 * column_vars,
            degree: 2,
            offset: 0,
        }],
    ];
    let stage_claims = vec![
        StageClaims {
            input: vec![row_input],
            output: row_result.output_claims,
        },
        StageClaims {
            input: vec![column_input],
            output: column_result.output_claims,
        },
    ];
    let verified = verify_stream(&proof, &shapes, &stage_claims, &claim_refs, &verifier_setup)
        .expect("verify stream");
    assert_eq!(verified.len(), 3);

    let mut changed_fixture = fixture.clone();
    if let Column::Bits(values) = &mut changed_fixture[0] {
        values[0] ^= 1;
    }
    let changed = commit_packed(&changed_fixture, 8, &setup).expect("changed commitment");
    let mut column_tamper = proof.clone();
    column_tamper.commitments[0] = changed.commitments[0];
    assert!(verify_stream(
        &column_tamper,
        &shapes,
        &stage_claims,
        &claim_refs,
        &verifier_setup,
    )
    .is_err());

    let mut round_tamper = proof.clone();
    let round = &mut round_tamper.stages[0].round_polynomials.round_polynomials[0];
    let mut coefficients = round.coeffs_except_linear_term().to_vec();
    coefficients[0] += Fr::from_u64(1);
    *round = CompressedPoly::new(coefficients);
    assert!(verify_stream(
        &round_tamper,
        &shapes,
        &stage_claims,
        &claim_refs,
        &verifier_setup,
    )
    .is_err());

    let mut claim_tamper = proof.clone();
    claim_tamper.reduced_claims[0] += Fr::from_u64(1);
    assert!(verify_stream(
        &claim_tamper,
        &shapes,
        &stage_claims,
        &claim_refs,
        &verifier_setup,
    )
    .is_err());
}
