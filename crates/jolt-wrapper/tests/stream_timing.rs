#![expect(
    clippy::expect_used,
    clippy::indexing_slicing,
    reason = "the ignored benchmark uses dimensions fixed by its fixture"
)]

use std::fmt::Write;
use std::time::Instant;

use bincode::config::standard;
use bincode::serde::encode_to_vec;
use jolt_crypto::Bn254;
use jolt_field::{Fr, Ring};
use jolt_hyperkzg::{HyperKZGScheme, HyperKZGVerifierSetup};
use jolt_poly::{BindingOrder, Polynomial, UnivariatePoly};
use jolt_sumcheck::prover::ProveRounds;
use jolt_sumcheck::SumcheckError;
use jolt_wrapper::stream::{
    commit_packed, prove_stream, verify_stream, Column, StageAEncoding, TensorStreamStatement,
    TensorTerm, WrapperProof,
};
use rayon::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct VerifierCost {
    ec_mul: usize,
    ec_add: usize,
    pairing_pairs: usize,
    fr_ops: usize,
    keccak: usize,
}

impl VerifierCost {
    fn stream(proof: &WrapperProof, statement: &TensorStreamStatement) -> Self {
        let committed = proof.stages[0]
            .committed_rounds
            .as_ref()
            .expect("timing gate uses committed stage A");
        let row_rounds = committed.round_commitments.len();
        let column_rounds = proof.stages[1].round_polynomials.round_polynomials.len();
        let factors = proof.stage_claims[0].len();
        let ell = proof.opening.com.len() + 1;
        let groups = proof.commitments.len();
        let compressed_keccak = |stage: usize| {
            proof.stages[stage]
                .round_polynomials
                .round_polynomials
                .iter()
                .map(|round| round.coeffs_except_linear_term().len() + 2)
                .sum::<usize>()
        };
        let prefix_keccak = groups + 4;
        let stage_a_keccak = 2 + 7 * row_rounds + 6;
        let stage_b_keccak = 5 * factors + compressed_keccak(1);
        let reduced_claim_keccak = proof.reduced_claims.len();
        let opening_keccak =
            proof.opening.com.len() + proof.opening.v.iter().map(Vec::len).sum::<usize>() + 3;
        let ec_mul = 3 * row_rounds + 2 + groups + ell + 6;
        let ec_add = ec_mul - 1;

        let column_vars = statement.column_count.next_power_of_two().trailing_zeros() as usize;
        let padded_groups = groups.next_power_of_two();
        let fr_ops = 25 * row_rounds
            + 12 * column_rounds
            + factors * (4 * column_vars + 1)
            + 3 * padded_groups
            + 15 * ell
            + 30;
        Self {
            ec_mul,
            ec_add,
            pairing_pairs: 8,
            fr_ops,
            keccak: prefix_keccak
                + stage_a_keccak
                + stage_b_keccak
                + reduced_claim_keccak
                + opening_keccak,
        }
    }

    fn estimated_n4_gas(self, proof: &WrapperProof) -> usize {
        let proof_g1 = proof.commitments.len()
            + proof
                .stages
                .iter()
                .filter_map(|stage| stage.committed_rounds.as_ref())
                .map(|stage| stage.round_commitments.len() + 3)
                .sum::<usize>()
            + proof.opening.com.len()
            + 1;
        let evm_calldata_bytes = proof.payload_bytes() + 32 * proof_g1;
        21_000
            + 16 * evm_calldata_bytes
            + 7_700 * self.ec_mul
            + 20 * self.fr_ops
            + 100 * self.keccak
            + 2 * 114_700
            + 183_400
    }
}

struct TimingRow {
    columns: Vec<Polynomial<Fr>>,
    rounds: usize,
    claim: Fr,
}

impl TimingRow {
    fn new(columns: Vec<Vec<Fr>>) -> Self {
        let rows = columns[0].len();
        let claim = (0..rows)
            .into_par_iter()
            .map(|row| columns.iter().map(|column| column[row]).product::<Fr>())
            .sum();
        Self {
            columns: columns.into_iter().map(Polynomial::new).collect(),
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

impl ProveRounds<Fr> for TimingRow {
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
                    .into_par_iter()
                    .map(|row| {
                        self.columns
                            .iter()
                            .map(|column| column.sumcheck_round_eval(row, x))
                            .product::<Fr>()
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

#[test]
#[ignore = "2^17 production stream benchmark"]
fn n3_g_shape_timing() {
    let rows = 1 << 17;
    let bit_columns: Vec<Column> = (0..163)
        .into_par_iter()
        .map(|column| {
            Column::Bits(
                (0..rows)
                    .map(|row| (mix(row as u64 ^ ((column as u64) << 32)) & 1) as u8)
                    .collect(),
            )
        })
        .collect();
    let chunk_columns: Vec<Column> = (0..54)
        .into_par_iter()
        .map(|column| {
            Column::U16(
                (0..rows)
                    .map(|row| mix(row as u64 ^ ((column as u64) << 32)) as u16)
                    .collect(),
            )
        })
        .collect();
    let field_columns: Vec<Column> = (0..20)
        .into_par_iter()
        .map(|column| {
            Column::Fr(
                (0..rows)
                    .map(|row| Fr::from_u64(mix(row as u64 ^ ((column as u64) << 32))))
                    .collect(),
            )
        })
        .collect();
    let mut columns = bit_columns;
    columns.extend(chunk_columns);
    columns.extend(field_columns);
    let row_columns: Vec<Vec<Fr>> = columns[..5]
        .iter()
        .map(|column| match column {
            Column::Bits(values) => values
                .iter()
                .map(|&value| Fr::from_u64(u64::from(value)))
                .collect(),
            Column::U16(_) | Column::Fr(_) => unreachable!(),
        })
        .collect();
    let mut results = String::new();
    for k in [8, 16] {
        let setup_start = Instant::now();
        let setup = HyperKZGScheme::<Bn254>::setup_from_secret(
            Fr::from_u64(23),
            rows * k,
            Bn254::g1_generator(),
            Bn254::g2_generator(),
        );
        let setup_seconds = setup_start.elapsed().as_secs_f64();
        let mut row = TimingRow::new(row_columns.clone());
        let statement = TensorStreamStatement {
            key_digest: [29; 32],
            rows,
            column_count: columns.len(),
            k,
            row_input_claim: row.claim,
            row_degree: 5,
            stage_a_encoding: StageAEncoding::KzgCommitted,
            terms: vec![TensorTerm {
                coefficient: Fr::from_u64(1),
                columns: vec![0, 1, 2, 3, 4],
            }],
        };
        let commit_start = Instant::now();
        let packed = commit_packed(&columns, k, &setup).expect("commit G columns");
        let commit_seconds = commit_start.elapsed().as_secs_f64();
        let prove_start = Instant::now();
        let proof = prove_stream(&packed, &statement, &mut row, &setup).expect("prove G stream");
        let prove_seconds = prove_start.elapsed().as_secs_f64();
        let verifier_setup = HyperKZGVerifierSetup::from(&setup);
        let verify_start = Instant::now();
        let verified = verify_stream(&proof, &statement, &verifier_setup).expect("verify G stream");
        let verify_seconds = verify_start.elapsed().as_secs_f64();
        let bincode_bytes = encode_to_vec(&proof, standard())
            .expect("serialize G proof")
            .len();
        assert_eq!(verified.len(), 2);
        assert_eq!(proof.bincode_bytes(), bincode_bytes);
        let verifier_cost = VerifierCost::stream(&proof, &statement);
        writeln!(
            results,
            "k={k} setup={setup_seconds:.3}s commit={commit_seconds:.3}s prove={prove_seconds:.3}s verify={verify_seconds:.3}s payload={}B bincode={bincode_bytes}B cost={verifier_cost:?} gas={}",
            proof.payload_bytes(),
            verifier_cost.estimated_n4_gas(&proof),
        )
        .expect("write timing line");
    }
    std::fs::write("/tmp/w4s-stream-timing.txt", results).expect("write timing result");
}

fn mix(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}
