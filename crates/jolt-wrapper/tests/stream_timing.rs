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
    commit_packed, prove_stream, verify_stream_with_cost, Column, PackedPolynomial, StageAEncoding,
    TensorStreamStatement, TensorTerm, VerifierCost, WrapperProof,
};
use rayon::prelude::*;

fn estimated_n4_gas(cost: VerifierCost, proof: &WrapperProof) -> usize {
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
    let evm_calldata_bytes = proof.payload_bytes() + 32 * proof_g1;
    21_000
        + 16 * evm_calldata_bytes
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
    let max_length_bytes = 32usize;
    let multiplication_complexity = max_length_bytes.div_ceil(8).pow(2);
    let iteration_count = 254 - 1;
    let modexp_gas = (multiplication_complexity * iteration_count / 3).max(200);
    modexp_gas + 3 * (inversions - 1) * 20
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
            Column::U16(_) | Column::U32(_) | Column::Fr(_) => unreachable!(),
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
        let (verified, verifier_cost) =
            verify_stream_with_cost(&proof, &statement, &verifier_setup).expect("verify G stream");
        let verify_seconds = verify_start.elapsed().as_secs_f64();
        let bincode_bytes = encode_to_vec(&proof, standard())
            .expect("serialize G proof")
            .len();
        assert_eq!(verified.len(), 2);
        assert_eq!(proof.bincode_bytes(), bincode_bytes);
        writeln!(
            results,
            "k={k} setup={setup_seconds:.3}s commit={commit_seconds:.3}s prove={prove_seconds:.3}s verify={verify_seconds:.3}s payload={}B bincode={bincode_bytes}B cost={verifier_cost:?} gas={}",
            proof.payload_bytes(),
            estimated_n4_gas(verifier_cost, &proof),
        )
        .expect("write timing line");
    }
    std::fs::write("/tmp/w4s-stream-timing.txt", results).expect("write timing result");
}

#[test]
#[ignore = "2^21 typed packed-column timing"]
fn typed_column_timing() {
    let rows = 1 << 18;
    let k = 8;
    let mut columns: Vec<Column> = (0..180)
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
    columns.extend(chunk_columns);
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
    columns.extend(field_columns);
    let setup = HyperKZGScheme::<Bn254>::setup_from_secret(
        Fr::from_u64(23),
        rows * k,
        Bn254::g1_generator(),
        Bn254::g2_generator(),
    );
    let packed = commit_packed(&columns, k, &setup).expect("commit columns");
    let point = vec![Fr::from_u64(3); rows.trailing_zeros() as usize];
    let eval_start = Instant::now();
    let evaluations = packed
        .column_evaluations(&point)
        .expect("column evaluations");
    let eval_seconds = eval_start.elapsed().as_secs_f64();
    let weights = (0..packed.layout.group_count)
        .map(|index| Fr::from_u64(index as u64 + 1))
        .collect::<Vec<_>>();
    let rlc_start = Instant::now();
    let rlc = packed.rlc_evaluations(&weights).expect("RLC evaluations");
    let rlc_seconds = rlc_start.elapsed().as_secs_f64();
    let storage_bytes = packed
        .polynomials
        .iter()
        .map(|polynomial| match polynomial {
            PackedPolynomial::Bits(values) => values.len(),
            PackedPolynomial::U16(values) => 2 * values.len(),
            PackedPolynomial::U32(values) => 4 * values.len(),
            PackedPolynomial::Fr(values) => 32 * values.len(),
        })
        .sum::<usize>();
    let dense_storage_bytes = packed.layout.group_count * rows * k * 32;
    assert_eq!(evaluations.len(), packed.layout.padded_column_count);
    assert_eq!(rlc.len(), rows * k);
    std::fs::write(
        "/tmp/w4s-typed-columns.txt",
        format!(
            "rows={rows} k={k} groups={} storage={storage_bytes}B dense_storage={dense_storage_bytes}B column_evaluations={eval_seconds:.6}s rlc={rlc_seconds:.6}s\n",
            packed.layout.group_count,
        ),
    )
    .expect("write timing result");
}

fn mix(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}
