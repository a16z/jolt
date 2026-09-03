use jolt_crypto::Bn254;
use jolt_field::{Fr, Ring, Zero};
use jolt_hyperkzg::{HyperKZGProverSetup, HyperKZGScheme, HyperKZGVerifierSetup, VerifierObserver};
use jolt_openings::AdditivelyHomomorphic;
use jolt_poly::MultilinearPoly;
use jolt_sumcheck::prover::ProveRounds;
use jolt_transcript::{AppendToTranscript, Keccak256Transcript, Transcript};
use serde::{Deserialize, Serialize};

use super::{
    RelationTable, RelationTableError, RelationTableProver, WitnessPart, DEGREE, FIXED_COLUMNS,
    TOTAL_COLUMNS, WIRES, WITNESS_COLUMNS,
};
use crate::stream::{
    commit_packed, prove_kzg_stage, prove_stage, verify_kzg_stage_observed,
    verify_stage_with_observed, ColumnReduction, Commitment, OpeningProof, PackedColumns,
    PackingLayout, StageMember, StageMemberSpec, StageProof, StreamError, VerifierCost,
};

const LABEL: &[u8] = b"jolt-relation-table-v1";

struct CountingTranscript {
    inner: Keccak256Transcript<Fr>,
    hashes: usize,
}

impl Default for CountingTranscript {
    fn default() -> Self {
        Self::new(b"")
    }
}

impl Transcript for CountingTranscript {
    type Challenge = Fr;

    fn new(label: &'static [u8]) -> Self {
        Self {
            inner: Keccak256Transcript::new(label),
            hashes: 1,
        }
    }

    fn append_bytes(&mut self, bytes: &[u8]) {
        self.hashes += 1;
        self.inner.append_bytes(bytes);
    }

    fn challenge(&mut self) -> Fr {
        self.hashes += 1;
        self.inner.challenge()
    }

    fn challenge_scalar(&mut self) -> Fr {
        self.hashes += 1;
        self.inner.challenge_scalar()
    }

    fn state(&self) -> [u8; 32] {
        self.inner.state()
    }
}

pub struct RelationTableProverKey {
    pub table: RelationTable,
    fixed: PackedColumns,
    key_digest: [u8; 32],
}

#[derive(Clone, Debug)]
pub struct RelationTableVerifierKey {
    pub fixed_commitments: Vec<Commitment>,
    pub layout: PackingLayout,
    pub key_digest: [u8; 32],
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RelationTableProof {
    pub wire_commitments: Vec<Commitment>,
    pub helper_commitments: Vec<Commitment>,
    pub row_stage: StageProof,
    pub column_stage: StageProof,
    pub column_claims: Vec<Fr>,
    pub reduced_claim: Fr,
    pub opening: OpeningProof,
}

impl RelationTableProof {
    pub fn payload_bytes(&self) -> usize {
        let row = self.row_stage.committed_rounds.as_ref().map_or(0, |proof| {
            32 * (proof.round_commitments.len() + 3 + 2 * proof.round_evaluations.len())
        });
        let column_scalars = self
            .column_stage
            .round_polynomials
            .round_polynomials
            .iter()
            .map(|round| round.coeffs_except_linear_term().len())
            .sum::<usize>();
        let opening_scalars = self.opening.v.iter().map(Vec::len).sum::<usize>() + 1;
        32 * (self.wire_commitments.len()
            + self.helper_commitments.len()
            + column_scalars
            + self.column_claims.len()
            + 1
            + self.opening.com.len()
            + 1
            + opening_scalars)
            + row
    }
}

pub fn setup(
    table: RelationTable,
    key_digest: [u8; 32],
    packing: usize,
    setup: &HyperKZGProverSetup<Bn254>,
) -> Result<(RelationTableProverKey, RelationTableVerifierKey), RelationTableError> {
    let fixed = commit_packed(&table.fixed_masked_columns(), packing, setup)?;
    if fixed.layout.group_count != 1 {
        return Err(RelationTableError::Claims);
    }
    let layout = PackingLayout::new(table.rows(), 3 * packing, packing)?;
    let verifier = RelationTableVerifierKey {
        fixed_commitments: fixed.commitments.clone(),
        layout,
        key_digest,
    };
    Ok((
        RelationTableProverKey {
            table,
            fixed,
            key_digest,
        },
        verifier,
    ))
}

pub fn prove(
    key: &RelationTableProverKey,
    assignment: &[Fr],
    setup: &HyperKZGProverSetup<Bn254>,
) -> Result<RelationTableProof, RelationTableError> {
    let mut witness = key.table.wire_witness(assignment)?;
    let wire = commit_packed(
        &key.table.masked_columns(&witness, WitnessPart::Wires),
        key.fixed.layout.k,
        setup,
    )?;
    let mut transcript = transcript(&key.key_digest, &key.fixed.commitments);
    absorb_commitments(&wire.commitments, &mut transcript);
    let beta = transcript.challenge_scalar();
    let gamma = transcript.challenge_scalar();
    key.table.add_copy_helpers(&mut witness, beta, gamma)?;
    let helper = commit_packed(
        &key.table.masked_columns(&witness, WitnessPart::Helpers),
        key.fixed.layout.k,
        setup,
    )?;
    absorb_commitments(&helper.commitments, &mut transcript);
    let tau = (0..key.fixed.layout.row_vars())
        .map(|_| transcript.challenge_scalar())
        .collect::<Vec<_>>();
    let relation_weights = std::array::from_fn(|_| transcript.challenge_scalar());
    let packed = phase_packed(&key.fixed, &wire, &helper)?;
    let mut row_prover = RelationTableProver::new(
        &key.table,
        &witness,
        tau.clone(),
        beta,
        gamma,
        relation_weights,
    );
    if !row_prover.input_claim().is_zero() {
        return Err(RelationTableError::Copy);
    }
    let (row_stage, row_result) =
        prove_kzg_stage(&mut row_prover, Fr::zero(), DEGREE, setup, &mut transcript)?;
    let column_claims = packed.column_evaluations(&row_result.point)?;
    let factor_columns = factor_columns(packed.layout.k);
    let actual_claims = factor_columns
        .iter()
        .map(|&column| column_claims[column])
        .collect::<Vec<_>>();
    let expected = RelationTable::final_value(
        key.table.rows(),
        &tau,
        beta,
        gamma,
        relation_weights,
        &row_result.point,
        &actual_claims,
    )?;
    if row_result.final_claim != expected {
        return Err(RelationTableError::Claims);
    }
    let mut reductions = factor_columns
        .iter()
        .map(|&column| ColumnReduction::new(column_claims.clone(), column))
        .collect::<Result<Vec<_>, StreamError>>()?;
    let mut members = reductions
        .iter_mut()
        .zip(&actual_claims)
        .map(|(reduction, &claim)| StageMember {
            prover: reduction as &mut dyn ProveRounds<Fr>,
            input_claim: claim,
            degree: 2,
            offset: 0,
        })
        .collect::<Vec<_>>();
    let (column_stage, column_result) = prove_stage(&mut members, &mut transcript)?;
    drop(members);
    let reduced_claim = reductions
        .first()
        .map(ColumnReduction::final_evaluation)
        .ok_or(RelationTableError::Claims)?;
    let expected_outputs = factor_columns
        .iter()
        .map(|&column| {
            ColumnReduction::expected_final(
                packed.layout.padded_column_count,
                column,
                &column_result.point,
                reduced_claim,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    if column_result.output_claims != expected_outputs {
        return Err(RelationTableError::Claims);
    }
    transcript.append(&reduced_claim);
    let weights = packed.layout.group_weights(&column_result.point)?;
    let point = packed
        .layout
        .packed_point(&row_result.point, &column_result.point)?;
    let combined = packed.rlc_evaluations(&weights)?;
    if combined.as_slice().evaluate(&point) != reduced_claim {
        return Err(RelationTableError::Claims);
    }
    let opening = HyperKZGScheme::<Bn254>::open(setup, &combined, &point, &mut transcript)
        .map_err(StreamError::HyperKzg)?;
    Ok(RelationTableProof {
        wire_commitments: wire.commitments,
        helper_commitments: helper.commitments,
        row_stage,
        column_stage,
        column_claims: actual_claims,
        reduced_claim,
        opening,
    })
}

pub fn verify(
    key: &RelationTableVerifierKey,
    proof: &RelationTableProof,
    setup: &HyperKZGVerifierSetup<Bn254>,
) -> Result<VerifierCost, RelationTableError> {
    if proof.wire_commitments.len() != key.fixed_commitments.len()
        || proof.helper_commitments.len() != key.fixed_commitments.len()
        || proof.column_claims.len() != TOTAL_COLUMNS
    {
        return Err(RelationTableError::Claims);
    }
    let mut transcript = counting_transcript(&key.key_digest, &key.fixed_commitments);
    absorb_commitments(&proof.wire_commitments, &mut transcript);
    let beta = transcript.challenge_scalar();
    let gamma = transcript.challenge_scalar();
    absorb_commitments(&proof.helper_commitments, &mut transcript);
    let tau = (0..key.layout.row_vars())
        .map(|_| transcript.challenge_scalar())
        .collect::<Vec<_>>();
    let relation_weights = std::array::from_fn(|_| transcript.challenge_scalar());
    let commitments = key
        .fixed_commitments
        .iter()
        .chain(&proof.wire_commitments)
        .chain(&proof.helper_commitments)
        .copied()
        .collect::<Vec<_>>();
    let mut cost = VerifierCost::default();
    let row_result = verify_kzg_stage_observed(
        &proof.row_stage,
        Fr::zero(),
        key.layout.row_vars(),
        DEGREE,
        setup,
        &mut transcript,
        &mut cost,
    )?;
    let expected = RelationTable::final_value_observed(
        key.layout.rows,
        &tau,
        beta,
        gamma,
        relation_weights,
        &row_result.point,
        &proof.column_claims,
        &mut cost,
    )?;
    if row_result.final_claim != expected {
        return Err(RelationTableError::Claims);
    }
    let factor_columns = factor_columns(key.layout.k);
    let shape = vec![
        StageMemberSpec {
            rounds: key.layout.column_vars(),
            degree: 2,
            offset: 0,
        };
        TOTAL_COLUMNS
    ];
    let column_result = verify_stage_with_observed(
        &proof.column_stage,
        &shape,
        &proof.column_claims,
        &mut transcript,
        &mut cost,
        |result, observer| {
            factor_columns
                .iter()
                .map(|&column| {
                    ColumnReduction::expected_final_observed(
                        key.layout.padded_column_count,
                        column,
                        &result.point,
                        proof.reduced_claim,
                        observer,
                    )
                })
                .collect()
        },
    )?;
    transcript.append(&proof.reduced_claim);
    let weights = group_weights_observed(key.layout, &column_result.point, &mut cost)?;
    let point = key
        .layout
        .packed_point(&row_result.point, &column_result.point)?;
    let commitment = HyperKZGScheme::<Bn254>::combine(&commitments, &weights);
    cost.ec_mul(commitments.len());
    cost.ec_add(commitments.len());
    HyperKZGScheme::<Bn254>::verify_observed(
        setup,
        &commitment,
        &point,
        &proof.reduced_claim,
        &proof.opening,
        &mut transcript,
        &mut cost,
    )
    .map_err(StreamError::HyperKzg)?;
    cost.keccak = transcript.hashes;
    Ok(cost)
}

fn transcript(key_digest: &[u8; 32], fixed_commitments: &[Commitment]) -> Keccak256Transcript<Fr> {
    let mut transcript = Keccak256Transcript::new(LABEL);
    transcript.append_bytes(key_digest);
    absorb_commitments(fixed_commitments, &mut transcript);
    transcript
}

fn counting_transcript(
    key_digest: &[u8; 32],
    fixed_commitments: &[Commitment],
) -> CountingTranscript {
    let mut transcript = CountingTranscript::new(LABEL);
    transcript.append_bytes(key_digest);
    absorb_commitments(fixed_commitments, &mut transcript);
    transcript
}

fn group_weights_observed<O: VerifierObserver>(
    layout: PackingLayout,
    column_point: &[Fr],
    observer: &mut O,
) -> Result<Vec<Fr>, RelationTableError> {
    if column_point.len() != layout.column_vars() {
        return Err(StreamError::PointDimension {
            expected: layout.column_vars(),
            actual: column_point.len(),
        }
        .into());
    }
    let group_point = &column_point[..layout.group_vars()];
    let mut evaluations = vec![Fr::from_u64(1); layout.padded_group_count];
    let mut size = 1;
    for &challenge in group_point {
        size *= 2;
        for index in (0..size).rev().step_by(2) {
            let scalar = evaluations[index / 2];
            evaluations[index] = observer.fr_mul(scalar, challenge);
            evaluations[index - 1] = scalar - evaluations[index];
        }
    }
    evaluations.truncate(layout.group_count);
    Ok(evaluations)
}

fn absorb_commitments<T: Transcript<Challenge = Fr>>(
    commitments: &[Commitment],
    transcript: &mut T,
) {
    for commitment in commitments {
        commitment.append_to_transcript(transcript);
    }
}

fn phase_packed(
    fixed: &PackedColumns,
    wire: &PackedColumns,
    helper: &PackedColumns,
) -> Result<PackedColumns, RelationTableError> {
    if fixed.layout != wire.layout || fixed.layout != helper.layout || fixed.layout.group_count != 1
    {
        return Err(RelationTableError::Claims);
    }
    let layout = PackingLayout::new(fixed.layout.rows, 3 * fixed.layout.k, fixed.layout.k)?;
    let polynomials = fixed
        .polynomials
        .iter()
        .chain(&wire.polynomials)
        .chain(&helper.polynomials)
        .cloned()
        .collect();
    let commitments = fixed
        .commitments
        .iter()
        .chain(&wire.commitments)
        .chain(&helper.commitments)
        .copied()
        .collect();
    Ok(PackedColumns {
        layout,
        polynomials,
        commitments,
    })
}

fn factor_columns(k: usize) -> Vec<usize> {
    (0..FIXED_COLUMNS)
        .chain((0..WIRES).map(|wire| k + FIXED_COLUMNS + wire))
        .chain((WIRES..WITNESS_COLUMNS).map(|helper| 2 * k + FIXED_COLUMNS + helper))
        .collect()
}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "tests fail on protocol errors")]
mod tests {
    use jolt_crypto::Bn254;
    use jolt_field::{Fr, Ring};
    use jolt_hyperkzg::{HyperKZGScheme, HyperKZGVerifierSetup};
    use jolt_r1cs::R1csBuilder;

    use super::*;

    #[test]
    fn small_relation_round_trip() {
        let mut builder = R1csBuilder::<Fr>::new();
        let x = builder.alloc(Fr::from_u64(3));
        let y = builder.alloc(Fr::from_u64(9));
        builder.assert_product(x, x, y);
        let witness = builder.witness().expect("assigned witness");
        let matrices = builder.into_matrices();
        let table = RelationTable::new(&matrices, 16).expect("lower relation");
        let pcs_setup = HyperKZGScheme::<Bn254>::setup_from_secret(
            Fr::from_u64(11),
            16 * 16,
            Bn254::g1_generator(),
            Bn254::g2_generator(),
        );
        let verifier_setup = HyperKZGVerifierSetup::from(&pcs_setup);
        let (prover_key, verifier_key) =
            setup(table, [7; 32], 16, &pcs_setup).expect("setup table");
        let proof = prove(&prover_key, &witness, &pcs_setup).expect("prove table");
        let _ = verify(&verifier_key, &proof, &verifier_setup).expect("verify table");
    }
}
