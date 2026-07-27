#![expect(
    dead_code,
    reason = "shared integration-test support is compiled independently per test file"
)]

use jolt_akita::{
    AkitaCommitment, AkitaField, AkitaNativeBatchPolynomials, AkitaNativeBatchStatement,
    AkitaScheme, AkitaSetupParams,
};
use jolt_openings::{
    CommitmentScheme, EvaluationClaim, OpeningsError, PrefixPacking, VerifierOpeningClaim,
};
use jolt_poly::{MultilinearPoly, Polynomial};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MaterializedPackedWitness<Id> {
    pub packing: PrefixPacking<Id>,
    pub polynomial: Polynomial<AkitaField>,
}

pub fn f(value: u64) -> AkitaField {
    AkitaField::from_u64(value)
}

pub fn layout(byte: u8) -> [u8; 32] {
    [byte; 32]
}

pub fn polynomial(num_vars: usize, offset: u64) -> Polynomial<AkitaField> {
    let len = 1usize << num_vars;
    Polynomial::new(
        (0..len)
            .map(|index| f(offset + 1 + 3 * index as u64 + (index as u64 % 5)))
            .collect(),
    )
}

pub fn setup_for(
    num_vars: usize,
    max_num_polys_per_commitment_group: usize,
    layout_digest: [u8; 32],
) -> (
    <AkitaScheme as CommitmentScheme>::ProverSetup,
    <AkitaScheme as CommitmentScheme>::VerifierSetup,
) {
    AkitaScheme::setup(AkitaSetupParams::new(
        num_vars,
        max_num_polys_per_commitment_group,
        layout_digest,
    ))
    .expect("Akita setup should succeed")
}

pub fn native_setup() -> (
    <AkitaScheme as CommitmentScheme>::ProverSetup,
    <AkitaScheme as CommitmentScheme>::VerifierSetup,
) {
    setup_for(13, 2, layout(7))
}

/// PCS setup pair sized for one packed commitment object.
pub fn packed_setup(
    packed_num_vars: usize,
    layout_digest: [u8; 32],
) -> (
    <AkitaScheme as CommitmentScheme>::ProverSetup,
    <AkitaScheme as CommitmentScheme>::VerifierSetup,
) {
    setup_for(packed_num_vars, 1, layout_digest)
}

pub fn native_statement(
    commitment: AkitaCommitment,
    point: &[AkitaField],
    evaluations: impl IntoIterator<Item = AkitaField>,
) -> AkitaNativeBatchStatement {
    evaluations
        .into_iter()
        .map(|evaluation| VerifierOpeningClaim {
            commitment: commitment.clone(),
            evaluation: EvaluationClaim::new(point.to_vec(), evaluation),
        })
        .collect()
}

pub fn single_statement(
    commitment: AkitaCommitment,
    point: &[AkitaField],
    eval: AkitaField,
) -> AkitaNativeBatchStatement {
    native_statement(commitment, point, [eval])
}

pub fn batch_polynomials<'a>(
    polynomials: impl IntoIterator<Item = &'a Polynomial<AkitaField>>,
) -> AkitaNativeBatchPolynomials<'a> {
    polynomials
        .into_iter()
        .map(|polynomial| polynomial as &(dyn MultilinearPoly<AkitaField> + 'a))
        .collect()
}

pub fn materialize_packed<Id>(
    polynomials: &[(Id, Polynomial<AkitaField>)],
) -> Result<MaterializedPackedWitness<Id>, OpeningsError>
where
    Id: Clone + Ord,
{
    let packing = PrefixPacking::new(
        polynomials
            .iter()
            .map(|(id, polynomial)| (id.clone(), polynomial.num_vars())),
    )?;
    let packed_len = 1usize
        .checked_shl(packing.packed_num_vars as u32)
        .ok_or_else(|| {
            OpeningsError::InvalidSetup("packed polynomial domain size overflow".to_owned())
        })?;
    let mut packed_evaluations = vec![f(0); packed_len];

    for (id, polynomial) in polynomials {
        let slot = &packing[id];
        let offset = slot.prefix_index() << slot.num_vars;
        for (local_index, evaluation) in polynomial.evaluations().iter().copied().enumerate() {
            packed_evaluations[offset + local_index] = evaluation;
        }
    }

    Ok(MaterializedPackedWitness {
        packing,
        polynomial: Polynomial::new(packed_evaluations),
    })
}

pub fn packed_claims<Id>(
    polynomials: &[(Id, Polynomial<AkitaField>)],
    packing: &PrefixPacking<Id>,
    packed_point: &[AkitaField],
) -> Vec<(Id, EvaluationClaim<AkitaField>)>
where
    Id: Copy + Ord,
{
    polynomials
        .iter()
        .map(|(id, polynomial)| {
            let logical_point = packing
                .logical_point(id, packed_point)
                .expect("packed point should produce logical suffix");
            (
                *id,
                EvaluationClaim::new(logical_point.clone(), polynomial.evaluate(&logical_point)),
            )
        })
        .collect()
}
