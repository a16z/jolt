#![expect(
    dead_code,
    reason = "shared integration-test support is compiled independently per test file"
)]

use jolt_akita::{
    AkitaBlackBoxBatchStatement, AkitaBlackBoxBatchWitness, AkitaCommitment, AkitaField,
    AkitaProverHint, AkitaScheme, AkitaSetupParams,
};
use jolt_openings::{
    CommitmentScheme, EvaluationClaim, OpeningsError, PrefixPackedClaim, PrefixPackedProverSetup,
    PrefixPackedVerifierSetup, PrefixPacking, VerifierOpeningClaim,
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
}

pub fn black_box_setup() -> (
    <AkitaScheme as CommitmentScheme>::ProverSetup,
    <AkitaScheme as CommitmentScheme>::VerifierSetup,
) {
    setup_for(4, 2, layout(7))
}

pub fn packed_setup<Id: Clone>(
    packing: PrefixPacking<Id>,
    layout_digest: [u8; 32],
) -> (
    PrefixPackedProverSetup<AkitaScheme, Id>,
    PrefixPackedVerifierSetup<AkitaScheme, Id>,
) {
    let (prover_pcs, verifier_pcs) = setup_for(packing.packed_num_vars, 1, layout_digest);
    (
        PrefixPackedProverSetup {
            pcs: prover_pcs,
            packing: packing.clone(),
        },
        PrefixPackedVerifierSetup {
            pcs: verifier_pcs,
            packing,
        },
    )
}

pub fn black_box_statement(
    commitment: AkitaCommitment,
    point: &[AkitaField],
    evaluations: impl IntoIterator<Item = AkitaField>,
) -> AkitaBlackBoxBatchStatement {
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
) -> AkitaBlackBoxBatchStatement {
    black_box_statement(commitment, point, [eval])
}

pub fn batch_witness<'a>(
    polynomials: impl IntoIterator<Item = &'a Polynomial<AkitaField>>,
    hint: AkitaProverHint,
) -> AkitaBlackBoxBatchWitness<'a> {
    (
        polynomials
            .into_iter()
            .map(|polynomial| polynomial as &(dyn MultilinearPoly<AkitaField> + 'a))
            .collect(),
        hint,
    )
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
        let offset = prefix_index(&slot.prefix) << slot.num_vars;
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
) -> Vec<PrefixPackedClaim<AkitaField, Id>>
where
    Id: Copy + Ord,
{
    polynomials
        .iter()
        .map(|(id, polynomial)| {
            let logical_point = packing
                .logical_point(id, packed_point)
                .expect("packed point should produce logical suffix");
            PrefixPackedClaim::new(
                *id,
                EvaluationClaim::new(logical_point.clone(), polynomial.evaluate(&logical_point)),
            )
        })
        .collect()
}

pub fn run_on_large_stack(test: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(test)
        .expect("test thread should spawn")
        .join()
        .expect("test thread should complete");
}

fn prefix_index(prefix: &[bool]) -> usize {
    prefix
        .iter()
        .fold(0usize, |acc, bit| (acc << 1) | usize::from(*bit))
}
