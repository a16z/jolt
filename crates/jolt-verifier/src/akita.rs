//! Prover-facing helpers for assembling Akita verifier artifacts.

use crate::{
    akita_packing::AkitaPackingScheme,
    akita_validation::{
        validate_akita_advice_commitment_aliases,
        validate_akita_packing_opening_proof_payload_shape,
        validate_akita_precommitted_commitment_aliases,
        validate_akita_precommitted_opening_proof_payload_shapes,
        validate_akita_proof_payload_shape, validate_akita_verifier_setup_config,
    },
    config::JoltProtocolConfig,
    preprocessing::JoltVerifierPreprocessing,
    proof::{ClearOnlyVectorCommitment, CommitmentPayload, JoltProof, LatticeCommitmentPayload},
    stages::stage8::{
        lattice_protocol_config_for_packed_witness_layout,
        validate_lattice_packed_witness_layout_config,
    },
    VerifierError,
};
use common::jolt_device::JoltDevice;
use jolt_akita::{AkitaBatchProof, AkitaCommitment, AkitaField, AkitaProverHint};
use jolt_field::{RingAccumulator, WithAccumulator};
use jolt_openings::{
    CommitmentScheme, PackingBatchProof, PackingWitnessLayout, PackingWitnessSource,
};
use jolt_transcript::Transcript;

pub use crate::akita_openings::{
    prove_akita_jolt_final_openings, prove_akita_jolt_final_openings_with_precommitted,
    prove_akita_packing_openings, prove_akita_stage8_clear_openings,
    prove_akita_stage8_clear_openings_with_precommitted, prove_and_attach_akita_opening_proofs,
    prove_and_attach_akita_opening_proofs_with_precommitted, AkitaPrecommittedOpeningInput,
    AkitaStage8ClearOpeningProofs,
};
pub use crate::akita_validity::{
    attach_akita_packing_validity_proof, prove_akita_jolt_packed_validity,
    prove_akita_packing_validity, AkitaPackingValidityProofArtifacts,
};
pub use crate::akita_witness::{build_akita_packing_jolt_witness, AkitaPackingJoltWitnessInput};

pub type AkitaClearVectorCommitment = ClearOnlyVectorCommitment<AkitaField>;
pub type AkitaPackingBatchProof = PackingBatchProof<AkitaBatchProof>;
pub type AkitaPackingProverSetup = <AkitaPackingScheme as CommitmentScheme>::ProverSetup;
pub type AkitaPackingVerifierSetup = <AkitaPackingScheme as CommitmentScheme>::VerifierSetup;
pub type AkitaVerifierPreprocessing =
    JoltVerifierPreprocessing<AkitaPackingScheme, AkitaClearVectorCommitment>;
pub type AkitaJoltProof = JoltProof<AkitaPackingScheme, AkitaClearVectorCommitment>;

#[derive(Clone, Debug)]
pub struct AkitaPackingWitnessArtifacts {
    pub protocol: JoltProtocolConfig,
    pub layout: PackingWitnessLayout,
    pub commitments: CommitmentPayload<AkitaCommitment>,
    pub hint: AkitaProverHint,
}

#[derive(Clone, Debug)]
pub struct AkitaCommittedPackedJoltWitness {
    pub artifacts: AkitaPackingWitnessArtifacts,
    pub witness: jolt_openings::SparsePackingWitness<AkitaField>,
}

impl AkitaPackingWitnessArtifacts {
    pub fn payload(&self) -> Option<&LatticeCommitmentPayload<AkitaCommitment>> {
        self.commitments.as_lattice()
    }
}

pub fn commit_akita_packing_jolt_witness(
    setup: &AkitaPackingProverSetup,
    input: AkitaPackingJoltWitnessInput<'_>,
) -> Result<AkitaCommittedPackedJoltWitness, VerifierError> {
    let witness = build_akita_packing_jolt_witness(input)?;
    let artifacts = commit_akita_packing_witness(setup, &witness)?;
    Ok(AkitaCommittedPackedJoltWitness { artifacts, witness })
}

pub fn commit_akita_packing_witness<S>(
    setup: &AkitaPackingProverSetup,
    source: &S,
) -> Result<AkitaPackingWitnessArtifacts, VerifierError>
where
    S: PackingWitnessSource<AkitaField>,
{
    let protocol = lattice_protocol_config_for_packed_witness_layout(source.layout())?;
    commit_akita_packing_witness_with_config(protocol, setup, source)
}

pub fn commit_akita_packing_witness_with_config<S>(
    protocol: JoltProtocolConfig,
    setup: &AkitaPackingProverSetup,
    source: &S,
) -> Result<AkitaPackingWitnessArtifacts, VerifierError>
where
    S: PackingWitnessSource<AkitaField>,
{
    let layout = source.layout().clone();
    validate_lattice_packed_witness_layout_config(&protocol, &layout)?;
    let (commitment, hint) =
        AkitaPackingScheme::commit_packing_source(setup, source).map_err(|error| {
            VerifierError::LatticePackingCommitmentFailed {
                reason: error.to_string(),
            }
        })?;
    let payload = LatticeCommitmentPayload::new(commitment, layout.digest, layout.dimension);
    crate::proof::validate_lattice_commitment_payload_config(&protocol, &payload)?;

    Ok(AkitaPackingWitnessArtifacts {
        protocol,
        layout,
        commitments: CommitmentPayload::Lattice(payload),
        hint,
    })
}

pub fn verify_akita_clear<T>(
    preprocessing: &AkitaVerifierPreprocessing,
    public_io: &JoltDevice,
    proof: &AkitaJoltProof,
    trusted_advice_commitment: Option<&AkitaCommitment>,
    config: &JoltProtocolConfig,
) -> Result<(), VerifierError>
where
    T: Transcript<Challenge = AkitaField>,
    <AkitaField as WithAccumulator>::Accumulator: RingAccumulator<Element = AkitaField>,
{
    validate_akita_verifier_setup_config(&preprocessing.pcs_setup, config)?;
    validate_akita_proof_payload_shape(&preprocessing.pcs_setup, &proof.commitments)?;
    validate_akita_packing_opening_proof_payload_shape(
        &proof.commitments,
        &proof.joint_opening_proof,
        "lattice joint opening proof",
    )?;
    if let Some(opening_proof) = &proof.lattice_packed_validity_opening_proof {
        validate_akita_packing_opening_proof_payload_shape(
            &proof.commitments,
            opening_proof,
            "lattice packed validity opening proof",
        )?;
    }
    validate_akita_precommitted_opening_proof_payload_shapes(
        &proof.commitments,
        &proof.lattice_precommitted_opening_proofs,
    )?;
    validate_akita_advice_commitment_aliases(
        &proof.commitments,
        proof.untrusted_advice_commitment.as_ref(),
        trusted_advice_commitment,
    )?;
    validate_akita_precommitted_commitment_aliases(
        preprocessing,
        &proof.commitments,
        trusted_advice_commitment,
    )?;
    crate::verifier::verify_clear_with_config::<
        AkitaField,
        AkitaPackingScheme,
        AkitaClearVectorCommitment,
        T,
    >(
        preprocessing,
        public_io,
        proof,
        trusted_advice_commitment,
        config,
    )
}
