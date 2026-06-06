use jolt_claims::protocols::wrapper_spartan_hyperkzg::WRAPPER_WITNESS_COMMITMENT_TRANSCRIPT_LABEL;
use jolt_crypto::PairingGroup;
#[cfg(feature = "zk")]
use jolt_crypto::{HomomorphicCommitment, VectorCommitment};
use jolt_field::Field;
#[cfg(feature = "zk")]
use jolt_field::{RingAccumulator, WithAccumulator};
#[cfg(feature = "zk")]
use jolt_hyperkzg::{
    HyperKZGCommitment, HyperKZGProof as HyperKZGOpeningProof, HyperKZGScheme,
    HyperKZGVerifierSetup,
};
#[cfg(feature = "zk")]
use jolt_openings::CommitmentScheme;
use jolt_transcript::{AppendToTranscript, Label, Transcript};

use crate::{
    stages::{hyperkzg, r1cs_relation, spartan},
    validate_proof_config, Error, WrapperProof, WrapperVerifierConfig,
};

#[cfg(feature = "zk")]
use crate::{validate_zk_proof_config, WrapperZkProof, WrapperZkVerifierConfig};

#[cfg(feature = "zk")]
use crate::stages::zk::blindfold;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CheckedInputs {
    pub relation_variables: usize,
    pub relation_constraints: usize,
    pub public_inputs: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct WrapperVerifierInputs<'a, F: Field> {
    pub public_inputs: &'a [F],
}

pub fn verify<P, T>(
    config: &WrapperVerifierConfig<P>,
    inputs: WrapperVerifierInputs<'_, P::ScalarField>,
    proof: &WrapperProof<P>,
) -> Result<(), Error>
where
    P: PairingGroup,
    P::ScalarField: Field + AppendToTranscript,
    P::G1: AppendToTranscript,
    T: Transcript<Challenge = P::ScalarField>,
{
    validate_proof_config(config, proof)?;

    let checked = validate_inputs(config, inputs);

    let mut transcript = T::new(config.transcript_label);

    let relation = r1cs_relation::verify(
        r1cs_relation::R1csRelationInputs {
            checked: &checked,
            relation: &config.key.relation,
            public_inputs: inputs.public_inputs,
            proof_relation: proof.relation,
        },
        &mut transcript,
    )?;
    absorb_witness_commitment(proof, &mut transcript);
    let spartan = spartan::verify(
        spartan::SpartanInputs {
            checked: &checked,
            proof,
            deps: spartan::deps(&relation),
        },
        &mut transcript,
    )?;

    let _hyperkzg = hyperkzg::verify(
        hyperkzg::HyperKzgInputs {
            checked: &checked,
            setup: &config.key.hyperkzg,
            proof,
            deps: hyperkzg::deps(&relation, &spartan),
        },
        &mut transcript,
    )?;

    Ok(())
}

#[cfg(feature = "zk")]
pub fn verify_zk<P, VC, T>(
    config: &WrapperZkVerifierConfig<P, VC>,
    inputs: WrapperVerifierInputs<'_, P::ScalarField>,
    proof: &WrapperZkProof<P, VC>,
) -> Result<(), Error>
where
    P: PairingGroup,
    P::ScalarField: Field + AppendToTranscript,
    P::G1: AppendToTranscript,
    VC: VectorCommitment<Field = P::ScalarField>,
    VC::Output: Copy + HomomorphicCommitment<P::ScalarField> + AppendToTranscript,
    T: Transcript<Challenge = P::ScalarField>,
    HyperKZGScheme<P>: jolt_crypto::Commitment<Output = HyperKZGCommitment<P>>
        + CommitmentScheme<
            Field = P::ScalarField,
            Proof = HyperKZGOpeningProof<P>,
            VerifierSetup = HyperKZGVerifierSetup<P>,
        > + jolt_openings::ZkOpeningScheme<HidingCommitment = VC::Output>,
    <P::ScalarField as WithAccumulator>::Accumulator: RingAccumulator<Element = P::ScalarField>,
{
    validate_zk_proof_config(config, proof)?;

    let checked = validate_zk_inputs(config, inputs);

    let mut transcript = T::new(config.transcript_label);

    let relation = r1cs_relation::verify(
        r1cs_relation::R1csRelationInputs {
            checked: &checked,
            relation: &config.key.relation,
            public_inputs: inputs.public_inputs,
            proof_relation: proof.relation,
        },
        &mut transcript,
    )?;
    absorb_zk_witness_commitment(proof, &mut transcript);
    let spartan = spartan::verify_zk::<P, VC, T>(
        spartan::SpartanZkInputs {
            checked: &checked,
            proof,
            vc_setup: &config.vc_setup,
            deps: spartan::deps(&relation),
        },
        &mut transcript,
    )?;

    let hyperkzg = hyperkzg::verify_zk::<P, VC, T>(
        hyperkzg::HyperKzgZkInputs {
            checked: &checked,
            setup: &config.key.hyperkzg,
            proof,
            deps: hyperkzg::zk_deps(&relation, &spartan),
        },
        &mut transcript,
    )?;

    let blindfold = blindfold::build(&spartan, &hyperkzg)?;
    jolt_blindfold::verify::<P::ScalarField, VC, T>(
        &blindfold.protocol,
        &proof.blindfold,
        &config.vc_setup,
        &mut transcript,
    )
    .map_err(|error| Error::BlindFoldVerificationFailed {
        reason: error.to_string(),
    })?;

    Ok(())
}

fn validate_inputs<P: PairingGroup>(
    config: &WrapperVerifierConfig<P>,
    inputs: WrapperVerifierInputs<'_, P::ScalarField>,
) -> CheckedInputs {
    CheckedInputs {
        relation_variables: config.key.relation.num_vars,
        relation_constraints: config.key.relation.num_constraints,
        public_inputs: inputs.public_inputs.len(),
    }
}

#[cfg(feature = "zk")]
fn validate_zk_inputs<P, VC>(
    config: &WrapperZkVerifierConfig<P, VC>,
    inputs: WrapperVerifierInputs<'_, P::ScalarField>,
) -> CheckedInputs
where
    P: PairingGroup,
    VC: VectorCommitment<Field = P::ScalarField>,
{
    CheckedInputs {
        relation_variables: config.key.relation.num_vars,
        relation_constraints: config.key.relation.num_constraints,
        public_inputs: inputs.public_inputs.len(),
    }
}

fn absorb_witness_commitment<P, T>(proof: &WrapperProof<P>, transcript: &mut T)
where
    P: PairingGroup,
    P::G1: AppendToTranscript,
    T: Transcript,
{
    transcript.append(&Label(WRAPPER_WITNESS_COMMITMENT_TRANSCRIPT_LABEL));
    transcript.append(&proof.hyperkzg.witness_commitment);
}

#[cfg(feature = "zk")]
fn absorb_zk_witness_commitment<P, VC, T>(proof: &WrapperZkProof<P, VC>, transcript: &mut T)
where
    P: PairingGroup,
    P::G1: AppendToTranscript,
    VC: VectorCommitment<Field = P::ScalarField>,
    T: Transcript,
{
    transcript.append(&Label(WRAPPER_WITNESS_COMMITMENT_TRANSCRIPT_LABEL));
    transcript.append(&proof.hyperkzg.witness_commitment);
}
