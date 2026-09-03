//! The committed adapter from a real Jolt proof to the limb table's inputs:
//! the verifier-key constants, the flattened check, the committed Dory
//! elements (in the joint opening's commitment order) and the wire values
//! the R1CS lane's named Dory scalars take in its witness.

use ark_bn254::Fq12;
use jolt_dory::{DoryCommitment, DoryProof, DoryVerifierSetup};
use jolt_field::Fr;
use jolt_verifier::proof::JoltCommitments;

use super::dory::{DorySetupInputs, DoryWitnessInputs, FlattenedCheck, WireValues};
use crate::relation::{DoryLinks, DoryScalar};

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum AdapterError {
    #[error("the opening proof has {proof} reduction rounds, the relation links {links}")]
    SigmaMismatch { proof: usize, links: usize },
    #[error("wire {wire:?} references variable {index} beyond the {len}-entry witness")]
    WitnessTooShort {
        wire: DoryScalar,
        index: usize,
        len: usize,
    },
    #[error("the relation links {links} commitment weights, the proof carries {commitments} commitments")]
    CommitmentCount { links: usize, commitments: usize },
}

/// Everything the table build needs from one Jolt proof.
pub struct JoltDoryInputs {
    pub setup: DorySetupInputs,
    pub check: FlattenedCheck,
    pub witness: DoryWitnessInputs,
    pub values: WireValues,
}

/// The joint opening's commitment order (`final_opening_polynomial_order`
/// without advice or committed program): `RamInc, RdInc, InstructionRa..,
/// BytecodeRa.., RamRa..`.
pub fn ordered_commitments(commitments: &JoltCommitments<DoryCommitment>) -> Vec<Fq12> {
    let mut out = Vec::with_capacity(
        2 + commitments.instruction_ra.len()
            + commitments.bytecode_ra.len()
            + commitments.ram_ra.len(),
    );
    out.push(Fq12::from(commitments.ram_inc.0));
    out.push(Fq12::from(commitments.rd_inc.0));
    for group in [
        &commitments.instruction_ra,
        &commitments.bytecode_ra,
        &commitments.ram_ra,
    ] {
        out.extend(group.iter().map(|c| Fq12::from(c.0)));
    }
    out
}

/// Builds the table inputs from the verifier setup, the proof's commitments
/// and joint opening proof, the relation's named Dory scalar wires and the
/// witness assignment `z` they index (`z[0] = 1`).
pub fn from_jolt(
    pcs_setup: &DoryVerifierSetup,
    commitments: &JoltCommitments<DoryCommitment>,
    opening_proof: &DoryProof,
    links: &DoryLinks,
    witness_values: &[Fr],
) -> Result<JoltDoryInputs, AdapterError> {
    let proof = opening_proof.0.clone();
    if proof.sigma != links.sigma {
        return Err(AdapterError::SigmaMismatch {
            proof: proof.sigma,
            links: links.sigma,
        });
    }
    let ordered = ordered_commitments(commitments);
    let weights = links
        .scalars
        .iter()
        .filter(|(wire, _)| matches!(wire, DoryScalar::CommitmentWeight(_)))
        .count();
    if weights != ordered.len() {
        return Err(AdapterError::CommitmentCount {
            links: weights,
            commitments: ordered.len(),
        });
    }
    let mut pairs = Vec::with_capacity(links.scalars.len());
    for (wire, variable) in &links.scalars {
        let index = variable.index();
        let value = *witness_values
            .get(index)
            .ok_or(AdapterError::WitnessTooShort {
                wire: wire.clone(),
                index,
                len: witness_values.len(),
            })?;
        pairs.push((wire.clone(), ark_bn254::Fr::from(value)));
    }
    Ok(JoltDoryInputs {
        setup: DorySetupInputs::from(&pcs_setup.0),
        check: FlattenedCheck::derive(proof.sigma, ordered.len()),
        witness: DoryWitnessInputs {
            commitments: ordered,
            proof,
        },
        values: WireValues::from_wires(pairs),
    })
}
