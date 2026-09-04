//! The committed adapter from a real Jolt proof to the limb table's inputs:
//! the verifier-key constants, the flattened check, the committed Dory
//! elements (in the joint opening's commitment order) and the wire values
//! the R1CS lane's named Dory scalars take in its witness.

use ark_bn254::{Fq12, Fr as ArkFr};
use jolt_dory::{DoryCommitment, DoryProof, DoryVerifierSetup};
use jolt_field::Fr;
use jolt_verifier::proof::JoltCommitments;

use super::dory::{DorySetupInputs, DoryWitnessInputs, FlattenedCheck, WireValues};
use crate::relation::{DoryLinks, DoryScalar};

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum AdapterError {
    /// The relation links a wire set different from the deferred check's bases.
    #[error("the linked Dory wires differ from the deferred check's bases ({links} linked, {check} used)")]
    WireSet { links: usize, check: usize },
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
    /// Digit-base order: the relation's linked wires in `DoryLinks` order
    /// (the R lane's consecutive scalar cells).
    pub wire_order: Vec<DoryScalar>,
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
/// witness assignment `z` they index (`z[0] = 1`), and the wrapper's offset
/// challenge `θ` (drawn from the stream transcript after phase 1a).
pub fn from_jolt(
    pcs_setup: &DoryVerifierSetup,
    commitments: &JoltCommitments<DoryCommitment>,
    opening_proof: &DoryProof,
    links: &DoryLinks,
    witness_values: &[Fr],
    offset: Fr,
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
        pairs.push((wire.clone(), ArkFr::from(value)));
    }
    let check = FlattenedCheck::derive(proof.sigma, ordered.len());
    let wire_order: Vec<DoryScalar> = links.scalars.iter().map(|(w, _)| w.clone()).collect();
    let used = check.wires();
    if wire_order.len() != used.len() || used.iter().any(|w| !wire_order.contains(w)) {
        return Err(AdapterError::WireSet {
            links: wire_order.len(),
            check: used.len(),
        });
    }
    Ok(JoltDoryInputs {
        setup: DorySetupInputs::from(&pcs_setup.0),
        check,
        witness: DoryWitnessInputs {
            commitments: ordered,
            proof,
        },
        values: WireValues::from_wires(pairs, ArkFr::from(offset)),
        wire_order,
    })
}
