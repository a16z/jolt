#[cfg(feature = "akita")]
use jolt_claims::protocols::jolt::formulas::dimensions::JoltFormulaDimensions;
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
#[cfg(feature = "akita")]
use jolt_lookup_tables::XLEN as RISCV_XLEN;
use jolt_openings::{BatchOpeningScheme, CommitmentScheme};
use jolt_transcript::Transcript;

#[cfg(feature = "akita")]
use crate::stages::stage8;
use crate::{
    config::{JoltProtocolConfig, PcsFamily},
    preprocessing::JoltVerifierPreprocessing,
    proof::{JoltProof, JoltProofClaims},
    verifier::CheckedInputs,
    VerifierError,
};

pub(crate) fn verify_lattice_packed_validity<F, PCS, VC, T, ZkProof>(
    config: &JoltProtocolConfig,
    preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    proof: &JoltProof<PCS, VC, ZkProof>,
    checked: &CheckedInputs,
    transcript: &mut T,
) -> Result<(), VerifierError>
where
    F: Field,
    PCS: CommitmentScheme<Field = F> + BatchOpeningScheme,
    PCS::Output: Clone,
    VC: VectorCommitment<Field = F>,
    T: Transcript<Challenge = F>,
{
    if proof.commitments.family() != PcsFamily::Lattice {
        return Ok(());
    }

    #[cfg(not(feature = "akita"))]
    {
        let _ = (config, preprocessing, proof, checked, transcript);
        Err(VerifierError::InvalidProtocolConfig {
            reason: "lattice packed validity verification requires the jolt-verifier akita feature"
                .to_string(),
        })
    }

    #[cfg(feature = "akita")]
    {
        if checked.zk {
            return Err(VerifierError::InvalidProtocolConfig {
                reason:
                    "lattice packed validity verification currently requires transparent claims"
                        .to_string(),
            });
        }
        let payload =
            proof
                .commitments
                .as_lattice()
                .ok_or_else(|| VerifierError::InvalidProtocolConfig {
                    reason: "lattice packed validity verification requires lattice commitments"
                        .to_string(),
                })?;
        let validity_claims = proof
            .clear_claims()?
            .stage7
            .lattice_packed_validity
            .as_ref()
            .ok_or(VerifierError::MissingLatticePackedValidityProof {
                field: "opening_claims",
            })?;
        let sumcheck_proof = proof
            .stages
            .lattice_packed_validity_sumcheck_proof
            .as_ref()
            .ok_or(VerifierError::MissingLatticePackedValidityProof {
                field: "sumcheck_proof",
            })?;
        let opening_proof = proof.lattice_packed_validity_opening_proof.as_ref().ok_or(
            VerifierError::MissingLatticePackedValidityProof {
                field: "opening_proof",
            },
        )?;

        let log_t = checked.trace_length.ilog2() as usize;
        let formula_dimensions = JoltFormulaDimensions::try_from(proof.one_hot_config.dimensions(
            log_t,
            2 * RISCV_XLEN,
            preprocessing.program.bytecode_len(),
            checked.ram_K,
        ))
        .map_err(|error| VerifierError::InvalidProtocolConfig {
            reason: format!("invalid lattice formula dimensions: {error}"),
        })?;
        let layout = stage8::derive_lattice_packed_witness_layout(
            config,
            log_t,
            proof.one_hot_config.committed_chunk_bits(),
            formula_dimensions.ra_layout,
            &checked.precommitted,
        )?;
        stage8::verify_lattice_packed_validity_proof::<F, PCS, T, _>(
            &preprocessing.pcs_setup,
            transcript,
            config,
            proof.one_hot_config.committed_chunk_bits(),
            &checked.precommitted,
            &layout,
            payload.packed_witness.clone(),
            sumcheck_proof,
            &validity_claims.opening_claims,
            opening_proof,
        )
    }
}

pub(crate) fn validate_lattice_layout_binding<PCS, VC, ZkProof>(
    config: &JoltProtocolConfig,
    preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    proof: &JoltProof<PCS, VC, ZkProof>,
    checked: &CheckedInputs,
) -> Result<(), VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    if proof.commitments.family() != PcsFamily::Lattice {
        return Ok(());
    }
    validate_lattice_precommitted_surface(config, checked)?;

    #[cfg(not(feature = "akita"))]
    {
        let _ = (config, preprocessing, checked);
        Err(VerifierError::InvalidProtocolConfig {
            reason: "lattice PCS mode requires the jolt-verifier akita feature".to_string(),
        })
    }

    #[cfg(feature = "akita")]
    {
        let log_t = checked.trace_length.ilog2() as usize;
        let formula_dimensions = JoltFormulaDimensions::try_from(proof.one_hot_config.dimensions(
            log_t,
            2 * RISCV_XLEN,
            preprocessing.program.bytecode_len(),
            checked.ram_K,
        ))
        .map_err(|error| VerifierError::InvalidProtocolConfig {
            reason: format!("invalid lattice formula dimensions: {error}"),
        })?;
        let layout = stage8::derive_lattice_packed_witness_layout(
            config,
            log_t,
            proof.one_hot_config.committed_chunk_bits(),
            formula_dimensions.ra_layout,
            &checked.precommitted,
        )?;
        stage8::validate_lattice_packed_witness_layout_config(config, &layout)?;
        stage8::validate_lattice_packed_witness_validity_config(
            config,
            proof.one_hot_config.committed_chunk_bits(),
            &checked.precommitted,
        )
    }
}

fn validate_lattice_precommitted_surface(
    config: &JoltProtocolConfig,
    checked: &CheckedInputs,
) -> Result<(), VerifierError> {
    let trusted_advice_present = checked.precommitted.trusted_advice.is_some();
    if config.lattice.advice.trusted != trusted_advice_present {
        let reason = if trusted_advice_present {
            "trusted advice precommitted schedule requires trusted advice lattice mode"
        } else {
            "trusted advice lattice mode requires a trusted advice precommitted schedule"
        };
        return Err(VerifierError::InvalidProtocolConfig {
            reason: reason.to_string(),
        });
    }

    let untrusted_advice_present = checked.precommitted.untrusted_advice.is_some();
    if config.lattice.advice.untrusted != untrusted_advice_present {
        let reason = if untrusted_advice_present {
            "untrusted advice precommitted schedule requires untrusted advice lattice mode"
        } else {
            "untrusted advice lattice mode requires an untrusted advice precommitted schedule"
        };
        return Err(VerifierError::InvalidProtocolConfig {
            reason: reason.to_string(),
        });
    }

    Ok(())
}

pub(crate) fn validate_lattice_validity_proof_surface<PCS, VC, ZkProof>(
    config: &JoltProtocolConfig,
    preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    proof: &JoltProof<PCS, VC, ZkProof>,
    checked: &CheckedInputs,
) -> Result<(), VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    let lattice = proof.commitments.family() == PcsFamily::Lattice;
    let validity_claims = match &proof.claims {
        JoltProofClaims::Clear(claims) => claims.stage7.lattice_packed_validity.as_ref(),
        JoltProofClaims::Zk { .. } => None,
    };

    if !lattice {
        if proof
            .stages
            .lattice_packed_validity_sumcheck_proof
            .is_some()
        {
            return Err(VerifierError::UnexpectedLatticePackedValidityProof {
                field: "sumcheck_proof",
            });
        }
        if proof.lattice_packed_validity_opening_proof.is_some() {
            return Err(VerifierError::UnexpectedLatticePackedValidityProof {
                field: "opening_proof",
            });
        }
        if validity_claims.is_some() {
            return Err(VerifierError::UnexpectedLatticePackedValidityProof {
                field: "opening_claims",
            });
        }
        return Ok(());
    }

    if proof
        .stages
        .lattice_packed_validity_sumcheck_proof
        .is_none()
    {
        return Err(VerifierError::MissingLatticePackedValidityProof {
            field: "sumcheck_proof",
        });
    }
    if proof.lattice_packed_validity_opening_proof.is_none() {
        return Err(VerifierError::MissingLatticePackedValidityProof {
            field: "opening_proof",
        });
    }
    let validity_claims =
        validity_claims.ok_or(VerifierError::MissingLatticePackedValidityProof {
            field: "opening_claims",
        })?;

    #[cfg(not(feature = "akita"))]
    {
        let _ = (config, preprocessing, checked, validity_claims);
        Err(VerifierError::InvalidProtocolConfig {
            reason: "lattice packed validity proof requires the jolt-verifier akita feature"
                .to_string(),
        })
    }

    #[cfg(feature = "akita")]
    {
        let log_t = checked.trace_length.ilog2() as usize;
        let formula_dimensions = JoltFormulaDimensions::try_from(proof.one_hot_config.dimensions(
            log_t,
            2 * RISCV_XLEN,
            preprocessing.program.bytecode_len(),
            checked.ram_K,
        ))
        .map_err(|error| VerifierError::InvalidProtocolConfig {
            reason: format!("invalid lattice formula dimensions: {error}"),
        })?;
        let layout = stage8::derive_lattice_packed_witness_layout(
            config,
            log_t,
            proof.one_hot_config.committed_chunk_bits(),
            formula_dimensions.ra_layout,
            &checked.precommitted,
        )?;
        let requirements = stage8::derive_lattice_packed_validity_requirements(
            config,
            proof.one_hot_config.committed_chunk_bits(),
            &checked.precommitted,
        )?;
        let statements = stage8::derive_lattice_packed_validity_statements(&layout, &requirements)?;
        let expected_opening_claims = stage8::lattice_packed_validity_opening_count(&statements);
        if validity_claims.opening_claims.len() != expected_opening_claims {
            return Err(VerifierError::LatticePackedValidityClaimCountMismatch {
                expected: expected_opening_claims,
                got: validity_claims.opening_claims.len(),
            });
        }
        Ok(())
    }
}
