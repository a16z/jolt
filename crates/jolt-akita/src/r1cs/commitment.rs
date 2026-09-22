//! Canonical commitment bytes shared with the Jolt transcript and later PCS checks.
use jolt_field::Fr;
use jolt_openings::OpeningsError;
use jolt_r1cs::bn254_bits::ByteVar;
use jolt_r1cs::fp128_bn254::{Fp128Error, Fp128Var};
use jolt_r1cs::R1csBuilder;
use jolt_transcript::r1cs::{Blake2bR1csError, LegacyBlake2bVar};
use thiserror::Error;

use crate::adapters::CommitmentFrame;
use crate::shape_guard::validate_commitment_profile_len;
use crate::{AkitaBackendFlavor, AkitaCommitment, AkitaVerifierSetup};

#[derive(Debug, Error)]
pub enum CommitmentR1csError {
    #[error("commitment does not match the fixed one-hot boundary profile")]
    Profile,
    #[error("setup encoding failed: {0}")]
    Encoding(String),
    #[error(transparent)]
    Setup(#[from] OpeningsError),
    #[error(transparent)]
    Field(#[from] Fp128Error),
    #[error(transparent)]
    Blake(#[from] Blake2bR1csError),
}

/// Fixed K16 single-group root. Setup/catalog trust is an application obligation.
pub struct AkitaCommitmentShape {
    header: AkitaCommitment,
    setup_encoding: Vec<u8>,
    catalog: Vec<u8>,
}

/// The canonical payload bytes and the very same decoded q128 handles.
pub struct AkitaCommitmentVars {
    bytes: Vec<ByteVar>,
    coefficients: Vec<Fp128Var>,
}

impl AkitaCommitmentVars {
    pub fn bytes(&self) -> &[ByteVar] {
        &self.bytes
    }
    pub fn coefficients(&self) -> &[Fp128Var] {
        &self.coefficients
    }
}

impl AkitaCommitmentShape {
    pub fn new(
        setup: &AkitaVerifierSetup,
        commitment: &AkitaCommitment,
    ) -> Result<Self, CommitmentR1csError> {
        if setup.one_hot_k != 16
            || commitment.backend_flavor != AkitaBackendFlavor::OneHot
            || commitment.layout_digest != setup.default_layout_digest
            || commitment.one_hot_k != 16
            || commitment.num_vars != 22
            || commitment.poly_count != 1
            || commitment.backend_coeff_len != 8
            || commitment.serialized_backend_bytes.len() != 128
            || setup.max_num_vars < 22
            || setup.max_num_polys_per_commitment_group < 1
        {
            return Err(CommitmentR1csError::Profile);
        }
        let scheme = setup.one_hot_k16_scheme()?;
        let mut matched = false;
        for row in scheme.schedules().rows() {
            let profiles = row.profiles();
            if profiles.precommitteds.is_empty()
                && profiles.final_group.group.num_vars() == commitment.num_vars
                && profiles.final_group.group.num_polynomials() == commitment.poly_count
            {
                validate_commitment_profile_len(commitment, &profiles.final_group)?;
                matched = true;
            }
        }
        if !matched {
            return Err(CommitmentR1csError::Profile);
        }
        let setup_encoding = bincode::serde::encode_to_vec(setup, bincode::config::standard())
            .map_err(|error| CommitmentR1csError::Encoding(error.to_string()))?;
        let catalog = setup
            .schedule_artifacts
            .one_hot()
            .ok_or(CommitmentR1csError::Profile)?
            .to_vec();
        let mut header = commitment.clone();
        header.serialized_backend_bytes.fill(0);
        Ok(Self {
            header,
            setup_encoding,
            catalog,
        })
    }

    /// Check the proof's fixed header before exposing its private encoded cells.
    pub fn witness_bytes<'a>(
        &self,
        commitment: &'a AkitaCommitment,
    ) -> Result<&'a [u8], CommitmentR1csError> {
        if commitment.backend_flavor != self.header.backend_flavor
            || commitment.layout_digest != self.header.layout_digest
            || commitment.num_vars != self.header.num_vars
            || commitment.poly_count != self.header.poly_count
            || commitment.one_hot_k != self.header.one_hot_k
            || commitment.backend_coeff_len != self.header.backend_coeff_len
            || commitment.serialized_backend_bytes.len()
                != self.header.serialized_backend_bytes.len()
        {
            return Err(CommitmentR1csError::Profile);
        }
        Ok(&commitment.serialized_backend_bytes)
    }

    pub fn setup_encoding(&self) -> &[u8] {
        &self.setup_encoding
    }
    pub fn catalog(&self) -> &[u8] {
        &self.catalog
    }
    pub fn layout_digest(&self) -> [u8; 32] {
        self.header.layout_digest
    }

    /// No payload value is trusted by the shape; canonicality is constrained here.
    pub fn allocate(
        &self,
        builder: &mut R1csBuilder<Fr>,
        witness: Option<&[u8]>,
    ) -> Result<AkitaCommitmentVars, CommitmentR1csError> {
        if witness.is_some_and(|bytes| bytes.len() != 128) {
            return Err(CommitmentR1csError::Profile);
        }
        let bytes = (0..128)
            .map(|i| ByteVar::allocate(builder, witness.and_then(|bytes| bytes.get(i)).copied()))
            .collect::<Vec<_>>();
        let mut coefficients = Vec::with_capacity(8);
        for chunk in bytes.chunks_exact(16) {
            let chunk: &[ByteVar; 16] =
                chunk.try_into().map_err(|_| CommitmentR1csError::Profile)?;
            coefficients.push(Fp128Var::from_le_bytes(builder, chunk)?);
        }
        Ok(AkitaCommitmentVars {
            bytes,
            coefficients,
        })
    }

    /// Append only the object's nine frames; the caller owns its outer label.
    pub fn append(
        &self,
        builder: &mut R1csBuilder<Fr>,
        transcript: &mut LegacyBlake2bVar,
        vars: &AkitaCommitmentVars,
    ) -> Result<(), CommitmentR1csError> {
        if vars.bytes.len() != 128 || vars.coefficients.len() != 8 {
            return Err(CommitmentR1csError::Profile);
        }
        transcript.check_schedule(9)?;
        for frame in self.header.frames() {
            match frame {
                CommitmentFrame::Label(label) => transcript.append_label(builder, label)?,
                CommitmentFrame::CountedLabel(label, count) => {
                    transcript.append_label_with_count(builder, label, count)?;
                }
                CommitmentFrame::Bytes(bytes) => transcript.append_bytes(
                    builder,
                    &bytes
                        .iter()
                        .copied()
                        .map(ByteVar::constant)
                        .collect::<Vec<_>>(),
                )?,
                CommitmentFrame::Word(value) => {
                    let mut bytes = vec![ByteVar::constant(0); 24];
                    bytes.extend(value.to_be_bytes().map(ByteVar::constant));
                    transcript.append_bytes(builder, &bytes)?;
                }
                CommitmentFrame::Payload(_) => transcript.append_bytes(builder, &vars.bytes)?,
            }
        }
        Ok(())
    }
}
