use std::{
    collections::BTreeMap,
    io::{Read, Write},
};

use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, SerializationError, Valid, Validate,
};
use num::FromPrimitive;
use strum::EnumCount;

use crate::subprotocols::univariate_skip::UniSkipFirstRoundProof;
use crate::{
    field::JoltField,
    poly::{
        commitment::{commitment_scheme::RecursionExt, hyrax::Hyrax},
        opening_proof::{OpeningId, OpeningPoint, Openings, SumcheckId},
    },
    subprotocols::sumcheck::SumcheckInstanceProof,
    transcripts::Transcript,
    zkvm::{
        config::{OneHotConfig, ReadWriteConfig},
        instruction::{CircuitFlags, InstructionFlags},
        recursion::{
            bijection::{ConstraintMapping, VarCountJaggedBijection},
            constraints_sys::ConstraintType,
            recursion_prover::RecursionProof,
            stage1::packed_gt_exp::PackedGtExpPublicInputs,
        },
        witness::{CommittedPolynomial, VirtualPolynomial},
    },
};
use ark_bn254::{Fq, Fq12};
use ark_grumpkin::Projective as GrumpkinProjective;

/// Constraint metadata for the recursion verifier
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct RecursionConstraintMetadata {
    pub constraint_types: Vec<ConstraintType>,
    pub jagged_bijection: VarCountJaggedBijection,
    pub jagged_mapping: ConstraintMapping,
    pub matrix_rows: Vec<usize>,
    pub dense_num_vars: usize,
    /// Public inputs for packed GT exp (base Fq12 and scalar bits for each GT exp)
    pub packed_gt_exp_public_inputs: Vec<PackedGtExpPublicInputs>,
}

/// Jolt proof structure organized by verification stages
pub struct JoltProof<F: JoltField, PCS: RecursionExt<F>, FS: Transcript> {
    // ============ Shared Data ============
    pub opening_claims: Claims<F>,
    pub commitments: Vec<PCS::Commitment>,

    // ============ Stage 1: R1CS Proof ============
    pub stage1_uni_skip_first_round_proof: UniSkipFirstRoundProof<F, FS>,
    pub stage1_sumcheck_proof: SumcheckInstanceProof<F, FS>,

    // ============ Stage 2: Spartan Virtual Remainder ============
    pub stage2_uni_skip_first_round_proof: UniSkipFirstRoundProof<F, FS>,
    pub stage2_sumcheck_proof: SumcheckInstanceProof<F, FS>,

    // ============ Stage 3: Spartan Shift ============
    pub stage3_sumcheck_proof: SumcheckInstanceProof<F, FS>,

    // ============ Stage 4: Read-Write Memory Checking ============
    pub stage4_sumcheck_proof: SumcheckInstanceProof<F, FS>,

    // ============ Stage 5: Registers & Bytecode ============
    pub stage5_sumcheck_proof: SumcheckInstanceProof<F, FS>,

    // ============ Stage 6: Hamming Weight & Ra Claims ============
    pub stage6_sumcheck_proof: SumcheckInstanceProof<F, FS>,

    // ============ Stage 7: Inc Claims & Virtualization ============
    pub stage7_sumcheck_proof: SumcheckInstanceProof<F, FS>,

    // ============ Stage 8: Dory Batch Opening ============
    /// Dory polynomial commitment opening proof
    pub stage8_opening_proof: PCS::Proof,
    /// Hint for combine_commitments offloading (the combined GT element)
    pub stage8_combine_hint: Option<Fq12>,

    // ============ Stage 9: Recursion Witness Generation ============
    /// PCS hint for recursion witness generation
    pub stage9_pcs_hint: Option<<PCS as RecursionExt<F>>::Hint>,

    // ============ Stage 10: Constraint System Metadata ============
    /// Constraint metadata extracted from recursion prover
    pub stage10_recursion_metadata: RecursionConstraintMetadata,

    // ============ Stages 11-13: Recursion SNARK ============
    /// Combined proof containing:
    /// - Stage 11: Recursion sumchecks (constraint, virtualization, jagged)
    /// - Stage 12: Dense polynomial commitment
    /// - Stage 13: Hyrax opening proof
    pub recursion_proof: RecursionProof<Fq, FS, Hyrax<1, GrumpkinProjective>>,

    // ============ Advice Proofs ============
    /// Trusted advice opening proof at point from RamValEvaluation
    pub trusted_advice_val_evaluation_proof: Option<PCS::Proof>,
    /// Trusted advice opening proof at point from RamValFinalEvaluation
    pub trusted_advice_val_final_proof: Option<PCS::Proof>,
    /// Untrusted advice opening proof at point from RamValEvaluation
    pub untrusted_advice_val_evaluation_proof: Option<PCS::Proof>,
    /// Untrusted advice opening proof at point from RamValFinalEvaluation
    pub untrusted_advice_val_final_proof: Option<PCS::Proof>,
    pub untrusted_advice_commitment: Option<PCS::Commitment>,

    // ============ Configuration ============
    pub trace_length: usize,
    pub ram_K: usize,
    pub bytecode_K: usize,
    pub rw_config: ReadWriteConfig,
    pub one_hot_config: OneHotConfig,
}

impl<F: JoltField, PCS: RecursionExt<F>, FS: Transcript> CanonicalSerialize
    for JoltProof<F, PCS, FS>
{
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.opening_claims
            .serialize_with_mode(&mut writer, compress)?;
        self.commitments
            .serialize_with_mode(&mut writer, compress)?;
        self.stage1_uni_skip_first_round_proof
            .serialize_with_mode(&mut writer, compress)?;
        self.stage1_sumcheck_proof
            .serialize_with_mode(&mut writer, compress)?;
        self.stage2_uni_skip_first_round_proof
            .serialize_with_mode(&mut writer, compress)?;
        self.stage2_sumcheck_proof
            .serialize_with_mode(&mut writer, compress)?;
        self.stage3_sumcheck_proof
            .serialize_with_mode(&mut writer, compress)?;
        self.stage4_sumcheck_proof
            .serialize_with_mode(&mut writer, compress)?;
        self.stage5_sumcheck_proof
            .serialize_with_mode(&mut writer, compress)?;
        self.stage6_sumcheck_proof
            .serialize_with_mode(&mut writer, compress)?;
        self.stage7_sumcheck_proof
            .serialize_with_mode(&mut writer, compress)?;
        self.stage8_opening_proof
            .serialize_with_mode(&mut writer, compress)?;
        self.stage8_combine_hint
            .serialize_with_mode(&mut writer, compress)?;
        self.stage9_pcs_hint
            .serialize_with_mode(&mut writer, compress)?;
        self.stage10_recursion_metadata
            .serialize_with_mode(&mut writer, compress)?;
        self.recursion_proof
            .serialize_with_mode(&mut writer, compress)?;
        self.trusted_advice_val_evaluation_proof
            .serialize_with_mode(&mut writer, compress)?;
        self.trusted_advice_val_final_proof
            .serialize_with_mode(&mut writer, compress)?;
        self.untrusted_advice_val_evaluation_proof
            .serialize_with_mode(&mut writer, compress)?;
        self.untrusted_advice_val_final_proof
            .serialize_with_mode(&mut writer, compress)?;
        self.untrusted_advice_commitment
            .serialize_with_mode(&mut writer, compress)?;
        self.trace_length
            .serialize_with_mode(&mut writer, compress)?;
        self.ram_K.serialize_with_mode(&mut writer, compress)?;
        self.bytecode_K.serialize_with_mode(&mut writer, compress)?;
        self.rw_config.serialize_with_mode(&mut writer, compress)?;
        self.one_hot_config
            .serialize_with_mode(&mut writer, compress)?;
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.opening_claims.serialized_size(compress)
            + self.commitments.serialized_size(compress)
            + self
                .stage1_uni_skip_first_round_proof
                .serialized_size(compress)
            + self.stage1_sumcheck_proof.serialized_size(compress)
            + self
                .stage2_uni_skip_first_round_proof
                .serialized_size(compress)
            + self.stage2_sumcheck_proof.serialized_size(compress)
            + self.stage3_sumcheck_proof.serialized_size(compress)
            + self.stage4_sumcheck_proof.serialized_size(compress)
            + self.stage5_sumcheck_proof.serialized_size(compress)
            + self.stage6_sumcheck_proof.serialized_size(compress)
            + self.stage7_sumcheck_proof.serialized_size(compress)
            + self.stage8_opening_proof.serialized_size(compress)
            + self.stage8_combine_hint.serialized_size(compress)
            + self.stage9_pcs_hint.serialized_size(compress)
            + self.stage10_recursion_metadata.serialized_size(compress)
            + self.recursion_proof.serialized_size(compress)
            + self
                .trusted_advice_val_evaluation_proof
                .serialized_size(compress)
            + self
                .trusted_advice_val_final_proof
                .serialized_size(compress)
            + self
                .untrusted_advice_val_evaluation_proof
                .serialized_size(compress)
            + self
                .untrusted_advice_val_final_proof
                .serialized_size(compress)
            + self.untrusted_advice_commitment.serialized_size(compress)
            + self.trace_length.serialized_size(compress)
            + self.ram_K.serialized_size(compress)
            + self.bytecode_K.serialized_size(compress)
            + self.rw_config.serialized_size(compress)
            + self.one_hot_config.serialized_size(compress)
    }
}

impl<F: JoltField, PCS: RecursionExt<F>, FS: Transcript> Valid for JoltProof<F, PCS, FS> {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl<F: JoltField, PCS: RecursionExt<F>, FS: Transcript> CanonicalDeserialize
    for JoltProof<F, PCS, FS>
{
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        Ok(Self {
            opening_claims: Claims::deserialize_with_mode(&mut reader, compress, validate)?,
            commitments: Vec::deserialize_with_mode(&mut reader, compress, validate)?,
            stage1_uni_skip_first_round_proof: UniSkipFirstRoundProof::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
            )?,
            stage1_sumcheck_proof: SumcheckInstanceProof::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
            )?,
            stage2_uni_skip_first_round_proof: UniSkipFirstRoundProof::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
            )?,
            stage2_sumcheck_proof: SumcheckInstanceProof::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
            )?,
            stage3_sumcheck_proof: SumcheckInstanceProof::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
            )?,
            stage4_sumcheck_proof: SumcheckInstanceProof::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
            )?,
            stage5_sumcheck_proof: SumcheckInstanceProof::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
            )?,
            stage6_sumcheck_proof: SumcheckInstanceProof::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
            )?,
            stage7_sumcheck_proof: SumcheckInstanceProof::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
            )?,
            stage8_opening_proof: PCS::Proof::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
            )?,
            stage8_combine_hint: Option::deserialize_with_mode(&mut reader, compress, validate)?,
            stage9_pcs_hint: Option::deserialize_with_mode(&mut reader, compress, validate)?,
            stage10_recursion_metadata: RecursionConstraintMetadata::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
            )?,
            recursion_proof: RecursionProof::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
            )?,
            trusted_advice_val_evaluation_proof: Option::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
            )?,
            trusted_advice_val_final_proof: Option::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
            )?,
            untrusted_advice_val_evaluation_proof: Option::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
            )?,
            untrusted_advice_val_final_proof: Option::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
            )?,
            untrusted_advice_commitment: Option::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
            )?,
            trace_length: usize::deserialize_with_mode(&mut reader, compress, validate)?,
            ram_K: usize::deserialize_with_mode(&mut reader, compress, validate)?,
            bytecode_K: usize::deserialize_with_mode(&mut reader, compress, validate)?,
            rw_config: ReadWriteConfig::deserialize_with_mode(&mut reader, compress, validate)?,
            one_hot_config: OneHotConfig::deserialize_with_mode(&mut reader, compress, validate)?,
        })
    }
}

pub struct Claims<F: JoltField>(pub Openings<F>);

impl<F: JoltField> CanonicalSerialize for Claims<F> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.0.len().serialize_with_mode(&mut writer, compress)?;
        for (key, (_opening_point, claim)) in self.0.iter() {
            key.serialize_with_mode(&mut writer, compress)?;
            claim.serialize_with_mode(&mut writer, compress)?;
        }
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        let mut size = self.0.len().serialized_size(compress);
        for (key, (_opening_point, claim)) in self.0.iter() {
            size += key.serialized_size(compress);
            size += claim.serialized_size(compress);
        }
        size
    }
}

impl<F: JoltField> Valid for Claims<F> {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl<F: JoltField> CanonicalDeserialize for Claims<F> {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let size = usize::deserialize_with_mode(&mut reader, compress, validate)?;
        let mut claims = BTreeMap::new();
        for _ in 0..size {
            let key = OpeningId::deserialize_with_mode(&mut reader, compress, validate)?;
            let claim = F::deserialize_with_mode(&mut reader, compress, validate)?;
            claims.insert(key, (OpeningPoint::default(), claim));
        }

        Ok(Claims(claims))
    }
}

// Compact encoding for OpeningId:
// Each variant uses a fused byte = BASE + sumcheck_id (1 byte total for advice, 2 bytes for committed/virtual)
// - [0, NUM_SUMCHECKS) = UntrustedAdvice(sumcheck_id)
// - [NUM_SUMCHECKS, 2*NUM_SUMCHECKS) = TrustedAdvice(sumcheck_id)
// - [2*NUM_SUMCHECKS, 3*NUM_SUMCHECKS) + poly_index = Committed(poly, sumcheck_id)
// - [3*NUM_SUMCHECKS, 4*NUM_SUMCHECKS) + poly_index = Virtual(poly, sumcheck_id)
const OPENING_ID_UNTRUSTED_ADVICE_BASE: u8 = 0;
const OPENING_ID_TRUSTED_ADVICE_BASE: u8 =
    OPENING_ID_UNTRUSTED_ADVICE_BASE + SumcheckId::COUNT as u8;
const OPENING_ID_COMMITTED_BASE: u8 = OPENING_ID_TRUSTED_ADVICE_BASE + SumcheckId::COUNT as u8;
const OPENING_ID_VIRTUAL_BASE: u8 = OPENING_ID_COMMITTED_BASE + SumcheckId::COUNT as u8;

impl CanonicalSerialize for OpeningId {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        match self {
            OpeningId::UntrustedAdvice(sumcheck_id) => {
                let fused = OPENING_ID_UNTRUSTED_ADVICE_BASE + (*sumcheck_id as u8);
                fused.serialize_with_mode(&mut writer, compress)
            }
            OpeningId::TrustedAdvice(sumcheck_id) => {
                let fused = OPENING_ID_TRUSTED_ADVICE_BASE + (*sumcheck_id as u8);
                fused.serialize_with_mode(&mut writer, compress)
            }
            OpeningId::Committed(committed_polynomial, sumcheck_id) => {
                let fused = OPENING_ID_COMMITTED_BASE + (*sumcheck_id as u8);
                fused.serialize_with_mode(&mut writer, compress)?;
                committed_polynomial.serialize_with_mode(&mut writer, compress)
            }
            OpeningId::Virtual(virtual_polynomial, sumcheck_id) => {
                let fused = OPENING_ID_VIRTUAL_BASE + (*sumcheck_id as u8);
                fused.serialize_with_mode(&mut writer, compress)?;
                virtual_polynomial.serialize_with_mode(&mut writer, compress)
            }
        }
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        match self {
            OpeningId::UntrustedAdvice(_) | OpeningId::TrustedAdvice(_) => 1,
            OpeningId::Committed(committed_polynomial, _) => {
                // 1 byte fused (variant + sumcheck_id) + poly index
                1 + committed_polynomial.serialized_size(compress)
            }
            OpeningId::Virtual(virtual_polynomial, _) => {
                // 1 byte fused (variant + sumcheck_id) + poly index
                1 + virtual_polynomial.serialized_size(compress)
            }
        }
    }
}

impl Valid for OpeningId {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl CanonicalDeserialize for OpeningId {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let fused = u8::deserialize_with_mode(&mut reader, compress, validate)?;
        match fused {
            _ if fused < OPENING_ID_TRUSTED_ADVICE_BASE => {
                let sumcheck_id = fused - OPENING_ID_UNTRUSTED_ADVICE_BASE;
                Ok(OpeningId::UntrustedAdvice(
                    SumcheckId::from_u8(sumcheck_id).ok_or(SerializationError::InvalidData)?,
                ))
            }
            _ if fused < OPENING_ID_COMMITTED_BASE => {
                let sumcheck_id = fused - OPENING_ID_TRUSTED_ADVICE_BASE;
                Ok(OpeningId::TrustedAdvice(
                    SumcheckId::from_u8(sumcheck_id).ok_or(SerializationError::InvalidData)?,
                ))
            }
            _ if fused < OPENING_ID_VIRTUAL_BASE => {
                let sumcheck_id = fused - OPENING_ID_COMMITTED_BASE;
                let polynomial =
                    CommittedPolynomial::deserialize_with_mode(&mut reader, compress, validate)?;
                Ok(OpeningId::Committed(
                    polynomial,
                    SumcheckId::from_u8(sumcheck_id).ok_or(SerializationError::InvalidData)?,
                ))
            }
            _ => {
                let sumcheck_id = fused - OPENING_ID_VIRTUAL_BASE;
                let polynomial =
                    VirtualPolynomial::deserialize_with_mode(&mut reader, compress, validate)?;
                Ok(OpeningId::Virtual(
                    polynomial,
                    SumcheckId::from_u8(sumcheck_id).ok_or(SerializationError::InvalidData)?,
                ))
            }
        }
    }
}

impl CanonicalSerialize for CommittedPolynomial {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        match self {
            Self::RdInc => 0u8.serialize_with_mode(writer, compress),
            Self::RamInc => 1u8.serialize_with_mode(writer, compress),
            Self::InstructionRa(i) => {
                2u8.serialize_with_mode(&mut writer, compress)?;
                (u8::try_from(*i).unwrap()).serialize_with_mode(writer, compress)
            }
            Self::BytecodeRa(i) => {
                3u8.serialize_with_mode(&mut writer, compress)?;
                (u8::try_from(*i).unwrap()).serialize_with_mode(writer, compress)
            }
            Self::RamRa(i) => {
                4u8.serialize_with_mode(&mut writer, compress)?;
                (u8::try_from(*i).unwrap()).serialize_with_mode(writer, compress)
            }
            Self::DoryDenseMatrix => 5u8.serialize_with_mode(writer, compress),
        }
    }

    fn serialized_size(&self, _compress: Compress) -> usize {
        match self {
            Self::RdInc | Self::RamInc | Self::DoryDenseMatrix => 1,
            Self::InstructionRa(_) | Self::BytecodeRa(_) | Self::RamRa(_) => 2,
        }
    }
}

impl Valid for CommittedPolynomial {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl CanonicalDeserialize for CommittedPolynomial {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        Ok(
            match u8::deserialize_with_mode(&mut reader, compress, validate)? {
                0 => Self::RdInc,
                1 => Self::RamInc,
                2 => {
                    let i = u8::deserialize_with_mode(reader, compress, validate)?;
                    Self::InstructionRa(i as usize)
                }
                3 => {
                    let i = u8::deserialize_with_mode(reader, compress, validate)?;
                    Self::BytecodeRa(i as usize)
                }
                4 => {
                    let i = u8::deserialize_with_mode(reader, compress, validate)?;
                    Self::RamRa(i as usize)
                }
                5 => Self::DoryDenseMatrix,
                _ => return Err(SerializationError::InvalidData),
            },
        )
    }
}

impl CanonicalSerialize for VirtualPolynomial {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        match self {
            Self::PC => 0u8.serialize_with_mode(&mut writer, compress),
            Self::UnexpandedPC => 1u8.serialize_with_mode(&mut writer, compress),
            Self::NextPC => 2u8.serialize_with_mode(&mut writer, compress),
            Self::NextUnexpandedPC => 3u8.serialize_with_mode(&mut writer, compress),
            Self::NextIsNoop => 4u8.serialize_with_mode(&mut writer, compress),
            Self::NextIsVirtual => 5u8.serialize_with_mode(&mut writer, compress),
            Self::NextIsFirstInSequence => 6u8.serialize_with_mode(&mut writer, compress),
            Self::LeftLookupOperand => 7u8.serialize_with_mode(&mut writer, compress),
            Self::RightLookupOperand => 8u8.serialize_with_mode(&mut writer, compress),
            Self::LeftInstructionInput => 9u8.serialize_with_mode(&mut writer, compress),
            Self::RightInstructionInput => 10u8.serialize_with_mode(&mut writer, compress),
            Self::Product => 11u8.serialize_with_mode(&mut writer, compress),
            Self::ShouldJump => 12u8.serialize_with_mode(&mut writer, compress),
            Self::ShouldBranch => 13u8.serialize_with_mode(&mut writer, compress),
            Self::WritePCtoRD => 14u8.serialize_with_mode(&mut writer, compress),
            Self::WriteLookupOutputToRD => 15u8.serialize_with_mode(&mut writer, compress),
            Self::Rd => 16u8.serialize_with_mode(&mut writer, compress),
            Self::Imm => 17u8.serialize_with_mode(&mut writer, compress),
            Self::Rs1Value => 18u8.serialize_with_mode(&mut writer, compress),
            Self::Rs2Value => 19u8.serialize_with_mode(&mut writer, compress),
            Self::RdWriteValue => 20u8.serialize_with_mode(&mut writer, compress),
            Self::Rs1Ra => 21u8.serialize_with_mode(&mut writer, compress),
            Self::Rs2Ra => 22u8.serialize_with_mode(&mut writer, compress),
            Self::RdWa => 23u8.serialize_with_mode(&mut writer, compress),
            Self::LookupOutput => 24u8.serialize_with_mode(&mut writer, compress),
            Self::InstructionRaf => 25u8.serialize_with_mode(&mut writer, compress),
            Self::InstructionRafFlag => 26u8.serialize_with_mode(&mut writer, compress),
            Self::InstructionRa(i) => {
                27u8.serialize_with_mode(&mut writer, compress)?;
                (*i as u32).serialize_with_mode(&mut writer, compress)
            }
            Self::RegistersVal => 28u8.serialize_with_mode(&mut writer, compress),
            Self::RamAddress => 29u8.serialize_with_mode(&mut writer, compress),
            Self::RamRa => 30u8.serialize_with_mode(&mut writer, compress),
            Self::RamReadValue => 31u8.serialize_with_mode(&mut writer, compress),
            Self::RamWriteValue => 32u8.serialize_with_mode(&mut writer, compress),
            Self::RamVal => 33u8.serialize_with_mode(&mut writer, compress),
            Self::RamValInit => 34u8.serialize_with_mode(&mut writer, compress),
            Self::RamValFinal => 35u8.serialize_with_mode(&mut writer, compress),
            Self::RamHammingWeight => 36u8.serialize_with_mode(&mut writer, compress),
            Self::UnivariateSkip => 37u8.serialize_with_mode(&mut writer, compress),
            Self::OpFlags(flags) => {
                38u8.serialize_with_mode(&mut writer, compress)?;
                (u8::try_from(*flags as usize).unwrap()).serialize_with_mode(&mut writer, compress)
            }
            Self::InstructionFlags(flags) => {
                39u8.serialize_with_mode(&mut writer, compress)?;
                (u8::try_from(*flags as usize).unwrap()).serialize_with_mode(&mut writer, compress)
            }
            Self::LookupTableFlag(flag) => {
                40u8.serialize_with_mode(&mut writer, compress)?;
                (u8::try_from(*flag).unwrap()).serialize_with_mode(&mut writer, compress)
            }
            Self::RecursionBase(i) => {
                41u8.serialize_with_mode(&mut writer, compress)?;
                (*i as u32).serialize_with_mode(&mut writer, compress)
            }
            Self::RecursionRhoPrev(i) => {
                42u8.serialize_with_mode(&mut writer, compress)?;
                (*i as u32).serialize_with_mode(&mut writer, compress)
            }
            Self::RecursionRhoCurr(i) => {
                43u8.serialize_with_mode(&mut writer, compress)?;
                (*i as u32).serialize_with_mode(&mut writer, compress)
            }
            Self::RecursionQuotient(i) => {
                44u8.serialize_with_mode(&mut writer, compress)?;
                (*i as u32).serialize_with_mode(&mut writer, compress)
            }
            Self::RecursionMulLhs(i) => {
                45u8.serialize_with_mode(&mut writer, compress)?;
                (*i as u32).serialize_with_mode(&mut writer, compress)
            }
            Self::RecursionMulRhs(i) => {
                46u8.serialize_with_mode(&mut writer, compress)?;
                (*i as u32).serialize_with_mode(&mut writer, compress)
            }
            Self::RecursionMulResult(i) => {
                47u8.serialize_with_mode(&mut writer, compress)?;
                (*i as u32).serialize_with_mode(&mut writer, compress)
            }
            Self::RecursionMulQuotient(i) => {
                48u8.serialize_with_mode(&mut writer, compress)?;
                (*i as u32).serialize_with_mode(&mut writer, compress)
            }
            Self::RecursionG1ScalarMulXA(i) => {
                49u8.serialize_with_mode(&mut writer, compress)?;
                (*i as u32).serialize_with_mode(&mut writer, compress)
            }
            Self::RecursionG1ScalarMulYA(i) => {
                50u8.serialize_with_mode(&mut writer, compress)?;
                (*i as u32).serialize_with_mode(&mut writer, compress)
            }
            Self::RecursionG1ScalarMulXT(i) => {
                51u8.serialize_with_mode(&mut writer, compress)?;
                (*i as u32).serialize_with_mode(&mut writer, compress)
            }
            Self::RecursionG1ScalarMulYT(i) => {
                52u8.serialize_with_mode(&mut writer, compress)?;
                (*i as u32).serialize_with_mode(&mut writer, compress)
            }
            Self::RecursionG1ScalarMulXANext(i) => {
                53u8.serialize_with_mode(&mut writer, compress)?;
                (*i as u32).serialize_with_mode(&mut writer, compress)
            }
            Self::RecursionG1ScalarMulYANext(i) => {
                54u8.serialize_with_mode(&mut writer, compress)?;
                (*i as u32).serialize_with_mode(&mut writer, compress)
            }
            Self::RecursionG1ScalarMulIndicator(i) => {
                55u8.serialize_with_mode(&mut writer, compress)?;
                (*i as u32).serialize_with_mode(&mut writer, compress)
            }
            Self::DorySparseConstraintMatrix => 56u8.serialize_with_mode(&mut writer, compress),
            Self::PackedGtExpRho(i) => {
                57u8.serialize_with_mode(&mut writer, compress)?;
                (*i as u32).serialize_with_mode(&mut writer, compress)
            }
            Self::PackedGtExpRhoNext(i) => {
                58u8.serialize_with_mode(&mut writer, compress)?;
                (*i as u32).serialize_with_mode(&mut writer, compress)
            }
            Self::PackedGtExpQuotient(i) => {
                59u8.serialize_with_mode(&mut writer, compress)?;
                (*i as u32).serialize_with_mode(&mut writer, compress)
            } // Note: PackedGtExpBit and PackedGtExpBase removed - they are public inputs
        }
    }

    fn serialized_size(&self, _compress: Compress) -> usize {
        match self {
            Self::PC
            | Self::UnexpandedPC
            | Self::NextPC
            | Self::NextUnexpandedPC
            | Self::NextIsNoop
            | Self::NextIsVirtual
            | Self::NextIsFirstInSequence
            | Self::LeftLookupOperand
            | Self::RightLookupOperand
            | Self::LeftInstructionInput
            | Self::RightInstructionInput
            | Self::Product
            | Self::ShouldJump
            | Self::ShouldBranch
            | Self::WritePCtoRD
            | Self::WriteLookupOutputToRD
            | Self::Rd
            | Self::Imm
            | Self::Rs1Value
            | Self::Rs2Value
            | Self::RdWriteValue
            | Self::Rs1Ra
            | Self::Rs2Ra
            | Self::RdWa
            | Self::LookupOutput
            | Self::InstructionRaf
            | Self::InstructionRafFlag
            | Self::RegistersVal
            | Self::RamAddress
            | Self::RamRa
            | Self::RamReadValue
            | Self::RamWriteValue
            | Self::RamVal
            | Self::RamValInit
            | Self::RamValFinal
            | Self::RamHammingWeight
            | Self::UnivariateSkip
            | Self::DorySparseConstraintMatrix => 1,
            Self::PackedGtExpRho(_)
            | Self::PackedGtExpRhoNext(_)
            | Self::PackedGtExpQuotient(_) => 5,
            Self::InstructionRa(_)
            | Self::OpFlags(_)
            | Self::InstructionFlags(_)
            | Self::LookupTableFlag(_) => 2,
            Self::RecursionBase(_)
            | Self::RecursionRhoPrev(_)
            | Self::RecursionRhoCurr(_)
            | Self::RecursionQuotient(_)
            | Self::RecursionMulLhs(_)
            | Self::RecursionMulRhs(_)
            | Self::RecursionMulResult(_)
            | Self::RecursionMulQuotient(_)
            | Self::RecursionG1ScalarMulXA(_)
            | Self::RecursionG1ScalarMulYA(_)
            | Self::RecursionG1ScalarMulXT(_)
            | Self::RecursionG1ScalarMulYT(_)
            | Self::RecursionG1ScalarMulXANext(_)
            | Self::RecursionG1ScalarMulYANext(_)
            | Self::RecursionG1ScalarMulIndicator(_) => 5, // 1 byte discriminator + 4 bytes u32
        }
    }
}

impl Valid for VirtualPolynomial {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl CanonicalDeserialize for VirtualPolynomial {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        Ok(
            match u8::deserialize_with_mode(&mut reader, compress, validate)? {
                0 => Self::PC,
                1 => Self::UnexpandedPC,
                2 => Self::NextPC,
                3 => Self::NextUnexpandedPC,
                4 => Self::NextIsNoop,
                5 => Self::NextIsVirtual,
                6 => Self::NextIsFirstInSequence,
                7 => Self::LeftLookupOperand,
                8 => Self::RightLookupOperand,
                9 => Self::LeftInstructionInput,
                10 => Self::RightInstructionInput,
                11 => Self::Product,
                12 => Self::ShouldJump,
                13 => Self::ShouldBranch,
                14 => Self::WritePCtoRD,
                15 => Self::WriteLookupOutputToRD,
                16 => Self::Rd,
                17 => Self::Imm,
                18 => Self::Rs1Value,
                19 => Self::Rs2Value,
                20 => Self::RdWriteValue,
                21 => Self::Rs1Ra,
                22 => Self::Rs2Ra,
                23 => Self::RdWa,
                24 => Self::LookupOutput,
                25 => Self::InstructionRaf,
                26 => Self::InstructionRafFlag,
                27 => {
                    let i = u32::deserialize_with_mode(&mut reader, compress, validate)?;
                    Self::InstructionRa(i as usize)
                }
                28 => Self::RegistersVal,
                29 => Self::RamAddress,
                30 => Self::RamRa,
                31 => Self::RamReadValue,
                32 => Self::RamWriteValue,
                33 => Self::RamVal,
                34 => Self::RamValInit,
                35 => Self::RamValFinal,
                36 => Self::RamHammingWeight,
                37 => Self::UnivariateSkip,
                38 => {
                    let discriminant = u8::deserialize_with_mode(&mut reader, compress, validate)?;
                    let flags = CircuitFlags::from_repr(discriminant)
                        .ok_or(SerializationError::InvalidData)?;
                    Self::OpFlags(flags)
                }
                39 => {
                    let discriminant = u8::deserialize_with_mode(&mut reader, compress, validate)?;
                    let flags = InstructionFlags::from_repr(discriminant)
                        .ok_or(SerializationError::InvalidData)?;
                    Self::InstructionFlags(flags)
                }
                40 => {
                    let flag = u8::deserialize_with_mode(&mut reader, compress, validate)?;
                    Self::LookupTableFlag(flag as usize)
                }
                41 => {
                    let i = u32::deserialize_with_mode(&mut reader, compress, validate)?;
                    Self::RecursionBase(i as usize)
                }
                42 => {
                    let i = u32::deserialize_with_mode(&mut reader, compress, validate)?;
                    Self::RecursionRhoPrev(i as usize)
                }
                43 => {
                    let i = u32::deserialize_with_mode(&mut reader, compress, validate)?;
                    Self::RecursionRhoCurr(i as usize)
                }
                44 => {
                    let i = u32::deserialize_with_mode(&mut reader, compress, validate)?;
                    Self::RecursionQuotient(i as usize)
                }
                45 => {
                    let i = u32::deserialize_with_mode(&mut reader, compress, validate)?;
                    Self::RecursionMulLhs(i as usize)
                }
                46 => {
                    let i = u32::deserialize_with_mode(&mut reader, compress, validate)?;
                    Self::RecursionMulRhs(i as usize)
                }
                47 => {
                    let i = u32::deserialize_with_mode(&mut reader, compress, validate)?;
                    Self::RecursionMulResult(i as usize)
                }
                48 => {
                    let i = u32::deserialize_with_mode(&mut reader, compress, validate)?;
                    Self::RecursionMulQuotient(i as usize)
                }
                49 => {
                    let i = u32::deserialize_with_mode(&mut reader, compress, validate)?;
                    Self::RecursionG1ScalarMulXA(i as usize)
                }
                50 => {
                    let i = u32::deserialize_with_mode(&mut reader, compress, validate)?;
                    Self::RecursionG1ScalarMulYA(i as usize)
                }
                51 => {
                    let i = u32::deserialize_with_mode(&mut reader, compress, validate)?;
                    Self::RecursionG1ScalarMulXT(i as usize)
                }
                52 => {
                    let i = u32::deserialize_with_mode(&mut reader, compress, validate)?;
                    Self::RecursionG1ScalarMulYT(i as usize)
                }
                53 => {
                    let i = u32::deserialize_with_mode(&mut reader, compress, validate)?;
                    Self::RecursionG1ScalarMulXANext(i as usize)
                }
                54 => {
                    let i = u32::deserialize_with_mode(&mut reader, compress, validate)?;
                    Self::RecursionG1ScalarMulYANext(i as usize)
                }
                55 => {
                    let i = u32::deserialize_with_mode(&mut reader, compress, validate)?;
                    Self::RecursionG1ScalarMulIndicator(i as usize)
                }
                56 => Self::DorySparseConstraintMatrix,
                57 => {
                    let i = u32::deserialize_with_mode(&mut reader, compress, validate)?;
                    Self::PackedGtExpRho(i as usize)
                }
                58 => {
                    let i = u32::deserialize_with_mode(&mut reader, compress, validate)?;
                    Self::PackedGtExpRhoNext(i as usize)
                }
                59 => {
                    let i = u32::deserialize_with_mode(&mut reader, compress, validate)?;
                    Self::PackedGtExpQuotient(i as usize)
                }
                // Note: 60/61 (PackedGtExpBit/Base) removed - they are now public inputs
                _ => return Err(SerializationError::InvalidData),
            },
        )
    }
}

pub fn serialize_and_print_size(
    item_name: &str,
    file_name: &str,
    item: &impl CanonicalSerialize,
) -> Result<(), SerializationError> {
    use std::fs::File;
    let mut file = File::create(file_name)?;
    item.serialize_compressed(&mut file)?;
    let file_size_bytes = file.metadata()?.len();
    let file_size_kb = file_size_bytes as f64 / 1024.0;
    tracing::info!("{item_name} Written to {file_name}");
    tracing::info!("{item_name} size: {file_size_kb:.1} kB");
    Ok(())
}
