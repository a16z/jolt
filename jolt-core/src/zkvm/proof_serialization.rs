#[cfg(not(feature = "zk"))]
use std::collections::BTreeMap;
use std::io::{Read, Write};

use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, SerializationError, Valid, Validate,
};
use num::FromPrimitive;
use strum::EnumCount;

#[cfg(not(feature = "zk"))]
use crate::poly::opening_proof::{OpeningPoint, Openings};
#[cfg(feature = "zk")]
use crate::subprotocols::blindfold::BlindFoldProof;
#[cfg(not(feature = "zk"))]
use crate::utils::serialization::MAX_OPENING_CLAIMS;
use crate::{
    curve::JoltCurve,
    field::JoltField,
    poly::{
        commitment::{commitment_scheme::CommitmentScheme, dory::DoryLayout},
        opening_proof::{OpeningId, PolynomialId, SumcheckId},
    },
};
use crate::{
    subprotocols::{
        sumcheck::SumcheckInstanceProof, univariate_skip::UniSkipFirstRoundProofVariant,
    },
    transcripts::Transcript,
    utils::serialization::{
        deserialize_bounded_vec, serialize_vec_with_len, serialized_vec_with_len_size,
        MAX_JOLT_COMMITMENTS,
    },
    zkvm::{
        config::{OneHotConfig, ReadWriteConfig},
        instruction::{CircuitFlags, InstructionFlags},
        witness::{CommittedPolynomial, VirtualPolynomial},
    },
};

#[derive(Clone)]
pub struct JoltProof<
    F: JoltField,
    C: JoltCurve<F = F>,
    PCS: CommitmentScheme<Field = F>,
    FS: Transcript,
> {
    pub commitments: Vec<PCS::Commitment>,
    pub stage1_uni_skip_first_round_proof: UniSkipFirstRoundProofVariant<F, C, FS>,
    pub stage1_sumcheck_proof: SumcheckInstanceProof<F, C, FS>,
    pub stage2_uni_skip_first_round_proof: UniSkipFirstRoundProofVariant<F, C, FS>,
    pub stage2_sumcheck_proof: SumcheckInstanceProof<F, C, FS>,
    pub stage3_sumcheck_proof: SumcheckInstanceProof<F, C, FS>,
    pub stage4_sumcheck_proof: SumcheckInstanceProof<F, C, FS>,
    pub stage5_sumcheck_proof: SumcheckInstanceProof<F, C, FS>,
    pub stage6_sumcheck_proof: SumcheckInstanceProof<F, C, FS>,
    pub stage7_sumcheck_proof: SumcheckInstanceProof<F, C, FS>,
    #[cfg(feature = "zk")]
    pub blindfold_proof: BlindFoldProof<F, C>,
    pub joint_opening_proof: PCS::Proof,
    pub untrusted_advice_commitment: Option<PCS::Commitment>,
    #[cfg(not(feature = "zk"))]
    pub opening_claims: Claims<F>,
    pub trace_length: usize,
    pub ram_K: usize,
    pub rw_config: ReadWriteConfig,
    pub one_hot_config: OneHotConfig,
    pub dory_layout: DoryLayout,
}

impl<F: JoltField, C: JoltCurve<F = F>, PCS: CommitmentScheme<Field = F>, FS: Transcript>
    CanonicalSerialize for JoltProof<F, C, PCS, FS>
{
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        serialize_vec_with_len(&self.commitments, &mut writer, compress)?;
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
        #[cfg(feature = "zk")]
        self.blindfold_proof
            .serialize_with_mode(&mut writer, compress)?;
        self.joint_opening_proof
            .serialize_with_mode(&mut writer, compress)?;
        self.untrusted_advice_commitment
            .serialize_with_mode(&mut writer, compress)?;
        #[cfg(not(feature = "zk"))]
        self.opening_claims
            .serialize_with_mode(&mut writer, compress)?;
        self.trace_length
            .serialize_with_mode(&mut writer, compress)?;
        self.ram_K.serialize_with_mode(&mut writer, compress)?;
        self.rw_config.serialize_with_mode(&mut writer, compress)?;
        self.one_hot_config
            .serialize_with_mode(&mut writer, compress)?;
        self.dory_layout.serialize_with_mode(writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        serialized_vec_with_len_size(&self.commitments, compress)
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
            + {
                #[cfg(feature = "zk")]
                {
                    self.blindfold_proof.serialized_size(compress)
                }
                #[cfg(not(feature = "zk"))]
                {
                    0
                }
            }
            + self.joint_opening_proof.serialized_size(compress)
            + self.untrusted_advice_commitment.serialized_size(compress)
            + {
                #[cfg(not(feature = "zk"))]
                {
                    self.opening_claims.serialized_size(compress)
                }
                #[cfg(feature = "zk")]
                {
                    0
                }
            }
            + self.trace_length.serialized_size(compress)
            + self.ram_K.serialized_size(compress)
            + self.rw_config.serialized_size(compress)
            + self.one_hot_config.serialized_size(compress)
            + self.dory_layout.serialized_size(compress)
    }
}

impl<F: JoltField, C: JoltCurve<F = F>, PCS: CommitmentScheme<Field = F>, FS: Transcript> Valid
    for JoltProof<F, C, PCS, FS>
{
    fn check(&self) -> Result<(), SerializationError> {
        self.commitments.check()?;
        self.stage1_uni_skip_first_round_proof.check()?;
        self.stage1_sumcheck_proof.check()?;
        self.stage2_uni_skip_first_round_proof.check()?;
        self.stage2_sumcheck_proof.check()?;
        self.stage3_sumcheck_proof.check()?;
        self.stage4_sumcheck_proof.check()?;
        self.stage5_sumcheck_proof.check()?;
        self.stage6_sumcheck_proof.check()?;
        self.stage7_sumcheck_proof.check()?;
        #[cfg(feature = "zk")]
        self.blindfold_proof.check()?;
        self.joint_opening_proof.check()?;
        self.untrusted_advice_commitment.check()?;
        #[cfg(not(feature = "zk"))]
        self.opening_claims.check()?;
        self.rw_config.check()?;
        self.one_hot_config.check()?;
        self.dory_layout.check()
    }
}

impl<F: JoltField, C: JoltCurve<F = F>, PCS: CommitmentScheme<Field = F>, FS: Transcript>
    CanonicalDeserialize for JoltProof<F, C, PCS, FS>
{
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let proof = Self {
            commitments: deserialize_bounded_vec(
                &mut reader,
                compress,
                validate,
                MAX_JOLT_COMMITMENTS,
            )?,
            stage1_uni_skip_first_round_proof:
                UniSkipFirstRoundProofVariant::deserialize_with_mode(
                    &mut reader,
                    compress,
                    validate,
                )?,
            stage1_sumcheck_proof: SumcheckInstanceProof::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
            )?,
            stage2_uni_skip_first_round_proof:
                UniSkipFirstRoundProofVariant::deserialize_with_mode(
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
            #[cfg(feature = "zk")]
            blindfold_proof: BlindFoldProof::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
            )?,
            joint_opening_proof: PCS::Proof::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
            )?,
            untrusted_advice_commitment: Option::<PCS::Commitment>::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
            )?,
            #[cfg(not(feature = "zk"))]
            opening_claims: Claims::deserialize_with_mode(&mut reader, compress, validate)?,
            trace_length: usize::deserialize_with_mode(&mut reader, compress, validate)?,
            ram_K: usize::deserialize_with_mode(&mut reader, compress, validate)?,
            rw_config: ReadWriteConfig::deserialize_with_mode(&mut reader, compress, validate)?,
            one_hot_config: OneHotConfig::deserialize_with_mode(&mut reader, compress, validate)?,
            dory_layout: DoryLayout::deserialize_with_mode(&mut reader, compress, validate)?,
        };
        if validate == Validate::Yes {
            proof.check()?;
        }
        Ok(proof)
    }
}

#[cfg(not(feature = "zk"))]
#[derive(Clone)]
pub struct Claims<F: JoltField>(pub Openings<F>);

#[cfg(not(feature = "zk"))]
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

#[cfg(not(feature = "zk"))]
impl<F: JoltField> Valid for Claims<F> {
    fn check(&self) -> Result<(), SerializationError> {
        self.0
            .iter()
            .try_for_each(|(id, (_point, claim))| -> Result<(), SerializationError> {
                id.check()?;
                claim.check()
            })
    }
}

#[cfg(not(feature = "zk"))]
impl<F: JoltField> CanonicalDeserialize for Claims<F> {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let size = usize::deserialize_with_mode(&mut reader, compress, validate)?;
        if size > MAX_OPENING_CLAIMS {
            return Err(SerializationError::InvalidData);
        }
        let mut claims = BTreeMap::new();
        for _ in 0..size {
            let key = OpeningId::deserialize_with_mode(&mut reader, compress, validate)?;
            let claim = F::deserialize_with_mode(&mut reader, compress, validate)?;
            if claims
                .insert(key, (OpeningPoint::default(), claim))
                .is_some()
            {
                return Err(SerializationError::InvalidData);
            }
        }
        let claims = Claims(claims);
        if validate == Validate::Yes {
            claims.check()?;
        }
        Ok(claims)
    }
}

impl CanonicalSerialize for DoryLayout {
    fn serialize_with_mode<W: Write>(
        &self,
        writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        u8::from(*self).serialize_with_mode(writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        u8::from(*self).serialized_size(compress)
    }
}

impl Valid for DoryLayout {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl CanonicalDeserialize for DoryLayout {
    fn deserialize_with_mode<R: Read>(
        reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let value = u8::deserialize_with_mode(reader, compress, validate)?;
        if value > 1 {
            return Err(SerializationError::InvalidData);
        }
        Ok(DoryLayout::from(value))
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
            OpeningId::Polynomial(PolynomialId::Committed(committed_polynomial), sumcheck_id) => {
                let fused = OPENING_ID_COMMITTED_BASE + (*sumcheck_id as u8);
                fused.serialize_with_mode(&mut writer, compress)?;
                committed_polynomial.serialize_with_mode(&mut writer, compress)
            }
            OpeningId::Polynomial(PolynomialId::Virtual(virtual_polynomial), sumcheck_id) => {
                let fused = OPENING_ID_VIRTUAL_BASE + (*sumcheck_id as u8);
                fused.serialize_with_mode(&mut writer, compress)?;
                virtual_polynomial.serialize_with_mode(&mut writer, compress)
            }
        }
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        match self {
            OpeningId::UntrustedAdvice(_) | OpeningId::TrustedAdvice(_) => 1,
            OpeningId::Polynomial(PolynomialId::Committed(committed_polynomial), _) => {
                1 + committed_polynomial.serialized_size(compress)
            }
            OpeningId::Polynomial(PolynomialId::Virtual(virtual_polynomial), _) => {
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
                Ok(OpeningId::committed(
                    polynomial,
                    SumcheckId::from_u8(sumcheck_id).ok_or(SerializationError::InvalidData)?,
                ))
            }
            _ => {
                let sumcheck_id = fused - OPENING_ID_VIRTUAL_BASE;
                let polynomial =
                    VirtualPolynomial::deserialize_with_mode(&mut reader, compress, validate)?;
                Ok(OpeningId::virt(
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
            Self::TrustedAdvice => 5u8.serialize_with_mode(writer, compress),
            Self::UntrustedAdvice => 6u8.serialize_with_mode(writer, compress),
        }
    }

    fn serialized_size(&self, _compress: Compress) -> usize {
        match self {
            Self::RdInc | Self::RamInc | Self::TrustedAdvice | Self::UntrustedAdvice => 1,
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
                5 => Self::TrustedAdvice,
                6 => Self::UntrustedAdvice,
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
            Self::Rd => 14u8.serialize_with_mode(&mut writer, compress),
            Self::Imm => 15u8.serialize_with_mode(&mut writer, compress),
            Self::Rs1Value => 16u8.serialize_with_mode(&mut writer, compress),
            Self::Rs2Value => 17u8.serialize_with_mode(&mut writer, compress),
            Self::RdWriteValue => 18u8.serialize_with_mode(&mut writer, compress),
            Self::Rs1Ra => 19u8.serialize_with_mode(&mut writer, compress),
            Self::Rs2Ra => 20u8.serialize_with_mode(&mut writer, compress),
            Self::RdWa => 21u8.serialize_with_mode(&mut writer, compress),
            Self::LookupOutput => 22u8.serialize_with_mode(&mut writer, compress),
            Self::InstructionRaf => 23u8.serialize_with_mode(&mut writer, compress),
            Self::InstructionRafFlag => 24u8.serialize_with_mode(&mut writer, compress),
            Self::InstructionRa(i) => {
                25u8.serialize_with_mode(&mut writer, compress)?;
                (u8::try_from(*i).unwrap()).serialize_with_mode(&mut writer, compress)
            }
            Self::RegistersVal => 26u8.serialize_with_mode(&mut writer, compress),
            Self::RamAddress => 27u8.serialize_with_mode(&mut writer, compress),
            Self::RamRa => 28u8.serialize_with_mode(&mut writer, compress),
            Self::RamReadValue => 29u8.serialize_with_mode(&mut writer, compress),
            Self::RamWriteValue => 30u8.serialize_with_mode(&mut writer, compress),
            Self::RamVal => 31u8.serialize_with_mode(&mut writer, compress),
            Self::RamValInit => 32u8.serialize_with_mode(&mut writer, compress),
            Self::RamValFinal => 33u8.serialize_with_mode(&mut writer, compress),
            Self::RamHammingWeight => 34u8.serialize_with_mode(&mut writer, compress),
            Self::UnivariateSkip => 35u8.serialize_with_mode(&mut writer, compress),
            Self::OpFlags(flags) => {
                36u8.serialize_with_mode(&mut writer, compress)?;
                (u8::try_from(*flags as usize).unwrap()).serialize_with_mode(&mut writer, compress)
            }
            Self::InstructionFlags(flags) => {
                37u8.serialize_with_mode(&mut writer, compress)?;
                (u8::try_from(*flags as usize).unwrap()).serialize_with_mode(&mut writer, compress)
            }
            Self::LookupTableFlag(flag) => {
                38u8.serialize_with_mode(&mut writer, compress)?;
                (u8::try_from(*flag).unwrap()).serialize_with_mode(&mut writer, compress)
            }
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
            | Self::UnivariateSkip => 1,
            Self::InstructionRa(_)
            | Self::OpFlags(_)
            | Self::InstructionFlags(_)
            | Self::LookupTableFlag(_) => 2,
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
                14 => Self::Rd,
                15 => Self::Imm,
                16 => Self::Rs1Value,
                17 => Self::Rs2Value,
                18 => Self::RdWriteValue,
                19 => Self::Rs1Ra,
                20 => Self::Rs2Ra,
                21 => Self::RdWa,
                22 => Self::LookupOutput,
                23 => Self::InstructionRaf,
                24 => Self::InstructionRafFlag,
                25 => {
                    let i = u8::deserialize_with_mode(&mut reader, compress, validate)?;
                    Self::InstructionRa(i as usize)
                }
                26 => Self::RegistersVal,
                27 => Self::RamAddress,
                28 => Self::RamRa,
                29 => Self::RamReadValue,
                30 => Self::RamWriteValue,
                31 => Self::RamVal,
                32 => Self::RamValInit,
                33 => Self::RamValFinal,
                34 => Self::RamHammingWeight,
                35 => Self::UnivariateSkip,
                36 => {
                    let discriminant = u8::deserialize_with_mode(&mut reader, compress, validate)?;
                    let flags = CircuitFlags::from_repr(discriminant)
                        .ok_or(SerializationError::InvalidData)?;
                    Self::OpFlags(flags)
                }
                37 => {
                    let discriminant = u8::deserialize_with_mode(&mut reader, compress, validate)?;
                    let flags = InstructionFlags::from_repr(discriminant)
                        .ok_or(SerializationError::InvalidData)?;
                    Self::InstructionFlags(flags)
                }
                38 => {
                    let flag = u8::deserialize_with_mode(&mut reader, compress, validate)?;
                    Self::LookupTableFlag(flag as usize)
                }
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

#[cfg(all(test, not(feature = "zk")))]
mod tests {
    use super::*;
    use ark_bn254::Fr;

    #[test]
    fn claims_reject_duplicate_keys() {
        let key = OpeningId::committed(
            CommittedPolynomial::InstructionRa(0),
            SumcheckId::HammingWeightClaimReduction,
        );
        let mut bytes = Vec::new();
        2usize.serialize_compressed(&mut bytes).unwrap();
        key.serialize_compressed(&mut bytes).unwrap();
        Fr::from(1u64).serialize_compressed(&mut bytes).unwrap();
        key.serialize_compressed(&mut bytes).unwrap();
        Fr::from(1u64).serialize_compressed(&mut bytes).unwrap();

        assert!(Claims::<Fr>::deserialize_compressed(&bytes[..]).is_err());
    }

    #[test]
    fn claims_reject_oversized_length_prefix() {
        let mut bytes = Vec::new();
        (MAX_OPENING_CLAIMS + 1)
            .serialize_compressed(&mut bytes)
            .unwrap();
        assert!(Claims::<Fr>::deserialize_compressed(&bytes[..]).is_err());
    }
}
