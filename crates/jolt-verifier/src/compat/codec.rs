//! Serialization compatibility for existing `jolt-core` proof artifacts.

use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, Read, SerializationError, Valid, Validate,
    Write,
};
use jolt_riscv::{CircuitFlags, InstructionFlags};

use crate::proof::DoryLayout;

use super::{
    config::{OneHotConfig, ReadWriteConfig},
    ids::{CommittedPolynomial, OpeningId, PolynomialId, SumcheckId, VirtualPolynomial},
};

const OPENING_ID_UNTRUSTED_ADVICE_BASE: u8 = 0;
const OPENING_ID_TRUSTED_ADVICE_BASE: u8 =
    OPENING_ID_UNTRUSTED_ADVICE_BASE + SumcheckId::COUNT as u8;
const OPENING_ID_COMMITTED_BASE: u8 = OPENING_ID_TRUSTED_ADVICE_BASE + SumcheckId::COUNT as u8;
const OPENING_ID_VIRTUAL_BASE: u8 = OPENING_ID_COMMITTED_BASE + SumcheckId::COUNT as u8;

impl CanonicalSerialize for SumcheckId {
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

impl Valid for SumcheckId {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl CanonicalDeserialize for SumcheckId {
    fn deserialize_with_mode<R: Read>(
        reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let value = u8::deserialize_with_mode(reader, compress, validate)?;
        Self::try_from(value).map_err(|()| SerializationError::InvalidData)
    }
}

impl CanonicalSerialize for ReadWriteConfig {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.ram_rw_phase1_num_rounds
            .serialize_with_mode(&mut writer, compress)?;
        self.ram_rw_phase2_num_rounds
            .serialize_with_mode(&mut writer, compress)?;
        self.registers_rw_phase1_num_rounds
            .serialize_with_mode(&mut writer, compress)?;
        self.registers_rw_phase2_num_rounds
            .serialize_with_mode(writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.ram_rw_phase1_num_rounds.serialized_size(compress)
            + self.ram_rw_phase2_num_rounds.serialized_size(compress)
            + self
                .registers_rw_phase1_num_rounds
                .serialized_size(compress)
            + self
                .registers_rw_phase2_num_rounds
                .serialized_size(compress)
    }
}

impl Valid for ReadWriteConfig {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl CanonicalDeserialize for ReadWriteConfig {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        Ok(Self {
            ram_rw_phase1_num_rounds: u8::deserialize_with_mode(&mut reader, compress, validate)?,
            ram_rw_phase2_num_rounds: u8::deserialize_with_mode(&mut reader, compress, validate)?,
            registers_rw_phase1_num_rounds: u8::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
            )?,
            registers_rw_phase2_num_rounds: u8::deserialize_with_mode(reader, compress, validate)?,
        })
    }
}

impl CanonicalSerialize for OneHotConfig {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.log_k_chunk
            .serialize_with_mode(&mut writer, compress)?;
        self.lookups_ra_virtual_log_k_chunk
            .serialize_with_mode(writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.log_k_chunk.serialized_size(compress)
            + self
                .lookups_ra_virtual_log_k_chunk
                .serialized_size(compress)
    }
}

impl Valid for OneHotConfig {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl CanonicalDeserialize for OneHotConfig {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        Ok(Self {
            log_k_chunk: u8::deserialize_with_mode(&mut reader, compress, validate)?,
            lookups_ra_virtual_log_k_chunk: u8::deserialize_with_mode(reader, compress, validate)?,
        })
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
        Self::try_from(value).map_err(|()| SerializationError::InvalidData)
    }
}

impl CanonicalSerialize for OpeningId {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        match self {
            Self::UntrustedAdvice(sumcheck_id) => {
                let fused = OPENING_ID_UNTRUSTED_ADVICE_BASE + u8::from(*sumcheck_id);
                fused.serialize_with_mode(writer, compress)
            }
            Self::TrustedAdvice(sumcheck_id) => {
                let fused = OPENING_ID_TRUSTED_ADVICE_BASE + u8::from(*sumcheck_id);
                fused.serialize_with_mode(writer, compress)
            }
            Self::Polynomial(PolynomialId::Committed(committed_polynomial), sumcheck_id) => {
                let fused = OPENING_ID_COMMITTED_BASE + u8::from(*sumcheck_id);
                fused.serialize_with_mode(&mut writer, compress)?;
                committed_polynomial.serialize_with_mode(writer, compress)
            }
            Self::Polynomial(PolynomialId::Virtual(virtual_polynomial), sumcheck_id) => {
                let fused = OPENING_ID_VIRTUAL_BASE + u8::from(*sumcheck_id);
                fused.serialize_with_mode(&mut writer, compress)?;
                virtual_polynomial.serialize_with_mode(writer, compress)
            }
        }
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        match self {
            Self::UntrustedAdvice(_) | Self::TrustedAdvice(_) => 1,
            Self::Polynomial(PolynomialId::Committed(committed_polynomial), _) => {
                1 + committed_polynomial.serialized_size(compress)
            }
            Self::Polynomial(PolynomialId::Virtual(virtual_polynomial), _) => {
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
                Ok(Self::UntrustedAdvice(
                    SumcheckId::try_from(sumcheck_id)
                        .map_err(|()| SerializationError::InvalidData)?,
                ))
            }
            _ if fused < OPENING_ID_COMMITTED_BASE => {
                let sumcheck_id = fused - OPENING_ID_TRUSTED_ADVICE_BASE;
                Ok(Self::TrustedAdvice(
                    SumcheckId::try_from(sumcheck_id)
                        .map_err(|()| SerializationError::InvalidData)?,
                ))
            }
            _ if fused < OPENING_ID_VIRTUAL_BASE => {
                let sumcheck_id = fused - OPENING_ID_COMMITTED_BASE;
                let polynomial =
                    CommittedPolynomial::deserialize_with_mode(&mut reader, compress, validate)?;
                Ok(Self::committed(
                    polynomial,
                    SumcheckId::try_from(sumcheck_id)
                        .map_err(|()| SerializationError::InvalidData)?,
                ))
            }
            _ => {
                let sumcheck_id = fused - OPENING_ID_VIRTUAL_BASE;
                let polynomial =
                    VirtualPolynomial::deserialize_with_mode(&mut reader, compress, validate)?;
                Ok(Self::virt(
                    polynomial,
                    SumcheckId::try_from(sumcheck_id)
                        .map_err(|()| SerializationError::InvalidData)?,
                ))
            }
        }
    }
}

impl CanonicalSerialize for CommittedPolynomial {
    fn serialize_with_mode<W: Write>(
        &self,
        writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        match self {
            Self::RdInc => 0u8.serialize_with_mode(writer, compress),
            Self::RamInc => 1u8.serialize_with_mode(writer, compress),
            Self::InstructionRa(i) => serialize_tagged_index(2, *i, writer, compress),
            Self::BytecodeRa(i) => serialize_tagged_index(3, *i, writer, compress),
            Self::RamRa(i) => serialize_tagged_index(4, *i, writer, compress),
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
                2 => Self::InstructionRa(read_index(reader, compress, validate)?),
                3 => Self::BytecodeRa(read_index(reader, compress, validate)?),
                4 => Self::RamRa(read_index(reader, compress, validate)?),
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
        writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        match self {
            Self::PC => 0u8.serialize_with_mode(writer, compress),
            Self::UnexpandedPC => 1u8.serialize_with_mode(writer, compress),
            Self::NextPC => 2u8.serialize_with_mode(writer, compress),
            Self::NextUnexpandedPC => 3u8.serialize_with_mode(writer, compress),
            Self::NextIsNoop => 4u8.serialize_with_mode(writer, compress),
            Self::NextIsVirtual => 5u8.serialize_with_mode(writer, compress),
            Self::NextIsFirstInSequence => 6u8.serialize_with_mode(writer, compress),
            Self::LeftLookupOperand => 7u8.serialize_with_mode(writer, compress),
            Self::RightLookupOperand => 8u8.serialize_with_mode(writer, compress),
            Self::LeftInstructionInput => 9u8.serialize_with_mode(writer, compress),
            Self::RightInstructionInput => 10u8.serialize_with_mode(writer, compress),
            Self::Product => 11u8.serialize_with_mode(writer, compress),
            Self::ShouldJump => 12u8.serialize_with_mode(writer, compress),
            Self::ShouldBranch => 13u8.serialize_with_mode(writer, compress),
            Self::Rd => 14u8.serialize_with_mode(writer, compress),
            Self::Imm => 15u8.serialize_with_mode(writer, compress),
            Self::Rs1Value => 16u8.serialize_with_mode(writer, compress),
            Self::Rs2Value => 17u8.serialize_with_mode(writer, compress),
            Self::RdWriteValue => 18u8.serialize_with_mode(writer, compress),
            Self::Rs1Ra => 19u8.serialize_with_mode(writer, compress),
            Self::Rs2Ra => 20u8.serialize_with_mode(writer, compress),
            Self::RdWa => 21u8.serialize_with_mode(writer, compress),
            Self::LookupOutput => 22u8.serialize_with_mode(writer, compress),
            Self::InstructionRaf => 23u8.serialize_with_mode(writer, compress),
            Self::InstructionRafFlag => 24u8.serialize_with_mode(writer, compress),
            Self::InstructionRa(i) => serialize_tagged_index(25, *i, writer, compress),
            Self::RegistersVal => 26u8.serialize_with_mode(writer, compress),
            Self::RamAddress => 27u8.serialize_with_mode(writer, compress),
            Self::RamRa => 28u8.serialize_with_mode(writer, compress),
            Self::RamReadValue => 29u8.serialize_with_mode(writer, compress),
            Self::RamWriteValue => 30u8.serialize_with_mode(writer, compress),
            Self::RamVal => 31u8.serialize_with_mode(writer, compress),
            Self::RamValInit => 32u8.serialize_with_mode(writer, compress),
            Self::RamValFinal => 33u8.serialize_with_mode(writer, compress),
            Self::RamHammingWeight => 34u8.serialize_with_mode(writer, compress),
            Self::UnivariateSkip => 35u8.serialize_with_mode(writer, compress),
            Self::OpFlags(flags) => serialize_tagged_index(36, *flags as usize, writer, compress),
            Self::InstructionFlags(flags) => {
                serialize_tagged_index(37, *flags as usize, writer, compress)
            }
            Self::LookupTableFlag(flag) => serialize_tagged_index(38, *flag, writer, compress),
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
                25 => Self::InstructionRa(read_index(&mut reader, compress, validate)?),
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
                    Self::OpFlags(
                        circuit_flag_from_u8(discriminant)
                            .ok_or(SerializationError::InvalidData)?,
                    )
                }
                37 => {
                    let discriminant = u8::deserialize_with_mode(&mut reader, compress, validate)?;
                    Self::InstructionFlags(
                        instruction_flag_from_u8(discriminant)
                            .ok_or(SerializationError::InvalidData)?,
                    )
                }
                38 => Self::LookupTableFlag(read_index(reader, compress, validate)?),
                _ => return Err(SerializationError::InvalidData),
            },
        )
    }
}

fn serialize_tagged_index<W: Write>(
    tag: u8,
    index: usize,
    mut writer: W,
    compress: Compress,
) -> Result<(), SerializationError> {
    tag.serialize_with_mode(&mut writer, compress)?;
    let index = u8::try_from(index).map_err(|_| SerializationError::InvalidData)?;
    index.serialize_with_mode(writer, compress)
}

fn read_index<R: Read>(
    reader: R,
    compress: Compress,
    validate: Validate,
) -> Result<usize, SerializationError> {
    Ok(u8::deserialize_with_mode(reader, compress, validate)? as usize)
}

fn circuit_flag_from_u8(value: u8) -> Option<CircuitFlags> {
    match value {
        value if value == CircuitFlags::AddOperands as u8 => Some(CircuitFlags::AddOperands),
        value if value == CircuitFlags::SubtractOperands as u8 => {
            Some(CircuitFlags::SubtractOperands)
        }
        value if value == CircuitFlags::MultiplyOperands as u8 => {
            Some(CircuitFlags::MultiplyOperands)
        }
        value if value == CircuitFlags::Load as u8 => Some(CircuitFlags::Load),
        value if value == CircuitFlags::Store as u8 => Some(CircuitFlags::Store),
        value if value == CircuitFlags::Jump as u8 => Some(CircuitFlags::Jump),
        value if value == CircuitFlags::WriteLookupOutputToRD as u8 => {
            Some(CircuitFlags::WriteLookupOutputToRD)
        }
        value if value == CircuitFlags::VirtualInstruction as u8 => {
            Some(CircuitFlags::VirtualInstruction)
        }
        value if value == CircuitFlags::Assert as u8 => Some(CircuitFlags::Assert),
        value if value == CircuitFlags::DoNotUpdateUnexpandedPC as u8 => {
            Some(CircuitFlags::DoNotUpdateUnexpandedPC)
        }
        value if value == CircuitFlags::Advice as u8 => Some(CircuitFlags::Advice),
        value if value == CircuitFlags::IsCompressed as u8 => Some(CircuitFlags::IsCompressed),
        value if value == CircuitFlags::IsFirstInSequence as u8 => {
            Some(CircuitFlags::IsFirstInSequence)
        }
        value if value == CircuitFlags::IsLastInSequence as u8 => {
            Some(CircuitFlags::IsLastInSequence)
        }
        _ => None,
    }
}

fn instruction_flag_from_u8(value: u8) -> Option<InstructionFlags> {
    match value {
        value if value == InstructionFlags::LeftOperandIsPC as u8 => {
            Some(InstructionFlags::LeftOperandIsPC)
        }
        value if value == InstructionFlags::RightOperandIsImm as u8 => {
            Some(InstructionFlags::RightOperandIsImm)
        }
        value if value == InstructionFlags::LeftOperandIsRs1Value as u8 => {
            Some(InstructionFlags::LeftOperandIsRs1Value)
        }
        value if value == InstructionFlags::RightOperandIsRs2Value as u8 => {
            Some(InstructionFlags::RightOperandIsRs2Value)
        }
        value if value == InstructionFlags::Branch as u8 => Some(InstructionFlags::Branch),
        value if value == InstructionFlags::IsNoop as u8 => Some(InstructionFlags::IsNoop),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
    use jolt_core::{
        poly::commitment::dory::DoryLayout as CoreDoryLayout,
        poly::opening_proof::{
            OpeningId as CoreOpeningId, PolynomialId as CorePolynomialId,
            SumcheckId as CoreSumcheckId,
        },
        zkvm::{
            config::{
                OneHotConfig as CoreOneHotConfig, OneHotParams as CoreOneHotParams,
                ReadWriteConfig as CoreReadWriteConfig,
            },
            instruction::{
                CircuitFlags as CoreCircuitFlags, InstructionFlags as CoreInstructionFlags,
            },
            witness::{
                CommittedPolynomial as CoreCommittedPolynomial,
                VirtualPolynomial as CoreVirtualPolynomial,
            },
        },
    };

    use super::*;

    #[test]
    fn sumcheck_ids_round_trip() -> Result<(), SerializationError> {
        for id in SumcheckId::ALL {
            round_trip(id)?;
        }
        assert!(SumcheckId::deserialize_compressed([SumcheckId::COUNT as u8].as_slice()).is_err());
        Ok(())
    }

    #[test]
    fn configs_match_jolt_core() -> Result<(), SerializationError> {
        for config in read_write_config_cases() {
            round_trip(config)?;
            assert_eq!(bytes(&config)?, bytes(&core_read_write_config(&config))?);
        }

        for config in one_hot_config_cases() {
            round_trip(config)?;
            assert_eq!(bytes(&config)?, bytes(&core_one_hot_config(&config))?);
        }

        Ok(())
    }

    #[test]
    fn dory_layout_matches_jolt_core() -> Result<(), SerializationError> {
        for (layout, core_layout) in [
            (DoryLayout::CycleMajor, CoreDoryLayout::CycleMajor),
            (DoryLayout::AddressMajor, CoreDoryLayout::AddressMajor),
        ] {
            round_trip(layout)?;
            assert_eq!(bytes(&layout)?, bytes(&core_layout)?);
        }

        assert!(DoryLayout::deserialize_compressed([2u8].as_slice()).is_err());
        assert_eq!(
            DoryLayout::CycleMajor.address_cycle_to_index(3, 4, 10, 20),
            64
        );
        assert_eq!(
            DoryLayout::AddressMajor.address_cycle_to_index(3, 4, 10, 20),
            43
        );
        assert_eq!(
            DoryLayout::CycleMajor.index_to_address_cycle(64, 10, 20),
            (3, 4)
        );
        assert_eq!(
            DoryLayout::AddressMajor.index_to_address_cycle(43, 10, 20),
            (3, 4)
        );
        Ok(())
    }

    #[test]
    fn one_hot_params_match_jolt_core() -> Result<(), String> {
        for (config, bytecode_k, ram_k) in [
            (
                OneHotConfig {
                    log_k_chunk: 4,
                    lookups_ra_virtual_log_k_chunk: 16,
                },
                100,
                1 << 20,
            ),
            (
                OneHotConfig {
                    log_k_chunk: 8,
                    lookups_ra_virtual_log_k_chunk: 32,
                },
                1024,
                4096,
            ),
        ] {
            let params = super::super::config::OneHotParams::try_from((config, bytecode_k, ram_k))?;
            let core_params =
                CoreOneHotParams::from_config(&core_one_hot_config(&config), bytecode_k, ram_k);

            assert_eq!(params.log_k_chunk, core_params.log_k_chunk);
            assert_eq!(
                params.lookups_ra_virtual_log_k_chunk,
                core_params.lookups_ra_virtual_log_k_chunk
            );
            assert_eq!(params.k_chunk, core_params.k_chunk);
            assert_eq!(params.bytecode_k, core_params.bytecode_k);
            assert_eq!(params.ram_k, core_params.ram_k);
            assert_eq!(params.instruction_d, core_params.instruction_d);
            assert_eq!(params.bytecode_d, core_params.bytecode_d);
            assert_eq!(params.ram_d, core_params.ram_d);
            assert_eq!(OneHotConfig::try_from(&params), Ok(config));

            for idx in 0..params.instruction_d {
                assert_eq!(
                    params.lookup_index_chunk(0x1234_5678_90ab_cdef_u128, idx),
                    Some(core_params.lookup_index_chunk(0x1234_5678_90ab_cdef_u128, idx))
                );
            }
            for idx in 0..params.bytecode_d {
                assert_eq!(
                    params.bytecode_pc_chunk(0x1234_5678, idx),
                    Some(core_params.bytecode_pc_chunk(0x1234_5678, idx))
                );
            }
            for idx in 0..params.ram_d {
                assert_eq!(
                    params.ram_address_chunk(0x1234_5678, idx),
                    Some(core_params.ram_address_chunk(0x1234_5678, idx))
                );
            }
        }
        Ok(())
    }

    #[test]
    fn committed_polynomial_ids_match_jolt_core() -> Result<(), SerializationError> {
        for polynomial in committed_polynomial_cases() {
            round_trip(polynomial)?;
            assert_eq!(bytes(&polynomial)?, bytes(&core_committed(polynomial))?);
        }
        assert!(CommittedPolynomial::deserialize_compressed([7u8].as_slice()).is_err());
        assert!(bytes(&CommittedPolynomial::InstructionRa(256)).is_err());
        Ok(())
    }

    #[test]
    fn virtual_polynomial_ids_match_jolt_core() -> Result<(), SerializationError> {
        for polynomial in virtual_polynomial_cases() {
            round_trip(polynomial)?;
            assert_eq!(bytes(&polynomial)?, bytes(&core_virtual(polynomial))?);
        }
        assert!(VirtualPolynomial::deserialize_compressed([39u8].as_slice()).is_err());
        assert!(VirtualPolynomial::deserialize_compressed([36u8, 14u8].as_slice()).is_err());
        assert!(VirtualPolynomial::deserialize_compressed([37u8, 6u8].as_slice()).is_err());
        assert!(bytes(&VirtualPolynomial::InstructionRa(256)).is_err());
        Ok(())
    }

    #[test]
    fn opening_ids_match_jolt_core() -> Result<(), SerializationError> {
        for sumcheck_id in SumcheckId::ALL {
            let ids = [
                OpeningId::UntrustedAdvice(sumcheck_id),
                OpeningId::TrustedAdvice(sumcheck_id),
                OpeningId::committed(CommittedPolynomial::RdInc, sumcheck_id),
                OpeningId::virt(VirtualPolynomial::PC, sumcheck_id),
            ];

            for id in ids {
                round_trip(id)?;
                assert_eq!(bytes(&id)?, bytes(&core_opening(id))?);
            }
        }

        for polynomial in committed_polynomial_cases() {
            let id = OpeningId::committed(polynomial, SumcheckId::RamReadWriteChecking);
            round_trip(id)?;
            assert_eq!(bytes(&id)?, bytes(&core_opening(id))?);
        }

        for polynomial in virtual_polynomial_cases() {
            let id = OpeningId::virt(polynomial, SumcheckId::RamReadWriteChecking);
            round_trip(id)?;
            assert_eq!(bytes(&id)?, bytes(&core_opening(id))?);
        }

        let invalid_sumcheck = OPENING_ID_VIRTUAL_BASE + SumcheckId::COUNT as u8;
        assert!(OpeningId::deserialize_compressed([invalid_sumcheck, 0u8].as_slice()).is_err());
        Ok(())
    }

    fn round_trip<T>(value: T) -> Result<(), SerializationError>
    where
        T: CanonicalSerialize + CanonicalDeserialize + PartialEq + core::fmt::Debug,
    {
        let encoded = bytes(&value)?;
        let decoded = T::deserialize_compressed(encoded.as_slice())?;
        assert_eq!(value, decoded);
        assert_eq!(encoded.len(), value.serialized_size(Compress::Yes));
        Ok(())
    }

    fn bytes(value: &impl CanonicalSerialize) -> Result<Vec<u8>, SerializationError> {
        let mut bytes = Vec::new();
        value.serialize_compressed(&mut bytes)?;
        Ok(bytes)
    }

    fn read_write_config_cases() -> Vec<ReadWriteConfig> {
        vec![
            ReadWriteConfig {
                ram_rw_phase1_num_rounds: 10,
                ram_rw_phase2_num_rounds: 12,
                registers_rw_phase1_num_rounds: 10,
                registers_rw_phase2_num_rounds: common::constants::REGISTER_COUNT.ilog2() as u8,
            },
            ReadWriteConfig {
                ram_rw_phase1_num_rounds: 4,
                ram_rw_phase2_num_rounds: 8,
                registers_rw_phase1_num_rounds: 6,
                registers_rw_phase2_num_rounds: 7,
            },
        ]
    }

    fn one_hot_config_cases() -> Vec<OneHotConfig> {
        vec![
            OneHotConfig::from(10),
            OneHotConfig::from(30),
            OneHotConfig {
                log_k_chunk: 4,
                lookups_ra_virtual_log_k_chunk: 32,
            },
        ]
    }

    fn committed_polynomial_cases() -> Vec<CommittedPolynomial> {
        vec![
            CommittedPolynomial::RdInc,
            CommittedPolynomial::RamInc,
            CommittedPolynomial::InstructionRa(0),
            CommittedPolynomial::InstructionRa(7),
            CommittedPolynomial::InstructionRa(255),
            CommittedPolynomial::BytecodeRa(0),
            CommittedPolynomial::BytecodeRa(7),
            CommittedPolynomial::BytecodeRa(255),
            CommittedPolynomial::RamRa(0),
            CommittedPolynomial::RamRa(7),
            CommittedPolynomial::RamRa(255),
            CommittedPolynomial::TrustedAdvice,
            CommittedPolynomial::UntrustedAdvice,
        ]
    }

    fn virtual_polynomial_cases() -> Vec<VirtualPolynomial> {
        let mut polynomials = vec![
            VirtualPolynomial::PC,
            VirtualPolynomial::UnexpandedPC,
            VirtualPolynomial::NextPC,
            VirtualPolynomial::NextUnexpandedPC,
            VirtualPolynomial::NextIsNoop,
            VirtualPolynomial::NextIsVirtual,
            VirtualPolynomial::NextIsFirstInSequence,
            VirtualPolynomial::LeftLookupOperand,
            VirtualPolynomial::RightLookupOperand,
            VirtualPolynomial::LeftInstructionInput,
            VirtualPolynomial::RightInstructionInput,
            VirtualPolynomial::Product,
            VirtualPolynomial::ShouldJump,
            VirtualPolynomial::ShouldBranch,
            VirtualPolynomial::Rd,
            VirtualPolynomial::Imm,
            VirtualPolynomial::Rs1Value,
            VirtualPolynomial::Rs2Value,
            VirtualPolynomial::RdWriteValue,
            VirtualPolynomial::Rs1Ra,
            VirtualPolynomial::Rs2Ra,
            VirtualPolynomial::RdWa,
            VirtualPolynomial::LookupOutput,
            VirtualPolynomial::InstructionRaf,
            VirtualPolynomial::InstructionRafFlag,
            VirtualPolynomial::InstructionRa(0),
            VirtualPolynomial::InstructionRa(7),
            VirtualPolynomial::InstructionRa(255),
            VirtualPolynomial::RegistersVal,
            VirtualPolynomial::RamAddress,
            VirtualPolynomial::RamRa,
            VirtualPolynomial::RamReadValue,
            VirtualPolynomial::RamWriteValue,
            VirtualPolynomial::RamVal,
            VirtualPolynomial::RamValInit,
            VirtualPolynomial::RamValFinal,
            VirtualPolynomial::RamHammingWeight,
            VirtualPolynomial::UnivariateSkip,
            VirtualPolynomial::LookupTableFlag(0),
            VirtualPolynomial::LookupTableFlag(7),
            VirtualPolynomial::LookupTableFlag(255),
        ];

        for flag in circuit_flag_cases() {
            polynomials.push(VirtualPolynomial::OpFlags(flag));
        }
        for flag in instruction_flag_cases() {
            polynomials.push(VirtualPolynomial::InstructionFlags(flag));
        }

        polynomials
    }

    fn circuit_flag_cases() -> [CircuitFlags; 14] {
        [
            CircuitFlags::AddOperands,
            CircuitFlags::SubtractOperands,
            CircuitFlags::MultiplyOperands,
            CircuitFlags::Load,
            CircuitFlags::Store,
            CircuitFlags::Jump,
            CircuitFlags::WriteLookupOutputToRD,
            CircuitFlags::VirtualInstruction,
            CircuitFlags::Assert,
            CircuitFlags::DoNotUpdateUnexpandedPC,
            CircuitFlags::Advice,
            CircuitFlags::IsCompressed,
            CircuitFlags::IsFirstInSequence,
            CircuitFlags::IsLastInSequence,
        ]
    }

    fn instruction_flag_cases() -> [InstructionFlags; 6] {
        [
            InstructionFlags::LeftOperandIsPC,
            InstructionFlags::RightOperandIsImm,
            InstructionFlags::LeftOperandIsRs1Value,
            InstructionFlags::RightOperandIsRs2Value,
            InstructionFlags::Branch,
            InstructionFlags::IsNoop,
        ]
    }

    fn core_sumcheck(id: SumcheckId) -> CoreSumcheckId {
        match id {
            SumcheckId::SpartanOuter => CoreSumcheckId::SpartanOuter,
            SumcheckId::SpartanProductVirtualization => {
                CoreSumcheckId::SpartanProductVirtualization
            }
            SumcheckId::SpartanShift => CoreSumcheckId::SpartanShift,
            SumcheckId::InstructionClaimReduction => CoreSumcheckId::InstructionClaimReduction,
            SumcheckId::InstructionInputVirtualization => {
                CoreSumcheckId::InstructionInputVirtualization
            }
            SumcheckId::InstructionReadRaf => CoreSumcheckId::InstructionReadRaf,
            SumcheckId::InstructionRaVirtualization => CoreSumcheckId::InstructionRaVirtualization,
            SumcheckId::RamReadWriteChecking => CoreSumcheckId::RamReadWriteChecking,
            SumcheckId::RamRafEvaluation => CoreSumcheckId::RamRafEvaluation,
            SumcheckId::RamOutputCheck => CoreSumcheckId::RamOutputCheck,
            SumcheckId::RamValCheck => CoreSumcheckId::RamValCheck,
            SumcheckId::RamRaClaimReduction => CoreSumcheckId::RamRaClaimReduction,
            SumcheckId::RamHammingBooleanity => CoreSumcheckId::RamHammingBooleanity,
            SumcheckId::RamRaVirtualization => CoreSumcheckId::RamRaVirtualization,
            SumcheckId::RegistersClaimReduction => CoreSumcheckId::RegistersClaimReduction,
            SumcheckId::RegistersReadWriteChecking => CoreSumcheckId::RegistersReadWriteChecking,
            SumcheckId::RegistersValEvaluation => CoreSumcheckId::RegistersValEvaluation,
            SumcheckId::BytecodeReadRaf => CoreSumcheckId::BytecodeReadRaf,
            SumcheckId::Booleanity => CoreSumcheckId::Booleanity,
            SumcheckId::AdviceClaimReductionCyclePhase => {
                CoreSumcheckId::AdviceClaimReductionCyclePhase
            }
            SumcheckId::AdviceClaimReduction => CoreSumcheckId::AdviceClaimReduction,
            SumcheckId::IncClaimReduction => CoreSumcheckId::IncClaimReduction,
            SumcheckId::HammingWeightClaimReduction => CoreSumcheckId::HammingWeightClaimReduction,
        }
    }

    fn core_read_write_config(config: &ReadWriteConfig) -> CoreReadWriteConfig {
        CoreReadWriteConfig {
            ram_rw_phase1_num_rounds: config.ram_rw_phase1_num_rounds,
            ram_rw_phase2_num_rounds: config.ram_rw_phase2_num_rounds,
            registers_rw_phase1_num_rounds: config.registers_rw_phase1_num_rounds,
            registers_rw_phase2_num_rounds: config.registers_rw_phase2_num_rounds,
        }
    }

    fn core_one_hot_config(config: &OneHotConfig) -> CoreOneHotConfig {
        CoreOneHotConfig {
            log_k_chunk: config.log_k_chunk,
            lookups_ra_virtual_log_k_chunk: config.lookups_ra_virtual_log_k_chunk,
        }
    }

    fn core_committed(polynomial: CommittedPolynomial) -> CoreCommittedPolynomial {
        match polynomial {
            CommittedPolynomial::RdInc => CoreCommittedPolynomial::RdInc,
            CommittedPolynomial::RamInc => CoreCommittedPolynomial::RamInc,
            CommittedPolynomial::InstructionRa(i) => CoreCommittedPolynomial::InstructionRa(i),
            CommittedPolynomial::BytecodeRa(i) => CoreCommittedPolynomial::BytecodeRa(i),
            CommittedPolynomial::RamRa(i) => CoreCommittedPolynomial::RamRa(i),
            CommittedPolynomial::TrustedAdvice => CoreCommittedPolynomial::TrustedAdvice,
            CommittedPolynomial::UntrustedAdvice => CoreCommittedPolynomial::UntrustedAdvice,
        }
    }

    fn core_virtual(polynomial: VirtualPolynomial) -> CoreVirtualPolynomial {
        match polynomial {
            VirtualPolynomial::PC => CoreVirtualPolynomial::PC,
            VirtualPolynomial::UnexpandedPC => CoreVirtualPolynomial::UnexpandedPC,
            VirtualPolynomial::NextPC => CoreVirtualPolynomial::NextPC,
            VirtualPolynomial::NextUnexpandedPC => CoreVirtualPolynomial::NextUnexpandedPC,
            VirtualPolynomial::NextIsNoop => CoreVirtualPolynomial::NextIsNoop,
            VirtualPolynomial::NextIsVirtual => CoreVirtualPolynomial::NextIsVirtual,
            VirtualPolynomial::NextIsFirstInSequence => {
                CoreVirtualPolynomial::NextIsFirstInSequence
            }
            VirtualPolynomial::LeftLookupOperand => CoreVirtualPolynomial::LeftLookupOperand,
            VirtualPolynomial::RightLookupOperand => CoreVirtualPolynomial::RightLookupOperand,
            VirtualPolynomial::LeftInstructionInput => CoreVirtualPolynomial::LeftInstructionInput,
            VirtualPolynomial::RightInstructionInput => {
                CoreVirtualPolynomial::RightInstructionInput
            }
            VirtualPolynomial::Product => CoreVirtualPolynomial::Product,
            VirtualPolynomial::ShouldJump => CoreVirtualPolynomial::ShouldJump,
            VirtualPolynomial::ShouldBranch => CoreVirtualPolynomial::ShouldBranch,
            VirtualPolynomial::Rd => CoreVirtualPolynomial::Rd,
            VirtualPolynomial::Imm => CoreVirtualPolynomial::Imm,
            VirtualPolynomial::Rs1Value => CoreVirtualPolynomial::Rs1Value,
            VirtualPolynomial::Rs2Value => CoreVirtualPolynomial::Rs2Value,
            VirtualPolynomial::RdWriteValue => CoreVirtualPolynomial::RdWriteValue,
            VirtualPolynomial::Rs1Ra => CoreVirtualPolynomial::Rs1Ra,
            VirtualPolynomial::Rs2Ra => CoreVirtualPolynomial::Rs2Ra,
            VirtualPolynomial::RdWa => CoreVirtualPolynomial::RdWa,
            VirtualPolynomial::LookupOutput => CoreVirtualPolynomial::LookupOutput,
            VirtualPolynomial::InstructionRaf => CoreVirtualPolynomial::InstructionRaf,
            VirtualPolynomial::InstructionRafFlag => CoreVirtualPolynomial::InstructionRafFlag,
            VirtualPolynomial::InstructionRa(i) => CoreVirtualPolynomial::InstructionRa(i),
            VirtualPolynomial::RegistersVal => CoreVirtualPolynomial::RegistersVal,
            VirtualPolynomial::RamAddress => CoreVirtualPolynomial::RamAddress,
            VirtualPolynomial::RamRa => CoreVirtualPolynomial::RamRa,
            VirtualPolynomial::RamReadValue => CoreVirtualPolynomial::RamReadValue,
            VirtualPolynomial::RamWriteValue => CoreVirtualPolynomial::RamWriteValue,
            VirtualPolynomial::RamVal => CoreVirtualPolynomial::RamVal,
            VirtualPolynomial::RamValInit => CoreVirtualPolynomial::RamValInit,
            VirtualPolynomial::RamValFinal => CoreVirtualPolynomial::RamValFinal,
            VirtualPolynomial::RamHammingWeight => CoreVirtualPolynomial::RamHammingWeight,
            VirtualPolynomial::UnivariateSkip => CoreVirtualPolynomial::UnivariateSkip,
            VirtualPolynomial::OpFlags(flag) => CoreVirtualPolynomial::OpFlags(core_circuit(flag)),
            VirtualPolynomial::InstructionFlags(flag) => {
                CoreVirtualPolynomial::InstructionFlags(core_instruction(flag))
            }
            VirtualPolynomial::LookupTableFlag(flag) => {
                CoreVirtualPolynomial::LookupTableFlag(flag)
            }
        }
    }

    fn core_opening(id: OpeningId) -> CoreOpeningId {
        match id {
            OpeningId::Polynomial(PolynomialId::Committed(polynomial), sumcheck_id) => {
                CoreOpeningId::Polynomial(
                    CorePolynomialId::Committed(core_committed(polynomial)),
                    core_sumcheck(sumcheck_id),
                )
            }
            OpeningId::Polynomial(PolynomialId::Virtual(polynomial), sumcheck_id) => {
                CoreOpeningId::Polynomial(
                    CorePolynomialId::Virtual(core_virtual(polynomial)),
                    core_sumcheck(sumcheck_id),
                )
            }
            OpeningId::UntrustedAdvice(sumcheck_id) => {
                CoreOpeningId::UntrustedAdvice(core_sumcheck(sumcheck_id))
            }
            OpeningId::TrustedAdvice(sumcheck_id) => {
                CoreOpeningId::TrustedAdvice(core_sumcheck(sumcheck_id))
            }
        }
    }

    fn core_circuit(flag: CircuitFlags) -> CoreCircuitFlags {
        match flag {
            CircuitFlags::AddOperands => CoreCircuitFlags::AddOperands,
            CircuitFlags::SubtractOperands => CoreCircuitFlags::SubtractOperands,
            CircuitFlags::MultiplyOperands => CoreCircuitFlags::MultiplyOperands,
            CircuitFlags::Load => CoreCircuitFlags::Load,
            CircuitFlags::Store => CoreCircuitFlags::Store,
            CircuitFlags::Jump => CoreCircuitFlags::Jump,
            CircuitFlags::WriteLookupOutputToRD => CoreCircuitFlags::WriteLookupOutputToRD,
            CircuitFlags::VirtualInstruction => CoreCircuitFlags::VirtualInstruction,
            CircuitFlags::Assert => CoreCircuitFlags::Assert,
            CircuitFlags::DoNotUpdateUnexpandedPC => CoreCircuitFlags::DoNotUpdateUnexpandedPC,
            CircuitFlags::Advice => CoreCircuitFlags::Advice,
            CircuitFlags::IsCompressed => CoreCircuitFlags::IsCompressed,
            CircuitFlags::IsFirstInSequence => CoreCircuitFlags::IsFirstInSequence,
            CircuitFlags::IsLastInSequence => CoreCircuitFlags::IsLastInSequence,
        }
    }

    fn core_instruction(flag: InstructionFlags) -> CoreInstructionFlags {
        match flag {
            InstructionFlags::LeftOperandIsPC => CoreInstructionFlags::LeftOperandIsPC,
            InstructionFlags::RightOperandIsImm => CoreInstructionFlags::RightOperandIsImm,
            InstructionFlags::LeftOperandIsRs1Value => CoreInstructionFlags::LeftOperandIsRs1Value,
            InstructionFlags::RightOperandIsRs2Value => {
                CoreInstructionFlags::RightOperandIsRs2Value
            }
            InstructionFlags::Branch => CoreInstructionFlags::Branch,
            InstructionFlags::IsNoop => CoreInstructionFlags::IsNoop,
        }
    }
}
