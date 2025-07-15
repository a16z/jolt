use crate::dag::jolt_dag::JoltDagProof;
use crate::dag::state_manager::{self, Claims, ProofData, ProofKeys, Proofs, StateManager};
use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::opening_proof::{OpeningsKeys, BIG_ENDIAN};
use crate::subprotocols::sumcheck::SumcheckInstanceProof;
use crate::utils::transcript::Transcript;

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, SerializationError, Valid};
use std::cell::RefCell;
use std::collections::HashMap;
use std::io::{Read, Write};
use std::rc::Rc;
use std::sync::Arc;

impl<F: JoltField, PCS: CommitmentScheme<Field = F>, ProofTranscript: Transcript> std::ops::Deref
    for Proofs<F, PCS, ProofTranscript>
{
    type Target = HashMap<ProofKeys, ProofData<F, PCS, ProofTranscript>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>, ProofTranscript: Transcript> std::ops::DerefMut
    for Proofs<F, PCS, ProofTranscript>
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<F: JoltField> std::ops::Deref for Claims<F> {
    type Target = HashMap<OpeningsKeys, F>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<F: JoltField> std::ops::DerefMut for Claims<F> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>, ProofTranscript: Transcript> CanonicalSerialize
    for ProofData<F, PCS, ProofTranscript>
where
    SumcheckInstanceProof<F, ProofTranscript>: CanonicalSerialize,
    crate::poly::opening_proof::ReducedOpeningProof<F, PCS, ProofTranscript>: CanonicalSerialize,
{
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: ark_serialize::Compress,
    ) -> Result<(), SerializationError> {
        match self {
            ProofData::SpartanOuterData(proof) => {
                0u8.serialize_with_mode(&mut writer, compress)?;
                proof.serialize_with_mode(writer, compress)
            }
            ProofData::BatchableSumcheckData(proof) => {
                1u8.serialize_with_mode(&mut writer, compress)?;
                proof.serialize_with_mode(writer, compress)
            }
            ProofData::SumcheckSwitchIndex(index) => {
                2u8.serialize_with_mode(&mut writer, compress)?;
                index.serialize_with_mode(writer, compress)
            }
            ProofData::RamK(k) => {
                3u8.serialize_with_mode(&mut writer, compress)?;
                k.serialize_with_mode(writer, compress)
            }
            ProofData::ReducedOpeningProof(proof) => {
                4u8.serialize_with_mode(&mut writer, compress)?;
                proof.serialize_with_mode(writer, compress)
            }
        }
    }

    fn serialized_size(&self, compress: ark_serialize::Compress) -> usize {
        1 + match self {
            ProofData::SpartanOuterData(proof) => proof.serialized_size(compress),
            ProofData::BatchableSumcheckData(proof) => proof.serialized_size(compress),
            ProofData::SumcheckSwitchIndex(index) => index.serialized_size(compress),
            ProofData::RamK(k) => k.serialized_size(compress),
            ProofData::ReducedOpeningProof(proof) => proof.serialized_size(compress),
        }
    }
}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>, ProofTranscript: Transcript>
    CanonicalDeserialize for ProofData<F, PCS, ProofTranscript>
where
    SumcheckInstanceProof<F, ProofTranscript>: CanonicalDeserialize,
    crate::poly::opening_proof::ReducedOpeningProof<F, PCS, ProofTranscript>: CanonicalDeserialize,
{
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: ark_serialize::Compress,
        validate: ark_serialize::Validate,
    ) -> Result<Self, SerializationError> {
        let variant = u8::deserialize_with_mode(&mut reader, compress, validate)?;
        match variant {
            0 => Ok(ProofData::SpartanOuterData(
                SumcheckInstanceProof::deserialize_with_mode(reader, compress, validate)?,
            )),
            1 => Ok(ProofData::BatchableSumcheckData(
                SumcheckInstanceProof::deserialize_with_mode(reader, compress, validate)?,
            )),
            2 => Ok(ProofData::SumcheckSwitchIndex(
                usize::deserialize_with_mode(reader, compress, validate)?,
            )),
            3 => Ok(ProofData::RamK(usize::deserialize_with_mode(
                reader, compress, validate,
            )?)),
            4 => Ok(ProofData::ReducedOpeningProof(
                crate::poly::opening_proof::ReducedOpeningProof::deserialize_with_mode(
                    reader, compress, validate,
                )?,
            )),
            _ => Err(SerializationError::InvalidData),
        }
    }

    fn deserialize_compressed<R: Read>(reader: R) -> Result<Self, SerializationError> {
        Self::deserialize_with_mode(
            reader,
            ark_serialize::Compress::Yes,
            ark_serialize::Validate::Yes,
        )
    }

    fn deserialize_compressed_unchecked<R: Read>(reader: R) -> Result<Self, SerializationError> {
        Self::deserialize_with_mode(
            reader,
            ark_serialize::Compress::Yes,
            ark_serialize::Validate::No,
        )
    }

    fn deserialize_uncompressed<R: Read>(reader: R) -> Result<Self, SerializationError> {
        Self::deserialize_with_mode(
            reader,
            ark_serialize::Compress::No,
            ark_serialize::Validate::Yes,
        )
    }

    fn deserialize_uncompressed_unchecked<R: Read>(reader: R) -> Result<Self, SerializationError> {
        Self::deserialize_with_mode(
            reader,
            ark_serialize::Compress::No,
            ark_serialize::Validate::No,
        )
    }
}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>, ProofTranscript: Transcript> Valid
    for ProofData<F, PCS, ProofTranscript>
where
    SumcheckInstanceProof<F, ProofTranscript>: Valid,
    crate::poly::opening_proof::ReducedOpeningProof<F, PCS, ProofTranscript>: Valid,
{
    fn check(&self) -> Result<(), SerializationError> {
        match self {
            ProofData::SpartanOuterData(proof) => proof.check(),
            ProofData::BatchableSumcheckData(proof) => proof.check(),
            ProofData::SumcheckSwitchIndex(_) => Ok(()),
            ProofData::RamK(_) => Ok(()),
            ProofData::ReducedOpeningProof(proof) => proof.check(),
        }
    }
}

impl CanonicalSerialize for ProofKeys {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: ark_serialize::Compress,
    ) -> Result<(), SerializationError> {
        match self {
            ProofKeys::SpartanOuterSumcheck => 0u8.serialize_with_mode(&mut writer, compress),
            ProofKeys::Stage2Sumcheck => 1u8.serialize_with_mode(&mut writer, compress),
            ProofKeys::Stage3Sumcheck => 2u8.serialize_with_mode(&mut writer, compress),
            ProofKeys::Stage4Sumcheck => 3u8.serialize_with_mode(&mut writer, compress),
            ProofKeys::RamSumcheckSwitchIndex => 4u8.serialize_with_mode(&mut writer, compress),
            ProofKeys::RamK => 5u8.serialize_with_mode(&mut writer, compress),
            ProofKeys::ReducedOpeningProof => 6u8.serialize_with_mode(&mut writer, compress),
        }
    }

    fn serialized_size(&self, _compress: ark_serialize::Compress) -> usize {
        1
    }
}

impl CanonicalDeserialize for ProofKeys {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: ark_serialize::Compress,
        validate: ark_serialize::Validate,
    ) -> Result<Self, SerializationError> {
        let variant = u8::deserialize_with_mode(&mut reader, compress, validate)?;
        match variant {
            0 => Ok(ProofKeys::SpartanOuterSumcheck),
            1 => Ok(ProofKeys::Stage2Sumcheck),
            2 => Ok(ProofKeys::Stage3Sumcheck),
            3 => Ok(ProofKeys::Stage4Sumcheck),
            4 => Ok(ProofKeys::RamSumcheckSwitchIndex),
            5 => Ok(ProofKeys::RamK),
            6 => Ok(ProofKeys::ReducedOpeningProof),
            _ => Err(SerializationError::InvalidData),
        }
    }

    fn deserialize_compressed<R: Read>(reader: R) -> Result<Self, SerializationError> {
        Self::deserialize_with_mode(
            reader,
            ark_serialize::Compress::Yes,
            ark_serialize::Validate::Yes,
        )
    }

    fn deserialize_compressed_unchecked<R: Read>(reader: R) -> Result<Self, SerializationError> {
        Self::deserialize_with_mode(
            reader,
            ark_serialize::Compress::Yes,
            ark_serialize::Validate::No,
        )
    }

    fn deserialize_uncompressed<R: Read>(reader: R) -> Result<Self, SerializationError> {
        Self::deserialize_with_mode(
            reader,
            ark_serialize::Compress::No,
            ark_serialize::Validate::Yes,
        )
    }

    fn deserialize_uncompressed_unchecked<R: Read>(reader: R) -> Result<Self, SerializationError> {
        Self::deserialize_with_mode(
            reader,
            ark_serialize::Compress::No,
            ark_serialize::Validate::No,
        )
    }
}

impl Valid for ProofKeys {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl CanonicalSerialize for OpeningsKeys {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: ark_serialize::Compress,
    ) -> Result<(), SerializationError> {
        match self {
            OpeningsKeys::SpartanZ(idx) => {
                0u8.serialize_with_mode(&mut writer, compress)?;
                idx.serialize_with_mode(&mut writer, compress)
            }
            OpeningsKeys::InstructionTypeFlag(idx) => {
                1u8.serialize_with_mode(&mut writer, compress)?;
                idx.serialize_with_mode(&mut writer, compress)
            }
            OpeningsKeys::InstructionRa(idx) => {
                2u8.serialize_with_mode(&mut writer, compress)?;
                idx.serialize_with_mode(&mut writer, compress)
            }
            OpeningsKeys::InstructionBooleanityRa(idx) => {
                3u8.serialize_with_mode(&mut writer, compress)?;
                idx.serialize_with_mode(&mut writer, compress)
            }
            OpeningsKeys::InstructionHammingRa(idx) => {
                4u8.serialize_with_mode(&mut writer, compress)?;
                idx.serialize_with_mode(&mut writer, compress)
            }
            OpeningsKeys::LookupTableFlag(idx) => {
                5u8.serialize_with_mode(&mut writer, compress)?;
                idx.serialize_with_mode(&mut writer, compress)
            }
            OpeningsKeys::InstructionRafFlag => 6u8.serialize_with_mode(&mut writer, compress),
            OpeningsKeys::OuterSumcheckAz => 7u8.serialize_with_mode(&mut writer, compress),
            OpeningsKeys::OuterSumcheckBz => 8u8.serialize_with_mode(&mut writer, compress),
            OpeningsKeys::OuterSumcheckCz => 9u8.serialize_with_mode(&mut writer, compress),
            OpeningsKeys::PCSumcheckUnexpandedPC => 10u8.serialize_with_mode(&mut writer, compress),
            OpeningsKeys::PCSumcheckNextPC => 11u8.serialize_with_mode(&mut writer, compress),
            OpeningsKeys::RegistersReadWriteVal => 12u8.serialize_with_mode(&mut writer, compress),
            OpeningsKeys::RegistersReadWriteRs1Ra => {
                13u8.serialize_with_mode(&mut writer, compress)
            }
            OpeningsKeys::RegistersReadWriteRs2Ra => {
                14u8.serialize_with_mode(&mut writer, compress)
            }
            OpeningsKeys::RegistersReadWriteRdWa => 15u8.serialize_with_mode(&mut writer, compress),
            OpeningsKeys::RegistersReadWriteInc => 16u8.serialize_with_mode(&mut writer, compress),
            OpeningsKeys::RegistersValEvaluationInc => {
                17u8.serialize_with_mode(&mut writer, compress)
            }
            OpeningsKeys::RegistersValEvaluationWa => {
                18u8.serialize_with_mode(&mut writer, compress)
            }
            OpeningsKeys::RamReadWriteCheckingVal => {
                19u8.serialize_with_mode(&mut writer, compress)
            }
            OpeningsKeys::RamReadWriteCheckingRa => 20u8.serialize_with_mode(&mut writer, compress),
            OpeningsKeys::RamReadWriteCheckingInc => {
                21u8.serialize_with_mode(&mut writer, compress)
            }
            OpeningsKeys::RamRafEvaluationRa => 22u8.serialize_with_mode(&mut writer, compress),
            OpeningsKeys::RamValInit => 23u8.serialize_with_mode(&mut writer, compress),
            OpeningsKeys::RamValFinal => 24u8.serialize_with_mode(&mut writer, compress),
            OpeningsKeys::RamValEvaluationInc => 25u8.serialize_with_mode(&mut writer, compress),
            OpeningsKeys::RamValEvaluationWa => 26u8.serialize_with_mode(&mut writer, compress),
            OpeningsKeys::ValFinalInc => 27u8.serialize_with_mode(&mut writer, compress),
            OpeningsKeys::ValFinalWa => 28u8.serialize_with_mode(&mut writer, compress),
            OpeningsKeys::RamHammingRa(idx) => {
                29u8.serialize_with_mode(&mut writer, compress)?;
                idx.serialize_with_mode(&mut writer, compress)
            }
            OpeningsKeys::RamBooleanityRa(idx) => {
                30u8.serialize_with_mode(&mut writer, compress)?;
                idx.serialize_with_mode(&mut writer, compress)
            }
            OpeningsKeys::RamRaVirtualization(idx) => {
                31u8.serialize_with_mode(&mut writer, compress)?;
                idx.serialize_with_mode(&mut writer, compress)
            }
            OpeningsKeys::OpeningReduction(idx) => {
                32u8.serialize_with_mode(&mut writer, compress)?;
                idx.serialize_with_mode(&mut writer, compress)
            }
        }
    }

    fn serialized_size(&self, compress: ark_serialize::Compress) -> usize {
        let mut size = 1; // size of variant tag
        match self {
            OpeningsKeys::SpartanZ(idx) => size += idx.serialized_size(compress),
            OpeningsKeys::InstructionTypeFlag(idx)
            | OpeningsKeys::InstructionRa(idx)
            | OpeningsKeys::InstructionBooleanityRa(idx)
            | OpeningsKeys::InstructionHammingRa(idx)
            | OpeningsKeys::LookupTableFlag(idx)
            | OpeningsKeys::RamHammingRa(idx)
            | OpeningsKeys::RamBooleanityRa(idx)
            | OpeningsKeys::RamRaVirtualization(idx)
            | OpeningsKeys::OpeningReduction(idx) => size += idx.serialized_size(compress),
            _ => {} // No additional data for other variants
        }
        size
    }
}

impl CanonicalDeserialize for OpeningsKeys {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: ark_serialize::Compress,
        validate: ark_serialize::Validate,
    ) -> Result<Self, SerializationError> {
        use crate::r1cs::inputs::JoltR1CSInputs;

        let variant = u8::deserialize_with_mode(&mut reader, compress, validate)?;
        match variant {
            0 => Ok(OpeningsKeys::SpartanZ(
                JoltR1CSInputs::deserialize_with_mode(reader, compress, validate)?,
            )),
            1 => Ok(OpeningsKeys::InstructionTypeFlag(
                usize::deserialize_with_mode(reader, compress, validate)?,
            )),
            2 => Ok(OpeningsKeys::InstructionRa(usize::deserialize_with_mode(
                reader, compress, validate,
            )?)),
            3 => Ok(OpeningsKeys::InstructionBooleanityRa(
                usize::deserialize_with_mode(reader, compress, validate)?,
            )),
            4 => Ok(OpeningsKeys::InstructionHammingRa(
                usize::deserialize_with_mode(reader, compress, validate)?,
            )),
            5 => Ok(OpeningsKeys::LookupTableFlag(usize::deserialize_with_mode(
                reader, compress, validate,
            )?)),
            6 => Ok(OpeningsKeys::InstructionRafFlag),
            7 => Ok(OpeningsKeys::OuterSumcheckAz),
            8 => Ok(OpeningsKeys::OuterSumcheckBz),
            9 => Ok(OpeningsKeys::OuterSumcheckCz),
            10 => Ok(OpeningsKeys::PCSumcheckUnexpandedPC),
            11 => Ok(OpeningsKeys::PCSumcheckNextPC),
            12 => Ok(OpeningsKeys::RegistersReadWriteVal),
            13 => Ok(OpeningsKeys::RegistersReadWriteRs1Ra),
            14 => Ok(OpeningsKeys::RegistersReadWriteRs2Ra),
            15 => Ok(OpeningsKeys::RegistersReadWriteRdWa),
            16 => Ok(OpeningsKeys::RegistersReadWriteInc),
            17 => Ok(OpeningsKeys::RegistersValEvaluationInc),
            18 => Ok(OpeningsKeys::RegistersValEvaluationWa),
            19 => Ok(OpeningsKeys::RamReadWriteCheckingVal),
            20 => Ok(OpeningsKeys::RamReadWriteCheckingRa),
            21 => Ok(OpeningsKeys::RamReadWriteCheckingInc),
            22 => Ok(OpeningsKeys::RamRafEvaluationRa),
            23 => Ok(OpeningsKeys::RamValInit),
            24 => Ok(OpeningsKeys::RamValFinal),
            25 => Ok(OpeningsKeys::RamValEvaluationInc),
            26 => Ok(OpeningsKeys::RamValEvaluationWa),
            27 => Ok(OpeningsKeys::ValFinalInc),
            28 => Ok(OpeningsKeys::ValFinalWa),
            29 => Ok(OpeningsKeys::RamHammingRa(usize::deserialize_with_mode(
                reader, compress, validate,
            )?)),
            30 => Ok(OpeningsKeys::RamBooleanityRa(usize::deserialize_with_mode(
                reader, compress, validate,
            )?)),
            31 => Ok(OpeningsKeys::RamRaVirtualization(
                usize::deserialize_with_mode(reader, compress, validate)?,
            )),
            32 => Ok(OpeningsKeys::OpeningReduction(
                usize::deserialize_with_mode(reader, compress, validate)?,
            )),
            _ => Err(SerializationError::InvalidData),
        }
    }
}

impl Valid for OpeningsKeys {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl CanonicalSerialize for crate::r1cs::inputs::JoltR1CSInputs {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: ark_serialize::Compress,
    ) -> Result<(), SerializationError> {
        use crate::r1cs::inputs::JoltR1CSInputs;

        match self {
            JoltR1CSInputs::PC => 0u8.serialize_with_mode(&mut writer, compress),
            JoltR1CSInputs::UnexpandedPC => 1u8.serialize_with_mode(&mut writer, compress),
            JoltR1CSInputs::Rd => 2u8.serialize_with_mode(&mut writer, compress),
            JoltR1CSInputs::Imm => 3u8.serialize_with_mode(&mut writer, compress),
            JoltR1CSInputs::RamAddress => 4u8.serialize_with_mode(&mut writer, compress),
            JoltR1CSInputs::Rs1Value => 5u8.serialize_with_mode(&mut writer, compress),
            JoltR1CSInputs::Rs2Value => 6u8.serialize_with_mode(&mut writer, compress),
            JoltR1CSInputs::RdWriteValue => 7u8.serialize_with_mode(&mut writer, compress),
            JoltR1CSInputs::RamReadValue => 8u8.serialize_with_mode(&mut writer, compress),
            JoltR1CSInputs::RamWriteValue => 9u8.serialize_with_mode(&mut writer, compress),
            JoltR1CSInputs::LeftInstructionInput => 10u8.serialize_with_mode(&mut writer, compress),
            JoltR1CSInputs::RightInstructionInput => {
                11u8.serialize_with_mode(&mut writer, compress)
            }
            JoltR1CSInputs::LeftLookupOperand => 12u8.serialize_with_mode(&mut writer, compress),
            JoltR1CSInputs::RightLookupOperand => 13u8.serialize_with_mode(&mut writer, compress),
            JoltR1CSInputs::Product => 14u8.serialize_with_mode(&mut writer, compress),
            JoltR1CSInputs::WriteLookupOutputToRD => {
                15u8.serialize_with_mode(&mut writer, compress)
            }
            JoltR1CSInputs::WritePCtoRD => 16u8.serialize_with_mode(&mut writer, compress),
            JoltR1CSInputs::ShouldBranch => 17u8.serialize_with_mode(&mut writer, compress),
            JoltR1CSInputs::NextUnexpandedPC => 18u8.serialize_with_mode(&mut writer, compress),
            JoltR1CSInputs::NextPC => 19u8.serialize_with_mode(&mut writer, compress),
            JoltR1CSInputs::LookupOutput => 20u8.serialize_with_mode(&mut writer, compress),
            JoltR1CSInputs::OpFlags(flags) => {
                21u8.serialize_with_mode(&mut writer, compress)?;
                // Serialize CircuitFlags - we'll just serialize it as u32 for now
                (*flags as u32).serialize_with_mode(&mut writer, compress)
            }
        }
    }

    fn serialized_size(&self, _compress: ark_serialize::Compress) -> usize {
        use crate::r1cs::inputs::JoltR1CSInputs;
        match self {
            JoltR1CSInputs::OpFlags(_) => 1 + 4, // discriminant + u32
            _ => 1,                              // Only the discriminant
        }
    }
}

impl CanonicalDeserialize for crate::r1cs::inputs::JoltR1CSInputs {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: ark_serialize::Compress,
        validate: ark_serialize::Validate,
    ) -> Result<Self, SerializationError> {
        use crate::r1cs::inputs::JoltR1CSInputs;

        let variant = u8::deserialize_with_mode(&mut reader, compress, validate)?;
        match variant {
            0 => Ok(JoltR1CSInputs::PC),
            1 => Ok(JoltR1CSInputs::UnexpandedPC),
            2 => Ok(JoltR1CSInputs::Rd),
            3 => Ok(JoltR1CSInputs::Imm),
            4 => Ok(JoltR1CSInputs::RamAddress),
            5 => Ok(JoltR1CSInputs::Rs1Value),
            6 => Ok(JoltR1CSInputs::Rs2Value),
            7 => Ok(JoltR1CSInputs::RdWriteValue),
            8 => Ok(JoltR1CSInputs::RamReadValue),
            9 => Ok(JoltR1CSInputs::RamWriteValue),
            10 => Ok(JoltR1CSInputs::LeftInstructionInput),
            11 => Ok(JoltR1CSInputs::RightInstructionInput),
            12 => Ok(JoltR1CSInputs::LeftLookupOperand),
            13 => Ok(JoltR1CSInputs::RightLookupOperand),
            14 => Ok(JoltR1CSInputs::Product),
            15 => Ok(JoltR1CSInputs::WriteLookupOutputToRD),
            16 => Ok(JoltR1CSInputs::WritePCtoRD),
            17 => Ok(JoltR1CSInputs::ShouldBranch),
            18 => Ok(JoltR1CSInputs::NextUnexpandedPC),
            19 => Ok(JoltR1CSInputs::NextPC),
            20 => Ok(JoltR1CSInputs::LookupOutput),
            21 => {
                let _flags_u32 = u32::deserialize_with_mode(&mut reader, compress, validate)?;
                Err(SerializationError::InvalidData)
            }
            _ => Err(SerializationError::InvalidData),
        }
    }
}

impl Valid for crate::r1cs::inputs::JoltR1CSInputs {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>, ProofTranscript: Transcript> Valid
    for Proofs<F, PCS, ProofTranscript>
where
    ProofData<F, PCS, ProofTranscript>: Valid,
{
    fn check(&self) -> Result<(), SerializationError> {
        for key in self.0.keys() {
            key.check()?;
        }

        for value in self.0.values() {
            value.check()?;
        }

        Ok(())
    }
}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>, ProofTranscript: Transcript> CanonicalSerialize
    for Proofs<F, PCS, ProofTranscript>
where
    ProofData<F, PCS, ProofTranscript>: CanonicalSerialize,
{
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: ark_serialize::Compress,
    ) -> Result<(), SerializationError> {
        let len = self.0.len() as u64;
        len.serialize_with_mode(&mut writer, compress)?;

        for (key, value) in self.0.iter() {
            key.serialize_with_mode(&mut writer, compress)?;
            value.serialize_with_mode(&mut writer, compress)?;
        }

        Ok(())
    }

    fn serialized_size(&self, compress: ark_serialize::Compress) -> usize {
        let mut size = 8; // size of length
        for (key, value) in self.0.iter() {
            size += key.serialized_size(compress);
            size += value.serialized_size(compress);
        }
        size
    }
}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>, ProofTranscript: Transcript>
    CanonicalDeserialize for Proofs<F, PCS, ProofTranscript>
where
    ProofData<F, PCS, ProofTranscript>: CanonicalDeserialize,
{
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: ark_serialize::Compress,
        validate: ark_serialize::Validate,
    ) -> Result<Self, SerializationError> {
        let len = u64::deserialize_with_mode(&mut reader, compress, validate)?;

        let mut map = HashMap::new();
        for _ in 0..len {
            let key = ProofKeys::deserialize_with_mode(&mut reader, compress, validate)?;
            let value = ProofData::deserialize_with_mode(&mut reader, compress, validate)?;
            map.insert(key, value);
        }

        Ok(Self(map))
    }

    fn deserialize_compressed<R: Read>(reader: R) -> Result<Self, SerializationError> {
        Self::deserialize_with_mode(
            reader,
            ark_serialize::Compress::Yes,
            ark_serialize::Validate::Yes,
        )
    }

    fn deserialize_compressed_unchecked<R: Read>(reader: R) -> Result<Self, SerializationError> {
        Self::deserialize_with_mode(
            reader,
            ark_serialize::Compress::Yes,
            ark_serialize::Validate::No,
        )
    }

    fn deserialize_uncompressed<R: Read>(reader: R) -> Result<Self, SerializationError> {
        Self::deserialize_with_mode(
            reader,
            ark_serialize::Compress::No,
            ark_serialize::Validate::Yes,
        )
    }

    fn deserialize_uncompressed_unchecked<R: Read>(reader: R) -> Result<Self, SerializationError> {
        Self::deserialize_with_mode(
            reader,
            ark_serialize::Compress::No,
            ark_serialize::Validate::No,
        )
    }
}

impl<F: JoltField> Valid for Claims<F> {
    fn check(&self) -> Result<(), SerializationError> {
        // Check all keys
        for key in self.0.keys() {
            key.check()?;
        }
        Ok(())
    }
}

impl<F: JoltField> CanonicalSerialize for Claims<F> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: ark_serialize::Compress,
    ) -> Result<(), SerializationError> {
        let len = self.0.len() as u64;
        len.serialize_with_mode(&mut writer, compress)?;

        for (key, value) in self.0.iter() {
            key.serialize_with_mode(&mut writer, compress)?;
            value.serialize_with_mode(&mut writer, compress)?;
        }

        Ok(())
    }

    fn serialized_size(&self, compress: ark_serialize::Compress) -> usize {
        let mut size = 8; // size of length
        for (key, value) in self.0.iter() {
            size += key.serialized_size(compress);
            size += value.serialized_size(compress);
        }
        size
    }
}

impl<F: JoltField> CanonicalDeserialize for Claims<F> {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: ark_serialize::Compress,
        validate: ark_serialize::Validate,
    ) -> Result<Self, SerializationError> {
        let len = u64::deserialize_with_mode(&mut reader, compress, validate)?;

        let mut map = HashMap::new();
        for _ in 0..len {
            let key = OpeningsKeys::deserialize_with_mode(&mut reader, compress, validate)?;
            let value = F::deserialize_with_mode(&mut reader, compress, validate)?;
            map.insert(key, value);
        }

        Ok(Self(map))
    }

    fn deserialize_compressed<R: Read>(reader: R) -> Result<Self, SerializationError> {
        Self::deserialize_with_mode(
            reader,
            ark_serialize::Compress::Yes,
            ark_serialize::Validate::Yes,
        )
    }

    fn deserialize_compressed_unchecked<R: Read>(reader: R) -> Result<Self, SerializationError> {
        Self::deserialize_with_mode(
            reader,
            ark_serialize::Compress::Yes,
            ark_serialize::Validate::No,
        )
    }

    fn deserialize_uncompressed<R: Read>(reader: R) -> Result<Self, SerializationError> {
        Self::deserialize_with_mode(
            reader,
            ark_serialize::Compress::No,
            ark_serialize::Validate::Yes,
        )
    }

    fn deserialize_uncompressed_unchecked<R: Read>(reader: R) -> Result<Self, SerializationError> {
        Self::deserialize_with_mode(
            reader,
            ark_serialize::Compress::No,
            ark_serialize::Validate::No,
        )
    }
}

// Conversion from StateManager (with prover state) to JoltDagProof
impl<'a, const WORD_SIZE: usize, F, PCS, ProofTranscript>
    From<&StateManager<'a, F, ProofTranscript, PCS>>
    for JoltDagProof<WORD_SIZE, F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    ProofTranscript: Transcript,
{
    fn from(state_manager: &StateManager<'a, F, ProofTranscript, PCS>) -> Self {
        use crate::jolt::vm::JoltVerifierPreprocessing;

        let (preprocessing, trace, program_io, _) = state_manager.get_prover_data();
        let trace_length = trace.len();

        let verifier_preprocessing = Arc::new(JoltVerifierPreprocessing::from(preprocessing));

        let commitments = state_manager.get_commitments();

        let dag_proofs = state_manager.proofs.borrow().clone();

        // Get claims from prover accumulator
        let prover_accumulator = state_manager.get_prover_accumulator();
        let prover_acc_borrow = prover_accumulator.borrow();
        let mut claims_map = HashMap::new();

        for (key, (_, value)) in prover_acc_borrow.evaluation_openings().iter() {
            claims_map.insert(*key, *value);
        }

        let claims = Claims(claims_map);

        // TODO(markosg04) get the real values for these and update ram and registers to use these
        let sumcheck_switch_index_registers = 0;
        let sumcheck_switch_index_ram = 0;

        JoltDagProof {
            verifier_preprocessing,
            program_io: program_io.clone(),
            trace_length,
            sumcheck_switch_index_registers,
            sumcheck_switch_index_ram,
            commitments,
            dag_proofs,
            claims,
        }
    }
}

// Conversion from JoltDagProof to StateManager (with verifier state)
impl<'a, const WORD_SIZE: usize, F, PCS, ProofTranscript>
    Into<StateManager<'a, F, ProofTranscript, PCS>>
    for JoltDagProof<WORD_SIZE, F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    ProofTranscript: Transcript,
{
    fn into(self) -> StateManager<'a, F, ProofTranscript, PCS> {
        use crate::poly::opening_proof::{OpeningPoint, VerifierOpeningAccumulator};

        let proof = self;

        let verifier_accumulator = VerifierOpeningAccumulator::<F, PCS>::new();
        let verifier_accumulator = Rc::new(RefCell::new(verifier_accumulator));

        let transcript = ProofTranscript::new(b"Jolt");
        let transcript = Rc::new(RefCell::new(transcript));

        let proofs = Rc::new(RefCell::new(proof.dag_proofs));
        let commitments = Rc::new(RefCell::new(Some(proof.commitments)));

        let program_data = state_manager::VerifierProgramData {
            preprocessing: proof.verifier_preprocessing.clone(),
            program_io: proof.program_io,
            trace_length: proof.trace_length,
        };

        let state_manager = StateManager::new_verifier(
            program_data,
            verifier_accumulator.clone(),
            transcript,
            proofs,
            commitments,
        );

        // Populate claims in the verifier accumulator
        let mut verifier_acc_borrow = verifier_accumulator.borrow_mut();
        for (key, claim) in proof.claims.iter() {
            let empty_point = OpeningPoint::<{ BIG_ENDIAN }, F>::new(vec![]);
            verifier_acc_borrow
                .evaluation_openings_mut()
                .insert(*key, (empty_point, *claim));
        }
        drop(verifier_acc_borrow);

        state_manager
    }
}
