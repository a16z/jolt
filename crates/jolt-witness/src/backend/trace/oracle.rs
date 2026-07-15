//! The exhaustive id-indexed oracle surface of the trace backend.
//!
//! Both matches below are total over [`JoltPolynomialId`] with **no wildcard
//! arm**: adding a variant in jolt-claims fails compilation here until the
//! variant is mapped to a derivation or added to an explicit exclusion arm
//! with its reason. The exclusion reasons are asserted in tests.

use jolt_claims::protocols::jolt::{
    JoltCommittedPolynomial, JoltPolynomialId, JoltVirtualPolynomial,
};

use super::*;
use crate::witnesses::{
    Imm, InstructionFlag, InstructionRafFlag, LeftInstructionInput, LeftLookupOperand,
    LookupOutput, LookupTableFlag, NextIsFirstInSequence, NextIsNoop, NextIsVirtual, NextPc,
    NextUnexpandedPc, OpFlag, Pc, Product, RamAddress, RamHammingWeight, RamReadValue,
    RamWriteValue, RdWriteValue, RightInstructionInput, RightLookupOperand, Rs1Value, Rs2Value,
    ShouldBranch, ShouldJump, UnexpandedPc,
};
use crate::{
    JoltWitnessOracle, PolynomialBatchStream, PolynomialEncoding, PolynomialStream, Shape,
};

/// Base-mode committed-program polynomials: precommitted from preprocessing,
/// never derived from the execution trace.
pub(crate) const COMMITTED_PROGRAM_REASON: &str =
    "committed-program polynomial served from preprocessing, not the execution trace";
/// Lattice-mode slots of the packed witness; base mode never constructs them.
pub(crate) const LATTICE_REASON: &str =
    "lattice-mode packed-witness polynomial; base mode never constructs it";
/// Openings produced by kernels during proving (owned by the proof session).
pub(crate) const PROTOCOL_INTERMEDIATE_REASON: &str =
    "protocol intermediate produced during proving, never served by a witness backend";
/// Vocabulary with no consumer on the modular stack; no derivation exists.
pub(crate) const UNSERVED_REASON: &str =
    "no consumer on the modular stack; no trace derivation is defined";

fn not_served(id: JoltPolynomialId, reason: &'static str) -> WitnessError {
    WitnessError::NotServed {
        oracle: format!("{id:?}"),
        reason,
    }
}

impl<T: TraceSource + Clone> TraceBackend<'_, T> {
    pub(crate) fn shape_of(&self, id: JoltPolynomialId) -> Result<Shape, WitnessError> {
        use JoltCommittedPolynomial as C;
        use JoltVirtualPolynomial as V;
        use PolynomialEncoding::{Compact, Dense, OneHot};
        match id {
            JoltPolynomialId::Committed(committed) => match committed {
                C::RdInc | C::RamInc => Ok(Shape::new(self.trace_log_rows(), Compact)),
                C::InstructionRa(index) => {
                    require_index(index, self.ra_layout()?.instruction())?;
                    Ok(Shape::new(self.one_hot_log_rows()?, OneHot))
                }
                C::BytecodeRa(index) => {
                    require_index(index, self.ra_layout()?.bytecode())?;
                    Ok(Shape::new(self.one_hot_log_rows()?, OneHot))
                }
                C::RamRa(index) => {
                    require_index(index, self.ra_layout()?.ram())?;
                    Ok(Shape::new(self.one_hot_log_rows()?, OneHot))
                }
                C::TrustedAdvice => {
                    if !self.config.include_trusted_advice {
                        return Err(WitnessError::UnknownOracle {
                            label: JOLT_VM_LABEL,
                        });
                    }
                    Ok(Shape::new(
                        Self::advice_log_rows(
                            self.preprocessing.memory_layout.max_trusted_advice_size as usize / 8,
                        ),
                        Compact,
                    ))
                }
                C::UntrustedAdvice => {
                    if !self.config.include_untrusted_advice {
                        return Err(WitnessError::UnknownOracle {
                            label: JOLT_VM_LABEL,
                        });
                    }
                    Ok(Shape::new(
                        Self::advice_log_rows(
                            self.preprocessing.memory_layout.max_untrusted_advice_size as usize / 8,
                        ),
                        Compact,
                    ))
                }
                C::BytecodeChunk(_) | C::ProgramImageInit => {
                    Err(not_served(id, COMMITTED_PROGRAM_REASON))
                }
                C::UnsignedIncChunk(_)
                | C::UnsignedIncMsb
                | C::TrustedAdviceBytes
                | C::UntrustedAdviceBytes
                | C::BytecodeRegisterSelector { .. }
                | C::BytecodeCircuitFlag { .. }
                | C::BytecodeInstructionFlag { .. }
                | C::BytecodeLookupSelector { .. }
                | C::BytecodeRafFlag { .. }
                | C::BytecodeUnexpandedPcBytes { .. }
                | C::BytecodeImmBytes { .. }
                | C::ProgramImageBytes => Err(not_served(id, LATTICE_REASON)),
            },
            JoltPolynomialId::Virtual(virtual_id) => match virtual_id {
                V::RamVal | V::RamRa => Ok(Shape::new(self.ram_read_write_log_rows()?, Dense)),
                V::RegistersVal | V::Rs1Ra | V::Rs2Ra | V::RdWa => {
                    Ok(Shape::new(self.register_read_write_log_rows()?, Dense))
                }
                V::RamValFinal => Ok(Shape::new(self.ram_log_k()?, Dense)),
                V::InstructionRa(index) => {
                    require_index(index, self.instruction_virtual_ra_count()?)?;
                    Ok(Shape::new(self.instruction_virtual_ra_log_rows()?, Dense))
                }
                V::LookupTableFlag(index) => {
                    require_index(index, LookupTableKind::<RV64_XLEN>::COUNT)?;
                    Ok(Shape::new(self.trace_log_rows(), Dense))
                }
                V::PC
                | V::UnexpandedPC
                | V::NextPC
                | V::NextUnexpandedPC
                | V::NextIsNoop
                | V::NextIsVirtual
                | V::NextIsFirstInSequence
                | V::LeftLookupOperand
                | V::RightLookupOperand
                | V::LeftInstructionInput
                | V::RightInstructionInput
                | V::Product
                | V::ShouldJump
                | V::ShouldBranch
                | V::Imm
                | V::Rs1Value
                | V::Rs2Value
                | V::RdWriteValue
                | V::LookupOutput
                | V::InstructionRafFlag
                | V::RamAddress
                | V::RamReadValue
                | V::RamWriteValue
                | V::RamHammingWeight
                | V::OpFlags(_)
                | V::InstructionFlags(_) => Ok(Shape::new(self.trace_log_rows(), Dense)),
                V::Rd | V::InstructionRaf | V::RamValInit => Err(not_served(id, UNSERVED_REASON)),
                V::UnivariateSkip
                | V::BytecodeValStage(_)
                | V::BytecodeReadRafAddrClaim
                | V::BooleanityAddrClaim
                | V::BytecodeClaimReductionIntermediate
                | V::ProgramImageInitContributionRw => {
                    Err(not_served(id, PROTOCOL_INTERMEDIATE_REASON))
                }
                V::FusedInc => Err(not_served(id, LATTICE_REASON)),
            },
        }
    }
}

impl<F: Field, T: TraceSource + Clone> JoltWitnessOracle<F> for TraceBackend<'_, T> {
    fn shape(&self, id: JoltPolynomialId) -> Result<Shape, WitnessError> {
        self.shape_of(id)
    }

    fn oracle_table(&self, id: JoltPolynomialId) -> Result<Vec<F>, WitnessError> {
        use JoltCommittedPolynomial as C;
        use JoltVirtualPolynomial as V;
        // Validates presence and index ranges (and rejects excluded ids)
        // before materialization, exactly like the arms below.
        let _ = self.shape_of(id)?;
        match id {
            JoltPolynomialId::Committed(committed) => match committed {
                C::RdInc
                | C::RamInc
                | C::InstructionRa(_)
                | C::BytecodeRa(_)
                | C::RamRa(_)
                | C::TrustedAdvice
                | C::UntrustedAdvice => self.materialize_compact_committed(committed),
                C::BytecodeChunk(_) | C::ProgramImageInit => {
                    Err(not_served(id, COMMITTED_PROGRAM_REASON))
                }
                C::UnsignedIncChunk(_)
                | C::UnsignedIncMsb
                | C::TrustedAdviceBytes
                | C::UntrustedAdviceBytes
                | C::BytecodeRegisterSelector { .. }
                | C::BytecodeCircuitFlag { .. }
                | C::BytecodeInstructionFlag { .. }
                | C::BytecodeLookupSelector { .. }
                | C::BytecodeRafFlag { .. }
                | C::BytecodeUnexpandedPcBytes { .. }
                | C::BytecodeImmBytes { .. }
                | C::ProgramImageBytes => Err(not_served(id, LATTICE_REASON)),
            },
            JoltPolynomialId::Virtual(virtual_id) => match virtual_id {
                V::RamVal | V::RamRa => self.materialize_ram_read_write_virtual(virtual_id),
                V::RegistersVal | V::Rs1Ra | V::Rs2Ra | V::RdWa => {
                    self.materialize_register_read_write_virtual(virtual_id)
                }
                V::RamValFinal => self.materialize_ram_val_final(),
                V::InstructionRa(index) => self.materialize_instruction_ra(index),
                V::PC => self.materialize_cycle::<F, Pc>(),
                V::UnexpandedPC => self.materialize_cycle::<F, UnexpandedPc>(),
                V::NextPC => self.materialize_cycle::<F, NextPc>(),
                V::NextUnexpandedPC => self.materialize_cycle::<F, NextUnexpandedPc>(),
                V::NextIsNoop => self.materialize_cycle::<F, NextIsNoop>(),
                V::NextIsVirtual => self.materialize_cycle::<F, NextIsVirtual>(),
                V::NextIsFirstInSequence => self.materialize_cycle::<F, NextIsFirstInSequence>(),
                V::LeftLookupOperand => self.materialize_cycle::<F, LeftLookupOperand>(),
                V::RightLookupOperand => self.materialize_cycle::<F, RightLookupOperand>(),
                V::LeftInstructionInput => self.materialize_cycle::<F, LeftInstructionInput>(),
                V::RightInstructionInput => self.materialize_cycle::<F, RightInstructionInput>(),
                V::Product => self.materialize_cycle::<F, Product>(),
                V::ShouldJump => self.materialize_cycle::<F, ShouldJump>(),
                V::ShouldBranch => self.materialize_cycle::<F, ShouldBranch>(),
                V::Imm => self.materialize_cycle::<F, Imm>(),
                V::Rs1Value => self.materialize_cycle::<F, Rs1Value>(),
                V::Rs2Value => self.materialize_cycle::<F, Rs2Value>(),
                V::RdWriteValue => self.materialize_cycle::<F, RdWriteValue>(),
                V::LookupOutput => self.materialize_cycle::<F, LookupOutput>(),
                V::InstructionRafFlag => self.materialize_cycle::<F, InstructionRafFlag>(),
                V::RamAddress => self.materialize_cycle::<F, RamAddress>(),
                V::RamReadValue => self.materialize_cycle::<F, RamReadValue>(),
                V::RamWriteValue => self.materialize_cycle::<F, RamWriteValue>(),
                V::RamHammingWeight => self.materialize_cycle::<F, RamHammingWeight>(),
                V::OpFlags(flag) => self.materialize_cycle_indexed::<F, OpFlag, _>(flag),
                V::InstructionFlags(flag) => {
                    self.materialize_cycle_indexed::<F, InstructionFlag, _>(flag)
                }
                V::LookupTableFlag(table) => {
                    self.materialize_cycle_indexed::<F, LookupTableFlag, _>(table)
                }
                V::Rd | V::InstructionRaf | V::RamValInit => Err(not_served(id, UNSERVED_REASON)),
                V::UnivariateSkip
                | V::BytecodeValStage(_)
                | V::BytecodeReadRafAddrClaim
                | V::BooleanityAddrClaim
                | V::BytecodeClaimReductionIntermediate
                | V::ProgramImageInitContributionRw => {
                    Err(not_served(id, PROTOCOL_INTERMEDIATE_REASON))
                }
                V::FusedInc => Err(not_served(id, LATTICE_REASON)),
            },
        }
    }

    fn committed_order(&self) -> Result<Vec<JoltCommittedPolynomial>, WitnessError> {
        self.committed_polynomial_order()
    }

    fn committed_stream<'a>(
        &'a self,
        id: JoltCommittedPolynomial,
        chunk_size: usize,
    ) -> Result<Box<dyn PolynomialStream<F> + 'a>, WitnessError>
    where
        F: 'a,
    {
        Ok(Box::new(TraceBackend::committed_stream(
            self, id, chunk_size,
        )?))
    }

    fn committed_batch_stream<'a>(
        &'a self,
        ids: &'a [JoltCommittedPolynomial],
        chunk_size: usize,
    ) -> Result<Box<dyn PolynomialBatchStream<F, JoltCommittedPolynomial> + 'a>, WitnessError>
    where
        F: 'a,
    {
        Ok(Box::new(TraceBackend::committed_batch_stream(
            self, ids, chunk_size,
        )?))
    }
}
