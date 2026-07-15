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
        let (dimensions, encoding) = match id {
            JoltPolynomialId::Committed(committed) => match committed {
                C::RdInc | C::RamInc => (self.trace_dimensions()?, PolynomialEncoding::Compact),
                C::InstructionRa(index) => {
                    require_index(index, self.ra_layout()?.instruction())?;
                    (self.one_hot_dimensions()?, PolynomialEncoding::OneHot)
                }
                C::BytecodeRa(index) => {
                    require_index(index, self.ra_layout()?.bytecode())?;
                    (self.one_hot_dimensions()?, PolynomialEncoding::OneHot)
                }
                C::RamRa(index) => {
                    require_index(index, self.ra_layout()?.ram())?;
                    (self.one_hot_dimensions()?, PolynomialEncoding::OneHot)
                }
                C::TrustedAdvice => {
                    if !self.config.include_trusted_advice {
                        return Err(WitnessError::UnknownOracle {
                            label: JOLT_VM_LABEL,
                        });
                    }
                    (
                        Self::advice_dimensions(
                            self.preprocessing.memory_layout.max_trusted_advice_size as usize / 8,
                        ),
                        PolynomialEncoding::Compact,
                    )
                }
                C::UntrustedAdvice => {
                    if !self.config.include_untrusted_advice {
                        return Err(WitnessError::UnknownOracle {
                            label: JOLT_VM_LABEL,
                        });
                    }
                    (
                        Self::advice_dimensions(
                            self.preprocessing.memory_layout.max_untrusted_advice_size as usize / 8,
                        ),
                        PolynomialEncoding::Compact,
                    )
                }
                C::BytecodeChunk(_) | C::ProgramImageInit => {
                    return Err(not_served(id, COMMITTED_PROGRAM_REASON));
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
                | C::ProgramImageBytes => {
                    return Err(not_served(id, LATTICE_REASON));
                }
            },
            JoltPolynomialId::Virtual(virtual_id) => match virtual_id {
                V::RamVal | V::RamRa => {
                    (self.ram_read_write_dimensions()?, PolynomialEncoding::Dense)
                }
                V::RegistersVal | V::Rs1Ra | V::Rs2Ra | V::RdWa => (
                    self.register_read_write_dimensions()?,
                    PolynomialEncoding::Dense,
                ),
                V::RamValFinal => (self.ram_final_dimensions()?, PolynomialEncoding::Dense),
                V::InstructionRa(index) => {
                    require_index(index, self.instruction_virtual_ra_count()?)?;
                    (
                        self.instruction_virtual_ra_dimensions()?,
                        PolynomialEncoding::Dense,
                    )
                }
                V::LookupTableFlag(index) => {
                    require_index(index, LookupTableKind::<RV64_XLEN>::COUNT)?;
                    (self.trace_dimensions()?, PolynomialEncoding::Dense)
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
                | V::InstructionFlags(_) => (self.trace_dimensions()?, PolynomialEncoding::Dense),
                V::Rd | V::InstructionRaf | V::RamValInit => {
                    return Err(not_served(id, UNSERVED_REASON));
                }
                V::UnivariateSkip
                | V::BytecodeValStage(_)
                | V::BytecodeReadRafAddrClaim
                | V::BooleanityAddrClaim
                | V::BytecodeClaimReductionIntermediate
                | V::ProgramImageInitContributionRw => {
                    return Err(not_served(id, PROTOCOL_INTERMEDIATE_REASON));
                }
                V::FusedInc => {
                    return Err(not_served(id, LATTICE_REASON));
                }
            },
        };
        Ok(Shape::new(dimensions, encoding))
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
                | V::InstructionFlags(_)
                | V::LookupTableFlag(_) => self.materialize_trace_virtual(virtual_id),
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
