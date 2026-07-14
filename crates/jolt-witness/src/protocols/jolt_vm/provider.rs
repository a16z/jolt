//! WitnessProvider implementation for the trace-backed Jolt VM witness.

use super::*;

impl<F: Field, T: TraceSource + Clone> crate::WitnessProvider<F, JoltVmNamespace>
    for TraceBackedJoltVmWitness<'_, T>
{
    fn describe_oracle(
        &self,
        oracle: OracleRef<JoltVmNamespace>,
    ) -> Result<OracleDescriptor<JoltVmNamespace>, WitnessError> {
        let dimensions = match oracle {
            OracleRef::Committed(
                JoltCommittedPolynomial::RdInc | JoltCommittedPolynomial::RamInc,
            ) => self.trace_dimensions()?,
            OracleRef::Committed(JoltCommittedPolynomial::InstructionRa(index)) => {
                require_index(index, self.ra_layout()?.instruction())?;
                self.one_hot_dimensions()?
            }
            OracleRef::Committed(JoltCommittedPolynomial::BytecodeRa(index)) => {
                require_index(index, self.ra_layout()?.bytecode())?;
                self.one_hot_dimensions()?
            }
            OracleRef::Committed(JoltCommittedPolynomial::RamRa(index)) => {
                require_index(index, self.ra_layout()?.ram())?;
                self.one_hot_dimensions()?
            }
            OracleRef::Committed(JoltCommittedPolynomial::TrustedAdvice) => {
                if !self.config.include_trusted_advice {
                    return Err(WitnessError::UnknownOracle {
                        namespace: JOLT_VM_NAMESPACE.name,
                    });
                }
                Self::advice_dimensions(
                    self.preprocessing.memory_layout.max_trusted_advice_size as usize / 8,
                )
            }
            OracleRef::Committed(JoltCommittedPolynomial::UntrustedAdvice) => {
                if !self.config.include_untrusted_advice {
                    return Err(WitnessError::UnknownOracle {
                        namespace: JOLT_VM_NAMESPACE.name,
                    });
                }
                Self::advice_dimensions(
                    self.preprocessing.memory_layout.max_untrusted_advice_size as usize / 8,
                )
            }
            OracleRef::Committed(
                JoltCommittedPolynomial::BytecodeChunk(_)
                | JoltCommittedPolynomial::ProgramImageInit
                | JoltCommittedPolynomial::UnsignedIncChunk(_)
                | JoltCommittedPolynomial::UnsignedIncMsb
                | JoltCommittedPolynomial::TrustedAdviceBytes
                | JoltCommittedPolynomial::UntrustedAdviceBytes
                | JoltCommittedPolynomial::BytecodeRegisterSelector { .. }
                | JoltCommittedPolynomial::BytecodeCircuitFlag { .. }
                | JoltCommittedPolynomial::BytecodeInstructionFlag { .. }
                | JoltCommittedPolynomial::BytecodeLookupSelector { .. }
                | JoltCommittedPolynomial::BytecodeRafFlag { .. }
                | JoltCommittedPolynomial::BytecodeUnexpandedPcBytes { .. }
                | JoltCommittedPolynomial::BytecodeImmBytes { .. }
                | JoltCommittedPolynomial::ProgramImageBytes,
            ) => {
                return Err(WitnessError::UnknownOracle {
                    namespace: JOLT_VM_NAMESPACE.name,
                });
            }
            OracleRef::Virtual(JoltVirtualPolynomial::RamVal | JoltVirtualPolynomial::RamRa) => {
                self.ram_read_write_dimensions()?
            }
            OracleRef::Virtual(
                JoltVirtualPolynomial::RegistersVal
                | JoltVirtualPolynomial::Rs1Ra
                | JoltVirtualPolynomial::Rs2Ra
                | JoltVirtualPolynomial::RdWa,
            ) => self.register_read_write_dimensions()?,
            OracleRef::Virtual(JoltVirtualPolynomial::RamValFinal) => {
                self.ram_final_dimensions()?
            }
            OracleRef::Virtual(JoltVirtualPolynomial::InstructionRa(index)) => {
                require_index(index, self.instruction_virtual_ra_count()?)?;
                self.instruction_virtual_ra_dimensions()?
            }
            OracleRef::Virtual(JoltVirtualPolynomial::LookupTableFlag(index)) => {
                require_index(index, LookupTableKind::<RV64_XLEN>::COUNT)?;
                self.trace_dimensions()?
            }
            OracleRef::Virtual(JoltVirtualPolynomial::InstructionRafFlag) => {
                self.trace_dimensions()?
            }
            OracleRef::Virtual(id) if supported_trace_virtual(id) => self.trace_dimensions()?,
            OracleRef::Virtual(_) => {
                return Err(WitnessError::UnknownOracle {
                    namespace: JOLT_VM_NAMESPACE.name,
                });
            }
        };
        Ok(OracleDescriptor::new(
            oracle,
            dimensions,
            oracle_encoding(oracle),
        ))
    }

    fn oracle_table(&self, oracle: OracleRef<JoltVmNamespace>) -> Result<Vec<F>, WitnessError> {
        let _ =
            <Self as crate::WitnessProvider<F, JoltVmNamespace>>::describe_oracle(self, oracle)?;
        match oracle {
            OracleRef::Virtual(
                id @ (JoltVirtualPolynomial::RamVal | JoltVirtualPolynomial::RamRa),
            ) => self.materialize_ram_read_write_virtual(id),
            OracleRef::Virtual(
                id @ (JoltVirtualPolynomial::RegistersVal
                | JoltVirtualPolynomial::Rs1Ra
                | JoltVirtualPolynomial::Rs2Ra
                | JoltVirtualPolynomial::RdWa),
            ) => self.materialize_register_read_write_virtual(id),
            OracleRef::Virtual(JoltVirtualPolynomial::RamValFinal) => {
                self.materialize_ram_val_final()
            }
            OracleRef::Virtual(JoltVirtualPolynomial::InstructionRa(index)) => {
                self.materialize_instruction_ra(index)
            }
            OracleRef::Virtual(id) => self.materialize_trace_virtual(id),
            OracleRef::Committed(id) => self.materialize_compact_committed(id),
        }
    }

    fn committed_stream<'a>(
        &'a self,
        id: JoltCommittedPolynomial,
        chunk_size: usize,
    ) -> Result<Box<dyn PolynomialStream<F> + 'a>, WitnessError>
    where
        F: 'a,
        JoltVmNamespace: 'a,
    {
        Ok(Box::new(TraceBackedJoltVmWitness::committed_stream(
            self, id, chunk_size,
        )?))
    }

    fn committed_batch_stream<'a>(
        &'a self,
        ids: &'a [JoltCommittedPolynomial],
        chunk_size: usize,
    ) -> Result<Box<dyn PolynomialBatchStream<F, JoltVmNamespace> + 'a>, WitnessError>
    where
        F: 'a,
        JoltVmNamespace: 'a,
        JoltCommittedPolynomial: 'a,
    {
        Ok(Box::new(TraceBackedJoltVmWitness::committed_batch_stream(
            self, ids, chunk_size,
        )?))
    }
}

impl<F: Field, T: TraceSource + Clone> crate::CommittedWitnessProvider<F, JoltVmNamespace>
    for TraceBackedJoltVmWitness<'_, T>
{
    fn committed_oracle_order(&self) -> Result<Vec<JoltCommittedPolynomial>, WitnessError> {
        self.committed_polynomial_order()
    }
}

pub(crate) const fn oracle_encoding(kind: OracleRef<JoltVmNamespace>) -> PolynomialEncoding {
    match kind {
        OracleRef::Committed(
            JoltCommittedPolynomial::InstructionRa(_)
            | JoltCommittedPolynomial::BytecodeRa(_)
            | JoltCommittedPolynomial::RamRa(_),
        ) => PolynomialEncoding::OneHot,
        OracleRef::Committed(_) => PolynomialEncoding::Compact,
        OracleRef::Virtual(_) => PolynomialEncoding::Dense,
    }
}
