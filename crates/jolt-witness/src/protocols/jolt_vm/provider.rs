//! WitnessProvider implementation for the trace-backed Jolt VM witness.

use super::*;

impl<F: Field, T: TraceSource + Clone> crate::WitnessProvider<F, JoltVmNamespace>
    for TraceBackedJoltVmWitness<'_, T>
{
    fn describe_oracle(
        &self,
        oracle: OracleRef<JoltVmNamespace>,
    ) -> Result<OracleDescriptor<JoltVmNamespace>, WitnessError> {
        let dimensions = match oracle.kind {
            OracleKind::Committed(
                JoltCommittedPolynomial::RdInc | JoltCommittedPolynomial::RamInc,
            ) => self.trace_dimensions()?,
            OracleKind::Committed(JoltCommittedPolynomial::InstructionRa(index)) => {
                require_index(index, self.ra_layout()?.instruction())?;
                self.one_hot_dimensions()?
            }
            OracleKind::Committed(JoltCommittedPolynomial::BytecodeRa(index)) => {
                require_index(index, self.ra_layout()?.bytecode())?;
                self.one_hot_dimensions()?
            }
            OracleKind::Committed(JoltCommittedPolynomial::RamRa(index)) => {
                require_index(index, self.ra_layout()?.ram())?;
                self.one_hot_dimensions()?
            }
            OracleKind::Committed(JoltCommittedPolynomial::TrustedAdvice) => {
                if !self.config.include_trusted_advice {
                    return Err(WitnessError::UnknownOracle {
                        namespace: JOLT_VM_NAMESPACE.name,
                    });
                }
                Self::advice_dimensions(
                    self.preprocessing.memory_layout.max_trusted_advice_size as usize / 8,
                )
            }
            OracleKind::Committed(JoltCommittedPolynomial::UntrustedAdvice) => {
                if !self.config.include_untrusted_advice {
                    return Err(WitnessError::UnknownOracle {
                        namespace: JOLT_VM_NAMESPACE.name,
                    });
                }
                Self::advice_dimensions(
                    self.preprocessing.memory_layout.max_untrusted_advice_size as usize / 8,
                )
            }
            OracleKind::Committed(
                JoltCommittedPolynomial::BytecodeChunk(_)
                | JoltCommittedPolynomial::ProgramImageInit,
            ) => {
                return Err(WitnessError::UnknownOracle {
                    namespace: JOLT_VM_NAMESPACE.name,
                });
            }
            OracleKind::Virtual(JoltVirtualPolynomial::RamVal | JoltVirtualPolynomial::RamRa) => {
                self.ram_read_write_dimensions()?
            }
            OracleKind::Virtual(
                JoltVirtualPolynomial::RegistersVal
                | JoltVirtualPolynomial::Rs1Ra
                | JoltVirtualPolynomial::Rs2Ra
                | JoltVirtualPolynomial::RdWa,
            ) => self.register_read_write_dimensions()?,
            OracleKind::Virtual(JoltVirtualPolynomial::RamValFinal) => {
                self.ram_final_dimensions()?
            }
            OracleKind::Virtual(JoltVirtualPolynomial::InstructionRa(index)) => {
                require_index(index, self.instruction_virtual_ra_count()?)?;
                self.instruction_virtual_ra_dimensions()?
            }
            OracleKind::Virtual(JoltVirtualPolynomial::LookupTableFlag(index)) => {
                require_index(index, LookupTableKind::<RV64_XLEN>::COUNT)?;
                self.trace_dimensions()?
            }
            OracleKind::Virtual(JoltVirtualPolynomial::InstructionRafFlag) => {
                self.trace_dimensions()?
            }
            OracleKind::Virtual(id) if supported_trace_virtual(id) => self.trace_dimensions()?,
            OracleKind::Virtual(_) => {
                return Err(WitnessError::UnknownOracle {
                    namespace: JOLT_VM_NAMESPACE.name,
                });
            }
        };
        Ok(OracleDescriptor::new(
            oracle,
            dimensions,
            oracle_encoding(oracle.kind),
        ))
    }

    fn view_requirements(
        &self,
        oracle: OracleRef<JoltVmNamespace>,
    ) -> Result<Vec<ViewRequirement<JoltVmNamespace>>, WitnessError> {
        let descriptor =
            <Self as crate::WitnessProvider<F, JoltVmNamespace>>::describe_oracle(self, oracle)?;
        let retention = match oracle.kind {
            OracleKind::Committed(
                JoltCommittedPolynomial::TrustedAdvice | JoltCommittedPolynomial::UntrustedAdvice,
            ) => RetentionHint::ThroughBlindFold,
            _ => RetentionHint::ThroughStage8,
        };
        Ok(vec![ViewRequirement::new(
            descriptor.reference,
            descriptor.encoding,
            MaterializationPolicy::BackendChoice,
            retention,
        )])
    }

    fn oracle_view(
        &self,
        requirement: ViewRequirement<JoltVmNamespace>,
    ) -> Result<PolynomialView<'_, F, JoltVmNamespace>, WitnessError> {
        let descriptor = <Self as crate::WitnessProvider<F, JoltVmNamespace>>::describe_oracle(
            self,
            requirement.oracle,
        )?;
        let values = match requirement.oracle.kind {
            OracleKind::Virtual(id)
                if matches!(
                    id,
                    JoltVirtualPolynomial::RamVal | JoltVirtualPolynomial::RamRa
                ) =>
            {
                self.materialize_ram_read_write_virtual(id)?
            }
            OracleKind::Virtual(id)
                if matches!(
                    id,
                    JoltVirtualPolynomial::RegistersVal
                        | JoltVirtualPolynomial::Rs1Ra
                        | JoltVirtualPolynomial::Rs2Ra
                        | JoltVirtualPolynomial::RdWa
                ) =>
            {
                self.materialize_register_read_write_virtual(id)?
            }
            OracleKind::Virtual(JoltVirtualPolynomial::RamValFinal) => {
                self.materialize_ram_val_final()?
            }
            OracleKind::Virtual(JoltVirtualPolynomial::InstructionRa(index)) => {
                self.materialize_instruction_ra(index)?
            }
            OracleKind::Virtual(id) => self.materialize_trace_virtual(id)?,
            OracleKind::Committed(id) => self.materialize_compact_committed(id)?,
        };
        Ok(PolynomialView::owned(descriptor, values))
    }

    fn try_evaluate_oracle_view(
        &self,
        requirement: ViewRequirement<JoltVmNamespace>,
        point: &[F],
    ) -> Result<Option<F>, WitnessError> {
        let directly_evaluates_committed = matches!(
            requirement.oracle.kind,
            OracleKind::Committed(
                JoltCommittedPolynomial::RdInc
                    | JoltCommittedPolynomial::RamInc
                    | JoltCommittedPolynomial::InstructionRa(_)
                    | JoltCommittedPolynomial::BytecodeRa(_)
                    | JoltCommittedPolynomial::RamRa(_)
            )
        );
        if requirement.encoding != PolynomialEncoding::Dense && !directly_evaluates_committed {
            return Ok(None);
        }

        match requirement.oracle.kind {
            OracleKind::Committed(
                id @ (JoltCommittedPolynomial::RdInc | JoltCommittedPolynomial::RamInc),
            ) => self.evaluate_committed_trace_dense(id, point).map(Some),
            OracleKind::Committed(
                id @ (JoltCommittedPolynomial::InstructionRa(_)
                | JoltCommittedPolynomial::BytecodeRa(_)
                | JoltCommittedPolynomial::RamRa(_)),
            ) => self.evaluate_committed_ra(id, point).map(Some),
            OracleKind::Virtual(JoltVirtualPolynomial::InstructionRa(index)) => {
                self.evaluate_instruction_ra(index, point).map(Some)
            }
            OracleKind::Virtual(
                id @ (JoltVirtualPolynomial::RamVal | JoltVirtualPolynomial::RamRa),
            ) => self.evaluate_ram_read_write_virtual(id, point).map(Some),
            OracleKind::Virtual(JoltVirtualPolynomial::RamValFinal) => {
                self.evaluate_ram_val_final(point).map(Some)
            }
            OracleKind::Virtual(
                id @ (JoltVirtualPolynomial::RegistersVal
                | JoltVirtualPolynomial::Rs1Ra
                | JoltVirtualPolynomial::Rs2Ra
                | JoltVirtualPolynomial::RdWa),
            ) => self
                .evaluate_register_read_write_virtual(id, point)
                .map(Some),
            OracleKind::Virtual(
                id @ (JoltVirtualPolynomial::InstructionRafFlag
                | JoltVirtualPolynomial::LookupTableFlag(_)
                | JoltVirtualPolynomial::RamHammingWeight),
            ) => self.evaluate_trace_virtual(id, point).map(Some),
            _ => Ok(None),
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

pub(crate) const fn oracle_encoding(
    kind: OracleKind<JoltCommittedPolynomial, JoltVirtualPolynomial>,
) -> PolynomialEncoding {
    match kind {
        OracleKind::Committed(
            JoltCommittedPolynomial::InstructionRa(_)
            | JoltCommittedPolynomial::BytecodeRa(_)
            | JoltCommittedPolynomial::RamRa(_),
        ) => PolynomialEncoding::OneHot,
        OracleKind::Committed(_) => PolynomialEncoding::Compact,
        OracleKind::Virtual(_) => PolynomialEncoding::Dense,
    }
}
