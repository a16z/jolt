//! Per-row trace value extraction and cycle-domain virtual polynomials.

use super::*;

impl<T: TraceSource + Clone> TraceBackedJoltVmWitness<'_, T> {
    pub(crate) fn materialize_trace_virtual<F: Field>(
        &self,
        id: JoltVirtualPolynomial,
    ) -> Result<Vec<F>, WitnessError> {
        if !supported_trace_virtual(id) {
            return Err(WitnessError::UnknownOracle {
                namespace: JOLT_VM_NAMESPACE.name,
            });
        }

        let rows = checked_pow2(self.config.log_t)?;
        let mut values = Vec::with_capacity(rows);
        let mut trace = self.trace.trace.clone();
        let mut current = trace.next_row().unwrap_or_default();
        for index in 0..rows {
            let next = (index + 1 < rows).then(|| trace.next_row().unwrap_or_default());
            let value = trace_virtual_value::<F>(&current, next.as_ref(), id, self.preprocessing)?;
            values.push(value);
            if let Some(row) = next {
                current = row;
            }
        }
        Ok(values)
    }

    pub(crate) fn evaluate_committed_trace_dense<F: Field>(
        &self,
        id: JoltCommittedPolynomial,
        point: &[F],
    ) -> Result<F, WitnessError> {
        if point.len() != self.config.log_t {
            return Err(WitnessError::InvalidDimensions {
                namespace: JOLT_VM_NAMESPACE.name,
                reason: format!(
                    "committed dense trace point has {} variables, expected {}",
                    point.len(),
                    self.config.log_t
                ),
            });
        }

        let rows = checked_pow2(self.config.log_t)?;
        let eq = eq_evals_msb(point)?;
        let mut stream = self.committed_stream(id, rows.max(1))?;
        let mut index = 0usize;
        let mut result = F::zero();
        loop {
            let next: Option<PolynomialChunk<F>> = stream.next_chunk()?;
            let Some(chunk) = next else {
                break;
            };
            match chunk {
                PolynomialChunk::I128(values) => {
                    for value in values {
                        if index >= rows {
                            return Err(WitnessError::InvalidWitnessData {
                                namespace: JOLT_VM_NAMESPACE.name,
                                reason: format!(
                                    "committed dense stream for {id:?} exceeded {rows} rows"
                                ),
                            });
                        }
                        result += eq[index] * F::from_i128(value);
                        index += 1;
                    }
                }
                PolynomialChunk::Dense(values) => {
                    for value in values {
                        if index >= rows {
                            return Err(WitnessError::InvalidWitnessData {
                                namespace: JOLT_VM_NAMESPACE.name,
                                reason: format!(
                                    "committed dense stream for {id:?} exceeded {rows} rows"
                                ),
                            });
                        }
                        result += eq[index] * value;
                        index += 1;
                    }
                }
                PolynomialChunk::Zeros(count) => {
                    index = index.checked_add(count).ok_or_else(|| {
                        WitnessError::InvalidWitnessData {
                            namespace: JOLT_VM_NAMESPACE.name,
                            reason: format!(
                                "committed dense stream for {id:?} zero chunk overflowed row count"
                            ),
                        }
                    })?;
                    if index > rows {
                        return Err(WitnessError::InvalidWitnessData {
                            namespace: JOLT_VM_NAMESPACE.name,
                            reason: format!(
                                "committed dense stream for {id:?} exceeded {rows} rows"
                            ),
                        });
                    }
                }
                _ => {
                    return Err(WitnessError::InvalidWitnessData {
                        namespace: JOLT_VM_NAMESPACE.name,
                        reason: format!("committed dense stream for {id:?} used non-dense chunks"),
                    });
                }
            }
        }
        if index != rows {
            return Err(WitnessError::InvalidWitnessData {
                namespace: JOLT_VM_NAMESPACE.name,
                reason: format!(
                    "committed dense stream for {id:?} produced {index} rows, expected {rows}"
                ),
            });
        }
        Ok(result)
    }

    pub(crate) fn evaluate_trace_virtual<F: Field>(
        &self,
        id: JoltVirtualPolynomial,
        point: &[F],
    ) -> Result<F, WitnessError> {
        if point.len() != self.config.log_t {
            return Err(WitnessError::InvalidDimensions {
                namespace: JOLT_VM_NAMESPACE.name,
                reason: format!(
                    "trace virtual point has {} variables, expected {}",
                    point.len(),
                    self.config.log_t
                ),
            });
        }

        let cycles = checked_pow2(self.config.log_t)?;
        let mut trace = self.trace.trace.clone();
        let mut current = trace.next_row().unwrap_or_default();
        let mut result = F::zero();
        for index in 0..cycles {
            let next = (index + 1 < cycles).then(|| trace.next_row().unwrap_or_default());
            let value = trace_virtual_value::<F>(&current, next.as_ref(), id, self.preprocessing)?;
            result += value * eq_index_msb(point, index as u128);
            if let Some(row) = next {
                current = row;
            }
        }
        Ok(result)
    }
}

pub(crate) const fn supported_trace_virtual(id: JoltVirtualPolynomial) -> bool {
    matches!(
        id,
        JoltVirtualPolynomial::PC
            | JoltVirtualPolynomial::UnexpandedPC
            | JoltVirtualPolynomial::NextPC
            | JoltVirtualPolynomial::NextUnexpandedPC
            | JoltVirtualPolynomial::NextIsNoop
            | JoltVirtualPolynomial::NextIsVirtual
            | JoltVirtualPolynomial::NextIsFirstInSequence
            | JoltVirtualPolynomial::LeftLookupOperand
            | JoltVirtualPolynomial::RightLookupOperand
            | JoltVirtualPolynomial::LeftInstructionInput
            | JoltVirtualPolynomial::RightInstructionInput
            | JoltVirtualPolynomial::Product
            | JoltVirtualPolynomial::ShouldJump
            | JoltVirtualPolynomial::ShouldBranch
            | JoltVirtualPolynomial::Imm
            | JoltVirtualPolynomial::Rs1Value
            | JoltVirtualPolynomial::Rs2Value
            | JoltVirtualPolynomial::RdWriteValue
            | JoltVirtualPolynomial::LookupOutput
            | JoltVirtualPolynomial::RamAddress
            | JoltVirtualPolynomial::RamReadValue
            | JoltVirtualPolynomial::RamWriteValue
            | JoltVirtualPolynomial::RamHammingWeight
            | JoltVirtualPolynomial::InstructionRafFlag
            | JoltVirtualPolynomial::OpFlags(_)
            | JoltVirtualPolynomial::InstructionFlags(_)
            | JoltVirtualPolynomial::LookupTableFlag(_)
    )
}

pub(crate) fn trace_virtual_value<F: Field>(
    row: &TraceRow,
    next: Option<&TraceRow>,
    id: JoltVirtualPolynomial,
    preprocessing: &JoltProgramPreprocessing,
) -> Result<F, WitnessError> {
    let instruction = JoltInstruction::try_from(row.instruction).map_err(|kind| {
        WitnessError::InvalidWitnessData {
            namespace: JOLT_VM_NAMESPACE.name,
            reason: format!("unsupported Jolt instruction kind in trace row: {kind:?}"),
        }
    })?;
    let circuit_flags = instruction.circuit_flags();
    let instruction_flags = instruction.instruction_flags();
    let query = JoltLookupQuery::new(row.instruction.instruction_kind, row);

    let value = match id {
        JoltVirtualPolynomial::PC => F::from_u64(pc_for_row(row, preprocessing)? as u64),
        JoltVirtualPolynomial::UnexpandedPC => F::from_u64(row.instruction.address as u64),
        JoltVirtualPolynomial::NextPC => next
            .map(|row| pc_for_row(row, preprocessing).map(|pc| F::from_u64(pc as u64)))
            .transpose()?
            .unwrap_or_else(F::zero),
        JoltVirtualPolynomial::NextUnexpandedPC => {
            next.map_or_else(F::zero, |row| F::from_u64(row.instruction.address as u64))
        }
        // The last cycle's missing successor counts as a no-op: the
        // product/shift family requires `NextIsNoop = 1` at `T - 1` (legacy
        // forces `not_next_noop = false` there -- "EqPlusOne does not do
        // overflow").
        JoltVirtualPolynomial::NextIsNoop => F::from_bool(next.is_none_or(row_is_noop)),
        JoltVirtualPolynomial::NextIsVirtual => F::from_bool(next.is_some_and(|row| {
            row_circuit_flags(row).is_ok_and(|flags| flags[CircuitFlags::VirtualInstruction])
        })),
        JoltVirtualPolynomial::NextIsFirstInSequence => F::from_bool(next.is_some_and(|row| {
            row_circuit_flags(row).is_ok_and(|flags| flags[CircuitFlags::IsFirstInSequence])
        })),
        JoltVirtualPolynomial::LeftLookupOperand => {
            let (left, _) = LookupQuery::<RV64_XLEN>::to_lookup_operands(&query);
            F::from_u64(left)
        }
        JoltVirtualPolynomial::RightLookupOperand => {
            let (_, right) = LookupQuery::<RV64_XLEN>::to_lookup_operands(&query);
            F::from_u128(right)
        }
        JoltVirtualPolynomial::LeftInstructionInput => {
            let (left, _) = LookupQuery::<RV64_XLEN>::to_instruction_inputs(&query);
            F::from_u64(left)
        }
        JoltVirtualPolynomial::RightInstructionInput => {
            let (_, right) = LookupQuery::<RV64_XLEN>::to_instruction_inputs(&query);
            F::from_i128(right)
        }
        JoltVirtualPolynomial::Product => {
            let (left, right) = LookupQuery::<RV64_XLEN>::to_instruction_inputs(&query);
            let product = S64::from_u64(left).mul_trunc::<2, 2>(&S128::from_i128(right));
            signed_128_to_field(product)
        }
        JoltVirtualPolynomial::ShouldJump => {
            let next_is_noop = next.is_some_and(row_is_noop);
            F::from_bool(circuit_flags[CircuitFlags::Jump] && !next_is_noop)
        }
        JoltVirtualPolynomial::ShouldBranch => {
            let lookup_output = LookupQuery::<RV64_XLEN>::to_lookup_output(&query);
            F::from_bool(instruction_flags[InstructionFlags::Branch] && lookup_output == 1)
        }
        JoltVirtualPolynomial::Imm => F::from_i128(row.instruction.operands.imm),
        JoltVirtualPolynomial::Rs1Value => {
            F::from_u64(row.registers.rs1.map_or(0, |read| read.value))
        }
        JoltVirtualPolynomial::Rs2Value => {
            F::from_u64(row.registers.rs2.map_or(0, |read| read.value))
        }
        JoltVirtualPolynomial::RdWriteValue => {
            F::from_u64(row.registers.rd.map_or(0, |write| write.post_value))
        }
        JoltVirtualPolynomial::LookupOutput => {
            F::from_u64(LookupQuery::<RV64_XLEN>::to_lookup_output(&query))
        }
        JoltVirtualPolynomial::InstructionRafFlag => {
            F::from_bool(!circuit_flags.is_interleaved_operands())
        }
        JoltVirtualPolynomial::LookupTableFlag(index) => {
            if index >= LookupTableKind::<RV64_XLEN>::COUNT {
                return Err(WitnessError::UnknownOracle {
                    namespace: JOLT_VM_NAMESPACE.name,
                });
            }
            let table_index =
                <JoltInstruction as InstructionLookupTable<RV64_XLEN>>::lookup_table(&instruction)
                    .map(|table| table.index());
            F::from_bool(table_index == Some(index))
        }
        JoltVirtualPolynomial::RamAddress => F::from_u64(match row.ram_access {
            RamAccess::Read(read) => read.address,
            RamAccess::Write(write) => write.address,
            RamAccess::NoOp => 0,
        }),
        JoltVirtualPolynomial::RamReadValue => F::from_u64(match row.ram_access {
            RamAccess::Read(read) => read.value,
            RamAccess::Write(write) => write.pre_value,
            RamAccess::NoOp => 0,
        }),
        JoltVirtualPolynomial::RamWriteValue => F::from_u64(match row.ram_access {
            RamAccess::Read(read) => read.value,
            RamAccess::Write(write) => write.post_value,
            RamAccess::NoOp => 0,
        }),
        JoltVirtualPolynomial::RamHammingWeight => {
            F::from_bool(ram_access_address(row.ram_access).is_some_and(|address| address != 0))
        }
        JoltVirtualPolynomial::OpFlags(flag) => F::from_bool(circuit_flags[flag]),
        JoltVirtualPolynomial::InstructionFlags(flag) => F::from_bool(instruction_flags[flag]),
        _ => {
            return Err(WitnessError::UnknownOracle {
                namespace: JOLT_VM_NAMESPACE.name,
            });
        }
    };

    Ok(value)
}

pub(crate) fn row_circuit_flags(
    row: &TraceRow,
) -> Result<jolt_riscv::CircuitFlagSet, WitnessError> {
    Ok(JoltInstruction::try_from(row.instruction)
        .map_err(|kind| WitnessError::InvalidWitnessData {
            namespace: JOLT_VM_NAMESPACE.name,
            reason: format!("unsupported Jolt instruction kind in trace row: {kind:?}"),
        })?
        .circuit_flags())
}

pub(crate) fn row_instruction_flags(
    row: &TraceRow,
) -> Result<jolt_riscv::InstructionFlagSet, WitnessError> {
    Ok(JoltInstruction::try_from(row.instruction)
        .map_err(|kind| WitnessError::InvalidWitnessData {
            namespace: JOLT_VM_NAMESPACE.name,
            reason: format!("unsupported Jolt instruction kind in trace row: {kind:?}"),
        })?
        .instruction_flags())
}

pub(crate) fn row_is_noop(row: &TraceRow) -> bool {
    row.instruction.instruction_kind == JoltInstructionKind::NoOp
}

#[derive(Clone, Debug, Default)]
pub(crate) struct PcLookupCache {
    values: HashMap<(usize, u16), usize>,
}

impl PcLookupCache {
    pub(crate) fn pc_for_row(
        &mut self,
        row: &TraceRow,
        preprocessing: &JoltProgramPreprocessing,
    ) -> Result<usize, WitnessError> {
        self.pc_for_row_optional(row, preprocessing)
            .ok_or_else(|| missing_pc_mapping(row))
    }

    pub(crate) fn pc_for_row_optional(
        &mut self,
        row: &TraceRow,
        preprocessing: &JoltProgramPreprocessing,
    ) -> Option<usize> {
        if row_is_noop(row) {
            return Some(0);
        }
        let key = pc_lookup_key(row);
        if let Some(&pc) = self.values.get(&key) {
            return Some(pc);
        }
        let pc = preprocessing.bytecode.get_pc(&row.instruction)?;
        let _ = self.values.insert(key, pc);
        Some(pc)
    }
}

pub(crate) fn pc_lookup_key(row: &TraceRow) -> (usize, u16) {
    (
        row.instruction.address,
        row.instruction.virtual_sequence_remaining.unwrap_or(0),
    )
}

pub(crate) fn pc_for_row(
    row: &TraceRow,
    preprocessing: &JoltProgramPreprocessing,
) -> Result<usize, WitnessError> {
    preprocessing
        .bytecode
        .get_pc(&row.instruction)
        .ok_or_else(|| missing_pc_mapping(row))
}

pub(crate) fn missing_pc_mapping(row: &TraceRow) -> WitnessError {
    WitnessError::InvalidWitnessData {
        namespace: JOLT_VM_NAMESPACE.name,
        reason: format!(
            "bytecode preprocessing is missing PC mapping for address {:#x} with virtual_sequence_remaining {:?}",
            row.instruction.address, row.instruction.virtual_sequence_remaining
        ),
    }
}

pub(crate) fn signed_128_to_field<F: Field>(value: S128) -> F {
    if let Some(value) = value.to_i128() {
        F::from_i128(value)
    } else {
        let magnitude = value.magnitude_as_u128();
        if value.is_positive {
            F::from_u128(magnitude)
        } else {
            -F::from_u128(magnitude)
        }
    }
}
