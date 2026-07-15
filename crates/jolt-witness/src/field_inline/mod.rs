use core::marker::PhantomData;

use jolt_claims::protocols::field_inline::{
    FieldInlineCommittedPolynomial, FieldInlinePolynomialId, FieldInlineVirtualPolynomial,
    FIELD_REGISTERS_LOG_K,
};
use jolt_field::Field;
use jolt_program::{
    execution::{JoltProgram, TraceOutput, TraceRow, TraceSource},
    field_inline::{
        FieldEncodedValue, FieldInlineBridge, FieldInlineTraceData, FieldRegisterRead,
        FieldRegisterWrite,
    },
    preprocess::JoltProgramPreprocessing,
};
use jolt_riscv::{field_inline_operand_shape, FieldInlineOperandShape, FieldInlineXRegisterRole};
use rayon::prelude::*;

use self::witnesses::{
    decode_value, FieldInvProduct, FieldOpFlag, FieldProduct, FieldRdInc, FieldRdValue,
    FieldRs1Value, FieldRs2Value, FieldValue,
};
use crate::backend::trace::{checked_pow2, TraceBackend};
use crate::witnesses::{Extract, ExtractIndexed, WitnessEnv};
use crate::{PolynomialChunk, PolynomialEncoding, PolynomialStream, Shape, WitnessError};

pub mod witnesses;

/// Error label for the field-inline witness backend.
pub const FIELD_INLINE_LABEL: &str = "jolt_vm.field_inline";

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FieldInlineRegisterReadRow<F: Field> {
    pub register: u8,
    pub value: F,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FieldInlineRegisterWriteRow<F: Field> {
    pub register: u8,
    pub pre_value: F,
    pub post_value: F,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FieldInlineRegisterReadWriteRow<F: Field> {
    pub rs1: Option<FieldInlineRegisterReadRow<F>>,
    pub rs2: Option<FieldInlineRegisterReadRow<F>>,
    pub rd: Option<FieldInlineRegisterWriteRow<F>>,
    pub rd_increment: F,
}

pub trait FieldInlineRegisterReadWriteRows<F: Field> {
    fn field_inline_register_read_write_rows(
        &self,
    ) -> Result<Vec<FieldInlineRegisterReadWriteRow<F>>, WitnessError>;
}

pub struct TraceBackedFieldInlineWitness<'a> {
    log_t: usize,
    program: &'a JoltProgram,
    preprocessing: &'a JoltProgramPreprocessing,
    trace_rows: Vec<TraceRow>,
    rows: usize,
}

fn collect_trace_rows<T: TraceSource + Clone>(
    trace: &TraceOutput<T>,
    rows: usize,
) -> Result<Vec<TraceRow>, WitnessError> {
    let mut source = trace.trace.clone();
    let mut trace_rows = Vec::with_capacity(rows);
    for _ in 0..rows {
        let Some(row) = source.next_row() else {
            return Ok(trace_rows);
        };
        trace_rows.push(row);
    }
    if source.next_row().is_some() {
        return Err(WitnessError::InvalidWitnessData {
            label: FIELD_INLINE_LABEL,
            reason: "trace length exceeds configured field-inline witness domain".to_owned(),
        });
    }
    Ok(trace_rows)
}

impl<'a> TraceBackedFieldInlineWitness<'a> {
    pub(crate) fn build<T: TraceSource + Clone>(
        log_t: usize,
        program: &'a JoltProgram,
        preprocessing: &'a JoltProgramPreprocessing,
        trace: &TraceOutput<T>,
    ) -> Result<Self, WitnessError> {
        let rows = checked_pow2(log_t)?;
        let trace_rows = collect_trace_rows(trace, rows)?;
        let witness = Self {
            log_t,
            program,
            preprocessing,
            trace_rows,
            rows,
        };
        witness.validate_inputs()?;
        Ok(witness)
    }

    fn trace_log_rows(&self) -> usize {
        self.log_t
    }

    fn field_register_log_rows(&self) -> Result<usize, WitnessError> {
        self.log_t
            .checked_add(FIELD_REGISTERS_LOG_K)
            .ok_or_else(|| WitnessError::InvalidDimensions {
                label: FIELD_INLINE_LABEL,
                reason: "field-register witness row count overflow".to_owned(),
            })
    }

    fn validate_inputs(&self) -> Result<(), WitnessError> {
        if !self.program.profile.supports_field_inline() {
            for (index, row) in self.trace_rows.iter().enumerate() {
                if row.field_inline.is_some()
                    || field_inline_operand_shape(row.instruction.instruction_kind).is_some()
                {
                    return Err(invalid_row(
                        index,
                        "field-inline trace data exists for an FR-disabled program",
                    ));
                }
            }
            return Err(WitnessError::UnavailableView {
                label: FIELD_INLINE_LABEL,
            });
        }

        let metadata = self
            .preprocessing
            .bytecode
            .field_inline
            .as_ref()
            .ok_or_else(|| WitnessError::InvalidWitnessData {
                label: FIELD_INLINE_LABEL,
                reason: "FR-enabled program is missing field-inline bytecode metadata".to_owned(),
            })?;
        metadata
            .validate(self.preprocessing.bytecode.bytecode.len())
            .map_err(|error| WitnessError::InvalidWitnessData {
                label: FIELD_INLINE_LABEL,
                reason: error.to_string(),
            })?;

        for (index, row) in self.trace_rows.iter().enumerate() {
            self.validate_row_shape(index, row)?;
        }
        validate_field_register_state(&self.trace_rows)
    }

    fn validate_row_shape(&self, index: usize, row: &TraceRow) -> Result<(), WitnessError> {
        let shape = field_inline_operand_shape(row.instruction.instruction_kind);
        let metadata = self
            .preprocessing
            .bytecode
            .field_inline
            .as_ref()
            .ok_or_else(|| WitnessError::InvalidWitnessData {
                label: FIELD_INLINE_LABEL,
                reason: "FR-enabled program is missing field-inline bytecode metadata".to_owned(),
            })?;

        match (shape, row.field_inline.as_deref()) {
            (None, None) => Ok(()),
            (None, Some(_)) => Err(invalid_row(
                index,
                "ordinary RV64 row carries field-inline trace data",
            )),
            (Some(_), None) => Err(invalid_row(
                index,
                "field-inline instruction is missing field-inline trace data",
            )),
            (Some(shape), Some(data)) => {
                let pc = self
                    .preprocessing
                    .bytecode
                    .get_pc(&row.instruction)
                    .ok_or_else(|| invalid_row(index, "field-inline row has no bytecode pc"))?;
                let bytecode_row = metadata.rows.get(pc).ok_or_else(|| {
                    invalid_row(index, "field-inline bytecode pc is out of range")
                })?;
                if !bytecode_row.active || bytecode_row.op != Some(shape.op) {
                    return Err(invalid_row(
                        index,
                        "field-inline trace op does not match bytecode metadata",
                    ));
                }
                validate_trace_data(index, row, shape, *data)
            }
        }
    }

    /// Materializes one cycle-domain witness column; rows beyond the trace
    /// are zero. All per-witness logic lives on `W`.
    fn materialize_cycle<F: Field, W: Extract + FieldValue<F> + Send>(
        &self,
    ) -> Result<Vec<F>, WitnessError> {
        self.walk_cycles(|row, env| W::extract(row, None, env).map(FieldValue::value))
    }

    /// [`Self::materialize_cycle`] for indexed witness families.
    fn materialize_cycle_indexed<F: Field, W: ExtractIndexed<I> + FieldValue<F>, I: Copy + Sync>(
        &self,
        index: I,
    ) -> Result<Vec<F>, WitnessError> {
        self.walk_cycles(|row, env| {
            W::extract_indexed(index, row, None, env).map(FieldValue::value)
        })
    }

    fn walk_cycles<F: Field>(
        &self,
        value: impl Fn(&TraceRow, &WitnessEnv<'_>) -> Result<F, WitnessError> + Sync,
    ) -> Result<Vec<F>, WitnessError> {
        let env = WitnessEnv {
            preprocessing: self.preprocessing,
        };
        let mut values = vec![F::from_u64(0); self.rows];
        values
            .par_iter_mut()
            .zip(self.trace_rows.par_iter())
            .try_for_each(|(slot, row)| {
                *slot = value(row, &env)?;
                Ok(())
            })?;
        Ok(values)
    }

    fn materialize_register_virtual<F: Field>(
        &self,
        id: FieldInlineVirtualPolynomial,
    ) -> Result<Vec<F>, WitnessError> {
        let register_count = field_register_count();
        let mut values = vec![F::from_u64(0); self.rows * register_count];

        if id == FieldInlineVirtualPolynomial::FieldRegistersVal {
            let trace_rows = self.trace_rows.as_slice();
            let trace_len = trace_rows.len();
            values
                .par_chunks_mut(self.rows)
                .enumerate()
                .for_each(|(register, values)| {
                    let mut current = F::zero();
                    for (cycle, value) in values.iter_mut().take(trace_len).enumerate() {
                        *value = current;
                        if let Some(write) = trace_rows[cycle]
                            .field_inline
                            .as_deref()
                            .and_then(|data| data.rd)
                        {
                            if usize::from(write.register) == register {
                                current = decode_value(write.post_value);
                            }
                        }
                    }
                    values[trace_len..].fill(current);
                });
            return Ok(values);
        }

        for (cycle, row) in self.trace_rows.iter().enumerate() {
            let Some(data) = row.field_inline.as_deref() else {
                continue;
            };
            let register = match id {
                FieldInlineVirtualPolynomial::FieldRs1Ra => data.rs1.map(|read| read.register),
                FieldInlineVirtualPolynomial::FieldRs2Ra => data.rs2.map(|read| read.register),
                FieldInlineVirtualPolynomial::FieldRdWa => data.rd.map(|write| write.register),
                _ => None,
            };
            if let Some(register) = register {
                values[usize::from(register) * self.rows + cycle] = F::from_u64(1);
            }
        }

        Ok(values)
    }
}

impl TraceBackedFieldInlineWitness<'_> {
    /// The exhaustive shape map over the field-inline id vocabulary — no
    /// wildcard arm, like the jolt-vm backend: a new jolt-claims variant
    /// fails compilation here until classified.
    pub fn shape(&self, id: FieldInlinePolynomialId) -> Result<Shape, WitnessError> {
        use FieldInlineVirtualPolynomial as V;
        match id {
            FieldInlinePolynomialId::Committed(FieldInlineCommittedPolynomial::FieldRdInc) => {
                Ok(Shape::new(self.trace_log_rows(), PolynomialEncoding::Dense))
            }
            FieldInlinePolynomialId::Virtual(virtual_id) => match virtual_id {
                V::FieldRs1Value
                | V::FieldRs2Value
                | V::FieldRdValue
                | V::FieldProduct
                | V::FieldInvProduct
                | V::FieldOpFlag(_) => {
                    Ok(Shape::new(self.trace_log_rows(), PolynomialEncoding::Dense))
                }
                V::FieldRs1Ra | V::FieldRs2Ra | V::FieldRdWa | V::FieldRegistersVal => Ok(
                    Shape::new(self.field_register_log_rows()?, PolynomialEncoding::Dense),
                ),
            },
        }
    }

    pub fn oracle_table<F: Field>(
        &self,
        id: FieldInlinePolynomialId,
    ) -> Result<Vec<F>, WitnessError> {
        use FieldInlineVirtualPolynomial as V;
        let _ = self.shape(id)?;
        match id {
            FieldInlinePolynomialId::Committed(FieldInlineCommittedPolynomial::FieldRdInc) => {
                self.materialize_cycle::<F, FieldRdInc<F>>()
            }
            FieldInlinePolynomialId::Virtual(virtual_id) => match virtual_id {
                V::FieldRs1Value => self.materialize_cycle::<F, FieldRs1Value<F>>(),
                V::FieldRs2Value => self.materialize_cycle::<F, FieldRs2Value<F>>(),
                V::FieldRdValue => self.materialize_cycle::<F, FieldRdValue<F>>(),
                V::FieldProduct => self.materialize_cycle::<F, FieldProduct<F>>(),
                V::FieldInvProduct => self.materialize_cycle::<F, FieldInvProduct<F>>(),
                V::FieldOpFlag(flag) => self.materialize_cycle_indexed::<F, FieldOpFlag, _>(flag),
                V::FieldRs1Ra | V::FieldRs2Ra | V::FieldRdWa | V::FieldRegistersVal => {
                    self.materialize_register_virtual(virtual_id)
                }
            },
        }
    }

    pub fn committed_stream<F: Field>(
        &self,
        id: FieldInlineCommittedPolynomial,
        chunk_size: usize,
    ) -> Result<FieldInlineCommittedStream<'_, F>, WitnessError> {
        if chunk_size == 0 {
            return Err(WitnessError::InvalidDimensions {
                label: FIELD_INLINE_LABEL,
                reason: "stream chunk size must be nonzero".to_owned(),
            });
        }
        match id {
            FieldInlineCommittedPolynomial::FieldRdInc => Ok(FieldInlineCommittedStream::<F> {
                trace_rows: &self.trace_rows,
                emitted: 0,
                rows: self.rows,
                chunk_size,
                preprocessing: self.preprocessing,
                all_zero: self.trace_rows.iter().all(|row| {
                    row.field_inline
                        .as_deref()
                        .and_then(|data| data.rd)
                        .is_none()
                }),
                _field: PhantomData,
            }),
        }
    }

    pub fn committed_order(&self) -> Vec<FieldInlineCommittedPolynomial> {
        vec![FieldInlineCommittedPolynomial::FieldRdInc]
    }
}

impl<F: Field> FieldInlineRegisterReadWriteRows<F> for TraceBackedFieldInlineWitness<'_> {
    fn field_inline_register_read_write_rows(
        &self,
    ) -> Result<Vec<FieldInlineRegisterReadWriteRow<F>>, WitnessError> {
        let env = WitnessEnv {
            preprocessing: self.preprocessing,
        };
        (0..self.rows)
            .map(|index| {
                self.trace_rows.get(index).map_or_else(
                    || Ok(FieldInlineRegisterReadWriteRow::default()),
                    |row| field_register_row(row, &env),
                )
            })
            .collect()
    }
}

impl<'a, T: TraceSource + Clone> TraceBackend<'a, T> {
    pub fn field_inline_witness(&self) -> Result<TraceBackedFieldInlineWitness<'a>, WitnessError> {
        TraceBackedFieldInlineWitness::build(
            self.config.log_t,
            self.program,
            self.preprocessing,
            &self.trace,
        )
    }

    /// Eagerly materializes and validates the field-inline witness view and
    /// stores it so this backend can serve the field-inline rows directly.
    pub fn with_field_inline(mut self) -> Result<Self, WitnessError> {
        self.field_inline = Some(self.field_inline_witness()?);
        Ok(self)
    }
}

impl<'a, T: TraceSource> TraceBackend<'a, T> {
    fn field_inline_view(&self) -> Result<&TraceBackedFieldInlineWitness<'a>, WitnessError> {
        self.field_inline
            .as_ref()
            .ok_or(WitnessError::UnavailableView {
                label: FIELD_INLINE_LABEL,
            })
    }
}

impl<F: Field, T: TraceSource> FieldInlineRegisterReadWriteRows<F> for TraceBackend<'_, T> {
    fn field_inline_register_read_write_rows(
        &self,
    ) -> Result<Vec<FieldInlineRegisterReadWriteRow<F>>, WitnessError> {
        self.field_inline_view()?
            .field_inline_register_read_write_rows()
    }
}

#[derive(Clone, Debug)]
pub struct FieldInlineCommittedStream<'a, F> {
    trace_rows: &'a [TraceRow],
    emitted: usize,
    rows: usize,
    chunk_size: usize,
    all_zero: bool,
    preprocessing: &'a JoltProgramPreprocessing,
    _field: PhantomData<F>,
}

impl<F: Field> PolynomialStream<F> for FieldInlineCommittedStream<'_, F> {
    fn next_chunk(&mut self) -> Result<Option<PolynomialChunk<F>>, WitnessError> {
        if self.emitted >= self.rows {
            return Ok(None);
        }
        if self.all_zero {
            let rows = self.rows - self.emitted;
            self.emitted = self.rows;
            return Ok(Some(PolynomialChunk::Zeros(rows)));
        }
        let env = WitnessEnv {
            preprocessing: self.preprocessing,
        };
        let end = self.emitted.saturating_add(self.chunk_size).min(self.rows);
        let mut values = Vec::with_capacity(end - self.emitted);
        while self.emitted < end {
            let value = match self.trace_rows.get(self.emitted) {
                Some(row) => FieldRdInc::extract(row, None, &env)?.0,
                None => F::zero(),
            };
            values.push(value);
            self.emitted += 1;
        }
        Ok(Some(PolynomialChunk::Dense(values)))
    }
}

fn field_register_row<F: Field>(
    row: &TraceRow,
    env: &WitnessEnv<'_>,
) -> Result<FieldInlineRegisterReadWriteRow<F>, WitnessError> {
    let Some(data) = row.field_inline.as_deref() else {
        return Ok(FieldInlineRegisterReadWriteRow::default());
    };
    let rs1 = data.rs1.map(|read| FieldInlineRegisterReadRow {
        register: read.register,
        value: decode_value(read.value),
    });
    let rs2 = data.rs2.map(|read| FieldInlineRegisterReadRow {
        register: read.register,
        value: decode_value(read.value),
    });
    let rd = data.rd.map(|write| FieldInlineRegisterWriteRow {
        register: write.register,
        pre_value: decode_value(write.pre_value),
        post_value: decode_value(write.post_value),
    });
    Ok(FieldInlineRegisterReadWriteRow {
        rs1,
        rs2,
        rd,
        rd_increment: FieldRdInc::extract(row, None, env)?.0,
    })
}

fn validate_trace_data(
    index: usize,
    row: &TraceRow,
    shape: FieldInlineOperandShape,
    data: FieldInlineTraceData,
) -> Result<(), WitnessError> {
    if data.op != Some(shape.op) {
        return Err(invalid_row(
            index,
            "field-inline trace payload op does not match instruction",
        ));
    }
    validate_read(
        index,
        "rs1",
        data.rs1,
        row.instruction.operands.rs1,
        shape.reads_fr_rs1,
    )?;
    validate_read(
        index,
        "rs2",
        data.rs2,
        row.instruction.operands.rs2,
        shape.reads_fr_rs2,
    )?;
    validate_write(
        index,
        data.rd,
        row.instruction.operands.rd,
        shape.writes_fr_rd,
    )?;

    if data.product.is_some() != shape.requires_product_payload() {
        return Err(invalid_row(
            index,
            "field-inline product payload presence does not match instruction",
        ));
    }
    if data.inv_product.is_some() != shape.requires_inverse_product_payload() {
        return Err(invalid_row(
            index,
            "field-inline inverse product payload presence does not match instruction",
        ));
    }
    validate_bridge(index, row, shape, data)
}

fn validate_read(
    index: usize,
    operand: &'static str,
    read: Option<FieldRegisterRead>,
    expected_register: Option<u8>,
    required: bool,
) -> Result<(), WitnessError> {
    match (required, read, expected_register) {
        (true, Some(read), Some(expected)) if read.register == expected => Ok(()),
        (true, Some(_), Some(_)) => Err(invalid_row(
            index,
            "field-inline read register does not match instruction operand",
        )),
        (true, _, _) => Err(invalid_row(
            index,
            "field-inline trace payload is missing read",
        )),
        (false, None, _) => Ok(()),
        (false, Some(_), _) => Err(invalid_row(index, operand)),
    }
}

fn validate_write(
    index: usize,
    write: Option<FieldRegisterWrite>,
    expected_register: Option<u8>,
    required: bool,
) -> Result<(), WitnessError> {
    match (required, write, expected_register) {
        (true, Some(write), Some(expected)) if write.register == expected => Ok(()),
        (true, Some(_), Some(_)) => Err(invalid_row(
            index,
            "field-inline write register does not match instruction operand",
        )),
        (true, _, _) => Err(invalid_row(
            index,
            "field-inline trace payload is missing write",
        )),
        (false, None, _) => Ok(()),
        (false, Some(_), _) => Err(invalid_row(
            index,
            "field-inline trace payload has an unexpected write",
        )),
    }
}

fn validate_bridge(
    index: usize,
    row: &TraceRow,
    shape: FieldInlineOperandShape,
    data: FieldInlineTraceData,
) -> Result<(), WitnessError> {
    match (shape.bridge_x_register_role, data.bridge) {
        (None, None) => Ok(()),
        (None, Some(_)) => Err(invalid_row(
            index,
            "pure field-inline instruction carries bridge payload",
        )),
        (
            Some(FieldInlineXRegisterRole::ReadRs1),
            Some(FieldInlineBridge::LoadFromX {
                x_register,
                x_value,
                field_value,
            }),
        ) => {
            if Some(x_register) != row.instruction.operands.rs1
                || Some(x_value) != row.registers.rs1.map(|read| read.value)
                || Some(field_value) != data.rd.map(|write| write.post_value)
            {
                return Err(invalid_row(
                    index,
                    "field-inline load bridge payload is inconsistent",
                ));
            }
            Ok(())
        }
        (
            Some(FieldInlineXRegisterRole::WriteRd),
            Some(FieldInlineBridge::StoreToX {
                field_register,
                field_value,
                x_register,
                x_value,
            }),
        ) => {
            if Some(field_register) != row.instruction.operands.rs1
                || Some(field_value) != data.rs1.map(|read| read.value)
                || Some(x_register) != row.instruction.operands.rd
                || Some(x_value) != row.registers.rd.map(|write| write.post_value)
            {
                return Err(invalid_row(
                    index,
                    "field-inline store bridge payload is inconsistent",
                ));
            }
            Ok(())
        }
        (Some(_), _) => Err(invalid_row(
            index,
            "field-inline bridge instruction is missing bridge payload",
        )),
    }
}

fn validate_field_register_state(rows: &[TraceRow]) -> Result<(), WitnessError> {
    let mut state = vec![FieldEncodedValue::zero(); field_register_count()];
    for (index, row) in rows.iter().enumerate() {
        let Some(data) = row.field_inline.as_deref() else {
            continue;
        };
        if let Some(read) = data.rs1 {
            validate_state_value(index, "rs1", &state, read.register, read.value)?;
        }
        if let Some(read) = data.rs2 {
            validate_state_value(index, "rs2", &state, read.register, read.value)?;
        }
        if let Some(write) = data.rd {
            validate_state_value(index, "rd", &state, write.register, write.pre_value)?;
            state[usize::from(write.register)] = write.post_value;
        }
    }
    Ok(())
}

fn validate_state_value(
    index: usize,
    operand: &'static str,
    state: &[FieldEncodedValue],
    register: u8,
    value: FieldEncodedValue,
) -> Result<(), WitnessError> {
    let expected = state
        .get(usize::from(register))
        .copied()
        .ok_or_else(|| invalid_row(index, "field register index is out of bounds"))?;
    if expected == value {
        Ok(())
    } else {
        Err(invalid_row(index, operand))
    }
}

fn field_register_count() -> usize {
    1usize << FIELD_REGISTERS_LOG_K
}

fn invalid_row(index: usize, reason: &'static str) -> WitnessError {
    WitnessError::InvalidWitnessData {
        label: FIELD_INLINE_LABEL,
        reason: format!("field-inline trace row {index}: {reason}"),
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use common::constants::RAM_START_ADDRESS;
    use jolt_claims::protocols::field_inline::FieldInlineOpFlag;
    use jolt_claims::protocols::jolt::{JoltCommittedPolynomial, JoltOneHotConfig};
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_program::{
        execution::{
            JoltProgram, OwnedTrace, RegisterRead, RegisterState, RegisterWrite, TraceOutput,
        },
        preprocess::{BytecodePreprocessing, JoltProgramPreprocessing, RAMPreprocessing},
    };
    use jolt_riscv::FieldInlineOp;
    use jolt_riscv::{
        JoltInstructionKind, JoltInstructionProfile, JoltInstructionRow, NormalizedOperands,
        RV64IMAC_JOLT, RV64IMAC_JOLT_FIELD_INLINE,
    };

    use super::*;
    use crate::backend::trace::{JoltVmWitnessConfig, JoltVmWitnessInputs};

    const ENTRY: u64 = RAM_START_ADDRESS;

    fn config(log_t: usize) -> JoltVmWitnessConfig {
        JoltVmWitnessConfig::new(
            log_t,
            64,
            JoltOneHotConfig {
                log_k_chunk: 4,
                lookups_ra_virtual_log_k_chunk: 16,
            },
        )
    }

    fn instruction(
        instruction_kind: JoltInstructionKind,
        offset: usize,
        rd: Option<u8>,
        rs1: Option<u8>,
        rs2: Option<u8>,
        imm: i128,
    ) -> JoltInstructionRow {
        JoltInstructionRow {
            instruction_kind,
            address: ENTRY as usize + offset * 4,
            operands: NormalizedOperands { rd, rs1, rs2, imm },
            virtual_sequence_remaining: None,
            is_first_in_sequence: false,
            is_compressed: false,
        }
    }

    fn preprocessing(
        bytecode: Vec<JoltInstructionRow>,
        profile: JoltInstructionProfile,
    ) -> JoltProgramPreprocessing {
        JoltProgramPreprocessing {
            bytecode: BytecodePreprocessing::preprocess(bytecode, ENTRY, profile).unwrap(),
            ram: RAMPreprocessing::default(),
            memory_layout: Default::default(),
            max_padded_trace_length: 8,
        }
    }

    fn program(bytecode: Vec<JoltInstructionRow>, profile: JoltInstructionProfile) -> JoltProgram {
        JoltProgram::from_parts_with_profile(
            Vec::new(),
            bytecode,
            Vec::new(),
            ENTRY + 4,
            ENTRY,
            profile,
        )
    }

    fn witness<'a>(
        program: &'a JoltProgram,
        preprocessing: &'a JoltProgramPreprocessing,
        rows: Vec<TraceRow>,
        log_t: usize,
    ) -> super::super::TraceBackend<'a, OwnedTrace> {
        super::super::TraceBackend::new(
            config(log_t),
            JoltVmWitnessInputs::new(
                program,
                preprocessing,
                TraceOutput::new(OwnedTrace::new(rows), Default::default(), None),
            ),
        )
    }

    fn enc(value: u64) -> FieldEncodedValue {
        FieldEncodedValue::from_u64(value)
    }

    fn fr(value: u64) -> Fr {
        Fr::from_u64(value)
    }

    fn row(instruction: JoltInstructionRow, data: FieldInlineTraceData) -> TraceRow {
        TraceRow {
            instruction,
            field_inline: Some(data.into()),
            ..TraceRow::default()
        }
    }

    fn load_imm(offset: usize, rd: u8, value: u64) -> (JoltInstructionRow, TraceRow) {
        let instruction = instruction(
            JoltInstructionKind::FIELD_LOAD_IMM,
            offset,
            Some(rd),
            None,
            None,
            i128::from(value),
        );
        let trace_row = row(
            instruction,
            FieldInlineTraceData {
                op: Some(FieldInlineOp::LoadImm),
                rd: Some(FieldRegisterWrite {
                    register: rd,
                    pre_value: enc(0),
                    post_value: enc(value),
                }),
                ..FieldInlineTraceData::default()
            },
        );
        (instruction, trace_row)
    }

    fn arithmetic_fixture() -> (Vec<JoltInstructionRow>, Vec<TraceRow>) {
        let (load_rs1, row0) = load_imm(0, 2, 5);
        let (load_rs2, row1) = load_imm(1, 3, 7);
        let mul = instruction(
            JoltInstructionKind::FIELD_MUL,
            2,
            Some(1),
            Some(2),
            Some(3),
            0,
        );
        let row2 = row(
            mul,
            FieldInlineTraceData {
                op: Some(FieldInlineOp::Mul),
                rs1: Some(FieldRegisterRead {
                    register: 2,
                    value: enc(5),
                }),
                rs2: Some(FieldRegisterRead {
                    register: 3,
                    value: enc(7),
                }),
                rd: Some(FieldRegisterWrite {
                    register: 1,
                    pre_value: enc(0),
                    post_value: enc(35),
                }),
                product: Some(enc(35)),
                ..FieldInlineTraceData::default()
            },
        );
        let store = instruction(
            JoltInstructionKind::FIELD_STORE_TO_X,
            3,
            Some(10),
            Some(1),
            None,
            0,
        );
        let row3 = TraceRow {
            instruction: store,
            registers: RegisterState {
                rd: Some(RegisterWrite {
                    register: 10,
                    pre_value: 0,
                    post_value: 35,
                }),
                ..RegisterState::default()
            },
            field_inline: Some(
                FieldInlineTraceData {
                    op: Some(FieldInlineOp::StoreToX),
                    rs1: Some(FieldRegisterRead {
                        register: 1,
                        value: enc(35),
                    }),
                    bridge: Some(FieldInlineBridge::StoreToX {
                        field_register: 1,
                        field_value: enc(35),
                        x_register: 10,
                        x_value: 35,
                    }),
                    ..FieldInlineTraceData::default()
                }
                .into(),
            ),
            ..TraceRow::default()
        };
        (
            vec![load_rs1, load_rs2, mul, store],
            vec![row0, row1, row2, row3],
        )
    }

    fn build_field_provider(
        bytecode: Vec<JoltInstructionRow>,
        rows: Vec<TraceRow>,
        log_t: usize,
    ) -> TraceBackedFieldInlineWitness<'static> {
        let program = Box::leak(Box::new(program(
            bytecode.clone(),
            RV64IMAC_JOLT_FIELD_INLINE,
        )));
        let preprocessing = Box::leak(Box::new(preprocessing(
            bytecode,
            RV64IMAC_JOLT_FIELD_INLINE,
        )));
        let witness = Box::leak(Box::new(witness(program, preprocessing, rows, log_t)));
        witness.field_inline_witness().unwrap()
    }

    fn owned_view(
        provider: &TraceBackedFieldInlineWitness<'_>,
        id: impl Into<FieldInlinePolynomialId>,
    ) -> Vec<Fr> {
        provider.oracle_table::<Fr>(id.into()).unwrap()
    }

    #[test]
    fn fr_off_provider_is_absent_without_field_data() {
        let bytecode = vec![instruction(
            JoltInstructionKind::ADDI,
            0,
            Some(1),
            Some(2),
            None,
            3,
        )];
        let program = program(bytecode.clone(), RV64IMAC_JOLT);
        let preprocessing = preprocessing(bytecode, RV64IMAC_JOLT);
        let row = TraceRow {
            instruction: instruction(JoltInstructionKind::ADDI, 0, Some(1), Some(2), None, 3),
            ..TraceRow::default()
        };
        let witness = witness(&program, &preprocessing, vec![row], 2);

        assert_eq!(
            witness.field_inline_witness().err(),
            Some(WitnessError::UnavailableView {
                label: FIELD_INLINE_LABEL,
            })
        );
    }

    #[test]
    fn fr_off_rejects_field_inline_trace_payload() {
        let bytecode = vec![instruction(
            JoltInstructionKind::ADDI,
            0,
            Some(1),
            Some(2),
            None,
            3,
        )];
        let program = program(bytecode.clone(), RV64IMAC_JOLT);
        let preprocessing = preprocessing(bytecode, RV64IMAC_JOLT);
        let field_row = row(
            instruction(
                JoltInstructionKind::FIELD_LOAD_IMM,
                0,
                Some(1),
                None,
                None,
                3,
            ),
            FieldInlineTraceData {
                op: Some(FieldInlineOp::LoadImm),
                rd: Some(FieldRegisterWrite {
                    register: 1,
                    pre_value: enc(0),
                    post_value: enc(3),
                }),
                ..FieldInlineTraceData::default()
            },
        );
        let witness = witness(&program, &preprocessing, vec![field_row], 2);

        assert!(matches!(
            witness.field_inline_witness(),
            Err(WitnessError::InvalidWitnessData { .. })
        ));
    }

    #[test]
    fn committed_order_and_requirements_are_blindfold_retained() {
        let (bytecode, rows) = arithmetic_fixture();
        let provider = build_field_provider(bytecode, rows, 3);

        let order = provider.committed_order();
        assert_eq!(order, vec![FieldInlineCommittedPolynomial::FieldRdInc]);

        let shape = provider
            .shape(FieldInlinePolynomialId::Committed(
                FieldInlineCommittedPolynomial::FieldRdInc,
            ))
            .unwrap();
        assert_eq!(shape.rows(), 8);
        assert_eq!(shape.encoding, PolynomialEncoding::Dense);
    }

    #[test]
    fn field_rd_inc_streams_field_deltas_and_padding() {
        let (bytecode, rows) = arithmetic_fixture();
        let provider = build_field_provider(bytecode, rows, 3);
        let mut stream = provider
            .committed_stream::<Fr>(FieldInlineCommittedPolynomial::FieldRdInc, 3)
            .unwrap();

        assert_eq!(
            stream.next_chunk(),
            Ok(Some(PolynomialChunk::Dense(vec![fr(5), fr(7), fr(35)])))
        );
        assert_eq!(
            stream.next_chunk(),
            Ok(Some(PolynomialChunk::Dense(vec![fr(0), fr(0), fr(0)])))
        );
        assert_eq!(
            stream.next_chunk(),
            Ok(Some(PolynomialChunk::Dense(vec![fr(0), fr(0)])))
        );
        assert_eq!(stream.next_chunk(), Ok(None));
    }

    #[test]
    fn trace_domain_virtual_views_decode_values_flags_and_products() {
        let (bytecode, rows) = arithmetic_fixture();
        let provider = build_field_provider(bytecode, rows, 3);

        let rd_values = owned_view(
            &provider,
            FieldInlinePolynomialId::Virtual(FieldInlineVirtualPolynomial::FieldRdValue),
        );
        assert_eq!(&rd_values[..4], &[fr(5), fr(7), fr(35), fr(0)]);

        let rs2_values = owned_view(
            &provider,
            FieldInlinePolynomialId::Virtual(FieldInlineVirtualPolynomial::FieldRs2Value),
        );
        assert_eq!(&rs2_values[..4], &[fr(0), fr(0), fr(7), fr(0)]);

        let products = owned_view(
            &provider,
            FieldInlinePolynomialId::Virtual(FieldInlineVirtualPolynomial::FieldProduct),
        );
        assert_eq!(&products[..4], &[fr(0), fr(0), fr(35), fr(0)]);

        let mul_flags = owned_view(
            &provider,
            FieldInlinePolynomialId::Virtual(FieldInlineVirtualPolynomial::FieldOpFlag(
                FieldInlineOpFlag::Mul,
            )),
        );
        assert_eq!(&mul_flags[..4], &[fr(0), fr(0), fr(1), fr(0)]);
    }

    #[test]
    fn register_domain_virtual_views_are_address_major() {
        let (bytecode, rows) = arithmetic_fixture();
        let provider = build_field_provider(bytecode, rows, 3);

        let registers_val = owned_view(
            &provider,
            FieldInlinePolynomialId::Virtual(FieldInlineVirtualPolynomial::FieldRegistersVal),
        );
        let index = |register: usize, cycle: usize| register * 8 + cycle;
        assert_eq!(registers_val[index(2, 0)], fr(0));
        assert_eq!(registers_val[index(2, 1)], fr(5));
        assert_eq!(registers_val[index(3, 2)], fr(7));
        assert_eq!(registers_val[index(1, 3)], fr(35));
        assert_eq!(registers_val[index(1, 7)], fr(35));

        let rs1_ra = owned_view(
            &provider,
            FieldInlinePolynomialId::Virtual(FieldInlineVirtualPolynomial::FieldRs1Ra),
        );
        assert_eq!(rs1_ra[index(2, 2)], fr(1));
        assert_eq!(rs1_ra[index(1, 3)], fr(1));
        assert_eq!(rs1_ra[index(3, 2)], fr(0));

        let rd_wa = owned_view(
            &provider,
            FieldInlinePolynomialId::Virtual(FieldInlineVirtualPolynomial::FieldRdWa),
        );
        assert_eq!(rd_wa[index(2, 0)], fr(1));
        assert_eq!(rd_wa[index(3, 1)], fr(1));
        assert_eq!(rd_wa[index(1, 2)], fr(1));
        assert_eq!(rd_wa[index(10, 3)], fr(0));
    }

    #[test]
    fn bridge_rows_keep_rv64_and_field_witnesses_separate() {
        let load = instruction(
            JoltInstructionKind::FIELD_LOAD_FROM_X,
            0,
            Some(1),
            Some(5),
            None,
            0,
        );
        let row0 = TraceRow {
            instruction: load,
            registers: RegisterState {
                rs1: Some(RegisterRead {
                    register: 5,
                    value: 11,
                }),
                ..RegisterState::default()
            },
            field_inline: Some(
                FieldInlineTraceData {
                    op: Some(FieldInlineOp::LoadFromX),
                    rd: Some(FieldRegisterWrite {
                        register: 1,
                        pre_value: enc(0),
                        post_value: enc(11),
                    }),
                    bridge: Some(FieldInlineBridge::LoadFromX {
                        x_register: 5,
                        x_value: 11,
                        field_value: enc(11),
                    }),
                    ..FieldInlineTraceData::default()
                }
                .into(),
            ),
            ..TraceRow::default()
        };
        let store = instruction(
            JoltInstructionKind::FIELD_STORE_TO_X,
            1,
            Some(6),
            Some(1),
            None,
            0,
        );
        let row1 = TraceRow {
            instruction: store,
            registers: RegisterState {
                rd: Some(RegisterWrite {
                    register: 6,
                    pre_value: 0,
                    post_value: 11,
                }),
                ..RegisterState::default()
            },
            field_inline: Some(
                FieldInlineTraceData {
                    op: Some(FieldInlineOp::StoreToX),
                    rs1: Some(FieldRegisterRead {
                        register: 1,
                        value: enc(11),
                    }),
                    bridge: Some(FieldInlineBridge::StoreToX {
                        field_register: 1,
                        field_value: enc(11),
                        x_register: 6,
                        x_value: 11,
                    }),
                    ..FieldInlineTraceData::default()
                }
                .into(),
            ),
            ..TraceRow::default()
        };

        let bytecode = vec![load, store];
        let program = program(bytecode.clone(), RV64IMAC_JOLT_FIELD_INLINE);
        let preprocessing = preprocessing(bytecode, RV64IMAC_JOLT_FIELD_INLINE);
        let witness = witness(&program, &preprocessing, vec![row0, row1], 2);
        let provider = witness.field_inline_witness().unwrap();

        let mut ordinary_stream = witness
            .committed_stream(JoltCommittedPolynomial::RdInc, 4)
            .unwrap();
        assert_eq!(
            ordinary_stream.next_chunk(),
            Ok(Some(PolynomialChunk::<Fr>::I128(vec![0, 11, 0, 0])))
        );

        let mut field_stream = provider
            .committed_stream::<Fr>(FieldInlineCommittedPolynomial::FieldRdInc, 4)
            .unwrap();
        assert_eq!(
            field_stream.next_chunk(),
            Ok(Some(PolynomialChunk::Dense(vec![
                fr(11),
                fr(0),
                fr(0),
                fr(0)
            ])))
        );
    }

    #[test]
    fn validation_rejects_missing_payload_and_inconsistent_state() {
        let (bytecode, mut rows) = arithmetic_fixture();
        rows[2].field_inline = None;
        let program = program(bytecode.clone(), RV64IMAC_JOLT_FIELD_INLINE);
        let preprocessing = preprocessing(bytecode.clone(), RV64IMAC_JOLT_FIELD_INLINE);
        let missing_payload_witness = witness(&program, &preprocessing, rows, 3);
        assert!(matches!(
            missing_payload_witness.field_inline_witness(),
            Err(WitnessError::InvalidWitnessData { .. })
        ));

        let (_, mut bad_rows) = arithmetic_fixture();
        let Some(data) = bad_rows[2].field_inline.as_mut() else {
            return;
        };
        std::sync::Arc::make_mut(data).rs1 = Some(FieldRegisterRead {
            register: 2,
            value: enc(6),
        });
        let inconsistent_state_witness = witness(&program, &preprocessing, bad_rows, 3);
        assert!(matches!(
            inconsistent_state_witness.field_inline_witness(),
            Err(WitnessError::InvalidWitnessData { .. })
        ));
    }

    #[test]
    fn field_inline_virtual_oracles_describe_dense_views() {
        let (bytecode, rows) = arithmetic_fixture();
        let provider = build_field_provider(bytecode, rows, 3);

        for id in [
            FieldInlinePolynomialId::Virtual(FieldInlineVirtualPolynomial::FieldRegistersVal),
            FieldInlinePolynomialId::Virtual(FieldInlineVirtualPolynomial::FieldRdWa),
            FieldInlinePolynomialId::Virtual(FieldInlineVirtualPolynomial::FieldOpFlag(
                FieldInlineOpFlag::LoadImm,
            )),
        ] {
            let shape = provider.shape(id).unwrap();
            assert_eq!(shape.encoding, PolynomialEncoding::Dense);
        }
    }
}
