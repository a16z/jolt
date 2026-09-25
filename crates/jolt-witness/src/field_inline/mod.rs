use jolt_claims::protocols::field_inline::{
    geometry::{
        bytecode::FIELD_INLINE_BYTECODE_STAGE1_FLAGS,
        spartan::FIELD_INLINE_SPARTAN_OUTER_R1CS_INPUTS,
    },
    FieldInlineCommittedPolynomial, FieldInlinePolynomialId, FieldInlineVirtualPolynomial,
    FIELD_REGISTERS_LOG_K,
};
use jolt_field::JoltField;
use jolt_program::{
    execution::{JoltProgram, TraceRow, TraceSource},
    field_inline::{
        FieldEncodedValue, FieldInlineBridge, FieldInlineTraceData, FieldRegisterRead,
        FieldRegisterWrite,
    },
    preprocess::JoltProgramPreprocessing,
};
use jolt_riscv::{
    field_inline_operand_shape, FieldInlineOp, FieldInlineOperandShape, JoltTraceRow,
};
use rayon::prelude::*;
use std::sync::Arc;

use self::witnesses::{
    decode_value, FieldInvProduct, FieldOpFlag, FieldProduct, FieldRdInc, FieldRdValue,
    FieldRs1Value, FieldRs2Value, FieldValue,
};
use crate::backend::trace::{checked_pow2, TraceBackend};
use crate::consumer::BUNDLE_PASS_CHUNK;
use crate::witnesses::{Extract, ExtractIndexed, WitnessEnv};
use crate::{
    stream_witnesses, BundleSource, PolynomialEncoding, RandomAccessRows, Shape, StreamConsumer,
    WitnessBundle, WitnessError, WitnessRow,
};

pub mod witnesses;

/// Error label for the field-inline witness backend.
pub const FIELD_INLINE_LABEL: &str = "jolt_vm.field_inline";

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FieldInlineRegisterReadRow<F: JoltField> {
    pub register: u8,
    pub value: F,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FieldInlineRegisterWriteRow<F: JoltField> {
    pub register: u8,
    pub pre_value: F,
    pub post_value: F,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FieldInlineRegisterReadWriteRow<F: JoltField> {
    pub rs1: Option<FieldInlineRegisterReadRow<F>>,
    pub rs2: Option<FieldInlineRegisterReadRow<F>>,
    pub rd: Option<FieldInlineRegisterWriteRow<F>>,
    pub rd_increment: F,
}

impl<F: JoltField> WitnessBundle for FieldInlineRegisterReadWriteRow<F> {
    type PolynomialId = FieldInlinePolynomialId;

    fn from_row(
        row: WitnessRow<'_>,
        _next: Option<WitnessRow<'_>>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        let Some(data) = row.field_inline()? else {
            return Ok(Self::default());
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
        Ok(Self {
            rs1,
            rs2,
            rd,
            rd_increment: rd.map_or_else(F::zero, |write| write.post_value - write.pre_value),
        })
    }

    fn annotated_ids() -> Vec<Self::PolynomialId> {
        vec![
            FieldInlineVirtualPolynomial::FieldRs1Value.into(),
            FieldInlineVirtualPolynomial::FieldRs2Value.into(),
            FieldInlineVirtualPolynomial::FieldRdValue.into(),
            FieldInlineCommittedPolynomial::FieldRdInc.into(),
        ]
    }
}

pub trait FieldInlineRegisterReadWriteRows<F: JoltField> {
    fn field_inline_register_read_write_rows(
        &self,
    ) -> Result<Vec<FieldInlineRegisterReadWriteRow<F>>, WitnessError>;
}

/// One active field-inline cycle's composed spartan-outer column values — the 15
/// appended R1CS columns in `FIELD_INLINE_SPARTAN_OUTER_R1CS_INPUTS` order: the five
/// value columns, then the ten op-flag columns in
/// [`FieldInlineOpFlag`](jolt_claims::protocols::field_inline::FieldInlineOpFlag)
/// declaration order.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FieldInlineSpartanRow<F> {
    pub rs1_value: F,
    pub rs2_value: F,
    pub rd_value: F,
    pub product: F,
    pub inv_product: F,
    pub flags: [F; 10],
}

impl<F: Copy> FieldInlineSpartanRow<F> {
    /// The row's 15 column values in the composed opening-column order.
    pub fn columns(&self) -> [F; 15] {
        [
            self.rs1_value,
            self.rs2_value,
            self.rd_value,
            self.product,
            self.inv_product,
            self.flags[0],
            self.flags[1],
            self.flags[2],
            self.flags[3],
            self.flags[4],
            self.flags[5],
            self.flags[6],
            self.flags[7],
            self.flags[8],
            self.flags[9],
        ]
    }
}

impl<F: JoltField> FieldInlineSpartanRow<F> {
    fn from_data(data: &FieldInlineTraceData) -> Self {
        let rs1_value = data
            .rs1
            .map_or_else(F::zero, |read| decode_value(read.value));
        let rs2_value = data
            .rs2
            .map_or_else(F::zero, |read| decode_value(read.value));
        let rd_value = data
            .rd
            .map_or_else(F::zero, |write| decode_value(write.post_value));
        Self {
            rs1_value,
            rs2_value,
            rd_value,
            product: rs1_value * rs2_value,
            inv_product: rs1_value * rd_value,
            flags: FIELD_INLINE_BYTECODE_STAGE1_FLAGS
                .map(|flag| F::from_bool(data.op == Some(witnesses::op(flag)))),
        }
    }
}

impl<F: JoltField> WitnessBundle for FieldInlineSpartanRow<F> {
    type PolynomialId = FieldInlinePolynomialId;

    fn from_row(
        row: WitnessRow<'_>,
        _next: Option<WitnessRow<'_>>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        Ok(row
            .field_inline()?
            .map_or_else(Self::default, Self::from_data))
    }

    fn annotated_ids() -> Vec<Self::PolynomialId> {
        FIELD_INLINE_SPARTAN_OUTER_R1CS_INPUTS
            .into_iter()
            .map(FieldInlinePolynomialId::Virtual)
            .collect()
    }
}

struct ActiveFieldInlineSpartanRow<F>(Option<(usize, FieldInlineSpartanRow<F>)>);

impl<F: JoltField> WitnessBundle for ActiveFieldInlineSpartanRow<F> {
    type PolynomialId = FieldInlinePolynomialId;

    fn from_row(
        row: WitnessRow<'_>,
        _next: Option<WitnessRow<'_>>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        Ok(Self(row.field_inline()?.map(|data| {
            (row.cycle, FieldInlineSpartanRow::from_data(data))
        })))
    }

    fn annotated_ids() -> Vec<Self::PolynomialId> {
        FieldInlineSpartanRow::<F>::annotated_ids()
    }
}

#[derive(Default)]
struct CollectFieldInlineSpartanRows<F> {
    rows: Vec<(usize, FieldInlineSpartanRow<F>)>,
}

impl<F: JoltField> StreamConsumer for CollectFieldInlineSpartanRows<F> {
    type Witness = ActiveFieldInlineSpartanRow<F>;

    fn consume(&mut self, chunk: &[Self::Witness]) {
        self.rows.extend(chunk.iter().filter_map(|row| row.0));
    }
}

/// The object-safe field-inline witness surface a prover reads off the witness plane:
/// shapes and dense tables over the field-inline id vocabulary, the committed-order
/// tail, and (via the supertrait) the register replay rows the read-write kernels fold.
pub trait FieldInlineWitnessOracle<F: JoltField>:
    FieldInlineRegisterReadWriteRows<F> + Send + Sync
{
    fn shape(&self, id: FieldInlinePolynomialId) -> Result<Shape, WitnessError>;

    /// Materializes the oracle's dense field-element evaluations, row-major
    /// over the domain declared by [`shape`](Self::shape).
    fn oracle_table(&self, id: FieldInlinePolynomialId) -> Result<Vec<F>, WitnessError>;

    /// The proof-payload order of the field-inline committed polynomials.
    fn committed_order(&self) -> Vec<FieldInlineCommittedPolynomial>;

    /// The composed spartan-outer field-inline column values, sparse over the cycle
    /// domain: `(cycle, row)` pairs sorted strictly increasing by cycle, covering at
    /// least every cycle where any of the 15 field-inline columns is non-zero (extra
    /// all-zero rows are harmless — the columns' values are what the composed kernels
    /// fold). The default derives the rows from the dense `oracle_table`s so fixture
    /// oracles stay valid; the trace-backed oracle overrides it with a direct sparse
    /// walk that never materializes the 15 dense tables.
    fn field_inline_spartan_rows(
        &self,
    ) -> Result<Vec<(usize, FieldInlineSpartanRow<F>)>, WitnessError> {
        let tables = FIELD_INLINE_SPARTAN_OUTER_R1CS_INPUTS
            .iter()
            .map(|&id| self.oracle_table(FieldInlinePolynomialId::Virtual(id)))
            .collect::<Result<Vec<_>, _>>()?;
        let cycles = tables.first().map_or(0, Vec::len);
        if tables.iter().any(|table| table.len() != cycles) {
            return Err(WitnessError::InvalidDimensions {
                label: FIELD_INLINE_LABEL,
                reason: "field-inline spartan column tables disagree on the cycle domain"
                    .to_owned(),
            });
        }
        let mut rows = Vec::new();
        for cycle in 0..cycles {
            let column = |index: usize| tables[index][cycle];
            let values: Vec<F> = (0..tables.len()).map(column).collect();
            if values.iter().all(F::is_zero) {
                continue;
            }
            rows.push((
                cycle,
                FieldInlineSpartanRow {
                    rs1_value: values[0],
                    rs2_value: values[1],
                    rd_value: values[2],
                    product: values[3],
                    inv_product: values[4],
                    flags: [
                        values[5], values[6], values[7], values[8], values[9], values[10],
                        values[11], values[12], values[13], values[14],
                    ],
                },
            ));
        }
        Ok(rows)
    }
}

impl<F: JoltField> FieldInlineWitnessOracle<F> for TraceBackedFieldInlineWitness {
    fn shape(&self, id: FieldInlinePolynomialId) -> Result<Shape, WitnessError> {
        TraceBackedFieldInlineWitness::shape(self, id)
    }

    fn oracle_table(&self, id: FieldInlinePolynomialId) -> Result<Vec<F>, WitnessError> {
        TraceBackedFieldInlineWitness::oracle_table::<F>(self, id)
    }

    fn committed_order(&self) -> Vec<FieldInlineCommittedPolynomial> {
        TraceBackedFieldInlineWitness::committed_order(self)
    }

    /// Only active rows survive the bounded bundle stream; no dense Spartan
    /// row vector or per-column tables are allocated.
    fn field_inline_spartan_rows(
        &self,
    ) -> Result<Vec<(usize, FieldInlineSpartanRow<F>)>, WitnessError> {
        let mut consumers = (CollectFieldInlineSpartanRows::default(),);
        stream_witnesses(
            &self.rows_source,
            0..self.trace_rows.len(),
            BUNDLE_PASS_CHUNK,
            &mut consumers,
        )?;
        Ok(consumers.0.rows)
    }
}

pub struct TraceBackedFieldInlineWitness {
    log_t: usize,
    program: Arc<JoltProgram>,
    preprocessing: Arc<JoltProgramPreprocessing>,
    trace_rows: Arc<Vec<TraceRow>>,
    rows_source: RandomAccessRows,
}

impl TraceBackedFieldInlineWitness {
    pub(crate) fn build(
        log_t: usize,
        program: &Arc<JoltProgram>,
        preprocessing: &Arc<JoltProgramPreprocessing>,
        trace_rows: &Arc<Vec<TraceRow>>,
        compact_rows: &Arc<Vec<JoltTraceRow>>,
    ) -> Result<Self, WitnessError> {
        let rows = checked_pow2(log_t)?;
        if trace_rows.len() > rows {
            return Err(WitnessError::InvalidWitnessData {
                label: FIELD_INLINE_LABEL,
                reason: "trace length exceeds configured field-inline witness domain".to_owned(),
            });
        }
        let mut witness = Self {
            log_t,
            program: Arc::clone(program),
            preprocessing: Arc::clone(preprocessing),
            trace_rows: Arc::clone(trace_rows),
            rows_source: RandomAccessRows::new(
                Arc::clone(compact_rows),
                rows,
                Arc::clone(preprocessing),
            )?,
        };
        witness.validate_inputs()?;
        witness.rows_source = witness
            .rows_source
            .with_field_inline(Arc::clone(trace_rows));
        Ok(witness)
    }

    pub(crate) fn rows_source(&self) -> &RandomAccessRows {
        &self.rows_source
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
                    || field_inline_operand_shape(row.instruction_kind()).is_some()
                {
                    return Err(invalid_row(
                        index,
                        "field-inline trace data exists for a program without field-inline",
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
                reason: "field-inline program is missing bytecode metadata".to_owned(),
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
        let shape = field_inline_operand_shape(row.instruction_kind());
        let metadata = self
            .preprocessing
            .bytecode
            .field_inline
            .as_ref()
            .ok_or_else(|| WitnessError::InvalidWitnessData {
                label: FIELD_INLINE_LABEL,
                reason: "field-inline program is missing bytecode metadata".to_owned(),
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
                    .get_pc(&row.instruction())
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
    fn materialize_cycle<
        F: JoltField,
        W: for<'a> Extract<WitnessRow<'a>> + FieldValue<F> + Send,
    >(
        &self,
    ) -> Result<Vec<F>, WitnessError> {
        self.walk_cycles(|row, env| W::extract(&row, None, env).map(FieldValue::value))
    }

    /// [`Self::materialize_cycle`] for indexed witness families.
    fn materialize_cycle_indexed<
        F: JoltField,
        W: for<'a> ExtractIndexed<I, WitnessRow<'a>> + FieldValue<F>,
        I: Copy + Sync,
    >(
        &self,
        index: I,
    ) -> Result<Vec<F>, WitnessError> {
        self.walk_cycles(|row, env| {
            W::extract_indexed(index, &row, None, env).map(FieldValue::value)
        })
    }

    fn walk_cycles<F: JoltField>(
        &self,
        value: impl Fn(WitnessRow<'_>, &WitnessEnv<'_>) -> Result<F, WitnessError> + Sync,
    ) -> Result<Vec<F>, WitnessError> {
        let env = WitnessEnv::new(&self.preprocessing);
        (0..self.rows_source.cycles())
            .into_par_iter()
            .map(|cycle| {
                let row =
                    self.rows_source
                        .row(cycle)
                        .ok_or_else(|| WitnessError::InvalidDimensions {
                            label: FIELD_INLINE_LABEL,
                            reason: format!("missing field witness cycle {cycle}"),
                        })?;
                value(row, &env)
            })
            .collect()
    }

    fn materialize_register_virtual<F: JoltField>(
        &self,
        id: FieldInlineVirtualPolynomial,
    ) -> Result<Vec<F>, WitnessError> {
        let register_count = field_register_count();
        let cycles = self.rows_source.cycles();
        let mut values = vec![F::from_u64(0); cycles * register_count];

        if id == FieldInlineVirtualPolynomial::FieldRegistersVal {
            let trace_rows = self.trace_rows.as_slice();
            let trace_len = trace_rows.len();
            values
                .par_chunks_mut(cycles)
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
                values[usize::from(register) * cycles + cycle] = F::from_u64(1);
            }
        }

        Ok(values)
    }
}

impl TraceBackedFieldInlineWitness {
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

    pub fn oracle_table<F: JoltField>(
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

    pub fn committed_order(&self) -> Vec<FieldInlineCommittedPolynomial> {
        vec![FieldInlineCommittedPolynomial::FieldRdInc]
    }
}

impl<F: JoltField> FieldInlineRegisterReadWriteRows<F> for TraceBackedFieldInlineWitness {
    fn field_inline_register_read_write_rows(
        &self,
    ) -> Result<Vec<FieldInlineRegisterReadWriteRow<F>>, WitnessError> {
        self.rows_source.bundles()
    }
}

impl<T: TraceSource> TraceBackend<T> {
    pub fn field_inline_witness(&self) -> Result<TraceBackedFieldInlineWitness, WitnessError> {
        TraceBackedFieldInlineWitness::build(
            self.config.log_t,
            &self.program,
            &self.preprocessing,
            &self.raw_trace_rows,
            &self.trace.trace,
        )
    }

    /// Eagerly materializes and validates the field-inline witness view and
    /// stores it so this backend can serve the field-inline rows directly.
    pub fn with_field_inline(mut self) -> Result<Self, WitnessError> {
        self.field_inline = Some(self.field_inline_witness()?);
        Ok(self)
    }
}

impl<T: TraceSource> TraceBackend<T> {
    fn field_inline_view(&self) -> Result<&TraceBackedFieldInlineWitness, WitnessError> {
        self.field_inline
            .as_ref()
            .ok_or(WitnessError::UnavailableView {
                label: FIELD_INLINE_LABEL,
            })
    }
}

impl<F: JoltField, T: TraceSource> FieldInlineRegisterReadWriteRows<F> for TraceBackend<T> {
    fn field_inline_register_read_write_rows(
        &self,
    ) -> Result<Vec<FieldInlineRegisterReadWriteRow<F>>, WitnessError> {
        self.field_inline_view()?
            .field_inline_register_read_write_rows()
    }
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
    let operands = row.instruction().operands;
    // The memory-sourced load keeps its field destination in the `rs2`
    // slot; both ingress operations read the old destination as field `rs1`.
    let field_rd = if shape.field_rd_in_rs2_slot {
        operands.rs2
    } else {
        operands.rd
    };
    let field_rs1 = if shape.field_rs1_is_field_rd {
        field_rd
    } else {
        operands.rs1
    };
    validate_read(index, "rs1", data.rs1, field_rs1, shape.reads_field_rs1)?;
    validate_read(index, "rs2", data.rs2, operands.rs2, shape.reads_field_rs2)?;
    validate_write(index, data.rd, field_rd, shape.writes_field_rd)?;

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
    match (shape.op, data.bridge) {
        (
            FieldInlineOp::Add
            | FieldInlineOp::Sub
            | FieldInlineOp::Mul
            | FieldInlineOp::Inv
            | FieldInlineOp::AssertEq
            | FieldInlineOp::AssertZero
            | FieldInlineOp::LoadImm,
            bridge,
        ) => {
            if bridge.is_none() {
                Ok(())
            } else {
                Err(invalid_row(
                    index,
                    "pure field-inline instruction carries bridge payload",
                ))
            }
        }
        (
            FieldInlineOp::LoadAccumulateFromRegister,
            Some(FieldInlineBridge::LoadAccumulateFromRegister {
                x_register,
                x_value,
                field_value,
            }),
        ) => {
            if Some(x_register) != row.instruction().operands.rs1
                || Some(x_value) != row.rs1_read().map(|read| read.value)
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
            FieldInlineOp::AdviceLimb,
            Some(FieldInlineBridge::AdviceLimb {
                field_register,
                field_value,
                x_register,
                x_value,
            }),
        ) => {
            let operands = row.instruction().operands;
            if Some(field_register) != operands.rs1
                || Some(field_value) != data.rs1.map(|read| read.value)
                || Some(x_register) != operands.rd
                || Some(x_value) != row.rd_write().map(|write| write.post_value)
            {
                return Err(invalid_row(
                    index,
                    "field-inline advice bridge payload is inconsistent",
                ));
            }
            Ok(())
        }
        (
            FieldInlineOp::LoadAccumulateFromMemory,
            Some(FieldInlineBridge::LoadAccumulateFromMemory {
                x_base,
                x_register,
                word,
                field_value,
            }),
        ) => {
            let operands = row.instruction().operands;
            if Some(x_base) != operands.rs1
                || Some(x_register) != operands.rd
                || Some(word) != row.rd_write().map(|write| write.post_value)
                || Some(field_value) != data.rd.map(|write| write.post_value)
            {
                return Err(invalid_row(
                    index,
                    "field-inline load word bridge payload is inconsistent",
                ));
            }
            Ok(())
        }
        (
            FieldInlineOp::LoadAccumulateFromRegister
            | FieldInlineOp::LoadAccumulateFromMemory
            | FieldInlineOp::AdviceLimb,
            _,
        ) => Err(invalid_row(
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
    use crate::{CollectBundles, RowSource};
    use common::constants::RAM_START_ADDRESS;
    use jolt_claims::protocols::field_inline::FieldInlineOpFlag;
    use jolt_claims::protocols::jolt::{
        JoltCommittedPolynomial, JoltOneHotConfig, JoltPolynomialId,
    };
    use jolt_field::{Fr, Ring};
    use jolt_program::{
        execution::{
            JoltProgram, OwnedTrace, RamAccess, RamRead, RegisterRead, RegisterState,
            RegisterWrite, TraceOutput,
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
    ) -> Arc<JoltProgramPreprocessing> {
        Arc::new(JoltProgramPreprocessing {
            bytecode: BytecodePreprocessing::preprocess(bytecode, ENTRY, profile).unwrap(),
            ram: RAMPreprocessing::default(),
            memory_layout: Default::default(),
            max_padded_trace_length: 8,
        })
    }

    fn program(
        bytecode: Vec<JoltInstructionRow>,
        profile: JoltInstructionProfile,
    ) -> Arc<JoltProgram> {
        Arc::new(JoltProgram::from_parts_with_profile(
            Vec::new(),
            bytecode,
            Vec::new(),
            ENTRY + 4,
            ENTRY,
            profile,
        ))
    }

    fn witness(
        program: &Arc<JoltProgram>,
        preprocessing: &Arc<JoltProgramPreprocessing>,
        rows: Vec<TraceRow>,
        log_t: usize,
    ) -> super::super::TraceBackend<OwnedTrace> {
        super::super::TraceBackend::new(
            config(log_t),
            JoltVmWitnessInputs::new(
                program,
                preprocessing,
                TraceOutput::new(OwnedTrace::new(rows), Default::default(), None, None),
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
        row_with_registers(instruction, RegisterState::default(), data)
    }

    fn row_with_registers(
        instruction: JoltInstructionRow,
        registers: RegisterState,
        data: FieldInlineTraceData,
    ) -> TraceRow {
        let mut row = TraceRow::new(instruction, registers, RamAccess::NoOp).unwrap();
        row.field_inline = Some(data.into());
        row
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
                ..FieldInlineTraceData::default()
            },
        );
        let advice = instruction(
            JoltInstructionKind::FIELD_ADVICE_LIMB,
            3,
            Some(10),
            Some(1),
            Some(0),
            0,
        );
        let row3 = row_with_registers(
            advice,
            RegisterState {
                rd: Some(RegisterWrite {
                    register: 10,
                    pre_value: 0,
                    post_value: 35,
                }),
                ..RegisterState::default()
            },
            FieldInlineTraceData {
                op: Some(FieldInlineOp::AdviceLimb),
                rs1: Some(FieldRegisterRead {
                    register: 1,
                    value: enc(35),
                }),
                rd: Some(FieldRegisterWrite {
                    register: 0,
                    pre_value: enc(0),
                    post_value: enc(0),
                }),
                bridge: Some(FieldInlineBridge::AdviceLimb {
                    field_register: 1,
                    field_value: enc(35),
                    x_register: 10,
                    x_value: 35,
                }),
                ..FieldInlineTraceData::default()
            },
        );
        (
            vec![load_rs1, load_rs2, mul, advice],
            vec![row0, row1, row2, row3],
        )
    }

    fn build_field_provider(
        bytecode: Vec<JoltInstructionRow>,
        rows: Vec<TraceRow>,
        log_t: usize,
    ) -> TraceBackedFieldInlineWitness {
        let program = program(bytecode.clone(), RV64IMAC_JOLT_FIELD_INLINE);
        let preprocessing = preprocessing(bytecode, RV64IMAC_JOLT_FIELD_INLINE);
        let witness = witness(&program, &preprocessing, rows, log_t);
        witness.field_inline_witness().unwrap()
    }

    fn owned_view(
        provider: &TraceBackedFieldInlineWitness,
        id: impl Into<FieldInlinePolynomialId>,
    ) -> Vec<Fr> {
        provider.oracle_table::<Fr>(id.into()).unwrap()
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct FieldLookaheadRow {
        cycle: usize,
        next_rd: Option<Fr>,
    }

    impl WitnessBundle for FieldLookaheadRow {
        type PolynomialId = FieldInlinePolynomialId;

        fn from_row(
            row: WitnessRow<'_>,
            next: Option<WitnessRow<'_>>,
            env: &WitnessEnv<'_>,
        ) -> Result<Self, WitnessError> {
            Ok(Self {
                cycle: row.cycle,
                next_rd: next
                    .map(|next| FieldRdValue::<Fr>::extract(&next, None, env).map(|value| value.0))
                    .transpose()?,
            })
        }

        fn annotated_ids() -> Vec<Self::PolynomialId> {
            Vec::new()
        }
    }

    #[test]
    fn field_bundles_preserve_sparse_cycles_lookahead_and_padding() {
        let (bytecode, mut rows) = arithmetic_fixture();
        rows.insert(0, TraceRow::default());
        rows.insert(2, TraceRow::default());
        rows.push(TraceRow::default());
        let program = program(bytecode.clone(), RV64IMAC_JOLT_FIELD_INLINE);
        let preprocessing = preprocessing(bytecode, RV64IMAC_JOLT_FIELD_INLINE);
        let backend = witness(&program, &preprocessing, rows, 3);
        assert_eq!(backend.trace.trace.len(), 6);
        assert_eq!(backend.raw_trace_rows.len(), 7);
        assert!(matches!(
            backend.bundles::<FieldInlineSpartanRow<Fr>>(),
            Err(WitnessError::UnavailableView {
                label: FIELD_INLINE_LABEL
            })
        ));
        assert!(matches!(
            backend.bundles::<FieldInlineRegisterReadWriteRow<Fr>>(),
            Err(WitnessError::UnavailableView {
                label: FIELD_INLINE_LABEL
            })
        ));
        let access = backend.random_access().unwrap();
        assert!(matches!(
            FieldRs1Value::<Fr>::extract(
                &access.row(4).unwrap(),
                None,
                &WitnessEnv::new(&preprocessing),
            ),
            Err(WitnessError::UnavailableView {
                label: FIELD_INLINE_LABEL
            })
        ));

        let backend = backend.with_field_inline().unwrap();
        let provider = backend.field_inline_view().unwrap();
        let registers: Vec<FieldInlineRegisterReadWriteRow<Fr>> =
            backend.field_inline_register_read_write_rows().unwrap();
        assert_eq!(registers.len(), 8);
        assert_eq!(registers[4].rs1.unwrap().value, fr(5));
        assert_eq!(registers[4].rs2.unwrap().value, fr(7));
        assert_eq!(registers[4].rd_increment, fr(35));
        for cycle in [0, 2, 6, 7] {
            assert_eq!(registers[cycle], FieldInlineRegisterReadWriteRow::default());
        }
        let sparse: Vec<(usize, FieldInlineSpartanRow<Fr>)> =
            provider.field_inline_spartan_rows().unwrap();
        assert_eq!(
            sparse.iter().map(|(cycle, _)| *cycle).collect::<Vec<_>>(),
            [1, 3, 4, 5]
        );
        assert_eq!(sparse[2].1.product, fr(35));
        assert_eq!(sparse[2].1.inv_product, fr(175));
        assert_eq!(sparse[2].1.flags[2], fr(1));
        let dense: Vec<FieldInlineSpartanRow<Fr>> = backend.bundles().unwrap();
        assert_eq!(dense.len(), 8);
        assert_eq!(dense[4], sparse[2].1);
        for cycle in [0, 2, 6, 7] {
            assert_eq!(dense[cycle], FieldInlineSpartanRow::default());
        }

        let mut consumers = (CollectBundles::<FieldLookaheadRow>::default(),);
        stream_witnesses(provider.rows_source(), 2..8, 2, &mut consumers).unwrap();
        assert_eq!(
            consumers.0.into_rows(),
            vec![
                FieldLookaheadRow {
                    cycle: 2,
                    next_rd: Some(fr(7))
                },
                FieldLookaheadRow {
                    cycle: 3,
                    next_rd: Some(fr(35))
                },
                FieldLookaheadRow {
                    cycle: 4,
                    next_rd: Some(fr(0))
                },
                FieldLookaheadRow {
                    cycle: 5,
                    next_rd: Some(fr(0))
                },
                FieldLookaheadRow {
                    cycle: 6,
                    next_rd: Some(fr(0))
                },
                FieldLookaheadRow {
                    cycle: 7,
                    next_rd: None
                },
            ],
        );
    }

    #[test]
    fn field_inline_disabled_provider_is_absent_without_field_data() {
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
        let row = TraceRow::from_instruction(instruction(
            JoltInstructionKind::ADDI,
            0,
            Some(1),
            Some(2),
            None,
            3,
        ))
        .unwrap();
        let witness = witness(&program, &preprocessing, vec![row], 2);

        assert_eq!(
            witness.field_inline_witness().err(),
            Some(WitnessError::UnavailableView {
                label: FIELD_INLINE_LABEL,
            })
        );
    }

    #[test]
    fn field_inline_disabled_rejects_field_inline_trace_payload() {
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
    fn field_rd_inc_materializes_field_deltas_and_padding() {
        let (bytecode, rows) = arithmetic_fixture();
        let provider = build_field_provider(bytecode, rows, 3);
        assert_eq!(
            owned_view(
                &provider,
                FieldInlinePolynomialId::Committed(FieldInlineCommittedPolynomial::FieldRdInc)
            ),
            vec![fr(5), fr(7), fr(35), fr(0), fr(0), fr(0), fr(0), fr(0)]
        );
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
            JoltInstructionKind::FIELD_LOAD_ACCUMULATE_FROM_REGISTER,
            0,
            Some(1),
            Some(5),
            None,
            0,
        );
        let row0 = row_with_registers(
            load,
            RegisterState {
                rs1: Some(RegisterRead {
                    register: 5,
                    value: 11,
                }),
                ..RegisterState::default()
            },
            FieldInlineTraceData {
                op: Some(FieldInlineOp::LoadAccumulateFromRegister),
                rs1: Some(FieldRegisterRead {
                    register: 1,
                    value: enc(0),
                }),
                rd: Some(FieldRegisterWrite {
                    register: 1,
                    pre_value: enc(0),
                    post_value: enc(11),
                }),
                bridge: Some(FieldInlineBridge::LoadAccumulateFromRegister {
                    x_register: 5,
                    x_value: 11,
                    field_value: enc(11),
                }),
                ..FieldInlineTraceData::default()
            },
        );
        let advice = instruction(
            JoltInstructionKind::FIELD_ADVICE_LIMB,
            1,
            Some(6),
            Some(1),
            Some(0),
            0,
        );
        let row1 = row_with_registers(
            advice,
            RegisterState {
                rd: Some(RegisterWrite {
                    register: 6,
                    pre_value: 0,
                    post_value: 11,
                }),
                ..RegisterState::default()
            },
            FieldInlineTraceData {
                op: Some(FieldInlineOp::AdviceLimb),
                rs1: Some(FieldRegisterRead {
                    register: 1,
                    value: enc(11),
                }),
                rd: Some(FieldRegisterWrite {
                    register: 0,
                    pre_value: enc(0),
                    post_value: enc(0),
                }),
                bridge: Some(FieldInlineBridge::AdviceLimb {
                    field_register: 1,
                    field_value: enc(11),
                    x_register: 6,
                    x_value: 11,
                }),
                ..FieldInlineTraceData::default()
            },
        );

        let bytecode = vec![load, advice];
        let program = program(bytecode.clone(), RV64IMAC_JOLT_FIELD_INLINE);
        let preprocessing = preprocessing(bytecode, RV64IMAC_JOLT_FIELD_INLINE);
        let witness = witness(&program, &preprocessing, vec![row0, row1], 2);
        let provider = witness.field_inline_witness().unwrap();

        let ordinary = crate::JoltWitnessOracle::<Fr>::oracle_table(
            &witness,
            JoltPolynomialId::Committed(JoltCommittedPolynomial::RdInc),
        )
        .unwrap();
        assert_eq!(ordinary, vec![fr(0), fr(11), fr(0), fr(0)]);

        assert_eq!(
            owned_view(
                &provider,
                FieldInlinePolynomialId::Committed(FieldInlineCommittedPolynomial::FieldRdInc)
            ),
            vec![fr(11), fr(0), fr(0), fr(0)]
        );
    }

    #[test]
    fn accumulating_loads_read_and_bind_the_nonzero_destination() {
        for (kind, op) in [
            (
                JoltInstructionKind::FIELD_LOAD_ACCUMULATE_FROM_REGISTER,
                FieldInlineOp::LoadAccumulateFromRegister,
            ),
            (
                JoltInstructionKind::FIELD_LOAD_ACCUMULATE_FROM_MEMORY,
                FieldInlineOp::LoadAccumulateFromMemory,
            ),
        ] {
            let memory_load = op == FieldInlineOp::LoadAccumulateFromMemory;
            let (seed, seed_row) = load_imm(0, 1, 3);
            let load = instruction(
                kind,
                1,
                Some(if memory_load { 6 } else { 1 }),
                Some(5),
                memory_load.then_some(1),
                0,
            );
            let mut accumulated = enc(11);
            accumulated.bytes_le[8] = 3;
            let bridge = if memory_load {
                FieldInlineBridge::LoadAccumulateFromMemory {
                    x_base: 5,
                    x_register: 6,
                    word: 11,
                    field_value: accumulated,
                }
            } else {
                FieldInlineBridge::LoadAccumulateFromRegister {
                    x_register: 5,
                    x_value: 11,
                    field_value: accumulated,
                }
            };
            let mut load_row = TraceRow::new(
                load,
                RegisterState {
                    rs1: Some(RegisterRead {
                        register: 5,
                        value: if memory_load { ENTRY } else { 11 },
                    }),
                    rd: memory_load.then_some(RegisterWrite {
                        register: 6,
                        pre_value: 0,
                        post_value: 11,
                    }),
                    ..RegisterState::default()
                },
                if memory_load {
                    RamAccess::Read(RamRead {
                        address: ENTRY,
                        value: 11,
                    })
                } else {
                    RamAccess::NoOp
                },
            )
            .unwrap();
            load_row.field_inline = Some(
                FieldInlineTraceData {
                    op: Some(op),
                    rs1: Some(FieldRegisterRead {
                        register: 1,
                        value: enc(3),
                    }),
                    rd: Some(FieldRegisterWrite {
                        register: 1,
                        pre_value: enc(3),
                        post_value: accumulated,
                    }),
                    bridge: Some(bridge),
                    ..FieldInlineTraceData::default()
                }
                .into(),
            );
            let bytecode = vec![seed, load];
            let program = program(bytecode.clone(), RV64IMAC_JOLT_FIELD_INLINE);
            let preprocessing = preprocessing(bytecode, RV64IMAC_JOLT_FIELD_INLINE);
            let rows = vec![seed_row, load_row];
            let provider = witness(&program, &preprocessing, rows.clone(), 2)
                .field_inline_witness()
                .unwrap();
            let registers: Vec<FieldInlineRegisterReadWriteRow<Fr>> =
                provider.field_inline_register_read_write_rows().unwrap();
            assert_eq!(registers[1].rs1.unwrap().value, fr(3));
            assert_eq!(registers[1].rd_increment, Fr::from_u128((3u128 << 64) + 8));

            for read in [
                None,
                Some(FieldRegisterRead {
                    register: 1,
                    value: enc(0),
                }),
            ] {
                let mut tampered = rows.clone();
                Arc::make_mut(tampered[1].field_inline.as_mut().unwrap()).rs1 = read;
                assert!(matches!(
                    witness(&program, &preprocessing, tampered, 2).field_inline_witness(),
                    Err(WitnessError::InvalidWitnessData { .. })
                ));
            }
        }
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
        Arc::make_mut(data).rs1 = Some(FieldRegisterRead {
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
