use jolt_claims::protocols::jolt::{
    formulas::{
        claim_reductions::hamming_weight::{self, HammingWeightClaimReductionDimensions},
        dimensions::TracePolynomialOrder,
    },
    AdviceClaimReductionLayout, JoltAdviceKind, JoltCommittedPolynomial,
};
use jolt_field::Field;
use jolt_poly::Polynomial;
#[cfg(feature = "field-inline")]
use jolt_witness::protocols::jolt_vm::field_inline::{
    FieldInlineRegisterReadWriteRow, FieldInlineRegisterReadWriteRows,
};
use jolt_witness::protocols::jolt_vm::{
    JoltVmNamespace, JoltVmRegisterReadWriteRow, JoltVmRegisterReadWriteRows,
    JoltVmSpartanOuterRow, JoltVmSpartanOuterRows, JoltVmStage2Rows, JoltVmStage2TraceRow,
    JoltVmStage3ShiftRow, JoltVmStage3ShiftRows, JoltVmStage5InstructionReadRafRows,
    JoltVmStage6Row, Stage5InstructionReadRafRow,
};
use jolt_witness::WitnessError;
use jolt_witness::{ViewRequirement, WitnessNamespace};

use crate::{BackendKernelMetadata, BackendRelationId, BackendValueSlot};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SumcheckSlot(pub u32);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckInstanceRequest<N: WitnessNamespace> {
    pub slot: SumcheckSlot,
    pub relation: BackendRelationId,
    pub optimization_ids: &'static [&'static str],
    pub witness_views: Vec<ViewRequirement<N>>,
    pub rounds: usize,
    pub degree: usize,
    pub input_claim: BackendValueSlot,
    pub output_claim: BackendValueSlot,
    #[cfg(feature = "zk")]
    pub committed_rounds: bool,
}

impl<N: WitnessNamespace> SumcheckInstanceRequest<N> {
    pub fn new(
        slot: SumcheckSlot,
        relation: BackendRelationId,
        witness_views: Vec<ViewRequirement<N>>,
        rounds: usize,
        degree: usize,
        input_claim: BackendValueSlot,
        output_claim: BackendValueSlot,
    ) -> Self {
        Self {
            slot,
            relation,
            optimization_ids: &[],
            witness_views,
            rounds,
            degree,
            input_claim,
            output_claim,
            #[cfg(feature = "zk")]
            committed_rounds: false,
        }
    }

    pub const fn with_optimization_ids(
        mut self,
        optimization_ids: &'static [&'static str],
    ) -> Self {
        self.optimization_ids = optimization_ids;
        self
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckRequest<N: WitnessNamespace> {
    pub label: &'static str,
    pub instances: Vec<SumcheckInstanceRequest<N>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckRegularBatchLinearTerm<F: Field> {
    pub polynomial: usize,
    pub coefficient: F,
}

impl<F: Field> SumcheckRegularBatchLinearTerm<F> {
    pub const fn new(polynomial: usize, coefficient: F) -> Self {
        Self {
            polynomial,
            coefficient,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckRegularBatchLinearFactor<F: Field> {
    pub constant: F,
    pub terms: Vec<SumcheckRegularBatchLinearTerm<F>>,
}

impl<F: Field> SumcheckRegularBatchLinearFactor<F> {
    pub const fn new(constant: F, terms: Vec<SumcheckRegularBatchLinearTerm<F>>) -> Self {
        Self { constant, terms }
    }

    pub fn from_terms(terms: Vec<SumcheckRegularBatchLinearTerm<F>>) -> Self {
        Self {
            constant: F::zero(),
            terms,
        }
    }
}

/// One additive product term of a regular-batch instance.
///
/// Evaluates to `scale * Π factors`, where each factor is a linear combination
/// of the instance polynomials. An instance's per-pair message is the sum of its
/// product terms, so a statement of the form `Σ_k scale_k · Π_j factor_{k,j}`
/// (such as the Stage 3 Spartan shift and instruction-input reductions) maps onto
/// one instance carrying several product terms.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckRegularBatchProduct<F: Field> {
    pub scale: F,
    pub factors: Vec<SumcheckRegularBatchLinearFactor<F>>,
}

impl<F: Field> SumcheckRegularBatchProduct<F> {
    pub const fn new(scale: F, factors: Vec<SumcheckRegularBatchLinearFactor<F>>) -> Self {
        Self { scale, factors }
    }

    pub fn degree(&self) -> usize {
        self.factors.len()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckRegularBatchInstance<F: Field> {
    pub label: &'static str,
    pub input_claim: F,
    pub polynomials: Vec<Polynomial<F>>,
    pub products: Vec<SumcheckRegularBatchProduct<F>>,
}

impl<F: Field> SumcheckRegularBatchInstance<F> {
    /// Single-product instance: message is `scale * Π factors`.
    pub fn new(
        label: &'static str,
        input_claim: F,
        scale: F,
        polynomials: Vec<Polynomial<F>>,
        factors: Vec<SumcheckRegularBatchLinearFactor<F>>,
    ) -> Self {
        Self {
            label,
            input_claim,
            polynomials,
            products: vec![SumcheckRegularBatchProduct::new(scale, factors)],
        }
    }

    /// Sum-of-products instance: message is `Σ_k scale_k · Π_j factor_{k,j}`.
    pub fn new_products(
        label: &'static str,
        input_claim: F,
        polynomials: Vec<Polynomial<F>>,
        products: Vec<SumcheckRegularBatchProduct<F>>,
    ) -> Self {
        Self {
            label,
            input_claim,
            polynomials,
            products,
        }
    }

    pub fn num_rounds(&self) -> usize {
        self.polynomials
            .first()
            .map_or(0, |polynomial| jolt_poly::Polynomial::num_vars(polynomial))
    }

    pub fn degree(&self) -> usize {
        self.products
            .iter()
            .map(SumcheckRegularBatchProduct::degree)
            .max()
            .unwrap_or(0)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckRegularBatchState<F: Field> {
    pub label: &'static str,
    pub instances: Vec<SumcheckRegularBatchInstance<F>>,
    validated: bool,
}

impl<F: Field> SumcheckRegularBatchState<F> {
    pub const fn new(label: &'static str, instances: Vec<SumcheckRegularBatchInstance<F>>) -> Self {
        Self {
            label,
            instances,
            validated: false,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.instances.is_empty()
    }

    pub(crate) const fn is_validated(&self) -> bool {
        self.validated
    }

    pub(crate) fn mark_validated(&mut self) {
        self.validated = true;
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SumcheckStage3ShiftRow {
    pub unexpanded_pc: u64,
    pub pc: u64,
    pub is_virtual: bool,
    pub is_first_in_sequence: bool,
    pub is_noop: bool,
}

impl SumcheckStage3ShiftRow {
    pub const fn new(
        unexpanded_pc: u64,
        pc: u64,
        is_virtual: bool,
        is_first_in_sequence: bool,
        is_noop: bool,
    ) -> Self {
        Self {
            unexpanded_pc,
            pc,
            is_virtual,
            is_first_in_sequence,
            is_noop,
        }
    }
}

pub fn stage3_shift_rows<W>(
    witness: &W,
    log_t: usize,
) -> Result<Vec<SumcheckStage3ShiftRow>, WitnessError>
where
    W: JoltVmStage3ShiftRows,
{
    Ok(witness
        .stage3_shift_rows(log_t)?
        .into_iter()
        .map(stage3_shift_row)
        .collect())
}

pub fn stage3_shift_row(row: JoltVmStage3ShiftRow) -> SumcheckStage3ShiftRow {
    SumcheckStage3ShiftRow::new(
        row.unexpanded_pc,
        row.pc,
        row.is_virtual,
        row.is_first_in_sequence,
        row.is_noop,
    )
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckStage3ShiftStateRequest<F: Field> {
    pub label: &'static str,
    pub log_t: usize,
    pub outer_point: Vec<F>,
    pub product_point: Vec<F>,
    pub gamma: F,
    pub rows: Vec<SumcheckStage3ShiftRow>,
}

impl<F: Field> SumcheckStage3ShiftStateRequest<F> {
    pub const fn new(
        label: &'static str,
        log_t: usize,
        outer_point: Vec<F>,
        product_point: Vec<F>,
        gamma: F,
        rows: Vec<SumcheckStage3ShiftRow>,
    ) -> Self {
        Self {
            label,
            log_t,
            outer_point,
            product_point,
            gamma,
            rows,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SumcheckRamReadWriteRow {
    pub remapped_ram_address: Option<usize>,
    pub ram_read_value: u64,
    pub ram_write_value: u64,
    pub ram_increment: i128,
}

pub fn ram_read_write_rows<W>(witness: &W) -> Result<Vec<SumcheckRamReadWriteRow>, WitnessError>
where
    W: JoltVmStage2Rows,
{
    Ok(ram_read_write_rows_from_trace(&witness.stage2_rows()?))
}

pub fn ram_read_write_rows_from_trace(
    rows: &[JoltVmStage2TraceRow],
) -> Vec<SumcheckRamReadWriteRow> {
    rows.iter().copied().map(ram_read_write_row).collect()
}

pub fn ram_read_write_row(row: JoltVmStage2TraceRow) -> SumcheckRamReadWriteRow {
    SumcheckRamReadWriteRow {
        remapped_ram_address: row.remapped_ram_address,
        ram_read_value: row.ram_read_value,
        ram_write_value: row.ram_write_value,
        ram_increment: row.ram_increment,
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SumcheckRegisterRead {
    pub register: u8,
    pub value: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SumcheckRegisterWrite {
    pub register: u8,
    pub pre_value: u64,
    pub post_value: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SumcheckRegistersReadWriteRow {
    pub rs1: Option<SumcheckRegisterRead>,
    pub rs2: Option<SumcheckRegisterRead>,
    pub rd: Option<SumcheckRegisterWrite>,
    pub rd_increment: i128,
}

pub fn register_read_write_rows<W>(
    witness: &W,
) -> Result<Vec<SumcheckRegistersReadWriteRow>, WitnessError>
where
    W: JoltVmRegisterReadWriteRows,
{
    Ok(witness
        .register_read_write_rows()?
        .into_iter()
        .map(register_read_write_row)
        .collect())
}

pub fn register_read_write_row(row: JoltVmRegisterReadWriteRow) -> SumcheckRegistersReadWriteRow {
    SumcheckRegistersReadWriteRow {
        rs1: row.rs1.map(|read| SumcheckRegisterRead {
            register: read.register,
            value: read.value,
        }),
        rs2: row.rs2.map(|read| SumcheckRegisterRead {
            register: read.register,
            value: read.value,
        }),
        rd: row.rd.map(|write| SumcheckRegisterWrite {
            register: write.register,
            pre_value: write.pre_value,
            post_value: write.post_value,
        }),
        rd_increment: row.rd_increment,
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SumcheckFieldRegisterRead<F: Field> {
    pub register: u8,
    pub value: F,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SumcheckFieldRegisterWrite<F: Field> {
    pub register: u8,
    pub pre_value: F,
    pub post_value: F,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SumcheckFieldRegistersReadWriteRow<F: Field> {
    pub rs1: Option<SumcheckFieldRegisterRead<F>>,
    pub rs2: Option<SumcheckFieldRegisterRead<F>>,
    pub rd: Option<SumcheckFieldRegisterWrite<F>>,
    pub rd_increment: F,
}

#[cfg(feature = "field-inline")]
pub fn field_register_read_write_rows<F, W>(
    witness: &W,
) -> Result<Vec<SumcheckFieldRegistersReadWriteRow<F>>, WitnessError>
where
    F: Field,
    W: FieldInlineRegisterReadWriteRows<F>,
{
    Ok(witness
        .field_inline_register_read_write_rows()?
        .into_iter()
        .map(field_register_read_write_row)
        .collect())
}

#[cfg(feature = "field-inline")]
pub fn field_register_read_write_row<F: Field>(
    row: FieldInlineRegisterReadWriteRow<F>,
) -> SumcheckFieldRegistersReadWriteRow<F> {
    SumcheckFieldRegistersReadWriteRow {
        rs1: row.rs1.map(|read| SumcheckFieldRegisterRead {
            register: read.register,
            value: read.value,
        }),
        rs2: row.rs2.map(|read| SumcheckFieldRegisterRead {
            register: read.register,
            value: read.value,
        }),
        rd: row.rd.map(|write| SumcheckFieldRegisterWrite {
            register: write.register,
            pre_value: write.pre_value,
            post_value: write.post_value,
        }),
        rd_increment: row.rd_increment,
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckRegistersReadWriteStateRequest<F: Field> {
    pub label: &'static str,
    pub kernel: BackendKernelMetadata,
    pub rows: Vec<SumcheckRegistersReadWriteRow>,
    pub r_cycle: Vec<F>,
    pub gamma: F,
    pub input_claim: F,
    pub log_t: usize,
    pub log_k: usize,
    pub phase1_num_rounds: usize,
    pub phase2_num_rounds: usize,
}

impl<F: Field> SumcheckRegistersReadWriteStateRequest<F> {
    #[expect(clippy::too_many_arguments)]
    pub fn new(
        label: &'static str,
        rows: Vec<SumcheckRegistersReadWriteRow>,
        r_cycle: Vec<F>,
        gamma: F,
        input_claim: F,
        log_t: usize,
        log_k: usize,
        phase1_num_rounds: usize,
        phase2_num_rounds: usize,
    ) -> Self {
        Self {
            label,
            kernel: BackendKernelMetadata::empty(),
            rows,
            r_cycle,
            gamma,
            input_claim,
            log_t,
            log_k,
            phase1_num_rounds,
            phase2_num_rounds,
        }
    }

    pub const fn with_kernel_metadata(mut self, kernel: BackendKernelMetadata) -> Self {
        self.kernel = kernel;
        self
    }

    pub const fn with_relation(mut self, relation: BackendRelationId) -> Self {
        self.kernel = self.kernel.with_relation(relation);
        self
    }

    pub const fn with_optimization_ids(
        mut self,
        optimization_ids: &'static [&'static str],
    ) -> Self {
        self.kernel = self.kernel.with_optimization_ids(optimization_ids);
        self
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckFieldRegistersReadWriteStateRequest<F: Field> {
    pub label: &'static str,
    pub kernel: BackendKernelMetadata,
    pub rows: Vec<SumcheckFieldRegistersReadWriteRow<F>>,
    pub r_cycle: Vec<F>,
    pub gamma: F,
    pub input_claim: F,
    pub log_t: usize,
    pub log_k: usize,
    pub phase1_num_rounds: usize,
    pub phase2_num_rounds: usize,
}

impl<F: Field> SumcheckFieldRegistersReadWriteStateRequest<F> {
    #[expect(clippy::too_many_arguments)]
    pub fn new(
        label: &'static str,
        rows: Vec<SumcheckFieldRegistersReadWriteRow<F>>,
        r_cycle: Vec<F>,
        gamma: F,
        input_claim: F,
        log_t: usize,
        log_k: usize,
        phase1_num_rounds: usize,
        phase2_num_rounds: usize,
    ) -> Self {
        Self {
            label,
            kernel: BackendKernelMetadata::empty(),
            rows,
            r_cycle,
            gamma,
            input_claim,
            log_t,
            log_k,
            phase1_num_rounds,
            phase2_num_rounds,
        }
    }

    pub const fn with_kernel_metadata(mut self, kernel: BackendKernelMetadata) -> Self {
        self.kernel = kernel;
        self
    }

    pub const fn with_relation(mut self, relation: BackendRelationId) -> Self {
        self.kernel = self.kernel.with_relation(relation);
        self
    }

    pub const fn with_optimization_ids(
        mut self,
        optimization_ids: &'static [&'static str],
    ) -> Self {
        self.kernel = self.kernel.with_optimization_ids(optimization_ids);
        self
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckFieldRegistersValEvaluationStateRequest<F: Field> {
    pub label: &'static str,
    pub kernel: BackendKernelMetadata,
    pub rows: Vec<SumcheckFieldRegistersReadWriteRow<F>>,
    pub r_address: Vec<F>,
    pub r_cycle: Vec<F>,
    pub input_claim: F,
    pub log_t: usize,
    pub log_k: usize,
}

impl<F: Field> SumcheckFieldRegistersValEvaluationStateRequest<F> {
    pub fn new(
        label: &'static str,
        rows: Vec<SumcheckFieldRegistersReadWriteRow<F>>,
        r_address: Vec<F>,
        r_cycle: Vec<F>,
        input_claim: F,
        log_t: usize,
        log_k: usize,
    ) -> Self {
        Self {
            label,
            kernel: BackendKernelMetadata::empty(),
            rows,
            r_address,
            r_cycle,
            input_claim,
            log_t,
            log_k,
        }
    }

    pub const fn with_kernel_metadata(mut self, kernel: BackendKernelMetadata) -> Self {
        self.kernel = kernel;
        self
    }

    pub const fn with_relation(mut self, relation: BackendRelationId) -> Self {
        self.kernel = self.kernel.with_relation(relation);
        self
    }

    pub const fn with_optimization_ids(
        mut self,
        optimization_ids: &'static [&'static str],
    ) -> Self {
        self.kernel = self.kernel.with_optimization_ids(optimization_ids);
        self
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckRamReadWriteStateRequest<F: Field> {
    pub label: &'static str,
    pub kernel: BackendKernelMetadata,
    pub rows: Vec<SumcheckRamReadWriteRow>,
    pub initial_ram_state: Vec<u64>,
    pub r_cycle: Vec<F>,
    pub gamma: F,
    pub input_claim: F,
    pub log_t: usize,
    pub log_k: usize,
    pub phase1_num_rounds: usize,
    pub phase2_num_rounds: usize,
}

impl<F: Field> SumcheckRamReadWriteStateRequest<F> {
    #[expect(clippy::too_many_arguments)]
    pub fn new(
        label: &'static str,
        rows: Vec<SumcheckRamReadWriteRow>,
        initial_ram_state: Vec<u64>,
        r_cycle: Vec<F>,
        gamma: F,
        input_claim: F,
        log_t: usize,
        log_k: usize,
        phase1_num_rounds: usize,
        phase2_num_rounds: usize,
    ) -> Self {
        Self {
            label,
            kernel: BackendKernelMetadata::empty(),
            rows,
            initial_ram_state,
            r_cycle,
            gamma,
            input_claim,
            log_t,
            log_k,
            phase1_num_rounds,
            phase2_num_rounds,
        }
    }

    pub const fn with_kernel_metadata(mut self, kernel: BackendKernelMetadata) -> Self {
        self.kernel = kernel;
        self
    }

    pub const fn with_relation(mut self, relation: BackendRelationId) -> Self {
        self.kernel = self.kernel.with_relation(relation);
        self
    }

    pub const fn with_optimization_ids(
        mut self,
        optimization_ids: &'static [&'static str],
    ) -> Self {
        self.kernel = self.kernel.with_optimization_ids(optimization_ids);
        self
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckRamValCheckStateRequest<F: Field> {
    pub label: &'static str,
    pub kernel: BackendKernelMetadata,
    pub rows: Vec<SumcheckRamReadWriteRow>,
    pub r_address: Vec<F>,
    pub r_cycle: Vec<F>,
    pub gamma: F,
    pub input_claim: F,
    pub log_t: usize,
}

impl<F: Field> SumcheckRamValCheckStateRequest<F> {
    pub fn new(
        label: &'static str,
        rows: Vec<SumcheckRamReadWriteRow>,
        r_address: Vec<F>,
        r_cycle: Vec<F>,
        gamma: F,
        input_claim: F,
        log_t: usize,
    ) -> Self {
        Self {
            label,
            kernel: BackendKernelMetadata::empty(),
            rows,
            r_address,
            r_cycle,
            gamma,
            input_claim,
            log_t,
        }
    }

    pub const fn with_kernel_metadata(mut self, kernel: BackendKernelMetadata) -> Self {
        self.kernel = kernel;
        self
    }

    pub const fn with_relation(mut self, relation: BackendRelationId) -> Self {
        self.kernel = self.kernel.with_relation(relation);
        self
    }

    pub const fn with_optimization_ids(
        mut self,
        optimization_ids: &'static [&'static str],
    ) -> Self {
        self.kernel = self.kernel.with_optimization_ids(optimization_ids);
        self
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckInstructionReadRafRow {
    pub lookup_index: u128,
    pub table_index: Option<usize>,
    pub interleaved_operands: bool,
}

impl SumcheckInstructionReadRafRow {
    pub const fn new(
        lookup_index: u128,
        table_index: Option<usize>,
        interleaved_operands: bool,
    ) -> Self {
        Self {
            lookup_index,
            table_index,
            interleaved_operands,
        }
    }
}

pub fn instruction_read_raf_rows<W>(
    witness: &W,
    log_t: usize,
) -> Result<Vec<SumcheckInstructionReadRafRow>, WitnessError>
where
    W: JoltVmStage5InstructionReadRafRows,
{
    Ok(witness
        .stage5_instruction_read_raf_rows(log_t)?
        .into_iter()
        .map(instruction_read_raf_row)
        .collect())
}

pub fn instruction_read_raf_row(row: Stage5InstructionReadRafRow) -> SumcheckInstructionReadRafRow {
    SumcheckInstructionReadRafRow::new(row.lookup_index, row.table_index, row.interleaved_operands)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckInstructionReadRafStateRequest<F: Field> {
    pub label: &'static str,
    pub kernel: BackendKernelMetadata,
    pub rows: Vec<SumcheckInstructionReadRafRow>,
    pub fixed_cycle_point: Vec<F>,
    pub gamma: F,
    pub input_claim: F,
    pub log_t: usize,
    pub address_bits: usize,
    pub ra_virtual_chunk_bits: usize,
    pub phases: usize,
}

impl<F: Field> SumcheckInstructionReadRafStateRequest<F> {
    #[expect(clippy::too_many_arguments)]
    pub fn new(
        label: &'static str,
        rows: Vec<SumcheckInstructionReadRafRow>,
        fixed_cycle_point: Vec<F>,
        gamma: F,
        input_claim: F,
        log_t: usize,
        address_bits: usize,
        ra_virtual_chunk_bits: usize,
        phases: usize,
    ) -> Self {
        Self {
            label,
            kernel: BackendKernelMetadata::empty(),
            rows,
            fixed_cycle_point,
            gamma,
            input_claim,
            log_t,
            address_bits,
            ra_virtual_chunk_bits,
            phases,
        }
    }

    pub const fn with_kernel_metadata(mut self, kernel: BackendKernelMetadata) -> Self {
        self.kernel = kernel;
        self
    }

    pub const fn with_relation(mut self, relation: BackendRelationId) -> Self {
        self.kernel = self.kernel.with_relation(relation);
        self
    }

    pub const fn with_optimization_ids(
        mut self,
        optimization_ids: &'static [&'static str],
    ) -> Self {
        self.kernel = self.kernel.with_optimization_ids(optimization_ids);
        self
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SumcheckStage6RaRow {
    pub instruction_lookup_index: u128,
    pub bytecode_index: usize,
    pub ram_address: Option<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SumcheckStage6IncRow {
    pub ram_increment: i128,
    pub rd_increment: i128,
}

pub fn stage6_ra_rows(rows: &[JoltVmStage6Row]) -> Vec<SumcheckStage6RaRow> {
    rows.iter()
        .map(|row| SumcheckStage6RaRow {
            instruction_lookup_index: row.instruction_lookup_index,
            bytecode_index: row.bytecode_index,
            ram_address: row.remapped_ram_address,
        })
        .collect()
}

pub fn stage6_inc_rows(rows: &[JoltVmStage6Row]) -> Vec<SumcheckStage6IncRow> {
    rows.iter()
        .map(|row| SumcheckStage6IncRow {
            ram_increment: row.ram_increment,
            rd_increment: row.rd_increment,
        })
        .collect()
}

pub fn stage6_hamming_weight(rows: &[JoltVmStage6Row]) -> Vec<bool> {
    rows.iter().map(|row| row.ram_access_nonzero).collect()
}

pub fn stage6_bytecode_pc_indices(rows: &[JoltVmStage6Row]) -> Vec<usize> {
    rows.iter().map(|row| row.bytecode_index).collect()
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckBytecodeReadRafExtraStageValues<F: Field> {
    pub stage: usize,
    pub bytecode_stage_values: Vec<F>,
    pub r_cycle: Vec<F>,
}

impl<F: Field> SumcheckBytecodeReadRafExtraStageValues<F> {
    pub fn new(stage: usize, bytecode_stage_values: Vec<F>, r_cycle: Vec<F>) -> Self {
        Self {
            stage,
            bytecode_stage_values,
            r_cycle,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckBytecodeReadRafStateRequest<F: Field> {
    pub label: &'static str,
    pub kernel: BackendKernelMetadata,
    pub bytecode_stage_values: Vec<[F; 5]>,
    pub extra_stage_values: Vec<SumcheckBytecodeReadRafExtraStageValues<F>>,
    pub pc_indices: Vec<usize>,
    pub r_cycles: [Vec<F>; 5],
    pub gamma_powers: Vec<F>,
    pub entry_bytecode_index: usize,
    pub input_claim: F,
    pub log_t: usize,
    pub log_k: usize,
    pub chunk_bits: usize,
}

impl<F: Field> SumcheckBytecodeReadRafStateRequest<F> {
    #[expect(clippy::too_many_arguments)]
    pub fn new(
        label: &'static str,
        bytecode_stage_values: Vec<[F; 5]>,
        pc_indices: Vec<usize>,
        r_cycles: [Vec<F>; 5],
        gamma_powers: Vec<F>,
        entry_bytecode_index: usize,
        input_claim: F,
        log_t: usize,
        log_k: usize,
        chunk_bits: usize,
    ) -> Self {
        Self {
            label,
            kernel: BackendKernelMetadata::empty(),
            bytecode_stage_values,
            extra_stage_values: Vec::new(),
            pc_indices,
            r_cycles,
            gamma_powers,
            entry_bytecode_index,
            input_claim,
            log_t,
            log_k,
            chunk_bits,
        }
    }

    pub const fn with_optimization_ids(
        mut self,
        optimization_ids: &'static [&'static str],
    ) -> Self {
        self.kernel = self.kernel.with_optimization_ids(optimization_ids);
        self
    }

    pub fn with_extra_stage_values(
        mut self,
        extra_stage_values: Vec<SumcheckBytecodeReadRafExtraStageValues<F>>,
    ) -> Self {
        self.extra_stage_values = extra_stage_values;
        self
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckBooleanityStateRequest<F: Field> {
    pub label: &'static str,
    pub kernel: BackendKernelMetadata,
    pub rows: Vec<SumcheckStage6RaRow>,
    pub r_address: Vec<F>,
    pub r_cycle: Vec<F>,
    pub gamma: F,
    pub input_claim: F,
    pub log_t: usize,
    pub chunk_bits: usize,
    pub instruction_chunks: usize,
    pub bytecode_chunks: usize,
    pub ram_chunks: usize,
}

impl<F: Field> SumcheckBooleanityStateRequest<F> {
    #[expect(clippy::too_many_arguments)]
    pub fn new(
        label: &'static str,
        rows: Vec<SumcheckStage6RaRow>,
        r_address: Vec<F>,
        r_cycle: Vec<F>,
        gamma: F,
        input_claim: F,
        log_t: usize,
        chunk_bits: usize,
        instruction_chunks: usize,
        bytecode_chunks: usize,
        ram_chunks: usize,
    ) -> Self {
        Self {
            label,
            kernel: BackendKernelMetadata::empty(),
            rows,
            r_address,
            r_cycle,
            gamma,
            input_claim,
            log_t,
            chunk_bits,
            instruction_chunks,
            bytecode_chunks,
            ram_chunks,
        }
    }

    pub const fn with_optimization_ids(
        mut self,
        optimization_ids: &'static [&'static str],
    ) -> Self {
        self.kernel = self.kernel.with_optimization_ids(optimization_ids);
        self
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckRamHammingBooleanityStateRequest<F: Field> {
    pub label: &'static str,
    pub kernel: BackendKernelMetadata,
    pub hamming_weight: Vec<bool>,
    pub r_cycle: Vec<F>,
    pub input_claim: F,
    pub log_t: usize,
}

impl<F: Field> SumcheckRamHammingBooleanityStateRequest<F> {
    pub fn new(
        label: &'static str,
        hamming_weight: Vec<bool>,
        r_cycle: Vec<F>,
        input_claim: F,
        log_t: usize,
    ) -> Self {
        Self {
            label,
            kernel: BackendKernelMetadata::empty(),
            hamming_weight,
            r_cycle,
            input_claim,
            log_t,
        }
    }

    pub const fn with_optimization_ids(
        mut self,
        optimization_ids: &'static [&'static str],
    ) -> Self {
        self.kernel = self.kernel.with_optimization_ids(optimization_ids);
        self
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckRamRaVirtualizationStateRequest<F: Field> {
    pub label: &'static str,
    pub kernel: BackendKernelMetadata,
    pub rows: Vec<SumcheckStage6RaRow>,
    pub r_address_chunks: Vec<Vec<F>>,
    pub r_cycle: Vec<F>,
    pub input_claim: F,
    pub log_t: usize,
    pub chunk_bits: usize,
}

impl<F: Field> SumcheckRamRaVirtualizationStateRequest<F> {
    pub fn new(
        label: &'static str,
        rows: Vec<SumcheckStage6RaRow>,
        r_address_chunks: Vec<Vec<F>>,
        r_cycle: Vec<F>,
        input_claim: F,
        log_t: usize,
        chunk_bits: usize,
    ) -> Self {
        Self {
            label,
            kernel: BackendKernelMetadata::empty(),
            rows,
            r_address_chunks,
            r_cycle,
            input_claim,
            log_t,
            chunk_bits,
        }
    }

    pub const fn with_optimization_ids(
        mut self,
        optimization_ids: &'static [&'static str],
    ) -> Self {
        self.kernel = self.kernel.with_optimization_ids(optimization_ids);
        self
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckInstructionRaVirtualizationStateRequest<F: Field> {
    pub label: &'static str,
    pub kernel: BackendKernelMetadata,
    pub rows: Vec<SumcheckStage6RaRow>,
    pub r_address_chunks: Vec<Vec<F>>,
    pub r_cycle: Vec<F>,
    pub gamma_powers: Vec<F>,
    pub input_claim: F,
    pub log_t: usize,
    pub chunk_bits: usize,
    pub virtual_polys: usize,
    pub committed_per_virtual: usize,
}

impl<F: Field> SumcheckInstructionRaVirtualizationStateRequest<F> {
    #[expect(clippy::too_many_arguments)]
    pub fn new(
        label: &'static str,
        rows: Vec<SumcheckStage6RaRow>,
        r_address_chunks: Vec<Vec<F>>,
        r_cycle: Vec<F>,
        gamma_powers: Vec<F>,
        input_claim: F,
        log_t: usize,
        chunk_bits: usize,
        virtual_polys: usize,
        committed_per_virtual: usize,
    ) -> Self {
        Self {
            label,
            kernel: BackendKernelMetadata::empty(),
            rows,
            r_address_chunks,
            r_cycle,
            gamma_powers,
            input_claim,
            log_t,
            chunk_bits,
            virtual_polys,
            committed_per_virtual,
        }
    }

    pub const fn with_optimization_ids(
        mut self,
        optimization_ids: &'static [&'static str],
    ) -> Self {
        self.kernel = self.kernel.with_optimization_ids(optimization_ids);
        self
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckIncClaimReductionStateRequest<F: Field> {
    pub label: &'static str,
    pub kernel: BackendKernelMetadata,
    pub rows: Vec<SumcheckStage6IncRow>,
    pub r_cycle_stage2: Vec<F>,
    pub r_cycle_stage4: Vec<F>,
    pub s_cycle_stage4: Vec<F>,
    pub s_cycle_stage5: Vec<F>,
    pub gamma: F,
    pub input_claim: F,
    pub log_t: usize,
}

impl<F: Field> SumcheckIncClaimReductionStateRequest<F> {
    #[expect(clippy::too_many_arguments)]
    pub fn new(
        label: &'static str,
        rows: Vec<SumcheckStage6IncRow>,
        r_cycle_stage2: Vec<F>,
        r_cycle_stage4: Vec<F>,
        s_cycle_stage4: Vec<F>,
        s_cycle_stage5: Vec<F>,
        gamma: F,
        input_claim: F,
        log_t: usize,
    ) -> Self {
        Self {
            label,
            kernel: BackendKernelMetadata::empty(),
            rows,
            r_cycle_stage2,
            r_cycle_stage4,
            s_cycle_stage4,
            s_cycle_stage5,
            gamma,
            input_claim,
            log_t,
        }
    }

    pub const fn with_optimization_ids(
        mut self,
        optimization_ids: &'static [&'static str],
    ) -> Self {
        self.kernel = self.kernel.with_optimization_ids(optimization_ids);
        self
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckFieldRegistersIncClaimReductionStateRequest<F: Field> {
    pub label: &'static str,
    pub kernel: BackendKernelMetadata,
    pub rows: Vec<SumcheckFieldRegistersReadWriteRow<F>>,
    pub r_cycle_read_write: Vec<F>,
    pub r_cycle_val_evaluation: Vec<F>,
    pub gamma: F,
    pub input_claim: F,
    pub log_t: usize,
}

impl<F: Field> SumcheckFieldRegistersIncClaimReductionStateRequest<F> {
    pub fn new(
        label: &'static str,
        rows: Vec<SumcheckFieldRegistersReadWriteRow<F>>,
        r_cycle_read_write: Vec<F>,
        r_cycle_val_evaluation: Vec<F>,
        gamma: F,
        input_claim: F,
        log_t: usize,
    ) -> Self {
        Self {
            label,
            kernel: BackendKernelMetadata::empty(),
            rows,
            r_cycle_read_write,
            r_cycle_val_evaluation,
            gamma,
            input_claim,
            log_t,
        }
    }

    pub const fn with_optimization_ids(
        mut self,
        optimization_ids: &'static [&'static str],
    ) -> Self {
        self.kernel = self.kernel.with_optimization_ids(optimization_ids);
        self
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckRamRaClaimReductionStateRequest<F: Field> {
    pub label: &'static str,
    pub kernel: BackendKernelMetadata,
    pub rows: Vec<SumcheckRamReadWriteRow>,
    pub r_address: Vec<F>,
    pub r_cycle_raf: Vec<F>,
    pub r_cycle_read_write: Vec<F>,
    pub r_cycle_val_check: Vec<F>,
    pub gamma: F,
    pub input_claim: F,
    pub log_t: usize,
    pub log_k: usize,
}

impl<F: Field> SumcheckRamRaClaimReductionStateRequest<F> {
    #[expect(clippy::too_many_arguments)]
    pub fn new(
        label: &'static str,
        rows: Vec<SumcheckRamReadWriteRow>,
        r_address: Vec<F>,
        r_cycle_raf: Vec<F>,
        r_cycle_read_write: Vec<F>,
        r_cycle_val_check: Vec<F>,
        gamma: F,
        input_claim: F,
        log_t: usize,
        log_k: usize,
    ) -> Self {
        Self {
            label,
            kernel: BackendKernelMetadata::empty(),
            rows,
            r_address,
            r_cycle_raf,
            r_cycle_read_write,
            r_cycle_val_check,
            gamma,
            input_claim,
            log_t,
            log_k,
        }
    }

    pub const fn with_kernel_metadata(mut self, kernel: BackendKernelMetadata) -> Self {
        self.kernel = kernel;
        self
    }

    pub const fn with_relation(mut self, relation: BackendRelationId) -> Self {
        self.kernel = self.kernel.with_relation(relation);
        self
    }

    pub const fn with_optimization_ids(
        mut self,
        optimization_ids: &'static [&'static str],
    ) -> Self {
        self.kernel = self.kernel.with_optimization_ids(optimization_ids);
        self
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckRegistersValEvaluationStateRequest<F: Field> {
    pub label: &'static str,
    pub kernel: BackendKernelMetadata,
    pub rows: Vec<SumcheckRegistersReadWriteRow>,
    pub r_address: Vec<F>,
    pub r_cycle: Vec<F>,
    pub input_claim: F,
    pub log_t: usize,
    pub log_k: usize,
}

impl<F: Field> SumcheckRegistersValEvaluationStateRequest<F> {
    pub fn new(
        label: &'static str,
        rows: Vec<SumcheckRegistersReadWriteRow>,
        r_address: Vec<F>,
        r_cycle: Vec<F>,
        input_claim: F,
        log_t: usize,
        log_k: usize,
    ) -> Self {
        Self {
            label,
            kernel: BackendKernelMetadata::empty(),
            rows,
            r_address,
            r_cycle,
            input_claim,
            log_t,
            log_k,
        }
    }

    pub const fn with_kernel_metadata(mut self, kernel: BackendKernelMetadata) -> Self {
        self.kernel = kernel;
        self
    }

    pub const fn with_relation(mut self, relation: BackendRelationId) -> Self {
        self.kernel = self.kernel.with_relation(relation);
        self
    }

    pub const fn with_optimization_ids(
        mut self,
        optimization_ids: &'static [&'static str],
    ) -> Self {
        self.kernel = self.kernel.with_optimization_ids(optimization_ids);
        self
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckRamRafStateRequest<F: Field> {
    pub label: &'static str,
    pub kernel: BackendKernelMetadata,
    pub rows: Vec<SumcheckRamReadWriteRow>,
    pub r_cycle: Vec<F>,
    pub input_claim: F,
    pub start_address: u64,
    pub log_t: usize,
    pub log_k: usize,
    pub phase1_num_rounds: usize,
    pub phase2_num_rounds: usize,
}

impl<F: Field> SumcheckRamRafStateRequest<F> {
    #[expect(clippy::too_many_arguments)]
    pub fn new(
        label: &'static str,
        rows: Vec<SumcheckRamReadWriteRow>,
        r_cycle: Vec<F>,
        input_claim: F,
        start_address: u64,
        log_t: usize,
        log_k: usize,
        phase1_num_rounds: usize,
        phase2_num_rounds: usize,
    ) -> Self {
        Self {
            label,
            kernel: BackendKernelMetadata::empty(),
            rows,
            r_cycle,
            input_claim,
            start_address,
            log_t,
            log_k,
            phase1_num_rounds,
            phase2_num_rounds,
        }
    }

    pub const fn with_relation(mut self, relation: BackendRelationId) -> Self {
        self.kernel = self.kernel.with_relation(relation);
        self
    }

    pub const fn with_optimization_ids(
        mut self,
        optimization_ids: &'static [&'static str],
    ) -> Self {
        self.kernel = self.kernel.with_optimization_ids(optimization_ids);
        self
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckRamOutputCheckStateRequest<F: Field> {
    pub label: &'static str,
    pub kernel: BackendKernelMetadata,
    pub final_ram_state: Vec<u64>,
    pub public_io_state: Vec<u64>,
    pub io_start: usize,
    pub io_end: usize,
    pub r_address: Vec<F>,
    pub log_t: usize,
    pub log_k: usize,
    pub phase1_num_rounds: usize,
    pub phase2_num_rounds: usize,
}

impl<F: Field> SumcheckRamOutputCheckStateRequest<F> {
    #[expect(clippy::too_many_arguments)]
    pub fn new(
        label: &'static str,
        final_ram_state: Vec<u64>,
        public_io_state: Vec<u64>,
        io_start: usize,
        io_end: usize,
        r_address: Vec<F>,
        log_t: usize,
        log_k: usize,
        phase1_num_rounds: usize,
        phase2_num_rounds: usize,
    ) -> Self {
        Self {
            label,
            kernel: BackendKernelMetadata::empty(),
            final_ram_state,
            public_io_state,
            io_start,
            io_end,
            r_address,
            log_t,
            log_k,
            phase1_num_rounds,
            phase2_num_rounds,
        }
    }

    pub const fn with_relation(mut self, relation: BackendRelationId) -> Self {
        self.kernel = self.kernel.with_relation(relation);
        self
    }

    pub const fn with_optimization_ids(
        mut self,
        optimization_ids: &'static [&'static str],
    ) -> Self {
        self.kernel = self.kernel.with_optimization_ids(optimization_ids);
        self
    }
}

impl<N: WitnessNamespace> SumcheckRequest<N> {
    pub const fn new(label: &'static str, instances: Vec<SumcheckInstanceRequest<N>>) -> Self {
        Self { label, instances }
    }

    pub fn is_empty(&self) -> bool {
        self.instances.is_empty()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckViewMaterializationRequest<N: WitnessNamespace> {
    pub slot: BackendValueSlot,
    pub requirement: ViewRequirement<N>,
}

impl<N: WitnessNamespace> SumcheckViewMaterializationRequest<N> {
    pub const fn new(slot: BackendValueSlot, requirement: ViewRequirement<N>) -> Self {
        Self { slot, requirement }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckMaterializationRequest<N: WitnessNamespace> {
    pub label: &'static str,
    pub views: Vec<SumcheckViewMaterializationRequest<N>>,
}

impl<N: WitnessNamespace> SumcheckMaterializationRequest<N> {
    pub const fn new(
        label: &'static str,
        views: Vec<SumcheckViewMaterializationRequest<N>>,
    ) -> Self {
        Self { label, views }
    }

    pub fn is_empty(&self) -> bool {
        self.views.is_empty()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckLinearProductQuery<F: Field> {
    pub slot: BackendValueSlot,
    pub point: Vec<F>,
    pub row_weights: Vec<F>,
    pub scale: F,
}

impl<F: Field> SumcheckLinearProductQuery<F> {
    pub const fn new(slot: BackendValueSlot, point: Vec<F>, row_weights: Vec<F>, scale: F) -> Self {
        Self {
            slot,
            point,
            row_weights,
            scale,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckLinearProductRequest<'a, F: Field> {
    pub label: &'static str,
    pub kernel: BackendKernelMetadata,
    pub witness_polynomials: &'a [Vec<F>],
    pub input_columns: &'a [usize],
    pub constant_column: usize,
    pub left_rows: &'a [Vec<(usize, F)>],
    pub right_rows: &'a [Vec<(usize, F)>],
    pub queries: Vec<SumcheckLinearProductQuery<F>>,
}

impl<'a, F: Field> SumcheckLinearProductRequest<'a, F> {
    pub const fn new(
        label: &'static str,
        witness_polynomials: &'a [Vec<F>],
        input_columns: &'a [usize],
        constant_column: usize,
        left_rows: &'a [Vec<(usize, F)>],
        right_rows: &'a [Vec<(usize, F)>],
        queries: Vec<SumcheckLinearProductQuery<F>>,
    ) -> Self {
        Self {
            label,
            kernel: BackendKernelMetadata::empty(),
            witness_polynomials,
            input_columns,
            constant_column,
            left_rows,
            right_rows,
            queries,
        }
    }

    pub const fn with_kernel_metadata(mut self, kernel: BackendKernelMetadata) -> Self {
        self.kernel = kernel;
        self
    }

    pub const fn with_relation(mut self, relation: BackendRelationId) -> Self {
        self.kernel = self.kernel.with_relation(relation);
        self
    }

    pub const fn with_optimization_ids(
        mut self,
        optimization_ids: &'static [&'static str],
    ) -> Self {
        self.kernel = self.kernel.with_optimization_ids(optimization_ids);
        self
    }

    pub fn is_empty(&self) -> bool {
        self.queries.is_empty()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckPrefixProductSumQuery<F: Field> {
    pub slot: BackendValueSlot,
    pub eq_point: Vec<F>,
    pub fixed_prefix: Vec<F>,
    pub suffix_vars: usize,
    pub row_weights_at_zero: Vec<F>,
    pub row_weights_at_one: Vec<F>,
    pub scale: F,
}

impl<F: Field> SumcheckPrefixProductSumQuery<F> {
    pub const fn new(
        slot: BackendValueSlot,
        eq_point: Vec<F>,
        fixed_prefix: Vec<F>,
        suffix_vars: usize,
        row_weights_at_zero: Vec<F>,
        row_weights_at_one: Vec<F>,
        scale: F,
    ) -> Self {
        Self {
            slot,
            eq_point,
            fixed_prefix,
            suffix_vars,
            row_weights_at_zero,
            row_weights_at_one,
            scale,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckPrefixProductSumRequest<'a, F: Field> {
    pub label: &'static str,
    pub kernel: BackendKernelMetadata,
    pub witness_polynomials: &'a [Vec<F>],
    pub input_columns: &'a [usize],
    pub constant_column: usize,
    pub left_rows: &'a [Vec<(usize, F)>],
    pub right_rows: &'a [Vec<(usize, F)>],
    pub queries: Vec<SumcheckPrefixProductSumQuery<F>>,
}

impl<'a, F: Field> SumcheckPrefixProductSumRequest<'a, F> {
    pub const fn new(
        label: &'static str,
        witness_polynomials: &'a [Vec<F>],
        input_columns: &'a [usize],
        constant_column: usize,
        left_rows: &'a [Vec<(usize, F)>],
        right_rows: &'a [Vec<(usize, F)>],
        queries: Vec<SumcheckPrefixProductSumQuery<F>>,
    ) -> Self {
        Self {
            label,
            kernel: BackendKernelMetadata::empty(),
            witness_polynomials,
            input_columns,
            constant_column,
            left_rows,
            right_rows,
            queries,
        }
    }

    pub const fn with_kernel_metadata(mut self, kernel: BackendKernelMetadata) -> Self {
        self.kernel = kernel;
        self
    }

    pub const fn with_relation(mut self, relation: BackendRelationId) -> Self {
        self.kernel = self.kernel.with_relation(relation);
        self
    }

    pub const fn with_optimization_ids(
        mut self,
        optimization_ids: &'static [&'static str],
    ) -> Self {
        self.kernel = self.kernel.with_optimization_ids(optimization_ids);
        self
    }

    pub fn is_empty(&self) -> bool {
        self.queries.is_empty()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckRowProductQuery<F: Field> {
    pub slot: BackendValueSlot,
    pub eq_point: Vec<F>,
    pub row_weights: Vec<F>,
    pub scale: F,
}

impl<F: Field> SumcheckRowProductQuery<F> {
    pub const fn new(
        slot: BackendValueSlot,
        eq_point: Vec<F>,
        row_weights: Vec<F>,
        scale: F,
    ) -> Self {
        Self {
            slot,
            eq_point,
            row_weights,
            scale,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckRowProductRequest<'a, F: Field> {
    pub label: &'static str,
    pub kernel: BackendKernelMetadata,
    pub witness_polynomials: &'a [Vec<F>],
    pub input_columns: &'a [usize],
    pub constant_column: usize,
    pub left_rows: &'a [Vec<(usize, F)>],
    pub right_rows: &'a [Vec<(usize, F)>],
    pub queries: Vec<SumcheckRowProductQuery<F>>,
}

impl<'a, F: Field> SumcheckRowProductRequest<'a, F> {
    pub const fn new(
        label: &'static str,
        witness_polynomials: &'a [Vec<F>],
        input_columns: &'a [usize],
        constant_column: usize,
        left_rows: &'a [Vec<(usize, F)>],
        right_rows: &'a [Vec<(usize, F)>],
        queries: Vec<SumcheckRowProductQuery<F>>,
    ) -> Self {
        Self {
            label,
            kernel: BackendKernelMetadata::empty(),
            witness_polynomials,
            input_columns,
            constant_column,
            left_rows,
            right_rows,
            queries,
        }
    }

    pub const fn with_kernel_metadata(mut self, kernel: BackendKernelMetadata) -> Self {
        self.kernel = kernel;
        self
    }

    pub const fn with_relation(mut self, relation: BackendRelationId) -> Self {
        self.kernel = self.kernel.with_relation(relation);
        self
    }

    pub const fn with_optimization_ids(
        mut self,
        optimization_ids: &'static [&'static str],
    ) -> Self {
        self.kernel = self.kernel.with_optimization_ids(optimization_ids);
        self
    }

    pub fn is_empty(&self) -> bool {
        self.queries.is_empty()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SumcheckProductUniskipRow {
    pub right_instruction: i128,
    pub left_instruction: u64,
    pub lookup_output: u64,
    pub jump_flag: bool,
    pub branch_flag: bool,
    pub next_is_noop: bool,
}

impl SumcheckProductUniskipRow {
    pub const fn new(
        left_instruction: u64,
        lookup_output: u64,
        jump_flag: bool,
        right_instruction: i128,
        branch_flag: bool,
        next_is_noop: bool,
    ) -> Self {
        Self {
            right_instruction,
            left_instruction,
            lookup_output,
            jump_flag,
            branch_flag,
            next_is_noop,
        }
    }
}

pub fn product_uniskip_rows_from_stage2_trace(
    rows: &[JoltVmStage2TraceRow],
) -> Vec<SumcheckProductUniskipRow> {
    rows.iter()
        .map(product_uniskip_row_from_stage2_trace)
        .collect()
}

pub fn product_uniskip_row_from_stage2_trace(
    row: &JoltVmStage2TraceRow,
) -> SumcheckProductUniskipRow {
    SumcheckProductUniskipRow::new(
        row.left_instruction_input,
        row.lookup_output,
        row.jump_flag,
        row.right_instruction_input,
        row.branch_flag,
        row.next_is_noop,
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SumcheckSpartanOuterRow {
    pub left_instruction_input: u64,
    pub right_instruction_input: i128,
    pub product_magnitude: u128,
    pub product_is_positive: bool,
    pub should_branch: bool,
    pub pc: u64,
    pub unexpanded_pc: u64,
    pub imm: i128,
    pub ram_address: u64,
    pub rs1_value: u64,
    pub rs2_value: u64,
    pub rd_write_value: u64,
    pub ram_read_value: u64,
    pub ram_write_value: u64,
    pub left_lookup_operand: u64,
    pub right_lookup_operand: u128,
    pub next_unexpanded_pc: u64,
    pub next_pc: u64,
    pub next_is_virtual: bool,
    pub next_is_first_in_sequence: bool,
    pub lookup_output: u64,
    pub should_jump: bool,
    pub flag_add_operands: bool,
    pub flag_subtract_operands: bool,
    pub flag_multiply_operands: bool,
    pub flag_load: bool,
    pub flag_store: bool,
    pub flag_jump: bool,
    pub flag_write_lookup_output_to_rd: bool,
    pub flag_virtual_instruction: bool,
    pub flag_assert: bool,
    pub flag_do_not_update_unexpanded_pc: bool,
    pub flag_advice: bool,
    pub flag_is_compressed: bool,
    pub flag_is_first_in_sequence: bool,
    pub flag_is_last_in_sequence: bool,
}

pub fn spartan_outer_rows<W>(witness: &W) -> Result<Vec<SumcheckSpartanOuterRow>, WitnessError>
where
    W: JoltVmSpartanOuterRows,
{
    Ok(witness
        .spartan_outer_rows()?
        .into_iter()
        .map(spartan_outer_row)
        .collect())
}

pub fn spartan_outer_row(row: JoltVmSpartanOuterRow) -> SumcheckSpartanOuterRow {
    SumcheckSpartanOuterRow {
        left_instruction_input: row.left_instruction_input,
        right_instruction_input: row.right_instruction_input,
        product_magnitude: row.product_magnitude,
        product_is_positive: row.product_is_positive,
        should_branch: row.should_branch,
        pc: row.pc,
        unexpanded_pc: row.unexpanded_pc,
        imm: row.imm,
        ram_address: row.ram_address,
        rs1_value: row.rs1_value,
        rs2_value: row.rs2_value,
        rd_write_value: row.rd_write_value,
        ram_read_value: row.ram_read_value,
        ram_write_value: row.ram_write_value,
        left_lookup_operand: row.left_lookup_operand,
        right_lookup_operand: row.right_lookup_operand,
        next_unexpanded_pc: row.next_unexpanded_pc,
        next_pc: row.next_pc,
        next_is_virtual: row.next_is_virtual,
        next_is_first_in_sequence: row.next_is_first_in_sequence,
        lookup_output: row.lookup_output,
        should_jump: row.should_jump,
        flag_add_operands: row.flag_add_operands,
        flag_subtract_operands: row.flag_subtract_operands,
        flag_multiply_operands: row.flag_multiply_operands,
        flag_load: row.flag_load,
        flag_store: row.flag_store,
        flag_jump: row.flag_jump,
        flag_write_lookup_output_to_rd: row.flag_write_lookup_output_to_rd,
        flag_virtual_instruction: row.flag_virtual_instruction,
        flag_assert: row.flag_assert,
        flag_do_not_update_unexpanded_pc: row.flag_do_not_update_unexpanded_pc,
        flag_advice: row.flag_advice,
        flag_is_compressed: row.flag_is_compressed,
        flag_is_first_in_sequence: row.flag_is_first_in_sequence,
        flag_is_last_in_sequence: row.flag_is_last_in_sequence,
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckSpartanOuterUniskipQuery<F: Field> {
    pub slot: BackendValueSlot,
    pub eq_point: Vec<F>,
    pub coeffs: Vec<i32>,
    pub scale: F,
}

impl<F: Field> SumcheckSpartanOuterUniskipQuery<F> {
    pub const fn new(slot: BackendValueSlot, eq_point: Vec<F>, coeffs: Vec<i32>, scale: F) -> Self {
        Self {
            slot,
            eq_point,
            coeffs,
            scale,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckSpartanOuterUniskipRequest<'a, F: Field> {
    pub label: &'static str,
    pub kernel: BackendKernelMetadata,
    pub rows: &'a [SumcheckSpartanOuterRow],
    pub queries: Vec<SumcheckSpartanOuterUniskipQuery<F>>,
}

impl<'a, F: Field> SumcheckSpartanOuterUniskipRequest<'a, F> {
    pub const fn new(
        label: &'static str,
        rows: &'a [SumcheckSpartanOuterRow],
        queries: Vec<SumcheckSpartanOuterUniskipQuery<F>>,
    ) -> Self {
        Self {
            label,
            kernel: BackendKernelMetadata::empty(),
            rows,
            queries,
        }
    }

    pub const fn with_kernel_metadata(mut self, kernel: BackendKernelMetadata) -> Self {
        self.kernel = kernel;
        self
    }

    pub const fn with_relation(mut self, relation: BackendRelationId) -> Self {
        self.kernel = self.kernel.with_relation(relation);
        self
    }

    pub const fn with_optimization_ids(
        mut self,
        optimization_ids: &'static [&'static str],
    ) -> Self {
        self.kernel = self.kernel.with_optimization_ids(optimization_ids);
        self
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckSpartanOuterRemainderQuery<F: Field> {
    pub slot: BackendValueSlot,
    pub eq_point: Vec<F>,
    pub fixed_prefix: Vec<F>,
    pub suffix_vars: usize,
    pub uniskip_challenge: F,
    pub uniskip_domain_size: usize,
    pub scale: F,
}

impl<F: Field> SumcheckSpartanOuterRemainderQuery<F> {
    pub const fn new(
        slot: BackendValueSlot,
        eq_point: Vec<F>,
        fixed_prefix: Vec<F>,
        suffix_vars: usize,
        uniskip_challenge: F,
        scale: F,
    ) -> Self {
        Self {
            slot,
            eq_point,
            fixed_prefix,
            suffix_vars,
            uniskip_challenge,
            uniskip_domain_size: 10,
            scale,
        }
    }

    pub const fn with_uniskip_domain_size(mut self, uniskip_domain_size: usize) -> Self {
        self.uniskip_domain_size = uniskip_domain_size;
        self
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckSpartanOuterRemainderRequest<'a, F: Field> {
    pub label: &'static str,
    pub kernel: BackendKernelMetadata,
    pub rows: &'a [SumcheckSpartanOuterRow],
    pub queries: Vec<SumcheckSpartanOuterRemainderQuery<F>>,
}

impl<'a, F: Field> SumcheckSpartanOuterRemainderRequest<'a, F> {
    pub const fn new(
        label: &'static str,
        rows: &'a [SumcheckSpartanOuterRow],
        queries: Vec<SumcheckSpartanOuterRemainderQuery<F>>,
    ) -> Self {
        Self {
            label,
            kernel: BackendKernelMetadata::empty(),
            rows,
            queries,
        }
    }

    pub const fn with_kernel_metadata(mut self, kernel: BackendKernelMetadata) -> Self {
        self.kernel = kernel;
        self
    }

    pub const fn with_relation(mut self, relation: BackendRelationId) -> Self {
        self.kernel = self.kernel.with_relation(relation);
        self
    }

    pub const fn with_optimization_ids(
        mut self,
        optimization_ids: &'static [&'static str],
    ) -> Self {
        self.kernel = self.kernel.with_optimization_ids(optimization_ids);
        self
    }

    pub fn is_empty(&self) -> bool {
        self.queries.is_empty()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckSpartanOuterRemainderStateRequest<'a, F: Field> {
    pub label: &'static str,
    pub kernel: BackendKernelMetadata,
    pub witness_polynomials: &'a [Vec<F>],
    pub input_columns: &'a [usize],
    pub constant_column: usize,
    pub left_rows: &'a [Vec<(usize, F)>],
    pub right_rows: &'a [Vec<(usize, F)>],
    pub eq_point: Vec<F>,
    pub row_weights_at_zero: Vec<F>,
    pub row_weights_at_one: Vec<F>,
    pub stream_challenge: F,
    pub scale: F,
}

impl<'a, F: Field> SumcheckSpartanOuterRemainderStateRequest<'a, F> {
    #[expect(
        clippy::too_many_arguments,
        reason = "Request constructor names every Stage 1 remainder kernel input explicitly."
    )]
    pub const fn new(
        label: &'static str,
        witness_polynomials: &'a [Vec<F>],
        input_columns: &'a [usize],
        constant_column: usize,
        left_rows: &'a [Vec<(usize, F)>],
        right_rows: &'a [Vec<(usize, F)>],
        eq_point: Vec<F>,
        row_weights_at_zero: Vec<F>,
        row_weights_at_one: Vec<F>,
        stream_challenge: F,
        scale: F,
    ) -> Self {
        Self {
            label,
            kernel: BackendKernelMetadata::empty(),
            witness_polynomials,
            input_columns,
            constant_column,
            left_rows,
            right_rows,
            eq_point,
            row_weights_at_zero,
            row_weights_at_one,
            stream_challenge,
            scale,
        }
    }

    pub const fn with_kernel_metadata(mut self, kernel: BackendKernelMetadata) -> Self {
        self.kernel = kernel;
        self
    }

    pub const fn with_relation(mut self, relation: BackendRelationId) -> Self {
        self.kernel = self.kernel.with_relation(relation);
        self
    }

    pub const fn with_optimization_ids(
        mut self,
        optimization_ids: &'static [&'static str],
    ) -> Self {
        self.kernel = self.kernel.with_optimization_ids(optimization_ids);
        self
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckSpartanOuterRemainderRowStateRequest<'a, F: Field> {
    pub label: &'static str,
    pub kernel: BackendKernelMetadata,
    pub rows: &'a [SumcheckSpartanOuterRow],
    pub eq_point: Vec<F>,
    pub uniskip_challenge: F,
    pub uniskip_domain_size: usize,
    pub stream_challenge: F,
    pub scale: F,
}

impl<'a, F: Field> SumcheckSpartanOuterRemainderRowStateRequest<'a, F> {
    pub const fn new(
        label: &'static str,
        rows: &'a [SumcheckSpartanOuterRow],
        eq_point: Vec<F>,
        uniskip_challenge: F,
        stream_challenge: F,
        scale: F,
    ) -> Self {
        Self {
            label,
            kernel: BackendKernelMetadata::empty(),
            rows,
            eq_point,
            uniskip_challenge,
            uniskip_domain_size: 10,
            stream_challenge,
            scale,
        }
    }

    pub const fn with_uniskip_domain_size(mut self, uniskip_domain_size: usize) -> Self {
        self.uniskip_domain_size = uniskip_domain_size;
        self
    }

    pub const fn with_kernel_metadata(mut self, kernel: BackendKernelMetadata) -> Self {
        self.kernel = kernel;
        self
    }

    pub const fn with_relation(mut self, relation: BackendRelationId) -> Self {
        self.kernel = self.kernel.with_relation(relation);
        self
    }

    pub const fn with_optimization_ids(
        mut self,
        optimization_ids: &'static [&'static str],
    ) -> Self {
        self.kernel = self.kernel.with_optimization_ids(optimization_ids);
        self
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckProductUniskipRequest<'a, F: Field> {
    pub label: &'static str,
    pub kernel: BackendKernelMetadata,
    pub rows: &'a [SumcheckProductUniskipRow],
    pub queries: Vec<SumcheckRowProductQuery<F>>,
}

impl<'a, F: Field> SumcheckProductUniskipRequest<'a, F> {
    pub const fn new(
        label: &'static str,
        rows: &'a [SumcheckProductUniskipRow],
        queries: Vec<SumcheckRowProductQuery<F>>,
    ) -> Self {
        Self {
            label,
            kernel: BackendKernelMetadata::empty(),
            rows,
            queries,
        }
    }

    pub const fn with_kernel_metadata(mut self, kernel: BackendKernelMetadata) -> Self {
        self.kernel = kernel;
        self
    }

    pub const fn with_relation(mut self, relation: BackendRelationId) -> Self {
        self.kernel = self.kernel.with_relation(relation);
        self
    }

    pub const fn with_optimization_ids(
        mut self,
        optimization_ids: &'static [&'static str],
    ) -> Self {
        self.kernel = self.kernel.with_optimization_ids(optimization_ids);
        self
    }

    pub fn is_empty(&self) -> bool {
        self.queries.is_empty()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckViewEvaluationRequest<F: Field, N: WitnessNamespace> {
    pub slot: BackendValueSlot,
    pub requirement: ViewRequirement<N>,
    pub point: Vec<F>,
}

impl<F: Field, N: WitnessNamespace> SumcheckViewEvaluationRequest<F, N> {
    pub const fn new(
        slot: BackendValueSlot,
        requirement: ViewRequirement<N>,
        point: Vec<F>,
    ) -> Self {
        Self {
            slot,
            requirement,
            point,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckEvaluationRequest<F: Field, N: WitnessNamespace> {
    pub label: &'static str,
    pub views: Vec<SumcheckViewEvaluationRequest<F, N>>,
}

impl<F: Field, N: WitnessNamespace> SumcheckEvaluationRequest<F, N> {
    pub const fn new(label: &'static str, views: Vec<SumcheckViewEvaluationRequest<F, N>>) -> Self {
        Self { label, views }
    }

    pub fn is_empty(&self) -> bool {
        self.views.is_empty()
    }
}

/// Request to materialize the Stage 7 hamming-weight RA-family pushforward
/// `G_i(k) = Σ_j eq(r_cycle, j) · ra_i(k, j)` for every committed RA polynomial.
///
/// The backend sources per-cycle one-hot chunk indices via the generic
/// [`WitnessProvider::committed_stream`](jolt_witness::WitnessProvider::committed_stream)
/// path (the chunk selector is already applied by the witness), assembles them
/// into the family layout, and reduces over the cycle hypercube against
/// `r_cycle`. RA ids are listed in canonical order (instruction, then bytecode,
/// then RAM); the returned `G` tables follow the same order.
#[derive(Clone, Debug)]
pub struct SumcheckRaPushforwardRequest<F: Field, N: WitnessNamespace> {
    pub label: &'static str,
    pub instruction_ids: Vec<N::CommittedId>,
    pub bytecode_ids: Vec<N::CommittedId>,
    pub ram_ids: Vec<N::CommittedId>,
    pub log_k_chunk: usize,
    pub r_cycle: Vec<F>,
    pub chunk_size: usize,
}

impl<F: Field, N: WitnessNamespace> SumcheckRaPushforwardRequest<F, N> {
    pub fn new(
        label: &'static str,
        instruction_ids: Vec<N::CommittedId>,
        bytecode_ids: Vec<N::CommittedId>,
        ram_ids: Vec<N::CommittedId>,
        log_k_chunk: usize,
        r_cycle: Vec<F>,
        chunk_size: usize,
    ) -> Self {
        Self {
            label,
            instruction_ids,
            bytecode_ids,
            ram_ids,
            log_k_chunk,
            r_cycle,
            chunk_size,
        }
    }

    pub fn num_polys(&self) -> usize {
        self.instruction_ids.len() + self.bytecode_ids.len() + self.ram_ids.len()
    }
}

#[derive(Clone, Debug)]
pub struct SumcheckStage7HammingStateRequest<F: Field, N: WitnessNamespace> {
    pub label: &'static str,
    pub instruction_ids: Vec<N::CommittedId>,
    pub bytecode_ids: Vec<N::CommittedId>,
    pub ram_ids: Vec<N::CommittedId>,
    pub log_k_chunk: usize,
    pub r_cycle: Vec<F>,
    pub r_addr_bool: Vec<F>,
    pub r_addr_virt: Vec<Vec<F>>,
    pub gamma_powers: Vec<F>,
    pub chunk_size: usize,
}

impl<F: Field, N: WitnessNamespace> SumcheckStage7HammingStateRequest<F, N> {
    #[expect(
        clippy::too_many_arguments,
        reason = "Stage 7 hamming requests are protocol records whose fields map one-to-one to verifier inputs."
    )]
    pub fn new(
        label: &'static str,
        instruction_ids: Vec<N::CommittedId>,
        bytecode_ids: Vec<N::CommittedId>,
        ram_ids: Vec<N::CommittedId>,
        log_k_chunk: usize,
        r_cycle: Vec<F>,
        r_addr_bool: Vec<F>,
        r_addr_virt: Vec<Vec<F>>,
        gamma_powers: Vec<F>,
        chunk_size: usize,
    ) -> Self {
        Self {
            label,
            instruction_ids,
            bytecode_ids,
            ram_ids,
            log_k_chunk,
            r_cycle,
            r_addr_bool,
            r_addr_virt,
            gamma_powers,
            chunk_size,
        }
    }

    pub fn num_polys(&self) -> usize {
        self.instruction_ids.len() + self.bytecode_ids.len() + self.ram_ids.len()
    }
}

impl<F: Field> SumcheckStage7HammingStateRequest<F, JoltVmNamespace> {
    pub fn jolt_hamming_weight_claim_reduction(
        dimensions: HammingWeightClaimReductionDimensions,
        r_cycle: Vec<F>,
        r_addr_bool: Vec<F>,
        r_addr_virt: Vec<Vec<F>>,
        hamming_gamma: F,
        chunk_size: usize,
    ) -> Self {
        let committed_polynomials =
            hamming_weight::claim_reduction_committed_polynomials(dimensions);
        Self::new(
            "stage7.hamming_weight_claim_reduction",
            committed_polynomials.instruction_ra,
            committed_polynomials.bytecode_ra,
            committed_polynomials.ram_ra,
            dimensions.log_k_chunk,
            r_cycle,
            r_addr_bool,
            r_addr_virt,
            powers(hamming_gamma, 3 * dimensions.layout.total()),
            chunk_size,
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SumcheckAdviceTraceOrder {
    CycleMajor,
    AddressMajor,
}

impl From<TracePolynomialOrder> for SumcheckAdviceTraceOrder {
    fn from(order: TracePolynomialOrder) -> Self {
        match order {
            TracePolynomialOrder::CycleMajor => Self::CycleMajor,
            TracePolynomialOrder::AddressMajor => Self::AddressMajor,
        }
    }
}

impl From<SumcheckAdviceTraceOrder> for TracePolynomialOrder {
    fn from(order: SumcheckAdviceTraceOrder) -> Self {
        match order {
            SumcheckAdviceTraceOrder::CycleMajor => Self::CycleMajor,
            SumcheckAdviceTraceOrder::AddressMajor => Self::AddressMajor,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SumcheckStage7AdviceAddressStateRequest<F: Field, N: WitnessNamespace> {
    pub label: &'static str,
    pub advice_id: N::CommittedId,
    pub trace_order: SumcheckAdviceTraceOrder,
    pub log_t: usize,
    pub log_k_chunk: usize,
    pub main_column_vars: usize,
    pub advice_column_vars: usize,
    pub advice_row_vars: usize,
    pub reference_opening_point: Vec<F>,
    pub cycle_phase_variables: Vec<F>,
    pub dummy_cycle_phase_rounds: usize,
    pub chunk_size: usize,
}

impl<F: Field, N: WitnessNamespace> SumcheckStage7AdviceAddressStateRequest<F, N> {
    #[expect(clippy::too_many_arguments)]
    pub fn new(
        label: &'static str,
        advice_id: N::CommittedId,
        trace_order: SumcheckAdviceTraceOrder,
        log_t: usize,
        log_k_chunk: usize,
        main_column_vars: usize,
        advice_column_vars: usize,
        advice_row_vars: usize,
        reference_opening_point: Vec<F>,
        cycle_phase_variables: Vec<F>,
        dummy_cycle_phase_rounds: usize,
        chunk_size: usize,
    ) -> Self {
        Self {
            label,
            advice_id,
            trace_order,
            log_t,
            log_k_chunk,
            main_column_vars,
            advice_column_vars,
            advice_row_vars,
            reference_opening_point,
            cycle_phase_variables,
            dummy_cycle_phase_rounds,
            chunk_size,
        }
    }

    pub const fn total_vars(&self) -> usize {
        self.advice_column_vars + self.advice_row_vars
    }

    pub fn total_rows(&self) -> Option<usize> {
        1usize.checked_shl(self.total_vars() as u32)
    }

    pub fn address_phase_rounds(&self) -> usize {
        self.total_vars()
            .saturating_sub(self.cycle_phase_variables.len())
    }
}

impl<F: Field> SumcheckStage7AdviceAddressStateRequest<F, JoltVmNamespace> {
    pub fn jolt_advice_address_phase(
        kind: JoltAdviceKind,
        layout: &AdviceClaimReductionLayout,
        reference_opening_point: Vec<F>,
        cycle_phase_variables: Vec<F>,
        chunk_size: usize,
    ) -> Self {
        Self::new(
            match kind {
                JoltAdviceKind::Trusted => "stage7.trusted_advice_address_phase",
                JoltAdviceKind::Untrusted => "stage7.untrusted_advice_address_phase",
            },
            match kind {
                JoltAdviceKind::Trusted => JoltCommittedPolynomial::TrustedAdvice,
                JoltAdviceKind::Untrusted => JoltCommittedPolynomial::UntrustedAdvice,
            },
            layout.trace_order().into(),
            layout.log_t(),
            layout.log_k_chunk(),
            layout.main_shape().column_vars(),
            layout.advice_shape().column_vars(),
            layout.advice_shape().row_vars(),
            reference_opening_point,
            cycle_phase_variables,
            layout.dummy_cycle_phase_rounds(),
            chunk_size,
        )
    }
}

fn powers<F: Field>(base: F, count: usize) -> Vec<F> {
    let mut powers = Vec::with_capacity(count);
    let mut current = F::one();
    for _ in 0..count {
        powers.push(current);
        current *= base;
    }
    powers
}
