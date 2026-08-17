#![expect(
    dead_code,
    reason = "the Atom catalogue names every atom a relation can ask for, including ones no \
              registered bundle reads yet"
)]

use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;

use jolt_riscv::{CircuitFlags, InstructionFlags, JoltInstructionKind};
use jolt_witness::__private::TraceRow;
use jolt_witness::witnesses::{
    BytecodePc, Imm, InstructionFlag, InstructionRafFlag, LeftInstructionInput, LeftLookupOperand,
    LookupIndex, LookupOutput, MappedPc, NextIsNoop, OpFlag, Pc, Product, RamAddress,
    RamHammingWeight, RamInc, RamReadValue, RamWriteValue, RdAddress, RdInc, RdWriteValue,
    RemappedRamAddress, RightInstructionInput, RightLookupOperand, Rs1Value, Rs2Value,
    ShouldBranch, ShouldJump, TableIndex, UnexpandedPc,
};
use jolt_witness::witnesses::{Extract, WitnessEnv};
use jolt_witness::{collect_bundles, RowSource, WitnessBundle, WitnessError};

use crate::commitment::CommittedColumnsWitness;
use crate::cuda::bytecode_read_raf::witness::BytecodeReadRafCycleWitness;
use crate::cuda::common::one_hot_witness::OneHotCycleWitness;
use crate::cuda::common::ram_address_witness::RamAddressWitness;
use crate::cuda::inc_claim_reduction::witness::IncClaimReductionWitness;
use crate::cuda::instruction_claim_reduction::witness::InstructionClaimReductionWitness;
use crate::cuda::instruction_input::witness::InstructionInputWitness;
use crate::cuda::instruction_ra_virtualization::witness::InstructionRaVirtualizationWitness;
use crate::cuda::ram_hamming_booleanity::witness::RamHammingBooleanityWitness;
use crate::cuda::ram_read_write::witness::RamReadWriteWitness;
use crate::cuda::ram_val_check::witness::RamValCheckWitness;
use crate::cuda::registers_claim_reduction::witness::RegistersClaimReductionWitness;
use crate::cuda::registers_read_write::witness::{
    RdPreValue, RegistersReadWriteWitness, Rs1Address, Rs2Address,
};
use crate::cuda::registers_val_evaluation::witness::RegistersValEvaluationWitness;
use crate::cuda::spartan_outer::witness::SpartanOuterWitness;
use crate::cuda::spartan_product::witness::SpartanProductWitness;
use crate::cuda::spartan_shift::witness::SpartanShiftWitness;
use crate::reference::bytecode_read_raf::BytecodeReadRafWitness;
use crate::reference::instruction_read_raf::InstructionReadRafWitness;
use crate::ProofSession;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NoopRow(pub bool);

impl Extract for NoopRow {
    fn extract(
        row: &TraceRow,
        _next: Option<&TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        Ok(Self(
            row.instruction.instruction_kind == JoltInstructionKind::NoOp,
        ))
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, WitnessBundle)]
pub struct NoopRowWitness {
    pub noop: NoopRow,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, WitnessBundle)]
pub struct ShiftResidualWitness {
    #[opening(InstructionFlags(InstructionFlags::IsNoop))]
    pub is_noop: InstructionFlag,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, WitnessBundle)]
pub struct ProductResidualWitness {
    #[opening(InstructionFlags(InstructionFlags::Branch))]
    pub branch: InstructionFlag,
    pub next_is_noop: NextIsNoop,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, WitnessBundle)]
pub struct InstructionInputResidualWitness {
    #[opening(InstructionFlags(InstructionFlags::LeftOperandIsRs1Value))]
    pub left_operand_is_rs1: InstructionFlag,
    #[opening(InstructionFlags(InstructionFlags::LeftOperandIsPC))]
    pub left_operand_is_pc: InstructionFlag,
    #[opening(InstructionFlags(InstructionFlags::RightOperandIsRs2Value))]
    pub right_operand_is_rs2: InstructionFlag,
    #[opening(InstructionFlags(InstructionFlags::RightOperandIsImm))]
    pub right_operand_is_imm: InstructionFlag,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, WitnessBundle)]
pub struct RegisterAddressResidualWitness {
    pub rs1_address: Rs1Address,
    pub rs2_address: Rs2Address,
    pub rd_address: RdAddress,
    pub rd_pre_value: RdPreValue,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, WitnessBundle)]
pub struct InstructionReadRafResidualWitness {
    pub table_index: TableIndex,
    #[opening(InstructionRafFlag)]
    pub raf_flag: InstructionRafFlag,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Atom {
    NoopRow,
    RdInc,
    RamInc,
    LookupIndex,
    TableIndex,
    InstructionRafFlag,
    MappedPc,
    BytecodePc,
    Pc,
    UnexpandedPc,
    RemappedRamAddress,
    RamAddress,
    RamReadValue,
    RamWriteValue,
    RamHammingWeight,
    LookupOutput,
    LeftLookupOperand,
    RightLookupOperand,
    LeftInstructionInput,
    RightInstructionInput,
    Imm,
    Product,
    Rs1Value,
    Rs2Value,
    RdWriteValue,
    Rs1Address,
    Rs2Address,
    RdAddress,
    RdPreValue,
    ShouldBranch,
    ShouldJump,
    NextIsNoop,
    OpFlag(CircuitFlags),
    InstructionFlag(InstructionFlags),
}

struct Column {
    values: Arc<dyn Any + Send + Sync>,
    bytes: usize,
}

#[derive(Default)]
pub(crate) struct TraceColumns {
    source: usize,
    cycles: usize,
    columns: HashMap<Atom, Column>,
}

impl TraceColumns {
    fn reset(&mut self, source: usize, cycles: usize) {
        self.source = source;
        self.cycles = cycles;
        self.columns.clear();
    }

    fn serves(&self, source: usize, cycles: usize) -> bool {
        self.source == source && self.cycles == cycles
    }

    pub(crate) fn column<T: Send + Sync + 'static>(&self, atom: Atom) -> Option<&[T]> {
        self.columns
            .get(&atom)?
            .values
            .downcast_ref::<Vec<T>>()
            .map(Vec::as_slice)
    }

    pub(crate) fn insert<T: Send + Sync + 'static>(&mut self, atom: Atom, values: Vec<T>) {
        let bytes = values.capacity() * size_of::<T>();
        let _ = self.columns.insert(
            atom,
            Column {
                values: Arc::new(values),
                bytes,
            },
        );
        if atom == Atom::MappedPc || atom == Atom::NoopRow {
            self.derive_pc();
            self.derive_bytecode_pc();
        }
    }

    fn derive_pc(&mut self) {
        let Some(mapped) = self.column::<MappedPc>(Atom::MappedPc) else {
            return;
        };
        let mut pc = Vec::with_capacity(mapped.len());
        for entry in mapped {
            match entry.0 {
                Some(index) => pc.push(Pc(index as u64)),
                None => return,
            }
        }
        let bytes = pc.capacity() * size_of::<Pc>();
        let _ = self.columns.insert(
            Atom::Pc,
            Column {
                values: Arc::new(pc),
                bytes,
            },
        );
    }

    fn derive_bytecode_pc(&mut self) {
        let Some(mapped) = self.column::<MappedPc>(Atom::MappedPc) else {
            return;
        };
        let Some(noop) = self.column::<NoopRow>(Atom::NoopRow) else {
            return;
        };
        if noop.len() != mapped.len() {
            return;
        }
        let bytecode_pc: Vec<BytecodePc> = mapped
            .iter()
            .zip(noop)
            .map(|(mapped, noop)| {
                if noop.0 {
                    BytecodePc(0)
                } else {
                    BytecodePc(mapped.0.unwrap_or(0))
                }
            })
            .collect();
        let bytes = bytecode_pc.capacity() * size_of::<BytecodePc>();
        let _ = self.columns.insert(
            Atom::BytecodePc,
            Column {
                values: Arc::new(bytecode_pc),
                bytes,
            },
        );
    }
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for TraceColumns {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        for (atom, column) in &self.columns {
            visitor.visit_simple(allocative::Key::new(atom_key(*atom)), column.bytes);
        }
        visitor.exit();
    }
}

#[cfg(feature = "allocative")]
fn atom_key(atom: Atom) -> &'static str {
    match atom {
        Atom::LookupIndex => "LookupIndex",
        Atom::MappedPc => "MappedPc",
        Atom::Pc => "Pc",
        Atom::RemappedRamAddress => "RemappedRamAddress",
        Atom::RdInc => "RdInc",
        Atom::RamInc => "RamInc",
        Atom::OpFlag(_) => "OpFlag",
        Atom::InstructionFlag(_) => "InstructionFlag",
        _ => "atom",
    }
}

pub(crate) trait CachedBundle: WitnessBundle + Clone + Send + Sync + 'static {
    const ATOMS: &'static [Atom];

    #[cfg(test)]
    fn atom_types() -> Vec<(Atom, std::any::TypeId)>;

    fn store(rows: &[Self], columns: &mut TraceColumns);

    fn restore(columns: &TraceColumns, cycles: usize) -> Option<Vec<Self>>;

    fn walk_residual<S: RowSource + ?Sized>(
        _session: &mut ProofSession,
        _source: &S,
        _cycles: usize,
    ) -> Result<bool, WitnessError> {
        Ok(false)
    }

    #[cfg(test)]
    fn residual_atoms() -> Option<&'static [Atom]> {
        None
    }
}

macro_rules! cached_residual {
    () => {};
    ($residual:ty) => {
        fn walk_residual<S: RowSource + ?Sized>(
            session: &mut ProofSession,
            source: &S,
            cycles: usize,
        ) -> Result<bool, WitnessError> {
            let _ = cached_bundles::<$residual, S>(session, source, cycles)?;
            Ok(true)
        }

        #[cfg(test)]
        fn residual_atoms() -> Option<&'static [Atom]> {
            Some(<$residual as CachedBundle>::ATOMS)
        }
    };
}

macro_rules! cached_bundle {
    ($bundle:ty $(where residual = $residual:ty)? {
        $($field:ident : $atom_type:ty = $atom:expr),+ $(,)?
    }) => {
        impl CachedBundle for $bundle {
            const ATOMS: &'static [Atom] = &[$($atom),+];

            cached_residual!($($residual)?);

            #[cfg(test)]
            fn atom_types() -> Vec<(Atom, std::any::TypeId)> {
                vec![$(($atom, std::any::TypeId::of::<$atom_type>())),+]
            }

            fn store(rows: &[Self], columns: &mut TraceColumns) {
                $( let mut $field: Vec<$atom_type> = Vec::with_capacity(rows.len()); )+
                for row in rows {
                    $( $field.push(row.$field); )+
                }
                $( columns.insert($atom, $field); )+
            }

            fn restore(columns: &TraceColumns, cycles: usize) -> Option<Vec<Self>> {
                $(
                    let $field = columns.column::<$atom_type>($atom)?;
                    if $field.len() != cycles {
                        return None;
                    }
                )+
                Some(
                    (0..cycles)
                        .map(|cycle| Self { $($field: $field[cycle]),+ })
                        .collect(),
                )
            }
        }
    };
}

cached_bundle! {
    CommittedColumnsWitness {
        rd_inc: RdInc = Atom::RdInc,
        ram_inc: RamInc = Atom::RamInc,
        lookup_index: LookupIndex = Atom::LookupIndex,
        bytecode_pc: MappedPc = Atom::MappedPc,
        ram_address: RemappedRamAddress = Atom::RemappedRamAddress,
    }
}

cached_bundle! {
    OneHotCycleWitness {
        lookup: LookupIndex = Atom::LookupIndex,
        pc: MappedPc = Atom::MappedPc,
        ram: RemappedRamAddress = Atom::RemappedRamAddress,
    }
}

cached_bundle! {
    BytecodeReadRafCycleWitness {
        pc: MappedPc = Atom::MappedPc,
    }
}

cached_bundle! {
    BytecodeReadRafWitness where residual = NoopRowWitness {
        bytecode_pc: BytecodePc = Atom::BytecodePc,
    }
}

cached_bundle! {
    InstructionReadRafWitness where residual = InstructionReadRafResidualWitness {
        lookup_index: LookupIndex = Atom::LookupIndex,
        table_index: TableIndex = Atom::TableIndex,
        raf_flag: InstructionRafFlag = Atom::InstructionRafFlag,
    }
}

cached_bundle! {
    RamAddressWitness {
        address: RemappedRamAddress = Atom::RemappedRamAddress,
    }
}

cached_bundle! {
    IncClaimReductionWitness {
        ram: RamInc = Atom::RamInc,
        rd: RdInc = Atom::RdInc,
    }
}

cached_bundle! {
    RamValCheckWitness {
        inc: RamInc = Atom::RamInc,
        address: RemappedRamAddress = Atom::RemappedRamAddress,
    }
}

cached_bundle! {
    RamReadWriteWitness {
        address: RemappedRamAddress = Atom::RemappedRamAddress,
        read_value: RamReadValue = Atom::RamReadValue,
        write_value: RamWriteValue = Atom::RamWriteValue,
    }
}

cached_bundle! {
    RamHammingBooleanityWitness {
        weight: RamHammingWeight = Atom::RamHammingWeight,
    }
}

cached_bundle! {
    InstructionRaVirtualizationWitness {
        lookup_index: LookupIndex = Atom::LookupIndex,
    }
}

cached_bundle! {
    InstructionClaimReductionWitness {
        output: LookupOutput = Atom::LookupOutput,
        left_lookup: LeftLookupOperand = Atom::LeftLookupOperand,
        right_lookup: RightLookupOperand = Atom::RightLookupOperand,
        left_input: LeftInstructionInput = Atom::LeftInstructionInput,
        right_input: RightInstructionInput = Atom::RightInstructionInput,
    }
}

cached_bundle! {
    InstructionInputWitness where residual = InstructionInputResidualWitness {
        left_operand_is_rs1: InstructionFlag =
            Atom::InstructionFlag(InstructionFlags::LeftOperandIsRs1Value),
        rs1_value: Rs1Value = Atom::Rs1Value,
        left_operand_is_pc: InstructionFlag =
            Atom::InstructionFlag(InstructionFlags::LeftOperandIsPC),
        unexpanded_pc: UnexpandedPc = Atom::UnexpandedPc,
        right_operand_is_rs2: InstructionFlag =
            Atom::InstructionFlag(InstructionFlags::RightOperandIsRs2Value),
        rs2_value: Rs2Value = Atom::Rs2Value,
        right_operand_is_imm: InstructionFlag =
            Atom::InstructionFlag(InstructionFlags::RightOperandIsImm),
        imm: Imm = Atom::Imm,
    }
}

cached_bundle! {
    RegistersClaimReductionWitness {
        rd_write: RdWriteValue = Atom::RdWriteValue,
        rs1: Rs1Value = Atom::Rs1Value,
        rs2: Rs2Value = Atom::Rs2Value,
    }
}

cached_bundle! {
    RegistersReadWriteWitness where residual = RegisterAddressResidualWitness {
        rs1_address: Rs1Address = Atom::Rs1Address,
        rs1_value: Rs1Value = Atom::Rs1Value,
        rs2_address: Rs2Address = Atom::Rs2Address,
        rs2_value: Rs2Value = Atom::Rs2Value,
        rd_address: RdAddress = Atom::RdAddress,
        rd_pre_value: RdPreValue = Atom::RdPreValue,
        rd_post_value: RdWriteValue = Atom::RdWriteValue,
    }
}

cached_bundle! {
    RegistersValEvaluationWitness {
        inc: RdInc = Atom::RdInc,
        address: RdAddress = Atom::RdAddress,
    }
}

cached_bundle! {
    SpartanOuterWitness {
        left_instruction_input: LeftInstructionInput = Atom::LeftInstructionInput,
        right_instruction_input: RightInstructionInput = Atom::RightInstructionInput,
        product: Product = Atom::Product,
        imm: Imm = Atom::Imm,
        right_lookup_operand: RightLookupOperand = Atom::RightLookupOperand,
        pc: Pc = Atom::Pc,
        unexpanded_pc: UnexpandedPc = Atom::UnexpandedPc,
        ram_address: RamAddress = Atom::RamAddress,
        rs1_value: Rs1Value = Atom::Rs1Value,
        rs2_value: Rs2Value = Atom::Rs2Value,
        rd_write_value: RdWriteValue = Atom::RdWriteValue,
        ram_read_value: RamReadValue = Atom::RamReadValue,
        ram_write_value: RamWriteValue = Atom::RamWriteValue,
        left_lookup_operand: LeftLookupOperand = Atom::LeftLookupOperand,
        lookup_output: LookupOutput = Atom::LookupOutput,
        should_branch: ShouldBranch = Atom::ShouldBranch,
        should_jump: ShouldJump = Atom::ShouldJump,
        add_operands: OpFlag = Atom::OpFlag(CircuitFlags::AddOperands),
        subtract_operands: OpFlag = Atom::OpFlag(CircuitFlags::SubtractOperands),
        multiply_operands: OpFlag = Atom::OpFlag(CircuitFlags::MultiplyOperands),
        load: OpFlag = Atom::OpFlag(CircuitFlags::Load),
        store: OpFlag = Atom::OpFlag(CircuitFlags::Store),
        jump: OpFlag = Atom::OpFlag(CircuitFlags::Jump),
        write_lookup_output_to_rd: OpFlag = Atom::OpFlag(CircuitFlags::WriteLookupOutputToRD),
        virtual_instruction: OpFlag = Atom::OpFlag(CircuitFlags::VirtualInstruction),
        assert_flag: OpFlag = Atom::OpFlag(CircuitFlags::Assert),
        do_not_update_unexpanded_pc: OpFlag = Atom::OpFlag(CircuitFlags::DoNotUpdateUnexpandedPC),
        advice: OpFlag = Atom::OpFlag(CircuitFlags::Advice),
        is_compressed: OpFlag = Atom::OpFlag(CircuitFlags::IsCompressed),
        is_first_in_sequence: OpFlag = Atom::OpFlag(CircuitFlags::IsFirstInSequence),
        is_last_in_sequence: OpFlag = Atom::OpFlag(CircuitFlags::IsLastInSequence),
    }
}

cached_bundle! {
    SpartanProductWitness where residual = ProductResidualWitness {
        left_instruction_input: LeftInstructionInput = Atom::LeftInstructionInput,
        right_instruction_input: RightInstructionInput = Atom::RightInstructionInput,
        lookup_output: LookupOutput = Atom::LookupOutput,
        jump: OpFlag = Atom::OpFlag(CircuitFlags::Jump),
        write_lookup_output_to_rd: OpFlag = Atom::OpFlag(CircuitFlags::WriteLookupOutputToRD),
        virtual_instruction: OpFlag = Atom::OpFlag(CircuitFlags::VirtualInstruction),
        branch: InstructionFlag = Atom::InstructionFlag(InstructionFlags::Branch),
        next_is_noop: NextIsNoop = Atom::NextIsNoop,
    }
}

cached_bundle! {
    SpartanShiftWitness where residual = ShiftResidualWitness {
        unexpanded_pc: UnexpandedPc = Atom::UnexpandedPc,
        pc: Pc = Atom::Pc,
        virtual_instruction: OpFlag = Atom::OpFlag(CircuitFlags::VirtualInstruction),
        is_first_in_sequence: OpFlag = Atom::OpFlag(CircuitFlags::IsFirstInSequence),
        is_noop: InstructionFlag = Atom::InstructionFlag(InstructionFlags::IsNoop),
    }
}

cached_bundle! {
    NoopRowWitness {
        noop: NoopRow = Atom::NoopRow,
    }
}

cached_bundle! {
    ShiftResidualWitness {
        is_noop: InstructionFlag = Atom::InstructionFlag(InstructionFlags::IsNoop),
    }
}

cached_bundle! {
    ProductResidualWitness {
        branch: InstructionFlag = Atom::InstructionFlag(InstructionFlags::Branch),
        next_is_noop: NextIsNoop = Atom::NextIsNoop,
    }
}

cached_bundle! {
    InstructionInputResidualWitness {
        left_operand_is_rs1: InstructionFlag =
            Atom::InstructionFlag(InstructionFlags::LeftOperandIsRs1Value),
        left_operand_is_pc: InstructionFlag =
            Atom::InstructionFlag(InstructionFlags::LeftOperandIsPC),
        right_operand_is_rs2: InstructionFlag =
            Atom::InstructionFlag(InstructionFlags::RightOperandIsRs2Value),
        right_operand_is_imm: InstructionFlag =
            Atom::InstructionFlag(InstructionFlags::RightOperandIsImm),
    }
}

cached_bundle! {
    RegisterAddressResidualWitness {
        rs1_address: Rs1Address = Atom::Rs1Address,
        rs2_address: Rs2Address = Atom::Rs2Address,
        rd_address: RdAddress = Atom::RdAddress,
        rd_pre_value: RdPreValue = Atom::RdPreValue,
    }
}

cached_bundle! {
    InstructionReadRafResidualWitness {
        table_index: TableIndex = Atom::TableIndex,
        raf_flag: InstructionRafFlag = Atom::InstructionRafFlag,
    }
}

pub(crate) fn witness_identity<T: ?Sized>(witness: &T) -> usize {
    std::ptr::from_ref(witness).cast::<()>() as usize
}

fn columns_for<'a, S: RowSource + ?Sized>(
    session: &'a mut ProofSession,
    source: &S,
    cycles: usize,
) -> &'a mut TraceColumns {
    let identity = witness_identity(source);
    let columns = session.state_or_insert_with(TraceColumns::default);
    if !columns.serves(identity, cycles) {
        columns.reset(identity, cycles);
    }
    columns
}

pub(crate) fn cached_columns(
    session: &ProofSession,
    identity: usize,
    cycles: usize,
) -> Option<&TraceColumns> {
    session
        .state::<TraceColumns>()
        .filter(|columns| columns.serves(identity, cycles))
}

pub(crate) fn cached_bundles<B, S>(
    session: &mut ProofSession,
    source: &S,
    cycles: usize,
) -> Result<Vec<B>, WitnessError>
where
    B: CachedBundle,
    S: RowSource + ?Sized,
{
    let restored = tracing::info_span!(
        "cuda_columns_restore",
        bundle = core::any::type_name::<B>(),
        atoms = B::ATOMS.len()
    )
    .in_scope(|| B::restore(columns_for(session, source, cycles), cycles));
    if let Some(rows) = restored {
        return Ok(rows);
    }
    if tracing::info_span!(
        "cuda_columns_residual",
        bundle = core::any::type_name::<B>()
    )
    .in_scope(|| B::walk_residual(session, source, cycles))?
    {
        let restored = B::restore(columns_for(session, source, cycles), cycles);
        if let Some(rows) = restored {
            return Ok(rows);
        }
    }
    let rows = collect_bundles::<B>(source, cycles)?;
    tracing::info_span!(
        "cuda_columns_store",
        bundle = core::any::type_name::<B>(),
        atoms = B::ATOMS.len()
    )
    .in_scope(|| B::store(&rows, columns_for(session, source, cycles)));
    Ok(rows)
}

pub(crate) fn store_columns<B, S>(session: &mut ProofSession, source: &S, cycles: usize, rows: &[B])
where
    B: CachedBundle,
    S: RowSource + ?Sized,
{
    if rows.len() != cycles {
        return;
    }
    B::store(rows, columns_for(session, source, cycles));
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test module: fixture and walk errors fail loudly"
)]
mod tests {
    use std::cell::Cell;
    use std::ops::Range;

    use jolt_claims::protocols::jolt::{JoltPolynomialId, JoltVirtualPolynomial};
    use jolt_witness::ChunkVisitor;
    use proptest::prelude::*;

    use super::*;
    use crate::cuda::common::testing::with_r1cs_witness;

    const LOG_T: usize = 8;

    const RAM_K: usize = 1 << 10;

    const fn one_hot() -> jolt_claims::protocols::jolt::JoltOneHotConfig {
        jolt_claims::protocols::jolt::JoltOneHotConfig {
            log_k_chunk: 8,
            lookups_ra_virtual_log_k_chunk: 32,
        }
    }

    struct CountingSource<'a, S: ?Sized> {
        inner: &'a S,
        walks: Cell<usize>,
    }

    impl<'a, S: RowSource + ?Sized> CountingSource<'a, S> {
        fn new(inner: &'a S) -> Self {
            Self {
                inner,
                walks: Cell::new(0),
            }
        }

        fn walks(&self) -> usize {
            self.walks.get()
        }
    }

    impl<S: RowSource + ?Sized> RowSource for CountingSource<'_, S> {
        fn visit_chunks(
            &self,
            range: Range<usize>,
            chunk_size: usize,
            visitor: &mut ChunkVisitor<'_>,
        ) -> Result<(), WitnessError> {
            self.walks.set(self.walks.get() + 1);
            self.inner.visit_chunks(range, chunk_size, visitor)
        }
    }

    fn expect_same<B: CachedBundle + std::fmt::Debug + PartialEq, S: RowSource + ?Sized>(
        session: &mut ProofSession,
        source: &S,
        cycles: usize,
        expected: &[B],
    ) {
        let got = cached_bundles::<B, S>(session, source, cycles).expect("cached bundles");
        assert_eq!(
            &got[..],
            expected,
            "{} diverged",
            core::any::type_name::<B>()
        );
    }

    macro_rules! each_bundle {
        ($body:ident) => {
            $body!(CommittedColumnsWitness);
            $body!(OneHotCycleWitness);
            $body!(BytecodeReadRafCycleWitness);
            $body!(BytecodeReadRafWitness);
            $body!(NoopRowWitness);
            $body!(ShiftResidualWitness);
            $body!(ProductResidualWitness);
            $body!(InstructionInputResidualWitness);
            $body!(RegisterAddressResidualWitness);
            $body!(InstructionReadRafResidualWitness);
            $body!(InstructionReadRafWitness);
            $body!(RamAddressWitness);
            $body!(IncClaimReductionWitness);
            $body!(RamValCheckWitness);
            $body!(RamReadWriteWitness);
            $body!(RamHammingBooleanityWitness);
            $body!(InstructionRaVirtualizationWitness);
            $body!(InstructionClaimReductionWitness);
            $body!(InstructionInputWitness);
            $body!(RegistersClaimReductionWitness);
            $body!(RegistersReadWriteWitness);
            $body!(RegistersValEvaluationWitness);
            $body!(SpartanOuterWitness);
            $body!(SpartanProductWitness);
            $body!(SpartanShiftWitness);
        };
    }

    macro_rules! each_residual_parent {
        ($body:ident) => {
            $body!(SpartanShiftWitness);
            $body!(SpartanProductWitness);
            $body!(InstructionInputWitness);
            $body!(RegistersReadWriteWitness);
            $body!(InstructionReadRafWitness);
            $body!(BytecodeReadRafWitness);
        };
    }

    fn warm_the_early_stages<S: RowSource + ?Sized>(
        session: &mut ProofSession,
        source: &S,
        cycles: usize,
    ) {
        let _ = cached_bundles::<CommittedColumnsWitness, S>(session, source, cycles)
            .expect("the fixture serves the committed columns bundle");
        let _ = cached_bundles::<SpartanOuterWitness, S>(session, source, cycles)
            .expect("the fixture serves the spartan outer bundle");
    }

    #[test]
    fn residual_walk_serves_every_registered_parent() {
        let cycles = 1usize << LOG_T;
        with_r1cs_witness(LOG_T, RAM_K, one_hot(), 43, |witness| {
            macro_rules! check {
                ($bundle:ty) => {{
                    let name = core::any::type_name::<$bundle>();
                    let source = CountingSource::new(witness);
                    let mut session = ProofSession::default();
                    warm_the_early_stages(&mut session, &source, cycles);
                    let warm = source.walks();

                    let columns = columns_for(&mut session, &source, cycles);
                    assert!(
                        <$bundle as CachedBundle>::restore(columns, cycles).is_none(),
                        "{name}: already fully cached after stage 0 and spartan outer, so its \
                         residual is dead code",
                    );
                    assert!(
                        <$bundle as CachedBundle>::walk_residual(&mut session, &source, cycles)
                            .expect("residual walk"),
                        "{name}: has no residual, so it does not belong in this list",
                    );
                    assert_eq!(
                        source.walks(),
                        warm + 1,
                        "{name}: the residual walk did not walk exactly once",
                    );
                    let columns = columns_for(&mut session, &source, cycles);
                    assert!(
                        <$bundle as CachedBundle>::restore(columns, cycles).is_some(),
                        "{name}: the residual walk did not complete the bundle, so the full \
                         bundle still has to be walked and the residual is pure overhead",
                    );

                    let expected = collect_bundles::<$bundle>(witness, cycles)
                        .expect("the fixture serves every bundle field");
                    let got = cached_bundles::<$bundle, _>(&mut session, &source, cycles)
                        .expect("cached bundles");
                    assert_eq!(got, expected, "{name}: the residual-served rows diverged");
                    assert_eq!(
                        source.walks(),
                        warm + 1,
                        "{name}: serving the full bundle walked again after the residual",
                    );
                }};
            }
            each_residual_parent!(check);
        });
    }

    #[test]
    fn fixture_residual_atoms_are_not_the_whole_parent() {
        macro_rules! check {
            ($bundle:ty) => {{
                let name = core::any::type_name::<$bundle>();
                let parent = <$bundle as CachedBundle>::ATOMS;
                let residual = <$bundle as CachedBundle>::residual_atoms()
                    .expect("every parent in this list declares a residual");
                assert!(
                    residual.len() <= parent.len(),
                    "{name}: the residual is wider than the parent",
                );
                assert_ne!(
                    residual, parent,
                    "{name}: the residual is the parent's whole atom set, so walking it saves \
                     nothing",
                );
            }};
        }
        each_residual_parent!(check);
    }

    #[test]
    fn fixture_bundle_atoms_are_unique() {
        macro_rules! check {
            ($bundle:ty) => {{
                let atoms = <$bundle as CachedBundle>::ATOMS;
                for (index, atom) in atoms.iter().enumerate() {
                    assert!(
                        !atoms[..index].contains(atom),
                        "{}: {atom:?} is claimed by two fields, so one column would overwrite \
                         the other",
                        core::any::type_name::<$bundle>(),
                    );
                }
            }};
        }
        each_bundle!(check);
    }

    const fn is_flag(atom: Atom) -> bool {
        matches!(atom, Atom::OpFlag(_) | Atom::InstructionFlag(_))
    }

    #[test]
    fn bundle_same_typed_atoms_are_flags_only() {
        macro_rules! check {
            ($bundle:ty) => {{
                let pairs = <$bundle as CachedBundle>::atom_types();
                for (index, (atom, id)) in pairs.iter().enumerate() {
                    for (other, previous) in &pairs[..index] {
                        if id != previous {
                            continue;
                        }
                        assert!(
                            is_flag(*atom) && is_flag(*other),
                            "{}: {atom:?} and {other:?} share a Rust type but are not flags, so \
                             transposing them would serve the wrong column silently instead of \
                             failing the downcast",
                            core::any::type_name::<$bundle>(),
                        );
                    }
                }
            }};
        }
        each_bundle!(check);
    }

    #[test]
    fn bundle_flag_atoms_match_the_derived_openings() {
        macro_rules! check {
            ($bundle:ty) => {{
                let declared: Vec<Atom> = <$bundle as CachedBundle>::ATOMS
                    .iter()
                    .copied()
                    .filter(|atom| is_flag(*atom))
                    .collect();
                let annotated: Vec<Atom> = <$bundle as WitnessBundle>::annotated_ids()
                    .into_iter()
                    .filter_map(|id| match id {
                        JoltPolynomialId::Virtual(JoltVirtualPolynomial::OpFlags(flag)) => {
                            Some(Atom::OpFlag(flag))
                        }
                        JoltPolynomialId::Virtual(JoltVirtualPolynomial::InstructionFlags(
                            flag,
                        )) => Some(Atom::InstructionFlag(flag)),
                        _ => None,
                    })
                    .collect();
                assert_eq!(
                    declared,
                    annotated,
                    "{}: the cached flag atoms disagree with the bundle's own `#[opening(..)]` \
                     annotations, so a flag column is bound to the wrong field",
                    core::any::type_name::<$bundle>(),
                );
            }};
        }
        each_bundle!(check);
    }

    #[test]
    fn fixture_flag_columns_that_coincide_are_covered_structurally() {
        with_r1cs_witness(LOG_T, RAM_K, one_hot(), 19, |witness| {
            let cycles = 1usize << LOG_T;
            let outer = collect_bundles::<SpartanOuterWitness>(witness, cycles)
                .expect("the fixture serves every bundle field");
            let assert_flag: Vec<bool> = outer.iter().map(|row| row.assert_flag.0).collect();
            let first_in_sequence: Vec<bool> =
                outer.iter().map(|row| row.is_first_in_sequence.0).collect();
            assert_eq!(
                assert_flag, first_in_sequence,
                "these two flags now differ in the fixture, so the equivalence test discriminates \
                 them and this gate can go",
            );
            assert!(
                <SpartanOuterWitness as CachedBundle>::ATOMS
                    .iter()
                    .filter(|atom| is_flag(**atom))
                    .count()
                    > 1,
                "no flag atoms left to transpose",
            );
        });
    }

    #[test]
    fn fixture_committed_atoms_have_several_consumers() {
        for atom in <CommittedColumnsWitness as CachedBundle>::ATOMS {
            let mut consumers = 0;
            macro_rules! count {
                ($bundle:ty) => {
                    if <$bundle as CachedBundle>::ATOMS.contains(atom) {
                        consumers += 1;
                    }
                };
            }
            each_bundle!(count);
            assert!(
                consumers > 1,
                "{atom:?} has a single consumer, so the sharing test is vacuous for it",
            );
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(2))]

        #[test]
        fn cached_bundles_matches_collect_bundles(seed in any::<u64>()) {
            let cycles = 1usize << LOG_T;
            with_r1cs_witness(LOG_T, RAM_K, one_hot(), seed, |witness| {
                let mut oracles = 0;
                macro_rules! oracle {
                    ($bundle:ty) => {{
                        let rows = collect_bundles::<$bundle>(witness, cycles)
                            .expect("the fixture serves every bundle field");
                        assert_eq!(rows.len(), cycles);
                        oracles += 1;
                    }};
                }
                each_bundle!(oracle);
                assert_eq!(
                    oracles, 25,
                    "the oracle pass must cover every registered bundle before the stub panics",
                );

                macro_rules! check {
                    ($bundle:ty) => {{
                        let expected = collect_bundles::<$bundle>(witness, cycles)
                            .expect("the fixture serves every bundle field");
                        let mut session = ProofSession::default();
                        expect_same::<$bundle, _>(&mut session, witness, cycles, &expected);
                    }};
                }
                each_bundle!(check);
            });
        }
    }

    #[test]
    fn cached_bundles_shares_columns_across_bundles() {
        let cycles = 1usize << LOG_T;
        with_r1cs_witness(LOG_T, RAM_K, one_hot(), 23, |witness| {
            let expected_commit = collect_bundles::<CommittedColumnsWitness>(witness, cycles)
                .expect("the fixture serves every bundle field");
            let expected_one_hot = collect_bundles::<OneHotCycleWitness>(witness, cycles)
                .expect("the fixture serves every bundle field");
            let expected_bytecode = collect_bundles::<BytecodeReadRafCycleWitness>(witness, cycles)
                .expect("the fixture serves every bundle field");
            let expected_ram = collect_bundles::<RamAddressWitness>(witness, cycles)
                .expect("the fixture serves every bundle field");
            let expected_inc = collect_bundles::<IncClaimReductionWitness>(witness, cycles)
                .expect("the fixture serves every bundle field");
            let expected_val_check = collect_bundles::<RamValCheckWitness>(witness, cycles)
                .expect("the fixture serves every bundle field");
            let expected_ra_virtual =
                collect_bundles::<InstructionRaVirtualizationWitness>(witness, cycles)
                    .expect("the fixture serves every bundle field");

            let source = CountingSource::new(witness);
            let mut session = ProofSession::default();
            expect_same(&mut session, &source, cycles, &expected_commit);
            let warm = source.walks();
            assert_eq!(warm, 1, "the first bundle must walk exactly once");

            expect_same(&mut session, &source, cycles, &expected_one_hot);
            expect_same(&mut session, &source, cycles, &expected_bytecode);
            expect_same(&mut session, &source, cycles, &expected_ram);
            expect_same(&mut session, &source, cycles, &expected_inc);
            expect_same(&mut session, &source, cycles, &expected_val_check);
            expect_same(&mut session, &source, cycles, &expected_ra_virtual);
            assert_eq!(
                source.walks(),
                warm,
                "a bundle whose atoms are all cached walked the trace again",
            );
        });
    }

    #[test]
    fn cached_bundles_derives_pc_from_mapped_pc() {
        let cycles = 1usize << LOG_T;
        with_r1cs_witness(LOG_T, RAM_K, one_hot(), 29, |witness| {
            let expected: Vec<Pc> = collect_bundles::<SpartanShiftWitness>(witness, cycles)
                .expect("the fixture serves every bundle field")
                .iter()
                .map(|row| row.pc)
                .collect();

            let source = CountingSource::new(witness);
            let mut session = ProofSession::default();
            let commit = collect_bundles::<CommittedColumnsWitness>(witness, cycles)
                .expect("the fixture serves every bundle field");
            expect_same(&mut session, &source, cycles, &commit);

            let columns = session
                .state::<TraceColumns>()
                .expect("the first bundle parked its columns");
            let got = columns
                .column::<Pc>(Atom::Pc)
                .expect("`Pc` derives from the cached `MappedPc` column");
            assert_eq!(got, &expected[..], "the derived `Pc` column diverged");
        });
    }

    #[test]
    fn cached_bundles_refreshes_for_a_different_source() {
        let cycles = 1usize << LOG_T;
        with_r1cs_witness(LOG_T, RAM_K, one_hot(), 31, |first| {
            with_r1cs_witness(LOG_T, RAM_K, one_hot(), 37, |second| {
                let expected_first = collect_bundles::<OneHotCycleWitness>(first, cycles)
                    .expect("the fixture serves every bundle field");
                let expected_second = collect_bundles::<OneHotCycleWitness>(second, cycles)
                    .expect("the fixture serves every bundle field");
                assert_ne!(
                    expected_first, expected_second,
                    "the two fixtures agree, so a stale column could not be observed",
                );

                let mut session = ProofSession::default();
                expect_same(&mut session, first, cycles, &expected_first);
                expect_same(&mut session, second, cycles, &expected_second);
            });
        });
    }
}
