//! Canonical compact instruction facts and their proof-owned shared carry.
use std::sync::{Arc, Weak};

use jolt_field::JoltField;
use jolt_lookup_tables::{LookupTableKind, XLEN as RISCV_XLEN};
#[cfg(feature = "parallel")]
use jolt_utils::par_collect_windows;
#[cfg(feature = "akita")]
use jolt_witness::witnesses::FusedInc;
use jolt_witness::witnesses::{
    BytecodePc, InstructionRafFlag, LookupIndex, RemappedRamAddress, TableIndex,
};
use jolt_witness::{
    stream_witnesses, JoltWitnessPlane, RandomAccessRows, StreamConsumer, WitnessBundle,
    WitnessError,
};

use crate::constant_suffix::ConstantSuffix;
use crate::{KernelError, ProofSession};

pub(crate) fn collect_par_map<B: WitnessBundle, V: Copy + Send>(
    access: &RandomAccessRows,
    cycles: usize,
    pack: impl Fn(B) -> V + Send + Sync,
) -> Result<Vec<V>, WitnessError> {
    let window = |index| access.window::<B>(index).map(&pack);
    #[cfg(feature = "parallel")]
    return par_collect_windows(cycles, window);
    #[cfg(not(feature = "parallel"))]
    (0..cycles).map(window).collect()
}

const _: () = assert!(
    LookupTableKind::<RISCV_XLEN>::COUNT < u8::MAX as usize,
    "InstructionCycleRow packs lookup table indices as u8"
);

/// One packed per-cycle row: the stage-5 facts plus the bytecode/RAM and
/// packed fused-inc sources used by later one-hot kernels. The lookup index
/// is split into native limbs and the PC/table/flags share one word, keeping
/// the retained row at 40 bytes in Akita mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
pub(crate) struct InstructionCycleRow {
    lookup_index_lo: u64,
    lookup_index_hi: u64,
    ram_address_plus_one: u64,
    #[cfg(feature = "akita")]
    fused_inc_magnitude: u64,
    packed_pc_and_flags: u64,
}

const PACKED_PC_BITS: u32 = 56;
const PACKED_TABLE_BITS: u32 = 6;
const PACKED_PC_MASK: u64 = (1 << PACKED_PC_BITS) - 1;
const PACKED_TABLE_MASK: u64 = (1 << PACKED_TABLE_BITS) - 1;
const PACKED_TABLE_SHIFT: u32 = PACKED_PC_BITS;
const PACKED_RAF_SHIFT: u32 = PACKED_TABLE_SHIFT + PACKED_TABLE_BITS;
#[cfg(feature = "akita")]
const PACKED_INC_SIGN_SHIFT: u32 = PACKED_RAF_SHIFT + 1;

const _: () = assert!(LookupTableKind::<RISCV_XLEN>::COUNT < 1 << PACKED_TABLE_BITS);

impl InstructionCycleRow {
    pub(crate) fn new(
        lookup_index: u128,
        table_index: Option<usize>,
        raf_flag: bool,
        bytecode_pc: usize,
        remapped_ram_address: Option<u64>,
        #[cfg(feature = "akita")] fused_inc: FusedInc,
    ) -> Self {
        debug_assert!(table_index.is_none_or(|index| index < u8::MAX as usize));
        #[cfg(feature = "akita")]
        debug_assert!(fused_inc.0.unsigned_abs() <= u64::MAX as u128);
        let pc = bytecode_pc as u64;
        assert!(pc <= PACKED_PC_MASK, "bytecode PC exceeds packed row");
        let table_plus_one = table_index.map_or(0, |index| index as u64 + 1);
        let packed_pc_and_flags =
            pc | (table_plus_one << PACKED_TABLE_SHIFT) | (u64::from(raf_flag) << PACKED_RAF_SHIFT);
        #[cfg(feature = "akita")]
        let packed_pc_and_flags =
            packed_pc_and_flags | (u64::from(fused_inc.0 < 0) << PACKED_INC_SIGN_SHIFT);
        Self {
            lookup_index_lo: lookup_index as u64,
            lookup_index_hi: (lookup_index >> 64) as u64,
            ram_address_plus_one: remapped_ram_address.map_or(0, |address| address + 1),
            #[cfg(feature = "akita")]
            fused_inc_magnitude: fused_inc.0.unsigned_abs() as u64,
            packed_pc_and_flags,
        }
    }

    #[inline(always)]
    pub(crate) fn lookup_index(&self) -> u128 {
        u128::from(self.lookup_index_lo) | (u128::from(self.lookup_index_hi) << 64)
    }

    #[inline]
    pub(crate) fn claim_byte(&self) -> u8 {
        let table = self.table_index().map_or(0, |index| index as u8 + 1);
        table | (u8::from(self.raf_flag()) << 7)
    }

    #[inline]
    pub(crate) fn table_index(&self) -> Option<usize> {
        let table_plus_one =
            ((self.packed_pc_and_flags >> PACKED_TABLE_SHIFT) & PACKED_TABLE_MASK) as usize;
        table_plus_one.checked_sub(1)
    }

    #[inline]
    pub(crate) fn bytecode_pc(&self) -> usize {
        (self.packed_pc_and_flags & PACKED_PC_MASK) as usize
    }

    #[inline]
    pub(crate) fn remapped_ram_address(&self) -> Option<u64> {
        self.ram_address_plus_one.checked_sub(1)
    }

    #[inline]
    pub(crate) fn raf_flag(&self) -> bool {
        self.packed_pc_and_flags & (1 << PACKED_RAF_SHIFT) != 0
    }

    #[cfg(feature = "akita")]
    #[inline]
    pub(crate) fn fused_inc_parts(&self) -> (u64, bool) {
        (
            self.fused_inc_magnitude,
            self.packed_pc_and_flags & (1 << PACKED_INC_SIGN_SHIFT) != 0,
        )
    }

    #[cfg(feature = "akita")]
    #[inline]
    pub(crate) fn fused_inc<F: JoltField>(&self) -> F {
        let (magnitude, negative) = self.fused_inc_parts();
        let magnitude = F::from_u64(magnitude);
        if negative {
            -magnitude
        } else {
            magnitude
        }
    }
}

#[cfg(feature = "akita")]
const _: () = assert!(std::mem::size_of::<InstructionCycleRow>() == 40);
#[cfg(not(feature = "akita"))]
const _: () = assert!(std::mem::size_of::<InstructionCycleRow>() == 32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, WitnessBundle)]
struct WideInstructionRow {
    lookup_index: LookupIndex,
    table_index: TableIndex,
    raf_flag: InstructionRafFlag,
    bytecode_pc: BytecodePc,
    remapped_ram_address: RemappedRamAddress,
    #[cfg(feature = "akita")]
    fused_inc: FusedInc,
}

struct PackRows {
    rows: Vec<InstructionCycleRow>,
}

impl StreamConsumer for PackRows {
    type Witness = WideInstructionRow;

    fn consume(&mut self, chunk: &[WideInstructionRow]) {
        self.rows.extend(chunk.iter().map(|row| {
            InstructionCycleRow::new(
                row.lookup_index.0,
                row.table_index.0,
                row.raf_flag.0,
                row.bytecode_pc.0,
                row.remapped_ram_address.0,
                #[cfg(feature = "akita")]
                row.fused_inc,
            )
        }));
    }
}

impl InstructionCycleRow {
    #[tracing::instrument(skip_all, name = "collect_instruction_cycle_rows", fields(cycles))]
    pub(crate) fn collect<F: JoltField>(
        witness: &dyn JoltWitnessPlane<F>,
        cycles: usize,
    ) -> Result<ConstantSuffix<Self>, KernelError<F>> {
        if let Some(access) = witness.random_access() {
            if cycles <= access.cycles() {
                let retained = access
                    .constant_suffix_start::<WideInstructionRow>(cycles)?
                    .saturating_add(1)
                    .min(cycles);
                let rows = collect_par_map(&access, retained, |row: WideInstructionRow| {
                    Self::new(
                        row.lookup_index.0,
                        row.table_index.0,
                        row.raf_flag.0,
                        row.bytecode_pc.0,
                        row.remapped_ram_address.0,
                        #[cfg(feature = "akita")]
                        row.fused_inc,
                    )
                })?;
                return Ok(ConstantSuffix::new(rows, cycles));
            }
        }
        let mut consumers = (PackRows {
            rows: Vec::with_capacity(cycles),
        },);
        stream_witnesses(witness, 0..cycles, 1 << 12, &mut consumers)?;
        Ok(consumers.0.rows.into())
    }
}

/// The collected stage-5 rows, parked in the [`ProofSession`] for the
/// stage-6b instruction RA virtualization kernel (its committed one-hot
/// chunks are chunks of the same per-cycle lookup index), the stage-6a/6b
/// booleanity and bytecode read-RAF kernels, and the stage-7 Hamming-weight
/// reduction.
///
/// Streaming witnesses keep this carry for the whole proof, since collecting
/// again would replay execution. For random-access witnesses, stage 5 and the
/// stage-6a address preparations park it so the rows outlive their kernels,
/// and the first [`InstructionCycleRow::shared`] call (any optimized stage-6b
/// cycle consumer, else the stage-7 Hamming-weight reduction) replaces it
/// with [`SharedInstructionRowsWeak`]. A registry with none of those
/// consumers optimized keeps the rows alive until
/// [`KernelContext::finish`](crate::KernelContext::finish).
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
pub(crate) struct SharedInstructionRows(pub(crate) Arc<ConstantSuffix<InstructionCycleRow>>);

/// Weak cache for slice-backed rows; it does not keep a collection alive.
/// A queued kernel drop can retain the last strong reference across stages.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
pub(crate) struct SharedInstructionRowsWeak(pub(crate) Weak<ConstantSuffix<InstructionCycleRow>>);

/// An early collection is accepted only for the exact immutable source. The
/// ordinary cross-stage carry takes over after the first sumcheck consumer.
#[cfg(feature = "akita")]
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
struct PreparedInstructionRows {
    rows: Arc<ConstantSuffix<InstructionCycleRow>>,
    #[cfg_attr(feature = "allocative", allocative(skip))]
    source: RandomAccessRows,
}

#[cfg(feature = "akita")]
impl InstructionCycleRow {
    pub(crate) fn prepared<F: JoltField>(
        session: &ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        cycles: usize,
    ) -> Result<Option<Arc<ConstantSuffix<Self>>>, KernelError<F>> {
        let Some(prepared) = session.state::<PreparedInstructionRows>() else {
            return Ok(None);
        };
        if prepared.rows.len() != cycles
            || !witness
                .random_access()
                .is_some_and(|source| source.same_source(&prepared.source))
        {
            return Err(KernelError::InvariantViolation {
                reason: "prepared instruction rows belong to a different witness source",
            });
        }
        Ok(Some(Arc::clone(&prepared.rows)))
    }

    /// Opt-in early collection, charged inside the commitment operation. The
    /// byte limit bounds the actual typed collection, not the padded domain.
    #[cfg(feature = "metal")]
    pub(crate) fn prepare_for_commit<F: JoltField>(
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        cycles: usize,
        byte_limit: u64,
    ) -> Result<(), KernelError<F>> {
        if Self::prepared(session, witness, cycles)?.is_some() {
            return Ok(());
        }
        let Some(source) = witness
            .random_access()
            .filter(|source| cycles <= source.cycles())
        else {
            return Ok(());
        };
        let retained = source
            .constant_suffix_start::<WideInstructionRow>(cycles)?
            .saturating_add(1)
            .min(cycles);
        let Some(bytes) = retained.checked_mul(size_of::<Self>()) else {
            return Ok(());
        };
        if bytes as u128 > u128::from(byte_limit) {
            return Ok(());
        }
        let _span = tracing::info_span!("prepare_shared_instruction_rows", bytes, cycles).entered();
        let rows = Arc::new(Self::collect(witness, cycles)?);
        session.park(PreparedInstructionRows { rows, source });
        Ok(())
    }
}

impl InstructionCycleRow {
    /// Reuse a length-matched carry; weak rows are valid only for random-access
    /// witnesses, whose collection does not replay execution.
    pub(crate) fn cached<F: JoltField>(
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        cycles: usize,
    ) -> Option<Arc<ConstantSuffix<Self>>> {
        if let Some(SharedInstructionRows(rows)) = session.state::<SharedInstructionRows>() {
            if rows.len() == cycles {
                return Some(Arc::clone(rows));
            }
            drop(session.take::<SharedInstructionRows>());
        }
        let _ = witness.random_access()?;
        session
            .state::<SharedInstructionRowsWeak>()
            .and_then(|weak| weak.0.upgrade())
            .filter(|rows| rows.len() == cycles)
    }

    /// Reuse cached rows or collect them, retaining a strong carry for a later
    /// consumer whatever the witness kind.
    pub(crate) fn shared_for_next_consumer<F: JoltField>(
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        cycles: usize,
    ) -> Result<Arc<ConstantSuffix<Self>>, KernelError<F>> {
        #[cfg(feature = "akita")]
        if let Some(rows) = Self::prepared(session, witness, cycles)? {
            drop(session.take::<PreparedInstructionRows>());
            session.park(SharedInstructionRows(Arc::clone(&rows)));
            return Ok(rows);
        }
        let rows = match Self::cached(session, witness, cycles) {
            Some(rows) => rows,
            None => Arc::new(Self::collect(witness, cycles)?),
        };
        session.park(SharedInstructionRows(Arc::clone(&rows)));
        Ok(rows)
    }

    /// Reuse cached rows or collect them, retaining a strong carry only for
    /// witnesses whose next collection would replay execution.
    pub(crate) fn shared<F: JoltField>(
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        cycles: usize,
    ) -> Result<Arc<ConstantSuffix<Self>>, KernelError<F>> {
        let rows = match Self::cached(session, witness, cycles) {
            Some(rows) => rows,
            None => Arc::new(Self::collect(witness, cycles)?),
        };
        if witness.random_access().is_some() {
            drop(session.take::<SharedInstructionRows>());
            session.park(SharedInstructionRowsWeak(Arc::downgrade(&rows)));
        } else {
            session.park(SharedInstructionRows(Arc::clone(&rows)));
        }
        Ok(rows)
    }
}
