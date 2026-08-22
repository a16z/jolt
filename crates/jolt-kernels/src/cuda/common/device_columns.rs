#![expect(
    dead_code,
    reason = "`Entry::bytes` is read only by the allocative visitor, and `DeviceBytes` is \
              implemented for every device buffer type the cache can hold"
)]

use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;

use cudarc::driver::CudaSlice;
use jolt_claims::protocols::jolt::JoltCommittedPolynomial;
use jolt_field::Field;
use jolt_witness::JoltWitnessPlane;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::context::{current_device, CudaKernelContext};
use super::device::{DeviceFrVec, LIMBS};
use super::pack::COLD;
use crate::cuda::common::devices::CycleWindow;
use crate::cuda::witness::{session_device_trace, session_device_trace_window};
use crate::{KernelError, ProofSession};

pub(crate) const ANY_SPAN: usize = usize::MAX;

pub(crate) const NO_SPAN: usize = 0;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum DeviceColumn {
    CommittedHot(JoltCommittedPolynomial),
    LookupIndexLimbs,
    MappedPcWord,
    RemappedRamWord,
}

pub(crate) trait DeviceBytes: Any + Send + Sync {
    fn device_bytes(&self) -> usize;
}

impl DeviceBytes for CudaSlice<u8> {
    fn device_bytes(&self) -> usize {
        self.len()
    }
}

impl DeviceBytes for CudaSlice<u32> {
    fn device_bytes(&self) -> usize {
        self.len() * size_of::<u32>()
    }
}

impl DeviceBytes for CudaSlice<u64> {
    fn device_bytes(&self) -> usize {
        self.len() * size_of::<u64>()
    }
}

impl DeviceBytes for DeviceFrVec {
    fn device_bytes(&self) -> usize {
        self.len() * LIMBS * size_of::<u64>()
    }
}

struct Entry {
    value: Arc<dyn Any + Send + Sync>,
    cycles: usize,
    span: usize,
    bytes: usize,
}

#[derive(Default)]
pub(crate) struct DeviceColumns {
    source: usize,
    entries: HashMap<(usize, DeviceColumn), Entry>,
}

impl DeviceColumns {
    fn retarget(&mut self, source: usize) {
        if self.source != source {
            self.source = source;
            self.entries.clear();
        }
    }

    fn get<T: DeviceBytes>(
        &self,
        column: DeviceColumn,
        cycles: usize,
        span: usize,
    ) -> Option<Arc<T>> {
        let entry = self.entries.get(&(current_device(), column))?;
        if entry.cycles != cycles || entry.span > span {
            return None;
        }
        Arc::clone(&entry.value).downcast::<T>().ok()
    }

    fn put<T: DeviceBytes>(
        &mut self,
        column: DeviceColumn,
        cycles: usize,
        span: usize,
        value: Arc<T>,
    ) {
        let bytes = value.device_bytes();
        let _ = self.entries.insert(
            (current_device(), column),
            Entry {
                value,
                cycles,
                span,
                bytes,
            },
        );
    }

    #[cfg(test)]
    pub(crate) fn residency(&self) -> Vec<(DeviceColumn, usize, usize)> {
        let mut resident: Vec<(DeviceColumn, usize, usize)> = self
            .entries
            .iter()
            .filter(|&(&(ordinal, _), _)| ordinal == current_device())
            .map(|(&(_, column), entry)| (column, entry.cycles, entry.span))
            .collect();
        resident
            .sort_unstable_by_key(|&(column, cycles, span)| (format!("{column:?}"), cycles, span));
        resident
    }

    #[cfg(test)]
    pub(crate) fn evict(&mut self, column: DeviceColumn) {
        let _ = self.entries.remove(&(current_device(), column));
    }
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for DeviceColumns {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        for (&(_, column), entry) in &self.entries {
            visitor.visit_simple(allocative::Key::new(column_key(column)), entry.bytes);
        }
        visitor.exit();
    }
}

#[cfg(feature = "allocative")]
const fn column_key(column: DeviceColumn) -> &'static str {
    match column {
        DeviceColumn::CommittedHot(_) => "CommittedHot",
        DeviceColumn::LookupIndexLimbs => "LookupIndexLimbs",
        DeviceColumn::MappedPcWord => "MappedPcWord",
        DeviceColumn::RemappedRamWord => "RemappedRamWord",
    }
}

pub(crate) fn witness_identity<T: ?Sized>(witness: &T) -> usize {
    std::ptr::from_ref(witness).cast::<()>() as usize
}

pub(crate) fn device_columns_for<'a, T: ?Sized>(
    session: &'a mut ProofSession,
    source: &T,
) -> &'a mut DeviceColumns {
    let identity = witness_identity(source);
    let columns = session.state_or_insert_with(DeviceColumns::default);
    columns.retarget(identity);
    columns
}

pub(crate) fn park_device_column<S: ?Sized, T: DeviceBytes>(
    session: &mut ProofSession,
    source: &S,
    column: DeviceColumn,
    cycles: usize,
    span: usize,
    value: Arc<T>,
) {
    device_columns_for(session, source).put(column, cycles, span, value);
}

pub(crate) fn device_column<S, T, E>(
    session: &mut ProofSession,
    source: &S,
    column: DeviceColumn,
    cycles: usize,
    required_span: usize,
    build: impl FnOnce(&mut ProofSession) -> Result<(T, usize), E>,
) -> Result<Arc<T>, E>
where
    S: ?Sized,
    T: DeviceBytes,
{
    if let Some(resident) =
        device_columns_for(session, source).get::<T>(column, cycles, required_span)
    {
        return Ok(resident);
    }
    let (built, span) = tracing::info_span!(
        "cuda_device_column_build",
        column = tracing::field::debug(column),
        cycles
    )
    .in_scope(|| build(session))?;
    let value = Arc::new(built);
    park_device_column(session, source, column, cycles, span, Arc::clone(&value));
    Ok(value)
}

pub(crate) struct DeviceTraceColumns {
    pub(crate) lookup: Arc<CudaSlice<u64>>,
    pub(crate) pc: Arc<CudaSlice<u32>>,
    pub(crate) ram: Arc<CudaSlice<u32>>,
}

fn live_span(words: &[u32]) -> usize {
    #[cfg(feature = "parallel")]
    let highest = words
        .par_iter()
        .filter(|&&word| word != COLD)
        .copied()
        .max();
    #[cfg(not(feature = "parallel"))]
    let highest = words.iter().filter(|&&word| word != COLD).copied().max();
    highest.map_or(NO_SPAN, |highest| highest as usize + 1)
}

pub(crate) fn device_lookup_limbs<F>(
    context: &CudaKernelContext,
    session: &mut ProofSession,
    witness: &dyn JoltWitnessPlane<F>,
    cycles: usize,
) -> Result<Arc<CudaSlice<u64>>, KernelError<F>>
where
    F: Field,
{
    device_column(
        session,
        witness,
        DeviceColumn::LookupIndexLimbs,
        cycles,
        ANY_SPAN,
        |session| {
            let trace = session_device_trace(context, session, witness, cycles)?;
            Ok((trace.lookup_index_limbs()?, NO_SPAN))
        },
    )
}

pub(crate) fn device_pc_words<F>(
    context: &CudaKernelContext,
    session: &mut ProofSession,
    witness: &dyn JoltWitnessPlane<F>,
    cycles: usize,
) -> Result<Arc<CudaSlice<u32>>, KernelError<F>>
where
    F: Field,
{
    device_column(
        session,
        witness,
        DeviceColumn::MappedPcWord,
        cycles,
        ANY_SPAN,
        |session| {
            let trace = session_device_trace(context, session, witness, cycles)?;
            Ok((trace.mapped_pc_words()?, NO_SPAN))
        },
    )
}

pub(crate) fn device_ram_words<F>(
    context: &CudaKernelContext,
    session: &mut ProofSession,
    witness: &dyn JoltWitnessPlane<F>,
    cycles: usize,
    addresses: usize,
) -> Result<Arc<CudaSlice<u32>>, KernelError<F>>
where
    F: Field,
{
    device_column(
        session,
        witness,
        DeviceColumn::RemappedRamWord,
        cycles,
        addresses,
        |session| {
            let trace = session_device_trace(context, session, witness, cycles)?;
            Ok(trace.remapped_ram_words(addresses)?)
        },
    )
}

pub(crate) fn windowed_trace_columns<F>(
    context: &CudaKernelContext,
    session: &mut ProofSession,
    witness: &dyn JoltWitnessPlane<F>,
    cycles: usize,
    window: &CycleWindow,
    families: [usize; 3],
    addresses: usize,
) -> Result<DeviceTraceColumns, KernelError<F>>
where
    F: Field,
{
    let trace =
        session_device_trace_window(context, session, witness, cycles, &window.residency(cycles))?;
    let lookup = if families[0] > 0 {
        Arc::new(trace.lookup_index_limbs()?)
    } else {
        Arc::new(context.alloc_u64(0)?)
    };
    let pc = if families[1] > 0 {
        Arc::new(trace.mapped_pc_words()?)
    } else {
        Arc::new(context.alloc_u32(0)?)
    };
    let ram = if families[2] > 0 {
        Arc::new(trace.remapped_ram_words(addresses)?.0)
    } else {
        Arc::new(context.alloc_u32(0)?)
    };
    Ok(DeviceTraceColumns { lookup, pc, ram })
}

pub(crate) fn device_trace_columns<F>(
    context: &CudaKernelContext,
    session: &mut ProofSession,
    witness: &dyn JoltWitnessPlane<F>,
    cycles: usize,
    families: [usize; 3],
    ram_addresses: usize,
) -> Result<DeviceTraceColumns, KernelError<F>>
where
    F: Field,
{
    let lookup = if families[0] > 0 {
        device_lookup_limbs::<F>(context, session, witness, cycles)?
    } else {
        Arc::new(context.alloc_u64(0)?)
    };
    let pc = if families[1] > 0 {
        device_pc_words::<F>(context, session, witness, cycles)?
    } else {
        Arc::new(context.alloc_u32(0)?)
    };
    let ram = if families[2] > 0 {
        device_ram_words::<F>(context, session, witness, cycles, ram_addresses)?
    } else {
        Arc::new(context.alloc_u32(0)?)
    };
    Ok(DeviceTraceColumns { lookup, pc, ram })
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test module: device operations and fixture errors fail loudly"
)]
mod tests {
    use std::sync::Arc;

    use jolt_claims::protocols::jolt::{JoltCommittedPolynomial, JoltOneHotConfig};
    use jolt_field::Fr;

    use super::{
        device_column, device_columns_for, device_trace_columns, DeviceColumn, ANY_SPAN, NO_SPAN,
    };
    use crate::cuda::common::context::shared_context;
    use crate::cuda::common::error::CudaError;
    use crate::cuda::common::one_hot_witness::{packed_columns, OneHotCycleWitness};
    use crate::cuda::common::pack::COLD;
    use crate::cuda::common::testing::with_r1cs_witness;
    use crate::optimized::support::collect_rows;
    use crate::ProofSession;

    const CYCLES: usize = 64;

    fn hot(cycles: usize, offset: u32) -> Vec<u32> {
        (0..cycles).map(|cycle| cycle as u32 + offset).collect()
    }

    #[test]
    fn a_second_ask_is_served_without_rebuilding() {
        let Some(context) = shared_context() else {
            return;
        };
        let source = [0u8; 8];
        let mut session = ProofSession::default();
        let mut builds = 0usize;
        for _ in 0..3 {
            let column = device_column(
                &mut session,
                &source,
                DeviceColumn::CommittedHot(JoltCommittedPolynomial::RdInc),
                CYCLES,
                ANY_SPAN,
                |_| -> Result<_, CudaError> {
                    builds += 1;
                    Ok((context.upload_u32_slice(&hot(CYCLES, 0))?, NO_SPAN))
                },
            )
            .expect("device column");
            assert_eq!(
                context.download_u32(&column).expect("download"),
                hot(CYCLES, 0),
                "the served column does not hold the built contents",
            );
        }
        assert_eq!(
            builds, 1,
            "the cache rebuilt a column it had already parked, so nothing is actually shared",
        );
    }

    #[test]
    fn a_foreign_witness_clears_the_cabinet() {
        let Some(context) = shared_context() else {
            return;
        };
        let first = [0u8; 8];
        let second = [1u8; 8];
        let key = DeviceColumn::CommittedHot(JoltCommittedPolynomial::RamInc);
        let mut session = ProofSession::default();
        let _ = device_column(
            &mut session,
            &first,
            key,
            CYCLES,
            ANY_SPAN,
            |_| -> Result<_, CudaError> {
                Ok((context.upload_u32_slice(&hot(CYCLES, 0))?, NO_SPAN))
            },
        )
        .expect("first witness");
        let mut builds = 0usize;
        let served = device_column(
            &mut session,
            &second,
            key,
            CYCLES,
            ANY_SPAN,
            |_| -> Result<_, CudaError> {
                builds += 1;
                Ok((context.upload_u32_slice(&hot(CYCLES, 100))?, NO_SPAN))
            },
        )
        .expect("second witness");
        assert_eq!(
            builds, 1,
            "a second witness was served the first witness's device column",
        );
        assert_eq!(
            context.download_u32(&served).expect("download"),
            hot(CYCLES, 100),
            "the rebuilt column does not hold the second witness's contents",
        );
        assert_eq!(
            device_columns_for(&mut session, &second).residency(),
            vec![(key, CYCLES, NO_SPAN)],
            "the first witness's entries survived the retarget",
        );
    }

    #[test]
    fn a_different_cycle_count_misses_without_evicting_its_neighbours() {
        let Some(context) = shared_context() else {
            return;
        };
        let source = [0u8; 8];
        let neighbour = DeviceColumn::MappedPcWord;
        let mut session = ProofSession::default();
        let _ = device_column(
            &mut session,
            &source,
            neighbour,
            CYCLES,
            ANY_SPAN,
            |_| -> Result<_, CudaError> {
                Ok((context.upload_u32_slice(&hot(CYCLES, 0))?, NO_SPAN))
            },
        )
        .expect("neighbour");
        let mut builds = 0usize;
        let _ = device_column(
            &mut session,
            &source,
            DeviceColumn::RemappedRamWord,
            2 * CYCLES,
            ANY_SPAN,
            |_| -> Result<_, CudaError> {
                builds += 1;
                Ok((context.upload_u32_slice(&hot(2 * CYCLES, 0))?, NO_SPAN))
            },
        )
        .expect("wider column");
        assert_eq!(builds, 1, "the wider column was served a shorter entry");
        assert_eq!(
            device_columns_for(&mut session, &source).residency().len(),
            2,
            "a differing cycle count evicted an unrelated column, which would silently \
             re-upload every other consumer's data",
        );
    }

    #[test]
    fn a_span_wider_than_the_asker_allows_is_not_served() {
        let Some(context) = shared_context() else {
            return;
        };
        let source = [0u8; 8];
        let key = DeviceColumn::RemappedRamWord;
        let mut session = ProofSession::default();
        let _ = device_column(
            &mut session,
            &source,
            key,
            CYCLES,
            ANY_SPAN,
            |_| -> Result<_, CudaError> {
                Ok((context.upload_u32_slice(&hot(CYCLES, 0))?, 1 << 12))
            },
        )
        .expect("wide span");
        let mut builds = 0usize;
        let _ = device_column(
            &mut session,
            &source,
            key,
            CYCLES,
            1 << 6,
            |_| -> Result<_, CudaError> {
                builds += 1;
                Ok((context.upload_u32_slice(&hot(CYCLES, 0))?, 1 << 6))
            },
        )
        .expect("narrow span");
        assert_eq!(
            builds, 1,
            "a column whose values reach past the asker's address count was served anyway, \
             so an out-of-range address could reach a kernel",
        );
        let mut builds = 0usize;
        let _ = device_column(
            &mut session,
            &source,
            key,
            CYCLES,
            1 << 20,
            |_| -> Result<_, CudaError> {
                builds += 1;
                Ok((context.upload_u32_slice(&hot(CYCLES, 0))?, 1 << 20))
            },
        )
        .expect("wider span");
        assert_eq!(
            builds, 0,
            "a column narrow enough for the asker was rebuilt, so consumers with different \
             address counts never share",
        );
    }

    #[test]
    fn a_build_error_parks_nothing() {
        let Some(_) = shared_context() else {
            return;
        };
        let source = [0u8; 8];
        let mut session = ProofSession::default();
        let failed = device_column::<_, cudarc::driver::CudaSlice<u32>, CudaError>(
            &mut session,
            &source,
            DeviceColumn::LookupIndexLimbs,
            CYCLES,
            ANY_SPAN,
            |_| {
                Err(CudaError::InvariantViolation {
                    reason: "the fixture build fails",
                })
            },
        );
        assert!(failed.is_err(), "the failing build reported success");
        assert!(
            device_columns_for(&mut session, &source)
                .residency()
                .is_empty(),
            "a failed build parked an entry, which a later consumer would be served",
        );
    }

    const RAM_K: usize = 1 << 10;

    const LOG_T: usize = 8;

    const fn one_hot_config() -> JoltOneHotConfig {
        JoltOneHotConfig {
            log_k_chunk: 8,
            lookups_ra_virtual_log_k_chunk: 32,
        }
    }

    #[test]
    fn device_trace_columns_match_the_packed_host_columns() {
        let Some(context) = shared_context() else {
            return;
        };
        let cycles = 1usize << LOG_T;
        with_r1cs_witness(LOG_T, RAM_K, one_hot_config(), 19, |witness| {
            let rows =
                collect_rows::<Fr, OneHotCycleWitness>(witness, cycles).expect("oracle rows");
            let expected = packed_columns(&rows).expect("host pack");
            assert!(
                expected.pc.iter().any(|&word| word != expected.pc[0])
                    && expected.ram.iter().any(|&word| word != expected.ram[0]),
                "the fixture's PC or RAM column is constant, so a builder that ignored the \
                 trace would pass",
            );
            assert!(
                expected.ram.contains(&COLD),
                "no cycle is RAM-cold, so the cold-cycle encoding is untested here",
            );

            let mut session = ProofSession::default();
            let got = device_trace_columns::<Fr>(
                context,
                &mut session,
                witness,
                cycles,
                [1, 1, 1],
                ANY_SPAN,
            )
            .expect("device trace columns");
            assert_eq!(
                context.download_u64(&got.lookup).expect("download lookup"),
                expected.lookup,
                "the device lookup-index limbs diverge from the host pack",
            );
            assert_eq!(
                context.download_u32(&got.pc).expect("download pc"),
                expected.pc,
                "the device mapped-PC words diverge from the host pack",
            );
            assert_eq!(
                context.download_u32(&got.ram).expect("download ram"),
                expected.ram,
                "the device remapped-RAM words diverge from the host pack",
            );

            let residency = device_columns_for(&mut session, witness).residency();
            assert_eq!(
                residency.len(),
                3,
                "the triple parked {} entries rather than one per column",
                residency.len(),
            );
        });
    }

    #[test]
    fn a_switched_off_family_parks_nothing() {
        let Some(context) = shared_context() else {
            return;
        };
        let cycles = 1usize << LOG_T;
        with_r1cs_witness(LOG_T, RAM_K, one_hot_config(), 23, |witness| {
            let mut session = ProofSession::default();
            let got = device_trace_columns::<Fr>(
                context,
                &mut session,
                witness,
                cycles,
                [0, 1, 0],
                ANY_SPAN,
            )
            .expect("device trace columns");
            assert!(
                got.lookup.is_empty() && got.ram.is_empty(),
                "a family the layout switched off still uploaded its column",
            );
            assert_eq!(
                device_columns_for(&mut session, witness).residency(),
                vec![(DeviceColumn::MappedPcWord, cycles, NO_SPAN)],
                "the switched-off families parked entries, so a later consumer with those \
                 families on would be served a buffer built for a different layout",
            );
        });
    }

    #[test]
    fn fixture_arc_identity_proves_sharing_rather_than_equal_contents() {
        let Some(context) = shared_context() else {
            return;
        };
        let source = [0u8; 8];
        let key = DeviceColumn::LookupIndexLimbs;
        let mut session = ProofSession::default();
        let first = device_column(
            &mut session,
            &source,
            key,
            CYCLES,
            ANY_SPAN,
            |_| -> Result<_, CudaError> {
                Ok((context.upload_u32_slice(&hot(CYCLES, 0))?, NO_SPAN))
            },
        )
        .expect("first");
        let second = device_column(
            &mut session,
            &source,
            key,
            CYCLES,
            ANY_SPAN,
            |_| -> Result<_, CudaError> {
                Ok((context.upload_u32_slice(&hot(CYCLES, 0))?, NO_SPAN))
            },
        )
        .expect("second");
        assert!(
            Arc::ptr_eq(&first, &second),
            "the two asks hold different allocations, so the second one uploaded again even \
             though the contents matched",
        );
    }
}
