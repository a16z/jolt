//! Sparse per-cycle RAM data shared across RAM kernels. [`RamEventLog`]
//! serves the RAM factors of stages 2–5; the stage-6b RA virtualization
//! collects the dense [`SharedRamAddresses`] column.

use core::marker::PhantomData;
use core::mem::MaybeUninit;
use core::ops::Range;
use std::sync::Arc;

use jolt_field::JoltField;
use jolt_utils::unsafe_allocate_zero_vec;
#[cfg(feature = "parallel")]
use jolt_utils::FirstErrorLatch;
use jolt_witness::witnesses::{RamInc, RamReadValue, RamWriteValue, RemappedRamAddress};
#[cfg(feature = "parallel")]
use jolt_witness::TraceRows;
use jolt_witness::{
    stream_witnesses, JoltWitnessPlane, StreamConsumer, WitnessBundle, WitnessError,
};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::support::{task_ranges, VISIT_ROWS};
use crate::{KernelError, ProofSession};

const SPLIT_CHUNK: usize = 1 << 16;

#[cfg(feature = "parallel")]
const PAR_CHUNK: usize = 1 << 14;

/// `addresses` sentinel for cycles with no (remappable) RAM access.
pub(crate) const NO_ACCESS: u32 = u32::MAX;

#[derive(Clone, Copy, Debug, WitnessBundle)]
struct RamAddressBundle {
    address: RemappedRamAddress,
}

#[derive(Clone, Copy, Debug, WitnessBundle)]
struct RamEventBundle {
    address: RemappedRamAddress,
    pre_value: RamReadValue,
    post_value: RamWriteValue,
    inc: RamInc,
}

impl RamEventBundle {
    /// `ra` vanishes off remapped accesses and `RamInc` off stores, and
    /// neither support contains the other: a store to an address that remaps
    /// to no word increments `RamInc` without an access.
    fn is_event(&self) -> bool {
        self.address.0.is_some() || self.inc.0 != 0
    }
}

/// One cycle that carries a remapped RAM access, a nonzero `RamInc`, or both.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
pub(crate) struct RamEvent {
    pub cycle: u32,
    /// The remapped word, below `ram_K`; `None` for a store whose address
    /// remaps to no word.
    pub address: Option<u32>,
    /// `RamReadValue` and `RamWriteValue` of the cycle.
    pub pre_value: u64,
    pub post_value: u64,
    /// `RamInc` of the cycle.
    pub inc: i128,
}

impl RamEvent {
    fn new(cycle: usize, bundle: &RamEventBundle, ram_k: usize) -> Result<Self, WitnessError> {
        let address = bundle
            .address
            .0
            .map(|address| {
                if address < ram_k as u64 {
                    Ok(address as u32)
                } else {
                    Err(WitnessError::InvalidWitnessData {
                        label: EVENT_LOG_LABEL,
                        reason: format!(
                            "cycle {cycle} accesses remapped word {address}, beyond ram_K {ram_k}"
                        ),
                    })
                }
            })
            .transpose()?;
        // `RamEventLog::collect` bounds the cycle domain and `ram_K` by 2^32,
        // so both narrowings are exact.
        Ok(Self {
            cycle: cycle as u32,
            address,
            pre_value: bundle.pre_value.0,
            post_value: bundle.post_value.0,
            inc: bundle.inc.0,
        })
    }
}

const EVENT_LOG_LABEL: &str = "RAM event log";

/// Every [`RamEvent`] of the cycle domain in cycle order: the sparse support
/// of the stage-2–5 RAM factors. Its size scales with the event count, not
/// the cycle count. The first consumer collects and parks it; the stage-5 RA
/// claim reduction, its last consumer, takes it.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
pub(crate) struct RamEventLog {
    cycles: usize,
    ram_k: usize,
    events: Vec<RamEvent>,
}

impl RamEventLog {
    /// Collect once, then return the session's log.
    #[expect(
        clippy::expect_used,
        reason = "the log is parked by this function right above the read"
    )]
    pub fn shared<'s, F: JoltField>(
        session: &'s mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        log_t: usize,
        ram_k: usize,
    ) -> Result<&'s Self, KernelError<F>> {
        if session.state::<Self>().is_none() {
            session.park(Self::collect(witness, log_t, ram_k)?);
        }
        let log = session.state::<Self>().expect("RAM event log parked above");
        log.check_domain(log_t, ram_k)?;
        Ok(log)
    }

    /// Remove the session's log for its last consumer, collecting it when no
    /// earlier consumer parked one.
    pub fn take<F: JoltField>(
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        log_t: usize,
        ram_k: usize,
    ) -> Result<Self, KernelError<F>> {
        let log = match session.take::<Self>() {
            Some(log) => log,
            None => Self::collect(witness, log_t, ram_k)?,
        };
        log.check_domain(log_t, ram_k)?;
        Ok(log)
    }

    pub fn events(&self) -> &[RamEvent] {
        &self.events
    }

    /// One serial visit per [`task_ranges`] task counts its events; a second
    /// writes them into that task's slots of one exactly sized allocation,
    /// so no task holds a vector of its own.
    fn collect<F: JoltField>(
        witness: &dyn JoltWitnessPlane<F>,
        log_t: usize,
        ram_k: usize,
    ) -> Result<Self, KernelError<F>> {
        const LABEL: &str = "RAM event trace rows";
        let rows = witness
            .rows()
            .ok_or(WitnessError::UnavailableView { label: LABEL })?;
        let cycles = 1usize << log_t;
        if rows.cycles() != cycles {
            return Err(WitnessError::InvalidDimensions {
                label: LABEL,
                reason: format!("{} rows for a 2^{log_t}-cycle stage", rows.cycles()),
            }
            .into());
        }
        if u32::try_from(cycles - 1).is_err() || u32::try_from(ram_k - 1).is_err() {
            return Err(KernelError::Unsupported {
                reason: "the RAM event log packs cycles and remapped addresses as u32",
            });
        }
        let ranges = task_ranges(cycles, 1);

        let count = |range: &Range<usize>| -> Result<usize, WitnessError> {
            let mut events = 0;
            rows.visit_bundles(
                range.clone(),
                VISIT_ROWS,
                |_, bundles: &[RamEventBundle]| {
                    events += bundles.iter().filter(|bundle| bundle.is_event()).count();
                    Ok(())
                },
            )?;
            Ok(events)
        };
        #[cfg(feature = "parallel")]
        let counts: Vec<usize> = ranges.par_iter().map(count).collect::<Result<_, _>>()?;
        #[cfg(not(feature = "parallel"))]
        let counts: Vec<usize> = ranges.iter().map(count).collect::<Result<_, _>>()?;

        let total = counts.iter().sum();
        let mut events = Vec::with_capacity(total);
        let mut spare = &mut events.spare_capacity_mut()[..total];
        let task_slots: Vec<&mut [MaybeUninit<RamEvent>]> = counts
            .iter()
            .map(|&count| {
                let (head, tail) = core::mem::take(&mut spare).split_at_mut(count);
                spare = tail;
                head
            })
            .collect();
        let disagreement = || WitnessError::InvalidWitnessData {
            label: EVENT_LOG_LABEL,
            reason: "trace rows disagree between the count and fill passes".to_owned(),
        };
        let fill = |range: &Range<usize>, slots: &mut [MaybeUninit<RamEvent>]| {
            let mut slots = slots.iter_mut();
            rows.visit_bundles(
                range.clone(),
                VISIT_ROWS,
                |first, bundles: &[RamEventBundle]| {
                    for (cycle, bundle) in (first..).zip(bundles) {
                        if bundle.is_event() {
                            let slot = slots.next().ok_or_else(disagreement)?;
                            let _ = slot.write(RamEvent::new(cycle, bundle, ram_k)?);
                        }
                    }
                    Ok(())
                },
            )?;
            if slots.next().is_some() {
                return Err(disagreement());
            }
            Ok(())
        };
        #[cfg(feature = "parallel")]
        ranges
            .par_iter()
            .zip(task_slots)
            .try_for_each(|(range, slots)| fill(range, slots))?;
        #[cfg(not(feature = "parallel"))]
        ranges
            .iter()
            .zip(task_slots)
            .try_for_each(|(range, slots)| fill(range, slots))?;
        // SAFETY: the task slots tile the first `total` spare elements, and
        // every task wrote each of its slots before returning `Ok`.
        unsafe { events.set_len(total) };
        Ok(Self {
            cycles,
            ram_k,
            events,
        })
    }

    /// Reuse by type must not cross trace or address domains.
    fn check_domain<F: JoltField>(&self, log_t: usize, ram_k: usize) -> Result<(), KernelError<F>> {
        for (table, expected, got) in [
            (
                "session-shared RAM event log cycles",
                1usize << log_t,
                self.cycles,
            ),
            ("session-shared RAM event log words", ram_k, self.ram_k),
        ] {
            if expected != got {
                return Err(KernelError::TableSizeMismatch {
                    table: table.to_owned(),
                    expected,
                    got,
                });
            }
        }
        Ok(())
    }

    /// Recover initial values from each address's first access; untouched
    /// addresses retain their final value.
    ///
    /// WARNING: RAM sumchecks enforce the assumed trace/final-image agreement.
    pub fn val_init<F: JoltField>(&self, val_final: Vec<F>) -> Vec<F> {
        let mut val_init = val_final;
        let mut seen = vec![false; val_init.len()];
        for event in &self.events {
            let Some(address) = event.address else {
                continue;
            };
            let address = address as usize;
            if !seen[address] {
                seen[address] = true;
                val_init[address] = F::from_u64(event.pre_value);
            }
        }
        val_init
    }

    /// The dense `RamInc` column over the cycle domain.
    pub fn dense_inc<F: JoltField>(&self) -> Vec<F> {
        let mut inc: Vec<F> = unsafe_allocate_zero_vec(self.cycles);
        for event in &self.events {
            inc[event.cycle as usize] = F::from_i128(event.inc);
        }
        inc
    }
}

/// Pack a remapped address.
fn encode_address<F: JoltField>(address: Option<u64>) -> Result<u32, KernelError<F>> {
    let Some(address) = address else {
        return Ok(NO_ACCESS);
    };
    match u32::try_from(address) {
        Ok(address) if address != NO_ACCESS => Ok(address),
        _ => Err(KernelError::Unsupported {
            reason: "optimized RAM kernels pack remapped addresses as u32 below the u32::MAX \
                     no-access sentinel",
        }),
    }
}

/// Session-shared remapped addresses; [`NO_ACCESS`] marks absent accesses.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
pub(crate) struct SharedRamAddresses(pub(crate) Arc<Vec<u32>>);

impl SharedRamAddresses {
    /// Collect once, then return shared references.
    #[expect(
        clippy::expect_used,
        reason = "the entry is parked by this function right above the read"
    )]
    pub fn shared<F: JoltField>(
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        log_t: usize,
    ) -> Result<Arc<Vec<u32>>, KernelError<F>> {
        if session.state::<Self>().is_none() {
            let cycles = 1usize << log_t;
            let [addresses] =
                collect_split_columns::<F, RamAddressBundle, u32, 1>(witness, cycles, |bundle| {
                    encode_address(bundle.address.0).map(|address| [address])
                })?;
            session.park(Self(Arc::new(addresses)));
        }
        let addresses = Arc::clone(
            &session
                .state::<Self>()
                .expect("RAM address column parked above")
                .0,
        );
        // Reuse by type must not cross trace domains.
        if addresses.len() != 1usize << log_t {
            return Err(KernelError::TableSizeMismatch {
                table: "session-shared RAM address column".to_owned(),
                expected: 1usize << log_t,
                got: addresses.len(),
            });
        }
        Ok(addresses)
    }
}

fn collect_split_columns<F: JoltField, B, T, const N: usize>(
    witness: &dyn JoltWitnessPlane<F>,
    cycles: usize,
    split: impl Fn(&B) -> Result<[T; N], KernelError<F>> + Send + Sync,
) -> Result<[Vec<T>; N], KernelError<F>>
where
    B: WitnessBundle + Copy + Send + Sync,
    T: Copy + Send + Sync,
{
    #[cfg(feature = "parallel")]
    if let Some(access) = witness.rows() {
        if cycles <= access.cycles() {
            return collect_split_columns_par(&access, cycles, &split);
        }
    }
    struct ColumnSplitter<F: JoltField, B, S, T, const N: usize> {
        columns: [Vec<T>; N],
        split: S,
        error: Option<KernelError<F>>,
        _bundle: PhantomData<B>,
    }
    impl<F: JoltField, B, S, T, const N: usize> StreamConsumer for ColumnSplitter<F, B, S, T, N>
    where
        B: WitnessBundle + Copy + Send + Sync,
        S: Fn(&B) -> Result<[T; N], KernelError<F>> + Send + Sync,
        T: Copy + Send + Sync,
    {
        type Witness = B;

        fn consume(&mut self, chunk: &[B]) {
            for bundle in chunk {
                match (self.split)(bundle) {
                    Ok(values) => {
                        for (column, value) in self.columns.iter_mut().zip(values) {
                            column.push(value);
                        }
                    }
                    Err(failure) => {
                        if self.error.is_none() {
                            self.error = Some(failure);
                        }
                    }
                }
            }
        }
    }
    let mut consumers = (ColumnSplitter {
        columns: core::array::from_fn(|_| Vec::with_capacity(cycles)),
        split,
        error: None,
        _bundle: PhantomData::<B>,
    },);
    stream_witnesses(witness, 0..cycles, SPLIT_CHUNK, &mut consumers)?;
    let ColumnSplitter { columns, error, .. } = consumers.0;
    if let Some(failure) = error {
        return Err(failure);
    }
    Ok(columns)
}

#[cfg(feature = "parallel")]
fn collect_split_columns_par<F: JoltField, B, T, const N: usize>(
    access: &TraceRows,
    cycles: usize,
    split: &(impl Fn(&B) -> Result<[T; N], KernelError<F>> + Sync),
) -> Result<[Vec<T>; N], KernelError<F>>
where
    B: WitnessBundle + Copy + Send + Sync,
    T: Copy + Send + Sync,
{
    let mut columns: [Vec<T>; N] = core::array::from_fn(|_| Vec::with_capacity(cycles));
    let chunk_count = cycles.div_ceil(PAR_CHUNK).max(1);
    let error = FirstErrorLatch::new();
    {
        let mut chunk_views: Vec<[&mut [MaybeUninit<T>]; N]> = Vec::with_capacity(chunk_count);
        let mut rests: [&mut [MaybeUninit<T>]; N] = columns
            .each_mut()
            .map(|column| &mut column.spare_capacity_mut()[..cycles]);
        for chunk_index in 0..chunk_count {
            let take = PAR_CHUNK.min(cycles - chunk_index * PAR_CHUNK);
            let mut views: [&mut [MaybeUninit<T>]; N] =
                core::array::from_fn(|_| Default::default());
            for (view, rest) in views.iter_mut().zip(rests.iter_mut()) {
                let (head, tail) = core::mem::take(rest).split_at_mut(take);
                *view = head;
                *rest = tail;
            }
            chunk_views.push(views);
        }
        chunk_views
            .into_par_iter()
            .enumerate()
            .for_each(|(chunk_index, mut views)| {
                let base = chunk_index * PAR_CHUNK;
                let take = PAR_CHUNK.min(cycles - base);
                for offset in 0..take {
                    let values = access
                        .window::<B>(base + offset)
                        .map_err(KernelError::from)
                        .and_then(|bundle| split(&bundle));
                    match values {
                        Ok(values) => {
                            for (view, value) in views.iter_mut().zip(values) {
                                let _ = view[offset].write(value);
                            }
                        }
                        Err(failure) => {
                            error.record(base + offset, failure);
                            return;
                        }
                    }
                }
            });
    }
    if let Some(failure) = error.take() {
        return Err(failure);
    }
    // SAFETY: successful chunks initialized every slot in each disjoint span.
    unsafe {
        for column in &mut columns {
            column.set_len(cycles);
        }
    }
    Ok(columns)
}

pub(crate) fn validate_addresses<F: JoltField>(
    addresses: &[u32],
    ram_k: usize,
) -> Result<(), KernelError<F>> {
    if addresses
        .iter()
        .any(|&address| address != NO_ACCESS && address as usize >= ram_k)
    {
        return Err(KernelError::InvariantViolation {
            reason: "RAM access address remapped beyond ram_K",
        });
    }
    Ok(())
}

#[cfg(test)]
#[expect(clippy::panic, clippy::unwrap_used, reason = "test module")]
mod tests {
    use jolt_field::Fr;
    use jolt_witness::testing::with_sample_backend;

    use super::super::testing::{ram_fixture, FixtureShape, RamOp};
    use super::*;

    #[test]
    fn rejects_session_carry_from_another_cycle_domain() {
        with_sample_backend(|witness| {
            let mut session = ProofSession::default();
            session.park(SharedRamAddresses(Arc::new(vec![NO_ACCESS; 2])));

            let error = match SharedRamAddresses::shared::<Fr>(&mut session, witness, 2) {
                Ok(_) => panic!("wrong-domain RAM addresses were accepted"),
                Err(error) => error,
            };
            assert!(matches!(
                error,
                KernelError::TableSizeMismatch {
                    expected: 4,
                    got: 2,
                    ..
                }
            ));
        });
    }

    /// Accesses and nonzero increments, never only their intersection, in
    /// cycle order from task-partitioned row visits.
    #[test]
    fn event_log_keeps_accesses_and_increments_in_cycle_order() {
        let shape = FixtureShape {
            log_t: 4,
            ram_k: 16,
        };
        let resident = ram_fixture(
            shape,
            vec![7],
            vec![
                RamOp::Read { word: 2 },
                RamOp::None,
                RamOp::UnmappedWrite { pre: 3, post: 8 },
                RamOp::Write { word: 2, post: 7 },
                RamOp::UnmappedWrite { pre: 4, post: 4 },
                RamOp::Write { word: 5, post: 2 },
                RamOp::UnmappedWrite { pre: 9, post: 1 },
            ],
        );
        // (cycle, word, pre, post, inc); cycle 7 is the fixture's
        // termination write, cycles 8.. are padding.
        let expected = [
            (0, Some(2), 7, 7, 0),
            (2, None, 3, 8, 5),
            (3, Some(2), 7, 7, 0),
            (5, Some(5), 0, 2, 2),
            (6, None, 9, 1, -8),
            (7, Some(1), 0, 1, 1),
        ];
        let log = RamEventLog::collect::<Fr>(&resident, shape.log_t, shape.ram_k).unwrap();
        let events: Vec<_> = log
            .events()
            .iter()
            .map(|e| (e.cycle, e.address, e.pre_value, e.post_value, e.inc))
            .collect();
        assert_eq!(events, expected);
        assert_eq!(log.events.capacity(), expected.len());
    }

    /// A witness without a row handle, an access beyond `ram_K` and a session
    /// log of another domain fail typed.
    #[test]
    fn event_log_failures_are_typed() {
        let shape = FixtureShape { log_t: 3, ram_k: 8 };
        let ops = vec![RamOp::Write { word: 3, post: 4 }, RamOp::Read { word: 3 }];
        let resident = ram_fixture(shape, Vec::new(), ops.clone());
        // A configured domain shorter than the trace leaves no valid handle.
        let mut shortened = ram_fixture(shape, Vec::new(), ops);
        shortened.config.log_t = 1;
        assert!(matches!(
            RamEventLog::collect::<Fr>(&shortened, 1, shape.ram_k),
            Err(KernelError::Witness(WitnessError::UnavailableView { .. }))
        ));
        assert!(matches!(
            RamEventLog::collect::<Fr>(&resident, shape.log_t, 2),
            Err(KernelError::Witness(
                WitnessError::InvalidWitnessData { .. }
            ))
        ));

        let mut session = ProofSession::default();
        session.park(RamEventLog::collect::<Fr>(&resident, shape.log_t, shape.ram_k).unwrap());
        assert!(matches!(
            RamEventLog::shared::<Fr>(&mut session, &resident, shape.log_t, 2 * shape.ram_k),
            Err(KernelError::TableSizeMismatch {
                expected: 16,
                got: 8,
                ..
            })
        ));
        assert!(matches!(
            RamEventLog::take::<Fr>(&mut session, &resident, shape.log_t + 1, shape.ram_k),
            Err(KernelError::TableSizeMismatch {
                expected: 16,
                got: 8,
                ..
            })
        ));
    }
}
