//! The streaming RAM family's trace view: [`RamEventLog`] holds every cycle
//! that carries a remapped RAM access or a nonzero `RamInc`, in cycle order,
//! so its size scales with the event count instead of the cycle count. The
//! first streaming RAM kernel of stage 2 collects and parks it; the stage-2
//! read-write and RAF kernels and the stage-4 value check read it, and the
//! stage-5 RA claim reduction, its last consumer, takes it.

use std::sync::Arc;

use jolt_field::JoltField;
use jolt_witness::witnesses::{RamReadValue, RamWriteValue, RemappedRamAddress};
use jolt_witness::{JoltWitnessPlane, WitnessBundle, WitnessError};

use super::ram_trace::val_init_from_first_accesses;
use super::streaming::{exact_task_fill, task_ranges, TaskSlots, VISIT_ROWS};
use super::support::BundleStore;
use crate::{KernelError, ProofSession};

const LABEL: &str = "RAM event log";

#[derive(Clone, Copy, Debug, WitnessBundle)]
struct RamEventBundle {
    address: RemappedRamAddress,
    pre_value: RamReadValue,
    post_value: RamWriteValue,
}

impl RamEventBundle {
    /// `ra` vanishes off remapped accesses and `RamInc` (`post − pre`, see
    /// [`RamEvent::inc`]) off stores, and neither support contains the other:
    /// a store whose address remaps to no word increments `RamInc` without an
    /// access.
    fn is_event(&self) -> bool {
        self.address.0.is_some() || self.pre_value.0 != self.post_value.0
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
}

const _: () = assert!(size_of::<RamEvent>() == 32);

impl RamEvent {
    /// The cycle's `RamInc`: the witness defines it as `post − pre` on stores
    /// and zero elsewhere, and `JoltTraceRow::ram_write_value` equals
    /// `ram_read_value` on every row that is not a store.
    pub fn inc(&self) -> i128 {
        i128::from(self.post_value) - i128::from(self.pre_value)
    }

    fn new(cycle: usize, bundle: &RamEventBundle, ram_k: usize) -> Result<Self, WitnessError> {
        let address = bundle
            .address
            .0
            .map(|address| match u32::try_from(address) {
                Ok(word) if address < ram_k as u64 => Ok(word),
                _ => Err(WitnessError::InvalidWitnessData {
                    label: LABEL,
                    reason: format!(
                        "cycle {cycle} accesses remapped word {address}, beyond ram_K {ram_k}"
                    ),
                }),
            })
            .transpose()?;
        // `RamEventLog::collect` bounds the cycle domain by 2^32.
        Ok(Self {
            cycle: cycle as u32,
            address,
            pre_value: bundle.pre_value.0,
            post_value: bundle.post_value.0,
        })
    }
}

/// Every [`RamEvent`] of the cycle domain in cycle order: the sparse support
/// of the stage-2–5 RAM factors. Clones share one backing.
#[derive(Clone)]
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
pub(crate) struct RamEventLog {
    log_t: usize,
    ram_k: usize,
    events: Arc<Vec<RamEvent>>,
}

impl RamEventLog {
    /// Collect once, then return a handle to the session's log.
    #[expect(
        clippy::expect_used,
        reason = "the log is parked by this function right above the read"
    )]
    pub fn shared<F: JoltField>(
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        log_t: usize,
        ram_k: usize,
    ) -> Result<Self, KernelError<F>> {
        if session.state::<Self>().is_none() {
            session.park(Self::collect(witness, log_t, ram_k)?);
        }
        let log = session.state::<Self>().expect("RAM event log parked above");
        log.check_domain(log_t, ram_k)?;
        Ok(log.clone())
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

    /// One serial visit per [`task_ranges`] task counts its events and a
    /// second writes them into that task's slots of one exactly sized
    /// allocation ([`exact_task_fill`]). Every remapped address is checked
    /// against `ram_K` before the log is shared.
    fn collect<F: JoltField>(
        witness: &dyn JoltWitnessPlane<F>,
        log_t: usize,
        ram_k: usize,
    ) -> Result<Self, KernelError<F>> {
        let rows = BundleStore::<RamEventBundle>::resident(witness, log_t, "RAM event trace rows")?;
        let cycles = 1usize << log_t;
        if u32::try_from(cycles - 1).is_err() {
            return Err(KernelError::Unsupported {
                reason: "the RAM event log packs cycles as u32",
            });
        }
        let events = exact_task_fill(
            &task_ranges(cycles, 1),
            |range| {
                let mut count = 0;
                rows.visit(range.clone(), VISIT_ROWS, |_, bundles| {
                    count += bundles.iter().filter(|bundle| bundle.is_event()).count();
                    Ok(())
                })?;
                Ok(count)
            },
            |range, slots: &mut TaskSlots<'_, RamEvent>| {
                rows.visit(range.clone(), VISIT_ROWS, |first, bundles| {
                    for (cycle, bundle) in (first..).zip(bundles) {
                        if bundle.is_event() {
                            slots.push(RamEvent::new(cycle, bundle, ram_k)?);
                        }
                    }
                    Ok(())
                })
            },
            |_| WitnessError::InvalidWitnessData {
                label: LABEL,
                reason: "trace rows disagree between the count and fill passes".to_owned(),
            },
        )?;
        Ok(Self {
            log_t,
            ram_k,
            events: Arc::new(events),
        })
    }

    /// Reuse by type must not cross trace or address domains.
    fn check_domain<F: JoltField>(&self, log_t: usize, ram_k: usize) -> Result<(), KernelError<F>> {
        for (table, expected, got) in [
            (
                "session-shared RAM event log cycle variables",
                log_t,
                self.log_t,
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

    pub fn val_init<F: JoltField>(&self, val_final: Vec<F>) -> Vec<F> {
        val_init_from_first_accesses(
            val_final,
            self.events.iter().filter_map(|event| {
                event
                    .address
                    .map(|address| (address as usize, event.pre_value))
            }),
        )
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use jolt_claims::protocols::jolt::geometry::ram::ram_inc;
    use jolt_field::{Fr, Ring};
    use jolt_witness::JoltWitnessOracle;

    use super::super::testing::{ram_fixture, FixtureShape, RamOp};
    use super::*;

    /// Accesses and nonzero increments, never only their intersection, in
    /// cycle order; each event's increment is the witness's `RamInc`.
    #[test]
    fn event_log_keeps_accesses_and_increments_in_cycle_order() {
        let shape = FixtureShape {
            log_t: 4,
            ram_k: 16,
        };
        let witness = ram_fixture(
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
        // (cycle, word, pre, post); cycle 7 is the fixture's termination
        // write, cycles 8.. are padding.
        let expected = [
            (0, Some(2), 7, 7),
            (2, None, 3, 8),
            (3, Some(2), 7, 7),
            (5, Some(5), 0, 2),
            (6, None, 9, 1),
            (7, Some(1), 0, 1),
        ];
        let log = RamEventLog::collect::<Fr>(&witness, shape.log_t, shape.ram_k).unwrap();
        let events: Vec<_> = log
            .events()
            .iter()
            .map(|e| (e.cycle, e.address, e.pre_value, e.post_value))
            .collect();
        assert_eq!(events, expected);
        assert_eq!(log.events.capacity(), expected.len());

        let inc: Vec<Fr> = witness.oracle_table(ram_inc().polynomial_id()).unwrap();
        let mut logged = vec![Fr::from_u64(0); 1 << shape.log_t];
        for event in log.events() {
            logged[event.cycle as usize] = Fr::from_i128(event.inc());
        }
        assert_eq!(logged, inc);
    }

    /// A witness without a row handle, a cycle count `usize` cannot hold, an
    /// access beyond `ram_K` and a session log of another domain fail typed.
    #[test]
    fn event_log_failures_are_typed() {
        let shape = FixtureShape { log_t: 3, ram_k: 8 };
        let ops = vec![RamOp::Write { word: 3, post: 4 }, RamOp::Read { word: 3 }];
        let witness = ram_fixture(shape, Vec::new(), ops.clone());
        // A configured domain shorter than the trace leaves no row handle.
        let mut shortened = ram_fixture(shape, Vec::new(), ops);
        shortened.config.log_t = 1;
        assert!(matches!(
            RamEventLog::collect::<Fr>(&shortened, 1, shape.ram_k),
            Err(KernelError::Witness(WitnessError::UnavailableView { .. }))
        ));
        assert!(matches!(
            RamEventLog::collect::<Fr>(&witness, usize::BITS as usize, shape.ram_k),
            Err(KernelError::Witness(WitnessError::InvalidDimensions { .. }))
        ));
        assert!(matches!(
            RamEventLog::collect::<Fr>(&witness, shape.log_t, 2),
            Err(KernelError::Witness(
                WitnessError::InvalidWitnessData { .. }
            ))
        ));

        let mut session = ProofSession::default();
        session.park(RamEventLog::collect::<Fr>(&witness, shape.log_t, shape.ram_k).unwrap());
        assert!(matches!(
            RamEventLog::shared::<Fr>(&mut session, &witness, shape.log_t, 2 * shape.ram_k),
            Err(KernelError::TableSizeMismatch {
                expected: 16,
                got: 8,
                ..
            })
        ));
        // A log over `2^log_t` cycles is not one over `2^(log_t + usize::BITS)`.
        let wide = shape.log_t + usize::BITS as usize;
        assert!(matches!(
            RamEventLog::shared::<Fr>(&mut session, &witness, wide, shape.ram_k),
            Err(KernelError::TableSizeMismatch { got: 3, .. })
        ));
        assert!(matches!(
            RamEventLog::take::<Fr>(&mut session, &witness, shape.log_t + 1, shape.ram_k),
            Err(KernelError::TableSizeMismatch {
                expected: 4,
                got: 3,
                ..
            })
        ));
    }
}
