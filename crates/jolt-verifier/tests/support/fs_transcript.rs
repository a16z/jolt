#![expect(
    clippy::expect_used,
    clippy::panic,
    reason = "the audit test double must fail loudly on a malformed replay tape"
)]

use std::{any::Any, cell::RefCell, collections::BTreeMap, sync::Arc};

use jolt_field::Field;
use jolt_transcript::{Keccak, Script, ScriptedTranscript, VerifierState};
use jolt_verifier::fs_audit::{self, FsScope};

/// A challenge draw's verifier scope, ordinal within that scope, and length.
///
/// One record is one codec message: `challenge_vector(len)` is one record of
/// `len` values, and `challenge_powers(len)` one record of its single base.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChallengeId {
    pub scope: FsScope,
    pub index: usize,
    pub len: usize,
}

/// One challenge draw and the values it returned.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChallengeRecord<F> {
    pub id: ChallengeId,
    pub values: Vec<F>,
}

/// Challenges produced while verifying one fixture, the proof-of-work check
/// bytes squeezed between them, and the messages the sponge absorbed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChallengeTape<F> {
    pub records: Vec<ChallengeRecord<F>>,
    /// Proof-of-work check bytes, in squeeze order. They are not challenges,
    /// but a replay must freeze them too: otherwise a mutated transcript
    /// fails the proof of work before reaching the binding under test.
    pub checks: Vec<Vec<u8>>,
    /// The encoding of every message absorbed through the transcript, in
    /// order. A replay absorbs these in place of the mutated messages, so the
    /// sponge stays the honest one. A protocol that runs on the sponge
    /// directly, such as Akita's embedded opening, then draws the honest
    /// challenges too.
    pub absorbs: Vec<Vec<u8>>,
}

impl<F: PartialEq> ChallengeTape<F> {
    /// First record where the tapes diverge in schedule (id sequence or
    /// length) or value. A dropped or inserted draw shifts every later
    /// squeeze without necessarily changing any shared-prefix value, so a
    /// pure `zip` would truncate to the shorter tape and miss it; the
    /// trailing check surfaces the first unmatched record instead.
    pub fn first_value_divergence(&self, other: &Self) -> Option<ChallengeId> {
        self.records
            .iter()
            .zip(&other.records)
            .find_map(|(left, right)| {
                (left.id != right.id || left.values != right.values).then_some(left.id)
            })
            .or_else(|| {
                self.records
                    .get(other.records.len())
                    .map(|record| record.id)
            })
            .or_else(|| {
                other
                    .records
                    .get(self.records.len())
                    .map(|record| record.id)
            })
    }
}

#[derive(Clone)]
struct ErasedRecord {
    id: ChallengeId,
    values: Arc<dyn Any + Send + Sync>,
}

enum Session {
    Record {
        records: Vec<ErasedRecord>,
        checks: Vec<Vec<u8>>,
        absorbs: Vec<Vec<u8>>,
        counters: BTreeMap<FsScope, usize>,
    },
    Replay {
        records: Vec<ErasedRecord>,
        checks: Vec<Vec<u8>>,
        absorbs: Vec<Vec<u8>>,
        counters: BTreeMap<FsScope, usize>,
        cursor: usize,
        check_cursor: usize,
        absorb_cursor: usize,
    },
}

thread_local! {
    static SESSION: RefCell<Option<Session>> = const { RefCell::new(None) };
}

fn next_id(counters: &mut BTreeMap<FsScope, usize>, len: usize) -> ChallengeId {
    let scope = fs_audit::current();
    let index = counters.entry(scope).or_default();
    let id = ChallengeId {
        scope,
        index: *index,
        len,
    };
    *index += 1;
    id
}

/// The [`Script`] of [`AuditTranscript`]: records every draw, check and
/// absorbed message, or replaces them with the active replay tape. It keeps the real challenge set,
/// so proofs carry their real nonces.
#[derive(Clone, Copy, Debug, Default)]
pub struct AuditScript;

impl<F: Field + 'static> Script<F> for AuditScript {
    const UNPRICED: bool = false;

    fn draw(&mut self, challenges: &mut [F]) {
        SESSION.with(|slot| {
            let mut slot = slot.borrow_mut();
            match slot.as_mut() {
                None => {}
                Some(Session::Record {
                    records, counters, ..
                }) => {
                    let id = next_id(counters, challenges.len());
                    records.push(ErasedRecord {
                        id,
                        values: Arc::new(challenges.to_vec()),
                    });
                }
                Some(Session::Replay {
                    records,
                    counters,
                    cursor,
                    ..
                }) => {
                    let id = next_id(counters, challenges.len());
                    let expected = records
                        .get(*cursor)
                        .unwrap_or_else(|| panic!("challenge replay exhausted at {id:?}"));
                    assert_eq!(
                        expected.id, id,
                        "challenge replay length or scope diverged at tape index {cursor}"
                    );
                    *cursor += 1;
                    challenges.clone_from_slice(
                        expected
                            .values
                            .downcast_ref::<Vec<F>>()
                            .expect("challenge replay field type changed"),
                    );
                }
            }
        });
    }

    // A frozen verifier takes every squeeze from the tape, so a replay over
    // changed inputs passes the proof-of-work checks the honest run passed.
    fn check_bytes(&mut self, bytes: &mut [u8]) {
        SESSION.with(|slot| match slot.borrow_mut().as_mut() {
            None => {}
            Some(Session::Record { checks, .. }) => checks.push(bytes.to_vec()),
            Some(Session::Replay {
                checks,
                check_cursor,
                ..
            }) => {
                let expected = checks
                    .get(*check_cursor)
                    .unwrap_or_else(|| panic!("check replay exhausted at {check_cursor}"));
                bytes.copy_from_slice(expected);
                *check_cursor += 1;
            }
        });
    }

    fn absorb_bytes(&mut self, bytes: &mut Vec<u8>) {
        SESSION.with(|slot| match slot.borrow_mut().as_mut() {
            None => {}
            Some(Session::Record { absorbs, .. }) => absorbs.push(bytes.clone()),
            Some(Session::Replay {
                absorbs,
                absorb_cursor,
                ..
            }) => {
                let expected = absorbs
                    .get(*absorb_cursor)
                    .unwrap_or_else(|| panic!("absorb replay exhausted at {absorb_cursor}"));
                bytes.clone_from(expected);
                *absorb_cursor += 1;
            }
        });
    }
}

/// The production verifier transcript with its draws and checks recorded or
/// replayed for the active session.
pub type AuditTranscript<'a, F> = ScriptedTranscript<VerifierState<'a, Keccak>, F, AuditScript>;

fn start(session: Session) {
    SESSION.with(|slot| {
        assert!(
            slot.borrow().is_none(),
            "nested Fiat-Shamir audit sessions are unsupported"
        );
        *slot.borrow_mut() = Some(session);
    });
}

/// Runs `verify` with challenge recording enabled.
pub fn record_challenges<F, R>(verify: impl FnOnce() -> R) -> (R, ChallengeTape<F>)
where
    F: Field + 'static,
{
    start(Session::Record {
        records: Vec::new(),
        checks: Vec::new(),
        absorbs: Vec::new(),
        counters: BTreeMap::new(),
    });
    let result = verify();
    let (records, checks, absorbs) = SESSION.with(|slot| match slot.borrow_mut().take() {
        Some(Session::Record {
            records,
            checks,
            absorbs,
            ..
        }) => (records, checks, absorbs),
        Some(Session::Replay { .. }) => panic!("recording session changed to replay"),
        None => panic!("recording session disappeared"),
    });
    let records = records
        .into_iter()
        .map(|record| ChallengeRecord {
            id: record.id,
            values: record
                .values
                .downcast_ref::<Vec<F>>()
                .expect("recorded challenge field type changed")
                .clone(),
        })
        .collect();
    (
        result,
        ChallengeTape {
            records,
            checks,
            absorbs,
        },
    )
}

/// Result of verifying with a frozen challenge tape.
pub struct ReplayResult<R> {
    pub output: R,
    pub consumed: usize,
    pub expected: usize,
}

/// Runs `verify` while returning the recorded values for every challenge
/// draw and proof-of-work check, and absorbing the recorded messages.
pub fn replay_challenges<F, R>(
    tape: &ChallengeTape<F>,
    verify: impl FnOnce() -> R,
) -> ReplayResult<R>
where
    F: Field + 'static,
{
    start(Session::Replay {
        records: tape
            .records
            .iter()
            .map(|record| ErasedRecord {
                id: record.id,
                values: Arc::new(record.values.clone()),
            })
            .collect(),
        checks: tape.checks.clone(),
        absorbs: tape.absorbs.clone(),
        counters: BTreeMap::new(),
        cursor: 0,
        check_cursor: 0,
        absorb_cursor: 0,
    });
    let output = verify();
    let (consumed, expected) = SESSION.with(|slot| match slot.borrow_mut().take() {
        Some(Session::Replay {
            records, cursor, ..
        }) => (cursor, records.len()),
        Some(Session::Record { .. }) => panic!("replay session changed to recording"),
        None => panic!("replay session disappeared"),
    });
    ReplayResult {
        output,
        consumed,
        expected,
    }
}
