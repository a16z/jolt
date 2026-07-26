#![expect(
    clippy::expect_used,
    clippy::panic,
    reason = "the audit test double must fail loudly on a malformed replay tape"
)]

use std::{any::Any, cell::RefCell, collections::BTreeMap, sync::Arc};

use jolt_field::Field;
use jolt_transcript::Transcript;
use jolt_verifier::fs_audit::{self, FsScope};

/// The transcript API used to derive a challenge.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChallengeKind {
    Challenge,
    Scalar,
    Vector(usize),
    ScalarPowers(usize),
}

/// A challenge's verifier scope and ordinal within that scope.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChallengeId {
    pub scope: FsScope,
    pub index: usize,
    pub kind: ChallengeKind,
}

/// One typed challenge call and all values it returned.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChallengeRecord<F> {
    pub id: ChallengeId,
    pub values: Vec<F>,
}

/// Challenges produced while verifying one fixture.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChallengeTape<F> {
    pub records: Vec<ChallengeRecord<F>>,
}

impl<F: PartialEq> ChallengeTape<F> {
    pub fn first_value_divergence(&self, other: &Self) -> Option<ChallengeId> {
        self.records
            .iter()
            .zip(&other.records)
            .find_map(|(left, right)| {
                (left.id != right.id || left.values != right.values).then_some(left.id)
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
        counters: BTreeMap<FsScope, usize>,
    },
    Replay {
        records: Vec<ErasedRecord>,
        counters: BTreeMap<FsScope, usize>,
        cursor: usize,
    },
}

thread_local! {
    static SESSION: RefCell<Option<Session>> = const { RefCell::new(None) };
}

fn next_id(counters: &mut BTreeMap<FsScope, usize>, kind: ChallengeKind) -> ChallengeId {
    let scope = fs_audit::current();
    let index = counters.entry(scope).or_default();
    let id = ChallengeId {
        scope,
        index: *index,
        kind,
    };
    *index += 1;
    id
}

fn draw<F>(kind: ChallengeKind, produce: impl FnOnce() -> Vec<F>) -> Vec<F>
where
    F: Field,
{
    let actual = produce();
    SESSION.with(|slot| {
        let mut slot = slot.borrow_mut();
        let Some(session) = slot.as_mut() else {
            return actual;
        };
        match session {
            Session::Record { records, counters } => {
                let id = next_id(counters, kind);
                records.push(ErasedRecord {
                    id,
                    values: Arc::new(actual.clone()),
                });
                actual
            }
            Session::Replay {
                records,
                counters,
                cursor,
            } => {
                let id = next_id(counters, kind);
                let expected = records
                    .get(*cursor)
                    .unwrap_or_else(|| panic!("challenge replay exhausted at {id:?}"));
                assert_eq!(
                    expected.id, id,
                    "challenge replay kind or scope diverged at tape index {cursor}"
                );
                *cursor += 1;
                expected
                    .values
                    .downcast_ref::<Vec<F>>()
                    .expect("challenge replay field type changed")
                    .clone()
            }
        }
    })
}

/// Transcript wrapper that records or replays challenge calls for the active session.
pub struct AuditTranscript<T> {
    inner: T,
}

impl<T: Default> Default for AuditTranscript<T> {
    fn default() -> Self {
        Self {
            inner: T::default(),
        }
    }
}

impl<T> Transcript for AuditTranscript<T>
where
    T: Transcript,
    T::Challenge: Field,
{
    type Challenge = T::Challenge;

    fn new(label: &'static [u8]) -> Self {
        Self {
            inner: T::new(label),
        }
    }

    fn append_bytes(&mut self, bytes: &[u8]) {
        self.inner.append_bytes(bytes);
    }

    fn challenge(&mut self) -> Self::Challenge {
        draw(ChallengeKind::Challenge, || vec![self.inner.challenge()])[0]
    }

    fn challenge_scalar(&mut self) -> Self::Challenge {
        draw(ChallengeKind::Scalar, || {
            vec![self.inner.challenge_scalar()]
        })[0]
    }

    fn challenge_vector(&mut self, len: usize) -> Vec<Self::Challenge> {
        draw(ChallengeKind::Vector(len), || {
            self.inner.challenge_vector(len)
        })
    }

    fn challenge_scalar_powers(&mut self, len: usize) -> Vec<Self::Challenge> {
        draw(ChallengeKind::ScalarPowers(len), || {
            self.inner.challenge_scalar_powers(len)
        })
    }

    fn state(&self) -> [u8; 32] {
        self.inner.state()
    }
}

/// Runs `verify` with challenge recording enabled.
pub fn record_challenges<F, R>(verify: impl FnOnce() -> R) -> (R, ChallengeTape<F>)
where
    F: Field,
{
    SESSION.with(|slot| {
        assert!(
            slot.borrow().is_none(),
            "nested Fiat-Shamir audit sessions are unsupported"
        );
        *slot.borrow_mut() = Some(Session::Record {
            records: Vec::new(),
            counters: BTreeMap::new(),
        });
    });

    let result = verify();
    let records = SESSION.with(|slot| match slot.borrow_mut().take() {
        Some(Session::Record { records, .. }) => records,
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
    (result, ChallengeTape { records })
}

/// Result of verifying with a frozen challenge tape.
pub struct ReplayResult<R> {
    pub output: R,
    pub consumed: usize,
    pub expected: usize,
}

/// Runs `verify` while returning the recorded values for every challenge call.
pub fn replay_challenges<F, R>(
    tape: &ChallengeTape<F>,
    verify: impl FnOnce() -> R,
) -> ReplayResult<R>
where
    F: Field,
{
    let records = tape
        .records
        .iter()
        .map(|record| ErasedRecord {
            id: record.id,
            values: Arc::new(record.values.clone()),
        })
        .collect();
    SESSION.with(|slot| {
        assert!(
            slot.borrow().is_none(),
            "nested Fiat-Shamir audit sessions are unsupported"
        );
        *slot.borrow_mut() = Some(Session::Replay {
            records,
            counters: BTreeMap::new(),
            cursor: 0,
        });
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
