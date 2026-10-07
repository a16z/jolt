//! A challenge-freezing sponge for Fiat-Shamir attack tests.
//!
//! [`AuditSponge`] runs the production [`Blake2b512`] duplex and shares its
//! [`Sponge::ID`], so it verifies honest Blake2b proofs unchanged. Inside a
//! [`record`] session it also writes every squeezed byte to a tape; inside a
//! [`replay`] session every squeeze returns the recorded bytes instead, so
//! the verifier's challenges no longer depend on anything it absorbs. A
//! mutation the frozen verifier accepts is therefore stopped only by
//! Fiat-Shamir binding.
//!
//! The tape is one byte stream indexed by squeeze position, not a list of
//! per-call records. Exact challenges rejection-sample a data-dependent number
//! of bytes, so only the byte stream is invariant under replay.
//!
//! Seeded forks ([`jolt_transcript::Fork`]) are not taped. A fork's output is
//! a function of its seed, squeezed from the taped transcript, and its
//! counter, a proof message, so freezing the transcript already freezes every
//! fork. A sponge is a fork when its first absorb is the fork domain tag; it
//! then runs plain Blake2b on any thread, inside a session or not (Akita
//! searches fold responses on worker threads).

#![expect(
    clippy::expect_used,
    clippy::panic,
    reason = "the audit test double must fail loudly outside a session or on a malformed tape"
)]

use std::{
    cell::RefCell,
    sync::{Arc, Mutex},
};

use jolt_field::{CanonicalEncoding, Field};
use jolt_transcript::DuplexSpongeInterface;
use jolt_transcript::{Blake2b512, Channel, ProtocolId, Sponge, VerifierTranscript};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Record,
    Replay,
}

#[derive(Debug)]
struct Tape {
    mode: Mode,
    bytes: Vec<u8>,
    /// Furthest stream position any sponge of the session squeezed to.
    high_water: usize,
}

thread_local! {
    static ACTIVE: RefCell<Option<Arc<Mutex<Tape>>>> = const { RefCell::new(None) };
}

/// The first absorb of every [`jolt_transcript::Fork`].
const FORK_TAG: &[u8] = b"jolt-transcript/fork/v1";

/// [`Blake2b512`] with a recorded or replayed squeeze stream.
///
/// [`Default`] binds the new sponge to the session active on the calling
/// thread, if any. A transcript sponge must have one; a fork drops it on its
/// first absorb.
#[derive(Clone)]
pub struct AuditSponge {
    inner: Blake2b512,
    position: usize,
    tape: Option<Arc<Mutex<Tape>>>,
    absorbed: bool,
}

impl Default for AuditSponge {
    fn default() -> Self {
        Self {
            inner: Blake2b512::default(),
            position: 0,
            tape: ACTIVE.with(|active| active.borrow().clone()),
            absorbed: false,
        }
    }
}

impl DuplexSpongeInterface for AuditSponge {
    type U = u8;

    fn absorb(&mut self, input: &[u8]) -> &mut Self {
        if !self.absorbed {
            self.absorbed = true;
            if input == FORK_TAG {
                self.tape = None;
            } else {
                assert!(
                    self.tape.is_some(),
                    "AuditSponge transcript constructed outside a record or replay session"
                );
            }
        }
        let _ = self.inner.absorb(input);
        self
    }

    fn squeeze(&mut self, output: &mut [u8]) -> &mut Self {
        let Some(tape) = &self.tape else {
            let _ = self.inner.squeeze(output);
            return self;
        };
        let start = self.position;
        let end = start + output.len();
        let mut tape = tape.lock().expect("audit tape lock poisoned");
        match tape.mode {
            Mode::Record => {
                let _ = self.inner.squeeze(output);
                assert_eq!(
                    start,
                    tape.bytes.len(),
                    "a second transcript sponge squeezed into one session's tape"
                );
                tape.bytes.extend_from_slice(output);
            }
            Mode::Replay => {
                let recorded = tape.bytes.get(start..end).unwrap_or_else(|| {
                    panic!(
                        "challenge replay exhausted: squeeze {start}..{end} past {} recorded bytes",
                        tape.bytes.len()
                    )
                });
                output.copy_from_slice(recorded);
            }
        }
        tape.high_water = tape.high_water.max(end);
        drop(tape);
        self.position = end;
        self
    }

    fn ratchet(&mut self) -> &mut Self {
        panic!("Jolt transcripts never ratchet; the audit tape cannot model one");
    }
}

impl Sponge for AuditSponge {
    const ID: &'static str = <Blake2b512 as Sponge>::ID;
}

/// Every byte squeezed during one recorded verification, in stream order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChallengeTape {
    pub bytes: Vec<u8>,
}

impl ChallengeTape {
    /// First stream offset at which the tapes differ, including one tape
    /// ending before the other.
    pub fn first_divergence(&self, other: &Self) -> Option<usize> {
        self.bytes
            .iter()
            .zip(&other.bytes)
            .position(|(left, right)| left != right)
            .or_else(|| {
                (self.bytes.len() != other.bytes.len())
                    .then_some(self.bytes.len().min(other.bytes.len()))
            })
    }
}

/// Clears the thread's session even when the closure under audit panics.
struct SessionGuard;

impl SessionGuard {
    fn start(tape: Tape) -> (Self, Arc<Mutex<Tape>>) {
        let tape = Arc::new(Mutex::new(tape));
        ACTIVE.with(|active| {
            let mut active = active.borrow_mut();
            assert!(
                active.is_none(),
                "nested Fiat-Shamir audit sessions are unsupported"
            );
            *active = Some(Arc::clone(&tape));
        });
        (Self, tape)
    }
}

impl Drop for SessionGuard {
    fn drop(&mut self) {
        ACTIVE.with(|active| *active.borrow_mut() = None);
    }
}

fn finish(guard: SessionGuard, tape: &Mutex<Tape>) -> (Vec<u8>, usize) {
    drop(guard);
    let tape = tape.lock().expect("audit tape lock poisoned");
    (tape.bytes.clone(), tape.high_water)
}

/// Runs `run` with every [`AuditSponge`] squeeze recorded.
pub fn record<R>(run: impl FnOnce() -> R) -> (R, ChallengeTape) {
    let (guard, tape) = SessionGuard::start(Tape {
        mode: Mode::Record,
        bytes: Vec::new(),
        high_water: 0,
    });
    let output = run();
    let (bytes, _) = finish(guard, &tape);
    (output, ChallengeTape { bytes })
}

/// Result of running against a frozen tape.
pub struct Replayed<R> {
    pub output: R,
    /// Bytes of the tape the run squeezed.
    pub consumed: usize,
    pub recorded: usize,
}

/// Runs `run` with every [`AuditSponge`] squeeze answered from `tape`.
pub fn replay<R>(tape: &ChallengeTape, run: impl FnOnce() -> R) -> Replayed<R> {
    let (guard, session) = SessionGuard::start(Tape {
        mode: Mode::Replay,
        bytes: tape.bytes.clone(),
        high_water: 0,
    });
    let output = run();
    let (_, consumed) = finish(guard, &session);
    Replayed {
        output,
        consumed,
        recorded: tape.bytes.len(),
    }
}

/// The value a production transcript draws when its squeezes return `bytes`.
fn draw_from<T>(
    bytes: &[u8],
    draw: impl FnOnce(&mut VerifierTranscript<'_, AuditSponge>) -> T,
) -> T {
    let tape = ChallengeTape {
        bytes: bytes.to_vec(),
    };
    let replayed = replay(&tape, || {
        let protocol = ProtocolId::new::<AuditSponge>("jolt/fs-attacks/decode");
        draw(&mut VerifierTranscript::new(&protocol, b"", &[]))
    });
    assert_eq!(
        replayed.consumed,
        bytes.len(),
        "challenge decode consumed a different byte count than the recorded draw"
    );
    replayed.output
}

/// Decodes an exactly uniform [`Channel::challenge`] from the bytes it squeezed.
pub fn decode_challenge<F: Field>(bytes: &[u8]) -> F {
    draw_from(bytes, |transcript| transcript.challenge())
}

/// Decodes a [`Channel::challenge_small`] from the bytes it squeezed.
pub fn decode_small_challenge<F: CanonicalEncoding>(bytes: &[u8]) -> F {
    draw_from(bytes, |transcript| transcript.challenge_small())
}
