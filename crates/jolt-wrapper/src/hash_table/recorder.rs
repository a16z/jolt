//! A transcript decorator logging every append (with its bytes) and squeeze
//! (with its decoder and result) of a verifier run, plus the wrapped
//! transcript's state after each event.
//!
//! The Jolt verifier constructs its transcript internally and returns nothing
//! of it, so the log lives in a thread-local the caller drains afterwards
//! ([`RecordingTranscript::take_log`]).

use std::any::type_name;
use std::cell::{Cell, RefCell};

use jolt_field::Fr;
use jolt_transcript::{AppendToTranscript, Transcript};

/// How a squeeze's 16 bytes became a field element (`plan-relation` §6).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Decoder {
    /// `Transcript::challenge`: little-endian `u128`, top 3 bits masked,
    /// placed in the high Montgomery limbs.
    Challenge125,
    /// `Transcript::challenge_scalar`: big-endian `u128`, reduced.
    Scalar128,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Event {
    /// `Transcript::new(label)`.
    Start {
        label: Vec<u8>,
    },
    /// An `append_bytes`; `labeled` when the bytes came through
    /// `Transcript::append` of a `Label`, `LabelWithCount` or `U64Word`.
    Append {
        bytes: Vec<u8>,
        labeled: bool,
    },
    Squeeze {
        decoder: Decoder,
        value: Fr,
    },
}

/// One logged event and the wrapped transcript's `state()` right after it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Recorded {
    pub event: Event,
    pub state: [u8; 32],
}

thread_local! {
    static LOG: RefCell<Vec<Recorded>> = const { RefCell::new(Vec::new()) };
    static LABELED: Cell<bool> = const { Cell::new(false) };
}

/// Forwards every call to `T` and logs it.
#[derive(Default)]
pub struct RecordingTranscript<T>(T);

impl<T: Transcript> RecordingTranscript<T> {
    /// Drain the thread's log.
    pub fn take_log() -> Vec<Recorded> {
        LOG.with(|log| std::mem::take(&mut *log.borrow_mut()))
    }

    fn record(&self, event: Event) {
        let state = self.0.state();
        LOG.with(|log| log.borrow_mut().push(Recorded { event, state }));
    }
}

impl<T: Transcript<Challenge = Fr>> Transcript for RecordingTranscript<T> {
    type Challenge = Fr;

    fn new(label: &'static [u8]) -> Self {
        let this = Self(T::new(label));
        this.record(Event::Start {
            label: label.to_vec(),
        });
        this
    }

    fn append_bytes(&mut self, bytes: &[u8]) {
        let labeled = LABELED.with(|flag| flag.replace(false));
        self.0.append_bytes(bytes);
        self.record(Event::Append {
            bytes: bytes.to_vec(),
            labeled,
        });
    }

    fn append<A: AppendToTranscript>(&mut self, value: &A) {
        // The domain-separation word types are the only appends the protocol
        // treats as constants; a raw `append_bytes` never carries one.
        let name = type_name::<A>();
        let labeled = name.ends_with("::Label")
            || name.ends_with("::LabelWithCount")
            || name.ends_with("::U64Word");
        LABELED.with(|flag| flag.set(labeled));
        value.append_to_transcript(self);
        LABELED.with(|flag| flag.set(false));
    }

    fn challenge(&mut self) -> Fr {
        let value = self.0.challenge();
        self.record(Event::Squeeze {
            decoder: Decoder::Challenge125,
            value,
        });
        value
    }

    fn challenge_scalar(&mut self) -> Fr {
        let value = self.0.challenge_scalar();
        self.record(Event::Squeeze {
            decoder: Decoder::Scalar128,
            value,
        });
        value
    }

    fn state(&self) -> [u8; 32] {
        self.0.state()
    }
}
