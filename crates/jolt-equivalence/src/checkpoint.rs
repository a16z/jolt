//! Checkpoint transcript for fine-grained Fiat-Shamir comparison.
//!
//! Wraps any [`Transcript`] and records every append/squeeze operation as
//! a [`TranscriptEvent`]. Two event logs can then be compared element by
//! element to find the *exact* operation where two systems diverge.
//!
//! # Usage
//!
//! ```ignore
//! // 1. Record the golden reference from jolt-core.
//! let mut golden = CheckpointTranscript::<Blake2bTranscript<Fr>>::new(LABEL);
//! // ...run jolt-core prover with `golden` as the transcript...
//! let golden_log = golden.into_log();
//!
//! // 2. Record the candidate from jolt-zkvm.
//! let mut candidate = CheckpointTranscript::<Blake2bTranscript<Fr>>::new(LABEL);
//! // ...run jolt-zkvm prover with `candidate` as the transcript...
//! let candidate_log = candidate.into_log();
//!
//! // 3. Compare.
//! let divergence = find_divergence(&golden_log, &candidate_log);
//! ```

use std::fmt;

use jolt_transcript::Transcript;

/// A single transcript operation.
#[derive(Clone)]
pub enum TranscriptEvent {
    /// Raw bytes were appended to the transcript.
    Append {
        /// The bytes that were absorbed.
        bytes: Vec<u8>,
        /// Transcript state *after* this append.
        state_after: [u8; 32],
    },
    /// A challenge was squeezed from the transcript.
    Squeeze {
        /// The 32-byte transcript state *after* the squeeze.
        state_after: [u8; 32],
    },
}

impl fmt::Debug for TranscriptEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fn hex(b: &[u8]) -> String {
            use std::fmt::Write;
            let mut s = String::with_capacity(b.len() * 2);
            for byte in b {
                let _ = write!(s, "{byte:02x}");
            }
            s
        }
        match self {
            TranscriptEvent::Append { bytes, state_after } => {
                let preview = if bytes.len() <= 32 {
                    hex(bytes)
                } else {
                    format!("{}... ({} bytes)", hex(&bytes[..16]), bytes.len())
                };
                write!(f, "Append({preview}) -> {}", &hex(state_after)[..16])
            }
            TranscriptEvent::Squeeze { state_after } => {
                write!(f, "Squeeze -> {}", &hex(state_after)[..16])
            }
        }
    }
}

/// Wraps a concrete transcript, recording every operation.
///
/// Implements [`Transcript`] so it can be used as a drop-in replacement.
/// After the protocol runs, call [`into_log`](Self::into_log) to extract
/// the event sequence.
pub struct CheckpointTranscript<T: Transcript> {
    inner: T,
    log: Vec<TranscriptEvent>,
}

impl<T: Transcript> CheckpointTranscript<T> {
    /// Consume the wrapper and return the recorded event log.
    pub fn into_log(self) -> Vec<TranscriptEvent> {
        self.log
    }

    /// Borrow the event log without consuming.
    pub fn log(&self) -> &[TranscriptEvent] {
        &self.log
    }

    /// Number of events recorded so far.
    pub fn len(&self) -> usize {
        self.log.len()
    }

    /// Whether the log is empty.
    pub fn is_empty(&self) -> bool {
        self.log.is_empty()
    }
}

impl<T: Transcript> Clone for CheckpointTranscript<T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            log: self.log.clone(),
        }
    }
}

impl<T: Transcript> Default for CheckpointTranscript<T> {
    fn default() -> Self {
        Self {
            inner: T::default(),
            log: Vec::new(),
        }
    }
}

impl<T: Transcript> Transcript for CheckpointTranscript<T> {
    type Challenge = <T as Transcript>::Challenge;

    fn new(label: &'static [u8]) -> Self {
        Self {
            inner: T::new(label),
            log: Vec::new(),
        }
    }

    fn append_bytes(&mut self, bytes: &[u8]) {
        self.inner.append_bytes(bytes);
        self.log.push(TranscriptEvent::Append {
            bytes: bytes.to_vec(),
            state_after: *self.inner.state(),
        });
    }

    fn challenge(&mut self) -> Self::Challenge {
        let c = self.inner.challenge();
        self.log.push(TranscriptEvent::Squeeze {
            state_after: *self.inner.state(),
        });
        c
    }

    fn state(&self) -> &[u8; 32] {
        self.inner.state()
    }
}

/// Describes where two transcript logs first diverge.
#[derive(Debug)]
pub struct TranscriptDivergence {
    /// Zero-based index of the diverging operation.
    pub op_index: usize,
    /// What the reference system did.
    pub expected: String,
    /// What the candidate system did.
    pub actual: String,
}

impl fmt::Display for TranscriptDivergence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "transcript divergence at op #{}: expected {}, got {}",
            self.op_index, self.expected, self.actual
        )
    }
}

/// Compare two transcript event logs and return the first divergence.
///
/// Returns `Ok(())` if the logs are identical (up to the length of the
/// shorter log — a length mismatch after identical prefixes is also reported).
pub fn find_divergence(
    reference: &[TranscriptEvent],
    candidate: &[TranscriptEvent],
) -> Result<(), TranscriptDivergence> {
    let min_len = reference.len().min(candidate.len());

    for i in 0..min_len {
        let states_match = match (&reference[i], &candidate[i]) {
            (
                TranscriptEvent::Append {
                    state_after: sa, ..
                },
                TranscriptEvent::Append {
                    state_after: sb, ..
                },
            ) => sa == sb,
            (
                TranscriptEvent::Squeeze { state_after: sa },
                TranscriptEvent::Squeeze { state_after: sb },
            ) => sa == sb,
            _ => false, // Different event kinds
        };

        if !states_match {
            return Err(TranscriptDivergence {
                op_index: i,
                expected: format!("{:?}", reference[i]),
                actual: format!("{:?}", candidate[i]),
            });
        }
    }

    if reference.len() != candidate.len() {
        return Err(TranscriptDivergence {
            op_index: min_len,
            expected: if min_len < reference.len() {
                format!("{:?}", reference[min_len])
            } else {
                "<end>".to_string()
            },
            actual: if min_len < candidate.len() {
                format!("{:?}", candidate[min_len])
            } else {
                "<end>".to_string()
            },
        });
    }

    Ok(())
}

/// Compare two logs and panic with a detailed message on divergence.
#[allow(clippy::print_stderr)]
pub fn assert_transcripts_match(reference: &[TranscriptEvent], candidate: &[TranscriptEvent]) {
    if let Err(div) = find_divergence(reference, candidate) {
        // Print context: a few events before the divergence point.
        let start = div.op_index.saturating_sub(3);
        eprintln!("=== Transcript divergence at op #{} ===", div.op_index);
        let end_ref = div.op_index.min(reference.len().saturating_sub(1));
        let end_cand = div.op_index.min(candidate.len().saturating_sub(1));
        eprintln!("Context (reference):");
        for (i, event) in reference.iter().enumerate().take(end_ref + 1).skip(start) {
            let marker = if i == div.op_index { ">>>" } else { "   " };
            eprintln!("  {marker} [{i}] {event:?}");
        }
        eprintln!("Context (candidate):");
        for (i, event) in candidate.iter().enumerate().take(end_cand + 1).skip(start) {
            let marker = if i == div.op_index { ">>>" } else { "   " };
            eprintln!("  {marker} [{i}] {event:?}");
        }
        panic!("{div}");
    }
}
