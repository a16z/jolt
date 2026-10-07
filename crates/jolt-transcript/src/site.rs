//! Diagnostic protocol sites and the transcript event log.
//!
//! Sites never reach the sponge: a proof and its challenges are identical with
//! and without the `logging` feature. Under `logging`, each transcript records
//! one [`TranscriptEvent`] per operation, tagged with the site set by the most
//! recent [`Channel::site`](crate::Channel::site) call.
use core::ops::Range;

/// Opaque 32-byte name of a protocol site, chosen by the protocol.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct SiteId(pub [u8; 32]);

impl SiteId {
    /// A site named by an ASCII label of at most 32 bytes, zero padded.
    ///
    /// # Panics
    ///
    /// Panics if `label` exceeds 32 bytes. Evaluate it in a `const` item to
    /// turn that into a compile error.
    #[must_use]
    #[expect(
        clippy::indexing_slicing,
        reason = "the loop index is below label.len(), which is asserted to fit"
    )]
    pub const fn label(label: &str) -> Self {
        let label = label.as_bytes();
        assert!(label.len() <= 32, "site label exceeds 32 bytes");
        let mut id = [0u8; 32];
        let mut i = 0;
        while i < label.len() {
            id[i] = label[i];
            i += 1;
        }
        Self(id)
    }
}

/// Kind of transcript operation recorded in the event log.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TranscriptOp {
    /// Absorbed a public value.
    Public,
    /// Sent (prover) or received (verifier) a prover message.
    Message,
    /// Squeezed a verifier challenge.
    Challenge,
}

/// One recorded transcript operation.
#[cfg(feature = "logging")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TranscriptEvent {
    /// Site active when the operation ran.
    pub site: SiteId,
    /// What the operation did.
    pub op: TranscriptOp,
    /// Bytes absorbed or squeezed.
    pub len: usize,
    /// Argument-string byte range of a prover message.
    pub narg: Option<Range<usize>>,
}

/// Event log of one transcript; empty and free without the `logging` feature.
#[derive(Clone, Debug, Default)]
pub(crate) struct Log {
    #[cfg(feature = "logging")]
    site: SiteId,
    #[cfg(feature = "logging")]
    events: Vec<TranscriptEvent>,
}

#[cfg(feature = "logging")]
impl Log {
    pub(crate) fn set_site(&mut self, site: SiteId) {
        self.site = site;
    }

    pub(crate) fn record(&mut self, op: TranscriptOp, len: usize, narg: Option<Range<usize>>) {
        self.events.push(TranscriptEvent {
            site: self.site,
            op,
            len,
            narg,
        });
    }

    pub(crate) fn events(&self) -> &[TranscriptEvent] {
        &self.events
    }
}

#[cfg(not(feature = "logging"))]
#[expect(
    clippy::unused_self,
    reason = "the event log compiles to nothing without the logging feature"
)]
impl Log {
    #[inline(always)]
    pub(crate) fn set_site(&mut self, _site: SiteId) {}

    #[inline(always)]
    pub(crate) fn record(&mut self, _op: TranscriptOp, _len: usize, _narg: Option<Range<usize>>) {}
}
