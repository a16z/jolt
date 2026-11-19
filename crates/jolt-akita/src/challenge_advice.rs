//! Checked challenge advice for Akita verification inside a guest.
//!
//! A host records the advice with [`record_checked_hints`] while it verifies a
//! proof natively. A guest passes it to [`verify_with_checked_hints`], so the
//! Akita verifier checks pre-decoded fold challenges against the transcript
//! instead of decoding them. Proof, setup and transcript bytes are unchanged.
//!
//! The advice session is thread-local, so it must live on the thread that
//! draws the fold challenges. See [`on_verifier_thread`].

use akita_pcs::AkitaError;
use core::fmt::{Display, Formatter, Result as FmtResult};

/// Why verification with checked challenge advice failed.
#[derive(Debug)]
pub enum CheckedHintsError<E> {
    /// The advice could not be installed, or verification left part of it unused.
    Advice(AkitaError),
    /// Verification itself failed.
    Verify(E),
}

impl<E: Display> Display for CheckedHintsError<E> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            Self::Advice(error) => write!(f, "checked challenge advice: {error}"),
            Self::Verify(error) => error.fmt(f),
        }
    }
}

/// Run `f` on the thread that runs Akita verification.
///
/// On a host, Akita verification without a setup accumulator runs on the
/// backend pool, while accumulator verification runs on the calling thread.
/// Running inside the pool covers both, because nested pool calls run inline on
/// the same worker. A RISC-V guest's verifier is single-threaded, so `f` runs
/// inline there instead of paying for a pool. A guest that verifies through the
/// pool must build this crate without `parallel`, as the manifest requires.
fn on_verifier_thread<R: Send>(f: impl FnOnce() -> R + Send) -> R {
    #[cfg(target_arch = "riscv64")]
    return f();
    #[cfg(not(target_arch = "riscv64"))]
    crate::adapters::with_backend_pool(f)
}

/// Run `verify` natively and record the checked challenge advice its fold
/// draws consume. Run only verification inside `verify`.
///
/// # Errors
///
/// Returns [`AkitaError`] if an advice session is already active on the
/// backend thread. `verify`'s own result is returned alongside the advice;
/// the advice is meaningful only when verification succeeds.
pub fn record_checked_hints<T: Send>(
    verify: impl FnOnce() -> T + Send,
) -> Result<(T, Vec<u8>), AkitaError> {
    on_verifier_thread(|| akita_challenges::record_checked_hints(verify))
}

/// Run `verify` with `advice` installed, then require that it consumed all of it.
///
/// `max_bytes` must come from trusted configuration such as the guest image,
/// never from `advice`. The advice session is released whether or not
/// verification succeeds.
///
/// # Errors
///
/// Returns [`CheckedHintsError::Advice`] when the advice cannot be installed or
/// is not fully consumed, and [`CheckedHintsError::Verify`] with `verify`'s error.
pub fn verify_with_checked_hints<T: Send, E: Send>(
    advice: &[u8],
    max_bytes: usize,
    verify: impl FnOnce() -> Result<T, E> + Send,
) -> Result<T, CheckedHintsError<E>> {
    on_verifier_thread(|| {
        akita_challenges::install_checked_hints(advice, max_bytes)
            .map_err(CheckedHintsError::Advice)?;
        match verify() {
            Ok(value) => akita_challenges::finish_checked_hints()
                .map(|()| value)
                .map_err(CheckedHintsError::Advice),
            Err(error) => {
                akita_challenges::abort_checked_hints().map_err(CheckedHintsError::Advice)?;
                Err(CheckedHintsError::Verify(error))
            }
        }
    })
}
