//! A vector-commitment placeholder for transparent-only protocol
//! configurations (the packed/lattice Jolt path).

use core::fmt::{Debug, Formatter, Result as FmtResult};
use core::marker::PhantomData;

use jolt_field::{CanonicalBytes, JoltField};
use serde::{Deserialize, Serialize};

use crate::{Commitment, HomomorphicCommitment, VectorCommitment};

/// A vector-commitment placeholder for transparent-only protocol
/// configurations that never produce or verify hiding commitments (the
/// packed/lattice Jolt path): the proof model requires *some*
/// [`VectorCommitment`] type parameter, but every zk arm is rejected
/// fail-closed before a commitment could be touched.
pub struct NoVectorCommitment<F>(PhantomData<fn() -> F>);

impl<F> Clone for NoVectorCommitment<F> {
    fn clone(&self) -> Self {
        Self(PhantomData)
    }
}

impl<F> Debug for NoVectorCommitment<F> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.write_str("NoVectorCommitment")
    }
}

impl<F> PartialEq for NoVectorCommitment<F> {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

impl<F> Eq for NoVectorCommitment<F> {}

/// The (empty) commitment value of [`NoVectorCommitment`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct NoCommitment;

// `AppendToTranscript` comes from jolt-transcript's blanket impl over
// `CanonicalBytes`: an empty canonical encoding, so absorbing a
// `NoCommitment` is a no-op.
impl CanonicalBytes for NoCommitment {
    const NUM_BYTES: usize = 0;

    fn to_bytes_le(&self, _out: &mut [u8]) {}
}

impl<F: JoltField> HomomorphicCommitment<F> for NoCommitment {
    fn add(_c1: &Self, _c2: &Self) -> Self {
        Self
    }

    fn linear_combine(_c1: &Self, _c2: &Self, _scalar: &F) -> Self {
        Self
    }
}

impl<F: JoltField> Commitment for NoVectorCommitment<F> {
    type Output = NoCommitment;
}

impl<F: JoltField> VectorCommitment for NoVectorCommitment<F> {
    type Field = F;
    type Setup = ();

    fn capacity(_setup: &Self::Setup) -> usize {
        0
    }

    #[expect(
        clippy::panic,
        reason = "transparent-only placeholder; every zk arm is rejected before a commitment could be requested"
    )]
    fn commit(
        _setup: &Self::Setup,
        _values: &[Self::Field],
        _blinding: &Self::Field,
    ) -> Self::Output {
        panic!("NoVectorCommitment never commits: the packed axis is transparent-only")
    }

    fn verify(
        _setup: &Self::Setup,
        _commitment: &Self::Output,
        _values: &[Self::Field],
        _blinding: &Self::Field,
    ) -> bool {
        false
    }
}
