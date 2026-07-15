//! Witness backends: implementors of the id-indexed oracle surface.

use jolt_claims::protocols::jolt::{JoltCommittedPolynomial, JoltPolynomialId};
use jolt_field::Field;

use crate::{PolynomialBatchStream, PolynomialStream, Shape, WitnessError};

pub mod trace;

/// The object-safe id-indexed witness surface of the Jolt VM protocol — what
/// `&dyn` consumers (the naive interpreter's kernels, the commitment slot)
/// need. Ids are jolt-claims vocabulary; this crate defines none.
///
/// Typed consumers (bundles over `stream_witnesses`) are statically
/// dispatched and do not go through this trait; both paths meet at the same
/// `Extract` impls.
pub trait JoltWitnessOracle<F: Field> {
    fn shape(&self, id: JoltPolynomialId) -> Result<Shape, WitnessError>;

    /// Materializes the oracle's dense field-element evaluations, row-major
    /// over the domain declared by [`shape`](Self::shape); one-hot grids are
    /// returned as flat address-major `(K x T)` tables.
    fn oracle_table(&self, id: JoltPolynomialId) -> Result<Vec<F>, WitnessError>;

    /// The proof-payload order of the committed polynomials this backend
    /// serves.
    fn committed_order(&self) -> Result<Vec<JoltCommittedPolynomial>, WitnessError>;

    fn committed_stream<'a>(
        &'a self,
        id: JoltCommittedPolynomial,
        chunk_size: usize,
    ) -> Result<Box<dyn PolynomialStream<F> + 'a>, WitnessError>
    where
        F: 'a;

    fn committed_batch_stream<'a>(
        &'a self,
        ids: &'a [JoltCommittedPolynomial],
        chunk_size: usize,
    ) -> Result<Box<dyn PolynomialBatchStream<F, JoltCommittedPolynomial> + 'a>, WitnessError>
    where
        F: 'a;
}
