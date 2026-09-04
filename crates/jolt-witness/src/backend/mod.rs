//! Witness backends: implementors of the id-indexed oracle surface.

use jolt_claims::protocols::jolt::{JoltCommittedPolynomial, JoltPolynomialId};
use jolt_field::Field;
use jolt_program::preprocess::JoltProgramPreprocessing;

#[cfg(feature = "field-inline")]
use crate::field_inline::FieldInlineWitnessOracle;
use crate::{RowSource, Shape, WitnessBundle, WitnessError};

#[cfg(any(test, feature = "test-utils"))]
pub mod fixed;
pub mod trace;

/// Stage-0 validation: every id a proof will request — bundle annotated
/// sets and the config's committed set — must be servable before witness
/// generation starts. The servable set is the backend's exhaustive match
/// (its `shape` resolving), never a curated list.
pub fn validate_servable<F: Field>(
    oracle: &dyn JoltWitnessOracle<F>,
    ids: impl IntoIterator<Item = JoltPolynomialId>,
) -> Result<(), WitnessError> {
    for id in ids {
        if let Err(error) = oracle.shape(id) {
            return Err(WitnessError::InvalidWitnessData {
                label: "stage-0 validation",
                reason: format!("requested id {id:?} is not servable: {error}"),
            });
        }
    }
    Ok(())
}

/// The typed witness surface of a backend: materialize one bundle type over
/// the full cycle domain. Implemented via the streaming pass
/// ([`crate::stream_witnesses`] with a collecting consumer), so backends and
/// the future streaming engine share the same walk.
pub trait BundleSource {
    fn bundles<B: WitnessBundle + Clone + Send + Sync>(&self) -> Result<Vec<B>, WitnessError>;
}

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

    /// The field-inline witness surface, when this backend serves one.
    /// Defaults to `None` so FR proving fails closed unless a backend
    /// explicitly attaches its field-inline view.
    #[cfg(feature = "field-inline")]
    fn field_inline(&self) -> Option<&dyn FieldInlineWitnessOracle<F>> {
        None
    }
}

/// The full program preprocessing behind the witness: the kernels whose
/// tables materialize from the program itself (the bytecode stage-value
/// fold, the reduction chunk grids, the program-image words) read it off the
/// witness plane inside `prepare`.
pub trait ProgramSource {
    fn program_preprocessing(&self) -> &JoltProgramPreprocessing;
}

/// The full witness plane a prover kernel prepares against: the id-indexed
/// oracle surface, the sequential row source (kernels collect their own
/// typed bundles through [`crate::collect_bundles`], so no stage recipe
/// stages row vectors on the side), and the program view.
/// Blanket-implemented; the supertrait set is exactly what kernels consume.
pub trait JoltWitnessPlane<F: Field>:
    JoltWitnessOracle<F> + RowSource + ProgramSource + Send + Sync
{
}

impl<F: Field, T> JoltWitnessPlane<F> for T where
    T: JoltWitnessOracle<F> + RowSource + ProgramSource + Send + Sync
{
}
