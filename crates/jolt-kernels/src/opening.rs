//! The stage-8 joint-opening materialization slot: the committed polynomials
//! embedded over the shared commitment grid, ready for the PCS batch opening.
//!
//! The batch opening RLCs every committed polynomial at one unified point over
//! the full grid domain, so each polynomial must present `grid.total_vars`
//! variables: the one-hot grids span it natively, while the dense trace
//! polynomials occupy a low-index prefix and zero-extend. WARNING: the
//! prefix embedding matches the verifier's `commitment_embedding_scale` only
//! under `TracePolynomialOrder::CycleMajor` (the address coordinates lead the
//! unified point); the address-major layout needs a strided embedding, and
//! advice needs a block embedding — the recipe guards both off. The slot
//! returns [`MultilinearPoly`] objects because the PCS opening drives them
//! lazily (`fold_rows`); the reference impl materializes every table dense
//! and simultaneously (a test oracle at harness scale, never a performance
//! path — an optimized backend returns lazy/sparse or device-backed
//! implementations).

use jolt_claims::protocols::jolt::geometry::committed_openings::final_opening_id;
use jolt_claims::protocols::jolt::JoltCommittedPolynomial;
use jolt_field::Field;
use jolt_poly::MultilinearPoly;
use jolt_witness::protocols::jolt_vm::JoltVmNamespace;
use jolt_witness::WitnessProvider;

use crate::commitment::CommitmentGrid;
use crate::views::dense_view;
use crate::{KernelError, ProofSession, ReferenceBackend};

/// The stage-8 joint-opening polynomial slot: materialize `polynomials` (in
/// the given order — the final-opening batch order) embedded over `grid`.
pub trait JointOpeningPolynomials<F: Field> {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn WitnessProvider<F, JoltVmNamespace>,
        polynomials: &[JoltCommittedPolynomial],
        grid: CommitmentGrid,
    ) -> Result<Vec<Box<dyn MultilinearPoly<F>>>, KernelError<F>>;
}

impl<F: Field> JointOpeningPolynomials<F> for ReferenceBackend {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn WitnessProvider<F, JoltVmNamespace>,
        polynomials: &[JoltCommittedPolynomial],
        grid: CommitmentGrid,
    ) -> Result<Vec<Box<dyn MultilinearPoly<F>>>, KernelError<F>> {
        let domain = 1usize << grid.total_vars;
        polynomials
            .iter()
            .map(|&polynomial| {
                let mut table = dense_view(witness, final_opening_id(polynomial))?;
                if table.len() > domain {
                    return Err(KernelError::TableSizeMismatch {
                        table: format!("{polynomial:?}"),
                        expected: domain,
                        got: table.len(),
                    });
                }
                table.resize(domain, F::zero());
                Ok(Box::new(table) as Box<dyn MultilinearPoly<F>>)
            })
            .collect()
    }
}
