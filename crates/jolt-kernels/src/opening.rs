//! The stage-8 joint-opening materialization slot: the committed polynomials
//! embedded over the shared commitment grid, ready for the PCS batch opening.
//!
//! The batch opening RLCs every committed polynomial at one unified point over
//! the full grid domain, so each polynomial must present `grid.total_vars`
//! variables: the one-hot grids span it natively, the dense trace
//! polynomials occupy a low-index prefix and zero-extend, and the advice
//! polynomials BLOCK-embed — their own balanced matrix (`2^σ_a` columns)
//! lands in the grid matrix's top-left corner, so advice coefficient
//! `row · 2^σ_a + col` sits at grid index `row · 2^σ_main + col` (strided,
//! not contiguous; the legacy `vmp_precommitted_contribution` layout the
//! commitment and `commitment_embedding_scale` agree on). WARNING: the
//! prefix embedding matches the verifier's `commitment_embedding_scale` only
//! under `TracePolynomialOrder::CycleMajor` (the address coordinates lead the
//! unified point); the address-major layout needs a strided embedding — the
//! recipe guards it off. The slot returns [`MultilinearPoly`] objects because
//! the PCS opening drives them lazily (`fold_rows`); the reference impl
//! materializes every table dense and simultaneously (a test oracle at
//! harness scale, never a performance path — an optimized backend returns
//! lazy/sparse or device-backed implementations).

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
                let table = dense_view(witness, final_opening_id(polynomial))?;
                if table.len() > domain {
                    return Err(KernelError::TableSizeMismatch {
                        table: format!("{polynomial:?}"),
                        expected: domain,
                        got: table.len(),
                    });
                }
                let embedded = match polynomial {
                    JoltCommittedPolynomial::TrustedAdvice
                    | JoltCommittedPolynomial::UntrustedAdvice => {
                        block_embed(&table, grid, polynomial)?
                    }
                    _ => {
                        let mut table = table;
                        table.resize(domain, F::zero());
                        table
                    }
                };
                Ok(Box::new(embedded) as Box<dyn MultilinearPoly<F>>)
            })
            .collect()
    }
}

/// Embed an advice polynomial's balanced matrix into the grid matrix's
/// top-left block: advice coefficient `row · 2^σ_a + col` lands at grid index
/// `row · 2^σ_main + col`.
fn block_embed<F: Field>(
    table: &[F],
    grid: CommitmentGrid,
    polynomial: JoltCommittedPolynomial,
) -> Result<Vec<F>, KernelError<F>> {
    if !table.len().is_power_of_two() {
        return Err(KernelError::TableSizeMismatch {
            table: format!("{polynomial:?}"),
            expected: table.len().next_power_of_two(),
            got: table.len(),
        });
    }
    let advice_vars = table.len().ilog2() as usize;
    let sigma_advice = advice_vars.div_ceil(2);
    let column_mask = (1usize << sigma_advice) - 1;
    let sigma_main = grid.total_vars.div_ceil(2);
    let mut embedded = vec![F::zero(); 1usize << grid.total_vars];
    for (index, value) in table.iter().enumerate() {
        let row = index >> sigma_advice;
        let column = index & column_mask;
        embedded[(row << sigma_main) | column] = *value;
    }
    Ok(embedded)
}
