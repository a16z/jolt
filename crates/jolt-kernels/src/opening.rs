//! The stage-8 joint-opening materialization slot: the committed polynomials
//! embedded over the shared commitment grid, ready for the PCS batch opening.
//!
//! The batch opening RLCs every committed polynomial at one unified point over
//! the full grid domain, so each polynomial must present `grid.total_vars`
//! variables. Cycle-major: the one-hot grids span it natively, the dense
//! trace polynomials occupy a low-index prefix and zero-extend.
//! Address-major: every trace polynomial scatters cycle-block-strided —
//! coefficient `(k, t)` at grid index `t · cycle_stride + k · one_hot_stride`
//! (the witness's native `k · T + t` views permute, dense polynomials sit at
//! address slot zero) — matching the address-major commit placement and the
//! verifier's `commitment_embedding_scale` under the `[r_cycle ‖ r_address]`
//! unified point. In both orders the precommitted polynomials (advice,
//! bytecode chunks, program image) BLOCK-embed — their own balanced matrix
//! (`2^σ_p` columns) lands in the grid matrix's top-left corner, so
//! coefficient `row · 2^σ_p + col` sits at grid index `row · 2^σ_main + col`
//! (strided, not contiguous; the legacy `vmp_precommitted_contribution`
//! layout the commitment and `commitment_embedding_scale` agree on). The
//! trace order enters a chunk table only through its coefficient
//! interleaving, which the recipe-supplied tables already carry. The slot
//! returns [`MultilinearPoly`] objects because the PCS opening drives them
//! lazily (`fold_rows`).

use std::collections::BTreeMap;

use jolt_claims::protocols::jolt::{
    JoltAdviceKind, JoltCommittedPolynomial, ProgramImageClaimReductionLayout,
};
use jolt_field::JoltField;
use jolt_poly::{sparse_segments_mle_msb, MultilinearPoly};
use jolt_witness::JoltWitnessPlane;

use crate::commitment::CommitmentGrid;
use crate::{KernelError, ProofSession};

/// A consuming factory for host committed-program opening tables. Backends
/// holding these polynomials in device memory need not materialize host copies.
pub type PrecommittedOpeningTables<'a, F> =
    Box<dyn FnOnce() -> Result<BTreeMap<JoltCommittedPolynomial, Vec<F>>, KernelError<F>> + 'a>;

/// The stage-8 joint-opening polynomial slot: materialize `polynomials` (in
/// the given order — the final-opening batch order) embedded over `grid`.
/// `precommitted_tables` materializes the committed-program polynomials (bytecode
/// chunks, program image) from prover-retained preprocessing when invoked. A
/// resident backend can omit that work. Host implementations consume the factory
/// once and move its tables into the opened polynomials without cloning them.
/// Receives the full witness plane (not just the oracle surface) so
/// implementations can rebuild the committed columns from typed trace
/// bundles instead of materialized `K × T` oracle grids.
pub trait JointOpeningPolynomials<F: JoltField> {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        polynomials: &[JoltCommittedPolynomial],
        precommitted_tables: PrecommittedOpeningTables<'_, F>,
        grid: CommitmentGrid,
    ) -> Result<Vec<Box<dyn MultilinearPoly<F>>>, KernelError<F>>;
}

/// A private contribution to stage 4's initial RAM evaluation. Points are
/// big-endian; the program image uses the full RAM address point, while
/// advice uses its block's address sub-point.
pub enum RamInitialOpening<'a, F: JoltField> {
    ProgramImage {
        layout: &'a ProgramImageClaimReductionLayout,
        point: &'a [F],
    },
    Advice {
        kind: JoltAdviceKind,
        point: &'a [F],
    },
}

/// Evaluate stage 4's private initial-RAM contributions together so a
/// device backend can share one batch. Return one scalar per request, in
/// request order. This slot neither draws challenges nor absorbs claims;
/// the stage coordinator owns those and the later PCS opening proof.
pub trait RamInitialOpeningEvaluation<F: JoltField> {
    fn evaluate(
        &self,
        session: &mut ProofSession,
        openings: &[RamInitialOpening<'_, F>],
        witness: &dyn JoltWitnessPlane<F>,
    ) -> Result<Vec<F>, KernelError<F>>;
}

pub(crate) fn evaluate_program_image<F: JoltField>(
    layout: &ProgramImageClaimReductionLayout,
    point: &[F],
    witness: &dyn JoltWitnessPlane<F>,
) -> F {
    let program = witness.program_preprocessing();
    sparse_segments_mle_msb(
        std::iter::once((
            layout.start_index() as u128,
            program.ram.bytecode_words.as_slice(),
        )),
        point,
    )
}
