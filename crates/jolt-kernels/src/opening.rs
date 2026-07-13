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

use jolt_claims::protocols::jolt::JoltCommittedPolynomial;
use jolt_field::Field;
use jolt_poly::MultilinearPoly;
use jolt_witness::protocols::jolt_vm::JoltVmNamespace;
use jolt_witness::WitnessProvider;

use crate::commitment::CommitmentGrid;
use crate::{KernelError, ProofSession};

/// The stage-8 joint-opening polynomial slot: materialize `polynomials` (in
/// the given order — the final-opening batch order) embedded over `grid`.
/// `precommitted_tables` carries the committed-program polynomials (bytecode
/// chunks, program image) the recipe materialized from the prover-retained
/// full program — they are preprocessing data, not witness oracles.
pub trait JointOpeningPolynomials<F: Field> {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn WitnessProvider<F, JoltVmNamespace>,
        polynomials: &[JoltCommittedPolynomial],
        precommitted_tables: &BTreeMap<JoltCommittedPolynomial, Vec<F>>,
        grid: CommitmentGrid,
    ) -> Result<Vec<Box<dyn MultilinearPoly<F>>>, KernelError<F>>;
}
