use allocative::Allocative;

use crate::field::{ChallengeFieldOps, FieldChallengeOps, JoltField};

pub mod commitment;
pub mod compact_polynomial;
pub mod dense_mlpoly;
pub mod eq_poly;
pub mod identity_poly;
pub mod multilinear_polynomial;
pub mod one_hot_polynomial;
pub mod opening_proof;
pub mod prefix_suffix;
pub mod program_io_polynomial;
pub mod ra_poly;
pub mod range_mask_polynomial;
pub mod rlc_polynomial;
pub mod spartan_interleaved_poly;
pub mod split_eq_poly;
pub mod unipoly;

/// The order in which polynomial variables are bound in sumcheck
#[derive(Clone, Copy, Debug, PartialEq, Allocative, Default)]
pub enum BindingOrder {
    #[default]
    LowToHigh,
    HighToLow,
}

pub trait PolynomialBinding<F: JoltField> {
    /// Returns whether or not the polynomial has been bound (in a sumcheck)
    fn is_bound(&self) -> bool;
    /// Binds the polynomial to a random field element `r`.
    fn bind(&mut self, r: F::Challenge, order: BindingOrder);
    /// Returns the final sumcheck claim about the polynomial.
    fn final_sumcheck_claim(&self) -> F;
}

pub trait PolynomialEvaluation<F: JoltField> {
    /// Returns the final sumcheck claim about the polynomial.
    /// This uses the algorithm in Lemma 4.3 in Thaler, Proofs and
    /// Arguments -- the point at which we evaluate the polynomial
    fn evaluate<C>(&self, r: &[C]) -> F
    where
        C: Copy + Send + Sync + Into<F> + ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>;

    /// Evaluates a batch of polynomials on the same point `r`.
    /// Returns: (evals, EQ table)
    /// where EQ table is EQ(x, r) for x \in {0, 1}^|r|. This is used for
    /// batched opening proofs (see opening_proof.rs)
    fn batch_evaluate<C>(polys: &[&Self], r: &[C]) -> Vec<F>
    where
        Self: Sized,
        C: Copy + Send + Sync + Into<F> + ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>;
    /// Computes this polynomial's contribution to the computation of a prover
    /// sumcheck message (i.e. a univariate polynomial of the given `degree`).
    fn sumcheck_evals(&self, index: usize, degree: usize, order: BindingOrder) -> Vec<F>;
}
