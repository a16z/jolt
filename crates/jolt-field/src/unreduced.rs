//! Deferred-reduction contracts: the unreduced value algebra around a field
//! ([`Unreduced`]) and the per-element multilinear-bind hook ([`Fold`]).
//!
//! The CPU prover's hot loops sum hundreds of products per output slot.
//! Reducing every product is wasted work when the products can be widened
//! into integer accumulators, summed with plain (carry-free or wrapping)
//! adds, and reduced once at the end. [`Unreduced`] is the single surface
//! for that pattern: it names the accumulator types and routes every
//! reduction back through the field type, so a backend's unreduced algebra
//! is enumerable from one `impl`.

use crate::{AdditiveGroup, Field};

/// The deferred-reduction companion surface of a field.
///
/// Three accumulator shapes cover the prover's patterns:
///
/// - [`Product`](Self::Product): full `Self × Self` widening products
///   ([`mul_unreduced`](Self::mul_unreduced)). Addition is wrapping
///   per-slot, i.e. the accumulator is the group `(Z/2^128)^n`; a sum
///   reduces exactly whenever the final integer value of every slot is
///   below `2^128` (intermediate dips below zero cancel exactly).
/// - [`SmallProduct`](Self::SmallProduct): narrower `Self × u64` products
///   ([`mul_u64_unreduced`](Self::mul_u64_unreduced)).
/// - [`Wide`](Self::Wide): a carry-free signed accumulator over `i32`
///   lanes for sums of small-scalar multiples
///   ([`scale_wide`](Self::scale_wide)); lane overflow bounds are
///   documented on the concrete lane types.
///
/// Each shape has a matching `reduce_*` back to a canonical element. The
/// per-type headroom (how many products fit before a slot can overflow) is
/// documented where the accumulation formula lives; **no runtime check
/// enforces it** beyond debug-mode overflow panics on the non-wrapping
/// lane types.
pub trait Unreduced: Field {
    /// Accumulator for full `Self × Self` widening products.
    type Product: AdditiveGroup;

    /// Accumulator for `Self × u64` widening products.
    type SmallProduct: AdditiveGroup;

    /// Carry-free `i32`-lane accumulator for small-scalar multiples.
    type Wide: AdditiveGroup + From<Self>;

    /// Whether delayed reduction over [`Product`](Self::Product) is exact:
    /// `reduce_product(Σᵢ mul_unreduced(aᵢ, bᵢ)) = Σᵢ aᵢ·bᵢ` for batches
    /// within the accumulator's documented headroom.
    ///
    /// Conservative default `false`; a field opts in only once its
    /// accumulator is proven exact. Callers that must stay term-for-term
    /// identical to `Mul` keep the per-term reduce path when this is
    /// `false`.
    const SUM_IS_EXACT: bool = false;

    /// Widening `self × other` with no reduction.
    fn mul_unreduced(self, other: Self) -> Self::Product;

    /// Widening `self × small` with no reduction.
    fn mul_u64_unreduced(self, small: u64) -> Self::SmallProduct;

    /// `self × small` as a wide lane value (equal to
    /// `Self::Wide::from(self)` scaled lane-wise by `small`).
    fn scale_wide(self, small: i32) -> Self::Wide;

    /// Reduces a full-product accumulator to a canonical element.
    fn reduce_product(accum: Self::Product) -> Self;

    /// Reduces a small-product accumulator to a canonical element.
    fn reduce_small_product(accum: Self::SmallProduct) -> Self;

    /// Reduces a wide lane accumulator to a canonical element.
    fn reduce_wide(wide: Self::Wide) -> Self;
}

/// Marks a field whose wide accumulator supports bounded unit-scale
/// commitment streams.
///
/// The accumulator is [`Unreduced::Wide`]. This separate role contract keeps
/// the commitment headroom out of extension fields and other unreduced types
/// that do not use this accumulation pattern.
pub trait WithCommitAccumulator: Unreduced {
    /// Maximum unit-scale additions before any signed lane can overflow.
    const MAX_COMMIT_ACCUMULATIONS: usize;
}

/// Per-element multilinear bind: `even + r·(odd − even)` for a challenge
/// `r` fixed across a whole polynomial-binding round.
///
/// This is a protocol-support hook rather than field algebra; it lives here
/// because implementations exploit the field representation — precomputing
/// a multiplication-by-`r` matrix from the challenge and folding each pair
/// with fewer reductions than a generic extension multiply. Implementations
/// must return exactly the canonical value of `even + r·(odd − even)`; the
/// loop structure and parallelism belong to the caller.
pub trait Fold: Field {
    /// Precomputed context for folding by a fixed challenge `r`.
    type Ctx: Copy + Send + Sync;

    /// Builds the fold context from the challenge `r`.
    fn precompute(r: Self) -> Self::Ctx;

    /// Folds one pair: `even + r·(odd − even)`.
    fn fold_one(ctx: &Self::Ctx, even: Self, odd: Self) -> Self;
}
