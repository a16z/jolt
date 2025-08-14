#![allow(missing_docs)]
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, Valid};
use ark_std::rand::RngCore;
use std::fmt::Debug;

/// --------- field ----------------------------------------------------------
pub trait Field:
    Sized + Clone + Copy + PartialEq + Send + Sync + CanonicalSerialize + CanonicalDeserialize + Valid
{
    fn zero() -> Self;
    fn one() -> Self;
    fn is_zero(&self) -> bool;

    fn add(&self, rhs: &Self) -> Self;
    fn sub(&self, rhs: &Self) -> Self;
    fn mul(&self, rhs: &Self) -> Self;
    fn inv(&self) -> Option<Self>;

    fn random<R: RngCore>(rng: &mut R) -> Self;

    fn from_u64(val: u64) -> Self;
    fn from_i64(val: i64) -> Self;
}

/// --------- group ----------------------------------------------------------
pub trait Group:
    Sized + Clone + PartialEq + Send + Sync + CanonicalSerialize + CanonicalDeserialize + Valid
{
    type Scalar: Field;

    fn identity() -> Self;
    fn add(&self, rhs: &Self) -> Self;
    fn neg(&self) -> Self;
    fn scale(&self, k: &Self::Scalar) -> Self;

    fn random<R: RngCore>(rng: &mut R) -> Self;
}

/// -------------------------------- pairing ----------------------------------
pub trait Pairing: Sized + Send + Sync {
    type G1: Group + Debug;
    type G2: Group + Debug;
    type GT: Group + Debug;

    /// e : G1 × G2 → GT
    fn pair(p: &Self::G1, q: &Self::G2) -> Self::GT;

    /// Multi-pairing: computes the product of pairings
    /// Π e(p_i, q_i)
    fn multi_pair(ps: &[Self::G1], qs: &[Self::G2]) -> Self::GT {
        assert_eq!(
            ps.len(),
            qs.len(),
            "multi_pair requires equal length vectors"
        );

        if ps.is_empty() {
            return Self::GT::identity();
        }

        ps.iter()
            .zip(qs.iter())
            .fold(Self::GT::identity(), |acc, (p, q)| {
                acc.add(&Self::pair(p, q))
            })
    }

    /// Multi-pairing with flexible caching support.
    ///
    /// For each side, you can either provide:
    /// - `points` + `None` cache: compute prepared values at runtime
    /// - `count` + `Some(cache)`: use first `count` cached prepared values
    ///
    /// This allows mixing cached generators (from setup) with runtime-computed points.
    fn multi_pair_cached(
        g1_points: Option<&[Self::G1]>,
        g1_count: Option<usize>,
        g1_cache: Option<&crate::curve::G1Cache>,
        g2_points: Option<&[Self::G2]>,
        g2_count: Option<usize>,
        g2_cache: Option<&crate::curve::G2Cache>,
    ) -> Self::GT;
}

pub trait MultiScalarMul<G: Group> {
    fn msm(bases: &[G], scalars: &[G::Scalar]) -> G;

    /// Fixed-base vectorized scalar multiplication where the same base is scaled by each scalar individually
    /// Computes: [base * scalars[0], base * scalars[1], ..., base * scalars[n-1]]
    fn fixed_base_vector_msm(
        base: &G,
        scalars: &[G::Scalar],
        g1_cache: Option<&crate::curve::G1Cache>,
        g2_cache: Option<&crate::curve::G2Cache>,
    ) -> Vec<G> {
        // Default implementation: scale each scalar individually
        // Caches are ignored in the default implementation
        let _ = (g1_cache, g2_cache);
        scalars.iter().map(|scalar| base.scale(scalar)).collect()
    }

    /// Fixed-scalar variable-base vectorized multiplication with add: vs[i] = vs[i] + scalar * bases[i]
    /// Modifies vs in place by adding the scaled bases
    /// This is optimized for cases like reduce_fold where we compute v_l = alpha * v_l + v_r
    fn fixed_scalar_variable_with_add(bases: &[G], vs: &mut [G], scalar: &G::Scalar) {
        assert_eq!(bases.len(), vs.len(), "bases and vs must have same length");
        // Default implementation: scale each base and add to vs
        for (base, v) in bases.iter().zip(vs.iter_mut()) {
            *v = v.add(&base.scale(scalar));
        }
    }

    /// Fixed-scalar variable-base vectorized multiplication with add using cached precomputed data
    /// vs[i] = vs[i] + scalar * bases[i] where bases come from cache
    ///
    /// This method allows using precomputed data when available, similar to multi_pair_cached.
    /// The default implementation panics - concrete implementations must override this.
    fn fixed_scalar_variable_with_add_cached(
        bases_count: usize,
        g1_cache: Option<&crate::curve::G1Cache>,
        g2_cache: Option<&crate::curve::G2Cache>,
        vs: &mut [G],
        scalar: &G::Scalar,
    ) {
        // Default implementation: panic as this must be implemented by concrete types
        // that know how to use the cache
        let _ = (bases_count, g1_cache, g2_cache, vs, scalar);
        panic!("fixed_scalar_variable_with_add_cached must be implemented by concrete MSM types");
    }

    /// Fixed-scalar vectorized multiplication with add: vs[i] = scalar * vs[i] + addends[i]
    /// Modifies vs in place by scaling each element and adding the corresponding addend
    /// This is optimized for cases like reduce_fold where we compute v_l = alpha * v_l + v_r
    fn fixed_scalar_scale_with_add(vs: &mut [G], addends: &[G], scalar: &G::Scalar) {
        assert_eq!(
            vs.len(),
            addends.len(),
            "vs and addends must have same length"
        );
        // Default implementation: scale each vs element and add the corresponding addend
        for (v, addend) in vs.iter_mut().zip(addends.iter()) {
            *v = v.scale(scalar).add(addend);
        }
    }
}
