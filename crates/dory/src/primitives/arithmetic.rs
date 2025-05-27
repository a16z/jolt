#![allow(missing_docs)]
use ark_std::rand::RngCore;

/// --------- field ----------------------------------------------------------
pub trait Field: Sized + Clone + PartialEq + Send + Sync {
    fn zero() -> Self;
    fn one() -> Self;

    fn add(&self, rhs: &Self) -> Self;
    fn sub(&self, rhs: &Self) -> Self;
    fn mul(&self, rhs: &Self) -> Self;
    fn inv(&self) -> Option<Self>;

    fn random<R: RngCore>(rng: &mut R) -> Self;
}

/// --------- group ----------------------------------------------------------
pub trait Group: Sized + Clone + PartialEq + Send + Sync {
    type Scalar: Field;

    fn identity() -> Self;
    fn add(&self, rhs: &Self) -> Self;
    fn neg(&self) -> Self;
    fn scale(&self, k: &Self::Scalar) -> Self;

    fn random<R: RngCore>(rng: &mut R) -> Self;
}

/// -------------------------------- pairing ----------------------------------
pub trait Pairing: Sized + Send + Sync {
    type G1: Group;
    type G2: Group;
    type GT: Group;

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
}

pub trait MultiScalarMul<G: Group> {
    fn msm(bases: &[G], scalars: &[G::Scalar]) -> G {
        assert_eq!(
            bases.len(),
            scalars.len(),
            "MSM requires equal length vectors"
        );

        if bases.is_empty() {
            return G::identity();
        }

        bases
            .iter()
            .zip(scalars)
            .fold(G::identity(), |acc, (base, scalar)| {
                acc.add(&base.scale(scalar))
            })
    }
}
