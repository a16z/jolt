#![allow(missing_docs)]

use super::{DoryDeserialize, DorySerialize};

pub trait Field:
    Sized
    + Clone
    + Copy
    + PartialEq
    + Send
    + Sync
    + DorySerialize
    + DoryDeserialize
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Neg<Output = Self>
    + for<'a> std::ops::Add<&'a Self, Output = Self>
    + for<'a> std::ops::Sub<&'a Self, Output = Self>
    + for<'a> std::ops::Mul<&'a Self, Output = Self>
{
    fn zero() -> Self;
    fn one() -> Self;
    fn is_zero(&self) -> bool;

    fn add(&self, rhs: &Self) -> Self;
    fn sub(&self, rhs: &Self) -> Self;
    fn mul(&self, rhs: &Self) -> Self;

    fn inv(self) -> Option<Self>;

    fn random() -> Self;

    fn from_u64(val: u64) -> Self;
    fn from_i64(val: i64) -> Self;
}

pub trait Group:
    Sized
    + Clone
    + Copy
    + PartialEq
    + Send
    + Sync
    + DorySerialize
    + DoryDeserialize
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Neg<Output = Self>
    + for<'a> std::ops::Add<&'a Self, Output = Self>
    + for<'a> std::ops::Sub<&'a Self, Output = Self>
{
    type Scalar: Field
        + std::ops::Mul<Self, Output = Self>
        + for<'a> std::ops::Mul<&'a Self, Output = Self>;

    fn identity() -> Self;
    fn add(&self, rhs: &Self) -> Self;
    fn neg(&self) -> Self;
    fn scale(&self, k: &Self::Scalar) -> Self;

    fn random() -> Self;
}

pub trait PairingCurve: Clone {
    type G1: Group;
    type G2: Group;
    type GT: Group; // Multiplicative subgroup F^* of the extension field

    /// Optional device-resident transparent reduce-loop implementation.
    /// Backends that do not install one keep the published host loop.
    fn resident_round_hooks() -> Option<ResidentRoundHooks<Self>>
    where
        Self: Sized,
    {
        None
    }

    /// e : G1 × G2 → GT
    fn pair(p: &Self::G1, q: &Self::G2) -> Self::GT;

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

    /// Optimized multi-pairing when G2 points come from setup/generators
    ///
    /// This variant should be used when the G2 points are from the prover setup
    /// (e.g., g2_vec generators). Backend implementations can optimize this by
    /// caching prepared G2 points.
    ///
    /// # Parameters
    /// - `ps`: G1 points (typically computed values like row commitments or v-vectors)
    /// - `qs`: G2 points from setup (e.g., `setup.g2_vec[..n]`)
    ///
    /// # Returns
    /// Product of pairings: Π e(p_i, q_i)
    ///
    /// # Default Implementation
    /// Delegates to `multi_pair`
    fn multi_pair_g2_setup(ps: &[Self::G1], qs: &[Self::G2]) -> Self::GT {
        Self::multi_pair(ps, qs)
    }

    /// Optimized multi-pairing when G1 points are from the prover setup.
    ///
    /// This variant should be used when the G1 points are from the prover setup
    /// (e.g., g1_vec generators). Backend implementations can optimize this by
    /// caching prepared G1 points.
    ///
    /// # Parameters
    /// - `ps`: G1 points from setup (e.g., `setup.g1_vec[..n]`)
    /// - `qs`: G2 points (typically computed values like v-vectors)
    ///
    /// # Returns
    /// Product of pairings: Π e(p_i, q_i)
    ///
    /// # Default Implementation
    /// Delegates to `multi_pair`
    fn multi_pair_g1_setup(ps: &[Self::G1], qs: &[Self::G2]) -> Self::GT {
        Self::multi_pair(ps, qs)
    }
}

/// Opaque backend state retained across Dory reduce rounds.
pub type ResidentRoundState = Box<dyn std::any::Any>;

/// Successful device-loop start and its leading-round budget.
pub struct ResidentRoundStart {
    pub state: ResidentRoundState,
    pub rounds: usize,
}

/// Value-exact backend operations for the transparent reduce-round prefix.
/// Transcript absorption remains in `evaluation_proof`.
pub struct ResidentRoundHooks<E: PairingCurve> {
    pub plan: fn(usize) -> usize,
    pub start: fn(
        &[E::G1],
        &[E::G2],
        &[E::G1],
        &[E::G2],
    ) -> Option<ResidentRoundStart>,
    /// The trailing `Option` is the first round's `v2_scalars` (`v2 =
    /// h2·scalars`), letting a backend serve D₂ as MSM+pair exactly like the
    /// host arm's `compute_d2`; `None` after the first challenge.
    pub first_message: fn(
        &mut ResidentRoundState,
        &[<E::G1 as Group>::Scalar],
        &[<E::G1 as Group>::Scalar],
        Option<&[<E::G1 as Group>::Scalar]>,
    ) -> crate::messages::FirstReduceMessage<E::G1, E::G2, E::GT>,
    pub apply_first: fn(
        &mut ResidentRoundState,
        &<E::G1 as Group>::Scalar,
        &<E::G1 as Group>::Scalar,
    ),
    pub second_message: fn(
        &mut ResidentRoundState,
        &[<E::G1 as Group>::Scalar],
        &[<E::G1 as Group>::Scalar],
    ) -> crate::messages::SecondReduceMessage<E::G1, E::G2, E::GT>,
    pub apply_second: fn(
        &mut ResidentRoundState,
        &<E::G1 as Group>::Scalar,
        &<E::G1 as Group>::Scalar,
    ),
    pub finish: fn(ResidentRoundState) -> (Vec<E::G1>, Vec<E::G2>),
}

impl<E: PairingCurve> Copy for ResidentRoundHooks<E> {}

impl<E: PairingCurve> Clone for ResidentRoundHooks<E> {
    fn clone(&self) -> Self {
        *self
    }
}

/// Dory requires MSMs and vector scaling ops, hence we expose a trait for optimized versions of such routines.
pub trait DoryRoutines<G: Group> {
    fn msm(bases: &[G], scalars: &[G::Scalar]) -> G;

    /// Fixed-base vectorized scalar multiplication where the same base is scaled by each scalar individually
    /// Computes: \[base * scalars\[0\], base * scalars\[1\], ..., base * scalars\[n-1\]\]
    fn fixed_base_vector_scalar_mul(base: &G, scalars: &[G::Scalar]) -> Vec<G>;

    /// vs\[i\] = vs\[i\] + scalar * bases\[i\]
    fn fixed_scalar_mul_bases_then_add(bases: &[G], vs: &mut [G], scalar: &G::Scalar);

    /// vs\[i\] = scalar * vs\[i\] + addends\[i\]
    fn fixed_scalar_mul_vs_then_add(vs: &mut [G], addends: &[G], scalar: &G::Scalar);

    /// Fold field vectors: left\[i\] = left\[i\] * scalar + right\[i\]
    fn fold_field_vectors(left: &mut [G::Scalar], right: &[G::Scalar], scalar: &G::Scalar) {
        assert_eq!(left.len(), right.len(), "Lengths must match");
        for i in 0..left.len() {
            left[i] = left[i] * *scalar + right[i];
        }
    }
}
