//! Defines the standards for messages between the prover and verifier
use crate::arithmetic::{Field, Group};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use serde::{Deserialize, Serialize};

/// The first prover message in the Dory-Reduce portion (Section 3.2) of the Dory protocol.
///
/// This consists of $D_{1L}$, $D_{1R}$, $D_{2L}$, $D_{2R}$, $E_{1\beta}$, and $E_{2\beta}$.
#[derive(
    Clone,
    Debug,
    Default,
    Deserialize,
    Eq,
    Hash,
    PartialEq,
    Serialize,
    CanonicalSerialize,
    CanonicalDeserialize,
)]
pub struct FirstReduceMessage<G1, G2, GT>
where
    G1: Group,
    G2: Group,
    GT: Group,
{
    /// $D_{1L}$
    pub d1_left: GT,
    /// $D_{1R}$
    pub d1_right: GT,
    /// $D_{2L}$
    pub d2_left: GT,
    /// $D_{2R}$
    pub d2_right: GT,
    /// $E_{1\beta}$ (extension - Section 4.2 of paper)
    pub e1_beta: G1,
    /// $E_{2\beta}$ (extension - Section 4.2 of paper)
    pub e2_beta: G2,
}

/// The the first verifier challenge in the Dory-Reduce portion (Section 3.2) of the Dory protocol.
///
/// The challenge, $\beta$, is a random scalar. Additionally, $\beta$ must be non-zero because
/// the protocol uses $\beta^{-1}$, which we also include here.
#[derive(
    Clone,
    Debug,
    Default,
    Deserialize,
    Eq,
    Hash,
    PartialEq,
    Serialize,
    CanonicalSerialize,
    CanonicalDeserialize,
)]
pub struct FirstReduceChallenge<Scalar>
where
    Scalar: Field,
{
    /// $\beta$
    pub beta: Scalar,
    /// $\beta^{-1}$
    pub beta_inverse: Scalar,
}

/// The second prover message in the Dory-Reduce portion (Section 3.2) of the Dory protocol.
///
/// This consists of $C_+$, $C_-$, $E_{1+}$, $E_{1-}$, $E_{2+}$, and $E_{2-}$.
#[derive(
    Clone,
    Debug,
    Default,
    Deserialize,
    Eq,
    Hash,
    PartialEq,
    Serialize,
    CanonicalSerialize,
    CanonicalDeserialize,
)]
pub struct SecondReduceMessage<G1, G2, GT>
where
    G1: Group,
    G2: Group,
    GT: Group,
{
    /// $C_+$
    pub c_plus: GT,
    /// $C_-$
    pub c_minus: GT,
    /// $E_{1+}$ (extension - Section 4.2 of paper)
    pub e1_plus: G1,
    /// $E_{1-}$ (extension - Section 4.2 of paper)
    pub e1_minus: G1,
    /// $E_{2+}$ (extension - Section 4.2 of paper)
    pub e2_plus: G2,
    /// $E_{2-}$ (extension - Section 4.2 of paper)
    pub e2_minus: G2,
}

/// The second verifier challenge in the Dory-Reduce portion (Section 3.2) of the Dory protocol.
///
/// The challenge, $\alpha$, is a random scalar. Additionally, $\alpha$ must be non-zero because
/// the protocol uses $\alpha^{-1}$, which we also include here.
#[derive(
    Clone,
    Debug,
    Default,
    Deserialize,
    Eq,
    Hash,
    PartialEq,
    Serialize,
    CanonicalSerialize,
    CanonicalDeserialize,
)]
pub struct SecondReduceChallenge<Scalar>
where
    Scalar: Field,
{
    /// $\alpha$
    pub alpha: Scalar,
    /// $\alpha^{-1}$
    pub alpha_inverse: Scalar,
}

/// This is the random chalenge (d, d^-1) for the Scalar-Product protocol
#[derive(
    Clone,
    Debug,
    Default,
    Deserialize,
    Eq,
    Hash,
    PartialEq,
    Serialize,
    CanonicalSerialize,
    CanonicalDeserialize,
)]
pub struct ScalarProductChallenge<Scalar>
where
    Scalar: Field,
{
    /// d challenge for Scalar-Product procedure
    pub d: Scalar,
    /// d^-1 challenge for Scalar-Product procedure
    pub d_inverse: Scalar,
}

/// The verifier challenge in the Fold-Scalars portion (Section 4.1) of the Dory protocol.
///
/// The challenge, $\gamma$, is a random scalar. Additionally, $\gamma$ must be non-zero because
/// the protocol uses $\gamma^{-1}$, which we also include here.
#[derive(
    Clone,
    Debug,
    Default,
    Deserialize,
    Eq,
    Hash,
    PartialEq,
    Serialize,
    CanonicalSerialize,
    CanonicalDeserialize,
)]
pub struct FoldScalarsChallenge<Scalar>
where
    Scalar: Field,
{
    /// $\gamma$
    pub gamma: Scalar,
    /// $\gamma^{-1}$
    pub gamma_inverse: Scalar,
}

/// The prover message in the Scalar-Product portion (Section 3.1) of the Dory protocol.
///
/// This consists of $E_1$ and $E_2$.
#[derive(
    Clone,
    Debug,
    Default,
    Deserialize,
    Eq,
    Hash,
    PartialEq,
    Serialize,
    CanonicalSerialize,
    CanonicalDeserialize,
)]
pub struct ScalarProductMessage<G1, G2>
where
    G1: Group,
    G2: Group,
{
    /// $E_1$
    pub e1: G1,
    /// $E_2$
    pub e2: G2,
}

/// The prover message in the VMV evaluation portion of the Dory protocol.
///
/// This consists of $C$, $D_2$, and $E_1$.
/// Note that the paper also includes `e2` but it turns out that the verifier
/// can compute this value, since $e2 = y \cdot \Gamma_{2,fin}$, which are known to V.
#[derive(
    Clone,
    Debug,
    Default,
    Deserialize,
    Eq,
    Hash,
    PartialEq,
    Serialize,
    CanonicalSerialize,
    CanonicalDeserialize,
)]
pub struct VMVMessage<G1, GT>
where
    G1: Group,
    GT: Group,
{
    /// $C$ = e(MSM(T_vec_prime, v_vec), Gamma_2_fin)
    pub c: GT,
    /// $D_2$ = e(MSM(Gamma_1[nu], v_vec), Gamma_2_fin)
    pub d2: GT,
    /// $E_1$ = MSM(T_vec_prime, L_vec)
    pub e1: G1,
}
