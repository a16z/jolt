//! Defines the structures which manage state during interactive execution of the prover and verifier
use crate::{
    arithmetic::{Field, Group, MultiScalarMul, Pairing},
    messages::{
        FirstReduceChallenge, FirstReduceMessage, FoldScalarsChallenge, ScalarProductMessage,
        SecondReduceChallenge, SecondReduceMessage,
    },
};

use super::ScalarProductChallenge;

/// Trait for the state and computation and state of the Dory protocol.
///
/// A type implementing this trait primarily stores the $v_i$ and $s_i$ vectors.
/// The trait methods define the operations needed to compute the messages exchanged.
/// This trait is not responsible for the actual messaging/proving. That is the job of the
/// [`ProofBuilder`](crate::ProofBuilder) trait. This is so that P and V actors can compute things as needed.
pub trait ProverState {
    /// The $\mathbb{G}_1$ group
    type G1: Group;
    /// The $\mathbb{G}_2$ group
    type G2: Group;
    /// The target group, $\mathbb{G}_T$
    type GT: Group;
    /// The scalar, $\mathbb{F}$, field of the groups
    type Scalar: Field;
    /// The setup type. This should contain the public parameters needed for the protocol.
    type Setup;

    /// Computes the [`FirstReduceMessage`] from the state.
    /// That is,
    /// $$\begin{aligned}
    /// D_{1L} &= \langle\vec{v_{1L}},\Gamma_2^{\prime}\rangle & D_{1R} &= \langle\vec{v_{1R}},\Gamma_2^{\prime}\rangle \\\\
    /// D_{2L} &= \langle\Gamma_1^{\prime},\vec{v_{2L}}\rangle & D_{2R} &= \langle\Gamma_1^{\prime},\vec{v_{2R}}\rangle \\\\
    /// E_{1\beta} &= \langle\Gamma_1,\vec{s_2}\rangle & E_{2\beta} &= \langle\vec{s_1},\Gamma_2\rangle
    /// \end{aligned}$$
    ///
    /// # Panics
    /// Panics if the state is not in an appropriate round. That is, if the last Reduce round has not been completed. This method
    /// assumes that the $v_i$ and $s_i$ vectors are of length at least 2.
    #[must_use]
    fn compute_first_reduce_message<M1, M2>(
        &self,
        setup: &Self::Setup,
    ) -> FirstReduceMessage<Self::G1, Self::G2, Self::GT>
    where
        Self::G1: Group,
        Self::G2: Group,
        M1: MultiScalarMul<Self::G1>,
        M2: MultiScalarMul<Self::G2>;
    /// Combines $\vec{v_i}$ and $\Gamma_i$ using the [`FirstReduceChallenge`].
    /// That is,
    /// $$\begin{aligned}
    /// \vec{v_1} &\leftarrow \vec{v_1} + \beta\Gamma_1 & \vec{v_2} &\leftarrow \vec{v_2} + \beta^{-1}\Gamma_2
    /// \end{aligned}$$
    ///
    /// # Panics
    /// Panics if the state is not in an appropriate round. That is, if the last Reduce round has not been completed. This method
    /// assumes that the $v_i$ and $s_i$ vectors are of length at least 2.
    #[must_use]
    fn reduce_combine<M1, M2>(
        self,
        setup: &Self::Setup,
        first_challenge: FirstReduceChallenge<Self::Scalar>,
    ) -> Self
    where
        M1: MultiScalarMul<Self::G1>,
        M2: MultiScalarMul<Self::G2>;
    /// Computes the [`SecondReduceMessage`] from the state.
    /// That is,
    /// $$\begin{aligned}
    /// C_+ &= \langle\vec{v_{1L}}, \vec{v_{2R}}\rangle & C_- &= \langle\vec{v_{1R}}, \vec{v_{2L}}\rangle \\\\
    /// E_{1+} &= \langle\vec{v_{1L}}, \vec{s_{2R}}\rangle & E_{1-} &= \langle\vec{v_{1R}}, \vec{s_{2L}}\rangle \\\\
    /// E_{2+} &= \langle\vec{s_{1L}}, \vec{v_{2R}}\rangle & E_{2-} &= \langle\vec{s_{1R}}, \vec{v_{2L}}\rangle
    /// \end{aligned}$$
    ///
    /// # Panics
    /// Panics if the state is not in an appropriate round. That is, if the last Reduce round has not been completed. This method
    /// assumes that the $v_i$ and $s_i$ vectors are of length at least 2.
    #[must_use]
    fn compute_second_reduce_message<M1, M2>(
        &self,
        setup: &Self::Setup,
    ) -> SecondReduceMessage<Self::G1, Self::G2, Self::GT>
    where
        Self::G1: Group,
        Self::G2: Group,
        M1: MultiScalarMul<Self::G1>,
        M2: MultiScalarMul<Self::G2>;
    /// Folds the $v_i$ and $s_i$ vectors using the [`SecondReduceChallenge`].
    /// That is,
    /// $$\begin{aligned}
    /// \vec{v_1}^\prime &\leftarrow \alpha \vec{v_{1L}} + \vec{v_{1R}} & \vec{v_2}^\prime &\leftarrow \alpha^{-1} \vec{v_{2L}} + \vec{v_{2R}} \\\\
    /// \vec{s_1}^\prime &\leftarrow \alpha \vec{s_{1L}} + \vec{s_{1R}} & \vec{s_2}^\prime &\leftarrow \alpha^{-1} \vec{s_{2L}} + \vec{s_{2R}}
    /// \end{aligned}$$
    ///
    /// # Panics
    /// Panics if the state is not in an appropriate round. That is, if the last Reduce round has not been completed. This method
    /// assumes that the $v_i$ and $s_i$ vectors are of length at least 2.
    #[must_use]
    fn reduce_fold<M1, M2>(
        self,
        setup: &Self::Setup,
        second_challenge: SecondReduceChallenge<Self::Scalar>,
    ) -> Self
    where
        M1: MultiScalarMul<Self::G1>,
        M2: MultiScalarMul<Self::G2>;
    /// Computes the [`ScalarProductMessage`] using [`FoldScalarsChallenge`]. That is,
    /// $$\begin{aligned}
    /// E_1 &= v_1 + \gamma s_1 H_2 & E_2 &= v_2 + \gamma^{-1} s_2 H_1
    /// \end{aligned}$$
    ///
    /// # Panics
    /// Panics if the state is not in the appropriate round. That is, if the last Reduce round has been completed. This method
    /// assumes that the $v_i$ and $s_i$ vectors are of length 1.
    #[must_use]
    fn compute_scalar_product_message<M1, M2>(
        self,
        setup: &Self::Setup,
        fold_scalars_challenge: FoldScalarsChallenge<Self::Scalar>,
    ) -> ScalarProductMessage<Self::G1, Self::G2>
    where
        Self::G1: Group,
        Self::G2: Group,
        M1: MultiScalarMul<Self::G1>,
        M2: MultiScalarMul<Self::G2>;
}

// Verifier
///
/// Trait for the verifier state and computation during the Dory protocol.
///
/// A type implementing this trait maintains verification state and the operations
/// needed to verify the messages from the prover.
pub trait VerifierState {
    /// The $\mathbb{G}_1$ group
    type G1: Group;
    /// The $\mathbb{G}_2$ group
    type G2: Group;
    /// The target group, $\mathbb{G}_T$
    type GT: Group;
    /// The scalar, $\mathbb{F}$, field of the groups
    type Scalar: Field;
    /// The setup type. This should contain the public parameters needed for verification.
    type Setup;

    /// This is the verifier side of the extended Dory-Reduce algorithm in section 3.2 & 4.2 of the paper.
    fn dory_reduce_verify_round(
        &mut self,
        setup: &Self::Setup,
        first_msg: &FirstReduceMessage<Self::G1, Self::G2, Self::GT>,
        second_msg: &SecondReduceMessage<Self::G1, Self::G2, Self::GT>,
        alpha_pair: (Self::Scalar, Self::Scalar),
        beta_pair: (Self::Scalar, Self::Scalar),
    ) -> bool;

    /// From the Dory-Reduce algorithm in section 3.2 of the paper.
    /// Updates C in the verifier state.
    /// C' <- C + χᵢ + β * D₂ + β⁻¹ * D₁ + α * C_plus + α⁻¹ * C_minus
    fn dory_reduce_verify_update_c(
        &mut self,
        setup: &Self::Setup,
        c_pair: (Self::GT, Self::GT),
        alpha_pair: (Self::Scalar, Self::Scalar),
        beta_pair: (Self::Scalar, Self::Scalar),
    );

    /// From the Dory-Reduce algorithm in section 3.2 of the paper.
    /// Updates D₁ and D₂ in the verifier state.
    /// D₁' <- α * D₁L + D₁R + α * β * Δ₁L + β * Δ₁R
    /// D₂' <- α⁻¹ * D₂L + D₂R + α⁻¹ * β⁻¹ * Δ₂L + β⁻¹ * Δ₂R
    fn dory_reduce_verify_update_ds(
        &mut self,
        setup: &Self::Setup,
        d_values: (Self::GT, Self::GT, Self::GT, Self::GT),
        alpha_pair: (Self::Scalar, Self::Scalar),
        beta_pair: (Self::Scalar, Self::Scalar),
    );

    /// From the extended Dory-Reduce algorithm in section 4.2 of the paper.
    /// Updates E₁ and E₂ in the extended verifier state.
    /// E₁' <- E₁ + β * E₁β + α * E₁+ + α⁻¹ * E₁-
    /// E₂' <- E₂ + β⁻¹ * E₂β + α * E₂+ + α⁻¹ * E₂-
    fn dory_reduce_verify_update_es(
        &mut self,
        e_beta_pair: (Self::G1, Self::G2),
        e_values: (Self::G1, Self::G1, Self::G2, Self::G2),
        alpha_pair: (Self::Scalar, Self::Scalar),
        beta_pair: (Self::Scalar, Self::Scalar),
    );

    /// Apply `Fold-Scalars` algorithm (verifier side) from extended Dory IP
    fn apply_fold_scalars(
        &mut self,
        setup: &Self::Setup,
        gamma_pair: FoldScalarsChallenge<Self::Scalar>,
    );

    /// Final verification step for Extended Dory-Reduce
    /// Verifies: e(E₁, H₂) * e(H₁, E₂) = C * e(H₁, H₂)^γ
    fn verify_final_pairing(
        &self,
        setup: &Self::Setup,
        message: &ScalarProductMessage<Self::G1, Self::G2>,
        d_pair: ScalarProductChallenge<Self::Scalar>,
    ) -> bool;
}

/// --------- Concrete ProverState and VerifierState ---------------------

pub struct DoryProverState<E: Pairing>
where
    E::G1: Group,
    E::G2: Group<Scalar = <E::G1 as Group>::Scalar>,
{
    // these follow notation from the paper
    /// v1 - P witness
    pub v1: Vec<E::G1>,
    /// v2 - P witness
    pub v2: Vec<E::G2>,
    /// s1 - scalars for extended Dory IP (see Section 4)
    pub s1: Vec<<E::G1 as Group>::Scalar>,
    /// s2 - scalars for extended Dory IP (see Section 4)
    pub s2: Vec<<E::G1 as Group>::Scalar>,
    /// number of rounds
    pub nu: usize,
}

impl<E: Pairing> DoryProverState<E>
where
    E::G1: Group,
    E::G2: Group<Scalar = <E::G1 as Group>::Scalar>,
{
    /// Constructor
    pub fn new(
        v1: Vec<E::G1>,
        v2: Vec<E::G2>,
        s1: Vec<<E::G1 as Group>::Scalar>,
        s2: Vec<<E::G1 as Group>::Scalar>,
        nu: usize,
    ) -> Self {
        Self { v1, v2, s1, s2, nu }
    }
}

/// Verifier state.
/// We track each commitment to be mutated during the interactive protocol
pub struct DoryVerifierState<E: Pairing>
where
    E::G1: Group,
    E::G2: Group<Scalar = <E::G1 as Group>::Scalar>,
{
    /// The inner pairing product <v1,v2>.
    pub c: E::GT,
    /// The commitment to v1: <v1,Γ_2>.
    pub d_1: E::GT,
    /// The commitment to v2: <Γ_1,v2>.
    pub d_2: E::GT,

    // extended use case:
    /// The commitment to s1: <v1,s2>.
    pub e_1: E::G1,
    /// The commitment to s2: <s1,v2>.
    pub e_2: E::G2,

    /// Tensors used for VMV (only for PCS protocol)
    pub s1_tensor: Option<Vec<<E::G1 as Group>::Scalar>>,
    /// Tensors used for VMV (only for PCS protocol)
    pub s2_tensor: Option<Vec<<E::G1 as Group>::Scalar>>,

    /// Current round number. Length of v1 and v2 should be 2^nu.
    pub nu: usize,
}
impl<E: Pairing> DoryVerifierState<E>
where
    E::G1: Group,
    E::G2: Group<Scalar = <E::G1 as Group>::Scalar>,
{
    /// Constructor
    pub fn new(c: E::GT, d_1: E::GT, d_2: E::GT, e_1: E::G1, e_2: E::G2, nu: usize) -> Self {
        Self {
            c,
            d_1,
            d_2,
            e_1,
            e_2,
            s1_tensor: None, // not used in non-pcs context
            s2_tensor: None, // not used in non-pcs context
            nu,
        }
    }

    /// Constructor
    pub fn new_with_s(
        c: E::GT,
        d_1: E::GT,
        d_2: E::GT,
        e_1: E::G1,
        e_2: E::G2,
        s1: Vec<<E::G1 as Group>::Scalar>,
        s2: Vec<<E::G1 as Group>::Scalar>,
        nu: usize,
    ) -> Self {
        Self {
            c,
            d_1,
            d_2,
            e_1,
            e_2,
            s1_tensor: Some(s1),
            s2_tensor: Some(s2),
            nu,
        }
    }
}
