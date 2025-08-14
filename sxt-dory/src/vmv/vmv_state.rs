//! Utilities related to the VMV commitment strategy for multilinear polynomials
//! Defines VMV states for both provers, verifiers
use crate::{
    arithmetic::{Field, Group, Pairing},
    poly::{compute_left_right_vec, Polynomial},
    setup::ProverSetup,
    state::DoryProverState,
    MultiScalarMul,
};

/// Prover structure for computing commitment by VMV
pub struct VMVProverState<E: Pairing>
where
    E::G1: Group,
    E::G2: Group<Scalar = <E::G1 as Group>::Scalar>,
{
    /// Evaluations of the columns of the matrix. That is, v = L^T * M.
    /// v[j] = <L, M[_, j]> = sum_{i=0}^{2^nu} L[i] M[i,j].
    pub(super) v_vec: Vec<<E::G1 as Group>::Scalar>,

    /// Commitments to the rows of the matrix.
    /// `T_vec_prime[i] = <M[i, _], Gamma_1[nu]> = sum_{j=0}^{2^nu} M[i,j] Gamma_1[nu][j]`.
    pub(super) t_vec_prime: Vec<<E as Pairing>::G1>,

    /// The left vector, L of LMR.
    pub(super) l_vec: Vec<<E::G1 as Group>::Scalar>,
    /// The right vector, R of LMR.
    pub(super) r_vec: Vec<<E::G1 as Group>::Scalar>,
    /// number of rows is 2^nu
    pub(super) nu: usize,
}

/// Verifier analogue of VMVProverState
pub struct VMVVerifierState<E: Pairing>
where
    E::G1: Group,
    E::G2: Group<Scalar = <E::G1 as Group>::Scalar>,
{
    /// The evaluation of the matrix. That is, y = LMR.
    pub(super) y: <E::G1 as Group>::Scalar,
    /// The commitment to the entire matrix. That is, `T = <T_vec_prime, Gamma_2[nu]>`.
    pub(super) t: <E as Pairing>::GT,
    /// The left tensor, l.
    pub(super) l_tensor: Vec<<E::G1 as Group>::Scalar>,
    /// The right tensor, r.
    pub(super) r_tensor: Vec<<E::G1 as Group>::Scalar>,
    /// number of rows is 2^nu
    pub(super) nu: usize,
}

/// Compute the size of the matrix M that is derived from the coefficients
/// 2^nu is the side length of M
pub fn compute_nu(num_vars: usize, sigma: usize) -> usize {
    if num_vars <= sigma * 2 {
        // No padding needed
        sigma
    } else {
        // Padding needed
        num_vars - sigma
    }
}

/// Compute the (Pedersen) commitments to the rows of the matrix M that is derived from coeffs `a`.
/// This produces T` in the paper.
pub fn commit_to_rows<E, M1, P>(
    polynomial: &P,
    sigma: usize,
    nu: usize,
    prover_setup: &ProverSetup<E>,
) -> Vec<E::G1>
where
    E: Pairing,
    M1: MultiScalarMul<E::G1>,
    P: Polynomial<<E::G1 as Group>::Scalar, E::G1> + ?Sized,
    E::G1: Group,
    <E::G1 as Group>::Scalar: Field + Clone,
{
    let bases = &prover_setup.g1_vec()[..1 << nu];
    let row_len = 1 << sigma;

    let mut res = polynomial.commit_rows::<M1>(bases, row_len);

    // Pad with identity elements if needed
    while res.len() < (1 << nu) {
        res.push(E::G1::identity());
    }

    res
}

/// Build the prover state for the VMV protocol
#[tracing::instrument(skip_all)]
pub fn build_vmv_prover_state<E, P>(
    polynomial: &P,                       // Multilinear polynomial coefficients
    b_point: &[<E::G1 as Group>::Scalar], // Evaluation point ( $v \in \mathbb{R}^d) for d variables
    row_commitments: Vec<E::G1>,
    sigma: usize,
    nu: usize,
) -> VMVProverState<E>
where
    E: Pairing,
    P: Polynomial<<E::G1 as Group>::Scalar, E::G1> + ?Sized + Sync,
    E::G1: Group,
    E::G2: Group<Scalar = <E::G1 as Group>::Scalar>,
{
    let (l_vec, r_vec) = compute_left_right_vec(b_point, sigma, nu);
    let v_vec = polynomial.vector_matrix_product(&l_vec, sigma, nu);

    VMVProverState {
        v_vec,
        t_vec_prime: row_commitments,
        l_vec,
        r_vec,
        nu,
    }
}

/// Convert a VMVProverState to a DoryProverState
pub fn vmv_state_to_dory_prover_state<E: Pairing>(
    vmv_state: VMVProverState<E>,
    _prover_setup: &ProverSetup<E>,
) -> (Vec<<E::G1 as Group>::Scalar>, DoryProverState<E>)
where
    E::G1: Group,
    E::G2: Group<Scalar = <E::G1 as Group>::Scalar>,
{
    // Extract values from VMV state
    // Note: the paper has a typo and we want to actually set s1 = R, s2 = L (as we do below)
    let v_vec = vmv_state.v_vec;
    let s1 = vmv_state.r_vec;
    let s2 = vmv_state.l_vec;
    let v1 = vmv_state.t_vec_prime; // row commitments
    let nu = vmv_state.nu; // nu

    // eval_vmv_re will calculate v2
    let v2 = Vec::new();

    // Create the DoryProverState
    let state = DoryProverState::new(v1, v2, s1, s2, nu);

    (v_vec, state)
}
