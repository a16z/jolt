//! Implementation of the extended Dory-Innerproduct protocol.
//! The protocol is outlined in sections 3.3 & 4.3 of the Dory paper.
use crate::builder::{ProofBuilder, VerificationBuilder};
use crate::state::{ProverState, VerifierState};

/// Prover side of extended Dory-innerproduct
pub fn inner_product_prove<Builder, State, G1, G2, GT, Scalar, Setup, M1, M2>(
    builder: Builder,
    state: State,
    setup: &Setup,
    num_rounds: usize,
) -> Builder
where
    G1: crate::arithmetic::Group,
    G2: crate::arithmetic::Group,
    GT: crate::arithmetic::Group,
    Scalar: crate::arithmetic::Field,
    Builder: ProofBuilder<G1 = G1, G2 = G2, GT = GT, Scalar = Scalar>,
    State: ProverState<G1 = G1, G2 = G2, GT = GT, Scalar = Scalar, Setup = Setup>,
    M1: crate::arithmetic::MultiScalarMul<G1>,
    M2: crate::arithmetic::MultiScalarMul<G2>,
{
    let (builder, state) = (0..num_rounds).fold((builder, state), |(builder, state), _| {
        let first_reduce_msg = state.compute_first_reduce_message::<M1, M2>(setup);
        let (challenge, builder) = builder.append_first_reduce_message(first_reduce_msg);

        let state = state.reduce_combine(setup, challenge);

        let second_reduce_msg = state.compute_second_reduce_message::<M1, M2>(setup);
        let (challenge, builder) = builder.append_second_reduce_message(second_reduce_msg);

        let folded_state = state.reduce_fold(setup, challenge);
        (builder, folded_state)
    });
    let (challenge, builder) = builder.challenge_fold_scalars();
    let scalar_product_msg = state.compute_scalar_product_message::<M1, M2>(setup, challenge);
    builder.append_scalar_product_message(scalar_product_msg)
}

/// Verifier analogue for the extended Dory-innerproduct
pub fn inner_product_verify<B, State, G1, G2, GT, Scalar, Setup>(
    mut builder: B,
    mut state: State,
    setup: &Setup,
    num_rounds: usize, // need rounds here? should be in state?
                       // @TODO(markosg04): add this folding impl thing?
) -> Result<(), usize>
where
    G1: crate::arithmetic::Group,
    G2: crate::arithmetic::Group,
    GT: crate::arithmetic::Group,
    Scalar: crate::arithmetic::Field,
    State: VerifierState<G1 = G1, G2 = G2, GT = GT, Scalar = Scalar, Setup = Setup>,
    B: VerificationBuilder<G1 = G1, G2 = G2, GT = GT, Scalar = Scalar>,
{
    for idx in 0..(num_rounds) {
        let (m1, m2) = builder.take_round(idx);

        let first = builder.process_first_reduce_message(&m1);
        let second = builder.process_second_reduce_message(&m2);

        let beta_pair = (first.beta, first.beta_inverse);
        let alpha_pair = (second.alpha, second.alpha_inverse);

        if !state.dory_reduce_verify_round(setup, &m1, &m2, alpha_pair, beta_pair) {
            return Err(idx);
        }
    }
    let fold_gamma = builder.challenge_fold_scalars();

    if !state.verify_final_pairing(setup, builder.process_scalar_product_message(), fold_gamma) {
        return Err(builder.rounds());
    }

    Ok(())
}
