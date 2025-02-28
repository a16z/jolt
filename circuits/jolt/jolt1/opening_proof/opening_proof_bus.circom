
pragma circom 2.2.1;

bus ReducedOpeningProof(rounds, sumcheck_claims_len) {
    SumcheckInstanceProof(2, rounds) sumcheck_proof;
    signal sumcheck_claims[sumcheck_claims_len];
    HyperKZGProof(rounds) joint_opening_proof;
}


bus VerifierOpening(num_opening_point) {
    // The random scalar field element used to combine the commits and evals.
    signal rho;
    /// The point at which the polynomial is being evaluated.
    signal opening_point[num_opening_point];
    /// The claimed opening.
    signal claim;
}