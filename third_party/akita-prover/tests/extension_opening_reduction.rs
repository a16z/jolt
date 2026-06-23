#![allow(missing_docs)]
#![cfg(not(feature = "zk"))]

use akita_algebra::poly::multilinear_eval;
use akita_field::{
    AkitaError, Ext2, ExtField, FieldCore, FpExt4, Prime128Offset275, Prime24Offset3,
    Prime30Offset35, Prime31Offset19, Prime32Offset99, Prime64Offset59,
};
use akita_prover::protocol::extension_opening_reduction::{
    ExtensionOpeningReductionProver, ExtensionOpeningReductionTerm, SparseExtensionOpeningWitness,
    SPARSE_TENSOR_FACTOR_MAX_LAZY_ROUNDS,
};
use akita_sumcheck::{SumcheckInstanceProver, SumcheckInstanceProverExt, SumcheckProof};
use akita_transcript::labels as tr_labels;
use akita_transcript::{AkitaTranscript, Transcript};
use akita_types::{
    check_extension_opening_reduction_output, derive_tensor_extension_opening_claim_from_partials,
    extension_opening_reduction_claim, extension_opening_reduction_eval_at_point,
    tensor_column_partials_from_base_evals, tensor_equality_factor_eval_at_point,
    tensor_equality_factor_evals, tensor_packed_witness_evals, tensor_reduction_claim_from_rows,
    tensor_row_partials_from_columns, ExtensionOpeningFactorTerm, ExtensionOpeningReductionFactor,
    ExtensionOpeningReductionRoundResult, ExtensionOpeningTensorPartials,
    EXTENSION_OPENING_REDUCTION_DEGREE,
};

type F = Prime128Offset275;

fn new_transcript() -> AkitaTranscript<F> {
    <AkitaTranscript<F> as Transcript<F>>::new(tr_labels::DOMAIN_AKITA_PROTOCOL)
}

fn sample_round(tr: &mut AkitaTranscript<F>) -> F {
    tr.challenge_scalar(tr_labels::CHALLENGE_SUMCHECK_ROUND)
}

fn verify_eor_rounds(
    input_claim: F,
    num_rounds: usize,
    proof: &SumcheckProof<F>,
    transcript: &mut AkitaTranscript<F>,
) -> Result<ExtensionOpeningReductionRoundResult<F>, AkitaError> {
    transcript.append_serde(tr_labels::ABSORB_SUMCHECK_CLAIM, &input_claim);
    let (final_claim, challenges) = proof.verify::<F, _, _>(
        input_claim,
        num_rounds,
        EXTENSION_OPENING_REDUCTION_DEGREE,
        transcript,
        sample_round,
    )?;
    Ok(ExtensionOpeningReductionRoundResult {
        final_claim,
        challenges,
    })
}

fn verify_eor_full(
    witness_evals: &[F],
    factor_evals: &[F],
    proof: &SumcheckProof<F>,
) -> Result<Vec<F>, AkitaError> {
    let input_claim = extension_opening_reduction_claim(witness_evals, factor_evals)?;
    let mut transcript = new_transcript();
    let result = verify_eor_rounds(
        input_claim,
        witness_evals.len().trailing_zeros() as usize,
        proof,
        &mut transcript,
    )?;
    let expected =
        extension_opening_reduction_eval_at_point(witness_evals, factor_evals, &result.challenges)?;
    if result.final_claim != expected {
        return Err(AkitaError::InvalidProof);
    }
    Ok(result.challenges)
}

#[test]
fn sparse_witness_sorted_constructor_combines_without_sorting() {
    let witness = SparseExtensionOpeningWitness::from_sorted_entries(
        8,
        vec![
            (1, F::from_u64(3)),
            (1, F::from_u64(5)),
            (3, F::zero()),
            (4, F::from_u64(7)),
        ],
    )
    .unwrap();
    assert_eq!(
        witness.entries(),
        &[(1, F::from_u64(8)), (4, F::from_u64(7))]
    );

    assert!(SparseExtensionOpeningWitness::from_sorted_entries(
        8,
        vec![(2, F::one()), (1, F::one())],
    )
    .is_err());

    let unique = SparseExtensionOpeningWitness::from_sorted_unique_entries(
        8,
        vec![(1, F::from_u64(3)), (4, F::from_u64(7))],
    )
    .unwrap();
    assert_eq!(
        unique.entries(),
        &[(1, F::from_u64(3)), (4, F::from_u64(7))]
    );
    assert!(SparseExtensionOpeningWitness::from_sorted_unique_entries(
        8,
        vec![(1, F::one()), (1, F::one())],
    )
    .is_err());
    assert!(
        SparseExtensionOpeningWitness::from_sorted_unique_entries(8, vec![(1, F::zero())],)
            .is_err()
    );
}

fn lifted_multilinear_eval<B, E>(evals: &[B], point: &[E]) -> E
where
    B: FieldCore,
    E: ExtField<B>,
{
    let mut layer = evals.iter().copied().map(E::lift_base).collect::<Vec<_>>();
    for &r in point {
        let one_minus_r = E::one() - r;
        let next_len = layer.len() / 2;
        for idx in 0..next_len {
            layer[idx] = layer[2 * idx] * one_minus_r + layer[2 * idx + 1] * r;
        }
        layer.truncate(next_len);
    }
    layer[0]
}

#[test]
fn tensor_partials_recompose_logical_extension_opening() {
    type B = Prime64Offset59;
    type E = Ext2<B>;

    let num_vars = 4;
    let base_evals = (0..(1usize << num_vars))
        .map(|idx| B::from_u64((17 * idx as u64 + 9) % 127))
        .collect::<Vec<_>>();
    let point = (0..num_vars)
        .map(|idx| {
            E::from_base_slice(&[B::from_u64(idx as u64 + 3), B::from_u64(5 * idx as u64 + 2)])
        })
        .collect::<Vec<_>>();

    let column_partials =
        tensor_column_partials_from_base_evals::<B, E>(num_vars, &base_evals, &point).unwrap();
    let row_partials = tensor_row_partials_from_columns::<B, E>(&column_partials).unwrap();
    let partials = ExtensionOpeningTensorPartials {
        column_partials,
        row_partials,
    };
    assert_eq!(
        partials.column_partials.len(),
        <E as ExtField<B>>::EXT_DEGREE
    );
    assert_eq!(partials.row_partials.len(), <E as ExtField<B>>::EXT_DEGREE);

    let logical_claim = derive_tensor_extension_opening_claim_from_partials::<B, E>(
        &point,
        &partials.column_partials,
    )
    .unwrap();
    assert_eq!(logical_claim, lifted_multilinear_eval(&base_evals, &point));
}

#[test]
fn tensor_row_reduction_matches_dense_sumcheck_claim() {
    type B = Prime64Offset59;
    type E = Ext2<B>;

    let num_vars = 4;
    let base_evals = (0..(1usize << num_vars))
        .map(|idx| B::from_u64((23 * idx as u64 + 11) % 131))
        .collect::<Vec<_>>();
    let point = (0..num_vars)
        .map(|idx| {
            E::from_base_slice(&[
                B::from_u64(3 * idx as u64 + 4),
                B::from_u64(7 * idx as u64 + 1),
            ])
        })
        .collect::<Vec<_>>();
    let eta = vec![E::from_base_slice(&[B::from_u64(19), B::from_u64(29)])];

    let packed_witness = tensor_packed_witness_evals::<B, E>(num_vars, &base_evals).unwrap();
    let column_partials =
        tensor_column_partials_from_base_evals::<B, E>(num_vars, &base_evals, &point).unwrap();
    let row_partials = tensor_row_partials_from_columns::<B, E>(&column_partials).unwrap();
    let partials = ExtensionOpeningTensorPartials {
        column_partials,
        row_partials,
    };
    let row_claim = tensor_reduction_claim_from_rows::<B, E>(&partials.row_partials, &eta).unwrap();
    let factor_evals = tensor_equality_factor_evals::<B, E>(&point[1..], &eta).unwrap();

    assert_eq!(packed_witness.len(), factor_evals.len());
    assert_eq!(
        extension_opening_reduction_claim(&packed_witness, &factor_evals).unwrap(),
        row_claim
    );

    let rho = vec![
        E::from_base_slice(&[B::from_u64(31), B::from_u64(37)]),
        E::from_base_slice(&[B::from_u64(41), B::from_u64(43)]),
        E::from_base_slice(&[B::from_u64(47), B::from_u64(53)]),
    ];
    assert_eq!(
        akita_sumcheck::multilinear_eval(&factor_evals, &rho).unwrap(),
        tensor_equality_factor_eval_at_point::<B, E>(&point[1..], &eta, &rho).unwrap()
    );
}

#[test]
fn singleton_factor_claim_matches_multilinear_opening() {
    let witness_evals: Vec<F> = (0..8).map(|i| F::from_u64((11 * i + 4) as u64)).collect();
    let opening_point = vec![F::from_u64(3), F::from_u64(5), F::from_u64(7)];
    let factor = ExtensionOpeningReductionFactor::singleton(opening_point.clone()).unwrap();

    let claim = factor.claim_for_witness(&witness_evals).unwrap();
    let expected = akita_sumcheck::multilinear_eval(&witness_evals, &opening_point).unwrap();
    assert_eq!(claim, expected);

    let rho = vec![F::from_u64(2), F::from_u64(9), F::from_u64(6)];
    let factor_evals = factor.evals().unwrap();
    let folded_factor = akita_sumcheck::multilinear_eval(&factor_evals, &rho).unwrap();
    assert_eq!(folded_factor, factor.evaluate(&rho).unwrap());
}

#[test]
fn row_factor_batches_multiple_opening_points() {
    let witness_evals: Vec<F> = (0..16).map(|i| F::from_u64((5 * i + 8) as u64)).collect();
    let point_a = vec![
        F::from_u64(2),
        F::from_u64(3),
        F::from_u64(4),
        F::from_u64(5),
    ];
    let point_b = vec![
        F::from_u64(7),
        F::from_u64(11),
        F::from_u64(13),
        F::from_u64(17),
    ];
    let coeff_a = F::from_u64(19);
    let coeff_b = F::from_u64(23);
    let factor = ExtensionOpeningReductionFactor::from_terms(vec![
        ExtensionOpeningFactorTerm::new(point_a.clone(), coeff_a),
        ExtensionOpeningFactorTerm::new(point_b.clone(), coeff_b),
    ])
    .unwrap();

    assert_eq!(factor.num_vars(), 4);
    assert_eq!(factor.terms().len(), 2);
    let claim = factor.claim_for_witness(&witness_evals).unwrap();
    let expected = coeff_a * akita_sumcheck::multilinear_eval(&witness_evals, &point_a).unwrap()
        + coeff_b * akita_sumcheck::multilinear_eval(&witness_evals, &point_b).unwrap();
    assert_eq!(claim, expected);

    let rho = vec![
        F::from_u64(29),
        F::from_u64(31),
        F::from_u64(37),
        F::from_u64(41),
    ];
    let factor_evals = factor.evals().unwrap();
    assert_eq!(
        akita_sumcheck::multilinear_eval(&factor_evals, &rho).unwrap(),
        factor.evaluate(&rho).unwrap()
    );
}

#[test]
fn factor_rejects_malformed_shapes() {
    let err = ExtensionOpeningReductionFactor::<F>::from_terms(Vec::new()).unwrap_err();
    assert!(matches!(err, akita_field::AkitaError::InvalidInput(_)));

    let err = ExtensionOpeningReductionFactor::from_terms(vec![
        ExtensionOpeningFactorTerm::new(vec![F::one(), F::zero()], F::one()),
        ExtensionOpeningFactorTerm::new(vec![F::one()], F::one()),
    ])
    .unwrap_err();
    assert!(matches!(err, akita_field::AkitaError::InvalidSize { .. }));
}

#[test]
fn extension_opening_reduction_proves_witness_factor_claim() {
    let witness_evals: Vec<F> = (0..16).map(|i| F::from_u64((3 * i + 5) as u64)).collect();
    let factor_evals: Vec<F> = (0..16).map(|i| F::from_u64((7 * i + 11) as u64)).collect();
    let expected_claim = extension_opening_reduction_claim(&witness_evals, &factor_evals).unwrap();

    let term =
        ExtensionOpeningReductionTerm::new(witness_evals.clone(), factor_evals.clone(), F::one())
            .unwrap();
    let mut prover = ExtensionOpeningReductionProver::new(vec![term], expected_claim).unwrap();
    assert_eq!(prover.degree_bound(), EXTENSION_OPENING_REDUCTION_DEGREE);
    assert_eq!(prover.input_claim(), expected_claim);

    let mut prover_transcript = new_transcript();
    let (proof, challenges, final_claim) = prover
        .prove::<F, _, _>(&mut prover_transcript, sample_round)
        .unwrap();

    let (final_witness, final_factor) = prover.final_witness_and_factor_evals().unwrap();
    assert_eq!(final_claim, final_witness * final_factor);
    assert_eq!(
        final_claim,
        extension_opening_reduction_eval_at_point(&witness_evals, &factor_evals, &challenges)
            .unwrap()
    );

    let verified_challenges = verify_eor_full(&witness_evals, &factor_evals, &proof).unwrap();
    assert_eq!(verified_challenges, challenges);
}

#[test]
fn batched_extension_opening_reduction_uses_one_common_rho() {
    let witness_a: Vec<F> = (0..16).map(|i| F::from_u64((3 * i + 5) as u64)).collect();
    let factor_a: Vec<F> = (0..16).map(|i| F::from_u64((7 * i + 11) as u64)).collect();
    let witness_b: Vec<F> = (0..16).map(|i| F::from_u64((13 * i + 17) as u64)).collect();
    let factor_b: Vec<F> = (0..16).map(|i| F::from_u64((19 * i + 23) as u64)).collect();
    let coeff_a = F::from_u64(29);
    let coeff_b = F::from_u64(31);
    let expected_claim = coeff_a
        * extension_opening_reduction_claim(&witness_a, &factor_a).unwrap()
        + coeff_b * extension_opening_reduction_claim(&witness_b, &factor_b).unwrap();

    let terms = vec![
        ExtensionOpeningReductionTerm::new(witness_a.clone(), factor_a.clone(), coeff_a).unwrap(),
        ExtensionOpeningReductionTerm::new(witness_b.clone(), factor_b.clone(), coeff_b).unwrap(),
    ];
    assert_eq!(
        ExtensionOpeningReductionProver::input_claim_from_terms(&terms).unwrap(),
        expected_claim
    );
    let mut prover = ExtensionOpeningReductionProver::new(terms, expected_claim).unwrap();
    assert_eq!(prover.input_claim(), expected_claim);
    assert_eq!(prover.degree_bound(), EXTENSION_OPENING_REDUCTION_DEGREE);

    let mut transcript = new_transcript();
    let (_proof, challenges, final_claim) = prover
        .prove::<F, _, _>(&mut transcript, sample_round)
        .unwrap();
    let expected_final = prover
        .final_terms()
        .unwrap()
        .into_iter()
        .fold(F::zero(), |acc, (coeff, witness, factor)| {
            acc + coeff * witness * factor
        });
    assert_eq!(final_claim, expected_final);
    assert_eq!(
        final_claim,
        coeff_a
            * extension_opening_reduction_eval_at_point(&witness_a, &factor_a, &challenges)
                .unwrap()
            + coeff_b
                * extension_opening_reduction_eval_at_point(&witness_b, &factor_b, &challenges)
                    .unwrap()
    );
}

#[test]
fn sparse_tensor_factor_matches_dense_factor_rounds() {
    type B = Prime64Offset59;
    type E = Ext2<B>;

    let tail_point = (0..5)
        .map(|idx| {
            E::from_base_slice(&[
                B::from_u64(3 * idx as u64 + 7),
                B::from_u64(5 * idx as u64 + 11),
            ])
        })
        .collect::<Vec<_>>();
    let eta = vec![E::from_base_slice(&[B::from_u64(17), B::from_u64(19)])];
    let coeff = E::from_base_slice(&[B::from_u64(23), B::from_u64(29)]);
    let entries = vec![
        (1, E::from_base_slice(&[B::from_u64(31), B::from_u64(37)])),
        (2, E::from_base_slice(&[B::from_u64(41), B::from_u64(43)])),
        (3, E::from_base_slice(&[B::from_u64(47), B::from_u64(53)])),
        (5, E::from_base_slice(&[B::from_u64(59), B::from_u64(61)])),
        (20, E::from_base_slice(&[B::from_u64(67), B::from_u64(71)])),
        (25, E::from_base_slice(&[B::from_u64(73), B::from_u64(79)])),
    ];
    let sparse_witness =
        SparseExtensionOpeningWitness::new(1usize << tail_point.len(), entries).unwrap();

    let dense_factor = tensor_equality_factor_evals::<B, E>(&tail_point, &eta).unwrap();
    let dense_term =
        ExtensionOpeningReductionTerm::new_sparse(sparse_witness.clone(), dense_factor, coeff)
            .unwrap();
    let lazy_term = ExtensionOpeningReductionTerm::new_sparse_tensor_factor::<B>(
        sparse_witness,
        tail_point.clone(),
        eta,
        coeff,
        2,
    )
    .unwrap();

    let expected_claim =
        ExtensionOpeningReductionProver::input_claim_from_terms(std::slice::from_ref(&dense_term))
            .unwrap();
    assert_eq!(
        ExtensionOpeningReductionProver::input_claim_from_terms(std::slice::from_ref(&lazy_term,))
            .unwrap(),
        expected_claim
    );

    let mut dense_prover =
        ExtensionOpeningReductionProver::new(vec![dense_term], expected_claim).unwrap();
    let mut lazy_prover =
        ExtensionOpeningReductionProver::new(vec![lazy_term], expected_claim).unwrap();
    let mut claim = expected_claim;
    for round in 0..tail_point.len() {
        let dense_round = dense_prover.compute_round_univariate(round, claim);
        let lazy_round = lazy_prover.compute_round_univariate(round, claim);
        assert_eq!(lazy_round, dense_round);

        let challenge = E::from_base_slice(&[
            B::from_u64(83 + 2 * round as u64),
            B::from_u64(89 + 3 * round as u64),
        ]);
        claim = dense_round.evaluate(&challenge);
        dense_prover.ingest_challenge(round, challenge);
        lazy_prover.ingest_challenge(round, challenge);
    }

    assert_eq!(lazy_prover.final_terms(), dense_prover.final_terms());
}

#[test]
fn sparse_tensor_factor_matches_dense_factor_rounds_at_production_lazy_depth() {
    type B = Prime32Offset99;
    type E = FpExt4<B>;

    let tail_point = (0..14)
        .map(|idx| {
            E::from_base_slice(&[
                B::from_u64(3 * idx as u64 + 7),
                B::from_u64(5 * idx as u64 + 11),
                B::from_u64(2 * idx as u64 + 1),
                B::from_u64(7 * idx as u64 + 3),
            ])
        })
        .collect::<Vec<_>>();
    let eta = vec![
        E::from_base_slice(&[
            B::from_u64(17),
            B::from_u64(19),
            B::from_u64(4),
            B::from_u64(6),
        ]),
        E::from_base_slice(&[
            B::from_u64(8),
            B::from_u64(2),
            B::from_u64(13),
            B::from_u64(5),
        ]),
    ];
    let coeff = E::from_base_slice(&[
        B::from_u64(23),
        B::from_u64(29),
        B::from_u64(9),
        B::from_u64(15),
    ]);
    let entries = [1usize, 2, 3, 5, 20, 25, 127, 4096, 8191, 10_000, 16_000]
        .into_iter()
        .enumerate()
        .map(|(entry_idx, table_idx)| {
            (
                table_idx,
                E::from_base_slice(&[
                    B::from_u64(31 + 2 * entry_idx as u64),
                    B::from_u64(37 + 3 * entry_idx as u64),
                    B::from_u64(5 + entry_idx as u64),
                    B::from_u64(11 + 4 * entry_idx as u64),
                ]),
            )
        })
        .collect::<Vec<_>>();
    let sparse_witness =
        SparseExtensionOpeningWitness::new(1usize << tail_point.len(), entries).unwrap();

    let dense_factor = tensor_equality_factor_evals::<B, E>(&tail_point, &eta).unwrap();
    let dense_term =
        ExtensionOpeningReductionTerm::new_sparse(sparse_witness.clone(), dense_factor, coeff)
            .unwrap();
    let lazy_term = ExtensionOpeningReductionTerm::new_sparse_tensor_factor::<B>(
        sparse_witness,
        tail_point.clone(),
        eta,
        coeff,
        SPARSE_TENSOR_FACTOR_MAX_LAZY_ROUNDS,
    )
    .unwrap();

    let expected_claim =
        ExtensionOpeningReductionProver::input_claim_from_terms(std::slice::from_ref(&dense_term))
            .unwrap();
    assert_eq!(
        ExtensionOpeningReductionProver::input_claim_from_terms(std::slice::from_ref(&lazy_term,))
            .unwrap(),
        expected_claim
    );

    let mut dense_prover =
        ExtensionOpeningReductionProver::new(vec![dense_term], expected_claim).unwrap();
    let mut lazy_prover =
        ExtensionOpeningReductionProver::new(vec![lazy_term], expected_claim).unwrap();
    let mut claim = expected_claim;
    for round in 0..tail_point.len() {
        let dense_round = dense_prover.compute_round_univariate(round, claim);
        let lazy_round = lazy_prover.compute_round_univariate(round, claim);
        assert_eq!(lazy_round, dense_round);

        let challenge = E::from_base_slice(&[
            B::from_u64(83 + 2 * round as u64),
            B::from_u64(89 + 3 * round as u64),
            B::from_u64(5 + round as u64),
            B::from_u64(11 + round as u64),
        ]);
        claim = dense_round.evaluate(&challenge);
        dense_prover.ingest_challenge(round, challenge);
        lazy_prover.ingest_challenge(round, challenge);
    }

    assert_eq!(lazy_prover.final_terms(), dense_prover.final_terms());
}

// Exercises the `FpExt4<Fp32>` lazy tensor factor across every Fp32-backed
// base prime: the largest `2^32 - 99` down through the 31-bit Mersenne-style
// `2^31 - 19` and smaller widths. Where the product accumulator is exact the lazy
// factor takes the delayed-reduction branch in `factor_pair`; otherwise it falls
// back to per-term reduction. Either way the lazy rounds must stay byte-identical to
// the dense factor materialized via per-term `Mul` (the byte-identical-proof
// guarantee for the small-field one-hot modes). The macro stamps one `#[test]` per
// prime so a regression pinpoints the offending modulus rather than hiding behind a
// single hard-coded prime.
macro_rules! sparse_tensor_factor_matches_dense_fp32_test {
    ($name:ident, $base:ty) => {
        #[test]
        fn $name() {
            type B = $base;
            type E = FpExt4<B>;

            let tail_point = (0..5)
                .map(|idx| {
                    E::from_base_slice(&[
                        B::from_u64(3 * idx as u64 + 7),
                        B::from_u64(5 * idx as u64 + 11),
                        B::from_u64(2 * idx as u64 + 1),
                        B::from_u64(7 * idx as u64 + 3),
                    ])
                })
                .collect::<Vec<_>>();
            let eta = vec![
                E::from_base_slice(&[
                    B::from_u64(17),
                    B::from_u64(19),
                    B::from_u64(4),
                    B::from_u64(6),
                ]),
                E::from_base_slice(&[
                    B::from_u64(8),
                    B::from_u64(2),
                    B::from_u64(13),
                    B::from_u64(5),
                ]),
            ];
            let coeff = E::from_base_slice(&[
                B::from_u64(23),
                B::from_u64(29),
                B::from_u64(9),
                B::from_u64(15),
            ]);
            let entries = vec![
                (
                    1,
                    E::from_base_slice(&[
                        B::from_u64(31),
                        B::from_u64(37),
                        B::from_u64(2),
                        B::from_u64(8),
                    ]),
                ),
                (
                    2,
                    E::from_base_slice(&[
                        B::from_u64(41),
                        B::from_u64(43),
                        B::from_u64(5),
                        B::from_u64(9),
                    ]),
                ),
                (
                    3,
                    E::from_base_slice(&[
                        B::from_u64(47),
                        B::from_u64(53),
                        B::from_u64(6),
                        B::from_u64(1),
                    ]),
                ),
                (
                    5,
                    E::from_base_slice(&[
                        B::from_u64(59),
                        B::from_u64(61),
                        B::from_u64(7),
                        B::from_u64(3),
                    ]),
                ),
                (
                    20,
                    E::from_base_slice(&[
                        B::from_u64(67),
                        B::from_u64(71),
                        B::from_u64(8),
                        B::from_u64(4),
                    ]),
                ),
                (
                    25,
                    E::from_base_slice(&[
                        B::from_u64(73),
                        B::from_u64(79),
                        B::from_u64(2),
                        B::from_u64(6),
                    ]),
                ),
            ];
            let sparse_witness =
                SparseExtensionOpeningWitness::new(1usize << tail_point.len(), entries).unwrap();

            let dense_factor = tensor_equality_factor_evals::<B, E>(&tail_point, &eta).unwrap();
            let dense_term = ExtensionOpeningReductionTerm::new_sparse(
                sparse_witness.clone(),
                dense_factor,
                coeff,
            )
            .unwrap();
            let lazy_term = ExtensionOpeningReductionTerm::new_sparse_tensor_factor::<B>(
                sparse_witness,
                tail_point.clone(),
                eta,
                coeff,
                2,
            )
            .unwrap();

            let expected_claim = ExtensionOpeningReductionProver::input_claim_from_terms(
                std::slice::from_ref(&dense_term),
            )
            .unwrap();
            assert_eq!(
                ExtensionOpeningReductionProver::input_claim_from_terms(std::slice::from_ref(
                    &lazy_term,
                ))
                .unwrap(),
                expected_claim
            );

            let mut dense_prover =
                ExtensionOpeningReductionProver::new(vec![dense_term], expected_claim).unwrap();
            let mut lazy_prover =
                ExtensionOpeningReductionProver::new(vec![lazy_term], expected_claim).unwrap();
            let mut claim = expected_claim;
            for round in 0..tail_point.len() {
                let dense_round = dense_prover.compute_round_univariate(round, claim);
                let lazy_round = lazy_prover.compute_round_univariate(round, claim);
                assert_eq!(lazy_round, dense_round);

                let challenge = E::from_base_slice(&[
                    B::from_u64(83 + 2 * round as u64),
                    B::from_u64(89 + 3 * round as u64),
                    B::from_u64(5 + round as u64),
                    B::from_u64(11 + round as u64),
                ]);
                claim = dense_round.evaluate(&challenge);
                dense_prover.ingest_challenge(round, challenge);
                lazy_prover.ingest_challenge(round, challenge);
            }

            assert_eq!(lazy_prover.final_terms(), dense_prover.final_terms());
        }
    };
}

sparse_tensor_factor_matches_dense_fp32_test!(
    sparse_tensor_factor_matches_dense_factor_rounds_fp32_prime32_offset99,
    Prime32Offset99
);
sparse_tensor_factor_matches_dense_fp32_test!(
    sparse_tensor_factor_matches_dense_factor_rounds_fp32_prime31_offset19,
    Prime31Offset19
);
sparse_tensor_factor_matches_dense_fp32_test!(
    sparse_tensor_factor_matches_dense_factor_rounds_fp32_prime30_offset35,
    Prime30Offset35
);
sparse_tensor_factor_matches_dense_fp32_test!(
    sparse_tensor_factor_matches_dense_factor_rounds_fp32_prime24_offset3,
    Prime24Offset3
);

#[test]
fn extension_opening_reduction_proves_transparent_factor_claim() {
    let witness_evals: Vec<F> = (0..16).map(|i| F::from_u64((3 * i + 5) as u64)).collect();
    let factor = ExtensionOpeningReductionFactor::from_terms(vec![
        ExtensionOpeningFactorTerm::new(
            vec![
                F::from_u64(2),
                F::from_u64(3),
                F::from_u64(4),
                F::from_u64(5),
            ],
            F::from_u64(7),
        ),
        ExtensionOpeningFactorTerm::new(
            vec![
                F::from_u64(11),
                F::from_u64(13),
                F::from_u64(17),
                F::from_u64(19),
            ],
            F::from_u64(23),
        ),
    ])
    .unwrap();
    let factor_evals = factor.evals().unwrap();
    let expected_claim = factor.claim_for_witness(&witness_evals).unwrap();

    let term =
        ExtensionOpeningReductionTerm::new(witness_evals.clone(), factor_evals.clone(), F::one())
            .unwrap();
    let mut prover = ExtensionOpeningReductionProver::new(vec![term], expected_claim).unwrap();
    assert_eq!(prover.input_claim(), expected_claim);

    let mut prover_transcript = new_transcript();
    let (proof, challenges, final_claim) = prover
        .prove::<F, _, _>(&mut prover_transcript, sample_round)
        .unwrap();
    let (final_witness, final_factor) = prover.final_witness_and_factor_evals().unwrap();
    assert_eq!(final_factor, factor.evaluate(&challenges).unwrap());
    check_extension_opening_reduction_output(final_claim, final_witness, final_factor).unwrap();

    let verified_challenges = verify_eor_full(&witness_evals, &factor_evals, &proof).unwrap();
    assert_eq!(verified_challenges, challenges);
}

#[test]
fn detached_verifier_checks_transparent_factor_against_opened_witness() {
    let witness_evals: Vec<F> = (0..8).map(|i| F::from_u64((17 * i + 3) as u64)).collect();
    let factor = ExtensionOpeningReductionFactor::singleton(vec![
        F::from_u64(2),
        F::from_u64(5),
        F::from_u64(11),
    ])
    .unwrap();
    let factor_evals = factor.evals().unwrap();
    let input_claim = factor.claim_for_witness(&witness_evals).unwrap();

    let term =
        ExtensionOpeningReductionTerm::new(witness_evals.clone(), factor_evals, F::one()).unwrap();
    let mut prover = ExtensionOpeningReductionProver::new(vec![term], input_claim).unwrap();
    let mut prover_transcript = new_transcript();
    let (proof, _challenges, _final_claim) = prover
        .prove::<F, _, _>(&mut prover_transcript, sample_round)
        .unwrap();

    let mut verifier_transcript = new_transcript();
    let verifier_result = verify_eor_rounds(
        input_claim,
        factor.num_vars(),
        &proof,
        &mut verifier_transcript,
    )
    .unwrap();

    let opened_witness = multilinear_eval(&witness_evals, &verifier_result.challenges).unwrap();
    let factor_eval = factor.evaluate(&verifier_result.challenges).unwrap();
    check_extension_opening_reduction_output(
        verifier_result.final_claim,
        opened_witness,
        factor_eval,
    )
    .unwrap();

    assert!(matches!(
        check_extension_opening_reduction_output(
            verifier_result.final_claim + F::one(),
            opened_witness,
            factor_eval,
        ),
        Err(akita_field::AkitaError::InvalidProof)
    ));
}

#[test]
fn extension_opening_reduction_rejects_wrong_final_oracle() {
    let witness_evals: Vec<F> = (0..8).map(|i| F::from_u64((i + 1) as u64)).collect();
    let factor_evals: Vec<F> = (0..8).map(|i| F::from_u64((2 * i + 9) as u64)).collect();

    let input_claim = extension_opening_reduction_claim(&witness_evals, &factor_evals).unwrap();
    let term =
        ExtensionOpeningReductionTerm::new(witness_evals.clone(), factor_evals, F::one()).unwrap();
    let mut prover = ExtensionOpeningReductionProver::new(vec![term], input_claim).unwrap();
    let mut prover_transcript = new_transcript();
    let (proof, _, _) = prover
        .prove::<F, _, _>(&mut prover_transcript, sample_round)
        .unwrap();

    let bad_factor_evals: Vec<F> = (0..8).map(|i| F::from_u64((2 * i + 10) as u64)).collect();
    let err = verify_eor_full(&witness_evals, &bad_factor_evals, &proof).unwrap_err();
    assert!(matches!(err, akita_field::AkitaError::InvalidProof));
}

#[test]
fn extension_opening_reduction_detached_round_verifier_returns_final_claim() {
    let witness_evals: Vec<F> = (0..4).map(|i| F::from_u64((5 * i + 1) as u64)).collect();
    let factor_evals: Vec<F> = (0..4).map(|i| F::from_u64((13 * i + 2) as u64)).collect();
    let input_claim = extension_opening_reduction_claim(&witness_evals, &factor_evals).unwrap();
    let term =
        ExtensionOpeningReductionTerm::new(witness_evals.clone(), factor_evals.clone(), F::one())
            .unwrap();
    let mut prover = ExtensionOpeningReductionProver::new(vec![term], input_claim).unwrap();

    let mut prover_transcript = new_transcript();
    let (proof, challenges, final_claim) = prover
        .prove::<F, _, _>(&mut prover_transcript, sample_round)
        .unwrap();

    let mut verifier_transcript = new_transcript();
    verifier_transcript.append_serde(
        tr_labels::ABSORB_SUMCHECK_CLAIM,
        &proof_claim(&witness_evals, &factor_evals),
    );
    let (detached_final_claim, detached_challenges) = proof
        .verify::<F, _, _>(
            proof_claim(&witness_evals, &factor_evals),
            challenges.len(),
            EXTENSION_OPENING_REDUCTION_DEGREE,
            &mut verifier_transcript,
            sample_round,
        )
        .unwrap();

    assert_eq!(detached_challenges, challenges);
    assert_eq!(detached_final_claim, final_claim);
}

#[test]
fn extension_opening_reduction_rejects_malformed_table_lengths() {
    let witness_evals = vec![F::one(), F::from_u64(2), F::from_u64(3)];
    let factor_evals = vec![F::one(), F::from_u64(2), F::from_u64(3)];
    assert!(ExtensionOpeningReductionTerm::new(witness_evals, factor_evals, F::one()).is_err());

    let witness_evals = vec![F::one(), F::from_u64(2)];
    let factor_evals = vec![F::one()];
    assert!(extension_opening_reduction_claim(&witness_evals, &factor_evals).is_err());
}

fn proof_claim(witness_evals: &[F], factor_evals: &[F]) -> F {
    extension_opening_reduction_claim(witness_evals, factor_evals).unwrap()
}

// ---------------------------------------------------------------------------
// Regression: EOR round messages must honor `DELAYED_PRODUCT_SUM_IS_EXACT`.
//
// `accumulate_dense_round`, `fused_fold_and_accumulate`, and the sparse
// `accumulate_entries_with_factor` sum `mul_to_product_accum` products and
// reduce once. That is only sound when the field's accumulator is exact w.r.t.
// per-term `Mul`. For a field that leaves `DELAYED_PRODUCT_SUM_IS_EXACT` at its
// conservative `false` default, the prover must reduce every product first, or
// the round coefficients silently drift and the prover's claim diverges.
//
// The existing byte-identical tests only cover fields whose flag is `true`
// (exact) or whose accumulator is trivially exact, so they cannot catch a
// regression on the `false` path. These tests drive the public prover API with
// a mock field whose product accumulator deliberately wraps mod 2^64, and
// assert the emitted round messages stay byte-identical to per-term `Mul`.
mod delayed_product_sum_contract {
    use super::*;
    use akita_field::unreduced::{HasOptimizedFold, HasUnreducedOps};
    use akita_field::{AdditiveGroup, Invertible, One, RingCore, Zero};
    use std::fmt;
    use std::iter::{Product, Sum};
    use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

    type Inner = Prime64Offset59;

    /// `u64` product accumulator that adds modulo `2^64`. Each stored value is a
    /// canonical residue `< p < 2^64`, but summing several near-`p` residues
    /// wraps, so `reduce(Σ mul_to_product_accum)` diverges from `Σ a*b` — exactly
    /// the hazard `DELAYED_PRODUCT_SUM_IS_EXACT = false` exists to flag.
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    struct WrappingU64Accum(u64);

    impl Zero for WrappingU64Accum {
        fn zero() -> Self {
            Self(0)
        }
        fn is_zero(&self) -> bool {
            self.0 == 0
        }
    }
    impl Add for WrappingU64Accum {
        type Output = Self;
        fn add(self, rhs: Self) -> Self {
            Self(self.0.wrapping_add(rhs.0))
        }
    }
    impl Add<&WrappingU64Accum> for WrappingU64Accum {
        type Output = Self;
        fn add(self, rhs: &Self) -> Self {
            Self(self.0.wrapping_add(rhs.0))
        }
    }
    impl AddAssign for WrappingU64Accum {
        fn add_assign(&mut self, rhs: Self) {
            self.0 = self.0.wrapping_add(rhs.0);
        }
    }
    impl Sub for WrappingU64Accum {
        type Output = Self;
        fn sub(self, rhs: Self) -> Self {
            Self(self.0.wrapping_sub(rhs.0))
        }
    }
    impl Sub<&WrappingU64Accum> for WrappingU64Accum {
        type Output = Self;
        fn sub(self, rhs: &Self) -> Self {
            Self(self.0.wrapping_sub(rhs.0))
        }
    }
    impl SubAssign for WrappingU64Accum {
        fn sub_assign(&mut self, rhs: Self) {
            self.0 = self.0.wrapping_sub(rhs.0);
        }
    }
    impl Neg for WrappingU64Accum {
        type Output = Self;
        fn neg(self) -> Self {
            Self(self.0.wrapping_neg())
        }
    }
    impl AdditiveGroup for WrappingU64Accum {}

    /// Field wrapper over `Prime64Offset59` whose only non-standard behavior is
    /// the lossy product accumulator above plus `DELAYED_PRODUCT_SUM_IS_EXACT =
    /// false`. All ordinary arithmetic delegates to the exact inner field, so a
    /// per-term `Mul` computation is trivially the ground truth.
    #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
    struct LossyField(Inner);

    impl LossyField {
        fn from_u64(v: u64) -> Self {
            Self(Inner::from_u64(v))
        }
    }
    impl fmt::Display for LossyField {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{}", self.0)
        }
    }
    impl Zero for LossyField {
        fn zero() -> Self {
            Self(Inner::zero())
        }
        fn is_zero(&self) -> bool {
            self.0.is_zero()
        }
    }
    impl One for LossyField {
        fn one() -> Self {
            Self(Inner::one())
        }
    }
    impl Add for LossyField {
        type Output = Self;
        fn add(self, rhs: Self) -> Self {
            Self(self.0 + rhs.0)
        }
    }
    impl Add<&LossyField> for LossyField {
        type Output = Self;
        fn add(self, rhs: &Self) -> Self {
            Self(self.0 + rhs.0)
        }
    }
    impl AddAssign for LossyField {
        fn add_assign(&mut self, rhs: Self) {
            self.0 += rhs.0;
        }
    }
    impl Sub for LossyField {
        type Output = Self;
        fn sub(self, rhs: Self) -> Self {
            Self(self.0 - rhs.0)
        }
    }
    impl Sub<&LossyField> for LossyField {
        type Output = Self;
        fn sub(self, rhs: &Self) -> Self {
            Self(self.0 - rhs.0)
        }
    }
    impl SubAssign for LossyField {
        fn sub_assign(&mut self, rhs: Self) {
            self.0 -= rhs.0;
        }
    }
    impl Neg for LossyField {
        type Output = Self;
        fn neg(self) -> Self {
            Self(-self.0)
        }
    }
    impl Mul for LossyField {
        type Output = Self;
        fn mul(self, rhs: Self) -> Self {
            Self(self.0 * rhs.0)
        }
    }
    impl Mul<&LossyField> for LossyField {
        type Output = Self;
        fn mul(self, rhs: &Self) -> Self {
            Self(self.0 * rhs.0)
        }
    }
    impl MulAssign for LossyField {
        fn mul_assign(&mut self, rhs: Self) {
            self.0 *= rhs.0;
        }
    }
    impl Sum for LossyField {
        fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
            iter.fold(Self::zero(), |acc, x| acc + x)
        }
    }
    impl<'a> Sum<&'a LossyField> for LossyField {
        fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
            iter.fold(Self::zero(), |acc, x| acc + *x)
        }
    }
    impl Product for LossyField {
        fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
            iter.fold(Self::one(), |acc, x| acc * x)
        }
    }
    impl<'a> Product<&'a LossyField> for LossyField {
        fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
            iter.fold(Self::one(), |acc, x| acc * *x)
        }
    }
    impl AdditiveGroup for LossyField {}
    impl RingCore for LossyField {}
    impl Invertible for LossyField {
        fn inverse(&self) -> Option<Self> {
            self.0.inverse().map(Self)
        }
    }
    impl FieldCore for LossyField {}

    impl HasOptimizedFold for LossyField {
        type FoldCtx = Self;
        fn precompute_fold(r: Self) -> Self {
            r
        }
        fn fold_one(r: &Self, even: Self, odd: Self) -> Self {
            even + *r * (odd - even)
        }
    }

    impl HasUnreducedOps for LossyField {
        type MulU64Accum = WrappingU64Accum;
        type ProductAccum = WrappingU64Accum;

        // Deliberately inexact: the accumulator wraps mod 2^64, so a delayed
        // batch sum diverges from per-term `Mul` once the sum crosses 2^64.
        const DELAYED_PRODUCT_SUM_IS_EXACT: bool = false;

        fn mul_u64_unreduced(self, small: u64) -> WrappingU64Accum {
            WrappingU64Accum((self.0 * Inner::from_u64(small)).to_limbs())
        }
        fn mul_to_product_accum(self, other: Self) -> WrappingU64Accum {
            WrappingU64Accum((self.0 * other.0).to_limbs())
        }
        fn reduce_mul_u64_accum(accum: WrappingU64Accum) -> Self {
            Self(Inner::from_u64(accum.0))
        }
        fn reduce_product_accum(accum: WrappingU64Accum) -> Self {
            Self(Inner::from_u64(accum.0))
        }
    }

    /// `p - 1`, the largest canonical residue (~2^64); prime-agnostic via `-1`.
    fn max_residue() -> LossyField {
        -LossyField::one()
    }

    /// Ground-truth degree-2 round message `c + l·X + q·X²` computed entirely
    /// with per-term `Mul`, with `l = claim - 2c - q`.
    fn reference_round_eval(
        witness: &[LossyField],
        factor: &[LossyField],
        claim: LossyField,
        x: LossyField,
    ) -> LossyField {
        let half = witness.len() / 2;
        let mut constant = LossyField::zero();
        let mut quadratic = LossyField::zero();
        for i in 0..half {
            let w0 = witness[2 * i];
            let w1 = witness[2 * i + 1];
            let a0 = factor[2 * i];
            let a1 = factor[2 * i + 1];
            constant += w0 * a0;
            quadratic += (w1 - w0) * (a1 - a0);
        }
        let linear = claim - constant - constant - quadratic;
        constant + linear * x + quadratic * x * x
    }

    /// Exact multilinear fold `even + r·(odd − even)`, matching the prover.
    fn reference_fold(table: &[LossyField], r: LossyField) -> Vec<LossyField> {
        (0..table.len() / 2)
            .map(|i| table[2 * i] + r * (table[2 * i + 1] - table[2 * i]))
            .collect()
    }

    /// Confirm the chosen tables actually trip the lossy accumulator, so the
    /// byte-identicality assertions below would fail if the prover summed wide
    /// products instead of reducing per term.
    fn assert_inputs_are_hazardous(witness: &[LossyField], factor: &[LossyField]) {
        let half = witness.len() / 2;
        let per_term = (0..half).fold(LossyField::zero(), |acc, i| {
            acc + witness[2 * i] * factor[2 * i]
        });
        let delayed = {
            let mut accum = WrappingU64Accum::zero();
            for i in 0..half {
                accum += witness[2 * i].mul_to_product_accum(factor[2 * i]);
            }
            LossyField::reduce_product_accum(accum)
        };
        assert_ne!(
            per_term, delayed,
            "test inputs must trigger the lossy delayed accumulator"
        );
    }

    // Dense path: round 0 exercises `accumulate_dense_round`; later rounds use
    // the cache filled by `fused_fold_and_accumulate`. Both must reduce per term
    // for this field, so every round message matches the per-term reference.
    #[test]
    fn dense_round_messages_honor_delayed_product_flag() {
        let zero = LossyField::zero();
        let one = LossyField::one();
        let two = LossyField::from_u64(2);
        let max = max_residue();
        // Even slots each multiply to ~2^64; four of them overflow a u64.
        let mut witness = vec![one, two, one, two, one, two, one, two];
        let mut factor = vec![max, zero, max, zero, max, zero, max, zero];
        assert_inputs_are_hazardous(&witness, &factor);

        let input_claim = extension_opening_reduction_claim(&witness, &factor).unwrap();
        let term =
            ExtensionOpeningReductionTerm::new(witness.clone(), factor.clone(), LossyField::one())
                .unwrap();
        let mut prover = ExtensionOpeningReductionProver::new(vec![term], input_claim).unwrap();
        let mut claim = prover.input_claim();

        let eval_points = [zero, one, two, LossyField::from_u64(3)];
        for round in 0..3 {
            let prover_poly = prover.compute_round_univariate(round, claim);
            for &x in &eval_points {
                assert_eq!(
                    prover_poly.evaluate(&x),
                    reference_round_eval(&witness, &factor, claim, x),
                    "dense round {round} diverged from per-term Mul at x={x:?}"
                );
            }
            let challenge = LossyField::from_u64(7 + round as u64);
            claim = prover_poly.evaluate(&challenge);
            prover.ingest_challenge(round, challenge);
            witness = reference_fold(&witness, challenge);
            factor = reference_fold(&factor, challenge);
        }
    }

    // Sparse path: round 0 exercises `accumulate_entries_with_factor`, which
    // must take the per-term branch for this field.
    #[test]
    fn sparse_round_messages_honor_delayed_product_flag() {
        let zero = LossyField::zero();
        let one = LossyField::one();
        let max = max_residue();
        // Dense-equivalent tables (even slots nonzero) for the reference.
        let witness = vec![one, zero, one, zero, one, zero, one, zero];
        let factor = vec![max, zero, max, zero, max, zero, max, zero];
        assert_inputs_are_hazardous(&witness, &factor);

        let entries = vec![(0, one), (2, one), (4, one), (6, one)];
        let sparse = SparseExtensionOpeningWitness::new(8, entries).unwrap();
        let term = ExtensionOpeningReductionTerm::new_sparse(sparse, factor.clone(), one).unwrap();
        let input_claim =
            ExtensionOpeningReductionProver::input_claim_from_terms(std::slice::from_ref(&term))
                .unwrap();
        let mut prover = ExtensionOpeningReductionProver::new(vec![term], input_claim).unwrap();

        let prover_poly = prover.compute_round_univariate(0, input_claim);
        for x in [zero, one, LossyField::from_u64(2), LossyField::from_u64(3)] {
            assert_eq!(
                prover_poly.evaluate(&x),
                reference_round_eval(&witness, &factor, input_claim, x),
                "sparse round 0 diverged from per-term Mul at x={x:?}"
            );
        }
    }
}
