use super::*;
use crate::DensePoly;
use akita_field::{Fp32, FpExt2, LiftBase, NegOneNr};
use akita_transcript::AkitaTranscript;
#[cfg(feature = "zk")]
use akita_types::FlatDigitBlocks;
use akita_types::{AkitaSetupSeed, FlatMatrix};

type F = Fp32<251>;
type E = FpExt2<F, NegOneNr>;

fn setup() -> AkitaExpandedSetup<F> {
    AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
        AkitaSetupSeed {
            max_num_vars: 3,
            max_num_batched_polys: 4,
            gen_ring_dim: 1,
            max_setup_len: 1,
            #[cfg(feature = "zk")]
            max_zk_b_len: 1,
            #[cfg(feature = "zk")]
            max_zk_d_len: 1,
            public_matrix_seed: [0u8; 32],
        },
        FlatMatrix::from_flat_data(vec![F::zero()], 1),
        #[cfg(feature = "zk")]
        FlatMatrix::from_flat_data(vec![F::zero()], 1),
        #[cfg(feature = "zk")]
        FlatMatrix::from_flat_data(vec![F::zero()], 1),
    )
}

#[test]
fn prover_claim_preparation_accepts_extension_points() {
    let point = [
        E::new(F::from_u64(1), F::from_u64(2)),
        E::new(F::from_u64(3), F::from_u64(4)),
    ];
    let polys = [
        DensePoly::<F, 2>::from_field_evals(2, &[F::from_u64(10), F::zero(), F::zero(), F::zero()])
            .expect("poly"),
        DensePoly::<F, 2>::from_field_evals(2, &[F::from_u64(11), F::zero(), F::zero(), F::zero()])
            .expect("poly"),
    ];
    let commitment = RingCommitment::<F, 2>::default();
    #[cfg(feature = "zk")]
    let hint = AkitaCommitmentHint::with_recomposed_inner_rows(
        Vec::new(),
        Vec::new(),
        vec![FlatDigitBlocks::empty()],
    );
    #[cfg(not(feature = "zk"))]
    let hint = AkitaCommitmentHint::new(Vec::new());
    let claims = (
        &point[..],
        vec![crate::CommittedPolynomials {
            polynomials: &polys[..],
            commitment: &commitment,
            hint,
        }],
    );

    let prepared = prepare_batched_prove_inputs::<F, E, DensePoly<F, 2>, 2>(&setup(), claims)
        .expect("extension-valued prover points should validate by shape");

    assert_eq!(prepared.opening_point, &point[..]);
    assert_eq!(prepared.opening_batch.num_claims(), 2);
    assert_eq!(
        prepared.opening_batch.num_polys_per_commitment_group(),
        &[2]
    );
    assert_eq!(prepared.opening_batch.claim_to_commitment_group(), &[0, 0]);
    assert_eq!(prepared.flat_polys, vec![&polys[0], &polys[1]]);
}

#[test]
fn recursive_extension_opening_reduction_pads_to_opening_cube() {
    let logical_w = RecursiveWitnessFlat::from_i8_digits(vec![1, -1, 2, 0, 3, -2]);
    let point = [
        E::new(F::from_u64(2), F::from_u64(3)),
        E::new(F::from_u64(5), F::from_u64(7)),
        E::new(F::from_u64(11), F::from_u64(13)),
    ];
    let logical_view = logical_w.view::<F, 2>().expect("valid suffix witness");
    let mut base_evals = logical_view.base_evals().expect("base evals");
    base_evals.resize(1usize << point.len(), F::zero());
    let expected_opening = base_evals
        .iter()
        .enumerate()
        .fold(E::zero(), |acc, (idx, &eval)| {
            let weight = point
                .iter()
                .enumerate()
                .fold(E::one(), |weight, (bit, &x)| {
                    if (idx >> bit) & 1 == 1 {
                        weight * x
                    } else {
                        weight * (E::one() - x)
                    }
                });
            acc + weight * E::lift_base(eval)
        });

    let mut transcript =
        AkitaTranscript::<F>::new(b"test/recursive-extension-opening-reduction-padding");
    #[cfg(feature = "zk")]
    let mut zk_hiding = ZkHidingProverState::new((1..=16).map(F::from_u64).collect::<Vec<_>>());
    let logical_polys = [&logical_view];
    let opening_batch = OpeningBatch::same_point(point.len(), 1).expect("opening batch");
    let proved = prove_extension_opening_reduction::<F, E, _, _, 2>(
        &logical_polys,
        &opening_batch,
        &point,
        #[cfg(feature = "zk")]
        None,
        true,
        &mut transcript,
        "recursive",
        #[cfg(feature = "zk")]
        &mut zk_hiding,
    )
    .expect("padded logical witnesses should reduce over the opening cube");

    assert_eq!(
        proved.reduction.proof.partials.len(),
        <E as ExtField<F>>::EXT_DEGREE
    );
    assert_eq!(proved.openings, vec![expected_opening]);
    assert_eq!(proved.reduction.proof.num_rounds(), point.len() - 1);
}
