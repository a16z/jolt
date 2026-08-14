//! Streaming (chunked) commitment for the Dory scheme.

use ark_bn254::{G1Affine, G1Projective};
use ark_ec::CurveGroup;
use dory::backends::arkworks::ArkG1;
use dory::primitives::arithmetic::DoryRoutines;
use jolt_crypto::ec::bn254::batch_addition::batch_g1_additions_multi_affine;
use jolt_crypto::Bn254G1;
use jolt_field::Fr;
use jolt_openings::{StreamingCommitment, ZkStreamingCommitment};
use rayon::prelude::*;

use crate::routines::JoltG1Routines;
use crate::scheme::{
    ark_to_jolt_fr, ark_to_jolt_g1, ark_to_jolt_g1_vec, ark_to_jolt_gt, commit_rows_tier_2,
    jolt_fr_to_ark, jolt_g1_vec_to_ark, srs_prefix, ArkFr,
};
use crate::types::{DoryCommitment, DoryHint, DoryPartialCommitment, DoryProverSetup};

impl crate::DoryScheme {
    #[tracing::instrument(skip_all, name = "DoryScheme::stream_finish_zk")]
    pub fn finish_zk(
        partial: DoryPartialCommitment,
        setup: &DoryProverSetup,
    ) -> (DoryCommitment, DoryHint) {
        validate_row_count(partial.row_commitments.len(), setup);
        let row_commitments = jolt_g1_vec_to_ark(partial.row_commitments);
        let (tier_2, commit_blind) = commit_rows_tier_2::<dory::ZK>(&row_commitments, setup);
        (
            DoryCommitment(ark_to_jolt_gt(&tier_2)),
            DoryHint::new(
                ark_to_jolt_g1_vec(row_commitments),
                ark_to_jolt_fr(&commit_blind),
            ),
        )
    }
}

impl StreamingCommitment for crate::DoryScheme {
    type PartialCommitment = DoryPartialCommitment;

    fn begin(_setup: &Self::ProverSetup) -> Self::PartialCommitment {
        DoryPartialCommitment {
            row_commitments: Vec::new(),
            scalar_affine_bases: None,
        }
    }

    /// Commits one full row of the polynomial as `MSM(g1_bases[..chunk.len()], chunk)`,
    /// matching the per-row work in [`DoryScheme::commit`](crate::DoryScheme::commit)'s
    /// dense path. Caller must feed every row at the same chunk width.
    #[tracing::instrument(skip_all, name = "DoryScheme::stream_feed")]
    fn feed(partial: &mut Self::PartialCommitment, chunk: &[Fr], setup: &Self::ProverSetup) {
        assert!(
            chunk.len().is_power_of_two(),
            "streaming: chunk length ({}) must be a power of two",
            chunk.len(),
        );
        assert!(
            chunk.len() <= setup.0.g1_vec.len(),
            "streaming: chunk length ({}) exceeds Dory SRS size ({})",
            chunk.len(),
            setup.0.g1_vec.len(),
        );

        let g1_bases = srs_prefix(&setup.0.g1_vec, chunk.len());
        let scalars: Vec<ArkFr> = chunk.iter().map(jolt_fr_to_ark).collect();
        let row_commitment = JoltG1Routines::msm(g1_bases, &scalars);
        partial.row_commitments.push(ark_to_jolt_g1(row_commitment));
    }

    /// Aggregates row commitments into the final tier-2 commitment, matching
    /// [`DoryScheme::commit`](crate::DoryScheme::commit). Asserts that the
    /// streamed row count is a power of two (the layout `DoryScheme::commit`
    /// produces).
    #[tracing::instrument(skip_all, name = "DoryScheme::stream_finish")]
    fn finish(partial: Self::PartialCommitment, setup: &Self::ProverSetup) -> Self::Output {
        let num_rows = partial.row_commitments.len();
        validate_row_count(num_rows, setup);

        let ark_rows = jolt_g1_vec_to_ark(partial.row_commitments);
        let (tier_2, _) = commit_rows_tier_2::<dory::Transparent>(&ark_rows, setup);
        DoryCommitment(ark_to_jolt_gt(&tier_2))
    }

    type OneHotChunkCommitment = Vec<Bn254G1>;
    type OneHotStreamContext = Vec<G1Affine>;

    #[tracing::instrument(skip_all, name = "DoryScheme::stream_feed_zeros")]
    fn feed_zeros(
        partial: &mut Self::PartialCommitment,
        row_width: usize,
        rows: usize,
        setup: &Self::ProverSetup,
    ) {
        if rows == 0 {
            return;
        }
        assert!(
            row_width.is_power_of_two(),
            "streaming: row width ({row_width}) must be a power of two",
        );
        assert!(
            row_width <= setup.0.g1_vec.len(),
            "streaming: row width ({}) exceeds Dory SRS size ({})",
            row_width,
            setup.0.g1_vec.len(),
        );
        partial
            .row_commitments
            .extend(std::iter::repeat_n(Bn254G1::default(), rows));
    }

    #[tracing::instrument(skip_all, name = "DoryScheme::stream_feed_u64")]
    fn feed_u64(partial: &mut Self::PartialCommitment, chunk: &[u64], setup: &Self::ProverSetup) {
        assert!(
            chunk.len().is_power_of_two(),
            "streaming: chunk length ({}) must be a power of two",
            chunk.len(),
        );
        assert!(
            chunk.len() <= setup.0.g1_vec.len(),
            "streaming: chunk length ({}) exceeds Dory SRS size ({})",
            chunk.len(),
            setup.0.g1_vec.len(),
        );

        let row_commitment = ark_ec::scalar_mul::variable_base::msm_u64::<G1Projective>(
            scalar_affine_bases(&mut partial.scalar_affine_bases, chunk.len(), setup),
            chunk,
            true,
        );
        partial
            .row_commitments
            .push(ark_to_jolt_g1(ArkG1(row_commitment)));
    }

    #[tracing::instrument(skip_all, name = "DoryScheme::stream_feed_i128")]
    fn feed_i128(partial: &mut Self::PartialCommitment, chunk: &[i128], setup: &Self::ProverSetup) {
        assert!(
            chunk.len().is_power_of_two(),
            "streaming: chunk length ({}) must be a power of two",
            chunk.len(),
        );
        assert!(
            chunk.len() <= setup.0.g1_vec.len(),
            "streaming: chunk length ({}) exceeds Dory SRS size ({})",
            chunk.len(),
            setup.0.g1_vec.len(),
        );

        let row_commitment = ark_ec::scalar_mul::variable_base::msm_i128::<G1Projective>(
            scalar_affine_bases(&mut partial.scalar_affine_bases, chunk.len(), setup),
            chunk,
            true,
        );
        partial
            .row_commitments
            .push(ark_to_jolt_g1(ArkG1(row_commitment)));
    }

    /// The parallel batch counterpart of [`feed_i128`](Self::feed_i128):
    /// windows materialize their own `row_width` values on the worker (a
    /// transient hundreds of KiB each) and commit in parallel (each MSM
    /// serial to avoid nested-pool oversubscription), appended in window
    /// order — the same row-commitment sequence as serial feeding, without
    /// ever staging the batch.
    #[tracing::instrument(
        skip_all,
        name = "DoryScheme::stream_feed_i128_rows_with",
        fields(rows = count / row_width.max(1))
    )]
    fn feed_i128_rows_with(
        partial: &mut Self::PartialCommitment,
        value: impl Fn(usize) -> i128 + Sync,
        count: usize,
        row_width: usize,
        setup: &Self::ProverSetup,
    ) {
        assert!(
            row_width.is_power_of_two(),
            "streaming: row width ({row_width}) must be a power of two",
        );
        assert!(
            row_width <= setup.0.g1_vec.len(),
            "streaming: row width ({}) exceeds Dory SRS size ({})",
            row_width,
            setup.0.g1_vec.len(),
        );
        assert!(
            count.is_multiple_of(row_width),
            "streaming: batch length ({count}) must be a multiple of the row width ({row_width})",
        );

        let bases = scalar_affine_bases(&mut partial.scalar_affine_bases, row_width, setup);
        let commitments: Vec<Bn254G1> = (0..count / row_width)
            .into_par_iter()
            .map(|window| {
                let base = window * row_width;
                let chunk: Vec<i128> = (base..base + row_width).map(&value).collect();
                ark_to_jolt_g1(ArkG1(ark_ec::scalar_mul::variable_base::msm_i128::<
                    G1Projective,
                >(bases, &chunk, true)))
            })
            .collect();
        partial.row_commitments.extend(commitments);
    }

    fn begin_one_hot_column_major_stream(
        setup: &Self::ProverSetup,
        row_width: usize,
    ) -> Self::OneHotStreamContext {
        assert!(
            row_width.is_power_of_two(),
            "streaming one-hot: row width ({row_width}) must be a power of two",
        );
        assert!(
            row_width <= setup.0.g1_vec.len(),
            "streaming one-hot: row width ({}) exceeds Dory SRS size ({})",
            row_width,
            setup.0.g1_vec.len(),
        );
        srs_prefix(&setup.0.g1_vec, row_width)
            .par_iter()
            .map(|base| base.0.into_affine())
            .collect()
    }

    #[tracing::instrument(skip_all, name = "DoryScheme::stream_process_one_hot_chunk")]
    fn process_one_hot_chunk(
        context: &mut Self::OneHotStreamContext,
        setup: &Self::ProverSetup,
        one_hot_k: usize,
        chunk: &[Option<usize>],
    ) -> Self::OneHotChunkCommitment {
        one_hot_chunk_commitments(context, setup, one_hot_k, chunk)
    }

    /// The parallel batch counterpart of
    /// [`process_one_hot_chunk`](Self::process_one_hot_chunk): windows
    /// materialize their own `chunk_width` hot addresses on the worker,
    /// share the cached affine bases read-only, and commit in parallel,
    /// collected in window order — the same chunk-commitment sequence as
    /// serial calls, without ever staging the batch.
    #[tracing::instrument(
        skip_all,
        name = "DoryScheme::stream_process_one_hot_chunks_with",
        fields(chunks = count / chunk_width.max(1))
    )]
    fn process_one_hot_chunks_with(
        context: &mut Self::OneHotStreamContext,
        setup: &Self::ProverSetup,
        one_hot_k: usize,
        hot_address: impl Fn(usize) -> Option<usize> + Sync,
        count: usize,
        chunk_width: usize,
    ) -> Vec<Self::OneHotChunkCommitment> {
        assert!(
            chunk_width != 0,
            "streaming one-hot: chunk width must be nonzero",
        );
        let context: &Self::OneHotStreamContext = context;
        (0..count.div_ceil(chunk_width))
            .into_par_iter()
            .map(|window| {
                let base = window * chunk_width;
                let top = (base + chunk_width).min(count);
                let chunk: Vec<Option<usize>> = (base..top).map(&hot_address).collect();
                one_hot_chunk_commitments(context, setup, one_hot_k, &chunk)
            })
            .collect()
    }

    #[tracing::instrument(skip_all, name = "DoryScheme::stream_finish_with_hint")]
    fn finish_with_hint(
        partial: Self::PartialCommitment,
        setup: &Self::ProverSetup,
    ) -> (Self::Output, Self::OpeningHint) {
        let num_rows = partial.row_commitments.len();
        validate_row_count(num_rows, setup);

        let ark_rows = jolt_g1_vec_to_ark(partial.row_commitments);
        let (tier_2, commit_blind) = commit_rows_tier_2::<dory::Transparent>(&ark_rows, setup);
        (
            DoryCommitment(ark_to_jolt_gt(&tier_2)),
            DoryHint::new(ark_to_jolt_g1_vec(ark_rows), ark_to_jolt_fr(&commit_blind)),
        )
    }

    #[tracing::instrument(skip_all, name = "DoryScheme::stream_finish_one_hot")]
    fn finish_one_hot_column_major_chunks(
        setup: &Self::ProverSetup,
        one_hot_k: usize,
        chunks: &[Self::OneHotChunkCommitment],
    ) -> (Self::Output, Self::OpeningHint) {
        finish_one_hot_column_major_chunks::<dory::Transparent>(setup, one_hot_k, chunks)
    }
}

impl ZkStreamingCommitment for crate::DoryScheme {
    fn finish_zk_with_hint(
        partial: Self::PartialCommitment,
        setup: &Self::ProverSetup,
    ) -> (Self::Output, Self::OpeningHint) {
        Self::finish_zk(partial, setup)
    }

    #[tracing::instrument(skip_all, name = "DoryScheme::stream_finish_one_hot_zk")]
    fn finish_zk_one_hot_column_major_chunks(
        setup: &Self::ProverSetup,
        one_hot_k: usize,
        chunks: &[Self::OneHotChunkCommitment],
    ) -> (Self::Output, Self::OpeningHint) {
        finish_one_hot_column_major_chunks::<dory::ZK>(setup, one_hot_k, chunks)
    }
}

fn finish_one_hot_column_major_chunks<M: dory::Mode>(
    setup: &DoryProverSetup,
    one_hot_k: usize,
    chunks: &[Vec<Bn254G1>],
) -> (DoryCommitment, DoryHint) {
    assert!(
        one_hot_k != 0,
        "streaming one-hot: one_hot_k must be nonzero",
    );
    assert!(
        !chunks.is_empty(),
        "streaming one-hot: cannot finish an empty chunk list",
    );
    for chunk in chunks {
        assert_eq!(
            chunk.len(),
            one_hot_k,
            "streaming one-hot: chunk row count must match one_hot_k",
        );
    }

    let chunk_count = chunks.len();
    let mut row_commitments = vec![Bn254G1::default(); chunk_count * one_hot_k];
    row_commitments
        .par_chunks_mut(chunk_count)
        .enumerate()
        .for_each(|(row, row_commitments)| {
            for (row_commitment, chunk) in row_commitments.iter_mut().zip(chunks) {
                #[expect(
                    clippy::indexing_slicing,
                    reason = "every chunk is asserted to have one_hot_k rows and par_chunks_mut(chunk_count) yields one_hot_k row slices"
                )]
                let value = chunk[row];
                *row_commitment = value;
            }
        });
    validate_row_count(row_commitments.len(), setup);

    let ark_rows = jolt_g1_vec_to_ark(row_commitments);
    let (tier_2, commit_blind) = commit_rows_tier_2::<M>(&ark_rows, setup);
    (
        DoryCommitment(ark_to_jolt_gt(&tier_2)),
        DoryHint::new(ark_to_jolt_g1_vec(ark_rows), ark_to_jolt_fr(&commit_blind)),
    )
}

/// One column-major one-hot chunk's `one_hot_k` partial row commitments —
/// the shared body behind the single and batch streaming entry points.
fn one_hot_chunk_commitments(
    bases: &[G1Affine],
    setup: &DoryProverSetup,
    one_hot_k: usize,
    chunk: &[Option<usize>],
) -> Vec<Bn254G1> {
    assert!(
        one_hot_k != 0,
        "streaming one-hot: one_hot_k must be nonzero",
    );
    assert!(
        chunk.len().is_power_of_two(),
        "streaming one-hot: chunk length ({}) must be a power of two",
        chunk.len(),
    );
    assert!(
        chunk.len() <= setup.0.g1_vec.len(),
        "streaming one-hot: chunk length ({}) exceeds Dory SRS size ({})",
        chunk.len(),
        setup.0.g1_vec.len(),
    );
    assert!(
        chunk.len() <= bases.len(),
        "streaming one-hot: chunk length ({}) exceeds cached base count ({})",
        chunk.len(),
        bases.len(),
    );
    let mut indices_per_k = vec![Vec::new(); one_hot_k];
    for (column, hot_row) in chunk.iter().copied().enumerate() {
        if let Some(hot_row) = hot_row {
            assert!(
                hot_row < one_hot_k,
                "streaming one-hot: hot row {hot_row} outside k={one_hot_k}",
            );
            #[expect(
                clippy::indexing_slicing,
                reason = "hot_row < one_hot_k is asserted above and indices_per_k has one_hot_k buckets"
            )]
            indices_per_k[hot_row].push(column);
        }
    }

    let additions = batch_g1_additions_multi_affine(srs_prefix(bases, chunk.len()), &indices_per_k);
    let mut row_commitments = vec![Bn254G1::default(); one_hot_k];
    for (row_commitment, (indices, addition)) in row_commitments
        .iter_mut()
        .zip(indices_per_k.iter().zip(additions))
    {
        if !indices.is_empty() {
            *row_commitment = ark_to_jolt_g1(ArkG1(addition.into()));
        }
    }
    row_commitments
}

fn validate_row_count(num_rows: usize, setup: &DoryProverSetup) {
    assert!(
        num_rows.is_power_of_two(),
        "streaming: row count ({num_rows}) must be a power of two",
    );
    assert!(
        num_rows <= setup.0.g2_vec.len(),
        "streaming: row count ({}) exceeds Dory SRS size ({})",
        num_rows,
        setup.0.g2_vec.len(),
    );
}

/// Fill and borrow the partial commitment's affine-base cache. Takes the
/// cache field (not the whole partial) so callers can hold the bases while
/// appending to the sibling `row_commitments` field.
fn scalar_affine_bases<'a>(
    cache: &'a mut Option<Vec<G1Affine>>,
    row_width: usize,
    setup: &DoryProverSetup,
) -> &'a [G1Affine] {
    let bases = cache.get_or_insert_with(|| {
        srs_prefix(&setup.0.g1_vec, row_width)
            .iter()
            .map(|base| base.0.into_affine())
            .collect()
    });
    if bases.len() < row_width {
        bases.extend(
            srs_prefix(&setup.0.g1_vec, row_width)
                .iter()
                .skip(bases.len())
                .map(|base| base.0.into_affine()),
        );
    }
    srs_prefix(bases, row_width)
}

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "tests unwrap successful PCS operations")]
    #![expect(clippy::indexing_slicing, reason = "tests index fixture data")]

    use jolt_field::FromPrimitiveInt;
    use jolt_field::RandomSampling;
    use jolt_openings::{
        CommitmentScheme, StreamingCommitment, ZkOpeningScheme, ZkStreamingCommitment,
    };
    use jolt_poly::{MultilinearPoly, OneHotIndexOrder, OneHotPolynomial};
    use jolt_transcript::Transcript;
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    use jolt_field::Fr;

    use crate::DoryScheme;

    #[test]
    fn streaming_matches_direct() {
        let num_vars: usize = 4;
        let num_cols = 1usize << num_vars.div_ceil(2);
        let num_rows = 1usize << (num_vars - num_vars.div_ceil(2));
        let mut rng = ChaCha20Rng::seed_from_u64(99);

        let prover_setup = DoryScheme::setup_prover(num_vars);

        let evals: Vec<Fr> = (0..num_rows * num_cols)
            .map(|_| <Fr as RandomSampling>::random(&mut rng))
            .collect();

        let poly = jolt_poly::Polynomial::new(evals.clone());
        let (direct, _) = DoryScheme::commit(poly.evaluations(), &prover_setup).unwrap();

        let mut partial = DoryScheme::begin(&prover_setup);
        for row in evals.chunks(num_cols) {
            DoryScheme::feed(&mut partial, row, &prover_setup);
        }
        let (streamed, hint) = DoryScheme::finish_with_hint(partial, &prover_setup);

        assert_eq!(
            direct, streamed,
            "streaming and direct commitments must match"
        );

        let point: Vec<Fr> = (0..num_vars)
            .map(|_| <Fr as RandomSampling>::random(&mut rng))
            .collect();
        let eval = poly.evaluate(&point);
        let mut prove_transcript = jolt_transcript::Blake2bTranscript::new(b"stream-open");
        let proof = DoryScheme::open(
            &poly,
            &point,
            eval,
            &prover_setup,
            Some(hint),
            &mut prove_transcript,
        )
        .unwrap();
        let verifier_setup = DoryScheme::verifier_setup(&prover_setup);
        let mut verify_transcript = jolt_transcript::Blake2bTranscript::new(b"stream-open");
        let result = DoryScheme::verify(
            &streamed,
            &point,
            eval,
            &proof,
            &verifier_setup,
            &mut verify_transcript,
        );
        assert!(result.is_ok(), "streaming hint should open: {result:?}");
    }

    #[test]
    fn streaming_u64_matches_direct() {
        let num_vars: usize = 4;
        let num_cols = 1usize << num_vars.div_ceil(2);
        let mut rng = ChaCha20Rng::seed_from_u64(199);

        let prover_setup = DoryScheme::setup_prover(num_vars);
        let evals_u64: Vec<u64> = (0..(1usize << num_vars))
            .map(|index| (index as u64 + 1) * 17)
            .collect();
        let evals = evals_u64
            .iter()
            .copied()
            .map(Fr::from_u64)
            .collect::<Vec<_>>();

        let poly = jolt_poly::Polynomial::new(evals.clone());
        let (direct, _) = DoryScheme::commit(poly.evaluations(), &prover_setup).unwrap();

        let mut partial = DoryScheme::begin(&prover_setup);
        for row in evals_u64.chunks(num_cols) {
            DoryScheme::feed_u64(&mut partial, row, &prover_setup);
        }
        let (streamed, hint) = DoryScheme::finish_with_hint(partial, &prover_setup);

        assert_eq!(
            direct, streamed,
            "u64 streaming and direct commitments must match"
        );

        let point: Vec<Fr> = (0..num_vars)
            .map(|_| <Fr as RandomSampling>::random(&mut rng))
            .collect();
        let eval = poly.evaluate(&point);
        let mut prove_transcript = jolt_transcript::Blake2bTranscript::new(b"u64-stream-open");
        let proof = DoryScheme::open(
            &poly,
            &point,
            eval,
            &prover_setup,
            Some(hint),
            &mut prove_transcript,
        )
        .unwrap();
        let verifier_setup = DoryScheme::verifier_setup(&prover_setup);
        let mut verify_transcript = jolt_transcript::Blake2bTranscript::new(b"u64-stream-open");
        let result = DoryScheme::verify(
            &streamed,
            &point,
            eval,
            &proof,
            &verifier_setup,
            &mut verify_transcript,
        );
        assert!(result.is_ok(), "u64 streaming hint should open: {result:?}");
    }

    #[test]
    fn streaming_zero_rows_match_explicit_zero_feeds() {
        let num_vars: usize = 6;
        let num_cols = 1usize << num_vars.div_ceil(2);
        let num_rows = 1usize << (num_vars - num_vars.div_ceil(2));
        let prover_setup = DoryScheme::setup_prover(num_vars);
        let zero_row = vec![Fr::from_u64(0); num_cols];

        let mut explicit = DoryScheme::begin(&prover_setup);
        for _ in 0..num_rows {
            DoryScheme::feed(&mut explicit, &zero_row, &prover_setup);
        }
        let (explicit_commitment, explicit_hint) =
            DoryScheme::finish_with_hint(explicit, &prover_setup);

        let mut fast = DoryScheme::begin(&prover_setup);
        DoryScheme::feed_zeros(&mut fast, num_cols, num_rows, &prover_setup);
        let (fast_commitment, fast_hint) = DoryScheme::finish_with_hint(fast, &prover_setup);

        assert_eq!(explicit_commitment, fast_commitment);
        assert_eq!(explicit_hint.row_commitments, fast_hint.row_commitments);
        assert_eq!(explicit_hint.commit_blind, fast_hint.commit_blind);
    }

    #[test]
    fn streaming_zero_rows_zk_open_and_verify() {
        let num_vars: usize = 6;
        let num_cols = 1usize << num_vars.div_ceil(2);
        let num_rows = 1usize << (num_vars - num_vars.div_ceil(2));
        let prover_setup = DoryScheme::setup_prover(num_vars);
        let verifier_setup = DoryScheme::verifier_setup(&prover_setup);
        let poly = jolt_poly::Polynomial::new(vec![Fr::from_u64(0); 1usize << num_vars]);

        let mut partial = DoryScheme::begin(&prover_setup);
        DoryScheme::feed_zeros(&mut partial, num_cols, num_rows, &prover_setup);
        let (commitment, hint) = DoryScheme::finish_zk(partial, &prover_setup);

        let mut rng = ChaCha20Rng::seed_from_u64(313);
        let point = (0..num_vars)
            .map(|_| <Fr as RandomSampling>::random(&mut rng))
            .collect::<Vec<_>>();
        let eval = Fr::from_u64(0);
        let mut prove_transcript = jolt_transcript::Blake2bTranscript::new(b"zero-zk-open");
        let (proof, _, _) = DoryScheme::open_zk(
            &poly,
            &point,
            eval,
            &prover_setup,
            hint,
            &mut prove_transcript,
        )
        .unwrap();
        let mut verify_transcript = jolt_transcript::Blake2bTranscript::new(b"zero-zk-open");
        let result = DoryScheme::verify_zk(
            &commitment,
            &point,
            &proof,
            &verifier_setup,
            &mut verify_transcript,
        );

        assert!(result.is_ok(), "zero-row ZK streaming hint should open");
    }

    #[test]
    fn streaming_one_hot_column_major_matches_direct() {
        let trace_rows = 8usize;
        let one_hot_k = 4usize;
        let num_vars = (trace_rows * one_hot_k).ilog2() as usize;
        let chunk_width = 1usize << num_vars.div_ceil(2);
        let prover_setup = DoryScheme::setup_prover(num_vars);
        let indices = [
            Some(0),
            Some(3),
            None,
            Some(1),
            Some(2),
            None,
            Some(3),
            Some(0),
        ];
        let poly = OneHotPolynomial::new_with_index_order(
            one_hot_k,
            indices
                .iter()
                .copied()
                .map(|value| value.map(|v| v as u8))
                .collect(),
            OneHotIndexOrder::ColumnMajor,
        );
        let (direct, direct_hint) = DoryScheme::commit(&poly, &prover_setup).unwrap();

        let mut context = DoryScheme::begin_one_hot_column_major_stream(&prover_setup, chunk_width);
        let chunks = indices
            .chunks(chunk_width)
            .map(|chunk| {
                DoryScheme::process_one_hot_chunk(&mut context, &prover_setup, one_hot_k, chunk)
            })
            .collect::<Vec<_>>();
        let (streamed, streamed_hint) =
            DoryScheme::finish_one_hot_column_major_chunks(&prover_setup, one_hot_k, &chunks);

        assert_eq!(direct, streamed);
        assert_eq!(direct_hint.row_commitments, streamed_hint.row_commitments);
        assert_eq!(direct_hint.commit_blind, streamed_hint.commit_blind);
    }

    #[test]
    fn streaming_one_hot_column_major_zk_opens_and_verifies() {
        let trace_rows = 8usize;
        let one_hot_k = 4usize;
        let num_vars = (trace_rows * one_hot_k).ilog2() as usize;
        let chunk_width = 1usize << num_vars.div_ceil(2);
        let prover_setup = DoryScheme::setup_prover(num_vars);
        let verifier_setup = DoryScheme::verifier_setup(&prover_setup);
        let indices = [
            Some(0),
            Some(3),
            None,
            Some(1),
            Some(2),
            None,
            Some(3),
            Some(0),
        ];
        let poly = OneHotPolynomial::new_with_index_order(
            one_hot_k,
            indices
                .iter()
                .copied()
                .map(|value| value.map(|v| v as u8))
                .collect(),
            OneHotIndexOrder::ColumnMajor,
        );
        let mut context = DoryScheme::begin_one_hot_column_major_stream(&prover_setup, chunk_width);
        let chunks = indices
            .chunks(chunk_width)
            .map(|chunk| {
                DoryScheme::process_one_hot_chunk(&mut context, &prover_setup, one_hot_k, chunk)
            })
            .collect::<Vec<_>>();
        let (commitment, hint) =
            DoryScheme::finish_zk_one_hot_column_major_chunks(&prover_setup, one_hot_k, &chunks);

        let mut rng = ChaCha20Rng::seed_from_u64(317);
        let point = (0..num_vars)
            .map(|_| <Fr as RandomSampling>::random(&mut rng))
            .collect::<Vec<_>>();
        let eval = poly.evaluate(&point);
        let mut prove_transcript = jolt_transcript::Blake2bTranscript::new(b"one-hot-zk-open");
        let (proof, _, _) = DoryScheme::open_zk(
            &poly,
            &point,
            eval,
            &prover_setup,
            hint,
            &mut prove_transcript,
        )
        .unwrap();
        let mut verify_transcript = jolt_transcript::Blake2bTranscript::new(b"one-hot-zk-open");
        let result = DoryScheme::verify_zk(
            &commitment,
            &point,
            &proof,
            &verifier_setup,
            &mut verify_transcript,
        );

        assert!(result.is_ok(), "one-hot ZK streaming hint should open");
    }

    #[test]
    fn streaming_i128_matches_direct() {
        let num_vars: usize = 4;
        let num_cols = 1usize << num_vars.div_ceil(2);
        let mut rng = ChaCha20Rng::seed_from_u64(211);

        let prover_setup = DoryScheme::setup_prover(num_vars);
        let evals_i128: Vec<i128> = (0..(1usize << num_vars))
            .map(|index| {
                let magnitude = (index as i128 + 1) * 19;
                if index % 3 == 0 {
                    -magnitude
                } else {
                    magnitude
                }
            })
            .collect();
        let evals = evals_i128
            .iter()
            .copied()
            .map(Fr::from_i128)
            .collect::<Vec<_>>();

        let poly = jolt_poly::Polynomial::new(evals.clone());
        let (direct, _) = DoryScheme::commit(poly.evaluations(), &prover_setup).unwrap();

        let mut partial = DoryScheme::begin(&prover_setup);
        for row in evals_i128.chunks(num_cols) {
            DoryScheme::feed_i128(&mut partial, row, &prover_setup);
        }
        let (streamed, hint) = DoryScheme::finish_with_hint(partial, &prover_setup);

        assert_eq!(
            direct, streamed,
            "i128 streaming and direct commitments must match"
        );

        let point: Vec<Fr> = (0..num_vars)
            .map(|_| <Fr as RandomSampling>::random(&mut rng))
            .collect();
        let eval = poly.evaluate(&point);
        let mut prove_transcript = jolt_transcript::Blake2bTranscript::new(b"i128-stream-open");
        let proof = DoryScheme::open(
            &poly,
            &point,
            eval,
            &prover_setup,
            Some(hint),
            &mut prove_transcript,
        )
        .unwrap();
        let verifier_setup = DoryScheme::verifier_setup(&prover_setup);
        let mut verify_transcript = jolt_transcript::Blake2bTranscript::new(b"i128-stream-open");
        let result = DoryScheme::verify(
            &streamed,
            &point,
            eval,
            &proof,
            &verifier_setup,
            &mut verify_transcript,
        );
        assert!(
            result.is_ok(),
            "i128 streaming hint should open: {result:?}"
        );
    }

    #[test]
    fn feed_i128_rows_with_matches_serial_feeds() {
        let num_vars: usize = 6;
        let num_cols = 1usize << num_vars.div_ceil(2);
        let prover_setup = DoryScheme::setup_prover(num_vars);
        let rows_i128: Vec<i128> = (0..(1usize << num_vars))
            .map(|index| {
                let magnitude = (index as i128 + 7) * 23;
                if index % 2 == 0 {
                    -magnitude
                } else {
                    magnitude
                }
            })
            .collect();

        let mut serial = DoryScheme::begin(&prover_setup);
        for row in rows_i128.chunks(num_cols) {
            DoryScheme::feed_i128(&mut serial, row, &prover_setup);
        }
        let (serial_commitment, serial_hint) = DoryScheme::finish_with_hint(serial, &prover_setup);

        let mut batched = DoryScheme::begin(&prover_setup);
        DoryScheme::feed_i128_rows_with(
            &mut batched,
            |index| rows_i128[index],
            rows_i128.len(),
            num_cols,
            &prover_setup,
        );
        let (batched_commitment, batched_hint) =
            DoryScheme::finish_with_hint(batched, &prover_setup);

        assert_eq!(serial_commitment, batched_commitment);
        assert_eq!(serial_hint.row_commitments, batched_hint.row_commitments);
    }

    #[test]
    fn process_one_hot_chunks_with_matches_serial_calls() {
        let trace_rows = 16usize;
        let one_hot_k = 4usize;
        let num_vars = (trace_rows * one_hot_k).ilog2() as usize;
        let chunk_width = 1usize << num_vars.div_ceil(2);
        let prover_setup = DoryScheme::setup_prover(num_vars);
        let indices: Vec<Option<usize>> = (0..trace_rows)
            .map(|cycle| (cycle % 5 != 4).then_some(cycle % one_hot_k))
            .collect();

        let mut context = DoryScheme::begin_one_hot_column_major_stream(&prover_setup, chunk_width);
        let serial = indices
            .chunks(chunk_width)
            .map(|chunk| {
                DoryScheme::process_one_hot_chunk(&mut context, &prover_setup, one_hot_k, chunk)
            })
            .collect::<Vec<_>>();
        let batched = DoryScheme::process_one_hot_chunks_with(
            &mut context,
            &prover_setup,
            one_hot_k,
            |index| indices[index],
            indices.len(),
            chunk_width,
        );

        assert_eq!(serial, batched);
    }
}
