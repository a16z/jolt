//! Dory PCS entry point sharing the CPU scheme's setup, commitment, proof,
//! and transcript formats. Hint combinations and the partitioned transparent
//! witness path's tier-two Miller loops run on CUDA. Final exponentiation
//! remains on the CPU. Standalone commitments, openings, setup, verification,
//! and ZK commitment finalization delegate to `DoryScheme`.

#[cfg(not(feature = "zk"))]
mod arena;
mod combine;
#[cfg(not(feature = "zk"))]
mod curve;
#[cfg(not(feature = "zk"))]
mod tier2;

use jolt_crypto::{Bn254G1, Commitment, VectorCommitment};
use jolt_dory::{
    DoryCommitment, DoryHint, DoryPartialCommitment, DoryProof, DoryProverSetup, DoryScheme,
    DoryVerifierSetup,
};
use jolt_field::Fr;
#[cfg(not(feature = "zk"))]
use jolt_field::Ring;
use jolt_openings::{
    AdditivelyHomomorphic, CommitmentScheme, OpeningsError, StreamingCommitment, ZkOpeningScheme,
    ZkStreamingCommitment,
};
use jolt_poly::MultilinearPoly;
use jolt_transcript::Transcript;
use jolt_verifier::{
    CommittedProgramPreprocessing, JoltVerifierPreprocessing, ProgramPreprocessing,
};

#[cfg(all(feature = "parallel", not(feature = "zk")))]
use rayon::prelude::*;

use crate::cuda::commitment::DeviceTier1Commitment;
#[cfg(not(feature = "zk"))]
use crate::cuda::commitment::FinishedColumn;
#[cfg(not(feature = "zk"))]
use crate::cuda::common::context::CudaKernelContext;
use crate::cuda::common::error::CudaError;
use crate::cuda::common::msm::{AffineLimbs, JacobianLimbs};

/// Dory PCS with CUDA hint combination and transparent witness tier two.
/// Selection is independent of the compute kernel registry; proof and setup
/// types are shared with `DoryScheme`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CudaDoryScheme;

impl CudaDoryScheme {
    pub fn setup_prover(max_num_vars: usize) -> DoryProverSetup {
        DoryScheme::setup_prover(max_num_vars)
    }

    pub fn setup_verifier(max_num_vars: usize) -> DoryVerifierSetup {
        DoryScheme::setup_verifier(max_num_vars)
    }

    /// Changes the PCS type of CPU verifier preprocessing without rebuilding
    /// its setup or program data, re-deriving the canonical program digest.
    pub fn adopt_verifier_preprocessing<VC>(
        source: JoltVerifierPreprocessing<DoryScheme, VC>,
    ) -> Result<JoltVerifierPreprocessing<Self, VC>, CudaError>
    where
        VC: VectorCommitment<Field = Fr>,
    {
        let program = match source.program {
            ProgramPreprocessing::Full(full) => ProgramPreprocessing::Full(full),
            ProgramPreprocessing::Committed(data) => {
                ProgramPreprocessing::Committed(CommittedProgramPreprocessing {
                    meta: data.meta,
                    memory_layout: data.memory_layout,
                    max_padded_trace_length: data.max_padded_trace_length,
                    program_image_commitment: data.program_image_commitment,
                    bytecode_commitment: data.bytecode_commitment,
                    trace_order: data.trace_order,
                })
            }
        };
        JoltVerifierPreprocessing::new(program, source.pcs_setup, source.vc_setup).map_err(|_| {
            CudaError::InvariantViolation {
                reason: "CUDA verifier preprocessing could not derive the program digest",
            }
        })
    }
}

impl DeviceTier1Commitment for CudaDoryScheme {
    const BATCHES_TIER2: bool = !cfg!(feature = "zk");

    #[cfg(not(feature = "zk"))]
    fn tier2_selected(
        context: &CudaKernelContext,
        setup: &Self::ProverSetup,
        columns: &[Vec<JacobianLimbs>],
        selection: &[usize],
    ) -> Result<Vec<(usize, FinishedColumn<Self>)>, CudaError> {
        let commitments = tier2::tier2_selected(context, setup, columns, selection)?;
        let blind = <Fr as Ring>::from_u64(0);
        let hints =
            tracing::info_span!("cuda_commit_hints", columns = selection.len()).in_scope(|| {
                #[cfg(feature = "parallel")]
                let hints = commitments
                    .par_iter()
                    .map(|&(index, _)| {
                        let rows = columns.get(index).ok_or(CudaError::InvariantViolation {
                            reason: "a tier-2 selection named a column outside the batch",
                        })?;
                        DoryScheme::partial_from_rows(setup, rows)
                    })
                    .collect::<Result<Vec<_>, _>>();
                #[cfg(not(feature = "parallel"))]
                let hints = commitments
                    .iter()
                    .map(|&(index, _)| {
                        let rows = columns.get(index).ok_or(CudaError::InvariantViolation {
                            reason: "a tier-2 selection named a column outside the batch",
                        })?;
                        DoryScheme::partial_from_rows(setup, rows)
                    })
                    .collect::<Result<Vec<_>, _>>();
                hints
            })?;
        commitments
            .into_iter()
            .zip(hints)
            .map(|((index, commitment), partial)| {
                Ok((
                    index,
                    (
                        commitment,
                        DoryHint::from_parts(partial.row_commitments, blind),
                    ),
                ))
            })
            .collect()
    }

    fn tier1_bases(setup: &Self::ProverSetup, count: usize) -> Result<Vec<AffineLimbs>, CudaError> {
        DoryScheme::tier1_bases(setup, count)
    }

    fn partial_from_rows(
        setup: &Self::ProverSetup,
        rows: &[JacobianLimbs],
    ) -> Result<Self::PartialCommitment, CudaError> {
        DoryScheme::partial_from_rows(setup, rows)
    }
}

impl Commitment for CudaDoryScheme {
    type Output = DoryCommitment;
}

impl CommitmentScheme for CudaDoryScheme {
    type Field = Fr;
    type Proof = DoryProof;
    type ProverSetup = DoryProverSetup;
    type VerifierSetup = DoryVerifierSetup;
    type OpeningHint = DoryHint;
    type SetupParams = <DoryScheme as CommitmentScheme>::SetupParams;

    fn setup(
        params: Self::SetupParams,
    ) -> Result<(Self::ProverSetup, Self::VerifierSetup), OpeningsError> {
        DoryScheme::setup(params)
    }

    fn verifier_setup(prover_setup: &Self::ProverSetup) -> Self::VerifierSetup {
        DoryScheme::verifier_setup(prover_setup)
    }

    fn commit<P: MultilinearPoly<Self::Field> + ?Sized>(
        poly: &P,
        setup: &Self::ProverSetup,
    ) -> Result<(Self::Output, Self::OpeningHint), OpeningsError> {
        DoryScheme::commit(poly, setup)
    }

    fn open<P: MultilinearPoly<Self::Field> + ?Sized>(
        poly: &P,
        point: &[Self::Field],
        eval: Self::Field,
        setup: &Self::ProverSetup,
        hint: Option<Self::OpeningHint>,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<Self::Proof, OpeningsError> {
        DoryScheme::open(poly, point, eval, setup, hint, transcript)
    }

    fn verify(
        commitment: &Self::Output,
        point: &[Self::Field],
        eval: Self::Field,
        proof: &Self::Proof,
        setup: &Self::VerifierSetup,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<(), OpeningsError> {
        DoryScheme::verify(commitment, point, eval, proof, setup, transcript)
    }
}

impl AdditivelyHomomorphic for CudaDoryScheme {
    fn combine(commitments: &[Self::Output], scalars: &[Self::Field]) -> Self::Output {
        DoryScheme::combine(commitments, scalars)
    }

    fn combine_hints(hints: Vec<Self::OpeningHint>, scalars: &[Self::Field]) -> Self::OpeningHint {
        combine::combine_hints(hints, scalars)
    }
}

impl StreamingCommitment for CudaDoryScheme {
    type PartialCommitment = DoryPartialCommitment;
    type OneHotChunkCommitment = <DoryScheme as StreamingCommitment>::OneHotChunkCommitment;
    type OneHotStreamContext = <DoryScheme as StreamingCommitment>::OneHotStreamContext;

    fn begin(setup: &Self::ProverSetup) -> Self::PartialCommitment {
        DoryScheme::begin(setup)
    }

    fn feed(
        partial: &mut Self::PartialCommitment,
        chunk: &[Self::Field],
        setup: &Self::ProverSetup,
    ) {
        DoryScheme::feed(partial, chunk, setup);
    }

    fn finish(partial: Self::PartialCommitment, setup: &Self::ProverSetup) -> Self::Output {
        DoryScheme::finish(partial, setup)
    }

    fn feed_zeros(
        partial: &mut Self::PartialCommitment,
        row_width: usize,
        rows: usize,
        setup: &Self::ProverSetup,
    ) {
        DoryScheme::feed_zeros(partial, row_width, rows, setup);
    }

    fn feed_u64(partial: &mut Self::PartialCommitment, chunk: &[u64], setup: &Self::ProverSetup) {
        DoryScheme::feed_u64(partial, chunk, setup);
    }

    fn feed_i128(partial: &mut Self::PartialCommitment, chunk: &[i128], setup: &Self::ProverSetup) {
        DoryScheme::feed_i128(partial, chunk, setup);
    }

    fn feed_i128_rows_with(
        partial: &mut Self::PartialCommitment,
        value: impl Fn(usize) -> i128 + Sync,
        count: usize,
        row_width: usize,
        setup: &Self::ProverSetup,
    ) {
        DoryScheme::feed_i128_rows_with(partial, value, count, row_width, setup);
    }

    fn begin_one_hot_column_major_stream(
        setup: &Self::ProverSetup,
        row_width: usize,
    ) -> Self::OneHotStreamContext {
        DoryScheme::begin_one_hot_column_major_stream(setup, row_width)
    }

    fn process_one_hot_chunk(
        context: &mut Self::OneHotStreamContext,
        setup: &Self::ProverSetup,
        one_hot_k: usize,
        chunk: &[Option<usize>],
    ) -> Self::OneHotChunkCommitment {
        DoryScheme::process_one_hot_chunk(context, setup, one_hot_k, chunk)
    }

    fn process_one_hot_chunks_with(
        context: &mut Self::OneHotStreamContext,
        setup: &Self::ProverSetup,
        one_hot_k: usize,
        hot_address: impl Fn(usize) -> Option<usize> + Sync,
        count: usize,
        chunk_width: usize,
    ) -> Vec<Self::OneHotChunkCommitment> {
        DoryScheme::process_one_hot_chunks_with(
            context,
            setup,
            one_hot_k,
            hot_address,
            count,
            chunk_width,
        )
    }

    fn finish_with_hint(
        partial: Self::PartialCommitment,
        setup: &Self::ProverSetup,
    ) -> (Self::Output, Self::OpeningHint) {
        DoryScheme::finish_with_hint(partial, setup)
    }

    fn finish_one_hot_column_major_chunks(
        setup: &Self::ProverSetup,
        one_hot_k: usize,
        chunks: &[Self::OneHotChunkCommitment],
    ) -> (Self::Output, Self::OpeningHint) {
        DoryScheme::finish_one_hot_column_major_chunks(setup, one_hot_k, chunks)
    }
}

impl ZkOpeningScheme for CudaDoryScheme {
    type HidingCommitment = Bn254G1;
    type Blind = Fr;

    fn commit_zk<P: MultilinearPoly<Self::Field> + ?Sized>(
        poly: &P,
        setup: &Self::ProverSetup,
    ) -> Result<(Self::Output, Self::OpeningHint), OpeningsError> {
        DoryScheme::commit_zk(poly, setup)
    }

    fn open_zk<P: MultilinearPoly<Self::Field> + ?Sized>(
        poly: &P,
        point: &[Self::Field],
        eval: Self::Field,
        setup: &Self::ProverSetup,
        hint: Self::OpeningHint,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<(Self::Proof, Self::HidingCommitment, Self::Blind), OpeningsError> {
        DoryScheme::open_zk(poly, point, eval, setup, hint, transcript)
    }

    fn verify_zk(
        commitment: &Self::Output,
        point: &[Self::Field],
        proof: &Self::Proof,
        setup: &Self::VerifierSetup,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<Self::HidingCommitment, OpeningsError> {
        DoryScheme::verify_zk(commitment, point, proof, setup, transcript)
    }
}

impl ZkStreamingCommitment for CudaDoryScheme {
    fn finish_zk_with_hint(
        partial: Self::PartialCommitment,
        setup: &Self::ProverSetup,
    ) -> (Self::Output, Self::OpeningHint) {
        DoryScheme::finish_zk_with_hint(partial, setup)
    }

    fn finish_zk_one_hot_column_major_chunks(
        setup: &Self::ProverSetup,
        one_hot_k: usize,
        chunks: &[Self::OneHotChunkCommitment],
    ) -> (Self::Output, Self::OpeningHint) {
        DoryScheme::finish_zk_one_hot_column_major_chunks(setup, one_hot_k, chunks)
    }
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test module: PCS and device operations fail loudly"
)]
mod tests {
    use jolt_crypto::Bn254G1;
    use jolt_field::{Field, Fr};
    use jolt_openings::{AdditivelyHomomorphic, CommitmentScheme};
    use jolt_poly::Polynomial;
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    use jolt_dory::{DoryHint, DoryScheme};

    use super::CudaDoryScheme;
    use crate::cuda::common::context::shared_context;
    use crate::cuda::common::msm::testing::point;

    const NUM_VARS: usize = 4;
    const PROVE_WIDE_HINTS: usize = 40;
    const PROVE_WIDE_ROWS: usize = 8_192;
    const PROVE_NARROW_HINTS: usize = 2;
    const PROVE_NARROW_ROWS: usize = 512;

    fn prove_scale_shape() -> Vec<usize> {
        let mut shape = vec![PROVE_WIDE_ROWS; PROVE_WIDE_HINTS];
        shape.extend(std::iter::repeat_n(PROVE_NARROW_ROWS, PROVE_NARROW_HINTS));
        shape
    }

    fn walked_row_commitments(rows: usize, start: u64) -> Vec<Bn254G1> {
        let step = point(1);
        let mut walk = point(start);
        (0..rows)
            .map(|_| {
                walk += step;
                Bn254G1::from(walk)
            })
            .collect()
    }

    fn shaped_hints(shape: &[usize], seed: u64) -> Vec<DoryHint> {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        shape
            .iter()
            .enumerate()
            .map(|(index, &rows)| {
                let start = 1 + index as u64 * (PROVE_WIDE_ROWS as u64 + 1);
                DoryHint::from_parts(
                    walked_row_commitments(rows, start),
                    <Fr as Field>::random(&mut rng),
                )
            })
            .collect()
    }

    fn assert_hints_match(got: &DoryHint, expected: &DoryHint, label: &str) {
        assert_eq!(
            got.row_commitments().len(),
            expected.row_commitments().len(),
            "{label}: combined row count diverged",
        );
        assert_eq!(
            got.commit_blind(),
            expected.commit_blind(),
            "{label}: combined commit blind diverged",
        );
        let divergence = got
            .row_commitments()
            .iter()
            .zip(expected.row_commitments())
            .position(|(got, expected)| got != expected);
        assert_eq!(divergence, None, "{label}: combined row diverged");
    }

    fn hints(rows: &[usize], seed: u64) -> Vec<DoryHint> {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let setup = DoryScheme::setup_prover(NUM_VARS);
        rows.iter()
            .map(|&count| {
                let poly = Polynomial::<Fr>::random(NUM_VARS, &mut rng);
                let (_, hint) = DoryScheme::commit(poly.evaluations(), &setup).unwrap();
                let (mut rows, blind) = (hint.row_commitments().to_vec(), hint.commit_blind());
                rows.truncate(count.max(1));
                DoryHint::from_parts(rows, blind)
            })
            .collect()
    }

    fn scalars(count: usize, seed: u64) -> Vec<Fr> {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        (0..count)
            .map(|_| <Fr as Field>::random(&mut rng))
            .collect()
    }

    #[test]
    fn combine_hints_matches_reference_dory() {
        if shared_context().is_none() {
            return;
        }
        for (index, shape) in [
            vec![4usize],
            vec![4, 4],
            vec![4, 4, 4, 4, 4],
            vec![1, 4, 2],
            vec![2, 1],
        ]
        .into_iter()
        .enumerate()
        {
            let source = hints(&shape, 1_000 + index as u64);
            let weights = scalars(shape.len(), 2_000 + index as u64);

            let expected = DoryScheme::combine_hints(source.clone(), &weights);
            let got = CudaDoryScheme::combine_hints(source, &weights);

            assert_eq!(
                got, expected,
                "combined hint diverged for row shape {shape:?}"
            );
        }
    }

    #[test]
    fn combine_hints_matches_reference_dory_at_prove_scale() {
        if shared_context().is_none() {
            return;
        }
        let shape = prove_scale_shape();
        let source = shaped_hints(&shape, 3_000);
        let weights = scalars(shape.len(), 4_000);

        let expected = DoryScheme::combine_hints(source.clone(), &weights);
        let got = super::combine::combine_on_device(&source, &weights)
            .expect("the device hint combination must not decline at prove scale");

        assert_hints_match(&got, &expected, "prove scale");
        assert_eq!(
            got.row_commitments().len(),
            PROVE_WIDE_ROWS,
            "the combined hint must be padded out to the widest source hint",
        );
    }
}
