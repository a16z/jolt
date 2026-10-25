//! Dory PCS entry point sharing the CPU scheme's setup, commitment, proof,
//! and transcript formats. All operations delegate to `DoryScheme`.

use jolt_crypto::{Bn254G1, Commitment, VectorCommitment};
use jolt_dory::{
    DoryCommitment, DoryHint, DoryPartialCommitment, DoryProof, DoryProverSetup, DoryScheme,
    DoryVerifierSetup,
};
use jolt_field::Fr;
use jolt_openings::{
    AdditivelyHomomorphic, CommitmentScheme, OpeningsError, StreamingCommitment, ZkOpeningScheme,
    ZkStreamingCommitment,
};
use jolt_poly::MultilinearPoly;
use jolt_transcript::Transcript;
use jolt_verifier::{
    CommittedProgramPreprocessing, JoltVerifierPreprocessing, ProgramPreprocessing,
};

use crate::cuda::commitment::DeviceTier1Commitment;
use crate::cuda::common::error::CudaError;
use crate::cuda::common::msm::{AffineLimbs, JacobianLimbs};

/// CPU-backed Dory PCS selected independently of the compute kernel registry.
/// Its proof and setup types are shared with `DoryScheme`.
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
        DoryScheme::combine_hints(hints, scalars)
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
