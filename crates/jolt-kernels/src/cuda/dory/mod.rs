mod arena;
mod combine;
mod curve;
mod gt;
mod handle;
mod routines;

use dory::backends::arkworks::ArkFr;
use dory::mode::Transparent;
use dory::setup::ProverSetup;
use jolt_crypto::{Bn254G1, Commitment, VectorCommitment};
use jolt_field::Fr;
use jolt_openings::{
    AdditivelyHomomorphic, CommitmentScheme, OpeningsError, StreamingCommitment, ZkOpeningScheme,
    ZkStreamingCommitment,
};
use jolt_poly::MultilinearPoly;
use jolt_transcript::Transcript;
use jolt_verifier::{JoltVerifierPreprocessing, ProgramPreprocessing};

use crate::cuda::commitment::DeviceTier1Commitment;
use crate::cuda::common::error::CudaError;
use crate::cuda::common::msm::{AffineLimbs, JacobianLimbs};

use jolt_dory::{
    compute_row_commitments, DoryCommitment, DoryHint, DoryPartialCommitment, DoryProof,
    DoryProverSetup, DoryScheme, DorySourceAdapter, DoryVerifierSetup,
};

use self::curve::{CudaBN254, CudaDoryTranscript};
use self::routines::{CudaG1Routines, CudaG2Routines};

const ARENA_SLACK: usize = 4_096;

fn resident_prover_setup(setup: &DoryProverSetup, bases: usize) -> ProverSetup<CudaBN254> {
    let width = bases.min(setup.0.g1_vec.len());
    let g1: Vec<_> = setup.0.g1_vec[..width].iter().map(|base| base.0).collect();
    let rows = bases.min(setup.0.g2_vec.len());
    let g2: Vec<_> = setup.0.g2_vec[..rows].iter().map(|base| base.0).collect();
    ProverSetup {
        g1_vec: handle::store_frozen(&g1),
        g2_vec: handle::store_frozen_g2(&g2),
        h1: handle::DeviceG1::store(&setup.0.h1.0),
        h2: handle::DeviceG2::store(&setup.0.h2.0),
        ht: setup.0.ht.into(),
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CudaDoryScheme;

impl CudaDoryScheme {
    pub fn setup_prover(max_num_vars: usize) -> DoryProverSetup {
        DoryScheme::setup_prover(max_num_vars)
    }

    pub fn setup_verifier(max_num_vars: usize) -> DoryVerifierSetup {
        DoryScheme::setup_verifier(max_num_vars)
    }

    pub fn adopt_verifier_preprocessing<VC>(
        source: JoltVerifierPreprocessing<DoryScheme, VC>,
    ) -> Result<JoltVerifierPreprocessing<Self, VC>, CudaError>
    where
        VC: VectorCommitment<Field = Fr>,
    {
        let program = match source.program {
            ProgramPreprocessing::Full(full) => ProgramPreprocessing::Full(full),
            ProgramPreprocessing::Committed(_) => {
                return Err(CudaError::NotImplemented {
                    kernel: "CudaDoryScheme cannot adopt a committed-program preprocessing",
                })
            }
        };
        Ok(JoltVerifierPreprocessing::new(
            program,
            source.preprocessing_digest,
            source.pcs_setup,
            source.vc_setup,
        ))
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

    #[tracing::instrument(skip_all, name = "CudaDoryScheme::open")]
    fn open<P: MultilinearPoly<Self::Field> + ?Sized>(
        poly: &P,
        point: &[Self::Field],
        _eval: Self::Field,
        setup: &Self::ProverSetup,
        hint: Option<Self::OpeningHint>,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<Self::Proof, OpeningsError> {
        let num_vars = point.len();
        let sigma = num_vars.div_ceil(2);
        let nu = num_vars - sigma;
        let adapter = DorySourceAdapter::new(poly);

        let (row_commitments, commit_blind) = match hint {
            Some(hint) => (
                hint.row_commitments
                    .into_iter()
                    .map(|point| point.into_inner())
                    .collect::<Vec<_>>(),
                ArkFr(hint.commit_blind.into()),
            ),
            None => (
                compute_row_commitments(poly, setup)?
                    .into_iter()
                    .map(|point| point.0)
                    .collect(),
                ArkFr(ark_bn254::Fr::from(0u64)),
            ),
        };

        let bases = 1usize << sigma;
        let capacity = 4 * bases.max(row_commitments.len()) + ARENA_SLACK;
        let _arena = arena::open(capacity, capacity).map_err(|error| {
            OpeningsError::ProveFailed(format!("the Dory arena did not open: {error:?}"))
        })?;
        gt::reset();
        let resident_setup = resident_prover_setup(setup, bases);
        let resident_rows = handle::store_all(&row_commitments);

        let ark_point: Vec<ArkFr> = point
            .iter()
            .rev()
            .map(|coordinate| ArkFr((*coordinate).into()))
            .collect();
        let mut dory_transcript = CudaDoryTranscript::new(transcript);

        let (proof, _blind) =
            dory::prove::<ArkFr, CudaBN254, CudaG1Routines, CudaG2Routines, _, _, Transparent>(
                &adapter,
                &ark_point,
                resident_rows,
                commit_blind,
                nu,
                sigma,
                &resident_setup,
                &mut dory_transcript,
            )
            .map_err(|error| {
                OpeningsError::ProveFailed(format!("dory::prove failed: {error:?}"))
            })?;

        if arena::poisoned() {
            return Err(OpeningsError::ProveFailed(
                "the Dory arena was poisoned during the opening".to_owned(),
            ));
        }

        let rebound = curve::rebind_proof(proof)
            .map_err(|reason| OpeningsError::ProveFailed(reason.to_owned()))?;
        Ok(DoryProof(rebound))
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

    fn open_batch(
        polynomials: &[&dyn MultilinearPoly<Self::Field>],
        point: &[Self::Field],
        evaluations: &[Self::Field],
        setup: &Self::ProverSetup,
        hint: Self::OpeningHint,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<Self::Proof, OpeningsError> {
        DoryScheme::open_batch(polynomials, point, evaluations, setup, hint, transcript)
    }

    fn verify_batch(
        commitment: &Self::Output,
        point: &[Self::Field],
        evaluations: &[Self::Field],
        proof: &Self::Proof,
        setup: &Self::VerifierSetup,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<(), OpeningsError> {
        DoryScheme::verify_batch(commitment, point, evaluations, proof, setup, transcript)
    }
}

impl AdditivelyHomomorphic for CudaDoryScheme {
    fn combine(commitments: &[Self::Output], scalars: &[Self::Field]) -> Self::Output {
        DoryScheme::combine(commitments, scalars)
    }

    #[tracing::instrument(skip_all, name = "CudaDoryScheme::combine_hints")]
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
    use jolt_field::{Fr, FromPrimitiveInt, RandomSampling};
    use jolt_openings::{AdditivelyHomomorphic, CommitmentScheme};
    use jolt_poly::Polynomial;
    use jolt_transcript::{Blake2bTranscript, Transcript};
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    use jolt_dory::{DoryHint, DoryProof, DoryProverSetup, DoryScheme};

    use super::CudaDoryScheme;
    use crate::cuda::common::context::shared_context;
    use crate::cuda::common::msm::testing::point;

    const NUM_VARS: usize = 4;
    const PROVE_SCALE_VARS: usize = 20;
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
                DoryHint::new(
                    walked_row_commitments(rows, start),
                    <Fr as RandomSampling>::random(&mut rng),
                )
            })
            .collect()
    }

    fn assert_hints_match(got: &DoryHint, expected: &DoryHint, label: &str) {
        assert_eq!(
            got.row_commitments.len(),
            expected.row_commitments.len(),
            "{label}: combined row count diverged",
        );
        assert_eq!(
            got.commit_blind, expected.commit_blind,
            "{label}: combined commit blind diverged",
        );
        let divergence = got
            .row_commitments
            .iter()
            .zip(&expected.row_commitments)
            .position(|(got, expected)| got != expected);
        assert_eq!(divergence, None, "{label}: combined row diverged");
    }

    fn open_pair(
        poly: &Polynomial<Fr>,
        point: &[Fr],
        eval: Fr,
        setup: &DoryProverSetup,
        hint: DoryHint,
    ) -> (DoryProof, DoryProof) {
        let mut transcript = Blake2bTranscript::new(b"dory");
        let got = CudaDoryScheme::open(
            poly,
            point,
            eval,
            setup,
            Some(hint.clone()),
            &mut transcript,
        )
        .expect("cuda open");

        let mut oracle_transcript = Blake2bTranscript::new(b"dory");
        let expected =
            DoryScheme::open(poly, point, eval, setup, Some(hint), &mut oracle_transcript)
                .expect("dory open");
        (got, expected)
    }

    fn opening_rows(num_vars: usize) -> usize {
        1usize << (num_vars - num_vars.div_ceil(2))
    }

    fn hints(rows: &[usize], seed: u64) -> Vec<DoryHint> {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let setup = DoryScheme::setup_prover(NUM_VARS);
        rows.iter()
            .map(|&count| {
                let poly = Polynomial::<Fr>::random(NUM_VARS, &mut rng);
                let (_, hint) = DoryScheme::commit(poly.evaluations(), &setup).unwrap();
                let mut hint = hint;
                hint.row_commitments.truncate(count.max(1));
                hint
            })
            .collect()
    }

    fn scalars(count: usize, seed: u64) -> Vec<Fr> {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        (0..count)
            .map(|_| <Fr as RandomSampling>::random(&mut rng))
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
            got.row_commitments.len(),
            PROVE_WIDE_ROWS,
            "the combined hint must be padded out to the widest source hint",
        );
    }

    #[test]
    fn reference_verifier_accepts_proof() {
        if shared_context().is_none() {
            return;
        }
        let mut rng = ChaCha20Rng::seed_from_u64(7);
        let prover_setup = CudaDoryScheme::setup_prover(NUM_VARS);
        let verifier_setup = CudaDoryScheme::setup_verifier(NUM_VARS);

        let poly = Polynomial::<Fr>::random(NUM_VARS, &mut rng);
        let point: Vec<Fr> = (0..NUM_VARS)
            .map(|_| <Fr as RandomSampling>::random(&mut rng))
            .collect();
        let eval = poly.evaluate(&point);

        let (commitment, hint) =
            CudaDoryScheme::commit(poly.evaluations(), &prover_setup).expect("cuda commit");
        let (expected_commitment, expected_hint) =
            DoryScheme::commit(poly.evaluations(), &prover_setup).expect("dory commit");
        assert_eq!(commitment, expected_commitment);
        assert_eq!(hint, expected_hint);
        assert_eq!(hint.row_commitments.len(), opening_rows(NUM_VARS));

        let (proof, expected) = open_pair(&poly, &point, eval, &prover_setup, hint);
        assert_eq!(proof, expected, "the proof diverged from DoryScheme's");

        let mut verify_transcript = Blake2bTranscript::new(b"dory");
        DoryScheme::verify(
            &commitment,
            &point,
            eval,
            &proof,
            &verifier_setup,
            &mut verify_transcript,
        )
        .expect("the DoryScheme verifier must accept a CudaDoryScheme proof");
    }

    #[test]
    fn open_matches_reference_dory_at_prove_scale() {
        if shared_context().is_none() {
            return;
        }
        let mut rng = ChaCha20Rng::seed_from_u64(11);
        let prover_setup = CudaDoryScheme::setup_prover(PROVE_SCALE_VARS);

        let poly = Polynomial::<Fr>::random(PROVE_SCALE_VARS, &mut rng);
        let point: Vec<Fr> = (0..PROVE_SCALE_VARS)
            .map(|_| <Fr as RandomSampling>::random(&mut rng))
            .collect();
        let eval = poly.evaluate(&point);

        let rows = opening_rows(PROVE_SCALE_VARS);
        let mut hint = shaped_hints(&[rows], 12)
            .pop()
            .expect("one shaped hint was requested");
        hint.commit_blind = Fr::from_u64(0);

        let (got, expected) = open_pair(&poly, &point, eval, &prover_setup, hint);
        assert_eq!(
            got, expected,
            "the proof diverged from DoryScheme's over {rows} rows at {PROVE_SCALE_VARS} variables",
        );
    }
}
