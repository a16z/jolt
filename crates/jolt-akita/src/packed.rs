use jolt_crypto::Commitment;
use jolt_openings::{
    has_packed_linear_view, prove_sparse_packed_linear_reduction, validate_packed_linear_statement,
    BatchOpeningResult, BatchOpeningScheme, BatchOpeningStatement, CommitmentScheme, OpeningsError,
    PackedLinearBatch, PackedLinearBatchBackend, PackedLinearBatchProof, PackedLinearWitnessSource,
    PackedWitnessLayout, PackedWitnessSource, ZkBatchOpeningScheme, ZkOpeningScheme,
};
use jolt_poly::{MultilinearPoly, Polynomial};
use jolt_transcript::{AppendToTranscript, Label, Transcript};
use serde::{Deserialize, Serialize};

use crate::backend::{
    bind_verifier_setup_key, packed_source_polynomial, packed_source_sparse_polynomial,
    prove_batch_with_native_polynomials,
};
use crate::types::{
    append_field_slice, AkitaBatchProof, AkitaCommitInput, AkitaCommitment, AkitaField,
    AkitaHidingCommitment, AkitaProverHint, AkitaProverSetup, AkitaSetupParams, AkitaVerifierSetup,
};
use crate::AkitaScheme;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AkitaPackedScheme;

type AkitaPackedAdapter = PackedLinearBatch<AkitaScheme>;

impl AkitaPackedScheme {
    pub fn commit_packed_witness(
        setup: &AkitaProverSetup,
        input: AkitaCommitInput,
    ) -> Result<(AkitaCommitment, AkitaProverHint), OpeningsError> {
        AkitaScheme::commit_packed_witness(setup, input)
    }

    pub fn commit_packed_source<S>(
        setup: &AkitaProverSetup,
        source: &S,
    ) -> Result<(AkitaCommitment, AkitaProverHint), OpeningsError>
    where
        S: PackedWitnessSource<AkitaField>,
    {
        AkitaScheme::commit_packed_source(setup, source)
    }

    pub fn prove_packed_source_batch<T, OpeningId, RelationId, S>(
        setup: &AkitaProverSetup,
        transcript: &mut T,
        statement: &BatchOpeningStatement<AkitaField, AkitaCommitment, OpeningId, RelationId>,
        source: &S,
        hint: AkitaProverHint,
    ) -> Result<PackedLinearBatchProof<AkitaBatchProof>, OpeningsError>
    where
        T: Transcript<Challenge = AkitaField>,
        S: PackedWitnessSource<AkitaField>,
    {
        if let Some(sparse_polynomial) = packed_source_sparse_polynomial(source)? {
            if !has_packed_linear_view(statement) {
                let native = prove_batch_with_native_polynomials(
                    setup,
                    transcript,
                    statement,
                    &[&sparse_polynomial],
                    vec![hint],
                )?;
                return Ok(PackedLinearBatchProof {
                    reduction: None,
                    native,
                });
            }

            let shape = validate_packed_source_prover_inputs(setup, statement, source, &hint)?;
            <AkitaScheme as PackedLinearBatchBackend>::bind_packed_prover_setup(setup, transcript);
            let source = AkitaPackedSource(source);
            let reduction =
                prove_sparse_packed_linear_reduction(shape.layout, statement, &source, transcript)?;
            let native_statement = singleton_statement(
                shape.commitment.clone(),
                &reduction.opening_point,
                reduction.opening_eval,
            );
            let native = prove_batch_with_native_polynomials(
                setup,
                transcript,
                &native_statement,
                &[&sparse_polynomial],
                vec![hint],
            )?;
            return Ok(PackedLinearBatchProof {
                reduction: Some(reduction.proof),
                native,
            });
        }

        let polynomial = packed_source_polynomial(source)?;
        <Self as BatchOpeningScheme>::prove_batch(
            setup,
            transcript,
            statement,
            std::slice::from_ref(&polynomial),
            vec![hint],
        )
    }
}

impl Commitment for AkitaPackedScheme {
    type Output = AkitaCommitment;
}

impl CommitmentScheme for AkitaPackedScheme {
    type Field = AkitaField;
    type Proof = PackedLinearBatchProof<AkitaBatchProof>;
    type ProverSetup = AkitaProverSetup;
    type VerifierSetup = AkitaVerifierSetup;
    type Polynomial = Polynomial<AkitaField>;
    type OpeningHint = AkitaProverHint;
    type SetupParams = AkitaSetupParams;

    fn setup(params: Self::SetupParams) -> (Self::ProverSetup, Self::VerifierSetup) {
        AkitaScheme::setup(params)
    }

    fn verifier_setup(prover_setup: &Self::ProverSetup) -> Self::VerifierSetup {
        AkitaScheme::verifier_setup(prover_setup)
    }

    fn commit<P: MultilinearPoly<Self::Field> + ?Sized>(
        _poly: &P,
        _setup: &Self::ProverSetup,
    ) -> (Self::Output, Self::OpeningHint) {
        unsupported_dense_packed_path("commit")
    }

    fn open(
        _poly: &Self::Polynomial,
        _point: &[Self::Field],
        _eval: Self::Field,
        _setup: &Self::ProverSetup,
        _hint: Option<Self::OpeningHint>,
        _transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Self::Proof {
        unsupported_dense_packed_path("open")
    }

    fn verify(
        commitment: &Self::Output,
        point: &[Self::Field],
        eval: Self::Field,
        proof: &Self::Proof,
        setup: &Self::VerifierSetup,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<(), OpeningsError> {
        <AkitaPackedAdapter as CommitmentScheme>::verify(
            commitment, point, eval, proof, setup, transcript,
        )
    }

    fn bind_opening_inputs(
        transcript: &mut impl Transcript<Challenge = Self::Field>,
        point: &[Self::Field],
        eval: &Self::Field,
    ) {
        AkitaScheme::bind_opening_inputs(transcript, point, eval);
    }
}

impl BatchOpeningScheme for AkitaPackedScheme {
    fn prove_batch<T, OpeningId, RelationId>(
        setup: &Self::ProverSetup,
        transcript: &mut T,
        statement: &BatchOpeningStatement<Self::Field, Self::Output, OpeningId, RelationId>,
        polynomials: &[Self::Polynomial],
        hints: Vec<Self::OpeningHint>,
    ) -> Result<Self::Proof, OpeningsError>
    where
        T: Transcript<Challenge = Self::Field>,
    {
        <AkitaPackedAdapter as BatchOpeningScheme>::prove_batch(
            setup,
            transcript,
            statement,
            polynomials,
            hints,
        )
    }

    fn verify_batch<T, OpeningId, RelationId>(
        setup: &Self::VerifierSetup,
        transcript: &mut T,
        statement: &BatchOpeningStatement<Self::Field, Self::Output, OpeningId, RelationId>,
        proof: &Self::Proof,
    ) -> Result<BatchOpeningResult<Self::Field, Self::Output>, OpeningsError>
    where
        T: Transcript<Challenge = Self::Field>,
    {
        <AkitaPackedAdapter as BatchOpeningScheme>::verify_batch(
            setup, transcript, statement, proof,
        )
    }
}

impl ZkOpeningScheme for AkitaPackedScheme {
    type HidingCommitment = AkitaHidingCommitment;
    type Blind = ();

    fn commit_zk<P: MultilinearPoly<Self::Field> + ?Sized>(
        _poly: &P,
        _setup: &Self::ProverSetup,
    ) -> (Self::Output, Self::OpeningHint) {
        unsupported_dense_packed_path("commit_zk")
    }

    fn open_zk(
        _poly: &Self::Polynomial,
        _point: &[Self::Field],
        _eval: Self::Field,
        _setup: &Self::ProverSetup,
        _hint: Self::OpeningHint,
        _transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> (Self::Proof, Self::HidingCommitment, Self::Blind) {
        unsupported_dense_packed_path("open_zk")
    }

    fn verify_zk(
        _commitment: &Self::Output,
        _point: &[Self::Field],
        _proof: &Self::Proof,
        _setup: &Self::VerifierSetup,
        _transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<Self::HidingCommitment, OpeningsError> {
        Err(transparent_zk_error())
    }

    fn bind_zk_opening_inputs(
        transcript: &mut impl Transcript<Challenge = Self::Field>,
        point: &[Self::Field],
        hiding_commitment: &Self::HidingCommitment,
    ) {
        transcript.append(&Label(b"akpk_zk_inputs"));
        append_field_slice(transcript, b"akpk_zk_point", point);
        hiding_commitment.append_to_transcript(transcript);
    }
}

impl ZkBatchOpeningScheme for AkitaPackedScheme {
    fn prove_batch_zk<T, OpeningId, RelationId>(
        _setup: &Self::ProverSetup,
        _transcript: &mut T,
        _statement: &BatchOpeningStatement<Self::Field, Self::Output, OpeningId, RelationId, ()>,
        _evals: &[Self::Field],
        _polynomials: &[Self::Polynomial],
        _hints: Vec<Self::OpeningHint>,
    ) -> Result<(Self::Proof, Self::HidingCommitment, Self::Blind), OpeningsError>
    where
        T: Transcript<Challenge = Self::Field>,
    {
        Err(transparent_zk_error())
    }

    fn verify_batch_zk<T, OpeningId, RelationId>(
        _setup: &Self::VerifierSetup,
        _transcript: &mut T,
        _statement: &BatchOpeningStatement<Self::Field, Self::Output, OpeningId, RelationId, ()>,
        _proof: &Self::Proof,
    ) -> Result<BatchOpeningResult<Self::Field, Self::Output, Self::HidingCommitment>, OpeningsError>
    where
        T: Transcript<Challenge = Self::Field>,
    {
        Err(transparent_zk_error())
    }
}

impl PackedLinearBatchBackend for AkitaScheme {
    type Layout = PackedWitnessLayout;

    fn prover_layout(setup: &Self::ProverSetup) -> Option<&Self::Layout> {
        setup.packed_layout.as_ref()
    }

    fn verifier_layout(setup: &Self::VerifierSetup) -> Option<&Self::Layout> {
        setup.packed_layout.as_ref()
    }

    fn validate_packed_prover_inputs(
        setup: &Self::ProverSetup,
        layout: &Self::Layout,
        commitment: &Self::Output,
        polynomials: &[Self::Polynomial],
        hints: &[Self::OpeningHint],
    ) -> Result<(), OpeningsError> {
        validate_packed_setup_shape(
            setup.max_num_vars,
            setup.default_layout_digest,
            layout,
            commitment,
        )?;
        if polynomials.len() != 1 {
            return Err(invalid_batch(format!(
                "Akita packed proof expects one packed witness polynomial, got {}",
                polynomials.len()
            )));
        }
        if polynomials[0].num_vars() != commitment.num_vars {
            return Err(invalid_batch(format!(
                "Akita packed witness polynomial has {} variables but commitment has {}",
                polynomials[0].num_vars(),
                commitment.num_vars
            )));
        }
        if hints.len() != 1 || !hints[0].matches_commitment(commitment) {
            return Err(invalid_batch(
                "Akita packed proof requires one hint matching the packed witness commitment",
            ));
        }
        Ok(())
    }

    fn validate_packed_verifier_inputs(
        setup: &Self::VerifierSetup,
        layout: &Self::Layout,
        commitment: &Self::Output,
    ) -> Result<(), OpeningsError> {
        validate_packed_setup_shape(
            setup.max_num_vars,
            setup.default_layout_digest,
            layout,
            commitment,
        )
    }

    fn bind_packed_prover_setup<T>(setup: &Self::ProverSetup, transcript: &mut T)
    where
        T: Transcript<Challenge = Self::Field>,
    {
        bind_verifier_setup_key(&setup.verifier, transcript);
    }

    fn bind_packed_verifier_setup<T>(setup: &Self::VerifierSetup, transcript: &mut T)
    where
        T: Transcript<Challenge = Self::Field>,
    {
        bind_verifier_setup_key(setup, transcript);
    }
}

struct AkitaPackedSource<'a, S>(&'a S);

impl<S> PackedLinearWitnessSource<AkitaField> for AkitaPackedSource<'_, S>
where
    S: PackedWitnessSource<AkitaField>,
{
    type Layout = PackedWitnessLayout;

    fn layout(&self) -> &Self::Layout {
        self.0.layout()
    }

    fn for_each_nonzero(&self, f: impl FnMut(usize, AkitaField)) {
        self.0.for_each_nonzero(f);
    }
}

struct PackedBatchShape<'a> {
    layout: &'a PackedWitnessLayout,
    commitment: AkitaCommitment,
}

fn validate_packed_source_prover_inputs<'a, OpeningId, RelationId, S>(
    setup: &'a AkitaProverSetup,
    statement: &BatchOpeningStatement<AkitaField, AkitaCommitment, OpeningId, RelationId>,
    source: &S,
    hint: &AkitaProverHint,
) -> Result<PackedBatchShape<'a>, OpeningsError>
where
    S: PackedWitnessSource<AkitaField>,
{
    let layout = setup
        .packed_layout
        .as_ref()
        .ok_or_else(|| invalid_batch("Akita packed opening requires setup layout"))?;
    let commitment = validate_packed_linear_statement(layout, statement)?;
    validate_packed_setup_shape(
        setup.max_num_vars,
        setup.default_layout_digest,
        layout,
        &commitment,
    )?;
    let source_layout = source.layout();
    if source_layout.digest != layout.digest || source_layout.dimension != layout.dimension {
        return Err(invalid_batch(
            "Akita packed witness source layout does not match packed statement",
        ));
    }
    if !hint.matches_commitment(&commitment) {
        return Err(invalid_batch(
            "Akita packed proof requires one hint matching the packed witness commitment",
        ));
    }
    Ok(PackedBatchShape { layout, commitment })
}

fn validate_packed_setup_shape(
    max_num_vars: usize,
    default_layout_digest: [u8; 32],
    layout: &PackedWitnessLayout,
    commitment: &AkitaCommitment,
) -> Result<(), OpeningsError> {
    if commitment.num_vars > max_num_vars {
        return Err(OpeningsError::PolynomialTooLarge {
            poly_size: commitment.num_vars,
            setup_max: max_num_vars,
        });
    }
    if commitment.num_vars != max_num_vars {
        return Err(invalid_batch(format!(
            "Akita packed commitment dimension {} does not match exact setup dimension {}",
            commitment.num_vars, max_num_vars
        )));
    }
    if commitment.layout_digest != default_layout_digest {
        return Err(invalid_batch(
            "Akita packed commitment layout digest does not match setup",
        ));
    }
    if commitment.layout_digest != layout.digest {
        return Err(invalid_batch(
            "Akita packed commitment layout digest does not match setup layout",
        ));
    }
    if commitment.num_vars != layout.dimension {
        return Err(invalid_batch(format!(
            "Akita packed commitment dimension {} does not match layout dimension {}",
            commitment.num_vars, layout.dimension
        )));
    }
    if commitment.poly_count != 1 {
        return Err(invalid_batch(format!(
            "Akita packed witness commitment must contain one polynomial, got {}",
            commitment.poly_count
        )));
    }
    Ok(())
}

fn singleton_statement(
    commitment: AkitaCommitment,
    point: &[AkitaField],
    eval: AkitaField,
) -> BatchOpeningStatement<AkitaField, AkitaCommitment> {
    BatchOpeningStatement {
        logical_point: point.to_vec(),
        pcs_point: point.to_vec(),
        layout_digest: commitment.layout_digest,
        claims: vec![jolt_openings::BatchOpeningClaim {
            id: (),
            relation: (),
            commitment,
            claim: eval,
            view: jolt_openings::PhysicalView::Direct,
            scale: AkitaField::one(),
        }],
    }
}

#[expect(
    clippy::panic,
    reason = "single-opening trait methods cannot return unsupported packed dense-path errors"
)]
fn unsupported_dense_packed_path(operation: &str) -> ! {
    panic!(
        "AkitaPackedScheme::{operation} cannot be used for dense proof-owned polynomials; use commit_packed_source/prove_packed_source_batch for W_pack or AkitaScheme direct openings for precommitted objects"
    )
}

fn invalid_batch(reason: impl Into<String>) -> OpeningsError {
    OpeningsError::InvalidBatch(reason.into())
}

fn transparent_zk_error() -> OpeningsError {
    invalid_batch("Akita packed batch openings do not support ZK mode yet")
}
