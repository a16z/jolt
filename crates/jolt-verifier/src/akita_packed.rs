use std::collections::BTreeSet;

use jolt_akita::{
    AkitaBatchProof, AkitaCommitment, AkitaField, AkitaHidingCommitment, AkitaProverHint,
    AkitaProverSetup, AkitaScheme, AkitaSetupParams, AkitaSparsePolynomial, AkitaVerifierSetup,
    AKITA_D,
};
use jolt_crypto::Commitment;
use jolt_openings::{
    has_packed_linear_view, packed_witness_source_polynomial, prove_sparse_packed_linear_reduction,
    validate_packed_linear_statement, BatchOpeningResult, BatchOpeningScheme,
    BatchOpeningStatement, CommitmentScheme, OpeningsError, PackedLinearBatch,
    PackedLinearBatchProof, PackedLinearProverSetup, PackedLinearSetupParams,
    PackedLinearVerifierSetup, PackedLinearWitnessSource, PackedWitnessLayout, PackedWitnessSource,
    ZkBatchOpeningScheme, ZkOpeningScheme,
};
use jolt_poly::{MultilinearPoly, Polynomial};
use jolt_transcript::{AppendToTranscript, Label, Transcript};
use serde::{Deserialize, Serialize};

type AkitaPackedAdapter = PackedLinearBatch<AkitaScheme, PackedWitnessLayout>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AkitaPackedScheme;

impl AkitaPackedScheme {
    pub fn commit_packed_source<S>(
        setup: &<Self as CommitmentScheme>::ProverSetup,
        source: &S,
    ) -> Result<(AkitaCommitment, AkitaProverHint), OpeningsError>
    where
        S: PackedWitnessSource<AkitaField>,
    {
        validate_source_fits_setup(setup.pcs.max_num_vars, source.layout())?;
        if let Some(polynomial) = packed_source_sparse_polynomial(source)? {
            return AkitaScheme::commit_sparse_polynomial(
                &setup.pcs,
                source.layout().digest,
                &polynomial,
            );
        }
        let polynomial = packed_witness_source_polynomial(source)?;
        AkitaScheme::commit_group(&setup.pcs, source.layout().digest, &[polynomial])
    }

    pub fn prove_packed_source_batch<T, OpeningId, RelationId, S>(
        setup: &<Self as CommitmentScheme>::ProverSetup,
        transcript: &mut T,
        statement: &BatchOpeningStatement<AkitaField, AkitaCommitment, OpeningId, RelationId>,
        source: &S,
        hint: AkitaProverHint,
    ) -> Result<PackedLinearBatchProof<AkitaBatchProof>, OpeningsError>
    where
        T: Transcript<Challenge = AkitaField>,
        S: PackedWitnessSource<AkitaField>,
    {
        validate_source_fits_setup(setup.pcs.max_num_vars, source.layout())?;
        if let Some(sparse_polynomial) = packed_source_sparse_polynomial(source)? {
            if !has_packed_linear_view(statement) {
                let native = AkitaScheme::prove_sparse_batch(
                    &setup.pcs,
                    transcript,
                    statement,
                    &sparse_polynomial,
                    hint,
                )?;
                return Ok(PackedLinearBatchProof {
                    reduction: None,
                    native,
                });
            }

            let shape = validate_packed_source_prover_inputs(setup, statement, source, &hint)?;
            let source = AkitaPackedSource(source);
            let reduction =
                prove_sparse_packed_linear_reduction(shape.layout, statement, &source, transcript)?;
            let native_statement = singleton_statement(
                shape.commitment.clone(),
                &reduction.opening_point,
                reduction.opening_eval,
            );
            let native = AkitaScheme::prove_sparse_batch(
                &setup.pcs,
                transcript,
                &native_statement,
                &sparse_polynomial,
                hint,
            )?;
            return Ok(PackedLinearBatchProof {
                reduction: Some(reduction.proof),
                native,
            });
        }

        let polynomial = packed_witness_source_polynomial(source)?;
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
    type ProverSetup = PackedLinearProverSetup<AkitaProverSetup, PackedWitnessLayout>;
    type VerifierSetup = PackedLinearVerifierSetup<AkitaVerifierSetup, PackedWitnessLayout>;
    type Polynomial = Polynomial<AkitaField>;
    type OpeningHint = AkitaProverHint;
    type SetupParams = PackedLinearSetupParams<AkitaSetupParams, PackedWitnessLayout>;

    fn setup(params: Self::SetupParams) -> (Self::ProverSetup, Self::VerifierSetup) {
        <AkitaPackedAdapter as CommitmentScheme>::setup(params)
    }

    fn verifier_setup(prover_setup: &Self::ProverSetup) -> Self::VerifierSetup {
        <AkitaPackedAdapter as CommitmentScheme>::verifier_setup(prover_setup)
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
        if has_packed_linear_view(statement) {
            validate_packed_adapter_prover_inputs(setup, statement, polynomials, &hints)?;
        }
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
        if has_packed_linear_view(statement) {
            validate_packed_adapter_verifier_inputs(setup, statement)?;
        }
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

fn validate_packed_adapter_prover_inputs<OpeningId, RelationId>(
    setup: &<AkitaPackedScheme as CommitmentScheme>::ProverSetup,
    statement: &BatchOpeningStatement<AkitaField, AkitaCommitment, OpeningId, RelationId>,
    polynomials: &[Polynomial<AkitaField>],
    hints: &[AkitaProverHint],
) -> Result<(), OpeningsError> {
    let shape = validate_packed_adapter_statement(&setup.pcs, &setup.layout, statement)?;
    if polynomials.len() != 1 {
        return Err(invalid_batch(format!(
            "Akita packed proof expects one packed witness polynomial, got {}",
            polynomials.len()
        )));
    }
    if polynomials[0].num_vars() != shape.commitment.num_vars {
        return Err(invalid_batch(format!(
            "Akita packed witness polynomial has {} variables but commitment has {}",
            polynomials[0].num_vars(),
            shape.commitment.num_vars
        )));
    }
    if hints.len() != 1 || !hints[0].matches_commitment(&shape.commitment) {
        return Err(invalid_batch(
            "Akita packed proof requires one hint matching the packed witness commitment",
        ));
    }
    Ok(())
}

fn validate_packed_adapter_verifier_inputs<OpeningId, RelationId>(
    setup: &<AkitaPackedScheme as CommitmentScheme>::VerifierSetup,
    statement: &BatchOpeningStatement<AkitaField, AkitaCommitment, OpeningId, RelationId>,
) -> Result<(), OpeningsError> {
    validate_packed_adapter_statement(&setup.pcs, &setup.layout, statement).map(|_| ())
}

fn validate_packed_adapter_statement<'a, Setup, OpeningId, RelationId>(
    setup: &Setup,
    layout: &'a PackedWitnessLayout,
    statement: &BatchOpeningStatement<AkitaField, AkitaCommitment, OpeningId, RelationId>,
) -> Result<PackedBatchShape<'a>, OpeningsError>
where
    Setup: AkitaPackedSetupShape,
{
    let commitment = validate_packed_linear_statement(layout, statement)?;
    validate_packed_setup_shape(
        setup.max_num_vars(),
        setup.default_layout_digest(),
        layout,
        &commitment,
    )?;
    Ok(PackedBatchShape { layout, commitment })
}

fn validate_packed_source_prover_inputs<'a, OpeningId, RelationId, S>(
    setup: &'a <AkitaPackedScheme as CommitmentScheme>::ProverSetup,
    statement: &BatchOpeningStatement<AkitaField, AkitaCommitment, OpeningId, RelationId>,
    source: &S,
    hint: &AkitaProverHint,
) -> Result<PackedBatchShape<'a>, OpeningsError>
where
    S: PackedWitnessSource<AkitaField>,
{
    let shape = validate_packed_adapter_statement(&setup.pcs, &setup.layout, statement)?;
    let source_layout = source.layout();
    if source_layout.digest != shape.layout.digest
        || source_layout.dimension != shape.layout.dimension
    {
        return Err(invalid_batch(
            "Akita packed witness source layout does not match packed statement",
        ));
    }
    if !hint.matches_commitment(&shape.commitment) {
        return Err(invalid_batch(
            "Akita packed proof requires one hint matching the packed witness commitment",
        ));
    }
    Ok(shape)
}

trait AkitaPackedSetupShape {
    fn max_num_vars(&self) -> usize;
    fn default_layout_digest(&self) -> [u8; 32];
}

impl AkitaPackedSetupShape for AkitaProverSetup {
    fn max_num_vars(&self) -> usize {
        self.max_num_vars
    }

    fn default_layout_digest(&self) -> [u8; 32] {
        self.default_layout_digest
    }
}

impl AkitaPackedSetupShape for AkitaVerifierSetup {
    fn max_num_vars(&self) -> usize {
        self.max_num_vars
    }

    fn default_layout_digest(&self) -> [u8; 32] {
        self.default_layout_digest
    }
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

fn packed_source_sparse_polynomial<S>(
    source: &S,
) -> Result<Option<AkitaSparsePolynomial>, OpeningsError>
where
    S: PackedWitnessSource<AkitaField>,
{
    let layout = source.layout();
    if layout.cells == 0 {
        return Err(invalid_batch(
            "Akita packed witness layout must contain at least one cell",
        ));
    }
    if layout.dimension >= usize::BITS as usize {
        return Err(invalid_batch(format!(
            "Akita packed witness dimension {} exceeds usize bit width",
            layout.dimension
        )));
    }
    let domain_size = 1usize << layout.dimension;
    if domain_size < AKITA_D {
        return Ok(None);
    }
    if layout.cells > domain_size {
        return Err(invalid_batch(format!(
            "Akita packed witness has {} cells but dimension {} supports {domain_size}",
            layout.cells, layout.dimension
        )));
    }

    let mut seen = BTreeSet::new();
    let mut ranks = Vec::new();
    let mut result = Ok(());
    source.for_each_nonzero(|rank, value| {
        if result.is_err() {
            return;
        }
        if rank >= layout.cells {
            result = Err(invalid_batch(format!(
                "Akita packed witness source emitted rank {rank} outside {} real cells",
                layout.cells
            )));
            return;
        }
        if !seen.insert(rank) {
            result = Err(invalid_batch(format!(
                "Akita packed witness source emitted rank {rank} more than once"
            )));
            return;
        }
        if value != AkitaField::one() {
            result = Err(invalid_batch(format!(
                "Akita sparse packed witness source emitted non-unit value at rank {rank}"
            )));
            return;
        }
        ranks.push(rank);
    });
    result?;

    AkitaSparsePolynomial::from_jolt_unit_indices(layout.dimension, ranks).map(Some)
}

fn validate_source_fits_setup(
    max_num_vars: usize,
    layout: &PackedWitnessLayout,
) -> Result<(), OpeningsError> {
    if layout.dimension > max_num_vars {
        return Err(OpeningsError::PolynomialTooLarge {
            poly_size: layout.dimension,
            setup_max: max_num_vars,
        });
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

fn append_field_slice<T>(transcript: &mut T, label: &'static [u8], values: &[AkitaField])
where
    T: Transcript<Challenge = AkitaField>,
{
    transcript.append(&jolt_transcript::LabelWithCount(label, values.len() as u64));
    for value in values {
        value.append_to_transcript(transcript);
    }
}

fn invalid_batch(reason: impl Into<String>) -> OpeningsError {
    OpeningsError::InvalidBatch(reason.into())
}

fn transparent_zk_error() -> OpeningsError {
    invalid_batch("Akita packed batch openings do not support ZK mode yet")
}
