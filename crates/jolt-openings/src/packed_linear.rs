use std::marker::PhantomData;

use jolt_crypto::Commitment;
use jolt_field::Field;
use jolt_poly::{EqPolynomial, MultilinearPoly};
use jolt_transcript::{AppendToTranscript, Label, LabelWithCount, Transcript, U64Word};
use serde::{Deserialize, Serialize};

use crate::{
    BatchOpeningResult, BatchOpeningScheme, BatchOpeningStatement, CommitmentScheme, OpeningsError,
    PackedFamilyRef, PackedLinearTerm, PhysicalView, ZkBatchOpeningScheme, ZkOpeningScheme,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackedLinearBatch<PCS>(PhantomData<PCS>);

impl<PCS> PackedLinearBatch<PCS> {
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

impl<PCS> Default for PackedLinearBatch<PCS> {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PackedLinearReductionProof {
    pub rounds: Vec<[Vec<u8>; 3]>,
    pub opening_eval: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PackedLinearBatchProof<NativeProof> {
    pub reduction: Option<PackedLinearReductionProof>,
    pub native: NativeProof,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PackedLinearFamily {
    pub id: PackedFamilyRef,
    pub offset: usize,
    pub rows: usize,
    pub limbs: usize,
    pub alphabet_size: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PackedLinearAddress {
    pub family: PackedFamilyRef,
    pub row: usize,
    pub limb: usize,
    pub symbol: usize,
}

pub trait PackedLinearLayout {
    fn digest(&self) -> [u8; 32];
    fn dimension(&self) -> usize;
    fn cells(&self) -> usize;
    fn family(&self, family: PackedFamilyRef) -> Result<Option<PackedLinearFamily>, OpeningsError>;
    fn rank(&self, address: PackedLinearAddress) -> Result<usize, OpeningsError>;
}

pub trait PackedLinearWitnessSource<F>
where
    F: Field,
{
    type Layout: PackedLinearLayout;

    fn layout(&self) -> &Self::Layout;

    fn for_each_nonzero(&self, f: impl FnMut(usize, F));
}

pub trait PackedLinearBatchBackend: BatchOpeningScheme {
    type Layout: PackedLinearLayout;

    fn prover_layout(setup: &Self::ProverSetup) -> Option<&Self::Layout>;

    fn verifier_layout(setup: &Self::VerifierSetup) -> Option<&Self::Layout>;

    fn validate_packed_prover_inputs(
        _setup: &Self::ProverSetup,
        layout: &Self::Layout,
        _commitment: &Self::Output,
        polynomials: &[Self::Polynomial],
        hints: &[Self::OpeningHint],
    ) -> Result<(), OpeningsError> {
        if polynomials.len() != 1 {
            return Err(invalid_batch(format!(
                "packed linear proof expects one packed polynomial, got {}",
                polynomials.len()
            )));
        }
        if polynomials[0].num_vars() != layout.dimension() {
            return Err(invalid_batch(format!(
                "packed linear polynomial has {} variables but layout has {}",
                polynomials[0].num_vars(),
                layout.dimension()
            )));
        }
        if hints.len() != 1 {
            return Err(invalid_batch(format!(
                "packed linear proof expects one opening hint, got {}",
                hints.len()
            )));
        }
        Ok(())
    }

    fn validate_packed_verifier_inputs(
        _setup: &Self::VerifierSetup,
        _layout: &Self::Layout,
        _commitment: &Self::Output,
    ) -> Result<(), OpeningsError> {
        Ok(())
    }

    fn bind_packed_prover_setup<T>(_setup: &Self::ProverSetup, _transcript: &mut T)
    where
        T: Transcript<Challenge = Self::Field>,
    {
    }

    fn bind_packed_verifier_setup<T>(_setup: &Self::VerifierSetup, _transcript: &mut T)
    where
        T: Transcript<Challenge = Self::Field>,
    {
    }
}

pub struct PackedLinearProverReduction<F> {
    pub proof: PackedLinearReductionProof,
    pub opening_point: Vec<F>,
    pub opening_eval: F,
}

pub struct PackedLinearVerifierReduction<F, C> {
    pub opening_point: Vec<F>,
    pub opening_eval: F,
    pub result: BatchOpeningResult<F, C>,
}

impl<PCS> Commitment for PackedLinearBatch<PCS>
where
    PCS: CommitmentScheme,
{
    type Output = PCS::Output;
}

impl<PCS> CommitmentScheme for PackedLinearBatch<PCS>
where
    PCS: CommitmentScheme,
{
    type Field = PCS::Field;
    type Proof = PackedLinearBatchProof<PCS::Proof>;
    type ProverSetup = PCS::ProverSetup;
    type VerifierSetup = PCS::VerifierSetup;
    type Polynomial = PCS::Polynomial;
    type OpeningHint = PCS::OpeningHint;
    type SetupParams = PCS::SetupParams;

    fn setup(params: Self::SetupParams) -> (Self::ProverSetup, Self::VerifierSetup) {
        PCS::setup(params)
    }

    fn verifier_setup(prover_setup: &Self::ProverSetup) -> Self::VerifierSetup {
        PCS::verifier_setup(prover_setup)
    }

    fn commit<P: MultilinearPoly<Self::Field> + ?Sized>(
        poly: &P,
        setup: &Self::ProverSetup,
    ) -> (Self::Output, Self::OpeningHint) {
        PCS::commit(poly, setup)
    }

    fn open(
        poly: &Self::Polynomial,
        point: &[Self::Field],
        eval: Self::Field,
        setup: &Self::ProverSetup,
        hint: Option<Self::OpeningHint>,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Self::Proof {
        PackedLinearBatchProof {
            reduction: None,
            native: PCS::open(poly, point, eval, setup, hint, transcript),
        }
    }

    fn verify(
        commitment: &Self::Output,
        point: &[Self::Field],
        eval: Self::Field,
        proof: &Self::Proof,
        setup: &Self::VerifierSetup,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<(), OpeningsError> {
        if proof.reduction.is_some() {
            return Err(OpeningsError::VerificationFailed);
        }
        PCS::verify(commitment, point, eval, &proof.native, setup, transcript)
    }

    fn bind_opening_inputs(
        transcript: &mut impl Transcript<Challenge = Self::Field>,
        point: &[Self::Field],
        eval: &Self::Field,
    ) {
        PCS::bind_opening_inputs(transcript, point, eval);
    }
}

impl<PCS> BatchOpeningScheme for PackedLinearBatch<PCS>
where
    PCS: PackedLinearBatchBackend,
    PCS::Output: AppendToTranscript,
{
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
        if !has_packed_linear_view(statement) {
            let native = PCS::prove_batch(setup, transcript, statement, polynomials, hints)?;
            return Ok(PackedLinearBatchProof {
                reduction: None,
                native,
            });
        }

        let layout = PCS::prover_layout(setup)
            .ok_or_else(|| invalid_batch("packed linear opening requires setup layout"))?;
        let commitment = validate_packed_linear_statement(layout, statement)?;
        PCS::validate_packed_prover_inputs(setup, layout, &commitment, polynomials, &hints)?;
        let hint = hints
            .into_iter()
            .next()
            .ok_or_else(|| invalid_batch("packed linear proof requires one opening hint"))?;
        PCS::bind_packed_prover_setup(setup, transcript);
        let reduction = prove_packed_linear_reduction(
            layout,
            statement,
            polynomial_evaluations(&polynomials[0]),
            transcript,
        )?;
        let native = PCS::open(
            &polynomials[0],
            &reduction.opening_point,
            reduction.opening_eval,
            setup,
            Some(hint),
            transcript,
        );
        Ok(PackedLinearBatchProof {
            reduction: Some(reduction.proof),
            native,
        })
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
        if !has_packed_linear_view(statement) {
            if proof.reduction.is_some() {
                return Err(OpeningsError::VerificationFailed);
            }
            return PCS::verify_batch(setup, transcript, statement, &proof.native);
        }

        let reduction_proof = proof
            .reduction
            .as_ref()
            .ok_or(OpeningsError::VerificationFailed)?;
        let layout = PCS::verifier_layout(setup)
            .ok_or_else(|| invalid_batch("packed linear opening requires setup layout"))?;
        let commitment = validate_packed_linear_statement(layout, statement)?;
        PCS::validate_packed_verifier_inputs(setup, layout, &commitment)?;
        PCS::bind_packed_verifier_setup(setup, transcript);
        let reduction =
            verify_packed_linear_reduction(layout, statement, reduction_proof, transcript)?;
        PCS::verify(
            &reduction.result.joint_commitment,
            &reduction.opening_point,
            reduction.opening_eval,
            &proof.native,
            setup,
            transcript,
        )?;
        Ok(reduction.result)
    }
}

impl<PCS> ZkOpeningScheme for PackedLinearBatch<PCS>
where
    PCS: ZkOpeningScheme,
{
    type HidingCommitment = PCS::HidingCommitment;
    type Blind = PCS::Blind;

    fn commit_zk<P: MultilinearPoly<Self::Field> + ?Sized>(
        poly: &P,
        setup: &Self::ProverSetup,
    ) -> (Self::Output, Self::OpeningHint) {
        PCS::commit_zk(poly, setup)
    }

    fn open_zk(
        poly: &Self::Polynomial,
        point: &[Self::Field],
        eval: Self::Field,
        setup: &Self::ProverSetup,
        hint: Self::OpeningHint,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> (Self::Proof, Self::HidingCommitment, Self::Blind) {
        let (native, hiding, blind) = PCS::open_zk(poly, point, eval, setup, hint, transcript);
        (
            PackedLinearBatchProof {
                reduction: None,
                native,
            },
            hiding,
            blind,
        )
    }

    fn verify_zk(
        commitment: &Self::Output,
        point: &[Self::Field],
        proof: &Self::Proof,
        setup: &Self::VerifierSetup,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<Self::HidingCommitment, OpeningsError> {
        if proof.reduction.is_some() {
            return Err(OpeningsError::VerificationFailed);
        }
        PCS::verify_zk(commitment, point, &proof.native, setup, transcript)
    }

    fn bind_zk_opening_inputs(
        transcript: &mut impl Transcript<Challenge = Self::Field>,
        point: &[Self::Field],
        hiding_commitment: &Self::HidingCommitment,
    ) {
        PCS::bind_zk_opening_inputs(transcript, point, hiding_commitment);
    }
}

impl<PCS> ZkBatchOpeningScheme for PackedLinearBatch<PCS>
where
    PCS: PackedLinearBatchBackend + ZkBatchOpeningScheme,
    PCS::Output: AppendToTranscript,
{
    fn prove_batch_zk<T, OpeningId, RelationId>(
        setup: &Self::ProverSetup,
        transcript: &mut T,
        statement: &BatchOpeningStatement<Self::Field, Self::Output, OpeningId, RelationId, ()>,
        evals: &[Self::Field],
        polynomials: &[Self::Polynomial],
        hints: Vec<Self::OpeningHint>,
    ) -> Result<(Self::Proof, Self::HidingCommitment, Self::Blind), OpeningsError>
    where
        T: Transcript<Challenge = Self::Field>,
    {
        if has_packed_linear_view(statement) {
            return Err(invalid_batch(
                "packed linear batch openings do not support ZK mode yet",
            ));
        }
        let (native, hiding, blind) =
            PCS::prove_batch_zk(setup, transcript, statement, evals, polynomials, hints)?;
        Ok((
            PackedLinearBatchProof {
                reduction: None,
                native,
            },
            hiding,
            blind,
        ))
    }

    fn verify_batch_zk<T, OpeningId, RelationId>(
        setup: &Self::VerifierSetup,
        transcript: &mut T,
        statement: &BatchOpeningStatement<Self::Field, Self::Output, OpeningId, RelationId, ()>,
        proof: &Self::Proof,
    ) -> Result<BatchOpeningResult<Self::Field, Self::Output, Self::HidingCommitment>, OpeningsError>
    where
        T: Transcript<Challenge = Self::Field>,
    {
        if has_packed_linear_view(statement) {
            return Err(invalid_batch(
                "packed linear batch openings do not support ZK mode yet",
            ));
        }
        if proof.reduction.is_some() {
            return Err(OpeningsError::VerificationFailed);
        }
        PCS::verify_batch_zk(setup, transcript, statement, &proof.native)
    }
}

pub fn has_packed_linear_view<F, C, OpeningId, RelationId, Claim>(
    statement: &BatchOpeningStatement<F, C, OpeningId, RelationId, Claim>,
) -> bool {
    statement
        .claims
        .iter()
        .any(|claim| matches!(claim.view, PhysicalView::PackedLinear { .. }))
}

pub fn validate_packed_linear_statement<F, C, OpeningId, RelationId, L>(
    layout: &L,
    statement: &BatchOpeningStatement<F, C, OpeningId, RelationId>,
) -> Result<C, OpeningsError>
where
    F: Field,
    C: Clone + Eq,
    L: PackedLinearLayout,
{
    let digest = layout.digest();
    if statement.claims.is_empty() {
        return Err(invalid_batch(
            "packed linear opening requires at least one claim",
        ));
    }
    if statement.layout_digest != digest {
        return Err(invalid_batch(
            "packed linear statement layout digest does not match setup layout",
        ));
    }
    let commitment = statement.claims[0].commitment.clone();
    for claim in &statement.claims {
        if claim.commitment != commitment {
            return Err(invalid_batch(
                "packed linear opening claims must use one packed commitment",
            ));
        }
        let PhysicalView::PackedLinear {
            layout_digest,
            terms,
        } = &claim.view
        else {
            return Err(invalid_batch(
                "packed linear opening requires PackedLinear physical views",
            ));
        };
        if layout_digest != &digest {
            return Err(invalid_batch(
                "packed linear view layout digest does not match statement layout",
            ));
        }
        if terms.is_empty() {
            return Err(invalid_batch(
                "packed linear view requires at least one term",
            ));
        }
        for term in terms {
            validate_term(layout, term)?;
        }
    }
    Ok(commitment)
}

pub fn prove_packed_linear_reduction<F, C, OpeningId, RelationId, L, T>(
    layout: &L,
    statement: &BatchOpeningStatement<F, C, OpeningId, RelationId>,
    packed_evals: Vec<F>,
    transcript: &mut T,
) -> Result<PackedLinearProverReduction<F>, OpeningsError>
where
    F: Field,
    C: Clone + Eq + AppendToTranscript,
    L: PackedLinearLayout,
    T: Transcript<Challenge = F>,
{
    let _ = validate_packed_linear_statement(layout, statement)?;
    bind_packed_statement(layout, statement, transcript)?;
    let gamma_powers = transcript.challenge_scalar_powers(statement.claims.len());
    let claimed_sum = reduced_claim(statement, &gamma_powers);
    let selector = packed_selector_evals(layout, statement, &gamma_powers)?;
    let (proof, sumcheck_point_lsb, opening_eval) =
        prove_product_sumcheck(selector, packed_evals, claimed_sum, transcript)?;
    Ok(PackedLinearProverReduction {
        proof,
        opening_point: native_opening_point(&sumcheck_point_lsb),
        opening_eval,
    })
}

pub fn prove_sparse_packed_linear_reduction<F, C, OpeningId, RelationId, L, S, T>(
    layout: &L,
    statement: &BatchOpeningStatement<F, C, OpeningId, RelationId>,
    source: &S,
    transcript: &mut T,
) -> Result<PackedLinearProverReduction<F>, OpeningsError>
where
    F: Field,
    C: Clone + Eq + AppendToTranscript,
    L: PackedLinearLayout,
    S: PackedLinearWitnessSource<F>,
    T: Transcript<Challenge = F>,
{
    let _ = validate_packed_linear_statement(layout, statement)?;
    validate_source_layout(layout, source.layout())?;
    bind_packed_statement(layout, statement, transcript)?;
    let gamma_powers = transcript.challenge_scalar_powers(statement.claims.len());
    let claimed_sum = reduced_claim(statement, &gamma_powers);
    let (proof, sumcheck_point_lsb, opening_eval) = prove_sparse_product_sumcheck(
        layout,
        statement,
        &gamma_powers,
        source,
        claimed_sum,
        transcript,
    )?;
    Ok(PackedLinearProverReduction {
        proof,
        opening_point: native_opening_point(&sumcheck_point_lsb),
        opening_eval,
    })
}

pub fn verify_packed_linear_reduction<F, C, OpeningId, RelationId, L, T>(
    layout: &L,
    statement: &BatchOpeningStatement<F, C, OpeningId, RelationId>,
    proof: &PackedLinearReductionProof,
    transcript: &mut T,
) -> Result<PackedLinearVerifierReduction<F, C>, OpeningsError>
where
    F: Field,
    C: Clone + Eq + AppendToTranscript,
    L: PackedLinearLayout,
    T: Transcript<Challenge = F>,
{
    let commitment = validate_packed_linear_statement(layout, statement)?;
    bind_packed_statement(layout, statement, transcript)?;
    let gamma_powers = transcript.challenge_scalar_powers(statement.claims.len());
    let coefficients = logical_coefficients(statement, &gamma_powers);
    let claimed_sum = reduced_claim(statement, &gamma_powers);
    let (sumcheck_point_lsb, final_claim) =
        verify_product_sumcheck::<F, _>(proof, claimed_sum, transcript)?;
    let selector_eval =
        packed_selector_eval(layout, statement, &gamma_powers, &sumcheck_point_lsb)?;
    let opening_eval = field_from_bytes::<F>(&proof.opening_eval)?;
    if final_claim != selector_eval * opening_eval {
        return Err(OpeningsError::VerificationFailed);
    }
    Ok(PackedLinearVerifierReduction {
        opening_point: native_opening_point(&sumcheck_point_lsb),
        opening_eval,
        result: BatchOpeningResult {
            coefficients,
            joint_commitment: commitment,
            reduced_opening: claimed_sum,
        },
    })
}

fn validate_source_layout<L, S>(layout: &L, source_layout: &S) -> Result<(), OpeningsError>
where
    L: PackedLinearLayout,
    S: PackedLinearLayout,
{
    if source_layout.digest() != layout.digest() || source_layout.dimension() != layout.dimension()
    {
        return Err(invalid_batch(
            "packed linear source layout does not match packed statement",
        ));
    }
    Ok(())
}

fn family_for_term<F, L>(
    layout: &L,
    term: &PackedLinearTerm<F>,
) -> Result<PackedLinearFamily, OpeningsError>
where
    L: PackedLinearLayout,
{
    layout
        .family(term.family)?
        .ok_or_else(|| invalid_batch("packed linear term references an unknown family"))
}

fn validate_term<F, L>(layout: &L, term: &PackedLinearTerm<F>) -> Result<(), OpeningsError>
where
    F: Field,
    L: PackedLinearLayout,
{
    let family = family_for_term(layout, term)?;
    let row_vars = log2_power_of_two(family.rows, "packed family rows")?;
    if term.row_point.len() != row_vars {
        return Err(invalid_batch(format!(
            "packed linear term row point has {} variables but family requires {row_vars}",
            term.row_point.len()
        )));
    }
    if !family.alphabet_size.is_power_of_two() {
        return Err(invalid_batch(
            "packed linear verifier currently requires power-of-two alphabets",
        ));
    }
    if !family.limbs.is_power_of_two() {
        return Err(invalid_batch(
            "packed linear verifier currently requires power-of-two limb counts",
        ));
    }
    layout
        .rank(PackedLinearAddress {
            family: term.family,
            row: 0,
            limb: term.limb,
            symbol: term.symbol,
        })
        .map(|_| ())
}

fn logical_coefficients<F, C, OpeningId, RelationId>(
    statement: &BatchOpeningStatement<F, C, OpeningId, RelationId>,
    gamma_powers: &[F],
) -> Vec<F>
where
    F: Field,
{
    statement
        .claims
        .iter()
        .zip(gamma_powers)
        .map(|(claim, gamma)| *gamma * claim.scale)
        .collect()
}

fn reduced_claim<F, C, OpeningId, RelationId>(
    statement: &BatchOpeningStatement<F, C, OpeningId, RelationId>,
    gamma_powers: &[F],
) -> F
where
    F: Field,
{
    statement
        .claims
        .iter()
        .zip(gamma_powers)
        .fold(F::zero(), |acc, (claim, gamma)| {
            acc + *gamma * claim.scale * claim.claim
        })
}

fn packed_selector_evals<F, C, OpeningId, RelationId, L>(
    layout: &L,
    statement: &BatchOpeningStatement<F, C, OpeningId, RelationId>,
    gamma_powers: &[F],
) -> Result<Vec<F>, OpeningsError>
where
    F: Field,
    L: PackedLinearLayout,
{
    let domain_size = checked_domain_size(layout.dimension())?;
    let mut selector = vec![F::zero(); domain_size];
    for (claim, gamma) in statement.claims.iter().zip(gamma_powers) {
        let PhysicalView::PackedLinear { terms, .. } = &claim.view else {
            return Err(invalid_batch(
                "packed linear selector requires PackedLinear views",
            ));
        };
        let claim_weight = *gamma * claim.scale;
        for term in terms {
            let family = family_for_term(layout, term)?;
            let row_weights = EqPolynomial::new(term.row_point.clone()).evaluations();
            if row_weights.len() != family.rows {
                return Err(invalid_batch(
                    "packed linear term row point does not match family row count",
                ));
            }
            let weight = claim_weight * term.coefficient;
            for (row, row_weight) in row_weights.iter().copied().enumerate() {
                if row_weight.is_zero() {
                    continue;
                }
                let rank = layout.rank(PackedLinearAddress {
                    family: term.family,
                    row,
                    limb: term.limb,
                    symbol: term.symbol,
                })?;
                selector[rank] += weight * row_weight;
            }
        }
    }
    Ok(selector)
}

fn packed_selector_eval<F, C, OpeningId, RelationId, L>(
    layout: &L,
    statement: &BatchOpeningStatement<F, C, OpeningId, RelationId>,
    gamma_powers: &[F],
    point: &[F],
) -> Result<F, OpeningsError>
where
    F: Field,
    L: PackedLinearLayout,
{
    if point.len() != layout.dimension() {
        return Err(invalid_batch(format!(
            "packed linear selector point has {} variables but layout has {}",
            point.len(),
            layout.dimension()
        )));
    }
    let mut result = F::zero();
    for (claim, gamma) in statement.claims.iter().zip(gamma_powers) {
        let PhysicalView::PackedLinear { terms, .. } = &claim.view else {
            return Err(invalid_batch(
                "packed linear selector requires PackedLinear views",
            ));
        };
        let claim_weight = *gamma * claim.scale;
        for term in terms {
            result += packed_term_selector_eval(layout, term, point)? * claim_weight;
        }
    }
    Ok(result)
}

fn packed_term_selector_eval<F, L>(
    layout: &L,
    term: &PackedLinearTerm<F>,
    point: &[F],
) -> Result<F, OpeningsError>
where
    F: Field,
    L: PackedLinearLayout,
{
    let family = family_for_term(layout, term)?;
    let row_vars = log2_power_of_two(family.rows, "packed family rows")?;
    if term.row_point.len() != row_vars {
        return Err(invalid_batch(format!(
            "packed linear term row point has {} variables but family requires {row_vars}",
            term.row_point.len()
        )));
    }
    let alphabet_vars = log2_power_of_two(family.alphabet_size, "packed alphabet")?;
    let limb_vars = log2_power_of_two(family.limbs, "packed limb count")?;
    let factors = [
        SelectorFactor::Fixed {
            value: term.symbol,
            bits: alphabet_vars,
        },
        SelectorFactor::Fixed {
            value: term.limb,
            bits: limb_vars,
        },
        SelectorFactor::RowEq {
            point: &term.row_point,
        },
    ];
    selector_eval_with_offset(point, family.offset, term.coefficient, &factors)
}

#[derive(Clone, Copy)]
enum SelectorFactor<'a, F> {
    Fixed { value: usize, bits: usize },
    RowEq { point: &'a [F] },
}

impl<F> SelectorFactor<'_, F>
where
    F: Field,
{
    fn bits(self) -> usize {
        match self {
            Self::Fixed { bits, .. } => bits,
            Self::RowEq { point } => point.len(),
        }
    }

    fn bit_weight(self, bit_index: usize, bit: usize) -> F {
        match self {
            Self::Fixed { value, .. } => {
                if ((value >> bit_index) & 1) == bit {
                    F::one()
                } else {
                    F::zero()
                }
            }
            Self::RowEq { point } => {
                let challenge = point[point.len() - 1 - bit_index];
                if bit == 1 {
                    challenge
                } else {
                    F::one() - challenge
                }
            }
        }
    }
}

#[derive(Clone, Copy)]
struct CarryMatrix<F>([[F; 2]; 2]);

impl<F> CarryMatrix<F>
where
    F: Field,
{
    fn identity() -> Self {
        Self([[F::one(), F::zero()], [F::zero(), F::one()]])
    }

    fn zero() -> Self {
        Self([[F::zero(); 2]; 2])
    }

    fn add_assign(&mut self, carry_in: usize, carry_out: usize, value: F) {
        self.0[carry_in][carry_out] += value;
    }

    fn mul(&self, rhs: &Self) -> Self {
        let a = &self.0;
        let b = &rhs.0;
        Self([
            [
                a[0][0] * b[0][0] + a[0][1] * b[1][0],
                a[0][0] * b[0][1] + a[0][1] * b[1][1],
            ],
            [
                a[1][0] * b[0][0] + a[1][1] * b[1][0],
                a[1][0] * b[0][1] + a[1][1] * b[1][1],
            ],
        ])
    }
}

fn selector_eval_with_offset<F>(
    point: &[F],
    offset: usize,
    scale: F,
    factors: &[SelectorFactor<'_, F>],
) -> Result<F, OpeningsError>
where
    F: Field,
{
    let total_bits = factors.iter().map(|factor| factor.bits()).sum::<usize>();
    if total_bits > point.len() {
        return Err(invalid_batch(format!(
            "packed linear selector needs {total_bits} bits but point has {}",
            point.len()
        )));
    }

    let mut matrix = CarryMatrix::identity();
    let mut bit_cursor = 0usize;
    for &factor in factors {
        for bit_index in 0..factor.bits() {
            let bit_matrix = selector_bit_matrix(
                point[bit_cursor],
                offset_bit(offset, bit_cursor),
                factor,
                bit_index,
            );
            matrix = matrix.mul(&bit_matrix);
            bit_cursor += 1;
        }
    }
    for (bit_index, &challenge) in point.iter().enumerate().skip(bit_cursor) {
        let bit_matrix = fixed_zero_bit_matrix(challenge, offset_bit(offset, bit_index));
        matrix = matrix.mul(&bit_matrix);
    }
    Ok(scale * matrix.0[0][0])
}

fn native_opening_point<F: Copy>(sumcheck_point_lsb: &[F]) -> Vec<F> {
    sumcheck_point_lsb.iter().rev().copied().collect()
}

fn selector_bit_matrix<F>(
    challenge: F,
    offset_bit: bool,
    factor: SelectorFactor<'_, F>,
    factor_bit_index: usize,
) -> CarryMatrix<F>
where
    F: Field,
{
    let mut matrix = CarryMatrix::zero();
    for local_bit in 0..=1 {
        let factor_weight = factor.bit_weight(factor_bit_index, local_bit);
        if factor_weight.is_zero() {
            continue;
        }
        add_transition(&mut matrix, challenge, offset_bit, local_bit, factor_weight);
    }
    matrix
}

fn fixed_zero_bit_matrix<F>(challenge: F, offset_bit: bool) -> CarryMatrix<F>
where
    F: Field,
{
    let mut matrix = CarryMatrix::zero();
    add_transition(&mut matrix, challenge, offset_bit, 0, F::one());
    matrix
}

fn add_transition<F>(
    matrix: &mut CarryMatrix<F>,
    challenge: F,
    offset_bit: bool,
    local_bit: usize,
    scale: F,
) where
    F: Field,
{
    for carry_in in 0..=1 {
        let sum = usize::from(offset_bit) + local_bit + carry_in;
        let output_bit = sum & 1;
        let carry_out = sum >> 1;
        let eq_weight = if output_bit == 1 {
            challenge
        } else {
            F::one() - challenge
        };
        matrix.add_assign(carry_in, carry_out, scale * eq_weight);
    }
}

fn prove_product_sumcheck<F, T>(
    mut left: Vec<F>,
    mut right: Vec<F>,
    claimed_sum: F,
    transcript: &mut T,
) -> Result<(PackedLinearReductionProof, Vec<F>, F), OpeningsError>
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    if left.len() != right.len() || !left.len().is_power_of_two() {
        return Err(invalid_batch(
            "packed linear sumcheck inputs must have equal power-of-two lengths",
        ));
    }
    let rounds = left.len().trailing_zeros() as usize;
    let mut proof_rounds = Vec::with_capacity(rounds);
    let mut point = Vec::with_capacity(rounds);
    let mut current_claim = claimed_sum;
    transcript.append(&LabelWithCount(b"akpk_sum_rounds", rounds as u64));

    while left.len() > 1 {
        let round = product_round(&left, &right);
        if round[0] + round[1] != current_claim {
            return Err(invalid_batch(
                "packed linear claims do not match packed witness evaluations",
            ));
        }
        append_round(transcript, &round);
        let challenge = transcript.challenge_scalar();
        point.push(challenge);
        fold_product_inputs(&mut left, &mut right, challenge);
        current_claim = eval_quadratic(round, challenge);
        proof_rounds.push(encode_round(round));
    }
    if left[0] * right[0] != current_claim {
        return Err(invalid_batch("packed linear sumcheck final claim mismatch"));
    }
    let opening_eval = right[0];
    opening_eval.append_to_transcript(transcript);
    Ok((
        PackedLinearReductionProof {
            rounds: proof_rounds,
            opening_eval: field_bytes(opening_eval),
        },
        point,
        opening_eval,
    ))
}

fn prove_sparse_product_sumcheck<F, C, OpeningId, RelationId, L, S, T>(
    layout: &L,
    statement: &BatchOpeningStatement<F, C, OpeningId, RelationId>,
    gamma_powers: &[F],
    source: &S,
    claimed_sum: F,
    transcript: &mut T,
) -> Result<(PackedLinearReductionProof, Vec<F>, F), OpeningsError>
where
    F: Field,
    L: PackedLinearLayout,
    S: PackedLinearWitnessSource<F>,
    T: Transcript<Challenge = F>,
{
    let mut right = sparse_product_input(source)?;
    let rounds = layout.dimension();
    let mut proof_rounds = Vec::with_capacity(rounds);
    let mut point = Vec::with_capacity(rounds);
    let mut current_claim = claimed_sum;
    transcript.append(&LabelWithCount(b"akpk_sum_rounds", rounds as u64));

    for _ in 0..rounds {
        let round = sparse_product_round(layout, statement, gamma_powers, &point, &right)?;
        if round[0] + round[1] != current_claim {
            return Err(invalid_batch(
                "packed linear claims do not match sparse packed witness evaluations",
            ));
        }
        append_round(transcript, &round);
        let challenge = transcript.challenge_scalar();
        point.push(challenge);
        fold_sparse_product_input(&mut right, challenge);
        current_claim = eval_quadratic(round, challenge);
        proof_rounds.push(encode_round(round));
    }

    let opening_eval = right.first().map_or_else(F::zero, |(_, eval)| *eval);
    let selector_eval = packed_selector_eval(layout, statement, gamma_powers, &point)?;
    if selector_eval * opening_eval != current_claim {
        return Err(invalid_batch("packed linear sumcheck final claim mismatch"));
    }
    opening_eval.append_to_transcript(transcript);
    Ok((
        PackedLinearReductionProof {
            rounds: proof_rounds,
            opening_eval: field_bytes(opening_eval),
        },
        point,
        opening_eval,
    ))
}

fn sparse_product_input<F, S>(source: &S) -> Result<Vec<(usize, F)>, OpeningsError>
where
    F: Field,
    S: PackedLinearWitnessSource<F>,
{
    let layout = source.layout();
    let domain_size = checked_domain_size(layout.dimension())?;
    if layout.cells() > domain_size {
        return Err(invalid_batch(format!(
            "packed linear witness has {} cells but dimension {} supports {domain_size}",
            layout.cells(),
            layout.dimension()
        )));
    }

    let mut entries = Vec::new();
    let mut error = None;
    source.for_each_nonzero(|rank, value| {
        if error.is_some() {
            return;
        }
        if rank >= layout.cells() {
            error = Some(invalid_batch(format!(
                "packed linear witness source emitted rank {rank} outside {} real cells",
                layout.cells()
            )));
            return;
        }
        if value.is_zero() {
            error = Some(invalid_batch(format!(
                "packed linear witness source emitted zero at rank {rank}"
            )));
            return;
        }
        entries.push((rank, value));
    });
    if let Some(error) = error {
        return Err(error);
    }

    entries.sort_by_key(|(rank, _)| *rank);
    for window in entries.windows(2) {
        if window[0].0 == window[1].0 {
            return Err(invalid_batch(format!(
                "packed linear witness source emitted rank {} more than once",
                window[0].0
            )));
        }
    }
    Ok(entries)
}

fn sparse_product_round<F, C, OpeningId, RelationId, L>(
    layout: &L,
    statement: &BatchOpeningStatement<F, C, OpeningId, RelationId>,
    gamma_powers: &[F],
    fixed_point: &[F],
    right: &[(usize, F)],
) -> Result<[F; 3], OpeningsError>
where
    F: Field,
    L: PackedLinearLayout,
{
    let mut evals = [F::zero(); 3];
    let mut cursor = 0usize;
    while cursor < right.len() {
        let pair_index = right[cursor].0 / 2;
        let mut right_0 = F::zero();
        let mut right_1 = F::zero();
        while cursor < right.len() && right[cursor].0 / 2 == pair_index {
            let (index, value) = right[cursor];
            if index & 1 == 0 {
                right_0 += value;
            } else {
                right_1 += value;
            }
            cursor += 1;
        }

        let left_0 = packed_selector_eval_at_index(
            layout,
            statement,
            gamma_powers,
            fixed_point,
            pair_index * 2,
        )?;
        let left_1 = packed_selector_eval_at_index(
            layout,
            statement,
            gamma_powers,
            fixed_point,
            pair_index * 2 + 1,
        )?;
        evals[0] += left_0 * right_0;
        evals[1] += left_1 * right_1;
        evals[2] += (left_1 + left_1 - left_0) * (right_1 + right_1 - right_0);
    }
    Ok(evals)
}

fn packed_selector_eval_at_index<F, C, OpeningId, RelationId, L>(
    layout: &L,
    statement: &BatchOpeningStatement<F, C, OpeningId, RelationId>,
    gamma_powers: &[F],
    fixed_point: &[F],
    index: usize,
) -> Result<F, OpeningsError>
where
    F: Field,
    L: PackedLinearLayout,
{
    if fixed_point.len() > layout.dimension() {
        return Err(invalid_batch(
            "packed linear selector fixed point exceeds layout dimension",
        ));
    }
    let remaining_bits = layout.dimension() - fixed_point.len();
    if index >= (1usize << remaining_bits) {
        return Err(invalid_batch(
            "packed linear selector index exceeds folded domain",
        ));
    }

    let mut point = Vec::with_capacity(layout.dimension());
    point.extend_from_slice(fixed_point);
    for bit in 0..remaining_bits {
        if (index >> bit) & 1 == 0 {
            point.push(F::zero());
        } else {
            point.push(F::one());
        }
    }
    packed_selector_eval(layout, statement, gamma_powers, &point)
}

fn fold_sparse_product_input<F>(right: &mut Vec<(usize, F)>, r: F)
where
    F: Field,
{
    let mut folded: Vec<(usize, F)> = Vec::with_capacity(right.len());
    for &(index, value) in right.iter() {
        let next_index = index / 2;
        let weight = if index & 1 == 0 { F::one() - r } else { r };
        let folded_value = value * weight;
        if folded_value.is_zero() {
            continue;
        }
        match folded.last_mut() {
            Some((last_index, last_value)) if *last_index == next_index => {
                *last_value += folded_value;
                if last_value.is_zero() {
                    let _ = folded.pop();
                }
            }
            _ => folded.push((next_index, folded_value)),
        }
    }
    *right = folded;
}

fn verify_product_sumcheck<F, T>(
    proof: &PackedLinearReductionProof,
    claimed_sum: F,
    transcript: &mut T,
) -> Result<(Vec<F>, F), OpeningsError>
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    transcript.append(&LabelWithCount(
        b"akpk_sum_rounds",
        proof.rounds.len() as u64,
    ));
    let mut point = Vec::with_capacity(proof.rounds.len());
    let mut current_claim = claimed_sum;
    for encoded_round in &proof.rounds {
        let round = decode_round(encoded_round)?;
        if round[0] + round[1] != current_claim {
            return Err(OpeningsError::VerificationFailed);
        }
        append_round(transcript, &round);
        let challenge = transcript.challenge_scalar();
        point.push(challenge);
        current_claim = eval_quadratic(round, challenge);
    }
    field_from_bytes::<F>(&proof.opening_eval)?.append_to_transcript(transcript);
    Ok((point, current_claim))
}

fn product_round<F>(left: &[F], right: &[F]) -> [F; 3]
where
    F: Field,
{
    let mut evals = [F::zero(); 3];
    for (left_pair, right_pair) in left.chunks_exact(2).zip(right.chunks_exact(2)) {
        let left_0 = left_pair[0];
        let left_1 = left_pair[1];
        let right_0 = right_pair[0];
        let right_1 = right_pair[1];
        evals[0] += left_0 * right_0;
        evals[1] += left_1 * right_1;
        evals[2] += (left_1 + left_1 - left_0) * (right_1 + right_1 - right_0);
    }
    evals
}

fn fold_product_inputs<F>(left: &mut Vec<F>, right: &mut Vec<F>, r: F)
where
    F: Field,
{
    let half = left.len() / 2;
    for index in 0..half {
        let left_0 = left[2 * index];
        let left_1 = left[2 * index + 1];
        let right_0 = right[2 * index];
        let right_1 = right[2 * index + 1];
        left[index] = left_0 + r * (left_1 - left_0);
        right[index] = right_0 + r * (right_1 - right_0);
    }
    left.truncate(half);
    right.truncate(half);
}

fn eval_quadratic<F>(evals: [F; 3], r: F) -> F
where
    F: Field,
{
    let two_inv = F::from_u64(2).inv_or_zero();
    let l0 = (r - F::one()) * (r - F::from_u64(2)) * two_inv;
    let l1 = F::zero() - r * (r - F::from_u64(2));
    let l2 = r * (r - F::one()) * two_inv;
    evals[0] * l0 + evals[1] * l1 + evals[2] * l2
}

fn append_round<F, T>(transcript: &mut T, round: &[F; 3])
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    transcript.append(&Label(b"akpk_sum_round"));
    for eval in round {
        eval.append_to_transcript(transcript);
    }
}

fn bind_packed_statement<F, C, OpeningId, RelationId, L, T>(
    layout: &L,
    statement: &BatchOpeningStatement<F, C, OpeningId, RelationId>,
    transcript: &mut T,
) -> Result<(), OpeningsError>
where
    F: Field,
    C: AppendToTranscript,
    L: PackedLinearLayout,
    T: Transcript<Challenge = F>,
{
    transcript.append(&Label(b"akpk_batch_stmt"));
    transcript.append_bytes(&layout.digest());
    transcript.append(&U64Word(layout.dimension() as u64));
    transcript.append(&U64Word(layout.cells() as u64));
    append_field_slice(transcript, b"akpk_logical_point", &statement.logical_point);
    append_field_slice(transcript, b"akpk_pcs_point", &statement.pcs_point);
    transcript.append(&LabelWithCount(
        b"akita_packed_claims",
        statement.claims.len() as u64,
    ));
    for claim in &statement.claims {
        claim.commitment.append_to_transcript(transcript);
        claim.claim.append_to_transcript(transcript);
        claim.scale.append_to_transcript(transcript);
        match &claim.view {
            PhysicalView::Direct => transcript.append_bytes(&[0]),
            PhysicalView::PackedLinear {
                layout_digest,
                terms,
            } => {
                transcript.append_bytes(&[1]);
                transcript.append_bytes(layout_digest);
                transcript.append(&LabelWithCount(b"akpk_view_terms", terms.len() as u64));
                for term in terms {
                    validate_term(layout, term)?;
                    transcript.append(&U64Word(term.family.namespace));
                    transcript.append(&U64Word(term.family.id));
                    transcript.append(&U64Word(term.family.index));
                    transcript.append(&U64Word(term.limb as u64));
                    transcript.append(&U64Word(term.symbol as u64));
                    append_field_slice(transcript, b"akpk_view_row_point", &term.row_point);
                    term.coefficient.append_to_transcript(transcript);
                }
            }
        }
    }
    Ok(())
}

fn checked_domain_size(num_vars: usize) -> Result<usize, OpeningsError> {
    if num_vars >= usize::BITS as usize {
        return Err(invalid_batch(format!(
            "packed linear dimension {num_vars} exceeds usize bit width"
        )));
    }
    Ok(1usize << num_vars)
}

fn log2_power_of_two(value: usize, label: &'static str) -> Result<usize, OpeningsError> {
    if value == 0 || !value.is_power_of_two() {
        return Err(invalid_batch(format!(
            "{label} must be a nonzero power of two"
        )));
    }
    Ok(value.trailing_zeros() as usize)
}

fn offset_bit(offset: usize, bit: usize) -> bool {
    bit < usize::BITS as usize && ((offset >> bit) & 1) != 0
}

fn field_bytes<F>(value: F) -> Vec<u8>
where
    F: Field,
{
    value.to_bytes_le_vec()
}

fn field_from_bytes<F>(bytes: &[u8]) -> Result<F, OpeningsError>
where
    F: Field,
{
    if bytes.len() != F::NUM_BYTES {
        return Err(invalid_batch(format!(
            "packed linear proof field encoding has {} bytes but expected {}",
            bytes.len(),
            F::NUM_BYTES
        )));
    }
    let value = F::from_le_bytes_mod_order(bytes);
    if value.to_bytes_le_vec() != bytes {
        return Err(invalid_batch(
            "packed linear proof field encoding is not canonical",
        ));
    }
    Ok(value)
}

fn encode_round<F>(round: [F; 3]) -> [Vec<u8>; 3]
where
    F: Field,
{
    [
        field_bytes(round[0]),
        field_bytes(round[1]),
        field_bytes(round[2]),
    ]
}

fn decode_round<F>(round: &[Vec<u8>; 3]) -> Result<[F; 3], OpeningsError>
where
    F: Field,
{
    Ok([
        field_from_bytes(&round[0])?,
        field_from_bytes(&round[1])?,
        field_from_bytes(&round[2])?,
    ])
}

fn polynomial_evaluations<F, P>(polynomial: &P) -> Vec<F>
where
    F: Field,
    P: MultilinearPoly<F>,
{
    let mut evals = Vec::with_capacity(1usize << polynomial.num_vars());
    polynomial.for_each_row(polynomial.num_vars(), &mut |_, row| {
        evals.extend_from_slice(row);
    });
    evals
}

fn append_field_slice<F, T>(transcript: &mut T, label: &'static [u8], values: &[F])
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    transcript.append(&LabelWithCount(label, values.len() as u64));
    for value in values {
        value.append_to_transcript(transcript);
    }
}

fn invalid_batch(reason: impl Into<String>) -> OpeningsError {
    OpeningsError::InvalidBatch(reason.into())
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    clippy::panic,
    reason = "tests assert successful packed reduction setup and fail loudly on malformed fixtures"
)]
mod tests {
    use serde::{Deserialize, Serialize};

    use super::*;
    use crate::mock::{MockCommitment, MockCommitmentScheme, MockProof};
    use crate::BatchOpeningClaim;
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_poly::Polynomial;
    use jolt_transcript::Blake2bTranscript;

    const FAMILY: PackedFamilyRef = PackedFamilyRef {
        namespace: 0x6a6f_6c74,
        id: 7,
        index: 0,
    };

    #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
    struct TestPackedPcs;

    #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
    struct TestLayout {
        digest: [u8; 32],
    }

    impl PackedLinearLayout for TestLayout {
        fn digest(&self) -> [u8; 32] {
            self.digest
        }

        fn dimension(&self) -> usize {
            3
        }

        fn cells(&self) -> usize {
            8
        }

        fn family(
            &self,
            family: PackedFamilyRef,
        ) -> Result<Option<PackedLinearFamily>, OpeningsError> {
            Ok((family == FAMILY).then_some(PackedLinearFamily {
                id: FAMILY,
                offset: 0,
                rows: 4,
                limbs: 1,
                alphabet_size: 2,
            }))
        }

        fn rank(&self, address: PackedLinearAddress) -> Result<usize, OpeningsError> {
            if address.family != FAMILY
                || address.limb != 0
                || address.row >= 4
                || address.symbol >= 2
            {
                return Err(OpeningsError::InvalidBatch(
                    "test packed address out of range".to_string(),
                ));
            }
            Ok(address.row * 2 + address.symbol)
        }
    }

    impl Commitment for TestPackedPcs {
        type Output = MockCommitment<Fr>;
    }

    impl CommitmentScheme for TestPackedPcs {
        type Field = Fr;
        type Proof = MockProof<Fr>;
        type ProverSetup = TestLayout;
        type VerifierSetup = TestLayout;
        type Polynomial = Polynomial<Fr>;
        type OpeningHint = ();
        type SetupParams = TestLayout;

        fn setup(params: Self::SetupParams) -> (Self::ProverSetup, Self::VerifierSetup) {
            (params.clone(), params)
        }

        fn verifier_setup(prover_setup: &Self::ProverSetup) -> Self::VerifierSetup {
            prover_setup.clone()
        }

        fn commit<P: MultilinearPoly<Self::Field> + ?Sized>(
            poly: &P,
            _setup: &Self::ProverSetup,
        ) -> (Self::Output, Self::OpeningHint) {
            MockCommitmentScheme::<Fr>::commit(poly, &())
        }

        fn open(
            poly: &Self::Polynomial,
            point: &[Self::Field],
            eval: Self::Field,
            _setup: &Self::ProverSetup,
            hint: Option<Self::OpeningHint>,
            transcript: &mut impl Transcript<Challenge = Self::Field>,
        ) -> Self::Proof {
            MockCommitmentScheme::<Fr>::open(poly, point, eval, &(), hint, transcript)
        }

        fn verify(
            commitment: &Self::Output,
            point: &[Self::Field],
            eval: Self::Field,
            proof: &Self::Proof,
            _setup: &Self::VerifierSetup,
            transcript: &mut impl Transcript<Challenge = Self::Field>,
        ) -> Result<(), OpeningsError> {
            MockCommitmentScheme::<Fr>::verify(commitment, point, eval, proof, &(), transcript)
        }

        fn bind_opening_inputs(
            transcript: &mut impl Transcript<Challenge = Self::Field>,
            point: &[Self::Field],
            eval: &Self::Field,
        ) {
            MockCommitmentScheme::<Fr>::bind_opening_inputs(transcript, point, eval);
        }
    }

    impl BatchOpeningScheme for TestPackedPcs {
        fn prove_batch<T, OpeningId, RelationId>(
            _setup: &Self::ProverSetup,
            _transcript: &mut T,
            _statement: &BatchOpeningStatement<Self::Field, Self::Output, OpeningId, RelationId>,
            _polynomials: &[Self::Polynomial],
            _hints: Vec<Self::OpeningHint>,
        ) -> Result<Self::Proof, OpeningsError>
        where
            T: Transcript<Challenge = Self::Field>,
        {
            Err(OpeningsError::InvalidBatch(
                "test PCS only supports packed adapter openings".to_string(),
            ))
        }

        fn verify_batch<T, OpeningId, RelationId>(
            _setup: &Self::VerifierSetup,
            _transcript: &mut T,
            _statement: &BatchOpeningStatement<Self::Field, Self::Output, OpeningId, RelationId>,
            _proof: &Self::Proof,
        ) -> Result<BatchOpeningResult<Self::Field, Self::Output>, OpeningsError>
        where
            T: Transcript<Challenge = Self::Field>,
        {
            Err(OpeningsError::InvalidBatch(
                "test PCS only supports packed adapter openings".to_string(),
            ))
        }
    }

    impl PackedLinearBatchBackend for TestPackedPcs {
        type Layout = TestLayout;

        fn prover_layout(setup: &Self::ProverSetup) -> Option<&Self::Layout> {
            Some(setup)
        }

        fn verifier_layout(setup: &Self::VerifierSetup) -> Option<&Self::Layout> {
            Some(setup)
        }
    }

    fn fr(value: u64) -> Fr {
        Fr::from_u64(value)
    }

    fn layout() -> TestLayout {
        TestLayout { digest: [17; 32] }
    }

    fn packed_polynomial() -> Polynomial<Fr> {
        Polynomial::new((0..8).map(|index| fr(3 + 2 * index)).collect())
    }

    fn packed_term(coefficient: Fr, symbol: usize, row_point: &[Fr]) -> PackedLinearTerm<Fr> {
        PackedLinearTerm::new(coefficient, FAMILY, 0, symbol).with_row_point(row_point.to_vec())
    }

    fn packed_view_eval(
        poly: &Polynomial<Fr>,
        row_point: &[Fr],
        terms: &[PackedLinearTerm<Fr>],
    ) -> Fr {
        let row_weights = EqPolynomial::new(row_point.to_vec()).evaluations();
        terms.iter().fold(fr(0), |acc, term| {
            acc + row_weights.iter().copied().enumerate().fold(
                fr(0),
                |term_acc, (row, row_weight)| {
                    term_acc
                        + term.coefficient * row_weight * poly.evaluations()[row * 2 + term.symbol]
                },
            )
        })
    }

    fn statement(
        commitment: MockCommitment<Fr>,
        poly: &Polynomial<Fr>,
        row_point: &[Fr],
    ) -> BatchOpeningStatement<Fr, MockCommitment<Fr>, usize, usize> {
        let first_terms = vec![packed_term(fr(2), 1, row_point)];
        let second_terms = vec![
            packed_term(fr(5), 0, row_point),
            packed_term(fr(7), 1, row_point),
        ];
        BatchOpeningStatement {
            logical_point: row_point.to_vec(),
            pcs_point: row_point.to_vec(),
            layout_digest: [17; 32],
            claims: vec![
                BatchOpeningClaim {
                    id: 0,
                    relation: 0,
                    commitment: commitment.clone(),
                    claim: packed_view_eval(poly, row_point, &first_terms),
                    view: PhysicalView::PackedLinear {
                        layout_digest: [17; 32],
                        terms: first_terms,
                    },
                    scale: fr(11),
                },
                BatchOpeningClaim {
                    id: 1,
                    relation: 1,
                    commitment,
                    claim: packed_view_eval(poly, row_point, &second_terms),
                    view: PhysicalView::PackedLinear {
                        layout_digest: [17; 32],
                        terms: second_terms,
                    },
                    scale: fr(13),
                },
            ],
        }
    }

    #[test]
    fn packed_linear_batch_roundtrip_many_views_one_commitment() {
        type PackedTestPcs = PackedLinearBatch<TestPackedPcs>;

        let layout = layout();
        let poly = packed_polynomial();
        let (commitment, hint) = TestPackedPcs::commit(&poly, &layout);
        let row_point = vec![fr(2), fr(5)];
        let statement = statement(commitment.clone(), &poly, &row_point);

        let mut prover_transcript = Blake2bTranscript::new(b"generic-packed-linear");
        let proof = <PackedTestPcs as BatchOpeningScheme>::prove_batch(
            &layout,
            &mut prover_transcript,
            &statement,
            std::slice::from_ref(&poly),
            vec![hint],
        )
        .expect("generic packed proof should prove");
        assert!(proof.reduction.is_some());

        let mut verifier_transcript = Blake2bTranscript::new(b"generic-packed-linear");
        let result = <PackedTestPcs as BatchOpeningScheme>::verify_batch(
            &layout,
            &mut verifier_transcript,
            &statement,
            &proof,
        )
        .expect("generic packed proof should verify");

        assert_eq!(result.joint_commitment, commitment);
        assert_eq!(result.coefficients.len(), statement.claims.len());
        assert_eq!(
            result.reduced_opening,
            result
                .coefficients
                .iter()
                .zip(&statement.claims)
                .fold(fr(0), |acc, (coefficient, claim)| {
                    acc + *coefficient * claim.claim
                })
        );
        assert_eq!(prover_transcript.state(), verifier_transcript.state());
    }

    #[test]
    fn packed_linear_batch_rejects_tampered_view_address() {
        type PackedTestPcs = PackedLinearBatch<TestPackedPcs>;

        let layout = layout();
        let poly = packed_polynomial();
        let (commitment, hint) = TestPackedPcs::commit(&poly, &layout);
        let row_point = vec![fr(2), fr(5)];
        let statement = statement(commitment, &poly, &row_point);
        let mut prover_transcript = Blake2bTranscript::new(b"generic-packed-linear-tamper");
        let proof = <PackedTestPcs as BatchOpeningScheme>::prove_batch(
            &layout,
            &mut prover_transcript,
            &statement,
            std::slice::from_ref(&poly),
            vec![hint],
        )
        .expect("generic packed proof should prove");

        let mut tampered = statement;
        let PhysicalView::PackedLinear { terms, .. } = &mut tampered.claims[0].view else {
            panic!("test statement uses packed views");
        };
        terms[0].symbol = 0;

        let mut verifier_transcript = Blake2bTranscript::new(b"generic-packed-linear-tamper");
        let result = <PackedTestPcs as BatchOpeningScheme>::verify_batch(
            &layout,
            &mut verifier_transcript,
            &tampered,
            &proof,
        );
        assert!(result.is_err(), "tampered packed address should reject");
    }
}
