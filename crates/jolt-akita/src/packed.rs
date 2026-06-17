use jolt_crypto::Commitment;
use jolt_field::{CanonicalBytes, FixedByteSize, Invertible, ReducingBytes};
use jolt_openings::{
    BatchOpeningResult, BatchOpeningScheme, BatchOpeningStatement, CommitmentScheme, OpeningsError,
    PackedLinearTerm, PhysicalView, ZkBatchOpeningScheme, ZkOpeningScheme,
};
use jolt_poly::{EqPolynomial, MultilinearPoly, Polynomial};
use jolt_transcript::{AppendToTranscript, Label, LabelWithCount, Transcript, U64Word};
use serde::{Deserialize, Serialize};

use crate::layout::{PackedCellAddress, PackedFamily, PackedWitnessLayout, PackedWitnessSource};
use crate::types::{
    append_field_slice, AkitaCommitInput, AkitaCommitment, AkitaField, AkitaHidingCommitment,
    AkitaPackedBatchProof, AkitaPackedReductionProof, AkitaProverHint, AkitaProverSetup,
    AkitaSetupParams, AkitaVerifierSetup,
};
use crate::AkitaScheme;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AkitaPackedScheme;

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
}

struct PackedBatchShape<'a> {
    layout: &'a PackedWitnessLayout,
    commitment: AkitaCommitment,
}

impl Commitment for AkitaPackedScheme {
    type Output = AkitaCommitment;
}

impl CommitmentScheme for AkitaPackedScheme {
    type Field = AkitaField;
    type Proof = AkitaPackedBatchProof;
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
        poly: &P,
        setup: &Self::ProverSetup,
    ) -> (Self::Output, Self::OpeningHint) {
        AkitaScheme::commit(poly, setup)
    }

    fn open(
        poly: &Self::Polynomial,
        point: &[Self::Field],
        eval: Self::Field,
        setup: &Self::ProverSetup,
        hint: Option<Self::OpeningHint>,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Self::Proof {
        let native = AkitaScheme::open(poly, point, eval, setup, hint, transcript);
        AkitaPackedBatchProof {
            reduction: None,
            native,
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
        AkitaScheme::verify(commitment, point, eval, &proof.native, setup, transcript)
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
        if !has_packed_view(statement) {
            let native =
                AkitaScheme::prove_batch(setup, transcript, statement, polynomials, hints)?;
            return Ok(AkitaPackedBatchProof {
                reduction: None,
                native,
            });
        }

        let shape = validate_packed_prover_inputs(setup, statement, polynomials, &hints)?;
        let hint = hints
            .into_iter()
            .next()
            .ok_or_else(|| invalid_batch("Akita packed proof requires one opening hint"))?;
        bind_packed_statement(statement, shape.layout, transcript)?;

        let gamma_powers = transcript.challenge_scalar_powers(statement.claims.len());
        let claimed_sum = reduced_claim(statement, &gamma_powers);
        let selector = packed_selector_evals(shape.layout, statement, &gamma_powers)?;
        let (reduction, sumcheck_point_lsb, opening_eval) = prove_product_sumcheck(
            selector,
            polynomials[0].evals().to_vec(),
            claimed_sum,
            transcript,
        )?;
        let opening_point = native_opening_point(&sumcheck_point_lsb);
        let native = AkitaScheme::open(
            &polynomials[0],
            &opening_point,
            opening_eval,
            setup,
            Some(hint),
            transcript,
        );
        Ok(AkitaPackedBatchProof {
            reduction: Some(reduction),
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
        if !has_packed_view(statement) {
            if proof.reduction.is_some() {
                return Err(OpeningsError::VerificationFailed);
            }
            return AkitaScheme::verify_batch(setup, transcript, statement, &proof.native);
        }

        let reduction = proof
            .reduction
            .as_ref()
            .ok_or(OpeningsError::VerificationFailed)?;
        let shape = validate_packed_verifier_inputs(setup, statement)?;
        bind_packed_statement(statement, shape.layout, transcript)?;

        let gamma_powers = transcript.challenge_scalar_powers(statement.claims.len());
        let coefficients = logical_coefficients(statement, &gamma_powers);
        let claimed_sum = reduced_claim(statement, &gamma_powers);
        let (sumcheck_point_lsb, final_claim) =
            verify_product_sumcheck(reduction, claimed_sum, transcript)?;
        let selector_eval =
            packed_selector_eval(shape.layout, statement, &gamma_powers, &sumcheck_point_lsb)?;
        let opening_eval = field_from_bytes(&reduction.opening_eval)?;
        if final_claim != selector_eval * opening_eval {
            return Err(OpeningsError::VerificationFailed);
        }
        let opening_point = native_opening_point(&sumcheck_point_lsb);
        AkitaScheme::verify(
            &shape.commitment,
            &opening_point,
            opening_eval,
            &proof.native,
            setup,
            transcript,
        )?;

        Ok(BatchOpeningResult {
            coefficients,
            joint_commitment: shape.commitment,
            reduced_opening: claimed_sum,
        })
    }
}

impl ZkOpeningScheme for AkitaPackedScheme {
    type HidingCommitment = AkitaHidingCommitment;
    type Blind = ();

    fn commit_zk<P: MultilinearPoly<Self::Field> + ?Sized>(
        poly: &P,
        setup: &Self::ProverSetup,
    ) -> (Self::Output, Self::OpeningHint) {
        AkitaScheme::commit_zk(poly, setup)
    }

    fn open_zk(
        poly: &Self::Polynomial,
        point: &[Self::Field],
        eval: Self::Field,
        setup: &Self::ProverSetup,
        hint: Self::OpeningHint,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> (Self::Proof, Self::HidingCommitment, Self::Blind) {
        (
            Self::open(poly, point, eval, setup, Some(hint), transcript),
            AkitaHidingCommitment {
                eval: field_bytes(eval),
            },
            (),
        )
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

fn has_packed_view<OpeningId, RelationId>(
    statement: &BatchOpeningStatement<AkitaField, AkitaCommitment, OpeningId, RelationId>,
) -> bool {
    statement
        .claims
        .iter()
        .any(|claim| matches!(claim.view, PhysicalView::PackedLinear { .. }))
}

fn validate_packed_prover_inputs<'a, OpeningId, RelationId>(
    setup: &'a AkitaProverSetup,
    statement: &BatchOpeningStatement<AkitaField, AkitaCommitment, OpeningId, RelationId>,
    polynomials: &[Polynomial<AkitaField>],
    hints: &[AkitaProverHint],
) -> Result<PackedBatchShape<'a>, OpeningsError> {
    let shape = validate_packed_statement(setup.packed_layout.as_ref(), statement)?;
    if shape.commitment.num_vars > setup.max_num_vars {
        return Err(OpeningsError::PolynomialTooLarge {
            poly_size: shape.commitment.num_vars,
            setup_max: setup.max_num_vars,
        });
    }
    if shape.commitment.num_vars != setup.max_num_vars {
        return Err(invalid_batch(format!(
            "Akita packed commitment dimension {} does not match exact setup dimension {}",
            shape.commitment.num_vars, setup.max_num_vars
        )));
    }
    if shape.commitment.layout_digest != setup.default_layout_digest {
        return Err(invalid_batch(
            "Akita packed commitment layout digest does not match setup",
        ));
    }
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
    Ok(shape)
}

fn validate_packed_verifier_inputs<'a, OpeningId, RelationId>(
    setup: &'a AkitaVerifierSetup,
    statement: &BatchOpeningStatement<AkitaField, AkitaCommitment, OpeningId, RelationId>,
) -> Result<PackedBatchShape<'a>, OpeningsError> {
    let shape = validate_packed_statement(setup.packed_layout.as_ref(), statement)?;
    if shape.commitment.num_vars > setup.max_num_vars {
        return Err(OpeningsError::PolynomialTooLarge {
            poly_size: shape.commitment.num_vars,
            setup_max: setup.max_num_vars,
        });
    }
    if shape.commitment.num_vars != setup.max_num_vars {
        return Err(invalid_batch(format!(
            "Akita packed commitment dimension {} does not match exact setup dimension {}",
            shape.commitment.num_vars, setup.max_num_vars
        )));
    }
    if shape.commitment.layout_digest != setup.default_layout_digest {
        return Err(invalid_batch(
            "Akita packed commitment layout digest does not match setup",
        ));
    }
    Ok(shape)
}

fn validate_packed_statement<'a, OpeningId, RelationId>(
    layout: Option<&'a PackedWitnessLayout>,
    statement: &BatchOpeningStatement<AkitaField, AkitaCommitment, OpeningId, RelationId>,
) -> Result<PackedBatchShape<'a>, OpeningsError> {
    let layout =
        layout.ok_or_else(|| invalid_batch("Akita packed opening requires setup layout"))?;
    if statement.claims.is_empty() {
        return Err(invalid_batch(
            "Akita packed opening requires at least one claim",
        ));
    }
    if statement.layout_digest != layout.digest {
        return Err(invalid_batch(
            "Akita packed statement layout digest does not match setup layout",
        ));
    }
    let commitment = statement.claims[0].commitment.clone();
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

    for claim in &statement.claims {
        if claim.commitment != commitment {
            return Err(invalid_batch(
                "Akita packed opening claims must use one packed witness commitment",
            ));
        }
        let PhysicalView::PackedLinear {
            layout_digest,
            terms,
        } = &claim.view
        else {
            return Err(invalid_batch(
                "Akita packed opening requires PackedLinear physical views",
            ));
        };
        if layout_digest != &layout.digest {
            return Err(invalid_batch(
                "Akita packed view layout digest does not match statement layout",
            ));
        }
        if terms.is_empty() {
            return Err(invalid_batch(
                "Akita packed linear view requires at least one term",
            ));
        }
        for term in terms {
            validate_term(layout, term)?;
        }
    }

    Ok(PackedBatchShape { layout, commitment })
}

fn validate_term(
    layout: &PackedWitnessLayout,
    term: &PackedLinearTerm<AkitaField>,
) -> Result<(), OpeningsError> {
    let family = family_for_term(layout, term)?;
    let rows = family.domain.rows().map_err(layout_error)?;
    let row_vars = log2_power_of_two(rows, "packed family rows")?;
    if term.row_point.len() != row_vars {
        return Err(invalid_batch(format!(
            "Akita packed term row point has {} variables but family requires {row_vars}",
            term.row_point.len()
        )));
    }
    if !family.alphabet.size().is_power_of_two() {
        return Err(invalid_batch(
            "Akita packed verifier currently requires power-of-two alphabets",
        ));
    }
    if !family.limbs.is_power_of_two() {
        return Err(invalid_batch(
            "Akita packed verifier currently requires power-of-two limb counts",
        ));
    }
    layout
        .rank(&PackedCellAddress {
            family: family.id.clone(),
            row: 0,
            limb: term.limb,
            symbol: term.symbol,
        })
        .map(|_| ())
        .map_err(layout_error)
}

fn family_for_term<'a>(
    layout: &'a PackedWitnessLayout,
    term: &PackedLinearTerm<AkitaField>,
) -> Result<&'a PackedFamily, OpeningsError> {
    layout
        .families
        .iter()
        .find(|family| family.id.physical_ref() == term.family)
        .ok_or_else(|| invalid_batch("Akita packed term references an unknown family"))
}

fn logical_coefficients<OpeningId, RelationId>(
    statement: &BatchOpeningStatement<AkitaField, AkitaCommitment, OpeningId, RelationId>,
    gamma_powers: &[AkitaField],
) -> Vec<AkitaField> {
    statement
        .claims
        .iter()
        .zip(gamma_powers)
        .map(|(claim, gamma)| *gamma * claim.scale)
        .collect()
}

fn reduced_claim<OpeningId, RelationId>(
    statement: &BatchOpeningStatement<AkitaField, AkitaCommitment, OpeningId, RelationId>,
    gamma_powers: &[AkitaField],
) -> AkitaField {
    statement
        .claims
        .iter()
        .zip(gamma_powers)
        .fold(AkitaField::zero(), |acc, (claim, gamma)| {
            acc + *gamma * claim.scale * claim.claim
        })
}

fn packed_selector_evals<OpeningId, RelationId>(
    layout: &PackedWitnessLayout,
    statement: &BatchOpeningStatement<AkitaField, AkitaCommitment, OpeningId, RelationId>,
    gamma_powers: &[AkitaField],
) -> Result<Vec<AkitaField>, OpeningsError> {
    let domain_size = checked_domain_size(layout.dimension)?;
    let mut selector = vec![AkitaField::zero(); domain_size];
    for (claim, gamma) in statement.claims.iter().zip(gamma_powers) {
        let PhysicalView::PackedLinear { terms, .. } = &claim.view else {
            return Err(invalid_batch(
                "Akita packed selector requires PackedLinear views",
            ));
        };
        let claim_weight = *gamma * claim.scale;
        for term in terms {
            let family = family_for_term(layout, term)?;
            let rows = family.domain.rows().map_err(layout_error)?;
            let row_weights = EqPolynomial::new(term.row_point.clone()).evaluations();
            if row_weights.len() != rows {
                return Err(invalid_batch(
                    "Akita packed term row point does not match family row count",
                ));
            }
            let weight = claim_weight * term.coefficient;
            for (row, row_weight) in row_weights.iter().copied().enumerate() {
                if row_weight.is_zero() {
                    continue;
                }
                let rank = layout
                    .rank(&PackedCellAddress {
                        family: family.id.clone(),
                        row,
                        limb: term.limb,
                        symbol: term.symbol,
                    })
                    .map_err(layout_error)?;
                selector[rank] += weight * row_weight;
            }
        }
    }
    Ok(selector)
}

fn packed_selector_eval<OpeningId, RelationId>(
    layout: &PackedWitnessLayout,
    statement: &BatchOpeningStatement<AkitaField, AkitaCommitment, OpeningId, RelationId>,
    gamma_powers: &[AkitaField],
    point: &[AkitaField],
) -> Result<AkitaField, OpeningsError> {
    if point.len() != layout.dimension {
        return Err(invalid_batch(format!(
            "Akita packed selector point has {} variables but layout has {}",
            point.len(),
            layout.dimension
        )));
    }
    let mut result = AkitaField::zero();
    for (claim, gamma) in statement.claims.iter().zip(gamma_powers) {
        let PhysicalView::PackedLinear { terms, .. } = &claim.view else {
            return Err(invalid_batch(
                "Akita packed selector requires PackedLinear views",
            ));
        };
        let claim_weight = *gamma * claim.scale;
        for term in terms {
            result += packed_term_selector_eval(layout, term, point)? * claim_weight;
        }
    }
    Ok(result)
}

fn packed_term_selector_eval(
    layout: &PackedWitnessLayout,
    term: &PackedLinearTerm<AkitaField>,
    point: &[AkitaField],
) -> Result<AkitaField, OpeningsError> {
    let family = family_for_term(layout, term)?;
    let rows = family.domain.rows().map_err(layout_error)?;
    let row_vars = log2_power_of_two(rows, "packed family rows")?;
    if term.row_point.len() != row_vars {
        return Err(invalid_batch(format!(
            "Akita packed term row point has {} variables but family requires {row_vars}",
            term.row_point.len()
        )));
    }
    let alphabet_vars = log2_power_of_two(family.alphabet.size(), "packed alphabet")?;
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
enum SelectorFactor<'a> {
    Fixed { value: usize, bits: usize },
    RowEq { point: &'a [AkitaField] },
}

impl SelectorFactor<'_> {
    fn bits(self) -> usize {
        match self {
            Self::Fixed { bits, .. } => bits,
            Self::RowEq { point } => point.len(),
        }
    }

    fn bit_weight(self, bit_index: usize, bit: usize) -> AkitaField {
        match self {
            Self::Fixed { value, .. } => {
                if ((value >> bit_index) & 1) == bit {
                    AkitaField::one()
                } else {
                    AkitaField::zero()
                }
            }
            Self::RowEq { point } => {
                let challenge = point[point.len() - 1 - bit_index];
                if bit == 1 {
                    challenge
                } else {
                    AkitaField::one() - challenge
                }
            }
        }
    }
}

#[derive(Clone, Copy)]
struct CarryMatrix([[AkitaField; 2]; 2]);

impl CarryMatrix {
    fn identity() -> Self {
        Self([
            [AkitaField::one(), AkitaField::zero()],
            [AkitaField::zero(), AkitaField::one()],
        ])
    }

    fn zero() -> Self {
        Self([[AkitaField::zero(); 2]; 2])
    }

    fn add_assign(&mut self, carry_in: usize, carry_out: usize, value: AkitaField) {
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

fn selector_eval_with_offset(
    point: &[AkitaField],
    offset: usize,
    scale: AkitaField,
    factors: &[SelectorFactor<'_>],
) -> Result<AkitaField, OpeningsError> {
    let total_bits = factors.iter().map(|factor| factor.bits()).sum::<usize>();
    if total_bits > point.len() {
        return Err(invalid_batch(format!(
            "Akita packed selector needs {total_bits} bits but point has {}",
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

fn native_opening_point(sumcheck_point_lsb: &[AkitaField]) -> Vec<AkitaField> {
    sumcheck_point_lsb.iter().rev().copied().collect()
}

fn selector_bit_matrix(
    challenge: AkitaField,
    offset_bit: bool,
    factor: SelectorFactor<'_>,
    factor_bit_index: usize,
) -> CarryMatrix {
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

fn fixed_zero_bit_matrix(challenge: AkitaField, offset_bit: bool) -> CarryMatrix {
    let mut matrix = CarryMatrix::zero();
    add_transition(&mut matrix, challenge, offset_bit, 0, AkitaField::one());
    matrix
}

fn add_transition(
    matrix: &mut CarryMatrix,
    challenge: AkitaField,
    offset_bit: bool,
    local_bit: usize,
    scale: AkitaField,
) {
    for carry_in in 0..=1 {
        let sum = usize::from(offset_bit) + local_bit + carry_in;
        let output_bit = sum & 1;
        let carry_out = sum >> 1;
        let eq_weight = if output_bit == 1 {
            challenge
        } else {
            AkitaField::one() - challenge
        };
        matrix.add_assign(carry_in, carry_out, scale * eq_weight);
    }
}

fn prove_product_sumcheck<T>(
    mut left: Vec<AkitaField>,
    mut right: Vec<AkitaField>,
    claimed_sum: AkitaField,
    transcript: &mut T,
) -> Result<(AkitaPackedReductionProof, Vec<AkitaField>, AkitaField), OpeningsError>
where
    T: Transcript<Challenge = AkitaField>,
{
    if left.len() != right.len() || !left.len().is_power_of_two() {
        return Err(invalid_batch(
            "Akita packed sumcheck inputs must have equal power-of-two lengths",
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
                "Akita packed claims do not match packed witness evaluations",
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
        return Err(invalid_batch("Akita packed sumcheck final claim mismatch"));
    }
    let opening_eval = right[0];
    opening_eval.append_to_transcript(transcript);
    Ok((
        AkitaPackedReductionProof {
            rounds: proof_rounds,
            opening_eval: field_bytes(opening_eval),
        },
        point,
        opening_eval,
    ))
}

fn verify_product_sumcheck<T>(
    proof: &AkitaPackedReductionProof,
    claimed_sum: AkitaField,
    transcript: &mut T,
) -> Result<(Vec<AkitaField>, AkitaField), OpeningsError>
where
    T: Transcript<Challenge = AkitaField>,
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
    field_from_bytes(&proof.opening_eval)?.append_to_transcript(transcript);
    Ok((point, current_claim))
}

fn product_round(left: &[AkitaField], right: &[AkitaField]) -> [AkitaField; 3] {
    let mut evals = [AkitaField::zero(); 3];
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

fn fold_product_inputs(left: &mut Vec<AkitaField>, right: &mut Vec<AkitaField>, r: AkitaField) {
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

fn eval_quadratic(evals: [AkitaField; 3], r: AkitaField) -> AkitaField {
    let two_inv = AkitaField::from_u64(2).inv_or_zero();
    let l0 = (r - AkitaField::one()) * (r - AkitaField::from_u64(2)) * two_inv;
    let l1 = AkitaField::zero() - r * (r - AkitaField::from_u64(2));
    let l2 = r * (r - AkitaField::one()) * two_inv;
    evals[0] * l0 + evals[1] * l1 + evals[2] * l2
}

fn append_round<T>(transcript: &mut T, round: &[AkitaField; 3])
where
    T: Transcript<Challenge = AkitaField>,
{
    transcript.append(&Label(b"akpk_sum_round"));
    for eval in round {
        eval.append_to_transcript(transcript);
    }
}

fn bind_packed_statement<OpeningId, RelationId, T>(
    statement: &BatchOpeningStatement<AkitaField, AkitaCommitment, OpeningId, RelationId>,
    layout: &PackedWitnessLayout,
    transcript: &mut T,
) -> Result<(), OpeningsError>
where
    T: Transcript<Challenge = AkitaField>,
{
    transcript.append(&Label(b"akpk_batch_stmt"));
    transcript.append_bytes(&layout.digest);
    transcript.append(&U64Word(layout.dimension as u64));
    transcript.append(&U64Word(layout.cells as u64));
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
            "Akita packed dimension {num_vars} exceeds usize bit width"
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

fn field_bytes(value: AkitaField) -> Vec<u8> {
    let mut bytes = vec![0u8; AkitaField::NUM_BYTES];
    value.to_bytes_le(&mut bytes);
    bytes
}

fn field_from_bytes(bytes: &[u8]) -> Result<AkitaField, OpeningsError> {
    if bytes.len() != AkitaField::NUM_BYTES {
        return Err(invalid_batch(format!(
            "Akita packed proof field encoding has {} bytes but expected {}",
            bytes.len(),
            AkitaField::NUM_BYTES
        )));
    }
    Ok(AkitaField::from_le_bytes_mod_order(bytes))
}

fn encode_round(round: [AkitaField; 3]) -> [Vec<u8>; 3] {
    [
        field_bytes(round[0]),
        field_bytes(round[1]),
        field_bytes(round[2]),
    ]
}

fn decode_round(round: &[Vec<u8>; 3]) -> Result<[AkitaField; 3], OpeningsError> {
    Ok([
        field_from_bytes(&round[0])?,
        field_from_bytes(&round[1])?,
        field_from_bytes(&round[2])?,
    ])
}

fn layout_error(error: impl ToString) -> OpeningsError {
    invalid_batch(error.to_string())
}

fn invalid_batch(reason: impl Into<String>) -> OpeningsError {
    OpeningsError::InvalidBatch(reason.into())
}

fn transparent_zk_error() -> OpeningsError {
    invalid_batch("Akita packed batch openings do not support ZK mode yet")
}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "tests assert successful fixture setup")]
mod tests {
    use super::*;

    fn f(value: u64) -> AkitaField {
        AkitaField::from_u64(value)
    }

    #[test]
    fn selector_eval_matches_dense_table_for_offset_family() {
        use crate::{PackedAlphabet, PackedFactDomain, PackedFamilyId, PackedFamilySpec};

        let layout = PackedWitnessLayout::new([
            PackedFamilySpec::direct(
                PackedFamilyId::Custom {
                    namespace: 99,
                    index: 0,
                },
                PackedFactDomain::TraceRows { log_t: 1 },
                1,
                PackedAlphabet::Bit,
            ),
            PackedFamilySpec::direct(
                PackedFamilyId::Custom {
                    namespace: 100,
                    index: 0,
                },
                PackedFactDomain::TraceRows { log_t: 2 },
                2,
                PackedAlphabet::Byte,
            ),
        ])
        .expect("layout should build");
        let term = PackedLinearTerm::new(
            f(7),
            (PackedFamilyId::Custom {
                namespace: 100,
                index: 0,
            })
            .physical_ref(),
            1,
            13,
        )
        .with_row_point(vec![f(3), f(5)]);
        let sumcheck_point_lsb = (0..layout.dimension)
            .map(|index| f(index as u64 + 2))
            .collect::<Vec<_>>();
        let got =
            packed_term_selector_eval(&layout, &term, &sumcheck_point_lsb).expect("selector eval");

        let mut table = vec![AkitaField::zero(); 1usize << layout.dimension];
        let family = family_for_term(&layout, &term).expect("family");
        let row_weights = EqPolynomial::new(term.row_point.clone()).evaluations();
        for (row, row_weight) in row_weights.iter().copied().enumerate() {
            let rank = layout
                .rank(&PackedCellAddress {
                    family: family.id.clone(),
                    row,
                    limb: term.limb,
                    symbol: term.symbol,
                })
                .expect("rank");
            table[rank] = term.coefficient * row_weight;
        }
        let expected = Polynomial::new(table).evaluate(&native_opening_point(&sumcheck_point_lsb));
        assert_eq!(got, expected);
    }
}
