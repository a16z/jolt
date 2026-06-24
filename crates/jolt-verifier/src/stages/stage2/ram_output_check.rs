//! The stage 2 `RamOutputCheck` sumcheck instance.
//!
//! A self-contained relation object driven identically by the prover (while
//! producing the stage 2 batch proof) and the verifier (after checking it). It
//! owns the RAM output-check address opening-point derivation and the `EqIoMask` /
//! `NegEqIoMaskValIo` public-value computation (against the committed public IO
//! memory), so the output claim algebra lives here once (and stays in lockstep with
//! the BlindFold constraint, which evaluates the same `ram::output_check` formula).
//!
//! The relation has no input opening; its claimed sum is the constant zero.

use core::marker::PhantomData;

use jolt_claims::protocols::jolt::relations;
use jolt_claims::protocols::jolt::{
    formulas::dimensions::ReadWriteDimensions, JoltOpeningId, JoltPublicId, JoltRelationId,
    RamOutputCheckPublic,
};
use jolt_claims::SymbolicSumcheck;
use jolt_field::Field;
use jolt_poly::{range_mask_mle_msb, sparse_segments_mle_msb, try_eq_mle};
use jolt_program::preprocess::PublicIoMemory;
use jolt_verifier_derive::OutputClaims;
use serde::{Deserialize, Serialize};

use crate::stages::relations::{ConcreteSumcheck, GetPoint, InputClaims, OpeningClaim};
use crate::VerifierError;

/// The produced RAM `val_final` opening, sharing the single output-check opening
/// point. Generic over the cell (`F` on the wire / serialized proof form,
/// `OpeningClaim<F>` on the clear path).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(RamOutputCheck)]
pub struct RamOutputCheckOutputClaims<C> {
    #[opening(RamValFinal)]
    pub val_final: C,
}

/// The RAM output check consumes no openings (its input claim is the constant
/// zero), so this carries only the cell marker. Hand-implements [`InputClaims`]
/// since the derive requires at least one `#[opening]` field.
pub struct RamOutputCheckInputClaims<C> {
    _cell: PhantomData<C>,
}

impl<C> Default for RamOutputCheckInputClaims<C> {
    fn default() -> Self {
        Self { _cell: PhantomData }
    }
}

impl<F: Field> RamOutputCheckInputClaims<OpeningClaim<F>> {
    pub fn from_upstream() -> Self {
        Self::default()
    }
}

impl<F: Field> InputClaims<F> for RamOutputCheckInputClaims<OpeningClaim<F>> {
    fn resolve_input(&self, _id: &JoltOpeningId) -> Option<F> {
        None
    }
}

pub struct RamOutputCheck<F: Field> {
    symbolic: relations::ram::OutputCheck,
    read_write_dimensions: ReadWriteDimensions,
    output_address_challenges: Vec<F>,
    public_memory: PublicIoMemory,
}

impl<F: Field> RamOutputCheck<F> {
    pub fn new(
        read_write_dimensions: ReadWriteDimensions,
        output_address_challenges: Vec<F>,
        public_memory: PublicIoMemory,
    ) -> Self {
        Self {
            symbolic: relations::ram::OutputCheck::new(read_write_dimensions),
            read_write_dimensions,
            output_address_challenges,
            public_memory,
        }
    }

    /// `(EqIoMask, NegEqIoMaskValIo)` at the produced output address point:
    /// `eq_io_mask = eq(output_address_challenges, addr) * range_mask(io, addr)`,
    /// and the negated `eq_io_mask * val_io` term that subtracts the committed
    /// public-IO contribution. Mirrors the BlindFold `ram_output_publics` helper.
    fn output_publics(&self, ram_output_address: &[F]) -> Result<(F, F), VerifierError> {
        let output_eq = try_eq_mle(&self.output_address_challenges, ram_output_address)
            .map_err(public_input_failed)?;
        let output_mask = range_mask_mle_msb(
            self.public_memory.io_mask_start,
            self.public_memory.io_mask_end,
            ram_output_address,
        )
        .map_err(public_input_failed)?;
        let io_num_vars = self.public_memory.io_num_vars();
        let split = ram_output_address
            .len()
            .checked_sub(io_num_vars)
            .ok_or_else(|| {
                public_input_failed(format!(
                    "RAM output address has {} variables but public IO needs {io_num_vars}",
                    ram_output_address.len()
                ))
            })?;
        let (r_hi, r_lo) = ram_output_address.split_at(split);
        let hi_scale = r_hi
            .iter()
            .fold(F::one(), |acc, challenge| acc * (F::one() - *challenge));
        let val_io = hi_scale
            * sparse_segments_mle_msb(
                self.public_memory
                    .segments
                    .iter()
                    .map(|segment| (segment.start_index, segment.words.as_slice())),
                r_lo,
            );
        let eq_io_mask = output_eq * output_mask;
        Ok((eq_io_mask, -eq_io_mask * val_io))
    }
}

fn public_input_failed(reason: impl ToString) -> VerifierError {
    VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::RamOutputCheck,
        reason: reason.to_string(),
    }
}

impl<F: Field> ConcreteSumcheck<F> for RamOutputCheck<F> {
    type Symbolic = relations::ram::OutputCheck;
    type Inputs<C> = RamOutputCheckInputClaims<C>;
    type Outputs<C> = RamOutputCheckOutputClaims<C>;

    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }

    fn derive_opening_points<C: GetPoint<F>>(
        &self,
        sumcheck_point: &[F],
        _inputs: &RamOutputCheckInputClaims<C>,
    ) -> Result<RamOutputCheckOutputClaims<Vec<F>>, VerifierError> {
        let address = self
            .read_write_dimensions
            .address_opening_point(sumcheck_point)
            .map_err(public_input_failed)?;
        Ok(RamOutputCheckOutputClaims { val_final: address })
    }

    fn resolve_public<C: GetPoint<F>>(
        &self,
        id: &JoltPublicId,
        _inputs: &RamOutputCheckInputClaims<C>,
        outputs: Option<&RamOutputCheckOutputClaims<OpeningClaim<F>>>,
    ) -> Result<F, VerifierError> {
        let outputs = outputs.ok_or(VerifierError::MissingStageClaimPublic { id: *id })?;
        let JoltPublicId::RamOutputCheck(public_id) = id else {
            return Err(VerifierError::MissingStageClaimPublic { id: *id });
        };
        let (eq_io_mask, neg_eq_io_mask_val_io) = self.output_publics(outputs.val_final.point())?;
        match public_id {
            RamOutputCheckPublic::EqIoMask => Ok(eq_io_mask),
            RamOutputCheckPublic::NegEqIoMaskValIo => Ok(neg_eq_io_mask_val_io),
        }
    }
}
