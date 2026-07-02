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

use jolt_claims::protocols::jolt::relations;
pub use jolt_claims::protocols::jolt::relations::ram::{
    RamOutputCheckInputClaims, RamOutputCheckOutputClaims,
};
use jolt_claims::protocols::jolt::{
    geometry::dimensions::ReadWriteDimensions, JoltDerivedId, JoltRelationId, RamOutputCheckPublic,
};
use jolt_claims::{NoChallenges, SymbolicSumcheck};
use jolt_field::Field;
use jolt_poly::{range_mask_mle_msb, sparse_segments_mle_msb, try_eq_mle};
use jolt_program::preprocess::PublicIoMemory;

use crate::stages::relations::ConcreteSumcheck;
use crate::VerifierError;

/// The RAM output check consumes no openings (its input claim is the constant
/// zero), so its consumed-claim *values* struct is empty. (Verifier-side
/// constructor for the moved [`RamOutputCheckInputClaims`].)
pub fn ram_output_check_input_values_from_upstream<F: Field>() -> RamOutputCheckInputClaims<F> {
    RamOutputCheckInputClaims::default()
}

/// The RAM output check consumes no openings, so its consumed-claim *points* struct
/// is empty.
pub fn ram_output_check_input_points_from_upstream<F: Field>() -> RamOutputCheckInputClaims<Vec<F>>
{
    RamOutputCheckInputClaims::default()
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

    /// Complete a two-phase construction: the output-check address reference point
    /// is drawn AFTER the batch's member gammas, so the stage-2 verifier builds
    /// this instance with a placeholder and injects the drawn point here, right
    /// after the draw. Sound because this relation draws no challenges of its own
    /// (`NoChallenges`) and `rounds()`/`degree()` are dims-only, so the placeholder
    /// is never read before injection; a premature `derive_output_term` on the
    /// placeholder fails loudly (`try_eq_mle` length mismatch).
    pub fn set_output_address_challenges(&mut self, output_address_challenges: Vec<F>) {
        self.output_address_challenges = output_address_challenges;
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

    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }

    /// This instance's point is embedded at the batch's phase-1 offset: the active
    /// stage-2 window (the RAM read-write leader's `log_t + log_k` rounds) starts
    /// at `batch_num_vars - (log_t + log_k)`, and this relation joins it after the
    /// leader's `phase1_num_rounds` cycle rounds — the pre-port verifier's
    /// `try_round_offset(log_t + log_k) + phase1_num_rounds()` slicing.
    fn instance_point_offset(&self, batch_num_vars: usize) -> Result<usize, VerifierError> {
        let dimensions = self.read_write_dimensions;
        let window_offset = batch_num_vars
            .checked_sub(dimensions.read_write_rounds())
            .ok_or_else(|| VerifierError::StageClaimSumcheckFailed {
                stage: JoltRelationId::RamOutputCheck,
                reason: format!(
                    "batch challenge vector has {batch_num_vars} entries, fewer than the \
                     active stage-2 window's {} rounds",
                    dimensions.read_write_rounds()
                ),
            })?;
        Ok(window_offset + dimensions.phase1_num_rounds())
    }

    fn derive_opening_points(
        &self,
        sumcheck_point: &[F],
        _input_points: &RamOutputCheckInputClaims<Vec<F>>,
    ) -> Result<RamOutputCheckOutputClaims<Vec<F>>, VerifierError> {
        let address = self
            .read_write_dimensions
            .address_opening_point(sumcheck_point)
            .map_err(public_input_failed)?;
        Ok(RamOutputCheckOutputClaims { val_final: address })
    }

    fn derive_output_term(
        &self,
        id: &JoltDerivedId,
        _input_points: &RamOutputCheckInputClaims<Vec<F>>,
        output_points: &RamOutputCheckOutputClaims<Vec<F>>,
        _challenges: &NoChallenges<F>,
    ) -> Result<F, VerifierError> {
        let JoltDerivedId::RamOutputCheck(public_id) = id else {
            return Err(VerifierError::MissingStageClaimDerived { id: *id });
        };
        let (eq_io_mask, neg_eq_io_mask_val_io) = self.output_publics(output_points.val_final())?;
        match public_id {
            RamOutputCheckPublic::EqIoMask => Ok(eq_io_mask),
            RamOutputCheckPublic::NegEqIoMaskValIo => Ok(neg_eq_io_mask_val_io),
        }
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use common::jolt_device::{JoltDevice, MemoryConfig};
    use jolt_field::Fr;

    /// The `instance_point_offset` override must reproduce the legacy phase-1
    /// slicing `(batch_num_vars - (log_t + log_k)) + phase1_num_rounds` the
    /// pre-port verifier computed via `try_round_offset(log_t + log_k)`.
    #[test]
    fn instance_point_offset_matches_legacy_phase1_formula() {
        for (log_t, log_k, phase1, phase2) in [(4usize, 3usize, 2usize, 1usize), (6, 5, 3, 2)] {
            let dimensions = ReadWriteDimensions::new(log_t, log_k, phase1, phase2);
            let public_memory = PublicIoMemory::new(&JoltDevice::new(&MemoryConfig {
                program_size: Some(1024),
                ..Default::default()
            }))
            .unwrap();
            let relation = RamOutputCheck::<Fr>::new(dimensions, Vec::new(), public_memory);
            // The real batch has `log_t + log_k` variables (the RAM read-write
            // leader); also probe a padded vector.
            for batch_num_vars in [log_t + log_k, log_t + log_k + 5] {
                let legacy = (batch_num_vars - (log_t + log_k)) + phase1;
                let offset = relation.instance_point_offset(batch_num_vars).unwrap();
                assert_eq!(offset, legacy);
                assert_eq!(offset + relation.rounds(), batch_num_vars);
            }
            assert!(relation.instance_point_offset(log_t + log_k - 1).is_err());
        }
    }
}
