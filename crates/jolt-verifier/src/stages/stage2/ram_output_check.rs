//! The stage 2 `RamOutputCheck` sumcheck instance.
//!
//! Owns the RAM output-check address opening-point derivation and the
//! `EqAddress` / `IoMask` / `ValIo` public-value computation (against the
//! committed public IO memory), in lockstep with the BlindFold constraint's
//! `ram::output_check` formula.
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
}

/// `(EqAddress, IoMask, ValIo)` at the produced output address point — one
/// value per derived multilinear: `eq(output_address_challenges, addr)`, the
/// `[io_start, io_end)` range mask, and the committed public-IO value. Shared
/// by the stage-2 clear path and the BlindFold statement builder so the
/// algebra lives in one place.
pub(crate) fn ram_output_check_publics<F: Field>(
    public_memory: &PublicIoMemory,
    output_address_challenges: &[F],
    ram_output_address: &[F],
) -> Result<(F, F, F), VerifierError> {
    let output_eq =
        try_eq_mle(output_address_challenges, ram_output_address).map_err(public_input_failed)?;
    let output_mask = range_mask_mle_msb(
        public_memory.io_mask_start,
        public_memory.io_mask_end,
        ram_output_address,
    )
    .map_err(public_input_failed)?;
    let io_num_vars = public_memory.io_num_vars();
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
            public_memory
                .segments
                .iter()
                .map(|segment| (segment.start_index, segment.words.as_slice())),
            r_lo,
        );
    Ok((output_eq, output_mask, val_io))
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

    /// Delegates to [`super::phase1_instance_point_offset`] (the phase-1 sub-point
    /// slicing shared with `RamRafEvaluation`).
    fn instance_point_offset(&self, batch_num_vars: usize) -> Result<usize, VerifierError> {
        super::phase1_instance_point_offset(self.read_write_dimensions, self.id(), batch_num_vars)
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
        let (eq_address, io_mask, val_io) = ram_output_check_publics(
            &self.public_memory,
            &self.output_address_challenges,
            output_points.val_final(),
        )?;
        match public_id {
            RamOutputCheckPublic::EqAddress => Ok(eq_address),
            RamOutputCheckPublic::IoMask => Ok(io_mask),
            RamOutputCheckPublic::ValIo => Ok(val_io),
        }
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use jolt_common::jolt_device::{JoltDevice, MemoryConfig};
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
