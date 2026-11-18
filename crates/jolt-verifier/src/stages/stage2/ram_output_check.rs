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
    RamOutputCheckChallenges, RamOutputCheckInputClaims, RamOutputCheckOutputClaims,
};
use jolt_claims::protocols::jolt::{
    geometry::dimensions::ReadWriteDimensions, JoltDerivedId, JoltRelationId, RamOutputCheckPublic,
};
use jolt_claims::SymbolicSumcheck;
use jolt_field::JoltField;
use jolt_poly::{range_mask_mle_msb, sparse_segments_mle_msb, try_eq_mle};
use jolt_program::preprocess::PublicIoMemory;
use jolt_transcript::FieldTranscript;

use crate::stages::relations::ConcreteSumcheck;
use crate::VerifierError;

#[derive(Clone)]
pub struct RamOutputCheck<F: JoltField> {
    symbolic: relations::ram::OutputCheck,
    read_write_dimensions: ReadWriteDimensions,
    public_memory: PublicIoMemory,
    _field: core::marker::PhantomData<F>,
}

impl<F: JoltField> RamOutputCheck<F> {
    pub fn new(read_write_dimensions: ReadWriteDimensions, public_memory: PublicIoMemory) -> Self {
        Self {
            symbolic: relations::ram::OutputCheck::new(read_write_dimensions),
            read_write_dimensions,
            public_memory,
            _field: core::marker::PhantomData,
        }
    }
}

/// `(EqAddress, IoMask, ValIo)` at the produced output address point — one
/// value per derived multilinear: `eq(output_address_challenges, addr)`, the
/// `[input_start, output_end)` range mask (zero when the layout has no input or
/// output capacity), and the committed public-IO value. Shared by the stage-2
/// clear path and the BlindFold statement builder so the algebra lives in one
/// place.
pub(crate) fn ram_output_check_publics<F: JoltField>(
    public_memory: &PublicIoMemory,
    output_address_challenges: &[F],
    ram_output_address: &[F],
) -> Result<(F, F, F), VerifierError> {
    let output_eq =
        try_eq_mle(output_address_challenges, ram_output_address).map_err(public_input_failed)?;
    let output_mask = if public_memory.io_mask_start == public_memory.io_mask_end {
        F::zero()
    } else {
        range_mask_mle_msb(
            public_memory.io_mask_start,
            public_memory.io_mask_end,
            ram_output_address,
        )
        .map_err(public_input_failed)?
    };
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

impl<F: JoltField> RamOutputCheck<F> {
    pub fn read_write_dimensions(&self) -> ReadWriteDimensions {
        self.read_write_dimensions
    }

    pub fn public_memory(&self) -> &PublicIoMemory {
        &self.public_memory
    }
}

impl<F: JoltField> ConcreteSumcheck<F> for RamOutputCheck<F> {
    type Symbolic = relations::ram::OutputCheck;

    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }

    /// Draws the output-check address reference point: one challenge per RAM
    /// address variable. This member is declared last in `Stage2BatchSumchecks`,
    /// so the generated aggregate draw lands the vector exactly where the legacy
    /// stage front drew it — after both batch gammas.
    ///
    /// Each coordinate is its own one-challenge codec message; one
    /// `challenge_vector` message would squeeze different bytes.
    fn draw_challenges<T: FieldTranscript<F>>(
        &self,
        transcript: &mut T,
    ) -> Result<RamOutputCheckChallenges<F>, VerifierError> {
        Ok(RamOutputCheckChallenges {
            output_address: (0..self.read_write_dimensions.log_k())
                .map(|_| transcript.challenge())
                .collect::<Result<_, _>>()?,
        })
    }

    /// Delegates to `super::phase1_instance_point_offset` (the phase-1 sub-point
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
        challenges: &RamOutputCheckChallenges<F>,
    ) -> Result<F, VerifierError> {
        let JoltDerivedId::RamOutputCheck(public_id) = id else {
            return Err(VerifierError::MissingStageClaimDerived { id: (*id).into() });
        };
        let (eq_address, io_mask, val_io) = ram_output_check_publics(
            &self.public_memory,
            &challenges.output_address,
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
    use common::constants::RAM_START_ADDRESS;
    use common::jolt_device::{JoltDevice, MemoryConfig};
    use jolt_field::{Fr, One, Zero};
    use jolt_poly::boolean_point_msb;

    /// The IoMask is one from `input_start` through the last output word and
    /// zero from `io_end`, the first I/O gap word, through the last word below
    /// RAM. Without input or output capacity the mask is empty and zero.
    #[test]
    fn io_mask_covers_inputs_and_outputs_but_not_the_io_gap() {
        let device = |max_input_size, max_output_size| {
            JoltDevice::new(&MemoryConfig {
                program_size: Some(1024),
                max_trusted_advice_size: 0,
                max_untrusted_advice_size: 0,
                max_input_size,
                max_output_size,
                ..Default::default()
            })
        };
        let io_mask = |device: &JoltDevice, address: u64| {
            let word = device.memory_layout.remapped_word_address(address).unwrap();
            let point = boolean_point_msb::<Fr>(4, usize::try_from(word).unwrap());
            let public_memory = PublicIoMemory::new(device).unwrap();
            ram_output_check_publics(&public_memory, &point, &point)
                .unwrap()
                .1
        };
        let io = device(16, 16);
        let layout = &io.memory_layout;
        assert_eq!(io_mask(&io, layout.input_start), Fr::one());
        assert_eq!(io_mask(&io, layout.output_end - 1), Fr::one());
        assert_eq!(io_mask(&io, layout.io_end), Fr::zero());
        assert_eq!(io_mask(&io, RAM_START_ADDRESS - 1), Fr::zero());

        let no_io = device(0, 0);
        assert_eq!(io_mask(&no_io, no_io.memory_layout.input_start), Fr::zero());
    }

    /// The `instance_point_offset` override must place the relation's own
    /// rounds exactly at the batch tail, and reject batch vectors shorter
    /// than the active stage-2 window.
    #[test]
    fn instance_point_offset_spans_the_batch_tail() {
        for (log_t, log_k, phase1, phase2) in [(4usize, 3usize, 2usize, 1usize), (6, 5, 3, 2)] {
            let dimensions = ReadWriteDimensions::new(log_t, log_k, phase1, phase2);
            let public_memory = PublicIoMemory::new(&JoltDevice::new(&MemoryConfig {
                program_size: Some(1024),
                ..Default::default()
            }))
            .unwrap();
            let relation = RamOutputCheck::<Fr>::new(dimensions, public_memory);
            // The real batch has `log_t + log_k` variables (the RAM read-write
            // leader); also probe a padded vector.
            for batch_num_vars in [log_t + log_k, log_t + log_k + 5] {
                let offset = relation.instance_point_offset(batch_num_vars).unwrap();
                assert_eq!(offset + relation.rounds(), batch_num_vars);
            }
            assert!(relation.instance_point_offset(log_t + log_k - 1).is_err());
        }
    }
}
