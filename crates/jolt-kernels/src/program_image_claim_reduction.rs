//! The program-image claim-reduction kernel (stage 6b cycle phase → stage 7
//! address phase): reduces the stage-4 `ProgramImageInitContributionRw`
//! contribution into a final `ProgramImageInit` opening over the shared
//! precommitted schedule.
//!
//! The shared [`PrecommittedReductionKernel`](crate::precommitted_reduction)
//! core runs over the padded program-image word vector as the value table and
//! the SHIFTED eq slice `eq(r_addr_rw, start_index + ·)` (indices wrapping
//! mod the RAM domain) as the eq table — the image occupies the RAM address
//! block starting at `start_index`, so the slice makes `Σ value · eq` exactly
//! the staged stage-4 contribution.

use jolt_claims::protocols::jolt::{PrecommittedReductionLayout, ProgramImageClaimReductionLayout};
use jolt_field::Field;

use crate::committed_program::program_image_words_padded;
use crate::precommitted_reduction::{
    permute_tables, PrecommittedReductionKernel, PrecommittedReductionProver,
};
use crate::views::eq_table;
use crate::{KernelError, ProofSession, ReferenceBackend};

/// The program-image claim-reduction slot. `r_addr_rw` is the stage-2 RAM
/// read-write address point (the staged contribution's point);
/// `bytecode_words` the prover-retained RAM-remapped image words;
/// `start_index` the image block's RAM word offset.
pub trait ProgramImageClaimReduction<F: Field> {
    fn prepare(
        &self,
        session: &mut ProofSession,
        layout: &ProgramImageClaimReductionLayout,
        r_addr_rw: &[F],
        start_index: usize,
        bytecode_words: &[u64],
    ) -> Result<Box<dyn PrecommittedReductionProver<F>>, KernelError<F>>;
}

impl<F: Field> ProgramImageClaimReduction<F> for ReferenceBackend {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        layout: &ProgramImageClaimReductionLayout,
        r_addr_rw: &[F],
        start_index: usize,
        bytecode_words: &[u64],
    ) -> Result<Box<dyn PrecommittedReductionProver<F>>, KernelError<F>> {
        let reduction = layout.precommitted().clone();
        let words = program_image_words_padded(bytecode_words);
        let padded_len = words.len();
        if padded_len != 1usize << reduction.poly_opening_round_permutation_be().len() {
            return Err(KernelError::TableSizeMismatch {
                table: "program image words".to_owned(),
                expected: 1usize << reduction.poly_opening_round_permutation_be().len(),
                got: padded_len,
            });
        }
        let ram_domain = 1usize << r_addr_rw.len();
        if start_index >= ram_domain || padded_len > ram_domain {
            return Err(KernelError::InvalidGeometry {
                reason: format!(
                    "program image block [{start_index}, +{padded_len}) cannot index the RAM domain {ram_domain}"
                ),
            });
        }

        let value: Vec<F> = words.iter().map(|&word| F::from_u64(word)).collect();
        // Addresses wrap mod the RAM domain — the padded block may cross the
        // top of the domain (the verifier's FinalScale DP drops the carry-out
        // the same way; the wrapped tail only ever multiplies padding zeros).
        let eq_ram = eq_table(r_addr_rw);
        let shifted_eq: Vec<F> = (0..padded_len)
            .map(|offset| eq_ram[(start_index + offset) & (ram_domain - 1)])
            .collect();

        let mut permuted = permute_tables(&reduction, vec![value, shifted_eq]).into_iter();
        let (value, eq) = match (permuted.next(), permuted.next()) {
            (Some(value), Some(eq)) => (value, eq),
            _ => {
                return Err(KernelError::Unsupported {
                    reason: "program image table permutation lost the value/eq tables",
                });
            }
        };
        Ok(Box::new(PrecommittedReductionKernel::new(
            reduction,
            value,
            eq,
            Vec::new(),
        )?))
    }
}
