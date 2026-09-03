//! Optimized precommitted claim-reduction family: the stage-6b cycle phases
//! and stage-7 address phases of the trusted/untrusted advice, committed
//! bytecode, and program-image reductions, plus the stage-4 advice opening
//! evaluation — byte-parity twins of the reference kernels in
//! [`crate::reference::advice_claim_reduction`],
//! [`crate::reference::bytecode_claim_reduction`],
//! [`crate::reference::program_image_claim_reduction`], and
//! [`crate::reference::precommitted_reduction`].
//!
//! The phase kernels themselves are the shared
//! [`crate::precommitted_reduction`] pair — the round loops, binds, and
//! table permutes there run threshold-gated rayon paths (the legacy
//! `PrecommittedProver` parallelism), so both tiers share one copy of the
//! subtle padded-claim round algebra and the 6b→7 carry stays the one
//! [`PrecommittedReductionCarry`] type: either tier's cycle kernel parks a
//! carry the other tier's address slot reclaims, and mixed-tier composition
//! holds by construction. What this module owns is the PREPARE side — the
//! table builders (ported from
//! `jolt-prover-legacy/src/zkvm/claim_reductions/{advice,bytecode,program_image}.rs`):
//!
//! - **Advice** (`AdviceClaimReductionProver::initialize`): the eq table is
//!   built from the LSB-permuted challenges directly (one parallel
//!   `EqPolynomial::evals`, no coefficient permute), and only the advice
//!   coefficient table pays the parallel permute-gather.
//! - **Program image** (`shifted_program_image_eq_slice`): the shifted eq
//!   slice `eq(r_addr_rw, start_index + ·)` is assembled from maximal
//!   aligned blocks (`EqPolynomial::evals_for_max_aligned_block`, wrap-aware)
//!   — `O(padded_len)` work instead of materializing the full
//!   `2^|r_addr_rw|` RAM-domain eq table the reference tier gathers from.
//!   The word vector permutes as raw `u64`s and converts to field elements
//!   in one parallel pass.
//! - **Bytecode** (`BytecodeClaimReductionProver::initialize`): the per-chunk
//!   coefficient grids build in parallel (one independent
//!   [`build_committed_bytecode_chunk_coeffs`] call per chunk over its own
//!   instruction slice — identical accumulation order per chunk), and the
//!   chunk-weight value fold and lane-weight eq template are parallel
//!   per-index maps (legacy folds the value grid with `into_par_iter`).
//!
//! Every construction is a rearrangement of exact field operations, so the
//! built tables — and through the shared kernels the round polynomials and
//! output claims — are byte-identical to the reference tier's.

use std::marker::PhantomData;

use jolt_claims::protocols::jolt::geometry::claim_reductions::advice::ram_val_check_advice_opening;
use jolt_claims::protocols::jolt::{
    AdviceClaimReductionLayout, BytecodeClaimReductionLayout, JoltAdviceKind, JoltChallengeId,
    PrecommittedReductionLayout, ProgramImageClaimReductionLayout,
};
use jolt_claims::{InputClaims, OutputClaims, SumcheckChallenges};
use jolt_field::JoltField;
use jolt_poly::EqPolynomial;
use jolt_riscv::JoltInstructionRow;
use jolt_verifier::stages::relations::{
    ConcreteSumcheck, ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckOutputClaims,
};
use jolt_verifier::stages::stage6b::committed_reduction_cycle_phase::{
    BytecodeReductionCyclePhase, ProgramImageReductionCyclePhase, TrustedAdviceCyclePhase,
    UntrustedAdviceCyclePhase,
};
use jolt_verifier::stages::stage6b::outputs::BytecodeReductionWeights;
use jolt_witness::{JoltWitnessOracle, JoltWitnessPlane};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::support::eq_table;
use crate::committed_program::{
    build_committed_bytecode_chunk_coeffs, chunk_index_to_lane_cycle, program_image_words_padded,
};
use crate::opening::AdviceOpeningEvaluation;
use crate::precommitted_reduction::{
    lsb_permutation, permute_challenges, permute_coefficients, permute_tables,
    AddressReductionKernel, CycleReductionKernel, PrecommittedReductionCarry,
};
use crate::{KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel};

/// Tables at least this large build in parallel; below it rayon dispatch
/// costs more than the work.
#[cfg(feature = "parallel")]
const PAR_THRESHOLD: usize = 1 << 10;

/// The precommitted cycle phases and the advice opening evaluation:
/// `PrepareKernel` front of the four stage-6b cycle-phase slots plus the
/// stage-4 `AdviceOpeningEvaluation` slot.
pub struct OptimizedPrecommittedCycle;

/// The stage-7 slot server for a precommitted address-phase relation `R`:
/// reclaims the carry the cycle kernel parked under `R`'s key — the same
/// [`PrecommittedReductionCarry`] the reference tier parks and reclaims, so
/// either tier's stage 6b feeds either tier's stage 7 — and mounts the
/// shared final-opening batch member. Same missing-carry contract as
/// [`ReferencePrecommittedAddress`](crate::reference::precommitted_reduction::ReferencePrecommittedAddress).
pub struct OptimizedPrecommittedAddress<R> {
    missing_carry: &'static str,
    _relation: PhantomData<fn() -> R>,
}

impl<R> OptimizedPrecommittedAddress<R> {
    pub fn new(missing_carry: &'static str) -> Self {
        Self {
            missing_carry,
            _relation: PhantomData,
        }
    }
}

impl<F, R> PrepareKernel<F, R> for OptimizedPrecommittedAddress<R>
where
    F: JoltField,
    R: ConcreteSumcheck<F> + 'static,
    AddressReductionKernel<F, R>: SumcheckKernel<F, Relation = R>,
    SumcheckInputClaims<F, R>: InputClaims<F>,
    SumcheckOutputClaims<F, R>: OutputClaims<F>,
    ConcreteSumcheckChallenges<F, R>: SumcheckChallenges<F, JoltChallengeId>,
{
    fn prepare(
        &self,
        session: &mut ProofSession,
        _witness: &dyn JoltWitnessPlane<F>,
        _inputs: ProverInputs<'_, F, R>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = R>>, KernelError<F>> {
        let carry = session.take::<PrecommittedReductionCarry<F, R>>().ok_or(
            KernelError::InvariantViolation {
                reason: self.missing_carry,
            },
        )?;
        Ok(Box::new(AddressReductionKernel::new(carry)))
    }
}

// ---------------------------------------------------------------- advice

impl<F: JoltField> AdviceOpeningEvaluation<F> for OptimizedPrecommittedCycle {
    #[tracing::instrument(skip_all, name = "OptimizedAdviceOpeningEvaluation::evaluate", fields(kind = ?kind))]
    fn evaluate(
        &self,
        _session: &mut ProofSession,
        kind: JoltAdviceKind,
        point: &[F],
        witness: &dyn JoltWitnessOracle<F>,
    ) -> Result<F, KernelError<F>> {
        let table = advice_table(witness, kind, point.len())?;
        let eq = eq_table(point);
        #[cfg(feature = "parallel")]
        if table.len() >= PAR_THRESHOLD {
            return Ok(table
                .par_iter()
                .zip(eq)
                .map(|(value, weight)| *value * weight)
                .sum());
        }
        Ok(table
            .iter()
            .zip(&eq)
            .map(|(value, weight)| *value * *weight)
            .sum())
    }
}

impl<F: JoltField> PrepareKernel<F, TrustedAdviceCyclePhase<F>> for OptimizedPrecommittedCycle {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, TrustedAdviceCyclePhase<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = TrustedAdviceCyclePhase<F>>>, KernelError<F>>
    {
        let r_val =
            inputs
                .relation
                .reference_opening_point()
                .ok_or(KernelError::InvariantViolation {
                    reason: "trusted-advice cycle phase carries no reference opening point",
                })?;
        Ok(Box::new(advice_reduction_kernel::<
            F,
            TrustedAdviceCyclePhase<F>,
        >(
            JoltAdviceKind::Trusted,
            inputs.relation.layout(),
            r_val,
            witness,
        )?))
    }
}

impl<F: JoltField> PrepareKernel<F, UntrustedAdviceCyclePhase<F>> for OptimizedPrecommittedCycle {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, UntrustedAdviceCyclePhase<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = UntrustedAdviceCyclePhase<F>>>, KernelError<F>>
    {
        let r_val =
            inputs
                .relation
                .reference_opening_point()
                .ok_or(KernelError::InvariantViolation {
                    reason: "untrusted-advice cycle phase carries no reference opening point",
                })?;
        Ok(Box::new(advice_reduction_kernel::<
            F,
            UntrustedAdviceCyclePhase<F>,
        >(
            JoltAdviceKind::Untrusted,
            inputs.relation.layout(),
            r_val,
            witness,
        )?))
    }
}

/// The advice reduction's cycle-phase kernel: the advice polynomial as the
/// value table and the eq table of the staged RAM value-check point, both in
/// Dory opening-round order. The eq table is built from the permuted
/// challenges directly — the coefficient permute and the challenge permute
/// are the same LSB relabeling, so `permuted_table[i] · permuted_eq[i]` pairs
/// exactly as the unpermuted product did.
fn advice_reduction_kernel<F: JoltField, R>(
    kind: JoltAdviceKind,
    layout: &AdviceClaimReductionLayout,
    r_val: &[F],
    witness: &dyn JoltWitnessPlane<F>,
) -> Result<CycleReductionKernel<F, R>, KernelError<F>> {
    let reduction = layout.precommitted().clone();
    let permutation = reduction.poly_opening_round_permutation_be();
    if r_val.len() != permutation.len() {
        return Err(KernelError::InvalidGeometry {
            reason: format!(
                "advice reference point has {} variables, schedule expects {}",
                r_val.len(),
                permutation.len()
            ),
        });
    }
    let table = advice_table(witness, kind, permutation.len())?;
    let (value, eq) = match lsb_permutation(permutation) {
        Some(old_lsb_to_new_lsb) => (
            permute_coefficients(&table, &old_lsb_to_new_lsb),
            eq_table(&permute_challenges(r_val, &old_lsb_to_new_lsb)),
        ),
        None => (table, eq_table(r_val)),
    };
    CycleReductionKernel::new(reduction, value, eq, Vec::new())
}

fn advice_table<F: JoltField>(
    witness: &dyn JoltWitnessOracle<F>,
    kind: JoltAdviceKind,
    expected_vars: usize,
) -> Result<Vec<F>, KernelError<F>> {
    let table = witness.oracle_table(ram_val_check_advice_opening(kind).polynomial_id())?;
    if table.len() != 1usize << expected_vars {
        return Err(KernelError::TableSizeMismatch {
            table: format!("{kind:?} advice"),
            expected: 1usize << expected_vars,
            got: table.len(),
        });
    }
    Ok(table)
}

// --------------------------------------------------------- program image

impl<F: JoltField> PrepareKernel<F, ProgramImageReductionCyclePhase<F>>
    for OptimizedPrecommittedCycle
{
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, ProgramImageReductionCyclePhase<F>>,
    ) -> Result<
        Box<dyn SumcheckKernel<F, Relation = ProgramImageReductionCyclePhase<F>>>,
        KernelError<F>,
    > {
        let layout = inputs.relation.layout();
        let program = witness.program_preprocessing();
        Ok(Box::new(program_image_reduction_kernel(
            layout,
            inputs.relation.r_addr_rw(),
            layout.start_index(),
            &program.ram.bytecode_words,
        )?))
    }
}

/// The program-image reduction's cycle-phase kernel: the padded word vector
/// (permuted as raw `u64`s, converted in one parallel pass) against the
/// blocked shifted eq slice.
fn program_image_reduction_kernel<F: JoltField>(
    layout: &ProgramImageClaimReductionLayout,
    r_addr_rw: &[F],
    start_index: usize,
    bytecode_words: &[u64],
) -> Result<CycleReductionKernel<F, ProgramImageReductionCyclePhase<F>>, KernelError<F>> {
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

    let shifted_eq = shifted_eq_slice(r_addr_rw, start_index, padded_len);
    let (words, shifted_eq) = match lsb_permutation(reduction.poly_opening_round_permutation_be()) {
        Some(old_lsb_to_new_lsb) => (
            permute_coefficients(&words, &old_lsb_to_new_lsb),
            permute_coefficients(&shifted_eq, &old_lsb_to_new_lsb),
        ),
        None => (words, shifted_eq),
    };
    let value = convert_words(&words);
    CycleReductionKernel::new(reduction, value, shifted_eq, Vec::new())
}

/// `eq(r_addr, start_index + offset)` for `offset < len`, indices wrapping mod
/// the RAM domain (the reference tier gathers the same entries out of the full
/// domain table; the wrapped tail only ever multiplies padding zeros but its
/// entries still enter the bound tables, so they must match exactly).
/// Assembled from maximal aligned power-of-two blocks — `O(len)` total work,
/// never the `O(2^|r_addr|)` full table.
fn shifted_eq_slice<F: JoltField>(r_addr: &[F], start_index: usize, len: usize) -> Vec<F> {
    let ram_domain = 1usize << r_addr.len();
    let mut out = Vec::with_capacity(len);
    let mut index = start_index & (ram_domain - 1);
    let mut remaining = len;
    while remaining > 0 {
        // Cap each block at the domain top so alignment never crosses the
        // wrap; a wrapped continuation restarts at index 0 (aligned to
        // everything).
        let span = remaining.min(ram_domain - index);
        let (block_size, block_evals) =
            EqPolynomial::<F>::evals_for_max_aligned_block(r_addr, index, span);
        out.extend(block_evals);
        index = (index + block_size) & (ram_domain - 1);
        remaining -= block_size;
    }
    out
}

/// Parallel `u64 → F` conversion of the (already permuted) word vector.
fn convert_words<F: JoltField>(words: &[u64]) -> Vec<F> {
    #[cfg(feature = "parallel")]
    if words.len() >= PAR_THRESHOLD {
        return words.par_iter().map(|&word| F::from_u64(word)).collect();
    }
    words.iter().map(|&word| F::from_u64(word)).collect()
}

// -------------------------------------------------------------- bytecode

impl<F: JoltField> PrepareKernel<F, BytecodeReductionCyclePhase<F>> for OptimizedPrecommittedCycle {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, BytecodeReductionCyclePhase<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = BytecodeReductionCyclePhase<F>>>, KernelError<F>>
    {
        let program = witness.program_preprocessing();
        Ok(Box::new(bytecode_reduction_kernel(
            inputs.relation.layout(),
            inputs.relation.weights(),
            &program.bytecode.bytecode,
        )?))
    }
}

/// The committed-bytecode reduction's cycle-phase kernel — the chunk-weight
/// value fold over the parallel-built per-chunk grids, the lane-weight eq
/// template, and the raw grids as aux tables (their fully bound coefficients
/// are the final per-chunk openings).
fn bytecode_reduction_kernel<F: JoltField>(
    layout: &BytecodeClaimReductionLayout,
    weights: &BytecodeReductionWeights<F>,
    bytecode: &[JoltInstructionRow],
) -> Result<CycleReductionKernel<F, BytecodeReductionCyclePhase<F>>, KernelError<F>> {
    let reduction = layout.precommitted().clone();
    let chunk_coeffs = parallel_chunk_coeffs(bytecode, layout.chunk_count(), layout)?;
    let chunk_len = chunk_coeffs[0].len();
    if chunk_len != 1usize << reduction.poly_opening_round_permutation_be().len() {
        return Err(KernelError::TableSizeMismatch {
            table: "committed bytecode chunk grid".to_owned(),
            expected: 1usize << reduction.poly_opening_round_permutation_be().len(),
            got: chunk_len,
        });
    }
    if weights.chunk_rbc_weights.len() != chunk_coeffs.len() {
        return Err(KernelError::TableSizeMismatch {
            table: "bytecode chunk weights".to_owned(),
            expected: chunk_coeffs.len(),
            got: weights.chunk_rbc_weights.len(),
        });
    }

    let chunk_cycle_len = 1usize << layout.log_bytecode_chunk_size();
    let eq_cycle = eq_table(&weights.r_bc);
    let eq_entry = |index: usize| -> F {
        let (lane, cycle) = chunk_index_to_lane_cycle(index, chunk_cycle_len, layout.trace_order());
        weights.lane_weights[lane] * eq_cycle[cycle]
    };
    let value_entry = |index: usize| -> F {
        chunk_coeffs
            .iter()
            .zip(&weights.chunk_rbc_weights)
            .map(|(coeffs, weight)| coeffs[index] * *weight)
            .sum()
    };
    #[cfg(feature = "parallel")]
    let (eq_template, value): (Vec<F>, Vec<F>) = if chunk_len >= PAR_THRESHOLD {
        (
            (0..chunk_len).into_par_iter().map(eq_entry).collect(),
            (0..chunk_len).into_par_iter().map(value_entry).collect(),
        )
    } else {
        (
            (0..chunk_len).map(eq_entry).collect(),
            (0..chunk_len).map(value_entry).collect(),
        )
    };
    #[cfg(not(feature = "parallel"))]
    let (eq_template, value): (Vec<F>, Vec<F>) = (
        (0..chunk_len).map(eq_entry).collect(),
        (0..chunk_len).map(value_entry).collect(),
    );

    let mut tables = Vec::with_capacity(2 + chunk_coeffs.len());
    tables.push(value);
    tables.push(eq_template);
    tables.extend(chunk_coeffs);
    let mut permuted = permute_tables(&reduction, tables).into_iter();
    let (value, eq) = match (permuted.next(), permuted.next()) {
        (Some(value), Some(eq)) => (value, eq),
        _ => {
            return Err(KernelError::InvariantViolation {
                reason: "bytecode reduction table permutation lost the value/eq tables",
            });
        }
    };
    CycleReductionKernel::new(reduction, value, eq, permuted.collect())
}

/// The per-chunk committed bytecode grids, one independent build per chunk in
/// parallel: each chunk's rows are a contiguous instruction slice and the
/// grid indexing is chunk-local, so a single-chunk build over the slice is
/// coefficient-identical to that chunk of the full build.
fn parallel_chunk_coeffs<F: JoltField>(
    bytecode: &[JoltInstructionRow],
    chunk_count: usize,
    layout: &BytecodeClaimReductionLayout,
) -> Result<Vec<Vec<F>>, KernelError<F>> {
    if chunk_count == 0 || !bytecode.len().is_multiple_of(chunk_count) {
        return Err(KernelError::InvalidGeometry {
            reason: format!(
                "invalid committed bytecode chunking: {chunk_count} chunks over {} rows",
                bytecode.len()
            ),
        });
    }
    let chunk_cycle_len = bytecode.len() / chunk_count;
    let build = |chunk: usize| -> Result<Vec<F>, KernelError<F>> {
        let slice = &bytecode[chunk * chunk_cycle_len..(chunk + 1) * chunk_cycle_len];
        build_committed_bytecode_chunk_coeffs(slice, 1, layout.trace_order())?
            .into_iter()
            .next()
            .ok_or(KernelError::InvariantViolation {
                reason: "single-chunk bytecode grid build produced no grid",
            })
    };
    #[cfg(feature = "parallel")]
    {
        (0..chunk_count).into_par_iter().map(build).collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        (0..chunk_count).map(build).collect()
    }
}

/// Byte parity against the reference kernels over a custom trace backend
/// (advice enabled with nonzero device bytes, multi-row committed bytecode,
/// nonzero program-image words — none of which the shared sample fixture
/// carries). Each kind drives the full production shape: cycle-phase
/// lockstep, `park_residue` into a per-pipeline session, stage-7 reclaim,
/// address-phase lockstep. The stage-7 kernels are prepared CROSS-TIER (the
/// reference slot reclaims the optimized pipeline's carry and vice versa),
/// so the mixed-tier composition promise — either tier's stage 6b feeds
/// either tier's stage 7 — is the very thing the address-phase parity pins.
#[cfg(all(test, not(feature = "akita")))]
#[expect(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "test module"
)]
mod tests {
    use common::jolt_device::{JoltDevice, MemoryLayout};
    use jolt_claims::protocols::jolt::{JoltOneHotConfig, TracePolynomialOrder};
    use jolt_field::{Fr, Ring};
    use jolt_program::execution::{JoltProgram, OwnedTrace, TraceOutput, TraceRow};
    use jolt_program::preprocess::{
        BytecodePreprocessing, JoltProgramPreprocessing, RAMPreprocessing,
    };
    use jolt_riscv::{JoltInstructionKind, JoltInstructionRow, NormalizedOperands, RV64IMAC_JOLT};
    use jolt_verifier::stages::relations::SumcheckInputPoints;
    use jolt_verifier::stages::stage6b::committed_reduction_cycle_phase::BytecodeReductionCyclePhaseChallenges;
    use jolt_verifier::stages::stage7::advice_address_phase::{
        TrustedAdviceAddressPhase, UntrustedAdviceAddressPhase,
    };
    use jolt_verifier::stages::stage7::committed_reduction_address_phase::{
        BytecodeReductionAddressPhase, ProgramImageReductionAddressPhase,
    };
    use jolt_verifier::stages::{CommittedProgramSchedule, PrecommittedSchedule};
    use jolt_witness::{JoltVmWitnessConfig, JoltVmWitnessInputs, ProgramSource, TraceBackend};

    use super::*;
    use crate::optimized::parity::{run_lockstep, synthetic_point};
    use crate::reference::precommitted_reduction::ReferencePrecommittedAddress;
    use crate::ReferenceBackend;

    const LOG_T: usize = 2;
    const LOG_K_CHUNK: usize = 4;
    /// 16 advice words (4 variables).
    const TRUSTED_ADVICE_MAX_BYTES: usize = 128;
    /// 8 advice words (3 variables).
    const UNTRUSTED_ADVICE_MAX_BYTES: usize = 64;
    const BYTECODE_CHUNK_COUNT: usize = 2;
    const IMAGE_START_INDEX: usize = 3;
    /// RAM address point length for the program-image relation: domain 32
    /// comfortably holds the 8-word image block at `IMAGE_START_INDEX`.
    const IMAGE_RAM_VARS: usize = 5;

    const MISSING_TRUSTED: &str =
        "stage 6b parked no trusted-advice reduction state for the scheduled address phase";
    const MISSING_UNTRUSTED: &str =
        "stage 6b parked no untrusted-advice reduction state for the scheduled address phase";
    const MISSING_BYTECODE: &str =
        "stage 6b parked no bytecode reduction state for the scheduled address phase";
    const MISSING_PROGRAM_IMAGE: &str =
        "stage 6b parked no program-image reduction state for the scheduled address phase";

    fn fr(value: u64) -> Fr {
        Fr::from_u64(value)
    }

    fn with_fixture<R>(
        trace_order: TracePolynomialOrder,
        f: impl FnOnce(&TraceBackend<OwnedTrace>, &PrecommittedSchedule) -> R,
    ) -> R {
        let instructions: Vec<JoltInstructionRow> = (0..3u64)
            .map(|index| JoltInstructionRow {
                instruction_kind: JoltInstructionKind::ADDI,
                address: (0x8000_0000u64 + 4 * index) as usize,
                operands: NormalizedOperands {
                    rd: Some(1 + index as u8),
                    rs1: Some(2),
                    rs2: None,
                    imm: 3 + index as i128,
                },
                virtual_sequence_remaining: None,
                is_first_in_sequence: false,
                is_compressed: false,
            })
            .collect();
        let memory_layout = MemoryLayout {
            max_trusted_advice_size: TRUSTED_ADVICE_MAX_BYTES as u64,
            max_untrusted_advice_size: UNTRUSTED_ADVICE_MAX_BYTES as u64,
            ..Default::default()
        };
        use std::sync::Arc;
        let preprocessing = Arc::new(JoltProgramPreprocessing {
            bytecode: BytecodePreprocessing::preprocess(
                instructions,
                0x8000_0000u64,
                RV64IMAC_JOLT,
            )
            .unwrap(),
            ram: RAMPreprocessing {
                min_bytecode_address: 0x8000_0000,
                bytecode_words: vec![7, 11, 13, 17, 19],
            },
            memory_layout: memory_layout.clone(),
            max_padded_trace_length: 1 << LOG_T,
        });
        let program = Arc::new(JoltProgram::default());
        let rows = vec![TraceRow::default(), TraceRow::default()];
        let device = JoltDevice {
            trusted_advice: (1..=24).collect(),
            untrusted_advice: (101..=116).collect(),
            memory_layout,
            ..Default::default()
        };
        let config = JoltVmWitnessConfig::new(
            LOG_T,
            64,
            JoltOneHotConfig {
                log_k_chunk: LOG_K_CHUNK as u8,
                lookups_ra_virtual_log_k_chunk: 16,
            },
        )
        .include_trusted_advice(true)
        .include_untrusted_advice(true);
        let inputs = JoltVmWitnessInputs::new(
            &program,
            &preprocessing,
            TraceOutput::new(OwnedTrace::new(rows), device, None, None),
        );
        let backend = TraceBackend::new(config, inputs);

        let bytecode_len = backend.program_preprocessing().bytecode.bytecode.len();
        assert!(
            bytecode_len.is_power_of_two() && bytecode_len.is_multiple_of(BYTECODE_CHUNK_COUNT),
            "fixture bytecode length {bytecode_len} defeats the chunking"
        );
        let program_image_len_words =
            program_image_words_padded(&backend.program_preprocessing().ram.bytecode_words).len();
        let schedule = PrecommittedSchedule::new(
            trace_order,
            LOG_T,
            LOG_K_CHUNK,
            Some(TRUSTED_ADVICE_MAX_BYTES),
            Some(UNTRUSTED_ADVICE_MAX_BYTES),
            Some(CommittedProgramSchedule {
                bytecode_len,
                bytecode_chunk_count: BYTECODE_CHUNK_COUNT,
                program_image_len_words,
                program_image_start_index: IMAGE_START_INDEX,
            }),
        )
        .unwrap();
        f(&backend, &schedule)
    }

    /// Both phases of one kind in lockstep: cycle-phase parity, a
    /// `park_residue` into each pipeline's session, cross-tier stage-7
    /// reclaim, address-phase parity. Kind-specific pieces (relations, claim
    /// extraction) stay with the callers; this drives the shared shape.
    struct PhasePair<'a, RC: ConcreteSumcheck<Fr>, RA: ConcreteSumcheck<Fr>>
    where
        SumcheckInputClaims<Fr, RC>: InputClaims<Fr>,
        SumcheckOutputClaims<Fr, RC>: OutputClaims<Fr>,
        ConcreteSumcheckChallenges<Fr, RC>: SumcheckChallenges<Fr, JoltChallengeId>,
        SumcheckInputClaims<Fr, RA>: InputClaims<Fr>,
        SumcheckOutputClaims<Fr, RA>: OutputClaims<Fr>,
        ConcreteSumcheckChallenges<Fr, RA>: SumcheckChallenges<Fr, JoltChallengeId>,
    {
        backend: &'a TraceBackend<OwnedTrace>,
        cycle_relation: &'a RC,
        cycle_claims: &'a SumcheckInputClaims<Fr, RC>,
        cycle_points: &'a SumcheckInputPoints<Fr, RC>,
        cycle_challenges_struct: &'a ConcreteSumcheckChallenges<Fr, RC>,
        address_relation: &'a RA,
        address_claims: &'a SumcheckInputClaims<Fr, RA>,
        address_points: &'a SumcheckInputPoints<Fr, RA>,
        address_challenges_struct: &'a ConcreteSumcheckChallenges<Fr, RA>,
        missing_carry: &'static str,
    }

    impl<RC, RA> PhasePair<'_, RC, RA>
    where
        RC: ConcreteSumcheck<Fr>,
        RA: ConcreteSumcheck<Fr> + 'static,
        SumcheckInputClaims<Fr, RC>: InputClaims<Fr>,
        SumcheckOutputClaims<Fr, RC>: OutputClaims<Fr> + PartialEq + core::fmt::Debug,
        ConcreteSumcheckChallenges<Fr, RC>: SumcheckChallenges<Fr, JoltChallengeId>,
        SumcheckInputClaims<Fr, RA>: InputClaims<Fr>,
        SumcheckOutputClaims<Fr, RA>: OutputClaims<Fr> + PartialEq + core::fmt::Debug,
        ConcreteSumcheckChallenges<Fr, RA>: SumcheckChallenges<Fr, JoltChallengeId>,
        ReferenceBackend: PrepareKernel<Fr, RC>,
        OptimizedPrecommittedCycle: PrepareKernel<Fr, RC>,
        AddressReductionKernel<Fr, RA>: SumcheckKernel<Fr, Relation = RA>,
    {
        /// Run the pair; `intermediate` extracts the staged handoff claim
        /// (the address phase's standalone input claim) from the cycle
        /// output claims. Returns the final address-phase output claims of
        /// the reference pipeline for kind-specific scrutiny.
        fn run(
            &self,
            cycle_rounds: usize,
            address_rounds: usize,
            intermediate: impl Fn(&SumcheckOutputClaims<Fr, RC>) -> Fr,
            seed: u64,
        ) -> SumcheckOutputClaims<Fr, RA> {
            let cycle_inputs = || ProverInputs {
                relation: self.cycle_relation,
                claims: self.cycle_claims,
                points: self.cycle_points,
                challenges: self.cycle_challenges_struct,
            };
            let mut session_ref = ProofSession::default();
            let mut session_opt = ProofSession::default();
            let mut reference = <ReferenceBackend as PrepareKernel<Fr, RC>>::prepare(
                &ReferenceBackend,
                &mut session_ref,
                self.backend,
                cycle_inputs(),
            )
            .unwrap();
            let mut optimized = <OptimizedPrecommittedCycle as PrepareKernel<Fr, RC>>::prepare(
                &OptimizedPrecommittedCycle,
                &mut session_opt,
                self.backend,
                cycle_inputs(),
            )
            .unwrap();

            // The honest standalone input claim off a throwaway reference
            // kernel: with an address phase scheduled, `output_claims` stages
            // the intermediate `Σ value·eq·scale`, which before any binding
            // IS the input claim. (These kernels never round-check, so the
            // harness's zero-claim probe cannot recover it.)
            let mut throwaway = <ReferenceBackend as PrepareKernel<Fr, RC>>::prepare(
                &ReferenceBackend,
                &mut ProofSession::default(),
                self.backend,
                cycle_inputs(),
            )
            .unwrap();
            let input_claim = intermediate(&throwaway.output_claims(self.cycle_claims).unwrap());

            let cycle_challenges = synthetic_point(cycle_rounds, seed);
            run_lockstep(
                reference.as_mut(),
                optimized.as_mut(),
                input_claim,
                &cycle_challenges,
            );
            let reference_outputs = reference.output_claims(self.cycle_claims).unwrap();
            let optimized_outputs = optimized.output_claims(self.cycle_claims).unwrap();
            assert_eq!(
                reference_outputs, optimized_outputs,
                "cycle outputs diverged"
            );
            let handoff_claim = intermediate(&reference_outputs);

            reference.park_residue(&mut session_ref);
            optimized.park_residue(&mut session_opt);
            assert!(
                session_ref
                    .state::<PrecommittedReductionCarry<Fr, RA>>()
                    .is_some()
                    && session_opt
                        .state::<PrecommittedReductionCarry<Fr, RA>>()
                        .is_some(),
                "cycle kernels parked no carry for the scheduled address phase"
            );

            // Cross-tier stage 7: the reference slot reclaims the OPTIMIZED
            // pipeline's carry and the optimized slot the reference one's —
            // the mixed-tier composition is what the address lockstep pins.
            let address_inputs = || ProverInputs {
                relation: self.address_relation,
                claims: self.address_claims,
                points: self.address_points,
                challenges: self.address_challenges_struct,
            };
            let mut reference_address = ReferencePrecommittedAddress::<RA>::new(self.missing_carry)
                .prepare(&mut session_opt, self.backend, address_inputs())
                .unwrap();
            let mut optimized_address = OptimizedPrecommittedAddress::<RA>::new(self.missing_carry)
                .prepare(&mut session_ref, self.backend, address_inputs())
                .unwrap();
            let address_challenges = synthetic_point(address_rounds, seed.wrapping_add(1000));
            run_lockstep(
                reference_address.as_mut(),
                optimized_address.as_mut(),
                handoff_claim,
                &address_challenges,
            );
            let reference_final = reference_address
                .output_claims(self.address_claims)
                .unwrap();
            let optimized_final = optimized_address
                .output_claims(self.address_claims)
                .unwrap();
            assert_eq!(reference_final, optimized_final, "address outputs diverged");

            // Missing-carry contract: both tiers' address slots refuse a
            // session stage 6b never parked into, with the same diagnostic.
            let Err(optimized_error) = OptimizedPrecommittedAddress::<RA>::new(self.missing_carry)
                .prepare(&mut ProofSession::default(), self.backend, address_inputs())
            else {
                panic!("optimized address slot accepted a session with no parked carry");
            };
            let Err(reference_error) = ReferencePrecommittedAddress::<RA>::new(self.missing_carry)
                .prepare(&mut ProofSession::default(), self.backend, address_inputs())
            else {
                panic!("reference address slot accepted a session with no parked carry");
            };
            assert!(
                matches!(
                    optimized_error,
                    KernelError::InvariantViolation { reason } if reason == self.missing_carry
                ),
                "optimized missing-carry error drifted: {optimized_error}"
            );
            assert_eq!(optimized_error.to_string(), reference_error.to_string());

            reference_final
        }
    }

    fn advice_pair(trace_order: TracePolynomialOrder, kind: JoltAdviceKind) {
        with_fixture(trace_order, |backend, schedule| {
            let layout = schedule.advice(kind).unwrap().clone();
            let dimensions = layout.dimensions();
            assert!(
                dimensions.has_address_phase(),
                "fixture geometry must schedule an address phase for {kind:?}"
            );
            let r_val = synthetic_point(
                layout
                    .precommitted()
                    .poly_opening_round_permutation_be()
                    .len(),
                43,
            );
            let cycle_rounds = dimensions.cycle_phase_total_rounds();
            let address_rounds = dimensions.address_phase_total_rounds();
            let cycle_challenges = synthetic_point(cycle_rounds, 7);
            let cycle_vars = layout
                .cycle_phase_variable_challenges(&cycle_challenges)
                .unwrap();

            match kind {
                JoltAdviceKind::Trusted => {
                    let cycle_relation = TrustedAdviceCyclePhase::new(&layout, Some(r_val.clone()));
                    let address_relation =
                        TrustedAdviceAddressPhase::new(&layout, Some(r_val), cycle_vars);
                    let pair = PhasePair {
                        backend,
                        cycle_relation: &cycle_relation,
                        cycle_claims: &Default::default(),
                        cycle_points: &Default::default(),
                        cycle_challenges_struct: &Default::default(),
                        address_relation: &address_relation,
                        address_claims: &Default::default(),
                        address_points: &Default::default(),
                        address_challenges_struct: &Default::default(),
                        missing_carry: MISSING_TRUSTED,
                    };
                    let _ = pair.run(cycle_rounds, address_rounds, |claims| claims.trusted, 7);
                }
                JoltAdviceKind::Untrusted => {
                    let cycle_relation =
                        UntrustedAdviceCyclePhase::new(&layout, Some(r_val.clone()));
                    let address_relation =
                        UntrustedAdviceAddressPhase::new(&layout, Some(r_val), cycle_vars);
                    let pair = PhasePair {
                        backend,
                        cycle_relation: &cycle_relation,
                        cycle_claims: &Default::default(),
                        cycle_points: &Default::default(),
                        cycle_challenges_struct: &Default::default(),
                        address_relation: &address_relation,
                        address_claims: &Default::default(),
                        address_points: &Default::default(),
                        address_challenges_struct: &Default::default(),
                        missing_carry: MISSING_UNTRUSTED,
                    };
                    let _ = pair.run(cycle_rounds, address_rounds, |claims| claims.untrusted, 11);
                }
            }
        });
    }

    #[test]
    fn trusted_advice_phases_match_reference_cycle_major() {
        advice_pair(TracePolynomialOrder::CycleMajor, JoltAdviceKind::Trusted);
    }

    #[test]
    fn trusted_advice_phases_match_reference_address_major() {
        advice_pair(TracePolynomialOrder::AddressMajor, JoltAdviceKind::Trusted);
    }

    #[test]
    fn untrusted_advice_phases_match_reference_cycle_major() {
        advice_pair(TracePolynomialOrder::CycleMajor, JoltAdviceKind::Untrusted);
    }

    #[test]
    fn untrusted_advice_phases_match_reference_address_major() {
        advice_pair(
            TracePolynomialOrder::AddressMajor,
            JoltAdviceKind::Untrusted,
        );
    }

    fn bytecode_pair(trace_order: TracePolynomialOrder) {
        with_fixture(trace_order, |backend, schedule| {
            let layout = schedule.bytecode.as_ref().unwrap().clone();
            let dimensions = layout.dimensions();
            assert!(
                dimensions.has_address_phase(),
                "fixture geometry must schedule a bytecode address phase"
            );
            let weights = BytecodeReductionWeights {
                r_bc: synthetic_point(layout.log_bytecode_chunk_size(), 51),
                chunk_rbc_weights: synthetic_point(layout.chunk_count(), 53),
                lane_weights: synthetic_point(
                    jolt_claims::protocols::jolt::geometry::claim_reductions::bytecode::COMMITTED_BYTECODE_LANE_CAPACITY,
                    57,
                ),
            };
            let cycle_relation = BytecodeReductionCyclePhase::new(&layout, weights.clone());
            let cycle_rounds = dimensions.cycle_phase_total_rounds();
            let address_rounds = dimensions.address_phase_total_rounds();
            let cycle_challenges = synthetic_point(cycle_rounds, 13);
            let cycle_vars = layout
                .cycle_phase_variable_challenges(&cycle_challenges)
                .unwrap();
            let address_relation =
                BytecodeReductionAddressPhase::new(&layout, Some(weights), cycle_vars);
            let pair = PhasePair {
                backend,
                cycle_relation: &cycle_relation,
                cycle_claims: &Default::default(),
                cycle_points: &Default::default(),
                cycle_challenges_struct: &BytecodeReductionCyclePhaseChallenges { eta: fr(29) },
                address_relation: &address_relation,
                address_claims: &Default::default(),
                address_points: &Default::default(),
                address_challenges_struct: &Default::default(),
                missing_carry: MISSING_BYTECODE,
            };
            let final_claims = pair.run(
                cycle_rounds,
                address_rounds,
                |claims| {
                    claims
                        .intermediate
                        .expect("cycle phase staged no intermediate")
                },
                13,
            );
            assert_eq!(
                final_claims.chunks.len(),
                layout.chunk_count(),
                "address phase produced the wrong chunk-opening count"
            );
        });
    }

    #[test]
    fn bytecode_reduction_phases_match_reference_cycle_major() {
        bytecode_pair(TracePolynomialOrder::CycleMajor);
    }

    #[test]
    fn bytecode_reduction_phases_match_reference_address_major() {
        bytecode_pair(TracePolynomialOrder::AddressMajor);
    }

    fn program_image_pair(trace_order: TracePolynomialOrder) {
        with_fixture(trace_order, |backend, schedule| {
            let layout = schedule.program_image.as_ref().unwrap().clone();
            let dimensions = layout.dimensions();
            assert!(
                dimensions.has_address_phase(),
                "fixture geometry must schedule a program-image address phase"
            );
            let r_addr_rw = synthetic_point(IMAGE_RAM_VARS, 61);
            let cycle_relation = ProgramImageReductionCyclePhase::new(&layout, r_addr_rw.clone());
            let cycle_rounds = dimensions.cycle_phase_total_rounds();
            let address_rounds = dimensions.address_phase_total_rounds();
            let cycle_challenges = synthetic_point(cycle_rounds, 17);
            let cycle_vars = layout
                .cycle_phase_variable_challenges(&cycle_challenges)
                .unwrap();
            let address_relation =
                ProgramImageReductionAddressPhase::new(&layout, Some(r_addr_rw), cycle_vars);
            let pair = PhasePair {
                backend,
                cycle_relation: &cycle_relation,
                cycle_claims: &Default::default(),
                cycle_points: &Default::default(),
                cycle_challenges_struct: &Default::default(),
                address_relation: &address_relation,
                address_claims: &Default::default(),
                address_points: &Default::default(),
                address_challenges_struct: &Default::default(),
                missing_carry: MISSING_PROGRAM_IMAGE,
            };
            let _ = pair.run(
                cycle_rounds,
                address_rounds,
                |claims| claims.program_image,
                17,
            );
        });
    }

    #[test]
    fn program_image_phases_match_reference_cycle_major() {
        program_image_pair(TracePolynomialOrder::CycleMajor);
    }

    #[test]
    fn program_image_phases_match_reference_address_major() {
        program_image_pair(TracePolynomialOrder::AddressMajor);
    }

    /// The blocked shifted eq slice against the reference tier's full-table
    /// gather, unaligned starts and wrapped tails included (the wrapped
    /// entries enter the bound tables, so they must match exactly).
    #[test]
    fn shifted_eq_slice_matches_full_table_gather() {
        let r = synthetic_point(4, 91);
        let full = eq_table(&r);
        let domain = full.len();
        for (start, len) in [(0, 16), (3, 8), (13, 8), (5, 2), (15, 4), (6, 1)] {
            let expected: Vec<Fr> = (0..len)
                .map(|offset| full[(start + offset) & (domain - 1)])
                .collect();
            assert_eq!(
                shifted_eq_slice(&r, start, len),
                expected,
                "shifted eq slice diverged for start {start}, len {len}"
            );
        }
    }

    /// The advice opening evaluation (stage 4) against the reference slot.
    #[test]
    fn advice_opening_evaluation_matches_reference() {
        with_fixture(TracePolynomialOrder::CycleMajor, |backend, schedule| {
            for kind in [JoltAdviceKind::Trusted, JoltAdviceKind::Untrusted] {
                let vars = schedule
                    .advice(kind)
                    .unwrap()
                    .precommitted()
                    .poly_opening_round_permutation_be()
                    .len();
                let point = synthetic_point(vars, 71);
                let reference_value = <ReferenceBackend as AdviceOpeningEvaluation<Fr>>::evaluate(
                    &ReferenceBackend,
                    &mut ProofSession::default(),
                    kind,
                    &point,
                    backend,
                )
                .unwrap();
                let optimized_value =
                    <OptimizedPrecommittedCycle as AdviceOpeningEvaluation<Fr>>::evaluate(
                        &OptimizedPrecommittedCycle,
                        &mut ProofSession::default(),
                        kind,
                        &point,
                        backend,
                    )
                    .unwrap();
                assert_eq!(reference_value, optimized_value);
                assert_ne!(
                    reference_value,
                    Fr::from_u64(0),
                    "degenerate advice fixture"
                );
            }
        });
    }
}
