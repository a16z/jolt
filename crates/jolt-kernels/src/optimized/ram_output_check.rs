//! The optimized RAM output-check (stage 2) kernel.
//!
//! Byte-parity contract: identical round polynomials and output claims to the
//! reference kernel (`reference/ram_output_check.rs`), which materializes the
//! `K`-sized address-eq table and binds four dense tables (eq, mask, val_io,
//! val_final) through the naive expression interpreter every round.
//!
//! Carries forward the former `OutputSumcheckProver` optimizations:
//!
//! - **Gruen split-eq factoring**: `eq(r_address, ·)` is held as an
//!   `E_out ⊗ E_in` tensor plus a per-round linear factor
//!   ([`GruenSplitEqPolynomial`]) — never materialized as a `K`-sized table,
//!   never bound. Each round samples `q(t) = Σ_y E(y) · mask(t,y) ·
//!   (val_final − val_io)(t,y)` at the naive prover's `t = 0..=3` nodes and
//!   emits `s(t) = ℓ(t) · q(t)` through the same `from_evals` interpolation,
//!   so the coefficients are byte-identical (the split-eq product is the
//!   exact partial bind of the dense eq table, and distributing
//!   `eq·mask·val_final − eq·mask·val_io` into `eq·mask·(val_final − val_io)`
//!   is exact field algebra).
//! - **Direct three-table walk**: the summand runs on the mask/final/io
//!   tables with inline arithmetic instead of the naive tier's per-point
//!   expression-tree interpretation.
//!
//! The legacy prover's leading zero-address shortcut (constant round
//! polynomials while the pair fold is provably zero) is intentionally not
//! replicated: those rounds' true polynomials are zero, this kernel computes
//! the zeros literally — matching the reference tier — and the engine's
//! batched-polynomial trim reproduces the wire bytes.

use jolt_claims::protocols::jolt::geometry::ram::ram_val_final;
use jolt_claims::protocols::jolt::{JoltDerivedId, RamOutputCheckPublic};
use jolt_field::JoltField;
use jolt_poly::{BindingOrder, GruenSplitEqPolynomial, Polynomial, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::{
    ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckInputPoints, SumcheckOutputPoints,
};
use jolt_verifier::stages::stage2::ram_output_check::{RamOutputCheck, RamOutputCheckOutputClaims};
use jolt_witness::JoltWitnessPlane;

use super::support::{pin_derived_term, GruenRoundMessage, RoundProgress};
use super::OptimizedBackend;
use crate::reference::views::dense_view;
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

impl<F: JoltField> PrepareKernel<F, RamOutputCheck<F>> for OptimizedBackend {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, RamOutputCheck<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = RamOutputCheck<F>>>, KernelError<F>> {
        let relation = inputs.relation;
        let output_address_challenges = inputs.challenges.output_address.as_slice();
        let ram_log_k = output_address_challenges.len();
        if relation.read_write_dimensions().output_check_rounds() != ram_log_k {
            return Err(KernelError::Unsupported {
                reason: "optimized RAM output check supports only the default read-write config \
                         (phase 1 = all cycle rounds)",
            });
        }

        // The public-IO tables, exactly as the reference builds them.
        let public_memory = relation.public_memory();
        let addresses = 1usize << ram_log_k;
        let mut val_io = vec![F::zero(); addresses];
        for segment in &public_memory.segments {
            for (offset, &word) in segment.words.iter().enumerate() {
                let index = segment.start_index as usize + offset;
                if index < addresses {
                    val_io[index] = F::from_u64(word);
                }
            }
        }
        let io_mask: Vec<F> = (0..addresses)
            .map(|k| {
                let in_io_region = (k as u128) >= public_memory.io_mask_start
                    && (k as u128) < public_memory.io_mask_end;
                if in_io_region {
                    F::one()
                } else {
                    F::zero()
                }
            })
            .collect();

        Ok(Box::new(OutputCheckKernel {
            progress: RoundProgress::new(ram_log_k),
            gruen: GruenSplitEqPolynomial::new(output_address_challenges, BindingOrder::LowToHigh),
            io_mask: Polynomial::new(io_mask),
            val_io: Polynomial::new(val_io),
            val_final: Polynomial::new(dense_view(witness, ram_val_final())?),
            bind_scratch: Vec::new(),
        }))
    }
}

#[cfg_attr(
    feature = "allocative",
    derive(allocative::Allocative),
    allocative(bound = "F: JoltField")
)]
struct OutputCheckKernel<F: JoltField> {
    progress: RoundProgress,
    gruen: GruenSplitEqPolynomial<F>,
    io_mask: Polynomial<F>,
    val_io: Polynomial<F>,
    val_final: Polynomial<F>,
    #[cfg_attr(feature = "allocative", allocative(visit = jolt_poly::visit_scalars))]
    bind_scratch: Vec<F>,
}
impl<F: JoltField> OutputCheckKernel<F> {
    /// `s(t) = ℓ(t) · q(t)` at the naive prover's `t = 0..=3` sample points,
    /// with `q(t) = Σ_y E(y) · mask(t, y) · (val_final − val_io)(t, y)`.
    fn message(
        &self,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        const POINTS: usize = 4;

        let mut q_evals = self.gruen.par_fold_out_in(
            || [F::zero(); POINTS],
            |acc, row, _x_in, e_in| {
                let pair = |table: &Polynomial<F>| {
                    let evals = table.evals();
                    (evals[2 * row], evals[2 * row + 1])
                };
                let (mask_0, mask_1) = pair(&self.io_mask);
                let (final_0, final_1) = pair(&self.val_final);
                let (io_0, io_1) = pair(&self.val_io);
                let mut mask = mask_0;
                let mask_step = mask_1 - mask_0;
                let mut diff = final_0 - io_0;
                let diff_step = (final_1 - io_1) - diff;
                for value in acc.iter_mut() {
                    *value += e_in * mask * diff;
                    mask += mask_step;
                    diff += diff_step;
                }
            },
            |_x_out, e_out, mut acc| {
                for value in &mut acc {
                    *value *= e_out;
                }
                acc
            },
            |mut a, b| {
                for (a, b) in a.iter_mut().zip(&b) {
                    *a += *b;
                }
                a
            },
        );

        self.gruen
            .checked_round_poly(&mut q_evals, previous_claim, round)
    }

    fn bind(&mut self, challenge: F) {
        self.gruen.bind(challenge);
        for table in [&mut self.io_mask, &mut self.val_io, &mut self.val_final] {
            table.bind_low_to_high_reusing_scratch(challenge, &mut self.bind_scratch);
        }
        self.progress.advance();
    }
}

impl<F: JoltField> ProveRounds<F> for OutputCheckKernel<F> {
    fn num_rounds(&self) -> usize {
        self.progress.total()
    }

    fn prove_round(
        &mut self,
        bind: Option<F>,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if let Some(challenge) = bind {
            self.bind(challenge);
        }
        self.message(round, previous_claim)
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind);
        Ok(())
    }
}

impl<F: JoltField> SumcheckKernel<F> for OutputCheckKernel<F> {
    type Relation = RamOutputCheck<F>;

    fn output_claims(
        &mut self,
        _inputs: &SumcheckInputClaims<F, Self::Relation>,
    ) -> Result<RamOutputCheckOutputClaims<F>, SumcheckKernelError<F>> {
        self.progress.require_complete()?;
        Ok(RamOutputCheckOutputClaims {
            val_final: self.val_final.evals()[0],
        })
    }

    /// Pin the three derived leaves to the verifier's scalar path: the bound
    /// Gruen scalar is the `EqAddress` value, the bound mask/io tables the
    /// other two.
    fn validate_derived_tables(
        &self,
        relation: &Self::Relation,
        input_points: &SumcheckInputPoints<F, Self::Relation>,
        output_points: &SumcheckOutputPoints<F, Self::Relation>,
        challenges: &ConcreteSumcheckChallenges<F, Self::Relation>,
    ) -> Result<(), SumcheckKernelError<F>> {
        self.progress.require_complete()?;
        for (public, got) in [
            (RamOutputCheckPublic::EqAddress, self.gruen.current_scalar()),
            (RamOutputCheckPublic::IoMask, self.io_mask.evals()[0]),
            (RamOutputCheckPublic::ValIo, self.val_io.evals()[0]),
        ] {
            let id = JoltDerivedId::from(public);
            pin_derived_term(relation, id, input_points, output_points, challenges, got)?;
        }
        Ok(())
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use common::constants::RAM_START_ADDRESS;
    use common::jolt_device::{JoltDevice, MemoryConfig};
    use jolt_claims::protocols::jolt::{JoltOneHotConfig, ReadWriteDimensions};
    use jolt_field::{Fr, Ring};
    use jolt_program::execution::{JoltProgram, MemoryImage, OwnedTrace, TraceOutput, TraceRow};
    use jolt_program::preprocess::{
        BytecodePreprocessing, JoltProgramPreprocessing, PublicIoMemory, RAMPreprocessing,
    };
    use jolt_riscv::{JoltInstructionKind, JoltInstructionRow, NormalizedOperands, RV64IMAC_JOLT};
    use jolt_verifier::stages::stage2::ram_output_check::{
        RamOutputCheckChallenges, RamOutputCheckInputClaims,
    };
    use jolt_witness::{JoltVmWitnessConfig, JoltVmWitnessInputs, TraceBackend};

    use super::super::testing::{assert_parity, random_scalars};
    use super::*;
    use crate::ReferenceBackend;

    /// A witness plane whose `RamValFinal` is nontrivial on both sides of the
    /// IO-mask boundary: real inputs/outputs (the oracle synthesizes the IO
    /// region from the device, so `val_final = val_io` there by construction)
    /// plus a final-memory image carrying post-execution DRAM bytes.
    fn with_output_check_plane<R>(
        log_t: usize,
        ram_k: usize,
        f: impl FnOnce(&dyn JoltWitnessPlane<Fr>, PublicIoMemory) -> R,
    ) -> R {
        let mut device = JoltDevice::new(&MemoryConfig {
            program_size: Some(1024),
            max_trusted_advice_size: 0,
            max_untrusted_advice_size: 0,
            max_input_size: 16,
            max_output_size: 16,
            ..Default::default()
        });
        device.inputs = vec![42, 1];
        device.outputs = vec![7];
        let public_memory = PublicIoMemory::new(&device).unwrap();

        let instruction = JoltInstructionRow {
            instruction_kind: JoltInstructionKind::ADDI,
            address: 0x8000_0000,
            operands: NormalizedOperands {
                rd: Some(1),
                rs1: Some(2),
                rs2: None,
                imm: 3,
            },
            virtual_sequence_remaining: None,
            is_first_in_sequence: false,
            is_compressed: false,
        };
        use std::sync::Arc;
        let preprocessing = Arc::new(JoltProgramPreprocessing {
            bytecode: BytecodePreprocessing::preprocess(
                vec![instruction],
                instruction.address as u64,
                RV64IMAC_JOLT,
            )
            .unwrap(),
            ram: RAMPreprocessing::default(),
            memory_layout: device.memory_layout.clone(),
            max_padded_trace_length: 1 << log_t,
        });
        let rows = vec![TraceRow {
            instruction,
            ..TraceRow::default()
        }];
        // Post-execution DRAM bytes (outside the IO mask): nonzero
        // `val_final − val_io` there keeps the later round polynomials
        // nontrivial while the Boolean-point sum stays zero.
        let final_memory = MemoryImage {
            bytes: vec![
                (RAM_START_ADDRESS, 0xAB),
                (RAM_START_ADDRESS + 8, 0x13),
                (RAM_START_ADDRESS + 17, 0x07),
            ],
        };

        let program = Arc::new(JoltProgram::default());
        let config = JoltVmWitnessConfig::new(
            log_t,
            ram_k,
            JoltOneHotConfig {
                log_k_chunk: 4,
                lookups_ra_virtual_log_k_chunk: 16,
            },
        );
        let inputs = JoltVmWitnessInputs::new(
            &program,
            &preprocessing,
            TraceOutput::new(OwnedTrace::new(rows), device, Some(final_memory), None),
        );
        let backend = TraceBackend::new(config, inputs);
        f(&backend, public_memory)
    }

    fn run_parity(log_t: usize, ram_k: usize, seed: u64) {
        with_output_check_plane(log_t, ram_k, |witness, public_memory| {
            let log_k = ram_k.trailing_zeros() as usize;
            let relation = RamOutputCheck::<Fr>::new(
                ReadWriteDimensions::new(log_t, log_k, log_t, log_k),
                public_memory,
            );
            let challenges = RamOutputCheckChallenges {
                output_address: random_scalars(log_k, seed ^ 0xADD1),
            };
            let claims = RamOutputCheckInputClaims::<Fr>::default();
            let points = RamOutputCheckInputClaims::<Vec<Fr>>::default();

            // Fixture guard: the DRAM image must reach `val_final` (a zero
            // table would make parity vacuous).
            let val_final = dense_view::<Fr>(witness, ram_val_final()).unwrap();
            assert_ne!(val_final[8], Fr::from_u64(0), "degenerate DRAM fixture");

            let inputs = ProverInputs {
                relation: &relation,
                claims: &claims,
                points: &points,
                challenges: &challenges,
            };
            let mut reference_session = ProofSession::default();
            let reference = PrepareKernel::<Fr, _>::prepare(
                &ReferenceBackend,
                &mut reference_session,
                witness,
                ProverInputs {
                    relation: &relation,
                    claims: &claims,
                    points: &points,
                    challenges: &challenges,
                },
            )
            .unwrap();
            let mut session = ProofSession::default();
            let optimized = PrepareKernel::<Fr, _>::prepare(
                &OptimizedBackend,
                &mut session,
                witness,
                ProverInputs {
                    relation: &relation,
                    claims: &claims,
                    points: &points,
                    challenges: &challenges,
                },
            )
            .unwrap();

            assert_parity(reference, optimized, Fr::from_u64(0), &inputs, seed);
        });
    }

    #[test]
    fn parity_k16() {
        run_parity(3, 16, 401);
    }

    #[test]
    fn parity_k32_deeper_address_domain() {
        run_parity(4, 32, 409);
    }

    #[test]
    fn parity_k16_alternate_seed() {
        run_parity(2, 16, 419);
    }
}
