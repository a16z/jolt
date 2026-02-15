use std::sync::Arc;

use crate::curve::JoltCurve;
use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::{StreamingCommitmentScheme, ZkEvalCommitment};
use crate::subprotocols::booleanity::{BooleanitySumcheckParams, BooleanitySumcheckProver};
use crate::subprotocols::streaming_schedule::LinearOnlySchedule;
use crate::subprotocols::sumcheck::SumcheckInstanceProof;
use crate::subprotocols::sumcheck_prover::SumcheckInstanceProver;
use crate::subprotocols::univariate_skip::UniSkipFirstRoundProofVariant;
use crate::transcripts::Transcript;
use crate::utils::math::Math;
use crate::utils::thread::drop_in_background_thread;
use crate::zkvm::bytecode::read_raf_checking::{
    BytecodeReadRafSumcheckParams, BytecodeReadRafSumcheckProver,
};
use crate::zkvm::claim_reductions::advice::ReductionPhase;
use crate::zkvm::claim_reductions::{
    AdviceClaimReductionParams, AdviceClaimReductionProver, AdviceKind,
    HammingWeightClaimReductionParams, HammingWeightClaimReductionProver,
    IncClaimReductionSumcheckParams, IncClaimReductionSumcheckProver,
    InstructionLookupsClaimReductionSumcheckParams, InstructionLookupsClaimReductionSumcheckProver,
    RaReductionParams, RamRaClaimReductionSumcheckProver, RegistersClaimReductionSumcheckParams,
    RegistersClaimReductionSumcheckProver,
};
use crate::zkvm::instruction_lookups::ra_virtual::InstructionRaSumcheckProver as LookupsRaSumcheckProver;
use crate::zkvm::instruction_lookups::{
    ra_virtual::InstructionRaSumcheckParams,
    read_raf_checking::{InstructionReadRafSumcheckParams, InstructionReadRafSumcheckProver},
};
use crate::zkvm::ram::{
    hamming_booleanity::{HammingBooleanitySumcheckParams, HammingBooleanitySumcheckProver},
    output_check::{OutputSumcheckParams, OutputSumcheckProver},
    prover_accumulate_advice,
    ra_virtual::{RamRaVirtualParams, RamRaVirtualSumcheckProver},
    raf_evaluation::{
        RafEvaluationSumcheckParams, RafEvaluationSumcheckProver as RamRafEvaluationSumcheckProver,
    },
    read_write_checking::{RamReadWriteCheckingParams, RamReadWriteCheckingProver},
    val_evaluation::{
        ValEvaluationSumcheckParams, ValEvaluationSumcheckProver as RamValEvaluationSumcheckProver,
    },
    val_final::{ValFinalSumcheckParams, ValFinalSumcheckProver},
};
use crate::zkvm::registers::{
    read_write_checking::{RegistersReadWriteCheckingParams, RegistersReadWriteCheckingProver},
    val_evaluation::RegistersValEvaluationSumcheckParams,
    val_evaluation::ValEvaluationSumcheckProver as RegistersValEvaluationSumcheckProver,
};
use crate::zkvm::spartan::{
    instruction_input::{InstructionInputParams, InstructionInputSumcheckProver},
    outer::{
        OuterRemainingStreamingSumcheck, OuterSharedState, OuterUniSkipParams, OuterUniSkipProver,
    },
    product::{
        ProductVirtualRemainderParams, ProductVirtualRemainderProver, ProductVirtualUniSkipParams,
        ProductVirtualUniSkipProver,
    },
    shift::{ShiftSumcheckParams, ShiftSumcheckProver},
};

#[cfg(feature = "allocative")]
use super::preprocessing::{write_boxed_instance_flamegraph_svg, write_instance_flamegraph_svg};
#[cfg(not(target_arch = "wasm32"))]
use crate::utils::profiling::print_current_memory_usage;
#[cfg(feature = "allocative")]
use crate::utils::profiling::print_data_structure_heap_usage;

use super::JoltCpuProver;

impl<
        'a,
        F: JoltField,
        C: JoltCurve,
        PCS: StreamingCommitmentScheme<Field = F> + ZkEvalCommitment<C>,
        ProofTranscript: Transcript,
    > JoltCpuProver<'a, F, C, PCS, ProofTranscript>
{
    #[allow(clippy::type_complexity)]
    #[tracing::instrument(skip_all)]
    pub(super) fn prove_stage1(
        &mut self,
    ) -> (
        UniSkipFirstRoundProofVariant<F, C, ProofTranscript>,
        SumcheckInstanceProof<F, C, ProofTranscript>,
        Vec<F::Challenge>,
    ) {
        #[cfg(not(target_arch = "wasm32"))]
        print_current_memory_usage("Stage 1 baseline");

        tracing::info!("Stage 1 proving");
        let uni_skip_params = OuterUniSkipParams::new(&self.spartan_key, &mut self.transcript);
        let mut uni_skip = OuterUniSkipProver::initialize(
            uni_skip_params.clone(),
            &self.trace,
            &self.preprocessing.shared.bytecode,
        );
        let first_round_proof = self.prove_uniskip(&mut uni_skip);

        let schedule = LinearOnlySchedule::new(uni_skip_params.tau.len() - 1);
        let shared = OuterSharedState::new(
            Arc::clone(&self.trace),
            &self.preprocessing.shared.bytecode,
            &uni_skip_params,
            &self.opening_accumulator,
        );
        let mut spartan_outer_remaining: OuterRemainingStreamingSumcheck<_, _> =
            OuterRemainingStreamingSumcheck::new(shared, schedule);

        let (sumcheck_proof, r_stage1, _initial_claim) = self.prove_batched_sumcheck(vec![
            &mut spartan_outer_remaining as &mut dyn SumcheckInstanceProver<_, _>,
        ]);

        (first_round_proof, sumcheck_proof, r_stage1)
    }

    #[allow(clippy::type_complexity)]
    #[tracing::instrument(skip_all)]
    pub(super) fn prove_stage2(
        &mut self,
    ) -> (
        UniSkipFirstRoundProofVariant<F, C, ProofTranscript>,
        SumcheckInstanceProof<F, C, ProofTranscript>,
        Vec<F::Challenge>,
    ) {
        #[cfg(not(target_arch = "wasm32"))]
        print_current_memory_usage("Stage 2 baseline");

        let uni_skip_params =
            ProductVirtualUniSkipParams::new(&self.opening_accumulator, &mut self.transcript);
        let mut uni_skip =
            ProductVirtualUniSkipProver::initialize(uni_skip_params.clone(), &self.trace);
        let first_round_proof = self.prove_uniskip(&mut uni_skip);

        let ram_read_write_checking_params = RamReadWriteCheckingParams::new(
            &self.opening_accumulator,
            &mut self.transcript,
            &self.one_hot_params,
            self.trace.len(),
            &self.rw_config,
        );
        let spartan_product_virtual_remainder_params = ProductVirtualRemainderParams::new(
            self.trace.len(),
            uni_skip_params,
            &self.opening_accumulator,
        );
        let instruction_claim_reduction_params =
            InstructionLookupsClaimReductionSumcheckParams::new(
                self.trace.len(),
                &self.opening_accumulator,
                &mut self.transcript,
            );
        let ram_raf_evaluation_params = RafEvaluationSumcheckParams::new(
            &self.program_io.memory_layout,
            &self.one_hot_params,
            &self.opening_accumulator,
        );
        let ram_output_check_params = OutputSumcheckParams::new(
            self.one_hot_params.ram_k,
            &self.program_io,
            &mut self.transcript,
        );
        let ram_read_write_checking = RamReadWriteCheckingProver::initialize(
            ram_read_write_checking_params,
            &self.trace,
            &self.preprocessing.shared.bytecode,
            &self.program_io.memory_layout,
            &self.initial_ram_state,
        );
        let spartan_product_virtual_remainder = ProductVirtualRemainderProver::initialize(
            spartan_product_virtual_remainder_params,
            Arc::clone(&self.trace),
        );
        let instruction_claim_reduction =
            InstructionLookupsClaimReductionSumcheckProver::initialize(
                instruction_claim_reduction_params,
                Arc::clone(&self.trace),
            );
        let ram_raf_evaluation = RamRafEvaluationSumcheckProver::initialize(
            ram_raf_evaluation_params,
            &self.trace,
            &self.program_io.memory_layout,
        );
        let ram_output_check = OutputSumcheckProver::initialize(
            ram_output_check_params,
            &self.initial_ram_state,
            &self.final_ram_state,
            &self.program_io.memory_layout,
        );

        #[cfg(feature = "allocative")]
        {
            print_data_structure_heap_usage("RamReadWriteCheckingProver", &ram_read_write_checking);
            print_data_structure_heap_usage(
                "ProductVirtualRemainderProver",
                &spartan_product_virtual_remainder,
            );
            print_data_structure_heap_usage(
                "InstructionLookupsClaimReductionSumcheckProver",
                &instruction_claim_reduction,
            );
            print_data_structure_heap_usage("RamRafEvaluationSumcheckProver", &ram_raf_evaluation);
            print_data_structure_heap_usage("OutputSumcheckProver", &ram_output_check);
        }

        let mut instances: Vec<Box<dyn SumcheckInstanceProver<_, _>>> = vec![
            Box::new(ram_read_write_checking),
            Box::new(spartan_product_virtual_remainder),
            Box::new(instruction_claim_reduction),
            Box::new(ram_raf_evaluation),
            Box::new(ram_output_check),
        ];

        #[cfg(feature = "allocative")]
        write_boxed_instance_flamegraph_svg(&instances, "stage2_start_flamechart.svg");
        tracing::info!("Stage 2 proving");

        let (sumcheck_proof, r_stage2, _initial_claim) =
            self.prove_batched_sumcheck(instances.iter_mut().map(|v| &mut **v as _).collect());

        #[cfg(feature = "allocative")]
        write_boxed_instance_flamegraph_svg(&instances, "stage2_end_flamechart.svg");
        drop_in_background_thread(instances);

        (first_round_proof, sumcheck_proof, r_stage2)
    }

    #[tracing::instrument(skip_all)]
    pub(super) fn prove_stage3(
        &mut self,
    ) -> (
        SumcheckInstanceProof<F, C, ProofTranscript>,
        Vec<F::Challenge>,
    ) {
        #[cfg(not(target_arch = "wasm32"))]
        print_current_memory_usage("Stage 3 baseline");

        let spartan_shift_params = ShiftSumcheckParams::new(
            self.trace.len().log_2(),
            &self.opening_accumulator,
            &mut self.transcript,
        );
        let spartan_instruction_input_params =
            InstructionInputParams::new(&self.opening_accumulator, &mut self.transcript);
        let spartan_registers_claim_reduction_params = RegistersClaimReductionSumcheckParams::new(
            self.trace.len(),
            &self.opening_accumulator,
            &mut self.transcript,
        );

        let spartan_shift = ShiftSumcheckProver::initialize(
            spartan_shift_params,
            Arc::clone(&self.trace),
            &self.preprocessing.shared.bytecode,
        );
        let spartan_instruction_input = InstructionInputSumcheckProver::initialize(
            spartan_instruction_input_params,
            &self.trace,
            &self.opening_accumulator,
        );
        let spartan_registers_claim_reduction = RegistersClaimReductionSumcheckProver::initialize(
            spartan_registers_claim_reduction_params,
            Arc::clone(&self.trace),
        );

        #[cfg(feature = "allocative")]
        {
            print_data_structure_heap_usage("ShiftSumcheckProver", &spartan_shift);
            print_data_structure_heap_usage(
                "InstructionInputSumcheckProver",
                &spartan_instruction_input,
            );
            print_data_structure_heap_usage(
                "RegistersClaimReductionSumcheckProver",
                &spartan_registers_claim_reduction,
            );
        }

        let mut instances: Vec<Box<dyn SumcheckInstanceProver<_, _>>> = vec![
            Box::new(spartan_shift),
            Box::new(spartan_instruction_input),
            Box::new(spartan_registers_claim_reduction),
        ];

        #[cfg(feature = "allocative")]
        write_boxed_instance_flamegraph_svg(&instances, "stage3_start_flamechart.svg");
        tracing::info!("Stage 3 proving");

        let (sumcheck_proof, r_stage3, _initial_claim) =
            self.prove_batched_sumcheck(instances.iter_mut().map(|v| &mut **v as _).collect());
        #[cfg(feature = "allocative")]
        write_boxed_instance_flamegraph_svg(&instances, "stage3_end_flamechart.svg");
        drop_in_background_thread(instances);

        (sumcheck_proof, r_stage3)
    }

    #[tracing::instrument(skip_all)]
    pub(super) fn prove_stage4(
        &mut self,
    ) -> (
        SumcheckInstanceProof<F, C, ProofTranscript>,
        Vec<F::Challenge>,
    ) {
        #[cfg(not(target_arch = "wasm32"))]
        print_current_memory_usage("Stage 4 baseline");

        prover_accumulate_advice(
            &self.advice.untrusted_advice_polynomial,
            &self.advice.trusted_advice_polynomial,
            &self.program_io.memory_layout,
            &self.one_hot_params,
            &mut self.opening_accumulator,
            self.rw_config
                .needs_single_advice_opening(self.trace.len().log_2()),
        );

        let registers_read_write_checking_params = RegistersReadWriteCheckingParams::new(
            self.trace.len(),
            &self.opening_accumulator,
            &mut self.transcript,
            &self.rw_config,
        );
        let ram_val_evaluation_params = ValEvaluationSumcheckParams::new_from_prover(
            &self.one_hot_params,
            &self.opening_accumulator,
            &self.initial_ram_state,
            self.trace.len(),
            &self.preprocessing.shared.ram,
            &self.program_io,
        );
        let ram_val_final_params = ValFinalSumcheckParams::new_from_prover(
            self.trace.len(),
            &self.opening_accumulator,
            &self.preprocessing.shared.ram,
            &self.program_io,
            self.one_hot_params.ram_k,
            &self.rw_config,
        );

        let registers_read_write_checking = RegistersReadWriteCheckingProver::initialize(
            registers_read_write_checking_params,
            self.trace.clone(),
            &self.preprocessing.shared.bytecode,
            &self.program_io.memory_layout,
        );
        let ram_val_evaluation = RamValEvaluationSumcheckProver::initialize(
            ram_val_evaluation_params,
            &self.trace,
            &self.preprocessing.shared.bytecode,
            &self.program_io.memory_layout,
        );
        let ram_val_final = ValFinalSumcheckProver::initialize(
            ram_val_final_params,
            &self.trace,
            &self.preprocessing.shared.bytecode,
            &self.program_io.memory_layout,
        );

        #[cfg(feature = "allocative")]
        {
            print_data_structure_heap_usage(
                "RegistersReadWriteCheckingProver",
                &registers_read_write_checking,
            );
            print_data_structure_heap_usage("RamValEvaluationSumcheckProver", &ram_val_evaluation);
            print_data_structure_heap_usage("ValFinalSumcheckProver", &ram_val_final);
        }

        let mut instances: Vec<Box<dyn SumcheckInstanceProver<_, _>>> = vec![
            Box::new(registers_read_write_checking),
            Box::new(ram_val_evaluation),
            Box::new(ram_val_final),
        ];

        #[cfg(feature = "allocative")]
        write_boxed_instance_flamegraph_svg(&instances, "stage4_start_flamechart.svg");
        tracing::info!("Stage 4 proving");

        let (sumcheck_proof, r_stage4, _initial_claim) =
            self.prove_batched_sumcheck(instances.iter_mut().map(|v| &mut **v as _).collect());
        #[cfg(feature = "allocative")]
        write_boxed_instance_flamegraph_svg(&instances, "stage4_end_flamechart.svg");
        drop_in_background_thread(instances);

        (sumcheck_proof, r_stage4)
    }

    #[tracing::instrument(skip_all)]
    pub(super) fn prove_stage5(
        &mut self,
    ) -> (
        SumcheckInstanceProof<F, C, ProofTranscript>,
        Vec<F::Challenge>,
    ) {
        #[cfg(not(target_arch = "wasm32"))]
        print_current_memory_usage("Stage 5 baseline");

        let lookups_read_raf_params = InstructionReadRafSumcheckParams::new(
            self.trace.len().log_2(),
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );
        let ram_ra_reduction_params = RaReductionParams::new(
            self.trace.len(),
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );
        let registers_val_evaluation_params =
            RegistersValEvaluationSumcheckParams::new(&self.opening_accumulator);

        let lookups_read_raf = InstructionReadRafSumcheckProver::initialize(
            lookups_read_raf_params,
            Arc::clone(&self.trace),
        );
        let ram_ra_reduction = RamRaClaimReductionSumcheckProver::initialize(
            ram_ra_reduction_params,
            &self.trace,
            &self.program_io.memory_layout,
            &self.one_hot_params,
        );
        let registers_val_evaluation = RegistersValEvaluationSumcheckProver::initialize(
            registers_val_evaluation_params,
            &self.trace,
            &self.preprocessing.shared.bytecode,
            &self.program_io.memory_layout,
        );

        #[cfg(feature = "allocative")]
        {
            print_data_structure_heap_usage("InstructionReadRafSumcheckProver", &lookups_read_raf);
            print_data_structure_heap_usage("RamRaClaimReductionSumcheckProver", &ram_ra_reduction);
            print_data_structure_heap_usage(
                "RegistersValEvaluationSumcheckProver",
                &registers_val_evaluation,
            );
        }

        let mut instances: Vec<Box<dyn SumcheckInstanceProver<_, _>>> = vec![
            Box::new(lookups_read_raf),
            Box::new(ram_ra_reduction),
            Box::new(registers_val_evaluation),
        ];

        #[cfg(feature = "allocative")]
        write_boxed_instance_flamegraph_svg(&instances, "stage5_start_flamechart.svg");
        tracing::info!("Stage 5 proving");

        let (sumcheck_proof, r_stage5, _initial_claim) =
            self.prove_batched_sumcheck(instances.iter_mut().map(|v| &mut **v as _).collect());
        #[cfg(feature = "allocative")]
        write_boxed_instance_flamegraph_svg(&instances, "stage5_end_flamechart.svg");
        drop_in_background_thread(instances);

        (sumcheck_proof, r_stage5)
    }

    #[tracing::instrument(skip_all)]
    pub(super) fn prove_stage6(
        &mut self,
    ) -> (
        SumcheckInstanceProof<F, C, ProofTranscript>,
        Vec<F::Challenge>,
    ) {
        #[cfg(not(target_arch = "wasm32"))]
        print_current_memory_usage("Stage 6 baseline");

        let bytecode_read_raf_params = BytecodeReadRafSumcheckParams::gen(
            &self.preprocessing.shared.bytecode,
            self.trace.len().log_2(),
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );

        let ram_hamming_booleanity_params =
            HammingBooleanitySumcheckParams::new(&self.opening_accumulator);

        let booleanity_params = BooleanitySumcheckParams::new(
            self.trace.len().log_2(),
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );

        let ram_ra_virtual_params = RamRaVirtualParams::new(
            self.trace.len(),
            &self.one_hot_params,
            &self.opening_accumulator,
        );
        let lookups_ra_virtual_params = InstructionRaSumcheckParams::new(
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );
        let inc_reduction_params = IncClaimReductionSumcheckParams::new(
            self.trace.len(),
            &self.opening_accumulator,
            &mut self.transcript,
        );

        if self.advice.trusted_advice_polynomial.is_some() {
            let trusted_advice_params = AdviceClaimReductionParams::new(
                AdviceKind::Trusted,
                &self.program_io.memory_layout,
                self.trace.len(),
                &self.opening_accumulator,
                &mut self.transcript,
                self.rw_config
                    .needs_single_advice_opening(self.trace.len().log_2()),
            );
            self.advice_reduction_prover_trusted = {
                let poly = self
                    .advice
                    .trusted_advice_polynomial
                    .clone()
                    .expect("trusted advice params exist but polynomial is missing");
                Some(AdviceClaimReductionProver::initialize(
                    trusted_advice_params,
                    poly,
                ))
            };
        }

        if self.advice.untrusted_advice_polynomial.is_some() {
            let untrusted_advice_params = AdviceClaimReductionParams::new(
                AdviceKind::Untrusted,
                &self.program_io.memory_layout,
                self.trace.len(),
                &self.opening_accumulator,
                &mut self.transcript,
                self.rw_config
                    .needs_single_advice_opening(self.trace.len().log_2()),
            );
            self.advice_reduction_prover_untrusted = {
                let poly = self
                    .advice
                    .untrusted_advice_polynomial
                    .clone()
                    .expect("untrusted advice params exist but polynomial is missing");
                Some(AdviceClaimReductionProver::initialize(
                    untrusted_advice_params,
                    poly,
                ))
            };
        }

        let mut bytecode_read_raf = BytecodeReadRafSumcheckProver::initialize(
            bytecode_read_raf_params,
            Arc::clone(&self.trace),
            Arc::clone(&self.preprocessing.shared.bytecode),
        );
        let mut ram_hamming_booleanity =
            HammingBooleanitySumcheckProver::initialize(ram_hamming_booleanity_params, &self.trace);

        let mut booleanity = BooleanitySumcheckProver::initialize(
            booleanity_params,
            &self.trace,
            &self.preprocessing.shared.bytecode,
            &self.program_io.memory_layout,
        );

        let mut ram_ra_virtual = RamRaVirtualSumcheckProver::initialize(
            ram_ra_virtual_params,
            &self.trace,
            &self.program_io.memory_layout,
            &self.one_hot_params,
        );
        let mut lookups_ra_virtual =
            LookupsRaSumcheckProver::initialize(lookups_ra_virtual_params, &self.trace);
        let mut inc_reduction =
            IncClaimReductionSumcheckProver::initialize(inc_reduction_params, self.trace.clone());

        #[cfg(feature = "allocative")]
        {
            print_data_structure_heap_usage("BytecodeReadRafSumcheckProver", &bytecode_read_raf);
            print_data_structure_heap_usage("BooleanitySumcheckProver", &booleanity);
            print_data_structure_heap_usage(
                "ram HammingBooleanitySumcheckProver",
                &ram_hamming_booleanity,
            );
            print_data_structure_heap_usage("RamRaSumcheckProver", &ram_ra_virtual);
            print_data_structure_heap_usage("LookupsRaSumcheckProver", &lookups_ra_virtual);
            print_data_structure_heap_usage("IncClaimReductionSumcheckProver", &inc_reduction);
            if let Some(ref advice) = self.advice_reduction_prover_trusted {
                print_data_structure_heap_usage("AdviceClaimReductionProver(trusted)", advice);
            }
            if let Some(ref advice) = self.advice_reduction_prover_untrusted {
                print_data_structure_heap_usage("AdviceClaimReductionProver(untrusted)", advice);
            }
        }

        let mut advice_trusted = self.advice_reduction_prover_trusted.take();
        let mut advice_untrusted = self.advice_reduction_prover_untrusted.take();

        let mut instances: Vec<&mut dyn SumcheckInstanceProver<_, _>> = vec![
            &mut bytecode_read_raf,
            &mut booleanity,
            &mut ram_hamming_booleanity,
            &mut ram_ra_virtual,
            &mut lookups_ra_virtual,
            &mut inc_reduction,
        ];
        if let Some(ref mut advice) = advice_trusted {
            instances.push(advice);
        }
        if let Some(ref mut advice) = advice_untrusted {
            instances.push(advice);
        }

        #[cfg(feature = "allocative")]
        write_instance_flamegraph_svg(&instances, "stage6_start_flamechart.svg");
        tracing::info!("Stage 6 proving");

        let (sumcheck_proof, r_stage6, _initial_claim) =
            self.prove_batched_sumcheck(instances.iter_mut().map(|v| &mut **v as _).collect());
        #[cfg(feature = "allocative")]
        write_instance_flamegraph_svg(&instances, "stage6_end_flamechart.svg");
        drop_in_background_thread(bytecode_read_raf);
        drop_in_background_thread(booleanity);
        drop_in_background_thread(ram_hamming_booleanity);
        drop_in_background_thread(ram_ra_virtual);
        drop_in_background_thread(lookups_ra_virtual);
        drop_in_background_thread(inc_reduction);

        self.advice_reduction_prover_trusted = advice_trusted;
        self.advice_reduction_prover_untrusted = advice_untrusted;

        (sumcheck_proof, r_stage6)
    }

    #[tracing::instrument(skip_all)]
    pub(super) fn prove_stage7(
        &mut self,
    ) -> (
        SumcheckInstanceProof<F, C, ProofTranscript>,
        Vec<F::Challenge>,
    ) {
        let hw_params = HammingWeightClaimReductionParams::new(
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );
        let hw_prover = HammingWeightClaimReductionProver::initialize(
            hw_params,
            &self.trace,
            &self.preprocessing.shared,
            &self.one_hot_params,
        );

        #[cfg(feature = "allocative")]
        print_data_structure_heap_usage("HammingWeightClaimReductionProver", &hw_prover);

        let mut instances: Vec<Box<dyn SumcheckInstanceProver<F, ProofTranscript>>> =
            vec![Box::new(hw_prover)];

        if let Some(mut advice_reduction_prover_trusted) =
            self.advice_reduction_prover_trusted.take()
        {
            if advice_reduction_prover_trusted
                .params
                .num_address_phase_rounds()
                > 0
            {
                advice_reduction_prover_trusted.params.phase = ReductionPhase::AddressVariables;
                instances.push(Box::new(advice_reduction_prover_trusted));
            }
        }
        if let Some(mut advice_reduction_prover_untrusted) =
            self.advice_reduction_prover_untrusted.take()
        {
            if advice_reduction_prover_untrusted
                .params
                .num_address_phase_rounds()
                > 0
            {
                advice_reduction_prover_untrusted.params.phase = ReductionPhase::AddressVariables;
                instances.push(Box::new(advice_reduction_prover_untrusted));
            }
        }

        #[cfg(feature = "allocative")]
        write_boxed_instance_flamegraph_svg(&instances, "stage7_start_flamechart.svg");
        tracing::info!("Stage 7 proving");

        let (sumcheck_proof, r_stage7, _initial_claim) =
            self.prove_batched_sumcheck(instances.iter_mut().map(|v| &mut **v as _).collect());
        #[cfg(feature = "allocative")]
        write_boxed_instance_flamegraph_svg(&instances, "stage7_end_flamechart.svg");
        drop_in_background_thread(instances);

        (sumcheck_proof, r_stage7)
    }
}
