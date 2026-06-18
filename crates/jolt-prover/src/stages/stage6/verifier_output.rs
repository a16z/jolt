use jolt_backends::{
    SumcheckBooleanityOutput, SumcheckBytecodeReadRafOutput, SumcheckIncClaimReductionOutput,
    SumcheckInstructionRaVirtualizationOutput, SumcheckRamHammingBooleanityOutput,
    SumcheckRamRaVirtualizationOutput,
};
use jolt_field::Field;
use jolt_verifier::stages::stage6::inputs::{
    AdviceCyclePhaseOutputClaim, BooleanityOutputOpeningClaims, BytecodeReadRafOutputOpeningClaims,
    IncClaimReductionOutputOpeningClaims, InstructionRaVirtualizationOutputOpeningClaims,
    RamHammingBooleanityOutputOpeningClaims, RamRaVirtualizationOutputOpeningClaims,
    Stage6AddressPhaseClaims, Stage6AdviceCyclePhaseClaims, Stage6Claims,
};

#[expect(
    clippy::too_many_arguments,
    reason = "Stage 6 backend output assembly mirrors the verifier output groups one-to-one."
)]
pub(super) fn output_claims_from_backend<F: Field>(
    bytecode_read_raf: SumcheckBytecodeReadRafOutput<F>,
    booleanity: SumcheckBooleanityOutput<F>,
    ram_hamming_booleanity: SumcheckRamHammingBooleanityOutput<F>,
    ram_ra_virtualization: SumcheckRamRaVirtualizationOutput<F>,
    instruction_ra_virtualization: SumcheckInstructionRaVirtualizationOutput<F>,
    inc_claim_reduction: SumcheckIncClaimReductionOutput<F>,
    trusted_advice: Option<AdviceCyclePhaseOutputClaim<F>>,
    untrusted_advice: Option<AdviceCyclePhaseOutputClaim<F>>,
) -> Stage6Claims<F> {
    Stage6Claims {
        bytecode_read_raf: BytecodeReadRafOutputOpeningClaims {
            bytecode_ra: bytecode_read_raf.bytecode_ra,
        },
        booleanity: BooleanityOutputOpeningClaims {
            instruction_ra: booleanity.instruction_ra,
            bytecode_ra: booleanity.bytecode_ra,
            ram_ra: booleanity.ram_ra,
        },
        ram_hamming_booleanity: RamHammingBooleanityOutputOpeningClaims {
            ram_hamming_weight: ram_hamming_booleanity.ram_hamming_weight,
        },
        ram_ra_virtualization: RamRaVirtualizationOutputOpeningClaims {
            ram_ra: ram_ra_virtualization.ram_ra,
        },
        instruction_ra_virtualization: InstructionRaVirtualizationOutputOpeningClaims {
            committed_instruction_ra: instruction_ra_virtualization.instruction_ra,
        },
        inc_claim_reduction: IncClaimReductionOutputOpeningClaims {
            ram_inc: inc_claim_reduction.ram_inc,
            rd_inc: inc_claim_reduction.rd_inc,
        },
        advice_cycle_phase: Stage6AdviceCyclePhaseClaims {
            trusted: trusted_advice,
            untrusted: untrusted_advice,
        },
        address_phase: Stage6AddressPhaseClaims {
            bytecode_read_raf: F::zero(),
            booleanity: F::zero(),
            bytecode_val_stages: None,
        },
        bytecode_claim_reduction: None,
        program_image_claim_reduction: None,
    }
}
