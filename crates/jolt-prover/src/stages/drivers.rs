//! The per-stage [`StageProver`](crate::driver::StageProver) /
//! [`KernelSource`](crate::driver::KernelSource) impl expansions: one
//! member-list callback invocation per stage batch, each in a module that
//! imports the batch's relation and aggregate names so the derive-emitted
//! tokens resolve. This file is the prove side's complete stage-driver
//! surface — no stage's member list, order, or presence appears anywhere
//! else in this crate.

mod stage1 {
    use jolt_verifier::stages::stage1::outer_remainder::OuterRemainder;
    use jolt_verifier::stages::stage1::outputs::{
        Stage1BatchChallenges, Stage1BatchInputClaims, Stage1BatchInputPoints,
        Stage1BatchOutputClaims, Stage1BatchOutputPoints, Stage1BatchSumchecks,
    };

    use crate::driver::impl_stage_prover;

    jolt_verifier::stage1_batch_sumchecks_members!(impl_stage_prover);
}

mod stage2 {
    use jolt_verifier::stages::stage2::instruction_claim_reduction::InstructionClaimReduction;
    use jolt_verifier::stages::stage2::outputs::{
        Stage2BatchChallenges, Stage2BatchInputClaims, Stage2BatchInputPoints,
        Stage2BatchOutputClaims, Stage2BatchOutputPoints, Stage2BatchSumchecks,
    };
    use jolt_verifier::stages::stage2::product_remainder::ProductRemainder;
    use jolt_verifier::stages::stage2::ram_output_check::RamOutputCheck;
    use jolt_verifier::stages::stage2::ram_raf_evaluation::RamRafEvaluation;
    use jolt_verifier::stages::stage2::ram_read_write_checking::RamReadWriteChecking;

    use crate::driver::impl_stage_prover;

    jolt_verifier::stage2_batch_sumchecks_members!(impl_stage_prover);
}

mod stage3 {
    use jolt_verifier::stages::stage3::outputs::{
        InstructionInput, RegistersClaimReduction, SpartanShift, Stage3Challenges,
        Stage3InputClaims, Stage3InputPoints, Stage3OutputClaims, Stage3OutputPoints,
        Stage3Sumchecks,
    };

    use crate::driver::impl_stage_prover;

    jolt_verifier::stage3_sumchecks_members!(impl_stage_prover);
}

mod stage4 {
    use jolt_verifier::stages::stage4::outputs::{
        Stage4Challenges, Stage4InputClaims, Stage4InputPoints, Stage4OutputClaims,
        Stage4OutputPoints, Stage4Sumchecks,
    };
    use jolt_verifier::stages::stage4::ram_val_check::RamValCheck;
    use jolt_verifier::stages::stage4::registers_read_write_checking::RegistersReadWriteChecking;

    use crate::driver::impl_stage_prover;

    jolt_verifier::stage4_sumchecks_members!(impl_stage_prover);
}

mod stage6a {
    use jolt_verifier::stages::stage6a::booleanity::BooleanityAddressPhase;
    use jolt_verifier::stages::stage6a::bytecode_read_raf::BytecodeReadRafAddressPhase;
    use jolt_verifier::stages::stage6a::outputs::{
        Stage6aChallenges, Stage6aInputClaims, Stage6aInputPoints, Stage6aOutputClaims,
        Stage6aOutputPoints, Stage6aSumchecks,
    };

    use crate::driver::impl_stage_prover;

    jolt_verifier::stage6a_sumchecks_members!(impl_stage_prover);
}

mod stage7 {
    use jolt_verifier::stages::stage7::advice_address_phase::{
        TrustedAdviceAddressPhase, UntrustedAdviceAddressPhase,
    };
    use jolt_verifier::stages::stage7::committed_reduction_address_phase::{
        BytecodeReductionAddressPhase, ProgramImageReductionAddressPhase,
    };
    use jolt_verifier::stages::stage7::hamming_weight_claim_reduction::HammingWeightClaimReduction;
    use jolt_verifier::stages::stage7::outputs::{
        Stage7Challenges, Stage7InputClaims, Stage7InputPoints, Stage7OutputClaims,
        Stage7OutputPoints, Stage7Sumchecks,
    };

    use crate::driver::impl_stage_prover;

    jolt_verifier::stage7_sumchecks_members!(impl_stage_prover);
}

mod stage5 {
    use jolt_verifier::stages::stage5::outputs::{
        Stage5Challenges, Stage5InputClaims, Stage5InputPoints, Stage5OutputClaims,
        Stage5OutputPoints, Stage5Sumchecks,
    };
    use jolt_verifier::stages::stage5::ram_ra_claim_reduction::RamRaClaimReduction;
    use jolt_verifier::stages::stage5::registers_val_evaluation::RegistersValEvaluation;
    use jolt_verifier::stages::stage5::InstructionReadRaf;

    use crate::driver::impl_stage_prover;

    jolt_verifier::stage5_sumchecks_members!(impl_stage_prover);
}
