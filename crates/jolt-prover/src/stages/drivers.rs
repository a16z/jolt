//! The per-stage [`StageProver`](crate::driver::StageProver) /
//! [`KernelSource`](crate::driver::KernelSource) impl expansions: one
//! member-list callback invocation per stage batch, each in a module that
//! imports the batch's relation and aggregate names so the derive-emitted
//! tokens resolve. This file is the prove side's complete stage-driver
//! surface — no stage's member list, order, or presence appears anywhere
//! else in this crate.

mod stage3 {
    use jolt_verifier::stages::stage3::outputs::{
        InstructionInput, RegistersClaimReduction, SpartanShift, Stage3Challenges,
        Stage3InputClaims, Stage3InputPoints, Stage3OutputClaims, Stage3OutputPoints,
        Stage3Sumchecks,
    };

    use crate::driver::impl_stage_prover;

    jolt_verifier::stage3_sumchecks_members!(impl_stage_prover);
}
