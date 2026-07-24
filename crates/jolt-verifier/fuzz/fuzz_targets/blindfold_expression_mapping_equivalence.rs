#![no_main]

//! Differential oracle for Jolt claim expressions crossing into BlindFold.
//!
//! Verifier-side BlindFold construction remaps `JoltChallengeId` factors into
//! public inputs while leaving openings and derived values in their respective
//! source families. This target generates bounded production-shaped claim
//! expressions and checks that the remapped expression evaluates identically.

use jolt_claims::{
    protocols::jolt::{
        AdviceClaimReductionPublic, BooleanityChallenge, BytecodeChunkReconstructionChallenge,
        BytecodeChunkReconstructionPublic, BytecodeClaimReductionChallenge,
        BytecodeClaimReductionPublic, BytecodeReadRafChallenge, BytecodeReadRafPublic,
        BytecodeRegisterLane, HammingWeightClaimReductionChallenge,
        HammingWeightClaimReductionPublic, IncClaimReductionChallenge,
        IncClaimReductionPublic, InstructionClaimReductionChallenge,
        InstructionClaimReductionPublic, InstructionInputChallenge, InstructionInputPublic,
        InstructionRaVirtualizationChallenge, InstructionRaVirtualizationPublic,
        InstructionReadRafChallenge, InstructionReadRafPublic, JoltAdviceKind, JoltChallengeId,
        JoltCommittedPolynomial, JoltDerivedId, JoltExpr, JoltOpeningId, JoltRelationId,
        JoltVirtualPolynomial, ProgramImageClaimReductionPublic, ProgramImageReconstructionPublic,
        RamHammingBooleanityPublic, RamOutputCheckPublic, RamRaClaimReductionChallenge,
        RamRaClaimReductionPublic, RamRaVirtualizationPublic, RamRafEvaluationPublic,
        RamReadWriteChallenge, RamReadWritePublic, RamValCheckChallenge, RamValCheckPublic,
        RegistersClaimReductionChallenge, RegistersClaimReductionPublic,
        RegistersReadWriteChallenge, RegistersReadWritePublic, RegistersValEvaluationPublic,
        SpartanOuterPublic, SpartanProductVirtualizationPublic, SpartanShiftChallenge,
        SpartanShiftPublic, TrustedAdviceReconstructionPublic, UntrustedAdviceReconstructionChallenge,
        UntrustedAdviceReconstructionPublic,
    },
    Source, Term,
};
use jolt_field::{Fr, FromPrimitiveInt};
use jolt_verifier::stages::zk::blindfold::evaluate_mapped_expression;
use libfuzzer_sys::fuzz_target;

const MAX_TERMS: usize = 12;
const MAX_FACTORS: usize = 5;

fn word(data: &[u8], cursor: &mut usize) -> u64 {
    let mut bytes = [0u8; 8];
    for byte in &mut bytes {
        *byte = data[*cursor % data.len()];
        *cursor += 1;
    }
    u64::from_le_bytes(bytes)
}

fn lookup<T: PartialEq>(ids: &[T], values: &[Fr], wanted: &T) -> Fr {
    for (id, value) in ids.iter().zip(values) {
        if id == wanted {
            return *value;
        }
    }
    panic!("generated expression referenced an unregistered source");
}

fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }

    let opening_ids = [
        JoltOpeningId::committed(
            JoltCommittedPolynomial::RdInc,
            JoltRelationId::IncClaimReduction,
        ),
        JoltOpeningId::committed(
            JoltCommittedPolynomial::RamInc,
            JoltRelationId::RamReadWriteChecking,
        ),
        JoltOpeningId::committed(
            JoltCommittedPolynomial::InstructionRa(0),
            JoltRelationId::InstructionRaVirtualization,
        ),
        JoltOpeningId::committed(
            JoltCommittedPolynomial::BytecodeRa(1),
            JoltRelationId::BytecodeReadRaf,
        ),
        JoltOpeningId::committed(
            JoltCommittedPolynomial::RamRa(2),
            JoltRelationId::RamRaVirtualization,
        ),
        JoltOpeningId::virtual_polynomial(
            JoltVirtualPolynomial::Product,
            JoltRelationId::SpartanProductVirtualization,
        ),
        JoltOpeningId::virtual_polynomial(JoltVirtualPolynomial::RamVal, JoltRelationId::RamValCheck),
        JoltOpeningId::virtual_polynomial(
            JoltVirtualPolynomial::RegistersVal,
            JoltRelationId::RegistersValEvaluation,
        ),
        JoltOpeningId::virtual_polynomial(
            JoltVirtualPolynomial::InstructionRaf,
            JoltRelationId::InstructionReadRaf,
        ),
        JoltOpeningId::trusted_advice(JoltRelationId::AdviceClaimReduction),
        JoltOpeningId::untrusted_advice(JoltRelationId::AdviceClaimReduction),
    ];
    let challenge_ids = [
        JoltChallengeId::from(RamReadWriteChallenge::Gamma),
        JoltChallengeId::from(RamValCheckChallenge::Gamma),
        JoltChallengeId::from(RamRaClaimReductionChallenge::Gamma),
        JoltChallengeId::from(BooleanityChallenge::Gamma),
        JoltChallengeId::from(IncClaimReductionChallenge::Gamma),
        JoltChallengeId::from(HammingWeightClaimReductionChallenge::Gamma),
        JoltChallengeId::from(BytecodeReadRafChallenge::Gamma),
        JoltChallengeId::from(BytecodeReadRafChallenge::Stage1Gamma),
        JoltChallengeId::from(BytecodeClaimReductionChallenge::Eta),
        JoltChallengeId::from(SpartanShiftChallenge::Gamma),
        JoltChallengeId::from(RegistersReadWriteChallenge::Gamma),
        JoltChallengeId::from(RegistersClaimReductionChallenge::Gamma),
        JoltChallengeId::from(InstructionClaimReductionChallenge::Gamma),
        JoltChallengeId::from(InstructionInputChallenge::Gamma),
        JoltChallengeId::from(InstructionReadRafChallenge::Gamma),
        JoltChallengeId::from(InstructionRaVirtualizationChallenge::Gamma),
        JoltChallengeId::from(UntrustedAdviceReconstructionChallenge::Gamma),
        JoltChallengeId::from(BytecodeChunkReconstructionChallenge::Gamma),
    ];
    let derived_ids = [
        JoltDerivedId::from(RamReadWritePublic::EqCycle),
        JoltDerivedId::from(RamValCheckPublic::LtCyclePlusGamma),
        JoltDerivedId::from(RamValCheckPublic::InitEval),
        JoltDerivedId::from(RamValCheckPublic::InitSelector(JoltAdviceKind::Trusted)),
        JoltDerivedId::from(RamRafEvaluationPublic::UnmapAddress),
        JoltDerivedId::from(RamOutputCheckPublic::IoMask),
        JoltDerivedId::from(RamRaClaimReductionPublic::EqCycleValCheck),
        JoltDerivedId::from(RamRaVirtualizationPublic::EqCycle),
        JoltDerivedId::from(RamHammingBooleanityPublic::EqCycle),
        JoltDerivedId::from(IncClaimReductionPublic::EqRegistersValEvaluation),
        JoltDerivedId::from(HammingWeightClaimReductionPublic::EqVirtualization(0)),
        JoltDerivedId::from(BytecodeReadRafPublic::StageValue(0)),
        JoltDerivedId::from(BytecodeReadRafPublic::StageCycleEq(0)),
        JoltDerivedId::from(BytecodeReadRafPublic::Entry),
        JoltDerivedId::from(AdviceClaimReductionPublic::FinalScale(JoltAdviceKind::Untrusted)),
        JoltDerivedId::from(BytecodeClaimReductionPublic::ChunkOutputWeight(0)),
        JoltDerivedId::from(ProgramImageClaimReductionPublic::FinalScale),
        JoltDerivedId::from(SpartanShiftPublic::EqPlusOneProduct),
        JoltDerivedId::from(SpartanProductVirtualizationPublic::TauKernel),
        JoltDerivedId::from(SpartanOuterPublic::AzWeight(0)),
        JoltDerivedId::from(SpartanOuterPublic::BzConstant),
        JoltDerivedId::from(RegistersReadWritePublic::EqCycle),
        JoltDerivedId::from(RegistersValEvaluationPublic::LtCycle),
        JoltDerivedId::from(RegistersClaimReductionPublic::EqSpartan),
        JoltDerivedId::from(InstructionClaimReductionPublic::EqSpartan),
        JoltDerivedId::from(InstructionInputPublic::EqProduct),
        JoltDerivedId::from(InstructionReadRafPublic::EqRafFlag),
        JoltDerivedId::from(InstructionRaVirtualizationPublic::EqCycle),
        JoltDerivedId::from(UntrustedAdviceReconstructionPublic::ByteDecode),
        JoltDerivedId::from(TrustedAdviceReconstructionPublic::ByteDecode),
        JoltDerivedId::from(ProgramImageReconstructionPublic::ByteDecode),
        JoltDerivedId::from(BytecodeChunkReconstructionPublic::RegisterSelectorWeight(
            BytecodeRegisterLane::Rs1,
        )),
        JoltDerivedId::from(BytecodeChunkReconstructionPublic::LaneWeight(0)),
    ];

    let mut cursor = 1;
    let opening_values: Vec<_> = opening_ids
        .iter()
        .map(|_| Fr::from_u64(word(data, &mut cursor)))
        .collect();
    let challenge_values: Vec<_> = challenge_ids
        .iter()
        .map(|_| Fr::from_u64(word(data, &mut cursor)))
        .collect();
    let derived_values: Vec<_> = derived_ids
        .iter()
        .map(|_| Fr::from_u64(word(data, &mut cursor)))
        .collect();

    let term_count = data[0] as usize % (MAX_TERMS + 1);
    let mut terms = Vec::with_capacity(term_count);
    for _ in 0..term_count {
        let coefficient = Fr::from_u64(word(data, &mut cursor));
        let factor_count = data[cursor % data.len()] as usize % (MAX_FACTORS + 1);
        cursor += 1;
        let mut factors = Vec::with_capacity(factor_count);
        for _ in 0..factor_count {
            let selector = word(data, &mut cursor) as usize;
            factors.push(match selector % 3 {
                0 => Source::Opening(opening_ids[(selector / 3) % opening_ids.len()]),
                1 => Source::Challenge(challenge_ids[(selector / 3) % challenge_ids.len()]),
                _ => Source::Derived(derived_ids[(selector / 3) % derived_ids.len()]),
            });
        }
        terms.push(Term {
            coefficient,
            factors,
        });
    }
    let expression: JoltExpr<Fr> = JoltExpr { terms };
    let direct = expression.evaluate(
        |id| lookup(&opening_ids, &opening_values, id),
        |id| lookup(&challenge_ids, &challenge_values, id),
        |id| lookup(&derived_ids, &derived_values, id),
    );
    let mapped = evaluate_mapped_expression(
        expression,
        |id| lookup(&opening_ids, &opening_values, id),
        |id| lookup(&challenge_ids, &challenge_values, id),
        |id| lookup(&derived_ids, &derived_values, id),
    );

    assert_eq!(
        direct, mapped,
        "BlindFold expression remapping changed the claim value"
    );
});
