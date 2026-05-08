use jolt_core::poly::opening_proof::SumcheckId as S;
use jolt_core::poly::opening_proof::{OpeningId, SumcheckId};
use jolt_core::zkvm::instruction::{CircuitFlags, InstructionFlags};
use jolt_core::zkvm::witness::VirtualPolynomial as V;
use jolt_core::zkvm::witness::{CommittedPolynomial, VirtualPolynomial};
use jolt_verifier::stages::common::OpeningClaimPlan;

pub(crate) fn core_opening_id(claim: &OpeningClaimPlan) -> Option<OpeningId> {
    let sumcheck = core_sumcheck_id(claim.symbol)?;
    match claim.claim_kind {
        "committed" => core_committed_polynomial(claim.oracle)
            .map(|polynomial| OpeningId::committed(polynomial, sumcheck)),
        "virtual" => core_virtual_polynomial(claim.oracle)
            .map(|polynomial| OpeningId::virt(polynomial, sumcheck)),
        _ => None,
    }
}

fn core_sumcheck_id(symbol: &'static str) -> Option<SumcheckId> {
    let mappings = [
        (
            "stage2.product_virtual.uniskip.",
            S::SpartanProductVirtualization,
        ),
        (
            "stage2.product_virtual.remainder.",
            S::SpartanProductVirtualization,
        ),
        ("stage2.ram_read_write.", S::RamReadWriteChecking),
        (
            "stage2.instruction_lookup.claim_reduction.",
            S::InstructionClaimReduction,
        ),
        ("stage2.ram_raf.", S::RamRafEvaluation),
        ("stage2.ram_output.", S::RamOutputCheck),
        ("stage3.spartan_shift.", S::SpartanShift),
        (
            "stage3.instruction_input.",
            S::InstructionInputVirtualization,
        ),
        (
            "stage3.registers_claim_reduction.",
            S::RegistersClaimReduction,
        ),
        ("stage6.bytecode_read_raf.", S::BytecodeReadRaf),
        ("stage6.booleanity.", S::Booleanity),
        ("stage6.hamming_booleanity.", S::RamHammingBooleanity),
        ("stage6.ram_ra_virtual.", S::RamRaVirtualization),
        (
            "stage6.instruction_ra_virtual.",
            S::InstructionRaVirtualization,
        ),
        ("stage6.inc_claim_reduction.", S::IncClaimReduction),
    ];
    mappings
        .iter()
        .find_map(|(prefix, id)| symbol.starts_with(prefix).then_some(*id))
}

fn core_committed_polynomial(oracle: &'static str) -> Option<CommittedPolynomial> {
    if let Some(index) = oracle_index(oracle, "InstructionRa_") {
        return Some(CommittedPolynomial::InstructionRa(index));
    }
    if let Some(index) = oracle_index(oracle, "BytecodeRa_") {
        return Some(CommittedPolynomial::BytecodeRa(index));
    }
    if let Some(index) = oracle_index(oracle, "RamRa_") {
        return Some(CommittedPolynomial::RamRa(index));
    }
    match oracle {
        "RamInc" => Some(CommittedPolynomial::RamInc),
        "RdInc" => Some(CommittedPolynomial::RdInc),
        _ => None,
    }
}

fn oracle_index(oracle: &'static str, prefix: &'static str) -> Option<usize> {
    oracle.strip_prefix(prefix)?.parse().ok()
}

fn core_virtual_polynomial(oracle: &'static str) -> Option<VirtualPolynomial> {
    match oracle {
        "UnivariateSkip" => Some(V::UnivariateSkip),
        "RamVal" => Some(V::RamVal),
        "RamRa" => Some(V::RamRa),
        "LeftInstructionInput" => Some(V::LeftInstructionInput),
        "RightInstructionInput" => Some(V::RightInstructionInput),
        "LookupOutput" => Some(V::LookupOutput),
        "InstructionFlagBranch" => Some(V::InstructionFlags(InstructionFlags::Branch)),
        "NextIsNoop" => Some(V::NextIsNoop),
        "LeftLookupOperand" => Some(V::LeftLookupOperand),
        "RightLookupOperand" => Some(V::RightLookupOperand),
        "RamValFinal" => Some(V::RamValFinal),
        "UnexpandedPC" => Some(V::UnexpandedPC),
        "PC" => Some(V::PC),
        "InstructionFlagIsNoop" => Some(V::InstructionFlags(InstructionFlags::IsNoop)),
        "InstructionFlagLeftOperandIsRs1Value" => {
            Some(V::InstructionFlags(InstructionFlags::LeftOperandIsRs1Value))
        }
        "Rs1Value" => Some(V::Rs1Value),
        "InstructionFlagLeftOperandIsPC" => {
            Some(V::InstructionFlags(InstructionFlags::LeftOperandIsPC))
        }
        "InstructionFlagRightOperandIsRs2Value" => Some(V::InstructionFlags(
            InstructionFlags::RightOperandIsRs2Value,
        )),
        "Rs2Value" => Some(V::Rs2Value),
        "InstructionFlagRightOperandIsImm" => {
            Some(V::InstructionFlags(InstructionFlags::RightOperandIsImm))
        }
        "Imm" => Some(V::Imm),
        "RdWriteValue" => Some(V::RdWriteValue),
        "HammingWeight" => Some(V::RamHammingWeight),
        _ => core_circuit_flag(oracle).map(V::OpFlags),
    }
}

fn core_circuit_flag(oracle: &'static str) -> Option<CircuitFlags> {
    match oracle {
        "OpFlagJump" => Some(CircuitFlags::Jump),
        "OpFlagWriteLookupOutputToRD" => Some(CircuitFlags::WriteLookupOutputToRD),
        "OpFlagVirtualInstruction" => Some(CircuitFlags::VirtualInstruction),
        "OpFlagIsFirstInSequence" => Some(CircuitFlags::IsFirstInSequence),
        _ => None,
    }
}
