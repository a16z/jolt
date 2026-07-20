//! Fixture-driven tamper suite for the akita path.
//!
//! Three layers, all over real packed-prover fixtures:
//!
//! - An exhaustive typed sweep ([`every_clear_claim_wire_rejects_offset`]):
//!   every field-element leaf of the clear claims is offset by one, one at a
//!   time, and the verifier must reject each. The visitor
//!   ([`for_each_scalar_mut`]) fully destructures every aggregate, so a future
//!   claim wire cannot be added without failing to compile until it is covered.
//! - A byte-level commitment sweep ([`every_commitment_wire_rejects_perturbation`]):
//!   every serde leaf of the `OneHotTrace` (and untrusted-advice) commitment objects
//!   — layout digest, declared dimensions, backend flavor, backend bytes — is
//!   perturbed; a deserialization failure or a verifier rejection both count.
//! - Proof-shape tampers ([`akita_proof_shape_tampers_reject`],
//!   [`akita_advice_commitment_presence_rejects`]): dropped reconstruction /
//!   auxiliary proofs, a swapped phase proof, an auxiliary evaluation offset,
//!   and an absent trusted-advice commitment.
//!
//! Together these are the active coverage behind the akita
//! `TamperCoverage::Active` manifest entries.

#![expect(
    clippy::expect_used,
    clippy::panic,
    reason = "fixture tamper tests should fail loudly when the stored proof shape changes"
)]

use jolt_claims::protocols::jolt::lattice::relations::{
    advice_reconstruction::{
        TrustedAdviceReconstructionOutputClaims, UntrustedAdviceReconstructionOutputClaims,
    },
    booleanity::LatticeBooleanityOutputClaims,
    bytecode_reconstruction::BytecodeChunkReconstructionOutputClaims,
    program_image_reconstruction::ProgramImageReconstructionOutputClaims,
    read_raf::LatticeBytecodeReadRafOutputClaims,
};
use jolt_field::Field;
use jolt_prover_legacy::zkvm::packed::{AkitaField, AkitaJoltProof, AkitaScheme};
use jolt_verifier::proof::{ClearProofClaims, JoltProofClaims};
use jolt_verifier::stages::{
    stage1::{
        outputs::{Stage1BatchOutputClaims, Stage1OutputClaims},
        OuterRemainderOutputClaims,
    },
    stage2::outputs::{
        InstructionClaimReductionOutputClaims, ProductRemainderOutputClaims,
        RamOutputCheckOutputClaims, RamRafEvaluationOutputClaims, RamReadWriteOutputClaims,
        Stage2BatchOutputClaims, Stage2OutputClaims,
    },
    stage3::outputs::{
        InstructionInputOutputClaims, RegistersClaimReductionOutputClaims,
        SpartanShiftOutputClaims, Stage3OutputClaims,
    },
    stage4::{
        outputs::Stage4OutputClaims, RamValCheckOutputClaims, RegistersReadWriteOutputClaims,
    },
    stage5::{
        outputs::Stage5OutputClaims, InstructionReadRafOutputClaims,
        RamRaClaimReductionOutputClaims, RegistersValEvaluationOutputClaims,
    },
    stage6a::outputs::{
        BooleanityAddressPhaseOutputClaims, BytecodeReadRafAddressPhaseOutputClaims,
        Stage6aOutputClaims,
    },
    stage6b::outputs::{
        BytecodeReductionCyclePhaseOutputClaims, InstructionRaVirtualizationOutputClaims,
        ProgramImageReductionCyclePhaseOutputClaims, RamHammingBooleanityOutputClaims,
        RamRaVirtualizationOutputClaims, Stage6bOutputClaims, TrustedAdviceCyclePhaseOutputClaims,
        UntrustedAdviceCyclePhaseOutputClaims,
    },
    stage7::{
        advice_address_phase::{
            TrustedAdviceAddressPhaseOutputClaims, UntrustedAdviceAddressPhaseOutputClaims,
        },
        committed_reduction_address_phase::{
            BytecodeReductionAddressPhaseOutputClaims,
            ProgramImageReductionAddressPhaseOutputClaims,
        },
        hamming_weight_claim_reduction::HammingWeightClaimReductionOutputClaims,
        outputs::Stage7OutputClaims,
    },
    stage8::reconstruction::ReconstructionOutputClaims,
};

use crate::support::akita_fixtures::{
    akita_advice_case, akita_committed_muldiv_case, akita_muldiv_case, AkitaFixtureCase,
};
use crate::support::assert_rejects;

/// The single packed `OneHotTrace` commitment object type (also the advice object
/// type): the concrete `Commitment::Output` of the akita scheme.
type AkitaCommitment = <AkitaScheme as jolt_crypto::Commitment>::Output;

fn one() -> AkitaField {
    AkitaField::from_u64(1)
}

fn clear_claims_mut(proof: &mut AkitaJoltProof) -> &mut ClearProofClaims<AkitaField> {
    match &mut proof.claims {
        JoltProofClaims::Clear(claims) => claims,
        JoltProofClaims::Zk { .. } => panic!("packed akita fixtures always carry clear claims"),
    }
}

/// Visit every field-element leaf of the clear claims by mutable reference, in
/// a fixed order. Each aggregate is fully destructured (no `..`), so a new
/// claim field is a compile error here until it is threaded through — the wire
/// coverage the sweep depends on cannot silently regress.
fn for_each_scalar_mut<F: Field>(claims: &mut ClearProofClaims<F>, f: &mut impl FnMut(&mut F)) {
    let f: &mut dyn FnMut(&mut F) = f;
    let ClearProofClaims {
        stage1,
        stage2,
        stage3,
        stage4,
        stage5,
        stage6a,
        stage6b,
        stage7,
        reconstruction,
    } = claims;
    visit_stage1(stage1, f);
    visit_stage2(stage2, f);
    visit_stage3(stage3, f);
    visit_stage4(stage4, f);
    visit_stage5(stage5, f);
    visit_stage6a(stage6a, f);
    visit_stage6b(stage6b, f);
    visit_stage7(stage7, f);
    visit_reconstruction(reconstruction, f);
}

fn visit_stage1<F: Field>(claims: &mut Stage1OutputClaims<F>, f: &mut dyn FnMut(&mut F)) {
    let Stage1OutputClaims {
        uniskip_output_claim,
        outer,
    } = claims;
    f(uniskip_output_claim);
    let Stage1BatchOutputClaims { outer_remainder } = outer;
    let OuterRemainderOutputClaims {
        left_instruction_input,
        right_instruction_input,
        product,
        should_branch,
        pc,
        unexpanded_pc,
        imm,
        ram_address,
        rs1_value,
        rs2_value,
        rd_write_value,
        ram_read_value,
        ram_write_value,
        left_lookup_operand,
        right_lookup_operand,
        next_unexpanded_pc,
        next_pc,
        next_is_virtual,
        next_is_first_in_sequence,
        lookup_output,
        should_jump,
        add_operands,
        subtract_operands,
        multiply_operands,
        load,
        store,
        jump,
        write_lookup_output_to_rd,
        virtual_instruction,
        assert,
        do_not_update_unexpanded_pc,
        advice,
        is_compressed,
        is_first_in_sequence,
        is_last_in_sequence,
    } = outer_remainder;
    for scalar in [
        left_instruction_input,
        right_instruction_input,
        product,
        should_branch,
        pc,
        unexpanded_pc,
        imm,
        ram_address,
        rs1_value,
        rs2_value,
        rd_write_value,
        ram_read_value,
        ram_write_value,
        left_lookup_operand,
        right_lookup_operand,
        next_unexpanded_pc,
        next_pc,
        next_is_virtual,
        next_is_first_in_sequence,
        lookup_output,
        should_jump,
        add_operands,
        subtract_operands,
        multiply_operands,
        load,
        store,
        jump,
        write_lookup_output_to_rd,
        virtual_instruction,
        assert,
        do_not_update_unexpanded_pc,
        advice,
        is_compressed,
        is_first_in_sequence,
        is_last_in_sequence,
    ] {
        f(scalar);
    }
}

fn visit_stage2<F: Field>(claims: &mut Stage2OutputClaims<F>, f: &mut dyn FnMut(&mut F)) {
    let Stage2OutputClaims {
        product_uniskip_output_claim,
        batch_outputs,
    } = claims;
    f(product_uniskip_output_claim);
    let Stage2BatchOutputClaims {
        ram_read_write,
        product_remainder,
        instruction_claim_reduction,
        ram_raf_evaluation,
        ram_output_check,
    } = batch_outputs;
    let RamReadWriteOutputClaims { val, ra, inc } = ram_read_write;
    let ProductRemainderOutputClaims {
        left_instruction_input,
        right_instruction_input,
        jump_flag,
        write_lookup_output_to_rd,
        lookup_output,
        branch_flag,
        next_is_noop,
        virtual_instruction,
    } = product_remainder;
    let InstructionClaimReductionOutputClaims {
        lookup_output: icr_lookup_output,
        left_lookup_operand,
        right_lookup_operand,
        left_instruction_input: icr_left_instruction_input,
        right_instruction_input: icr_right_instruction_input,
    } = instruction_claim_reduction;
    let RamRafEvaluationOutputClaims { ram_ra } = ram_raf_evaluation;
    let RamOutputCheckOutputClaims { val_final } = ram_output_check;
    for scalar in [
        val,
        ra,
        inc,
        left_instruction_input,
        right_instruction_input,
        jump_flag,
        write_lookup_output_to_rd,
        lookup_output,
        branch_flag,
        next_is_noop,
        virtual_instruction,
        icr_lookup_output,
        left_lookup_operand,
        right_lookup_operand,
        icr_left_instruction_input,
        icr_right_instruction_input,
        ram_ra,
        val_final,
    ] {
        f(scalar);
    }
}

fn visit_stage3<F: Field>(claims: &mut Stage3OutputClaims<F>, f: &mut dyn FnMut(&mut F)) {
    let Stage3OutputClaims {
        shift,
        instruction_input,
        registers_claim_reduction,
    } = claims;
    let SpartanShiftOutputClaims {
        unexpanded_pc,
        pc,
        is_virtual,
        is_first_in_sequence,
        is_noop,
    } = shift;
    let InstructionInputOutputClaims {
        left_operand_is_rs1,
        rs1_value,
        left_operand_is_pc,
        unexpanded_pc: ii_unexpanded_pc,
        right_operand_is_rs2,
        rs2_value,
        right_operand_is_imm,
        imm,
    } = instruction_input;
    let RegistersClaimReductionOutputClaims {
        rd_write_value,
        rs1_value: rcr_rs1_value,
        rs2_value: rcr_rs2_value,
    } = registers_claim_reduction;
    for scalar in [
        unexpanded_pc,
        pc,
        is_virtual,
        is_first_in_sequence,
        is_noop,
        left_operand_is_rs1,
        rs1_value,
        left_operand_is_pc,
        ii_unexpanded_pc,
        right_operand_is_rs2,
        rs2_value,
        right_operand_is_imm,
        imm,
        rd_write_value,
        rcr_rs1_value,
        rcr_rs2_value,
    ] {
        f(scalar);
    }
}

fn visit_stage4<F: Field>(claims: &mut Stage4OutputClaims<F>, f: &mut dyn FnMut(&mut F)) {
    let Stage4OutputClaims {
        registers_read_write,
        ram_val_check,
    } = claims;
    let RegistersReadWriteOutputClaims {
        registers_val,
        rs1_ra,
        rs2_ra,
        rd_wa,
        rd_inc,
    } = registers_read_write;
    for scalar in [registers_val, rs1_ra, rs2_ra, rd_wa, rd_inc] {
        f(scalar);
    }
    let RamValCheckOutputClaims {
        untrusted_advice,
        trusted_advice,
        program_image,
        ram_ra,
        ram_inc,
    } = ram_val_check;
    for scalar in [untrusted_advice, trusted_advice, program_image]
        .into_iter()
        .flatten()
    {
        f(scalar);
    }
    for scalar in [ram_ra, ram_inc] {
        f(scalar);
    }
}

fn visit_stage5<F: Field>(claims: &mut Stage5OutputClaims<F>, f: &mut dyn FnMut(&mut F)) {
    let Stage5OutputClaims {
        instruction_read_raf,
        ram_ra_claim_reduction,
        registers_val_evaluation,
    } = claims;
    let InstructionReadRafOutputClaims {
        lookup_table_flags,
        instruction_ra,
        instruction_raf_flag,
    } = instruction_read_raf;
    for scalar in lookup_table_flags.iter_mut() {
        f(scalar);
    }
    for scalar in instruction_ra.iter_mut() {
        f(scalar);
    }
    f(instruction_raf_flag);
    let RamRaClaimReductionOutputClaims { ram_ra } = ram_ra_claim_reduction;
    f(ram_ra);
    let RegistersValEvaluationOutputClaims { rd_inc, rd_wa } = registers_val_evaluation;
    for scalar in [rd_inc, rd_wa] {
        f(scalar);
    }
}

fn visit_stage6a<F: Field>(claims: &mut Stage6aOutputClaims<F>, f: &mut dyn FnMut(&mut F)) {
    let Stage6aOutputClaims {
        bytecode_read_raf,
        booleanity,
    } = claims;
    let BytecodeReadRafAddressPhaseOutputClaims {
        intermediate,
        val_stages,
    } = bytecode_read_raf;
    f(intermediate);
    for scalar in val_stages.iter_mut() {
        f(scalar);
    }
    let BooleanityAddressPhaseOutputClaims {
        intermediate: booleanity_intermediate,
    } = booleanity;
    f(booleanity_intermediate);
}

fn visit_stage6b<F: Field>(claims: &mut Stage6bOutputClaims<F>, f: &mut dyn FnMut(&mut F)) {
    let Stage6bOutputClaims {
        bytecode_read_raf,
        booleanity,
        ram_hamming_booleanity,
        ram_ra_virtualization,
        instruction_ra_virtualization,
        trusted_advice,
        untrusted_advice,
        bytecode_reduction,
        program_image_reduction,
    } = claims;
    let LatticeBytecodeReadRafOutputClaims {
        bytecode_ra,
        fused_inc,
    } = bytecode_read_raf;
    for scalar in bytecode_ra.iter_mut() {
        f(scalar);
    }
    f(fused_inc);
    let LatticeBooleanityOutputClaims {
        instruction_ra,
        bytecode_ra: booleanity_bytecode_ra,
        ram_ra,
        unsigned_inc_chunks,
        unsigned_inc_msb,
    } = booleanity;
    for scalar in instruction_ra.iter_mut() {
        f(scalar);
    }
    for scalar in booleanity_bytecode_ra.iter_mut() {
        f(scalar);
    }
    for scalar in ram_ra.iter_mut() {
        f(scalar);
    }
    for scalar in unsigned_inc_chunks.iter_mut() {
        f(scalar);
    }
    f(unsigned_inc_msb);
    let RamHammingBooleanityOutputClaims { ram_hamming_weight } = ram_hamming_booleanity;
    f(ram_hamming_weight);
    let RamRaVirtualizationOutputClaims {
        ram_ra: virt_ram_ra,
    } = ram_ra_virtualization;
    for scalar in virt_ram_ra.iter_mut() {
        f(scalar);
    }
    let InstructionRaVirtualizationOutputClaims {
        committed_instruction_ra,
    } = instruction_ra_virtualization;
    for scalar in committed_instruction_ra.iter_mut() {
        f(scalar);
    }
    if let Some(TrustedAdviceCyclePhaseOutputClaims { trusted }) = trusted_advice {
        f(trusted);
    }
    if let Some(UntrustedAdviceCyclePhaseOutputClaims { untrusted }) = untrusted_advice {
        f(untrusted);
    }
    if let Some(BytecodeReductionCyclePhaseOutputClaims {
        intermediate,
        chunks,
    }) = bytecode_reduction
    {
        if let Some(scalar) = intermediate {
            f(scalar);
        }
        for scalar in chunks.iter_mut() {
            f(scalar);
        }
    }
    if let Some(ProgramImageReductionCyclePhaseOutputClaims { program_image }) =
        program_image_reduction
    {
        f(program_image);
    }
}

fn visit_stage7<F: Field>(claims: &mut Stage7OutputClaims<F>, f: &mut dyn FnMut(&mut F)) {
    let Stage7OutputClaims {
        hamming_weight_claim_reduction,
        trusted_advice,
        untrusted_advice,
        bytecode_address_phase,
        program_image_address_phase,
    } = claims;
    let HammingWeightClaimReductionOutputClaims {
        instruction_ra,
        bytecode_ra,
        ram_ra,
        unsigned_inc_chunks,
        unsigned_inc_msb,
    } = hamming_weight_claim_reduction;
    for scalar in instruction_ra.iter_mut() {
        f(scalar);
    }
    for scalar in bytecode_ra.iter_mut() {
        f(scalar);
    }
    for scalar in ram_ra.iter_mut() {
        f(scalar);
    }
    for scalar in unsigned_inc_chunks.iter_mut() {
        f(scalar);
    }
    f(unsigned_inc_msb);
    if let Some(TrustedAdviceAddressPhaseOutputClaims { trusted }) = trusted_advice {
        f(trusted);
    }
    if let Some(UntrustedAdviceAddressPhaseOutputClaims { untrusted }) = untrusted_advice {
        f(untrusted);
    }
    if let Some(BytecodeReductionAddressPhaseOutputClaims { chunks }) = bytecode_address_phase {
        for scalar in chunks.iter_mut() {
            f(scalar);
        }
    }
    if let Some(ProgramImageReductionAddressPhaseOutputClaims { program_image }) =
        program_image_address_phase
    {
        f(program_image);
    }
}

fn visit_reconstruction<F: Field>(
    claims: &mut ReconstructionOutputClaims<F>,
    f: &mut dyn FnMut(&mut F),
) {
    let ReconstructionOutputClaims {
        untrusted_advice,
        trusted_advice,
        bytecode,
        program_image,
    } = claims;
    if let Some(UntrustedAdviceReconstructionOutputClaims { bytes }) = untrusted_advice {
        f(bytes);
    }
    if let Some(TrustedAdviceReconstructionOutputClaims { bytes }) = trusted_advice {
        f(bytes);
    }
    if let Some(BytecodeChunkReconstructionOutputClaims {
        register_selectors,
        circuit_flags,
        instruction_flags,
        lookup_selectors,
        raf_flags,
        pc_bytes,
        imm_bytes,
    }) = bytecode
    {
        for lane in [
            register_selectors,
            circuit_flags,
            instruction_flags,
            lookup_selectors,
            raf_flags,
            pc_bytes,
            imm_bytes,
        ] {
            for scalar in lane.iter_mut() {
                f(scalar);
            }
        }
    }
    if let Some(ProgramImageReconstructionOutputClaims { bytes }) = program_image {
        f(bytes);
    }
}

fn clear_claim_scalar_count(case: &AkitaFixtureCase) -> usize {
    let mut proof = case.proof.clone();
    let mut count = 0usize;
    for_each_scalar_mut(clear_claims_mut(&mut proof), &mut |_| count += 1);
    count
}

/// Offset each clear-claim scalar in turn by one and assert the verifier
/// rejects the mutated proof.
fn sweep_clear_claim_offsets(case: &AkitaFixtureCase, scalar_count: usize) {
    for target in 0..scalar_count {
        let mut proof = case.proof.clone();
        let mut index = 0usize;
        for_each_scalar_mut(clear_claims_mut(&mut proof), &mut |scalar| {
            if index == target {
                *scalar += one();
            }
            index += 1;
        });
        assert_rejects(case.verify_proof(&proof));
    }
}

/// Every clear-claim scalar of every fixture case rejects a one-off offset.
#[test]
fn every_clear_claim_wire_rejects_offset() {
    let muldiv = akita_muldiv_case();
    let muldiv_scalars = clear_claim_scalar_count(muldiv);
    assert!(
        muldiv_scalars >= 150,
        "muldiv exposes only {muldiv_scalars} clear-claim scalars, expected at least 150"
    );
    sweep_clear_claim_offsets(muldiv, muldiv_scalars);

    let advice = akita_advice_case();
    sweep_clear_claim_offsets(advice, clear_claim_scalar_count(advice));

    let committed = akita_committed_muldiv_case();
    sweep_clear_claim_offsets(committed, clear_claim_scalar_count(committed));
}

/// Collect every perturbable serde leaf under `value`, keyed by a dot/index
/// path rooted at `prefix`. A non-empty all-number array (a limb/byte column)
/// is one leaf; an empty array has none.
fn leaf_paths(prefix: &str, value: &serde_json::Value, out: &mut Vec<String>) {
    match value {
        serde_json::Value::Object(map) => {
            for (key, child) in map {
                leaf_paths(&format!("{prefix}.{key}"), child, out);
            }
        }
        serde_json::Value::Array(items) => {
            if items.is_empty() {
                // No leaf to perturb.
            } else if items.iter().all(serde_json::Value::is_number) {
                out.push(prefix.to_string());
            } else {
                for (index, child) in items.iter().enumerate() {
                    leaf_paths(&format!("{prefix}[{index}]"), child, out);
                }
            }
        }
        serde_json::Value::Number(_) | serde_json::Value::String(_) => {
            out.push(prefix.to_string());
        }
        serde_json::Value::Bool(_) | serde_json::Value::Null => {}
    }
}

/// Perturb the leaf at `path` (as produced by [`leaf_paths`], whose first
/// segment names the root and is skipped).
fn perturb_leaf(value: &mut serde_json::Value, path: &str) {
    let mut cursor = value;
    for segment in path.split('.').skip(1) {
        let (key, indices) = match segment.find('[') {
            Some(split) => (&segment[..split], &segment[split..]),
            None => (segment, ""),
        };
        if !key.is_empty() {
            cursor = cursor.get_mut(key).expect("path segment must exist");
        }
        for index in indices.split(['[', ']']).filter(|s| !s.is_empty()) {
            let index: usize = index.parse().expect("index must parse");
            cursor = cursor.get_mut(index).expect("indexed segment must exist");
        }
    }
    match cursor {
        serde_json::Value::Number(number) => {
            let flipped = number.as_u64().map_or(1, |n| n ^ 1);
            *cursor = serde_json::Value::from(flipped);
        }
        serde_json::Value::String(text) => {
            text.push('0');
        }
        serde_json::Value::Array(items) => {
            let first = items.first_mut().expect("swept arrays are non-empty");
            let flipped = first.as_u64().map_or(1, |n| n ^ 1);
            *first = serde_json::Value::from(flipped);
        }
        other => panic!("unsupported leaf {other:?}"),
    }
}

/// Perturb every serde leaf of `commitment` one at a time; a mutation that no
/// longer deserializes is rejected at the boundary (the strongest rejection),
/// otherwise the rebuilt proof must fail to verify.
fn sweep_commitment(
    case: &AkitaFixtureCase,
    commitment: &AkitaCommitment,
    minimum_leaves: usize,
    rebuild: impl Fn(AkitaCommitment) -> AkitaJoltProof,
) {
    let value = serde_json::to_value(commitment).expect("commitment serializes");
    let mut paths = Vec::new();
    leaf_paths("commitment", &value, &mut paths);
    assert!(
        paths.len() >= minimum_leaves,
        "expected at least {minimum_leaves} commitment leaves, found {}",
        paths.len()
    );
    for path in paths {
        let mut mutated = value.clone();
        perturb_leaf(&mut mutated, &path);
        match serde_json::from_value::<AkitaCommitment>(mutated) {
            Err(_) => {}
            Ok(commitment) => assert_rejects(case.verify_proof(&rebuild(commitment))),
        }
    }
}

/// Every commitment-object wire — the `OneHotTrace` layout digest, declared
/// dimensions, backend flavor, and backend bytes, plus the untrusted-advice
/// object when present — rejects a leaf-level perturbation.
#[test]
fn every_commitment_wire_rejects_perturbation() {
    for case in [
        akita_muldiv_case(),
        akita_advice_case(),
        akita_committed_muldiv_case(),
    ] {
        sweep_commitment(case, &case.proof.commitments, 6, |commitment| {
            let mut proof = case.proof.clone();
            proof.commitments = commitment;
            proof
        });
    }

    let advice = akita_advice_case();
    if let Some(untrusted) = &advice.proof.untrusted_advice_commitment {
        sweep_commitment(advice, untrusted, 6, |commitment| {
            let mut proof = advice.proof.clone();
            proof.untrusted_advice_commitment = Some(commitment);
            proof
        });
    }
}

/// Proof-shape tampers: a swapped phase proof, dropped reconstruction /
/// auxiliary proofs, and an auxiliary evaluation offset — each fail-closed.
#[test]
fn akita_proof_shape_tampers_reject() {
    let muldiv = akita_muldiv_case();
    let mut proof = muldiv.proof.clone();
    proof.stages.stage6b_sumcheck_proof = proof.stages.stage3_sumcheck_proof.clone();
    assert_rejects(muldiv.verify_proof(&proof));

    let advice = akita_advice_case();
    let mut proof = advice.proof.clone();
    proof.stages.reconstruction_sumcheck_proof = None;
    assert_rejects(advice.verify_proof(&proof));

    let mut proof = advice.proof.clone();
    proof.joint_opening_proof.auxiliary = None;
    assert_rejects(advice.verify_proof(&proof));

    let committed = akita_committed_muldiv_case();
    let mut proof = committed.proof.clone();
    proof.joint_opening_proof.auxiliary = None;
    assert_rejects(committed.verify_proof(&proof));

    let mut proof = committed.proof.clone();
    if let Some(auxiliary) = proof.joint_opening_proof.auxiliary.as_mut() {
        auxiliary.evaluations[0] += one();
    }
    assert_rejects(committed.verify_proof(&proof));
}

/// The advice case fails closed when its trusted-advice commitment is absent:
/// the reconstruction outputs have nothing to bind against.
#[test]
fn akita_advice_commitment_presence_rejects() {
    let advice = akita_advice_case();
    let result = jolt_verifier::verify::<
        AkitaField,
        AkitaScheme,
        jolt_prover_legacy::zkvm::packed::AkitaVc,
        jolt_prover_legacy::zkvm::packed::AkitaTranscript,
    >(
        &advice.preprocessing,
        &advice.public_io,
        &advice.proof,
        None,
    );
    assert_rejects(result);
}
