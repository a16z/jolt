// #![cfg_attr(feature = "guest", no_std)]
//
// Trust model:
// This guest program expects its input bytes to be a **guest-optimized encoding** produced by a
// trusted host pipeline (e.g. `jolt_sdk::decompress_transport_bytes_to_guest_bytes`) that has
// already decoded and validated the transport encoding in native execution.
//
// As a result, `GuestDeserialize` is intentionally **unchecked** for performance and this example
// uses `unwrap()` in a few hot deserialization paths. This is appropriate for the trusted
// aggregation setting; do not treat the guest encoding as a general wire format.

use jolt_sdk::{self as jolt};

extern crate alloc;

use jolt::GuestDeserialize;
use jolt::{
    BytecodePCMapper, BytecodePreprocessing, Instruction, JoltDevice, JoltSharedPreprocessing,
    JoltVerifierPreprocessing, RV64IMACProof, RV64IMACVerifier, F, PCS,
};

use jolt::{end_cycle_tracking, start_cycle_tracking};
use std::sync::Arc;

mod embedded_bytes {
    include!("./embedded_bytes.rs");
}

include!("./provable_macro.rs");

#[inline(always)]
fn deser_with_marker<T: GuestDeserialize, R: std::io::Read>(
    reader: &mut R,
    label: &'static str,
) -> T {
    start_cycle_tracking(label);
    let v = T::guest_deserialize(&mut *reader).unwrap();
    end_cycle_tracking(label);
    v
}

#[inline(always)]
fn deserialize_preprocessing_marked<R: std::io::Read>(
    reader: &mut R,
) -> JoltVerifierPreprocessing<F, PCS> {
    let generators: <PCS as jolt::CommitmentScheme>::VerifierSetup = deser_with_marker(
        reader,
        "deserialize preprocessing/generators (PCS::VerifierSetup)",
    );
    start_cycle_tracking("deserialize preprocessing/shared (JoltSharedPreprocessing)");
    let shared = {
        start_cycle_tracking("deserialize preprocessing/shared/bytecode_preprocessing");
        let bytecode_inner = deserialize_bytecode_preprocessing_marked(reader);
        end_cycle_tracking("deserialize preprocessing/shared/bytecode_preprocessing");
        let bytecode = Arc::new(bytecode_inner);
        let ram = deser_with_marker(reader, "deserialize preprocessing/shared/ram_preprocessing");
        let memory_layout =
            deser_with_marker(reader, "deserialize preprocessing/shared/memory_layout");
        let max_padded_trace_length = deser_with_marker(
            reader,
            "deserialize preprocessing/shared/max_padded_trace_length",
        );
        JoltSharedPreprocessing {
            bytecode,
            ram,
            memory_layout,
            max_padded_trace_length,
        }
    };
    end_cycle_tracking("deserialize preprocessing/shared (JoltSharedPreprocessing)");
    JoltVerifierPreprocessing { generators, shared }
}

#[inline(always)]
fn deserialize_bytecode_preprocessing_marked<R: std::io::Read>(
    reader: &mut R,
) -> BytecodePreprocessing {
    // Field order must match `BytecodePreprocessing`'s declaration order in
    // `jolt-core/src/zkvm/bytecode/mod.rs`.
    let code_size: usize = deser_with_marker(
        reader,
        "deserialize preprocessing/shared/bytecode/code_size (usize)",
    );

    // Deserialize `Vec<Instruction>` manually so we can chunk-profile the per-instruction cost.
    start_cycle_tracking("deserialize preprocessing/shared/bytecode/bytecode_vec");
    let bytecode_len: usize = deser_with_marker(
        reader,
        "deserialize preprocessing/shared/bytecode/bytecode_len (usize)",
    );

    start_cycle_tracking("deserialize preprocessing/shared/bytecode/bytecode_alloc");
    let mut bytecode: Vec<Instruction> = Vec::with_capacity(bytecode_len);
    end_cycle_tracking("deserialize preprocessing/shared/bytecode/bytecode_alloc");

    // Use repeated markers (same label) per chunk to avoid per-instruction log spam.
    const CHUNK: usize = 128;
    let mut remaining = bytecode_len;
    while remaining > 0 {
        let this_chunk = remaining.min(CHUNK);
        start_cycle_tracking("deserialize preprocessing/shared/bytecode/bytecode_chunk");
        for _ in 0..this_chunk {
            bytecode.push(Instruction::guest_deserialize(&mut *reader).unwrap());
        }
        end_cycle_tracking("deserialize preprocessing/shared/bytecode/bytecode_chunk");
        remaining -= this_chunk;
    }
    end_cycle_tracking("deserialize preprocessing/shared/bytecode/bytecode_vec");

    let pc_map: BytecodePCMapper =
        deser_with_marker(reader, "deserialize preprocessing/shared/bytecode/pc_map");

    BytecodePreprocessing {
        code_size,
        bytecode,
        pc_map,
    }
}

#[inline(always)]
fn deserialize_proof_marked<R: std::io::Read>(reader: &mut R) -> RV64IMACProof {
    // Field order must match `JoltProof`'s declaration order in
    // `jolt-core/src/zkvm/proof_serialization.rs`.
    let opening_claims = deser_with_marker(reader, "deserialize proof/opening_claims (Claims)");
    let commitments = deser_with_marker(reader, "deserialize proof/commitments (Vec<Commitment>)");

    let stage1_uni_skip_first_round_proof = deser_with_marker(
        reader,
        "deserialize proof/stage1_uni_skip_first_round_proof",
    );
    let stage1_sumcheck_proof =
        deser_with_marker(reader, "deserialize proof/stage1_sumcheck_proof");

    let stage2_uni_skip_first_round_proof = deser_with_marker(
        reader,
        "deserialize proof/stage2_uni_skip_first_round_proof",
    );
    let stage2_sumcheck_proof =
        deser_with_marker(reader, "deserialize proof/stage2_sumcheck_proof");

    let stage3_sumcheck_proof =
        deser_with_marker(reader, "deserialize proof/stage3_sumcheck_proof");
    let stage4_sumcheck_proof =
        deser_with_marker(reader, "deserialize proof/stage4_sumcheck_proof");
    let stage5_sumcheck_proof =
        deser_with_marker(reader, "deserialize proof/stage5_sumcheck_proof");
    let stage6_sumcheck_proof =
        deser_with_marker(reader, "deserialize proof/stage6_sumcheck_proof");
    let stage7_sumcheck_proof =
        deser_with_marker(reader, "deserialize proof/stage7_sumcheck_proof");

    let joint_opening_proof: <PCS as jolt::CommitmentScheme>::Proof =
        deser_with_marker(reader, "deserialize proof/joint_opening_proof (PCS::Proof)");
    let untrusted_advice_commitment = deser_with_marker(
        reader,
        "deserialize proof/untrusted_advice_commitment (Option<Commitment>)",
    );

    let trace_length = deser_with_marker(reader, "deserialize proof/trace_length (usize)");
    let ram_k = deser_with_marker(reader, "deserialize proof/ram_K (usize)");
    let bytecode_k = deser_with_marker(reader, "deserialize proof/bytecode_K (usize)");
    let rw_config = deser_with_marker(reader, "deserialize proof/rw_config (ReadWriteConfig)");
    let one_hot_config =
        deser_with_marker(reader, "deserialize proof/one_hot_config (OneHotConfig)");
    let dory_layout = deser_with_marker(reader, "deserialize proof/dory_layout (DoryLayout)");

    RV64IMACProof {
        opening_claims,
        commitments,
        stage1_uni_skip_first_round_proof,
        stage1_sumcheck_proof,
        stage2_uni_skip_first_round_proof,
        stage2_sumcheck_proof,
        stage3_sumcheck_proof,
        stage4_sumcheck_proof,
        stage5_sumcheck_proof,
        stage6_sumcheck_proof,
        stage7_sumcheck_proof,
        joint_opening_proof,
        untrusted_advice_commitment,
        trace_length,
        ram_K: ram_k,
        bytecode_K: bytecode_k,
        rw_config,
        one_hot_config,
        dory_layout,
    }
}

#[inline(always)]
fn deserialize_device_marked<R: std::io::Read>(reader: &mut R) -> JoltDevice {
    // Field order must match `JoltDevice`'s declaration order in `common/src/jolt_device.rs`.
    let inputs = deser_with_marker(reader, "deserialize device/inputs (Vec<u8>)");
    let trusted_advice = deser_with_marker(reader, "deserialize device/trusted_advice (Vec<u8>)");
    let untrusted_advice =
        deser_with_marker(reader, "deserialize device/untrusted_advice (Vec<u8>)");
    let outputs = deser_with_marker(reader, "deserialize device/outputs (Vec<u8>)");
    let panic = deser_with_marker(reader, "deserialize device/panic (bool)");
    let memory_layout =
        deser_with_marker(reader, "deserialize device/memory_layout (MemoryLayout)");
    JoltDevice {
        inputs,
        trusted_advice,
        untrusted_advice,
        outputs,
        panic,
        memory_layout,
    }
}

provable_with_config! {
fn verify(bytes: &[u8]) -> u32 {
    let use_embedded = !embedded_bytes::EMBEDDED_BYTES.is_empty();
    let data_bytes = if use_embedded {
        embedded_bytes::EMBEDDED_BYTES
    } else {
        bytes
    };

    let mut cursor = std::io::Cursor::new(data_bytes);

    start_cycle_tracking("deserialize preprocessing");
    let verifier_preprocessing: JoltVerifierPreprocessing<F, PCS> =
        deserialize_preprocessing_marked(&mut cursor);
    end_cycle_tracking("deserialize preprocessing");

    start_cycle_tracking("deserialize count of proofs");
    // Deserialize number of proofs to verify
    let n: u32 = deser_with_marker(&mut cursor, "deserialize count of proofs/u32");
    end_cycle_tracking("deserialize count of proofs");

    let mut all_valid = true;
    for _ in 0..n {
        start_cycle_tracking("deserialize proof");
        let proof = deserialize_proof_marked(&mut cursor);
        end_cycle_tracking("deserialize proof");

        start_cycle_tracking("deserialize device");
        let device = deserialize_device_marked(&mut cursor);
        end_cycle_tracking("deserialize device");

        start_cycle_tracking("verification");
        let verifier = RV64IMACVerifier::new(&verifier_preprocessing, proof, device, None, None);
        let is_valid = verifier.is_ok_and(|verifier| verifier.verify().is_ok());
        end_cycle_tracking("verification");
        all_valid = all_valid && is_valid;
    }

    all_valid as u32
}
}
