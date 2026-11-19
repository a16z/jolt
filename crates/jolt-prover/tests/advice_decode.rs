//! Prover-chosen advice that the SDK cannot decode, traced. The guest sees
//! none of it before the decode, so the SDK must execute `spoil_proof`'s
//! unsatisfiable assertion: a guest panic alone is a provable outcome.

#![cfg(all(
    feature = "prover-fixtures",
    not(any(feature = "zk", feature = "akita", feature = "field-inline"))
))]
#![expect(clippy::expect_used, reason = "integration tests should fail loudly")]

use common::constants::RAM_START_ADDRESS;
use common::jolt_device::{JoltDevice, MemoryConfig};
use jolt_host::{default_target_dir, Program};
use jolt_program::image::decode_elf;
use tracer::instruction::Cycle;
use tracer::AdviceTape;

/// Honest VirtualAssertEQ rows panic the tracer on unequal operands, so an
/// unequal row is the SDK spoiling the proof.
fn spoiled(trace: &[Cycle]) -> bool {
    trace.iter().any(|cycle| {
        matches!(cycle, Cycle::VirtualAssertEQ(row)
            if row.register_state.rs1 != row.register_state.rs2)
    })
}

/// Traces `elf` over `advice_tape` and also returns the tape the run leaves,
/// which a compute_advice build has written.
fn trace_with_tape(
    guest: &Program,
    elf: &[u8],
    inputs: &[u8],
    advice_tape: Option<Vec<u8>>,
) -> (Vec<Cycle>, JoltDevice, Vec<u8>) {
    let image = decode_elf(elf, guest.instruction_profile()).expect("decode the guest ELF");
    let memory_config = MemoryConfig {
        program_size: Some(image.program_end - RAM_START_ADDRESS),
        ..MemoryConfig::default()
    };
    let (_, trace, _, device, advice_tape) = tracer::trace(
        elf,
        None,
        inputs,
        &[],
        &[],
        &memory_config,
        advice_tape.map(AdviceTape::from_bytes),
    );
    (trace, device, advice_tape.into_bytes())
}

/// A `Vec` length the guest heap cannot hold fails inside the tape decoder,
/// before the guest can check it.
#[test]
fn unreservable_vec_length_on_the_advice_tape_spoils_the_proof() {
    let mut guest = Program::new("advice-demo-guest");
    guest.set_std(true);
    guest.set_func("subset");
    let target_dir = default_target_dir();
    let target_dir = target_dir.to_string_lossy();
    guest.build_with_features(&target_dir, &["compute_advice"]);
    guest.build(&target_dir);
    let first_pass_elf = guest
        .get_elf_compute_advice_contents()
        .expect("the compute_advice build");
    let elf = guest.get_elf_contents().expect("the proving build");
    let inputs = postcard::to_stdvec(&(vec![2usize, 5, 3], vec![3usize, 2, 7, 5]))
        .expect("serialize the sets");

    let (_, _, honest_tape) = trace_with_tape(&guest, &first_pass_elf, &inputs, None);
    let (trace, device, _) = trace_with_tape(&guest, &elf, &inputs, Some(honest_tape.clone()));
    assert!(!device.panic);
    assert!(!spoiled(&trace));

    // Past `isize::MAX` bytes, and past the heap.
    for length in [u64::MAX, 1 << 40] {
        // Every word of the tape is the length, wherever the decoder reads it.
        let tape = length.to_le_bytes().repeat(honest_tape.len() / 8);
        let (trace, device, _) = trace_with_tape(&guest, &elf, &inputs, Some(tape));
        assert!(
            spoiled(&trace),
            "a Vec of {length} elements must spoil the proof, not only panic"
        );
        assert!(device.panic);
    }
}

#[test]
fn malformed_untrusted_advice_input_spoils_the_proof() {
    let mut guest = Program::new("advice-consumer-guest");
    let public_sum = postcard::to_stdvec(&12u64).expect("serialize the public input");
    let trusted = postcard::to_stdvec(&7u64).expect("serialize the trusted advice");

    let untrusted = postcard::to_stdvec(&5u64).expect("serialize the untrusted advice");
    let (_, trace, _, device) = guest.trace(&public_sum, &untrusted, &trusted);
    assert!(!device.panic);
    assert!(!spoiled(&trace));

    // No `u64` varint is longer than ten bytes.
    let (_, trace, _, device) = guest.trace(&public_sum, &[0xff; 11], &trusted);
    assert!(
        spoiled(&trace),
        "undecodable untrusted advice must spoil the proof, not only panic"
    );
    assert!(device.panic);
}

/// The decoder reserves the vector a length declares before it can reject
/// anything, and an allocation the heap refuses aborts the guest. Each vector
/// here fits the heap alone: the refused reservation is the untrusted one only
/// because the entry point decodes untrusted advice last.
#[test]
fn untrusted_advice_input_too_large_for_the_heap_spoils_the_proof() {
    let mut guest = Program::new("advice-demo-guest");
    guest.set_std(true);
    guest.set_func("block_total");
    guest.set_heap_size(1 << 20);
    // One byte each as advice, 264 bytes each in a decoded vector.
    let pinned = postcard::to_stdvec(&vec![None::<[u64; 32]>; 1000]).expect("serialize the blocks");

    let blocks = postcard::to_stdvec(&vec![Some([3u64; 32]), None]).expect("serialize the blocks");
    let (_, trace, _, device) = guest.trace(&[], &blocks, &pinned);
    assert!(!device.panic);
    assert!(!spoiled(&trace));

    let blocks = postcard::to_stdvec(&vec![None::<[u64; 32]>; 3000]).expect("serialize the blocks");
    let (_, trace, _, device) = guest.trace(&[], &blocks, &pinned);
    assert!(
        spoiled(&trace),
        "advice the heap cannot hold must spoil the proof, not only abort"
    );
    assert!(device.panic);
}
