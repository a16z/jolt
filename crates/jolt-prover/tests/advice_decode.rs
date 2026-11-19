//! Prover-chosen advice that the SDK cannot decode, traced. The guest sees
//! none of it before the decode, so the SDK must reach `spoil_proof`'s
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

struct Run {
    device: JoltDevice,
    /// The trace holds spoil_proof's unsatisfiable assertion.
    spoiled: bool,
    advice_tape: Vec<u8>,
}

fn trace(
    guest: &Program,
    elf: &[u8],
    inputs: &[u8],
    untrusted_advice: &[u8],
    trusted_advice: &[u8],
    advice_tape: Option<Vec<u8>>,
) -> Run {
    let image = decode_elf(elf, guest.instruction_profile()).expect("decode the guest ELF");
    let memory_config = MemoryConfig {
        program_size: Some(image.program_end - RAM_START_ADDRESS),
        ..MemoryConfig::default()
    };
    let (_, trace, _, device, advice_tape) = tracer::trace(
        elf,
        None,
        inputs,
        untrusted_advice,
        trusted_advice,
        &memory_config,
        advice_tape.map(AdviceTape::from_bytes),
    );
    // Honest VirtualAssertEQ rows panic the tracer on unequal operands, so an
    // unequal row is spoil_proof's.
    let spoiled = trace.iter().any(|cycle| {
        matches!(cycle, Cycle::VirtualAssertEQ(row)
            if row.register_state.rs1 != row.register_state.rs2)
    });
    Run {
        device,
        spoiled,
        advice_tape: advice_tape.into_bytes(),
    }
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

    let first_pass = trace(&guest, &first_pass_elf, &inputs, &[], &[], None);
    let honest = trace(
        &guest,
        &elf,
        &inputs,
        &[],
        &[],
        Some(first_pass.advice_tape.clone()),
    );
    assert!(!honest.device.panic);
    assert!(!honest.spoiled);

    // Past `isize::MAX` bytes, and past the heap.
    for length in [u64::MAX, 1 << 40] {
        // Every word of the tape is the length, wherever the decoder reads it.
        let tape = length
            .to_le_bytes()
            .repeat(first_pass.advice_tape.len() / 8);
        let run = trace(&guest, &elf, &inputs, &[], &[], Some(tape));
        assert!(
            run.spoiled,
            "a Vec of {length} elements must spoil the proof, not only panic"
        );
        assert!(run.device.panic);
    }
}

#[test]
fn malformed_untrusted_advice_input_spoils_the_proof() {
    let mut guest = Program::new("advice-consumer-guest");
    guest.build(&default_target_dir().to_string_lossy());
    let elf = guest.get_elf_contents().expect("the proving build");
    let public_sum = postcard::to_stdvec(&12u64).expect("serialize the public input");
    let trusted = postcard::to_stdvec(&7u64).expect("serialize the trusted advice");

    let untrusted = postcard::to_stdvec(&5u64).expect("serialize the untrusted advice");
    let honest = trace(&guest, &elf, &public_sum, &untrusted, &trusted, None);
    assert!(!honest.device.panic);
    assert!(!honest.spoiled);

    // No `u64` varint is longer than ten bytes.
    let run = trace(&guest, &elf, &public_sum, &[0xff; 11], &trusted, None);
    assert!(
        run.spoiled,
        "undecodable untrusted advice must spoil the proof, not only panic"
    );
    assert!(run.device.panic);
}
