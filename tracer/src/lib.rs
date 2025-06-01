#![cfg_attr(not(feature = "std"), no_std)]
#![allow(dead_code)]
#![allow(clippy::legacy_numeric_constants)]

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, vec::Vec};

use common::{self, constants::RAM_START_ADDRESS, jolt_device::MemoryConfig};
use emulator::{
    cpu::{self, Xlen},
    default_terminal::DefaultTerminal,
    Emulator,
};

use instruction::{RV32IMCycle, RV32IMInstruction};
use object::{Object, ObjectSection, SectionKind};

mod emulator;
pub mod instruction;

pub use common::jolt_device::JoltDevice;

#[tracing::instrument(skip_all)]
pub fn trace_old(
    elf_contents: Vec<u8>,
    inputs: &[u8],
    memory_config: &MemoryConfig,
) -> (Vec<RV32IMCycle>, JoltDevice) {
    let term = DefaultTerminal::new();
    let mut emulator = Emulator::new(Box::new(term));
    emulator.update_xlen(get_xlen());

    let mut jolt_device = JoltDevice::new(memory_config);
    jolt_device.inputs = inputs.to_vec();
    emulator.get_mut_cpu().get_mut_mmu().jolt_device = jolt_device;

    emulator.setup_program(elf_contents);

    let mut prev_pc = 0;
    loop {
        if step_emulator(&mut emulator, &mut prev_pc, true) {
            break;
        }
    }

    let execution_trace = std::mem::take(&mut emulator.get_mut_cpu().trace);
    let device = std::mem::take(&mut emulator.get_mut_cpu().get_mut_mmu().jolt_device);

    (execution_trace, device)
}

#[tracing::instrument(skip_all)]
pub fn trace(
    elf_contents: Vec<u8>,
    inputs: &[u8],
    memory_config: &MemoryConfig,
    checkpoint_interval: Option<usize>,
) -> (Vec<RV32IMCycle>, JoltDevice, Option<Vec<Emulator>>) {
    println!("@@@@@@@@@@@@@@ Launching TEST MODE @@@@@@@@@@@@@@@@");
    trace_test(elf_contents.clone(), inputs, memory_config);
    unimplemented!("This is a test mode. The trace function is not implemented in test mode.");
}

#[tracing::instrument(skip_all)]
pub fn trace_new(
    elf_contents: Vec<u8>,
    inputs: &[u8],
    memory_config: &MemoryConfig,
    checkpoint_interval: Option<usize>,
) -> (Vec<RV32IMCycle>, JoltDevice, Option<Vec<Emulator>>) {
    let mut emulator = setup_emulator(elf_contents, inputs, memory_config);
    // checkpoints are emulator states
    let checkpoints = run_and_get_checkpoints(&mut emulator, checkpoint_interval);
    let execution_trace = std::mem::take(&mut emulator.get_mut_cpu().trace);
    let device = std::mem::take(&mut emulator.get_mut_cpu().get_mut_mmu().jolt_device);
    return (execution_trace, device, checkpoints);
}

#[tracing::instrument(skip_all)]
fn step_emulator(emulator: &mut Emulator, prev_pc: &mut u64, tracing: bool) -> bool {
    let pc = emulator.get_cpu().read_pc();
    // This is a trick to see if the program has terminated by throwing itself
    // into an infinite loop. It seems to be a good heuristic for now but we
    // should eventually migrate to an explicit shutdown signal.
    if *prev_pc == pc {
        return true;
    }
    emulator.tick(tracing);
    *prev_pc = pc;
    false
}

/*
Function to generate an incremental trace. It is an alternative to the trace function but instead of generating the
entire trace at once, it return a checkpoint of the trace at a given interval.
*/
#[tracing::instrument(skip_all)]
pub fn run_and_get_checkpoints(
    emulator: &mut Emulator,
    checkpoint_interval: Option<usize>,
) -> Option<Vec<Emulator>> {
    let mut prev_pc = 0;
    let mut checkpoints = Vec::new();

    match checkpoint_interval {
        Some(interval) => {
            let mut count = 0;
            loop {
                if count % interval == 0 {
                    checkpoints.push(emulator.clone());
                }
                count += 1;
                if step_emulator(emulator, &mut prev_pc, true) {
                    break;
                }
            }
        }
        None => loop {
            if step_emulator(emulator, &mut prev_pc, true) {
                break;
            }
        },
    }

    checkpoint_interval.map(|_| checkpoints)
}

#[tracing::instrument(skip_all)]
fn setup_emulator(elf_contents: Vec<u8>, inputs: &[u8], memory_config: &MemoryConfig) -> Emulator {
    let term = DefaultTerminal::new();
    let mut emulator = Emulator::new(Box::new(term));
    emulator.update_xlen(get_xlen());

    let mut jolt_device = JoltDevice::new(memory_config);
    jolt_device.inputs = inputs.to_vec();
    emulator.get_mut_cpu().get_mut_mmu().jolt_device = jolt_device;

    emulator.setup_program(elf_contents);
    return emulator;
}

pub struct LazyTraceIterator {
    emulator: Emulator,
    prev_pc: u64,
    current_traces: Vec<RV32IMCycle>,
}

impl Iterator for LazyTraceIterator {
    type Item = RV32IMCycle;

    fn next(&mut self) -> Option<Self::Item> {
        //Iterate over t returning in FIFO order before calling tick() again.
        if let Some(trace) = self.current_traces.pop() {
            return Some(trace);
        }

        if step_emulator(&mut self.emulator, &mut self.prev_pc, true) {
            return None;
        }

        self.current_traces = std::mem::take(&mut self.emulator.get_mut_cpu().trace);
        self.current_traces.reverse();
        self.current_traces.pop()
    }
}

#[tracing::instrument(skip_all)]
pub fn trace_from_checkpoint(checkpoint: &Emulator) -> LazyTraceIterator {
    LazyTraceIterator {
        emulator: checkpoint.clone(),
        prev_pc: 0,
        current_traces: Vec::new(),
    }
}

fn trace_test(elf_contents: Vec<u8>, inputs: &[u8], memory_config: &MemoryConfig) {
    println!("Running old trace function.");
    let (trace1, device1) = trace_old(elf_contents.clone(), inputs, memory_config);
    println!("Asserting that the trace is not empty.");
    assert!(trace1.len() > 0, "The trace should not be empty.");

    println!("Running new trace function with checkpoint_interval = None.");
    let (trace2, device2, checkpoints2) =
        trace_new(elf_contents.clone(), inputs, memory_config, None);
    println!("Asserting that checkpoints2.is_none.");
    assert!(
        checkpoints2.is_none(),
        "Checkpoints should not be collected when checkpoint_interval is None"
    );
    println!("Asserting that the traces are equal.");
    assert!(trace1 == trace2, "The traces should be equal.");

    println!("Running new trace function with checkpoint_interval = Some(20).");
    let (trace3, device3, checkpoints3) =
        trace_new(elf_contents.clone(), inputs, memory_config, Some(20));
    println!("Asserting that checkpoints3.is_some.");
    assert!(
        checkpoints3.is_some(),
        "Checkpoints should be collected when checkpoint_interval is Some"
    );
    println!("Getting trace_iter from the first checkpoint.");
    // generate the trace from the first checkpoint
    let trace_iter: LazyTraceIterator = trace_from_checkpoint(&checkpoints3.unwrap()[0]);
    println!("collecting the trace from trace_iter.");
    let trace4 = trace_iter.collect::<Vec<_>>();
    // for i in trace_iter {
    //     println!("{:?}", i);
    // }
    println!("Asserting that the traces are equal.");
    assert!(
        trace1 == trace4,
        "The traces should be equal.\n\
        ========== Trace4 ========\n\
        {:?}\n\
        ========== Trace1 ========\n\
        {:?}\n",
        trace4,
        trace1
    );

    print!("All tests passed successfully!\n");
}

#[tracing::instrument(skip_all)]
pub fn decode(elf: &[u8]) -> (Vec<RV32IMInstruction>, Vec<(u64, u8)>) {
    let obj = object::File::parse(elf).unwrap();

    let sections = obj
        .sections()
        .filter(|s| s.address() >= RAM_START_ADDRESS)
        .collect::<Vec<_>>();

    let mut instructions = Vec::new();
    let mut data = Vec::new();

    for section in sections {
        let raw_data = section.data().unwrap();

        if let SectionKind::Text = section.kind() {
            for (chunk, word) in raw_data.chunks(4).enumerate() {
                let word = u32::from_le_bytes(word.try_into().unwrap());
                let address = chunk as u64 * 4 + section.address();

                if let Ok(inst) = RV32IMInstruction::decode(word, address) {
                    instructions.push(inst);
                    continue;
                }
                // Unrecognized instruction, or from a ReadOnlyData section
                instructions.push(RV32IMInstruction::UNIMPL);
            }
        }
        let address = section.address();
        for (offset, byte) in raw_data.iter().enumerate() {
            data.push((address + offset as u64, *byte));
        }
    }

    (instructions, data)
}

fn get_xlen() -> Xlen {
    match common::constants::XLEN {
        32 => cpu::Xlen::Bit32,
        64 => cpu::Xlen::Bit64,
        _ => panic!("Emulator only supports 32 / 64 bit registers."),
    }
}
