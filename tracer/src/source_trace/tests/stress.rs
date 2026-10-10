use super::*;
use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
};

#[test]
fn seeded_state_equivalence() {
    use rand::{rngs::StdRng, Rng, SeedableRng};

    for seed in 0..64 {
        let mut f = Fixture::new();
        let base = f.layout().heap_end - 64;
        f.address(31, base);
        let mut rng = StdRng::seed_from_u64(seed);
        for index in 0..64 {
            let rd = rng.gen_range(0..16);
            let rs1 = rng.gen_range(0..16);
            let rs2 = rng.gen_range(0..16);
            let word = match rng.gen_range(0..12) {
                0 => r(
                    0x33,
                    [0, 1, 2, 3, 4, 5, 6, 7][rng.gen_range(0..8)],
                    0,
                    rd,
                    rs1,
                    rs2,
                ),
                1 => r(0x33, [0, 5][rng.gen_range(0..2)], 0x20, rd, rs1, rs2),
                2 => i(
                    0x13,
                    [0, 2, 3, 4, 6, 7][rng.gen_range(0..6)],
                    rd,
                    rs1,
                    rng.gen_range(-2048..2048),
                ),
                3 => i(0x13, 1, rd, rs1, rng.gen_range(0..64)),
                4 => i(
                    0x13,
                    5,
                    rd,
                    rs1,
                    rng.gen_range(0..64) | if rng.gen_bool(0.5) { 0x400 } else { 0 },
                ),
                5 => (rng.gen::<u32>() & 0xffff_f000) | (u32::from(rd) << 7) | 0x37,
                6 => (rng.gen::<u32>() & 0xffff_f000) | (u32::from(rd) << 7) | 0x17,
                7 => b(
                    [0, 1, 4, 5, 6, 7][rng.gen_range(0..6)],
                    rs1,
                    rs2,
                    if index == 63 { 4 } else { 8 },
                ),
                8 | 9 => {
                    let funct3 = [0, 1, 2, 3, 4, 5, 6][rng.gen_range(0..7)];
                    let width = [1, 2, 4, 8, 1, 2, 4][funct3 as usize];
                    i(0x03, funct3, rd, 31, rng.gen_range(0..(64 / width)) * width)
                }
                10 => {
                    let funct3 = rng.gen_range(0..4);
                    let width = 1 << funct3;
                    s(funct3, 31, rs2, rng.gen_range(0..(64 / width)) * width)
                }
                _ => r(0x3b, [0, 1, 5][rng.gen_range(0..3)], 0, rd, rs1, rs2),
            };
            f.words.push(word);
        }
        f.words.push(HALT);
        let program = f.program();
        let output = SourceTracerBackend::with_row_capacity(70)
            .trace(&program, f.inputs.clone())
            .unwrap();
        check_replay(&program, &f.inputs, &output);
        check_lockstep(&program, &f.inputs, output.trace.rows());
    }
}

thread_local! {
    static ALLOCATIONS: Cell<Option<usize>> = const { Cell::new(None) };
}

struct CountingAllocator;

impl CountingAllocator {
    fn count_allocation(&self) {
        let _ = ALLOCATIONS.try_with(|counter| {
            if let Some(count) = counter.get() {
                counter.set(Some(count + 1));
            }
        });
    }
}

// SAFETY: every allocation operation forwards the same pointer/layout to
// System, preserving its GlobalAlloc contract; the counter owns no allocation.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        self.count_allocation();
        // SAFETY: the caller supplied the GlobalAlloc layout contract.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        self.count_allocation();
        // SAFETY: the caller supplied the GlobalAlloc layout contract.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: this allocator forwards every allocation to System.
        unsafe { System.dealloc(pointer, layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        self.count_allocation();
        // SAFETY: this allocator forwards every allocation to System.
        unsafe { System.realloc(pointer, layout, size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

#[test]
fn allocation_free_step_loop() {
    assert!(
        !std::env::var("JOLT_BACKTRACE").is_ok_and(|value| value.eq_ignore_ascii_case("full")),
        "JOLT_BACKTRACE=full allocates a register snapshot per call and is excluded from the allocation contract",
    );
    let mut f = Fixture::new();
    let layout = f.layout();
    f.address(5, layout.heap_end - 8);
    f.address(6, layout.output_start);
    f.words.extend([
        0x0000_13b7, // lui x7, 1: 4096 iterations
        i(0x13, 0, 8, 0, 0),
        i(0x13, 0, 8, 8, 1),
        j(1, 32),
        s(3, 5, 8, 0),
        i(0x03, 3, 9, 5, 0),
        s(0, 6, 9, 0),
        i(0x13, 0, 6, 6, 1),
        i(0x13, 0, 7, 7, -1),
        b(1, 7, 0, -28),
        HALT,
        i(0x13, 4, 10, 8, -1),
        i(0x67, 0, 0, 1, 0),
    ]);
    let program = f.program();
    let mut execution = SourceExecution::new(&program, f.inputs).unwrap();
    let mut rows = Vec::with_capacity(4096 * 10 + 11);
    ALLOCATIONS.with(|counter| counter.set(Some(0)));
    let result = loop {
        match execution.step(&mut rows) {
            Ok(true) => {}
            Ok(false) => break Ok(()),
            Err(error) => break Err(error),
        }
    };
    let allocations = ALLOCATIONS.with(|counter| counter.replace(None).unwrap());
    result.unwrap();
    assert_eq!(allocations, 0);
    assert_eq!(rows.len(), 4096 * 10 + 11);
    assert_eq!(execution.emulator.get_cpu().x[8], 4096);
    let outputs = &execution
        .emulator
        .get_cpu()
        .mmu
        .jolt_device
        .as_ref()
        .unwrap()
        .outputs;
    assert_eq!(outputs.len(), 4096);
    for (offset, &byte) in outputs.iter().enumerate() {
        assert_eq!(byte, (offset + 1) as u8);
    }
}
