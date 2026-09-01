#![expect(clippy::print_stdout, clippy::expect_used, reason = "bench harness")]

use std::time::Instant;

use jolt_field::{Fr, FromPrimitiveInt};
use jolt_kernels::cuda::{shared_context, DeviceOneHotColumns, FoldTuning, LANES, SHARED_BUDGET};

const SAMPLES: usize = 3;

const LOG_CYCLES: usize = 21;

#[derive(Clone, Copy)]
enum Shape {
    Uniform,
    Loop { body: usize },
    Single,
}

impl Shape {
    fn label(self) -> String {
        match self {
            Self::Uniform => "uniform".to_string(),
            Self::Loop { body } => format!("loop({body})"),
            Self::Single => "single".to_string(),
        }
    }

    fn column(self, cycles: usize, addresses: usize) -> Vec<u32> {
        match self {
            Self::Uniform => (0..cycles)
                .map(|cycle| {
                    let mixed = (cycle as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                    ((mixed >> 33) % addresses as u64) as u32
                })
                .collect(),
            Self::Loop { body } => {
                let body = body.min(addresses).max(1);
                (0..cycles).map(|cycle| (cycle % body) as u32).collect()
            }
            Self::Single => vec![0u32; cycles],
        }
    }

    fn widest(self, cycles: usize, addresses: usize) -> usize {
        let column = self.column(cycles, addresses);
        let mut counts = vec![0usize; addresses];
        for address in column {
            if let Some(slot) = counts.get_mut(address as usize) {
                *slot += 1;
            }
        }
        counts.into_iter().max().unwrap_or(0)
    }
}

fn probe_to_host(context: &jolt_kernels::cuda::CudaKernelContext) {
    println!("DeviceFrVec::to_host cost by length (best of {SAMPLES}):");
    println!(
        "{:>10}  {:>9}  {:>9}  {:>10}  {:>9}  {:>9}  {:>10}",
        "elements", "KiB", "d2h ms", "raw ms", "raw MB/s", "h2d ms", "h2d ns/el",
    );
    for log_len in [10usize, 13, 16, 19, 22] {
        let len = 1usize << log_len;
        let device = context.alloc(len).expect("allocate a device vector");
        let _ = device.to_host().expect("warm the download");
        let mut best = f64::MAX;
        for _ in 0..SAMPLES {
            let started = Instant::now();
            let host = device.to_host().expect("timed download");
            best = best.min(started.elapsed().as_secs_f64() * 1e3);
            assert_eq!(host.len(), len, "the download lost elements");
        }
        let mut raw = f64::MAX;
        for _ in 0..SAMPLES {
            let started = Instant::now();
            let _ = device.to_limbs().expect("timed raw download");
            raw = raw.min(started.elapsed().as_secs_f64() * 1e3);
        }
        let host = device.to_host().expect("host copy for the upload probe");
        let mut up = f64::MAX;
        for _ in 0..SAMPLES {
            let started = Instant::now();
            let _ = context.upload(&host).expect("timed upload");
            up = up.min(started.elapsed().as_secs_f64() * 1e3);
        }
        let bytes = (len * 32) as f64;
        println!(
            "{len:>10}  {:>9.0}  {best:>9.3}  {raw:>10.3}  {:>9.1}  {up:>9.3}  {:>10.1}",
            bytes / 1024.0,
            bytes / (raw / 1e3) / 1e6,
            up * 1e6 / len as f64,
        );
    }
    println!();
}

fn main() {
    let Some(context) = shared_context() else {
        println!("no CUDA device; nothing to measure");
        return;
    };
    probe_to_host(context);
    let cycles = 1usize << LOG_CYCLES;
    let point: Vec<Fr> = (0..LOG_CYCLES)
        .map(|index| Fr::from_u64(index as u64 * 7 + 11))
        .collect();
    let tuning = FoldTuning::default();

    println!(
        "ohf_fold, {} cycles per call, LANES = {LANES}, SHARED_BUDGET = {} KiB",
        cycles,
        SHARED_BUDGET / 1024,
    );
    println!(
        "bucket_bytes = addresses * LANES * 8; the shared-memory privatization is taken only when \
         that fits the budget (addresses <= {})",
        SHARED_BUDGET / (LANES * size_of::<u64>()),
    );
    println!();
    println!(
        "{:>9}  {:>6}  {:>12}  {:>10}  {:>9}  {:>11}  {:>9}",
        "addresses", "shared", "shape", "widest", "ms", "Mcycles/s", "vs uniform",
    );

    for chunk_bits in [6usize, 9, 13] {
        let addresses = 1usize << chunk_bits;
        let shared = addresses * LANES * size_of::<u64>() <= SHARED_BUDGET;
        let mut uniform_ms = 0.0f64;
        for shape in [
            Shape::Uniform,
            Shape::Loop { body: 512 },
            Shape::Loop { body: 64 },
            Shape::Single,
        ] {
            let pc = shape.column(cycles, addresses);
            let columns =
                DeviceOneHotColumns::new(context, &[], &pc, &[], [0, 1, 0], chunk_bits, cycles)
                    .expect("build one-hot columns");

            let _ = columns
                .fold_cycles(context, &point, tuning)
                .expect("warm the fold");
            let mut best = f64::MAX;
            for _ in 0..SAMPLES {
                let started = Instant::now();
                let _ = columns
                    .fold_cycles(context, &point, tuning)
                    .expect("timed fold");
                best = best.min(started.elapsed().as_secs_f64() * 1e3);
            }
            if matches!(shape, Shape::Uniform) {
                uniform_ms = best;
            }
            println!(
                "{addresses:>9}  {:>6}  {:>12}  {:>10}  {best:>9.2}  {:>11.1}  {:>9}",
                if shared { "yes" } else { "no" },
                shape.label(),
                shape.widest(cycles, addresses),
                cycles as f64 / best / 1e3,
                format!("{:.2}x", best / uniform_ms),
            );
        }
        println!();
    }

    const POLYS: usize = 40;
    const CHUNK_BITS: usize = 4;
    let addresses = 1usize << CHUNK_BITS;
    let bucket_bytes = addresses * LANES * size_of::<u64>();
    println!(
        "instruction-family geometry: polys = {POLYS}, addresses = {addresses}, \
         bucket_bytes = {bucket_bytes} B, so shared memory fits {} polys per block",
        SHARED_BUDGET / bucket_bytes,
    );
    let lookup: Vec<u64> = (0..cycles * 2)
        .map(|index| (index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
        .collect();
    let columns = DeviceOneHotColumns::new(
        context,
        &lookup,
        &[],
        &[],
        [POLYS, 0, 0],
        CHUNK_BITS,
        cycles,
    )
    .expect("build the instruction-family columns");
    println!(
        "{:>16}  {:>7}  {:>9}  {:>11}  {:>9}",
        "polys_per_block", "groups", "ms", "GB/s cols", "vs 1",
    );
    let mut base_ms = 0.0f64;
    for polys_per_block in [1usize, 2, 4, 8, 16, 32] {
        let tuning = FoldTuning {
            polys_per_block,
            ..FoldTuning::default()
        };
        let _ = columns
            .fold_cycles(context, &point, tuning)
            .expect("warm the fold");
        let mut best = f64::MAX;
        for _ in 0..SAMPLES {
            let started = Instant::now();
            let _ = columns
                .fold_cycles(context, &point, tuning)
                .expect("timed fold");
            best = best.min(started.elapsed().as_secs_f64() * 1e3);
        }
        if polys_per_block == 1 {
            base_ms = best;
        }
        let groups = POLYS.div_ceil(polys_per_block);
        let column_bytes = (groups * cycles * (2 * size_of::<u64>() + 4 * size_of::<u64>())) as f64;
        println!(
            "{polys_per_block:>16}  {groups:>7}  {best:>9.2}  {:>11.1}  {:>9}",
            column_bytes / (best / 1e3) / 1e9,
            format!("{:.2}x", best / base_ms),
        );
    }
}
