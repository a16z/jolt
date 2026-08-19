#![expect(clippy::print_stdout, reason = "bench harness")]

use std::time::Instant;

use jolt_crypto::{Bn254, Bn254G1, JoltGroup};
use jolt_dory::{DoryHint, DoryScheme};
use jolt_field::{Fr, FromPrimitiveInt};
use jolt_kernels::cuda::{shared_context, CudaDoryScheme};
use jolt_openings::AdditivelyHomomorphic;

const HINTS: usize = 42;
const ROWS: usize = 8192;

fn hints() -> Vec<DoryHint> {
    let step = Bn254::g1_generator();
    let mut walk = step;
    (0..HINTS)
        .map(|index| {
            let mut commitments = Vec::with_capacity(ROWS);
            for row in 0..ROWS {
                walk += step;
                commitments.push(if (index + row) % 37 == 0 {
                    Bn254G1::identity()
                } else {
                    walk
                });
            }
            DoryHint::new(commitments, Fr::from_u64(index as u64 + 1))
        })
        .collect()
}

fn main() {
    if shared_context().is_none() {
        println!("no CUDA device");
        return;
    }
    let source = hints();
    let scalars: Vec<Fr> = (0..HINTS)
        .map(|index| Fr::from_u64(index as u64 * 1_000_003 + 7))
        .collect();

    let now = Instant::now();
    let expected = DoryScheme::combine_hints(source.clone(), &scalars);
    let host = now.elapsed();

    let _ = CudaDoryScheme::combine_hints(source.clone(), &scalars);
    let now = Instant::now();
    let got = CudaDoryScheme::combine_hints(source.clone(), &scalars);
    let device = now.elapsed();

    assert_eq!(got, expected, "device combination diverged");
    println!("hints={HINTS} rows={ROWS}");
    println!("  host   {:?}", host);
    println!("  device {:?}", device);
    println!(
        "  speedup {:.2}x",
        host.as_secs_f64() / device.as_secs_f64()
    );
}
