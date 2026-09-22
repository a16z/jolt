//! Complete R2/K4 indexed-stream, candidate, norm and first-acceptance relation size.
use jolt_akita::r1cs::D64RetryProfile;
use jolt_r1cs::{bn254_bits::ByteVar, R1csBuilder};
use std::error::Error;

#[expect(
    clippy::print_stdout,
    reason = "intentional wrapper constraint-size diagnostic"
)]
fn main() -> Result<(), Box<dyn Error>> {
    let mut builder = R1csBuilder::new();
    let root = std::array::from_fn(|_| ByteVar::allocate(&mut builder, None));
    let _ = D64RetryProfile::new(2, 4)?.sample(&mut builder, &root, 0)?;
    let matrices = builder.into_matrices();
    let nonzeros: usize = matrices
        .a
        .iter()
        .chain(&matrices.b)
        .chain(&matrices.c)
        .map(Vec::len)
        .sum();
    println!(
        "D64 first acceptance R=2 K=4: rows={}, variables={}, nonzeros={nonzeros}",
        matrices.num_constraints, matrices.num_vars
    );
    Ok(())
}
