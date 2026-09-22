//! Complete fixed-shape D64 candidate acceptance matrix census.
use std::error::Error;

use jolt_akita::r1cs::operator_norm::D64ShellVar;
use jolt_field::Fr;
use jolt_r1cs::R1csBuilder;

#[expect(
    clippy::print_stdout,
    reason = "intentional component matrix diagnostic"
)]
fn main() -> Result<(), Box<dyn Error>> {
    let mut builder = R1csBuilder::<Fr>::new();
    let coefficients = std::array::from_fn(|_| builder.alloc_witness(None));
    D64ShellVar::bind(&mut builder, coefficients, None)?.enforce_accepted(&mut builder)?;
    let matrices = builder.into_matrices();
    let nonzeros: usize = matrices
        .a
        .iter()
        .chain(&matrices.b)
        .chain(&matrices.c)
        .map(Vec::len)
        .sum();
    println!(
        "D64 shell + all32 exact upper bounds: rows={}, variables={}, nonzeros={nonzeros}",
        matrices.num_constraints, matrices.num_vars
    );
    Ok(())
}
