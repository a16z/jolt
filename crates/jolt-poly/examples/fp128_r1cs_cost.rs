//! Exact constraint sizes; this is not a proving-time benchmark.
use std::error::Error;

use jolt_field::Fr;
use jolt_poly::r1cs::evaluate_fp128_bn254;
use jolt_r1cs::fp128_bn254::Fp128Var;
use jolt_r1cs::{ConstraintMatrices, R1csBuilder};

#[expect(
    clippy::print_stdout,
    reason = "this diagnostic reports constraint sizes to stdout"
)]
fn report(label: &str, matrices: ConstraintMatrices<Fr>) {
    let nonzeros: usize = matrices
        .a
        .iter()
        .chain(&matrices.b)
        .chain(&matrices.c)
        .map(Vec::len)
        .sum();
    println!(
        "{label}: rows={}, variables={}, nonzeros={nonzeros}",
        matrices.num_constraints, matrices.num_vars
    );
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut builder = R1csBuilder::new();
    let a = Fp128Var::allocate(&mut builder, None)?;
    report("one canonical input", builder.clone().into_matrices());
    let b = Fp128Var::allocate(&mut builder, None)?;
    report("two canonical inputs", builder.clone().into_matrices());
    let _ = a.multiply(&mut builder, &b)?;
    report("product including inputs", builder.into_matrices());

    let mut builder = R1csBuilder::new();
    let point = Fp128Var::allocate(&mut builder, None)?;
    let coefficients: Vec<_> = (0..4)
        .map(|_| Fp128Var::allocate(&mut builder, None))
        .collect::<Result<_, _>>()?;
    let _ = evaluate_fp128_bn254(&mut builder, &coefficients, &point)?;
    report(
        "degree-three Horner including inputs",
        builder.into_matrices(),
    );
    Ok(())
}
