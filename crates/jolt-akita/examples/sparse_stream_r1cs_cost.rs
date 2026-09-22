//! Exact sparse-stream relation sizes, including root and claimed-output bits.
use jolt_akita::r1cs::AkitaSparseStreamVar;
use jolt_r1cs::{bn254_bits::ByteVar, R1csBuilder};
use std::error::Error;

#[expect(
    clippy::print_stdout,
    reason = "intentional constraint-size diagnostic"
)]
fn main() -> Result<(), Box<dyn Error>> {
    for length in [64, 128] {
        let mut builder = R1csBuilder::new();
        let root = std::array::from_fn(|_| ByteVar::allocate(&mut builder, None));
        let mut stream = AkitaSparseStreamVar::new(&builder, &root, 0)?;
        for byte in stream.read(&mut builder, length)? {
            let claimed = ByteVar::allocate(&mut builder, None);
            builder.assert_equal(byte.expression(), claimed.expression());
        }
        let matrices = builder.into_matrices();
        let nonzeros: usize = matrices
            .a
            .iter()
            .chain(&matrices.b)
            .chain(&matrices.c)
            .map(Vec::len)
            .sum();
        println!(
            "sparse-stream bytes={length}: rows={}, variables={}, nonzeros={nonzeros}",
            matrices.num_constraints, matrices.num_vars
        );
    }
    Ok(())
}
