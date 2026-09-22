//! Exact constraint sizes, including private input and digest binding.
use std::error::Error;

use jolt_r1cs::bn254_bits::ByteVar;
use jolt_r1cs::R1csBuilder;
use jolt_transcript::r1cs::{blake2b256, blake2b512};

#[expect(
    clippy::print_stdout,
    reason = "intentional constraint-size diagnostic"
)]
fn main() -> Result<(), Box<dyn Error>> {
    for length in [0, 128, 129] {
        for size in [32, 64] {
            let mut builder = R1csBuilder::new();
            let message: Vec<_> = (0..length)
                .map(|_| ByteVar::allocate(&mut builder, None))
                .collect();
            let digest = if size == 32 {
                blake2b256(&mut builder, &message)?.to_vec()
            } else {
                blake2b512(&mut builder, &message)?.to_vec()
            };
            for byte in digest {
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
                "blake2b{} bytes={length}: rows={}, variables={}, nonzeros={nonzeros}",
                size * 8,
                matrices.num_constraints,
                matrices.num_vars
            );
        }
    }
    Ok(())
}
