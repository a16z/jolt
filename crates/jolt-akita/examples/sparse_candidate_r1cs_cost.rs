//! Candidate-only costs: allocated tape bits, exact sampling and dense outputs.
use jolt_akita::r1cs::D64CandidateProfile;
use jolt_r1cs::{bn254_bits::ByteVar, R1csBuilder};
use std::error::Error;

#[expect(
    clippy::print_stdout,
    reason = "intentional constraint-size diagnostic"
)]
fn main() -> Result<(), Box<dyn Error>> {
    for trials in [1, 2, 4] {
        let profile = D64CandidateProfile::selective_l2(trials)?;
        let mut builder = R1csBuilder::new();
        let tape: Vec<_> = (0..profile.tape_len())
            .map(|_| ByteVar::allocate(&mut builder, None))
            .collect();
        let _ = profile.sample(&mut builder, &tape)?;
        let matrices = builder.into_matrices();
        let nonzeros: usize = matrices
            .a
            .iter()
            .chain(&matrices.b)
            .chain(&matrices.c)
            .map(Vec::len)
            .sum();
        println!(
            "candidate trials={trials}, bytes={}: rows={}, variables={}, nonzeros={nonzeros}",
            profile.tape_len(),
            matrices.num_constraints,
            matrices.num_vars
        );
    }
    Ok(())
}
