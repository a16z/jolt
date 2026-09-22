//! Indexed-stream/retry relation diagnostic: [R K [coordinate | root_hex coordinate]].
//! An optional root checks native sampler parity, not root authentication.
use akita_challenges::{
    FoldChallengeDrawDomain, FoldDraw, OperatorNormRejection, D64_SELECTIVE_L2_CHALLENGE_CONFIG,
};
use jolt_akita::r1cs::D64RetryProfile;
use jolt_field::{CanonicalBytes, CanonicalEncoding, Fr, Ring};
use jolt_r1cs::{bn254_bits::ByteVar, LinearCombination, R1csBuilder};
use sha2::{Digest, Sha256};
use std::{error::Error, fmt::Write};

struct RootOracle([u8; 32]);
impl FoldDraw for RootOracle {
    fn absorb_and_squeeze(&mut self, _: &[u8], _: &[u8]) -> [u8; 32] {
        self.0
    }
}

#[expect(
    clippy::print_stdout,
    reason = "intentional wrapper constraint-size and native parity diagnostic"
)]
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let (rounds, trials, coordinate, input) = match args.as_slice() {
        [] => (2, 4, 0, None),
        [r, k] => (r.parse()?, k.parse()?, 0, None),
        [r, k, coordinate] => (r.parse()?, k.parse()?, coordinate.parse::<u64>()?, None),
        [r, k, root, coordinate] => {
            if root.len() != 64 || !root.is_ascii() {
                return Err("root must be 64 hex characters".into());
            }
            let bytes: Vec<_> = root
                .as_bytes()
                .chunks_exact(2)
                .map(|s| Ok(u8::from_str_radix(std::str::from_utf8(s)?, 16)?))
                .collect::<Result<_, Box<dyn Error>>>()?;
            (
                r.parse()?,
                k.parse()?,
                coordinate.parse::<u64>()?,
                Some(<[u8; 32]>::try_from(bytes).map_err(|_| "root length")?),
            )
        }
        _ => {
            return Err(
                "usage: sparse_retry_r1cs_cost [R K [coordinate | root_hex coordinate]]".into(),
            )
        }
    };
    let mut builder = R1csBuilder::new();
    let root = std::array::from_fn(|i| {
        ByteVar::allocate(&mut builder, input.and_then(|root| root.get(i).copied()))
    });
    let output = D64RetryProfile::new(rounds, trials)?.sample(&mut builder, &root, coordinate)?;
    let consumed = if let Some(root) = input {
        let count = usize::try_from(coordinate)?
            .checked_add(1)
            .ok_or("coordinate overflow")?;
        let native = RootOracle(root).draw_folding_challenges_with_rejection(
            FoldChallengeDrawDomain::EvaluationTrace,
            64,
            0,
            count,
            1,
            &D64_SELECTIVE_L2_CHALLENGE_CONFIG,
            0,
            Some(OperatorNormRejection::D64_SELECTIVE_L2),
        )?;
        let expected = native
            .as_slice()
            .get(usize::try_from(coordinate)?)
            .ok_or("native coordinate")?;
        let mut dense = [Fr::from_u64(0); 64];
        for (&position, &coefficient) in expected.positions.iter().zip(&expected.coeffs) {
            let magnitude = Fr::from_u64(coefficient.unsigned_abs().into());
            *dense
                .get_mut(usize::try_from(position)?)
                .ok_or("native position")? = if coefficient < 0 {
                -magnitude
            } else {
                magnitude
            };
        }
        for (&variable, expected) in output.coefficients().iter().zip(dense) {
            if builder.evaluate(&LinearCombination::variable(variable))? != expected {
                return Err("native output mismatch".into());
            }
        }
        Some(
            builder
                .evaluate(&output.consumed_bytes())?
                .to_u64_checked()
                .ok_or("consumed bytes")?,
        )
    } else {
        None
    };
    let selected: Option<Vec<_>> = input
        .map(|_| {
            output
                .selected()
                .iter()
                .map(|v| builder.evaluate(v).map(|v| v.to_u64_checked()))
                .collect::<Result<_, _>>()
        })
        .transpose()?;
    let witness = input.map(|_| builder.witness()).transpose()?;
    let matrices = builder.into_matrices();
    if let Some(witness) = witness {
        matrices
            .check_witness(&witness)
            .map_err(|row| format!("unsatisfied row {row}"))?;
    }
    let nonzeros: usize = matrices
        .a
        .iter()
        .chain(&matrices.b)
        .chain(&matrices.c)
        .map(Vec::len)
        .sum();
    let mut hash = Sha256::new();
    hash.update(b"akita-sparse-retry-matrices/v1");
    hash.update(u64::try_from(matrices.num_vars)?.to_le_bytes());
    hash.update(u64::try_from(matrices.num_constraints)?.to_le_bytes());
    for rows in [&matrices.a, &matrices.b, &matrices.c] {
        for row in rows {
            hash.update(u64::try_from(row.len())?.to_le_bytes());
            for (column, value) in row {
                hash.update(u64::try_from(*column)?.to_le_bytes());
                hash.update(value.to_bytes_le_vec());
            }
        }
    }
    let mut fingerprint = String::new();
    for byte in hash.finalize() {
        write!(&mut fingerprint, "{byte:02x}")?;
    }
    println!("D64 first acceptance R={rounds} K={trials}: rows={}, variables={}, nonzeros={nonzeros}, matrix_sha256={fingerprint}, consumed={consumed:?}, selected={selected:?}, native_and_r1cs_checked={}",matrices.num_constraints,matrices.num_vars,input.is_some());
    Ok(())
}
