//! Trusted-local frozen-artifact diagnostic, not a verifier or proof generator.
use std::error::Error;
use std::fs::File;
use std::path::PathBuf;

use jolt_akita::r1cs::terminal::TerminalZ;
use jolt_field::Fr;
use jolt_r1cs::fp128_bn254::{Fp128Var, MODULUS};
use jolt_r1cs::integer_bn254::SignedVar;
use jolt_r1cs::R1csBuilder;
use serde::Deserialize;

#[derive(Deserialize)]
struct Terminal {
    z: Vec<i128>,
    t: Vec<[u8; 16]>,
    e: Vec<[u8; 16]>,
    a_prefix: Vec<[u8; 16]>,
    position_weights: Vec<[u8; 16]>,
    challenge_positions: Vec<Vec<usize>>,
    challenge_coefficients: Vec<Vec<i8>>,
}
#[derive(Deserialize)]
struct Quotients {
    a_row_quotients: Vec<String>,
    consistency: Consistency,
}
#[derive(Deserialize)]
struct Consistency {
    row_quotients: Vec<String>,
}

impl Terminal {
    fn centered(bytes: [u8; 16]) -> i128 {
        let value = u128::from_le_bytes(bytes);
        if value > (MODULUS - 1) / 2 {
            -((MODULUS - value) as i128)
        } else {
            value as i128
        }
    }
    fn allocate(
        builder: &mut R1csBuilder<Fr>,
        bytes: [u8; 16],
    ) -> Result<SignedVar, Box<dyn Error>> {
        let field = Fp128Var::allocate(builder, Some(u128::from_le_bytes(bytes)))?;
        Ok(SignedVar::centered(
            builder,
            &field,
            Some(Self::centered(bytes)),
        )?)
    }
}

#[expect(
    clippy::print_stdout,
    reason = "intentional constraint-size diagnostic"
)]
fn main() -> Result<(), Box<dyn Error>> {
    let path = PathBuf::from(
        std::env::args()
            .nth(1)
            .ok_or("pass a trusted frozen census directory")?,
    );
    let terminal: Terminal = serde_json::from_reader(File::open(path.join("terminal.json"))?)?;
    let quotients: Quotients =
        serde_json::from_reader(File::open(path.join("terminal-row-transport.json"))?)?;
    let mut builder = R1csBuilder::new();
    let z = TerminalZ::allocate(
        &mut builder,
        &terminal.z.iter().copied().map(Some).collect::<Vec<_>>(),
    )?;
    let mut coefficients = Vec::with_capacity(16384);
    for position in 0..256 {
        for j in 0..64 {
            let bytes = *terminal
                .a_prefix
                .get(position * 64 + (64 - j) % 64)
                .ok_or("short A")?;
            coefficients.push(if j == 0 {
                Terminal::centered(bytes)
            } else {
                -Terminal::centered(bytes)
            });
        }
    }
    let mut t = Vec::new();
    let mut e = Vec::new();
    let mut signs = Vec::new();
    for block in 0..7 {
        let positions = terminal
            .challenge_positions
            .get(block)
            .ok_or("missing positions")?;
        let values = terminal
            .challenge_coefficients
            .get(block)
            .ok_or("missing signs")?;
        if positions.len() != values.len() {
            return Err("sparse length mismatch".into());
        }
        for (&position, &sign) in positions.iter().zip(values) {
            if position >= 64 {
                return Err("sparse position out of range".into());
            }
            signs.push(if position == 0 { sign } else { -sign });
            let coordinate = (64 - position) % 64;
            t.push(Terminal::allocate(
                &mut builder,
                *terminal.t.get(block * 192 + coordinate).ok_or("short t")?,
            )?);
            e.push(Terminal::allocate(
                &mut builder,
                *terminal.e.get(block * 64 + coordinate).ok_or("short e")?,
            )?);
        }
    }
    z.enforce_a_row(
        &mut builder,
        &coefficients,
        &signs.iter().copied().zip(&t).collect::<Vec<_>>(),
        Some(
            quotients
                .a_row_quotients
                .first()
                .ok_or("missing A quotient")?
                .parse()?,
        ),
    )?;
    let weights = terminal
        .position_weights
        .iter()
        .map(|&x| Terminal::allocate(&mut builder, x))
        .collect::<Result<Vec<_>, _>>()?;
    z.enforce_consistency_row(
        &mut builder,
        0,
        &weights,
        &signs.iter().copied().zip(&e).collect::<Vec<_>>(),
        Some(
            quotients
                .consistency
                .row_quotients
                .first()
                .ok_or("missing consistency quotient")?
                .parse()?,
        ),
    )?;
    let witness = builder.witness()?;
    let matrices = builder.into_matrices();
    matrices
        .check_witness(&witness)
        .map_err(|row| format!("unsatisfied row {row}"))?;
    let nonzeros: usize = matrices
        .a
        .iter()
        .chain(&matrices.b)
        .chain(&matrices.c)
        .map(Vec::len)
        .sum();
    println!("complete z/ranges/norm + canonical coefficients + A row0 + consistency row0: rows={}, variables={}, nonzeros={nonzeros}; assignment satisfied", matrices.num_constraints, matrices.num_vars);
    Ok(())
}
