//! Historical algebra-only census diagnostic; not a Blake-only proof fixture.
use std::error::Error;
use std::path::PathBuf;

use jolt_akita::r1cs::scalar::TerminalPointVar;
use jolt_akita::r1cs::terminal::TerminalZ;
use jolt_field::Fr;
use jolt_r1cs::fp128_bn254::{Fp128Var, MODULUS};
use jolt_r1cs::integer_bn254::SignedVar;
use jolt_r1cs::R1csBuilder;
use serde::Deserialize;

#[derive(Deserialize)]
struct Terminal {
    z: Vec<i128>,
    e: Vec<[u8; 16]>,
    protocol_point: Vec<[u8; 16]>,
    opening_value: [u8; 16],
    position_weights: Vec<[u8; 16]>,
    challenge_positions: Vec<Vec<usize>>,
    challenge_coefficients: Vec<Vec<i8>>,
}
#[derive(Deserialize)]
struct Quotients {
    consistency: Consistency,
}
#[derive(Deserialize)]
struct Consistency {
    row_quotients: Vec<String>,
}

impl Terminal {
    fn centered(bytes: [u8; 16]) -> Result<i128, Box<dyn Error>> {
        let x = u128::from_le_bytes(bytes);
        if x >= MODULUS {
            return Err("noncanonical historical field value".into());
        }
        Ok(if x > (MODULUS - 1) / 2 {
            -((MODULUS - x) as i128)
        } else {
            x as i128
        })
    }

    #[expect(
        clippy::print_stdout,
        reason = "intentional complete component matrix diagnostic"
    )]
    fn report(label: &str, builder: &R1csBuilder<Fr>) -> Result<(), Box<dyn Error>> {
        let witness = builder.witness()?;
        let matrices = builder.clone().into_matrices();
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
        println!(
            "{label}: rows={}, variables={}, nonzeros={nonzeros}; historical assignment satisfied",
            matrices.num_constraints, matrices.num_vars
        );
        Ok(())
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let path = PathBuf::from(
        std::env::args()
            .nth(1)
            .ok_or("pass a trusted historical scalar census directory")?,
    );
    let terminal: Terminal =
        serde_json::from_reader(std::fs::File::open(path.join("terminal.json"))?)?;
    let quotients: Quotients = serde_json::from_reader(std::fs::File::open(
        path.join("terminal-row-transport.json"),
    )?)?;
    let mut builder = R1csBuilder::new();
    let point = terminal
        .protocol_point
        .iter()
        .map(|&b| Fp128Var::allocate(&mut builder, Some(u128::from_le_bytes(b))))
        .collect::<Result<Vec<_>, _>>()?;
    let e = terminal
        .e
        .iter()
        .map(|&b| Fp128Var::allocate(&mut builder, Some(u128::from_le_bytes(b))))
        .collect::<Result<Vec<_>, _>>()?;
    let claim = Fp128Var::allocate(
        &mut builder,
        Some(u128::from_le_bytes(terminal.opening_value)),
    )?;
    let prepared = TerminalPointVar::prepare(&mut builder, &point)?;
    prepared.enforce_scalar_opening(&mut builder, &e, &claim)?;
    Terminal::report(
        "point preparation + scalar opening + canonical inputs",
        &builder,
    )?;
    let z = TerminalZ::allocate(
        &mut builder,
        &terminal.z.iter().copied().map(Some).collect::<Vec<_>>(),
    )?;
    let mut rhs_values = Vec::new();
    let mut signs = Vec::new();
    for block in 0..7 {
        let positions = terminal
            .challenge_positions
            .get(block)
            .ok_or("missing positions")?;
        let coefficients = terminal
            .challenge_coefficients
            .get(block)
            .ok_or("missing coefficients")?;
        if positions.len() != coefficients.len() {
            return Err("sparse lengths disagree".into());
        }
        for (&position, &coefficient) in positions.iter().zip(coefficients) {
            if position >= 64 {
                return Err("sparse position out of range".into());
            }
            let index = block * 64 + (64 - position) % 64;
            let value = e.get(index).ok_or("missing e")?;
            let assignment =
                Terminal::centered(*terminal.e.get(index).ok_or("missing e encoding")?)?;
            rhs_values.push(SignedVar::centered(&mut builder, value, Some(assignment))?);
            signs.push(if position == 0 {
                coefficient
            } else {
                -coefficient
            });
        }
    }
    let assignments = terminal
        .position_weights
        .iter()
        .map(|&b| Ok(Some(Terminal::centered(b)?)))
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
    prepared.enforce_consistency_row(
        &mut builder,
        &z,
        0,
        &signs.into_iter().zip(&rhs_values).collect::<Vec<_>>(),
        &assignments,
        Some(
            quotients
                .consistency
                .row_quotients
                .first()
                .ok_or("missing quotient")?
                .parse()?,
        ),
    )?;
    Terminal::report(
        "plus complete z norm and point-bound consistency row0",
        &builder,
    )
}
