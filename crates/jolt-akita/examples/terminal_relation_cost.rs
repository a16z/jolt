//! Trusted-local full terminal relation diagnostic, not proof acceptance.
use std::{error::Error, fmt::Write, fs::File, path::PathBuf};

use akita_types::TerminalFoldParams;
use jolt_akita::r1cs::terminal::TerminalZ;
use jolt_akita::r1cs::terminal_relation::{
    TerminalChallengesVar, TerminalRelationInputs, TerminalRelationProfile, TerminalRelationWitness,
};
use jolt_field::{CanonicalBytes, Fr, Ring};
use jolt_r1cs::bn254_bits::ByteVar;
use jolt_r1cs::fp128_bn254::{Fp128Var, MODULUS};
use jolt_r1cs::{ConstraintMatrices, R1csBuilder};
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

#[derive(Deserialize)]
struct Terminal {
    z: Vec<i128>,
    e: Vec<[u8; 16]>,
    t: Vec<[u8; 16]>,
    e_canonical_bytes: Vec<u8>,
    t_canonical_bytes: Vec<u8>,
    a_prefix: Vec<[u8; 16]>,
    protocol_point: Vec<[u8; 16]>,
    opening_value: [u8; 16],
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
    fn mutate(&mut self, mode: &str) -> Result<(), Box<dyn Error>> {
        match mode {
            "positive" | "unknown" => {}
            "coherent-e" | "coherent-t" => {
                let (fields, bytes) = if mode == "coherent-e" {
                    (&mut self.e, &mut self.e_canonical_bytes)
                } else {
                    (&mut self.t, &mut self.t_canonical_bytes)
                };
                let field = fields.first_mut().ok_or("empty segment")?;
                *field = ((u128::from_le_bytes(*field) + 1) % MODULUS).to_le_bytes();
                bytes
                    .get_mut(..16)
                    .ok_or("short segment")?
                    .copy_from_slice(field);
            }
            "claim" => {
                self.opening_value =
                    ((u128::from_le_bytes(self.opening_value) + 1) % MODULUS).to_le_bytes();
            }
            _ => {
                return Err(
                    "mode must be positive, unknown, coherent-e, coherent-t, or claim".into(),
                )
            }
        }
        Ok(())
    }

    fn segment(
        builder: &mut R1csBuilder<Fr>,
        fields: &[[u8; 16]],
        bytes: &[u8],
        known: bool,
    ) -> Result<Vec<Fp128Var>, Box<dyn Error>> {
        if fields.iter().flatten().copied().collect::<Vec<_>>() != bytes {
            return Err("canonical segment mismatch".into());
        }
        bytes
            .chunks_exact(16)
            .map(|chunk| {
                let variables: [ByteVar; 16] = chunk
                    .iter()
                    .map(|&value| ByteVar::allocate(builder, known.then_some(value)))
                    .collect::<Vec<_>>()
                    .try_into()
                    .map_err(|_| "field width")?;
                Ok(Fp128Var::from_le_bytes(builder, &variables)?)
            })
            .collect()
    }

    fn assemble(
        &self,
        params: &TerminalFoldParams,
        quotients: &Quotients,
        known: bool,
    ) -> Result<R1csBuilder<Fr>, Box<dyn Error>> {
        let a: Vec<_> = self
            .a_prefix
            .iter()
            .copied()
            .map(u128::from_le_bytes)
            .collect();
        let profile = TerminalRelationProfile::new(params, &a)?;
        let mut builder = R1csBuilder::new();
        let z = TerminalZ::allocate(
            &mut builder,
            &self
                .z
                .iter()
                .map(|&value| known.then_some(value))
                .collect::<Vec<_>>(),
        )?;
        let e = Self::segment(&mut builder, &self.e, &self.e_canonical_bytes, known)?;
        let t = Self::segment(&mut builder, &self.t, &self.t_canonical_bytes, known)?;
        let point = self
            .protocol_point
            .iter()
            .map(|&bytes| {
                Fp128Var::allocate(&mut builder, known.then_some(u128::from_le_bytes(bytes)))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let claim = Fp128Var::allocate(
            &mut builder,
            known.then_some(u128::from_le_bytes(self.opening_value)),
        )?;
        if self.challenge_positions.len() != 7 || self.challenge_coefficients.len() != 7 {
            return Err("challenge shape".into());
        }
        let mut dense = Vec::new();
        for (positions, coefficients) in self
            .challenge_positions
            .iter()
            .zip(&self.challenge_coefficients)
        {
            if positions.len() != coefficients.len() {
                return Err("sparse shape".into());
            }
            let mut values = [0i8; 64];
            for (&position, &coefficient) in positions.iter().zip(coefficients) {
                let value = values.get_mut(position).ok_or("position outside D64")?;
                if *value != 0 {
                    return Err("duplicate sparse position".into());
                }
                *value = coefficient;
            }
            dense.push(values.map(|value| {
                builder.alloc_witness(known.then(|| {
                    let magnitude = Fr::from_u64(u64::from(value.unsigned_abs()));
                    if value < 0 {
                        -magnitude
                    } else {
                        magnitude
                    }
                }))
            }));
        }
        let challenges = TerminalChallengesVar::bind(&mut builder, &dense)?;
        let a_quotients = quotients
            .a_row_quotients
            .iter()
            .map(|x| x.parse::<i128>().map(|x| known.then_some(x)))
            .collect::<Result<Vec<_>, _>>()?;
        let consistency_quotients = quotients
            .consistency
            .row_quotients
            .iter()
            .map(|x| x.parse::<i128>().map(|x| known.then_some(x)))
            .collect::<Result<Vec<_>, _>>()?;
        profile.enforce(
            &mut builder,
            TerminalRelationInputs {
                z: &z,
                e: &e,
                t: &t,
                point: &point,
                claim: &claim,
                challenges: &challenges,
            },
            TerminalRelationWitness {
                a_quotients: &a_quotients,
                consistency_quotients: &consistency_quotients,
            },
        )?;
        Ok(builder)
    }

    fn fingerprint(matrix: &ConstraintMatrices<Fr>) -> Result<String, Box<dyn Error>> {
        let mut hash = Sha256::new();
        hash.update(b"akita-terminal-relation-matrices/v1");
        hash.update((matrix.num_vars as u64).to_le_bytes());
        hash.update((matrix.num_constraints as u64).to_le_bytes());
        let mut encoded = [0u8; 32];
        for rows in [&matrix.a, &matrix.b, &matrix.c] {
            for row in rows {
                hash.update((row.len() as u64).to_le_bytes());
                for (column, coefficient) in row {
                    hash.update((*column as u64).to_le_bytes());
                    coefficient.to_bytes_le(&mut encoded);
                    hash.update(encoded);
                }
            }
        }
        let mut output = String::with_capacity(64);
        for byte in hash.finalize() {
            write!(&mut output, "{byte:02x}")?;
        }
        Ok(output)
    }
}

#[expect(
    clippy::print_stdout,
    reason = "intentional complete-relation diagnostic"
)]
fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let directory = PathBuf::from(args.next().ok_or("pass reviewed census directory")?);
    let mode = args.next().unwrap_or_else(|| "positive".into());
    let mut terminal: Terminal =
        serde_json::from_reader(File::open(directory.join("terminal.json"))?)?;
    let census: Value = serde_json::from_reader(File::open(directory.join("census.json"))?)?;
    let params: TerminalFoldParams = serde_json::from_value(
        census
            .pointer("/terminal_geometry/schedule/terminal")
            .ok_or("terminal params")?
            .clone(),
    )?;
    let quotients: Quotients =
        serde_json::from_reader(File::open(directory.join("terminal-row-transport.json"))?)?;
    terminal.mutate(&mode)?;
    let builder = terminal.assemble(&params, &quotients, mode != "unknown")?;
    let witness = if mode == "unknown" {
        None
    } else {
        Some(builder.witness()?)
    };
    let matrix = builder.into_matrices();
    let result = witness
        .as_ref()
        .map(|witness| matrix.check_witness(witness));
    match (&*mode, result) {
        ("positive", Some(Ok(()))) | ("unknown", None) => {}
        ("coherent-e" | "coherent-t" | "claim", Some(Err(_))) => {}
        _ => return Err(format!("unexpected R1CS result: {result:?}").into()),
    }
    let nonzeros: usize = matrix
        .a
        .iter()
        .chain(&matrix.b)
        .chain(&matrix.c)
        .map(Vec::len)
        .sum();
    println!(
        "{}",
        json!({"mode":mode,"rows":matrix.num_constraints,"variables":matrix.num_vars,"nonzeros":nonzeros,"static_a_terms":3_145_728,"matrix_sha256":Terminal::fingerprint(&matrix)?,"r1cs_result":format!("{result:?}"),"witness_construction":"succeeded; result is actual matrix evaluation, not host rejection"})
    );
    Ok(())
}
