#![feature(iter_intersperse)]

mod mle_ast;
use crate::mle_ast::MleAst;
mod util;
use crate::util::ZkLeanReprField;
mod subtable;
use crate::subtable::NamedSubtable;
mod instruction;
use crate::instruction::NamedInstruction;
mod r1cs;
use crate::r1cs::*;
mod flags;
use crate::flags::*;

use clap::Parser;

use common::constants::{DEFAULT_MAX_INPUT_SIZE, DEFAULT_MAX_OUTPUT_SIZE};

/// Simple argument parsing to allow writing to a file.
#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Args {
    /// File to write output to instead of stdout
    #[arg(short, long)]
    file: Option<String>,
}

/// Evaluate all the subtable MLEs using a given [`ZkLeanReprField`] and print the results.
fn print_all_subtables<F: ZkLeanReprField>(
    f: &mut impl std::io::Write,
    reg_size: usize,
) -> std::io::Result<()> {
    for subtable in NamedSubtable::<F>::enumerate() {
        subtable.zklean_pretty_print(f, reg_size)?;
    }
    Ok(())
}

/// Evaluate all the instruction MLEs using a given [`ZkLeanReprField`] and size and print the
/// results.
fn print_all_instructions<F: ZkLeanReprField, const WORD_SIZE: usize>(
    f: &mut impl std::io::Write,
    c: usize,
    log_m: usize,
) -> std::io::Result<()> {
    for instruction in NamedInstruction::<WORD_SIZE>::enumerate() {
        instruction.zklean_pretty_print::<F>(f, c, log_m)?;
    }
    Ok(())
}

fn main() -> std::io::Result<()> {
    let args = Args::parse();
    let mut f: Box<dyn std::io::Write> = match args.file {
        None => Box::new(std::io::stdout()),
        Some(fname) => Box::new(std::fs::File::create(fname)?),
    };

    f.write(b"import ZkLean\n")?;
    f.write(b"\n")?;
    f.write(b"/-\nSubtable MLEs in AST form\n-/\n")?;
    //print_all_subtables::<MleAst<2048>>(&mut f.as_mut(), 8)?;
    print_all_subtables::<MleAst<16000>>(&mut f.as_mut(), 16)?;
    f.write(b"\n")?;
    f.write(b"/-\nCombining polynomials in AST form\n-/\n")?;
    print_all_instructions::<MleAst<2048>, 32>(&mut f.as_mut(), 4, 16)?;
    //print_all_instructions::<MleAst<4096>, 64>(&mut f.as_mut(), 8, 16)?;
    f.write(b"\n")?;

    let r1cs = ZkLeanR1CSConstraints::extract(DEFAULT_MAX_INPUT_SIZE, DEFAULT_MAX_OUTPUT_SIZE);
    r1cs.zklean_pretty_print(&mut f)?;
    f.write(b"\n")?;

    let cases = ZkLeanLookupCases::extract();
    cases.zklean_pretty_print(&mut f)?;

    Ok(())
}
