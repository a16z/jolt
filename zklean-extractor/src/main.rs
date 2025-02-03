mod mle_ast;
use crate::mle_ast::MleAst;
mod util;
use crate::util::subtables;

use clap::Parser;
use util::ZkLeanReprField;
use std::io::Write;

/// Simple argument parsing to allow writing to a file.
#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Args {
    /// File to write output to instead of stdout
    #[arg(short, long)]
    file: Option<String>,
}

/// Evaluate all the subtable MLEs in [`subtables`] using a given [`ZkLeanReprField`] and print the
/// results.
fn print_all_subtables<F: ZkLeanReprField>(
    f: &mut impl Write,
    reg_size: usize,
    kind: &'static str,
) -> std::io::Result<()> {
    for (name, subtable) in subtables(reg_size) {
        let mle_name = format!("{name}_{reg_size}_{kind}_subtable");
        subtable
            .evaluate_mle(&F::register('x', reg_size))
            .write_lean_mle(f, &mle_name, reg_size)?;
    }
    Ok(())
}

fn main() -> std::io::Result<()> {
    let args = Args::parse();
    let mut f: Box<dyn Write> = match args.file {
        None => Box::new(std::io::stdout()),
        Some(fname) => Box::new(std::fs::File::create(fname)?),
    };

    f.write(b"import ZkLean.LookupTable\n")?;
    f.write(b"\n")?;
    f.write(b"/-\nSubtable MLEs in AST form\n-/\n")?;
    print_all_subtables::<MleAst<2048>>(&mut f.as_mut(), 8, "ast")?;
    print_all_subtables::<MleAst<4096>>(&mut f.as_mut(), 16, "ast")?;

    Ok(())
}
