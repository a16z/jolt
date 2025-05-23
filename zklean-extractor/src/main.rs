#![feature(iter_intersperse, generic_const_exprs, generic_const_items)]
#![allow(incomplete_features)] // Silence warnings for generic_const_exprs

use std::path::PathBuf;

mod constants;
use crate::constants::*;
mod mle_ast;
use crate::mle_ast::*;
mod util;
//use crate::util::*;
mod subtable;
use crate::subtable::*;
mod instruction;
use crate::instruction::*;
mod r1cs;
use crate::r1cs::*;
mod flags;
use crate::flags::*;
mod modules;
use crate::modules::*;

use build_fs_tree::{Build, MergeableFileSystemTree};
use clap::Parser;

/// Simple argument parsing to allow writing to a file.
#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Args {
    /// File to write output to instead of stdout; ignored if -p is specified
    #[arg(short, long)]
    file: Option<String>,

    /// Path to save Jolt ZkLean package to
    #[arg(short, long)]
    package_path: Option<PathBuf>,

    /// Directory to use as a package template instead of `./package-template`; ignored if -p is
    /// not specified
    #[arg(short, long)]
    template_dir: Option<PathBuf>,

    /// Don't complain if the directory specified with -p already exists. NB: This will clobber any
    /// files in the target directory that collide with generated files or files in the template!
    /// Ignored if -p is not specified.
    #[arg(short, long, default_value_t = false)]
    overwrite: bool,
}

fn write_flat_file(
    f: &mut impl std::io::Write,
    modules: Vec<Box<dyn AsModule>>,
) -> std::io::Result<()> {
    let mut import_set = std::collections::HashSet::new();
    let mut contents: Vec<u8> = vec![];

    for module in modules {
        let mut module = module.as_module()?;

        for import in module.imports {
            let _ = import_set.insert(import);
        }

        let mut separator = Vec::from(b"\n\n");
        contents.append(&mut separator);
        contents.append(&mut module.contents);
    }

    for i in import_set {
        f.write_fmt(format_args!("import {i}\n"))?;
    }

    f.write_all(&contents)?;

    Ok(())
}

type ParameterSet = RV32IParameterSet;

fn main() -> Result<(), FSError> {
    let args = Args::parse();

    let modules: Vec<Box<dyn AsModule>> = vec![
        Box::new(ZkLeanR1CSConstraints::<ParameterSet>::extract()),
        Box::new(ZkLeanSubtables::<MleAst<16000>, ParameterSet>::extract()),
        Box::new(ZkLeanInstructions::<ParameterSet>::extract()),
        Box::new(ZkLeanLookupCases::<ParameterSet>::extract()),
    ];

    if let Some(package_path) = args.package_path {
        let tree = make_jolt_zk_lean_package(&args.template_dir, modules)?;
        if args.overwrite {
            MergeableFileSystemTree::from(tree).build(&package_path)
        } else {
            tree.build(&package_path)
        }?;
        println!("Created Lean4 package at {package_path:?}");
    } else {
        let mut f: Box<dyn std::io::Write> = match args.file {
            None => Box::new(std::io::stdout()),
            Some(fname) => Box::new(std::fs::File::create(fname)?),
        };
        write_flat_file(&mut f, modules)?;
    }

    Ok(())
}
