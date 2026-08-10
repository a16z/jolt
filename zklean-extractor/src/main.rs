//#![feature(iter_intersperse, generic_const_exprs, generic_const_items)]
//#![allow(incomplete_features)] // Silence warnings for generic_const_exprs

use std::path::PathBuf;

use zklean_extractor::constants::*;
use zklean_extractor::instruction::*;
use zklean_extractor::lean_tests::*;
use zklean_extractor::lookup_artifact::{verify_checkout_revision, LookupArtifact};
use zklean_extractor::lookup_table_flags::*;
use zklean_extractor::lookups::*;
use zklean_extractor::modules::*;
use zklean_extractor::r1cs::*;
use zklean_extractor::sumchecks::*;

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

    /// Write a standalone, provenance-bearing Lean lookup artifact
    #[arg(long, value_name = "PATH", conflicts_with_all = ["file", "package_path", "template_dir"])]
    lookup_artifact_path: Option<PathBuf>,

    /// Exact Jolt Git revision represented by a standalone lookup artifact
    #[arg(long, value_name = "40_HEX", requires = "lookup_artifact_path")]
    source_revision: Option<String>,
}

fn write_flat_file(
    f: &mut impl std::io::Write,
    modules: Vec<Box<dyn AsModule>>,
) -> std::io::Result<()> {
    let modules = modules
        .into_iter()
        .map(|module| module.as_module())
        .collect::<std::io::Result<Vec<_>>>()?;
    let generated_imports = modules
        .iter()
        .map(|module| format!("Jolt.{}", module.name))
        .collect::<std::collections::HashSet<_>>();
    let mut import_set = std::collections::HashSet::new();
    let mut contents: Vec<u8> = vec![];

    for module in modules {
        for import in module.imports {
            if !generated_imports.contains(&import) {
                let _ = import_set.insert(import);
            }
        }

        let mut separator = Vec::from(b"\n\n");
        contents.append(&mut separator);
        contents.extend(module.contents);
    }

    for i in import_set {
        f.write_fmt(format_args!("import {i}\n"))?;
    }

    f.write_all(&contents)?;

    Ok(())
}

type ParameterSet = RV64IParameterSet;

fn extract_modules<const XLEN: usize>() -> Vec<Box<dyn AsModule>> {
    let mut rng = rand_core::OsRng;

    let mut modules: Vec<Box<dyn AsModule>> = vec![
        Box::new(ZkLeanR1CSConstraints::<ParameterSet>::extract()),
        Box::new(ZkLeanInstructions::<ParameterSet>::extract()),
        Box::new(ZkLeanSumchecks::<ark_bn254::Fr>::extract::<XLEN>()),
    ];
    let lookup_tables = ZkLeanLookupTables::<XLEN>::extract().expect("lookup extraction failed");
    modules.extend(
        lookup_tables
            .as_modules()
            .expect("lookup module generation failed")
            .into_iter()
            .map(|module| Box::new(module) as Box<dyn AsModule>),
    );
    modules.push(Box::new(ZkLeanLookupTableFlags::<XLEN>::extract()));
    modules.push(Box::new(ZkLeanTests::<XLEN>::extract(&mut rng)));
    modules
}

fn main() -> Result<(), FSError> {
    let args = Args::parse();

    if let Some(artifact_path) = args.lookup_artifact_path {
        let source_revision = args.source_revision.ok_or_else(|| {
            FSError::TemplateError(
                "--source-revision is required with --lookup-artifact-path".to_string(),
            )
        })?;
        verify_checkout_revision(&source_revision).map_err(FSError::TemplateError)?;
        let artifact =
            LookupArtifact::extract::<64>(&source_revision).map_err(FSError::TemplateError)?;
        artifact.write_to(&artifact_path, args.overwrite)?;
        println!("Created Lean lookup artifact at {artifact_path:?}");
        return Ok(());
    }

    let modules = match ParameterSet::XLEN {
        32 => extract_modules::<32>(),
        64 => extract_modules::<64>(),
        _ => panic!("Unsupported architecture size"),
    };

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
