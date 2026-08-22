use std::env;
use std::error::Error;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;

fn compile_asm(
    compiler: &str,
    source: &Path,
    object: &Path,
    asm_dir: &Path,
    apple_arch: Option<&str>,
) -> Result<(), Box<dyn Error>> {
    let mut command = Command::new(compiler);
    let _ = command.args(["-c", "-I"]).arg(asm_dir);
    if let Some(arch) = apple_arch {
        let _ = command.args(["-arch", arch]);
    }
    let status = command.arg(source).arg("-o").arg(object).status()?;
    if !status.success() {
        return Err(io::Error::other(format!("{compiler} failed with {status}")).into());
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    for architecture in ["aarch64", "x86_64"] {
        let asm_dir = Path::new("asm").join(architecture);
        let files = vec![
            "fp128_add.S",
            "fp128_add_body.inc",
            "fp128_load_a7f7.inc",
            "fp128_sub.S",
            "fp128_sub_body.inc",
            "fp128_mul.S",
            "fp128_mul_body.inc",
        ];
        for file in files {
            println!("cargo:rerun-if-changed={}", asm_dir.join(file).display());
        }
    }

    if env::var_os("CARGO_FEATURE_FP128_PROOF_LINKAGE").is_none() {
        return Ok(());
    }

    let target_arch = env::var("CARGO_CFG_TARGET_ARCH")?;
    if !matches!(target_arch.as_str(), "aarch64" | "x86_64") {
        return Ok(());
    }

    let asm_dir = Path::new("asm").join(&target_arch);

    let out_dir = PathBuf::from(
        env::var_os("OUT_DIR")
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "Cargo did not set OUT_DIR"))?,
    );
    let compiler = env::var("CC").unwrap_or_else(|_| "cc".to_owned());
    let apple_arch = if env::var("CARGO_CFG_TARGET_VENDOR").as_deref() == Ok("apple") {
        Some(match target_arch.as_str() {
            "aarch64" => "arm64",
            "x86_64" => "x86_64",
            _ => unreachable!(),
        })
    } else {
        None
    };

    let stems = ["fp128_add", "fp128_sub", "fp128_mul"];
    for stem in stems {
        compile_asm(
            &compiler,
            &asm_dir.join(format!("{stem}.S")),
            &out_dir.join(format!("{stem}.o")),
            &asm_dir,
            apple_arch,
        )?;
    }
    Ok(())
}
