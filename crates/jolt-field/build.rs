use std::env;
use std::error::Error;
use std::io;
use std::path::{Path, PathBuf};

fn compile_asm(
    compiler: &cc::Tool,
    source: &Path,
    object: &Path,
    asm_dir: &Path,
) -> Result<(), Box<dyn Error>> {
    let mut command = compiler.to_command();
    let _ = command.args(["-c", "-I"]).arg(asm_dir);
    let status = command.arg(source).arg("-o").arg(object).status()?;
    if !status.success() {
        return Err(io::Error::other(format!(
            "{} failed with {status}",
            compiler.path().display()
        ))
        .into());
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    for architecture in ["aarch64", "x86_64"] {
        let asm_dir = Path::new("asm").join(architecture);
        let mut files = vec![
            "fp128_add.S",
            "fp128_add_body.inc",
            "fp128_load_a7f7.inc",
            "fp128_sub.S",
            "fp128_sub_body.inc",
            "fp128_mul.S",
            "fp128_mul_body.inc",
        ];
        if architecture == "x86_64" {
            files.extend(["fp128_mul_bmi2_adx.S", "fp128_mul_bmi2_adx_body.inc"]);
        }
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

    let target = env::var("TARGET")?;
    let host = env::var("HOST")?;
    let compiler = cc::Build::new()
        .host(&host)
        .target(&target)
        .try_get_compiler()?;

    let asm_dir = Path::new("asm").join(&target_arch);

    let out_dir = PathBuf::from(
        env::var_os("OUT_DIR")
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "Cargo did not set OUT_DIR"))?,
    );
    let mut stems = vec!["fp128_add", "fp128_sub", "fp128_mul"];
    if target_arch == "x86_64" {
        stems.push("fp128_mul_bmi2_adx");
    }
    for stem in stems {
        compile_asm(
            &compiler,
            &asm_dir.join(format!("{stem}.S")),
            &out_dir.join(format!("{stem}.o")),
            &asm_dir,
        )?;
    }
    Ok(())
}
