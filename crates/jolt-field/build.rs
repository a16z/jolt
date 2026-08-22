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
    apple: bool,
) -> Result<(), Box<dyn Error>> {
    let mut command = Command::new(compiler);
    let _ = command.args(["-c", "-I"]).arg(asm_dir);
    if apple {
        let _ = command.args(["-arch", "arm64"]);
    }
    let status = command.arg(source).arg("-o").arg(object).status()?;
    if !status.success() {
        return Err(io::Error::other(format!("{compiler} failed with {status}")).into());
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let asm_dir = PathBuf::from("asm/aarch64");
    for file in [
        "fp128_add.S",
        "fp128_add_body.inc",
        "fp128_sub.S",
        "fp128_sub_body.inc",
    ] {
        println!("cargo:rerun-if-changed={}", asm_dir.join(file).display());
    }

    if env::var_os("CARGO_FEATURE_FP128_PROOF_LINKAGE").is_none() {
        return Ok(());
    }

    if env::var("CARGO_CFG_TARGET_ARCH").as_deref() != Ok("aarch64") {
        return Ok(());
    }

    let out_dir = PathBuf::from(
        env::var_os("OUT_DIR")
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "Cargo did not set OUT_DIR"))?,
    );
    let compiler = env::var("CC").unwrap_or_else(|_| "cc".to_owned());
    let apple = env::var("CARGO_CFG_TARGET_VENDOR").as_deref() == Ok("apple");

    for stem in ["fp128_add", "fp128_sub"] {
        compile_asm(
            &compiler,
            &asm_dir.join(format!("{stem}.S")),
            &out_dir.join(format!("{stem}.o")),
            &asm_dir,
            apple,
        )?;
    }
    Ok(())
}
