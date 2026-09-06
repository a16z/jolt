use std::env;
use std::error::Error;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;

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

fn require_fp64_registered_target() -> Result<(), Box<dyn Error>> {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?);
    let matrix_tool = manifest_dir
        .join("../..")
        .join("scripts/fp64_certified_matrix.py");
    let output = Command::new("python3")
        .arg(matrix_tool)
        .args(["validate-build", "--target-triple"])
        .arg(env::var("TARGET")?)
        .arg("--architecture")
        .arg(env::var("CARGO_CFG_TARGET_ARCH")?)
        .arg("--vendor")
        .arg(env::var("CARGO_CFG_TARGET_VENDOR")?)
        .arg("--target-os")
        .arg(env::var("CARGO_CFG_TARGET_OS")?)
        .arg("--target-env")
        .arg(env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default())
        .arg("--endian")
        .arg(env::var("CARGO_CFG_TARGET_ENDIAN")?)
        .arg("--pointer-width")
        .arg(env::var("CARGO_CFG_TARGET_POINTER_WIDTH")?)
        .arg("--target-features")
        .arg(env::var("CARGO_CFG_TARGET_FEATURE")?)
        .arg("--profile")
        .arg(env::var("PROFILE")?)
        .arg("--opt-level")
        .arg(env::var("OPT_LEVEL")?)
        .arg("--debug")
        .arg(env::var("DEBUG")?)
        .arg("--rustc")
        .arg(env::var("RUSTC")?)
        .output()?;
    if !output.status.success() {
        return Err(io::Error::other(format!(
            "Fp64 proof build is outside the registered matrix:\n{}",
            String::from_utf8_lossy(&output.stderr)
        ))
        .into());
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("cargo:rerun-if-changed=../../proofs/hol-light/fp64-certified-builds.json");
    println!("cargo:rerun-if-changed=../../scripts/fp64_certified_matrix.py");
    for variable in [
        "TARGET",
        "CARGO_CFG_TARGET_ARCH",
        "CARGO_CFG_TARGET_VENDOR",
        "CARGO_CFG_TARGET_OS",
        "CARGO_CFG_TARGET_ENV",
        "CARGO_CFG_TARGET_ENDIAN",
        "CARGO_CFG_TARGET_POINTER_WIDTH",
        "CARGO_CFG_TARGET_FEATURE",
        "PROFILE",
        "OPT_LEVEL",
        "DEBUG",
        "RUSTC",
        "JOLT_FP64_PROOF_MATRIX_CONTRACT",
        "RUSTFLAGS",
        "CARGO_ENCODED_RUSTFLAGS",
        "CARGO_BUILD_RUSTFLAGS",
        "CARGO_BUILD_TARGET",
        "CARGO_INCREMENTAL",
        "RUSTC_WRAPPER",
        "RUSTC_WORKSPACE_WRAPPER",
        "CARGO_PROFILE_RELEASE_OPT_LEVEL",
        "CARGO_PROFILE_RELEASE_LTO",
        "CARGO_PROFILE_RELEASE_CODEGEN_UNITS",
        "CARGO_PROFILE_RELEASE_DEBUG",
        "CARGO_PROFILE_RELEASE_DEBUG_ASSERTIONS",
        "CARGO_PROFILE_RELEASE_OVERFLOW_CHECKS",
        "CARGO_PROFILE_RELEASE_SPLIT_DEBUGINFO",
        "CARGO_PROFILE_RELEASE_STRIP",
        "CARGO_PROFILE_RELEASE_RPATH",
        "CARGO_PROFILE_RELEASE_INCREMENTAL",
        "CARGO_PROFILE_RELEASE_PANIC",
    ] {
        println!("cargo:rerun-if-env-changed={variable}");
    }
    for architecture in ["aarch64", "x86_64"] {
        let asm_dir = Path::new("asm").join(architecture);
        let mut files = vec![
            "fp64_add.S",
            "fp64_add_body.inc",
            "fp64_add_linux_body.inc",
            "fp64_sub.S",
            "fp64_sub_body.inc",
            "fp64_sub_linux_body.inc",
            "fp64_mul.S",
            "fp64_mul_body.inc",
            "fp64_mul_linux_body.inc",
            "fp128_add.S",
            "fp128_add_body.inc",
            "fp128_load_a7f7.inc",
            "fp128_sub.S",
            "fp128_sub_body.inc",
            "fp128_mul.S",
            "fp128_mul_body.inc",
        ];
        if architecture == "x86_64" {
            files.extend([
                "fp64_mul_bmi2.S",
                "fp64_mul_bmi2_body.inc",
                "fp128_mul_bmi2_adx.S",
                "fp128_mul_bmi2_adx_body.inc",
            ]);
        }
        for file in files {
            println!("cargo:rerun-if-changed={}", asm_dir.join(file).display());
        }
    }

    let fp64_linkage = env::var_os("CARGO_FEATURE_FP64_PROOF_LINKAGE").is_some();
    let fp128_linkage = env::var_os("CARGO_FEATURE_FP128_PROOF_LINKAGE").is_some();
    if !fp64_linkage && !fp128_linkage {
        return Ok(());
    }

    if fp64_linkage {
        require_fp64_registered_target()?;
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
    let mut stems = Vec::new();
    if fp64_linkage {
        stems.extend(["fp64_add", "fp64_sub", "fp64_mul"]);
        if target_arch == "x86_64" {
            stems.push("fp64_mul_bmi2");
        }
    }
    if fp128_linkage {
        stems.extend(["fp128_add", "fp128_sub", "fp128_mul"]);
        if target_arch == "x86_64" {
            stems.push("fp128_mul_bmi2_adx");
        }
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
