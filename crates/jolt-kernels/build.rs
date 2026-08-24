use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const KERNEL_FILES: &[&str] = &[
    "prelude.cu",
    "probe.cu",
    "arith.cu",
    "tables.cu",
    "sumcheck_common.cu",
    "precommitted_reduction.cu",
    "msm.cu",
    "commit_increments.cu",
    "pairing.cu",
    "opening.cu",
    "scan.cu",
    "lt_poly.cu",
    "half_fold.cu",
    "dense_product.cu",
    "ra_poly.cu",
    "ram_ra_reduction.cu",
    "suffixes.cu",
    "prefixes.cu",
    "prefix_mle.cu",
    "combine.cu",
    "unreduced.cu",
    "product_accum.cu",
    "read_write_matrix.cu",
    "rs2_claim.cu",
    "address_major_matrix.cu",
    "address_phase.cu",
    "cycle_rounds.cu",
    "ram_read_write.cu",
    "registers_read_write.cu",
    "instruction_ra_virtualization.cu",
    "ram_ra_virtualization.cu",
    "booleanity_cycle.cu",
    "bytecode_read_raf.cu",
    "spartan_outer.cu",
    "one_hot_fold.cu",
    "hamming_weight_claim_reduction.cu",
    "ram_output_check.cu",
    "spartan_product.cu",
    "spartan_shift.cu",
    "booleanity_address.cu",
    "bytecode_read_raf_address.cu",
    "instruction_input.cu",
];

fn main() -> Result<(), Box<dyn Error>> {
    println!("cargo:rerun-if-changed=build.rs");
    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return Ok(());
    }
    println!("cargo:rerun-if-env-changed=JOLT_CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=JOLT_NVCC");

    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?);
    let kernel_dir = manifest.join("src").join("cuda").join("kernels");
    let out_dir = PathBuf::from(env::var("OUT_DIR")?);

    let mut source = String::new();
    for name in KERNEL_FILES {
        let path = kernel_dir.join(name);
        println!("cargo:rerun-if-changed={}", path.display());
        let text = fs::read_to_string(&path)
            .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
        source.push_str(&text);
        source.push('\n');
    }

    let present: Vec<String> = fs::read_dir(&kernel_dir)?
        .filter_map(Result::ok)
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .filter(|name| {
            Path::new(name)
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("cu"))
        })
        .filter(|name| !KERNEL_FILES.contains(&name.as_str()))
        .collect();
    if !present.is_empty() {
        return Err(format!(
            "{} is not listed in build.rs's KERNEL_FILES, so its kernels would be absent from \
             the compiled module",
            present.join(", ")
        )
        .into());
    }

    let source_path = out_dir.join("kernels_all.cu");
    fs::write(&source_path, source.as_bytes())?;

    let arch = env::var("JOLT_CUDA_ARCH").unwrap_or_else(|_| "native".to_owned());
    let nvcc = env::var("JOLT_NVCC").unwrap_or_else(|_| "nvcc".to_owned());
    let cubin_path = out_dir.join("kernels.cubin");

    let result = Command::new(&nvcc)
        .arg(format!("-arch={arch}"))
        .arg("--split-compile=0")
        .arg("-Wno-deprecated-declarations")
        .arg("-cubin")
        .arg("-o")
        .arg(&cubin_path)
        .arg(&source_path)
        .output();

    let output = match result {
        Ok(output) => output,
        Err(error) => {
            return Err(format!(
                "could not run `{nvcc}` ({error}). Building jolt-kernels with the `cuda` feature \
                 compiles the kernels ahead of time, which needs the CUDA toolkit on PATH; set \
                 JOLT_NVCC to point at it."
            )
            .into())
        }
    };
    if !output.status.success() {
        return Err(format!(
            "`{nvcc} -arch={arch}` failed for {}:\n{}",
            source_path.display(),
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }
    for line in String::from_utf8_lossy(&output.stderr).lines() {
        if !line.trim().is_empty() {
            println!("cargo:warning={line}");
        }
    }

    println!("cargo:rustc-env=JOLT_CUDA_KERNEL_ARCH={arch}");
    Ok(())
}
