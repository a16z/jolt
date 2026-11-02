use common::jolt_device::MemoryConfig;
use jolt_host::{default_target_dir, GuestElf};
use jolt_inlines_blake2 as _;
use jolt_inlines_keccak256 as _;
use std::{
    fs,
    fs::File,
    io::{Error, Result},
    process::Command,
};

pub fn build(
    guest: &str,
    function: &str,
    memory: &MemoryConfig,
    linker_memory_size: u64,
) -> Result<GuestElf> {
    let target = default_target_dir().join(format!("{guest}-{function}"));
    fs::create_dir_all(&target)?;
    // Cargo re-links the ELF even on cache hits; retain the lock until its bytes are read.
    let lock = File::create(target.join(".jolt-build.lock"))?;
    lock.lock()?;
    let status = Command::new(std::env::var_os("JOLT_PATH").unwrap_or_else(|| "jolt".into()))
        .args(["build", "-p", guest, "--mode", "std", "--backtrace", "off"])
        .args(["--stack-size", &memory.stack_size.to_string()])
        .args(["--heap-size", &memory.heap_size.to_string()])
        .args(["--memory-size", &linker_memory_size.to_string()])
        .args(["--", "--release", "--target-dir"])
        .arg(&target)
        .args(["--features", "guest"])
        .env("JOLT_FUNC_NAME", function)
        .status()?;
    if !status.success() {
        return Err(Error::other(format!("guest build failed: {status}")));
    }
    GuestElf::read(
        target
            .join("riscv64imac-zero-linux-musl/release")
            .join(guest),
    )
}
