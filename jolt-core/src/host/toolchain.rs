use std::{
    fs::{self, read_to_string, File},
    io::Write,
    path::PathBuf,
};

use dirs::home_dir;
use eyre::{bail, Result};
use reqwest::blocking::Client;

const TOOLCHAIN_TAG: &str = "nightly-3c5f0ec3f4f98a2d211061a83bade8d62c6a6135";

/// Installs the toolchain if it is not already
pub fn install_toolchain() -> Result<()> {
    if has_toolchain() {
        return Ok(());
    }

    let client = Client::builder()
        .timeout(None)
        .user_agent("Mozilla/5.0")
        .build()?;

    let toolchain_url = toolchain_url();
    download_toolchain(&client, &toolchain_url)?;
    unpack_toolchain()?;
    link_toolchain()?;

    write_tag_file()
}

fn write_tag_file() -> Result<()> {
    let tag_path = toolchain_tag_file();
    let mut tag_file = File::create(tag_path)?;
    tag_file.write_all(TOOLCHAIN_TAG.as_bytes())?;
    Ok(())
}

fn link_toolchain() -> Result<()> {
    let link_path = jolt_dir().join("rust/build/host/stage2");
    let output = std::process::Command::new("rustup")
        .args([
            "toolchain",
            "link",
            "riscv32i-jolt-zkvm-elf",
            link_path.to_str().unwrap(),
        ])
        .output()?;

    if !output.status.success() {
        bail!("{}", String::from_utf8(output.stderr)?);
    }

    Ok(())
}

fn unpack_toolchain() -> Result<()> {
    let output = std::process::Command::new("tar")
        .args(["-xzf", "rust-toolchain.tar.gz"])
        .current_dir(jolt_dir())
        .output()?;

    if !output.status.success() {
        bail!("{}", String::from_utf8(output.stderr)?);
    }

    Ok(())
}

fn download_toolchain(client: &Client, url: &str) -> Result<()> {
    let bytes = client.get(url).send()?.bytes()?;
    let jolt_dir = jolt_dir();
    if !jolt_dir.exists() {
        fs::create_dir(&jolt_dir)?;
    }

    let path = jolt_dir.join("rust-toolchain.tar.gz");
    fs::write(path, &bytes)?;

    Ok(())
}

fn toolchain_url() -> String {
    let target = target_lexicon::HOST;
    format!(
        "https://github.com/a16z/rust/releases/download/{}/rust-toolchain-{}.tar.gz",
        TOOLCHAIN_TAG, target,
    )
}

fn has_toolchain() -> bool {
    let tag_path = toolchain_tag_file();
    if let Ok(tag) = read_to_string(tag_path) {
        tag == TOOLCHAIN_TAG
    } else {
        false
    }
}

fn jolt_dir() -> PathBuf {
    home_dir().unwrap().join(".jolt")
}

fn toolchain_tag_file() -> PathBuf {
    jolt_dir().join(".toolchaintag")
}
