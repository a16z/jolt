use crate::host::TOOLCHAIN_VERSION;
use std::{
    fs::{self, read_to_string, File},
    future::Future,
    io::Write,
    path::PathBuf,
};

use dirs::home_dir;
use eyre::{bail, eyre, Result};
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::Client;
#[cfg(not(target_arch = "wasm32"))]
use tokio::runtime::Runtime;

const TOOLCHAIN_TAG: &str = include_str!("../../../guest-toolchain-tag");
const DOWNLOAD_RETRIES: usize = 5;
const DELAY_BASE_MS: u64 = 500;

#[cfg(not(target_arch = "wasm32"))]
/// Installs the toolchain if it is not already
pub fn install_toolchain() -> Result<()> {
    if !has_toolchain() {
        let client = Client::builder().user_agent("Mozilla/5.0").build()?;
        let rt = Runtime::new()?;

        for channel in ["stable", "nightly"] {
            let toolchain_url = toolchain_url(channel);
            rt.block_on(retry_times(DOWNLOAD_RETRIES, DELAY_BASE_MS, || {
                download_toolchain(&client, &toolchain_url)
            }))?;
            unpack_toolchain(channel)?;
            remove_archive()?;
            link_toolchain(channel)?;
            write_tag_file()?;
            tracing::info!(
                "\"{channel}-jolt-{TOOLCHAIN_VERSION}\" toolchain installed successfully at {:?}",
                jolt_dir()
            );
        }
    }
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
pub fn install_no_std_toolchain() -> Result<()> {
    std::process::Command::new("rustup")
        .args(["target", "add", "riscv64imac-unknown-none-elf"])
        .output()?;
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
async fn retry_times<F, T, E>(times: usize, base_ms: u64, f: F) -> Result<T>
where
    F: Fn() -> E,
    E: Future<Output = Result<T>>,
{
    for i in 0..times {
        tracing::info!("Attempt {}/{}", i + 1, times);
        match f().await {
            Ok(t) => return Ok(t),
            Err(e) => {
                let timeout = delay_timeout(i, base_ms);
                tracing::warn!(
                    "Toolchain download error {i}/{times}: {e}. Retrying in {timeout}ms"
                );
                tokio::time::sleep(std::time::Duration::from_millis(timeout)).await;
            }
        }
    }
    Err(eyre!("failed after {} retries", times))
}

fn delay_timeout(i: usize, base_ms: u64) -> u64 {
    let timeout = 2u64.pow(i as u32) * base_ms;
    rand::random::<u64>() % timeout
}

fn write_tag_file() -> Result<()> {
    let tag_path = toolchain_tag_file();
    let mut tag_file = File::create(tag_path)?;
    tag_file.write_all(TOOLCHAIN_TAG.as_bytes())?;
    Ok(())
}

fn link_toolchain(channel: &str) -> Result<()> {
    let link_path = jolt_dir().join(format!("{channel}/rust/build/host/stage2"));
    let toolchain_name = format!("{channel}-jolt-{TOOLCHAIN_VERSION}");
    let output = std::process::Command::new("rustup")
        .args([
            "toolchain",
            "link",
            &toolchain_name,
            link_path.to_str().unwrap(),
        ])
        .output()?;

    if !output.status.success() {
        bail!("{}", String::from_utf8(output.stderr)?);
    }

    Ok(())
}

fn unpack_toolchain(channel: &str) -> Result<()> {
    let channel_dir = jolt_dir().join(channel);
    if !channel_dir.exists() {
        fs::create_dir(&channel_dir)?;
    }
    let output = std::process::Command::new("tar")
        .args([
            "-xzf",
            "rust-toolchain.tar.gz",
            "-C",
            channel_dir.to_str().unwrap(),
        ])
        .current_dir(jolt_dir())
        .output()?;

    if !output.status.success() {
        bail!("{}", String::from_utf8(output.stderr)?);
    }

    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
async fn download_toolchain(client: &Client, url: &str) -> Result<()> {
    use tracing::info;

    let jolt_dir = jolt_dir();
    let output_path = jolt_dir.join("rust-toolchain.tar.gz");
    info!("Downloading toolchain to {output_path:?}");
    if !jolt_dir.exists() {
        fs::create_dir(&jolt_dir)?;
    }

    // Check for partial download
    let (mut file, resume_from) = if output_path.exists() {
        let file_size = fs::metadata(&output_path)?.len();
        if let Ok(head_response) = client.head(url).send().await {
            // Manually parse content-length from headers
            let content_length = head_response
                .headers()
                .get("content-length")
                .and_then(|h| h.to_str().ok())
                .and_then(|s| s.parse::<u64>().ok());

            if let Some(content_length) = content_length {
                if file_size >= content_length {
                    fs::remove_file(&output_path)?;
                    (File::create(&output_path)?, 0)
                } else {
                    (File::options().append(true).open(&output_path)?, file_size)
                }
            } else {
                fs::remove_file(&output_path)?;
                (File::create(&output_path)?, 0)
            }
        } else {
            fs::remove_file(&output_path)?;
            (File::create(&output_path)?, 0)
        }
    } else {
        (File::create(&output_path)?, 0)
    };

    let mut response = client.get(url).send().await?;
    if resume_from > 0 {
        response = client
            .get(url)
            .header("Range", format!("bytes={resume_from}-"))
            .send()
            .await?;
    }

    if response.status().is_success() || response.status().as_u16() == 206 {
        let total_size = if resume_from > 0 {
            response
                .headers()
                .get("content-range")
                .and_then(|h| h.to_str().ok())
                .and_then(|s| s.split('/').nth(1))
                .and_then(|s| s.parse::<u64>().ok())
                .unwrap_or(response.content_length().unwrap_or(0) + resume_from)
        } else {
            response.content_length().unwrap_or(0)
        };

        let pb = ProgressBar::new(total_size);
        pb.set_position(resume_from);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")?
                .progress_chars("#>-"),
        );

        let mut downloaded = resume_from;
        while let Some(chunk) = response.chunk().await.unwrap() {
            file.write_all(&chunk)?;
            let new = downloaded + (chunk.len() as u64);
            pb.set_position(new);
            downloaded = new;
        }

        pb.finish_with_message("Download complete");

        Ok(())
    } else {
        Err(match response.error_for_status() {
            Ok(_) => eyre!("failed to download toolchain"),
            Err(err) => eyre!("failed to download toolchain: {}", err),
        })
    }
}

fn remove_archive() -> Result<()> {
    let toolchain_archive = jolt_dir().join("rust-toolchain.tar.gz");
    if toolchain_archive.exists() {
        fs::remove_file(&toolchain_archive)?;
    }
    Ok(())
}

fn toolchain_url(channel: &str) -> String {
    let target = target_lexicon::HOST;
    format!(
        "https://github.com/a16z/rust/releases/download/{channel}-{TOOLCHAIN_TAG}/rust-toolchain-{channel}-{target}.tar.gz",
    )
}

#[cfg(not(target_arch = "wasm32"))]
pub fn uninstall_no_std_toolchain() -> Result<()> {
    std::process::Command::new("rustup")
        .args(["target", "remove", "riscv64imac-unknown-none-elf"])
        .output()?;

    tracing::info!("\"riscv\" toolchains uninstalled successfully");
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
/// Uninstalls the toolchain if it is already installed
pub fn uninstall_toolchain() -> Result<()> {
    if !has_toolchain() {
        tracing::info!("Toolchain is not installed");
        return Ok(());
    }

    for channel in ["stable", "nightly"] {
        let toolchain_name = &format!("{channel}-jolt-{TOOLCHAIN_VERSION}");
        // Remove the linked toolchain from rustup
        let output = std::process::Command::new("rustup")
            .args(["toolchain", "remove", toolchain_name])
            .output()?;

        if !output.status.success() {
            bail!(
                "Failed to remove toolchain: {}",
                String::from_utf8(output.stderr)?
            );
        }

        // Remove the unpacked toolchain directory
        let link_path = jolt_dir().join(format!("{channel}/rust/build/host/stage2"));
        if link_path.exists() {
            fs::remove_dir_all(&link_path)?;
        }

        // Remove the downloaded toolchain archive
        remove_archive()?;
    }

    // Remove the toolchain tag file
    let tag_file = toolchain_tag_file();
    if tag_file.exists() {
        fs::remove_file(&tag_file)?;
    }

    tracing::info!("\"Jolt\" toolchain uninstalled successfully");
    Ok(())
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
