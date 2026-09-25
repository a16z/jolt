//! Reports the default Metal device and its limits.
//!
//! Exit status: 0 when a supported device is present, 2 when Metal is
//! unavailable (non-macOS, no device, or below Apple GPU family 7), 1 on any
//! other error. `scripts/metal-report.sh` and the macOS CI job read both the
//! output and the status.

#![expect(clippy::print_stdout, reason = "the probe's output is its report")]

use std::process::ExitCode;

use jolt_metal::runtime::Device;
use jolt_metal::ErrorClass;

fn main() -> ExitCode {
    let device = match Device::system_default() {
        Ok(device) => device,
        Err(error) => {
            println!("device: none");
            println!("error: {error}");
            return match error.class() {
                ErrorClass::Unavailable => ExitCode::from(2),
                _ => ExitCode::FAILURE,
            };
        }
    };
    let limits = device.limits();
    println!("device: {}", device.name());
    println!("apple_family: {}", limits.apple_family);
    println!("max_buffer_length: {}", limits.max_buffer_length);
    println!(
        "recommended_max_working_set_size: {}",
        limits.recommended_max_working_set_size
    );
    println!(
        "max_threads_per_threadgroup: {}",
        limits.max_threads_per_threadgroup
    );
    ExitCode::SUCCESS
}
