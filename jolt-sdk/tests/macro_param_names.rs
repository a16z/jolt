//! Guest parameters may use any identifier, including names the generated
//! host-side code also needs internally.
#![cfg(feature = "host")]
#![allow(unexpected_cfgs, dead_code, clippy::too_many_arguments)]
extern crate jolt_sdk as jolt;

#[jolt::provable]
fn generated_names(
    program: u32,
    preprocessing: u32,
    backend: u32,
    target_dir: u32,
    path: u32,
    output: u32,
    panic: u32,
    proof: u32,
    input_bytes: Vec<u8>,
    trusted_advice_commitment: u32,
    io_device: u32,
) -> u32 {
    program
        + preprocessing
        + backend
        + target_dir
        + path
        + output
        + panic
        + proof
        + input_bytes.len() as u32
        + trusted_advice_commitment
        + io_device
}

#[jolt::provable]
fn advice_names(
    untrusted_advice_bytes: jolt::UntrustedAdvice<Vec<u8>>,
    trusted_advice_bytes: jolt::TrustedAdvice<Vec<u8>>,
) -> u32 {
    (untrusted_advice_bytes.len() + trusted_advice_bytes.len()) as u32
}

#[test]
fn provable_accepts_generated_names_as_parameters() {}
