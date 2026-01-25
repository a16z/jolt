//! Native decompression/conversion layer for guest verification inputs.
//!
//! Transport bytes use arkworks canonical encoding (typically compressed) and should be decoded
//! with validation in native execution. The output of this module is a Jolt-owned guest encoding
//! (see `jolt_core::zkvm::guest_serde`) designed to be cheap to decode inside zkVM programs.

use std::io;

use ark_serialize::CanonicalDeserialize;

use crate::host_utils::{JoltDevice, JoltVerifierPreprocessing, RV64IMACProof, F, PCS};

/// Convert transport bytes (canonical, compressed) into guest bytes (decompressed+converted).
///
/// - Transport decode: **validated** (`Validate::Yes` via `deserialize_compressed`)
/// - Guest decode: **unvalidated** (trusted output of this function)
pub fn decompress_transport_bytes_to_guest_bytes(transport: &[u8]) -> io::Result<Vec<u8>> {
    use jolt_core::zkvm::guest_serde::GuestSerialize as _;

    let mut cursor = std::io::Cursor::new(transport);

    // Transport decode (validated by default).
    let verifier_preprocessing =
        JoltVerifierPreprocessing::<F, PCS>::deserialize_compressed(&mut cursor).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "transport preprocessing decode failed",
            )
        })?;
    let n = u32::deserialize_compressed(&mut cursor).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "transport proof count decode failed",
        )
    })?;

    // Decode proofs/devices from transport.
    let mut proofs: Vec<RV64IMACProof> = Vec::with_capacity(n as usize);
    let mut devices: Vec<JoltDevice> = Vec::with_capacity(n as usize);
    for _ in 0..n {
        let proof = RV64IMACProof::deserialize_compressed(&mut cursor).map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidData, "transport proof decode failed")
        })?;
        let device = JoltDevice::deserialize_compressed(&mut cursor).map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidData, "transport device decode failed")
        })?;
        proofs.push(proof);
        devices.push(device);
    }

    // Emit guest bytes (decompressed+converted encoding).
    let mut out = Vec::new();
    verifier_preprocessing.guest_serialize(&mut out)?;
    n.guest_serialize(&mut out)?;
    for (p, d) in proofs.iter().zip(devices.iter()) {
        p.guest_serialize(&mut out)?;
        d.guest_serialize(&mut out)?;
    }
    Ok(out)
}
