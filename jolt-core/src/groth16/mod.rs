//! Groth16 Circuit for Stage 1 Verification
//!
//! This module implements a Groth16 circuit that verifies Stage 1 of the Jolt proof
//! (Spartan outer sumcheck / R1CS constraint checking).
//!
//! ## Design Philosophy
//!
//! **Goal**: Cheap EVM verification, NOT zero-knowledge
//! **Strategy**: All data is public inputs (no privacy needed)
//!
//! ## Challenge Handling
//!
//! The circuit avoids expensive Blake2b hashing by:
//! 1. Prover generates Fiat-Shamir challenges outside the circuit (via Blake2b)
//! 2. Challenges are passed as public inputs to the circuit
//! 3. Circuit verifies: "given these challenges, does the proof check out?"
//!
//! This saves thousands of constraints compared to implementing Blake2b in-circuit.
//!
//! ## Features
//!
//! - `groth16-stable`: Use stable arkworks (v0.5.0)
//! - `groth16-git`: Use git master arkworks (for testing latest optimizations)
//!
//! ## Usage
//!
//! ```ignore
//! use jolt_core::groth16::{Stage1Circuit, Stage1CircuitData};
//! use ark_groth16::Groth16;
//! use ark_bn254::Bn254;
//!
//! // Extract circuit data from Stage 1 proof
//! let circuit_data = Stage1CircuitData::from_stage1_proof(...);
//! let circuit = Stage1Circuit::from_data(circuit_data);
//!
//! // Groth16 setup, prove, verify
//! let (pk, vk) = Groth16::<Bn254>::setup(circuit.clone(), &mut rng)?;
//! let proof = Groth16::<Bn254>::prove(&pk, circuit, &mut rng)?;
//! let is_valid = Groth16::<Bn254>::verify(&vk, &public_inputs, &proof)?;
//! ```

#[cfg(any(feature = "groth16-stable", feature = "groth16-git"))]
pub mod circuit;

#[cfg(any(feature = "groth16-stable", feature = "groth16-git"))]
pub mod witness;

#[cfg(any(feature = "groth16-stable", feature = "groth16-git"))]
pub mod gadgets;

#[cfg(any(feature = "groth16-stable", feature = "groth16-git"))]
pub mod benchmarks;

#[cfg(any(feature = "groth16-stable", feature = "groth16-git"))]
#[cfg(test)]
mod tests;

#[cfg(any(feature = "groth16-stable", feature = "groth16-git"))]
pub use circuit::{Stage1Circuit, Stage1CircuitConfig};

#[cfg(any(feature = "groth16-stable", feature = "groth16-git"))]
pub use witness::Stage1CircuitData;
