mod compute_to_cpu;
mod kernel_resolution;
mod module_copy;
mod party_to_compute;
mod protocol_to_concrete;
mod roles;
mod support;
mod verify;

pub use compute_to_cpu::lower_compute_to_cpu;
pub use kernel_resolution::{resolve_compute_kernels_with, ComputeKernelSpec, KernelRegistry};
pub use party_to_compute::{lower_party_to_compute, PartyToComputeLowering};
pub use protocol_to_concrete::lower_piop_and_fiat_shamir;
pub use roles::{
    derive_prover_role, derive_verifier_role, project_party, project_prover_party,
    project_verifier_party,
};
pub use verify::{verify_concrete_transcript, VerifyError};
