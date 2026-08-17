//! Host-side offload seams for the optimized kernels.
//!
//! Everything the Metal adapters need from the `optimized` tier that is not
//! part of the CPU proving path lives here: `prepare_metal_*` entry points,
//! the `metal_*` state-machine methods on the optimized kernel types
//! (inherent impls on types defined in `crate::optimized`), and the row/plan
//! types that exist only for device residency. The parent `metal` module is
//! compiled only under `all(feature = "metal", target_os = "macos")`, so no
//! item in this tree carries its own gate.

pub(crate) mod booleanity;
pub(crate) mod bytecode_read_raf;
pub(crate) mod hamming_weight_claim_reduction;
pub(crate) mod instruction_input;
pub(crate) mod instruction_ra_virtualization;
pub(crate) mod instruction_read_raf;
pub(crate) mod ram_trace;
pub(crate) mod registers_val_evaluation;
pub(crate) mod spartan_outer;
mod support;
