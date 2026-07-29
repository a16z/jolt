//! Optimized kernels: performance-tier [`PrepareKernel`](crate::PrepareKernel)
//! implementors, byte-parity-equivalent to their [`reference`](crate::reference)
//! counterparts (identical round polynomials and output claims — field
//! arithmetic is exact, so algebraically identical reorganizations are
//! byte-identical).

pub mod instruction_claim_reduction;
pub mod instruction_input;
pub mod instruction_ra_virtualization;
pub mod instruction_read_raf;
