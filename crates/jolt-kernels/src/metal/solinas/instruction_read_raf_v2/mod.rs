//! Backend-neutral slice-A model for the InstructionReadRaf Metal v2 design.
//!
//! This directory is not wired into the production backend. The design packet
//! maps to code as follows:
//!
//! - producer-owned facts and the stable 82-segment topology: [`carrier`];
//! - six-lane suffix mapping, traffic, work, and occupancy arithmetic: [`model`];
//! - host transcript, phase, handoff, and output ordering: [`protocol`];
//! - scalar cycle-order phase and address-polynomial oracle: [`oracle`].
//!
//! The producer adapter remains unresolved. It must attach these receipts to
//! the existing `BooleanityRows` allocation and any compact planes without a
//! member-local upload or repack.

pub mod carrier;
pub mod model;
pub mod oracle;
pub mod protocol;
