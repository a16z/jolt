//! CUDA Spartan outer input packing.
//!
//! CUDA gathers native values into narrow lanes, wide lanes with separate sign
//! bits, and packed flags; optimized keeps typed rows. A device shift adds
//! successor PC and flag columns, with zero values beyond the final row.

mod columns;
mod witness;
