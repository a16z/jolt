//! Lazy polynomial data access from an execution trace.
//!
//! [`TracePolynomials`] provides on-demand access to virtual polynomial
//! evaluation data backed by a `CycleRow` trace. Nothing is pre-materialized
//! until the consumer requests it.
//!
//! This is the sole interface between the proving layer (jolt-zkvm) and
//! trace-derived polynomial data. The prover never touches `CycleRow`
//! directly — all trace knowledge is encapsulated here.

use jolt_field::Field;
use jolt_host::CycleRow;

/// Lazy polynomial data backed by a `CycleRow` trace.
///
/// Provides access to sparse read-write entries for RAM and register
/// checking evaluators, backed by an in-memory `CycleRow` trace.
pub struct TracePolynomials<'a, R> {
    trace: &'a [R],
    padded_len: usize,
}

impl<'a, R: CycleRow> TracePolynomials<'a, R> {
    pub fn new(trace: &'a [R]) -> Self {
        let padded_len = trace.len().next_power_of_two();
        Self { trace, padded_len }
    }

    /// Padded trace length (next power of 2).
    pub fn padded_len(&self) -> usize {
        self.padded_len
    }

    /// Raw trace length (before padding).
    pub fn trace_len(&self) -> usize {
        self.trace.len()
    }

    /// Build sparse RAM read-write entries for `SparseRwEvaluator`.
    ///
    /// Each cycle with a RAM access produces one entry sorted by cycle index.
    pub fn ram_entries<F: Field>(&self) -> Vec<crate::RwEntry<F>> {
        let mut entries = Vec::new();
        for (j, cycle) in self.trace.iter().enumerate() {
            if let Some(addr) = cycle.ram_access_address() {
                let read_val = cycle.ram_read_value().unwrap_or(0);
                let write_val = cycle.ram_write_value().unwrap_or(0);
                entries.push(crate::RwEntry {
                    bind_pos: j,
                    free_pos: addr as usize,
                    ra: F::one(),
                    val: F::from_u64(read_val),
                    prev_val: F::from_u64(read_val),
                    next_val: F::from_u64(write_val),
                });
            }
        }
        entries
    }

    /// Build sparse register read-write entries for `SparseRwEvaluator`.
    pub fn register_entries<F: Field>(&self) -> Vec<crate::RwEntry<F>> {
        let mut entries = Vec::new();
        for (j, cycle) in self.trace.iter().enumerate() {
            // RS1 read
            if let Some((idx, val)) = cycle.rs1_read() {
                entries.push(crate::RwEntry {
                    bind_pos: j,
                    free_pos: idx as usize,
                    ra: F::one(),
                    val: F::from_u64(val),
                    prev_val: F::from_u64(val),
                    next_val: F::from_u64(val),
                });
            }
            // RS2 read
            if let Some((idx, val)) = cycle.rs2_read() {
                entries.push(crate::RwEntry {
                    bind_pos: j,
                    free_pos: idx as usize,
                    ra: F::one(),
                    val: F::from_u64(val),
                    prev_val: F::from_u64(val),
                    next_val: F::from_u64(val),
                });
            }
            // RD write
            if let Some((idx, pre, post)) = cycle.rd_write() {
                entries.push(crate::RwEntry {
                    bind_pos: j,
                    free_pos: idx as usize,
                    ra: F::one(),
                    val: F::from_u64(pre),
                    prev_val: F::from_u64(pre),
                    next_val: F::from_u64(post),
                });
            }
        }
        // Sort by (bind_pos, free_pos) for sparse evaluator
        entries.sort_by_key(|e| (e.bind_pos, e.free_pos));
        entries
    }
}

/// Sparse matrix entry for read-write checking evaluators.
///
/// Re-exported from jolt-witness for use by jolt-zkvm's evaluators.
#[derive(Clone, Debug)]
pub struct RwEntry<F> {
    pub bind_pos: usize,
    pub free_pos: usize,
    pub ra: F,
    pub val: F,
    pub prev_val: F,
    pub next_val: F,
}

#[cfg(test)]
mod tests {
    // Tests require a concrete CycleRow impl (from jolt-host).
    // Integration tests in jolt-zkvm exercise the full path.
}
