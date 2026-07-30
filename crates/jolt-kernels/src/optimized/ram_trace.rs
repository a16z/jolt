//! The shared RAM access columns: one typed trace walk serving every
//! RAM-family kernel in this backend, parked in the [`ProofSession`] so the
//! stage-2 kernel's walk is reused by stages 4 and 5.
//!
//! The columns are the sparse view of the `(K × T)` RAM grids: per cycle,
//! the remapped word address (or a no-access sentinel) plus the pre- and
//! post-access word values. `ra(k, j)` is 1 exactly at `(addresses[j], j)`;
//! `val(k, j)` walks from the initial state through the writes.

use std::sync::Arc;

use jolt_field::Field;
use jolt_witness::witnesses::{RamReadValue, RamWriteValue, RemappedRamAddress};
use jolt_witness::{JoltWitnessPlane, WitnessBundle};

use super::support::collect_rows;
use crate::{KernelError, ProofSession};

/// `addresses` sentinel for cycles with no (remappable) RAM access.
pub(crate) const NO_ACCESS: u64 = u64::MAX;

#[derive(Clone, Copy, Debug, WitnessBundle)]
struct RamAccessBundle {
    address: RemappedRamAddress,
    pre_value: RamReadValue,
    post_value: RamWriteValue,
}

/// Column-major per-cycle RAM access data over the full padded cycle domain.
pub(crate) struct RamAccessColumns {
    /// Remapped word address per cycle; [`NO_ACCESS`] when the cycle makes no
    /// remappable RAM access (no-ops and address 0).
    pub addresses: Vec<u64>,
    /// Pre-access word value per cycle (a read's value, a write's pre-value);
    /// 0 on no-access cycles.
    pub pre_values: Vec<u64>,
    /// Post-access word value per cycle (equals the pre-value for reads).
    pub post_values: Vec<u64>,
}

impl RamAccessColumns {
    fn collect<F: Field>(
        witness: &dyn JoltWitnessPlane<F>,
        log_t: usize,
    ) -> Result<Self, KernelError<F>> {
        let cycles = 1usize << log_t;
        let bundles: Vec<RamAccessBundle> = collect_rows(witness, cycles)?;
        let mut addresses = Vec::with_capacity(cycles);
        let mut pre_values = Vec::with_capacity(cycles);
        let mut post_values = Vec::with_capacity(cycles);
        for bundle in bundles {
            addresses.push(bundle.address.0.unwrap_or(NO_ACCESS));
            pre_values.push(bundle.pre_value.0);
            post_values.push(bundle.post_value.0);
        }
        Ok(Self {
            addresses,
            pre_values,
            post_values,
        })
    }

    /// The session-shared columns: collected on first request (whichever RAM
    /// kernel prepares first), cloned out as an [`Arc`] afterwards.
    #[expect(
        clippy::expect_used,
        reason = "the entry is parked by this function right above the read"
    )]
    pub fn shared<F: Field>(
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        log_t: usize,
    ) -> Result<Arc<Self>, KernelError<F>> {
        if session.state::<Arc<Self>>().is_none() {
            let columns = Arc::new(Self::collect(witness, log_t)?);
            session.park(columns);
        }
        let columns = Arc::clone(
            session
                .state::<Arc<Self>>()
                .expect("RAM access columns parked above"),
        );
        debug_assert_eq!(
            columns.addresses.len(),
            1usize << log_t,
            "parked RAM access columns cover a different cycle domain than requested"
        );
        Ok(columns)
    }

    /// Bounds-check every accessed address against the proof's `K`, matching
    /// the grid materializers' fail-loud contract.
    pub fn validate_addresses<F: Field>(&self, ram_k: usize) -> Result<(), KernelError<F>> {
        if self
            .addresses
            .iter()
            .any(|&address| address != NO_ACCESS && address >= ram_k as u64)
        {
            return Err(KernelError::InvariantViolation {
                reason: "RAM access address remapped beyond ram_K",
            });
        }
        Ok(())
    }

    /// The `RamInc` column: `post − pre` per cycle — exactly the
    /// `RamInc` extractor's value (reads and no-ops cancel to 0; the raw
    /// pre/post values are remap-independent, matching the oracle walk on
    /// every cycle including unremappable-address writes).
    pub fn inc_column<F: Field>(&self) -> Vec<F> {
        let inc = |(&post, &pre): (&u64, &u64)| F::from_i128(i128::from(post) - i128::from(pre));
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            self.post_values
                .par_iter()
                .zip(&self.pre_values)
                .map(inc)
                .collect()
        }
        #[cfg(not(feature = "parallel"))]
        self.post_values
            .iter()
            .zip(&self.pre_values)
            .map(inc)
            .collect()
    }

    /// The address-eq fold of the one-hot `ra` grid:
    /// `out[j] = Σ_k eq(r_address, k) · ra(k, j) = eq_address[addresses[j]]`
    /// (0 on no-access cycles). Reproduces `views::address_fold` of the dense
    /// grid without materializing it.
    pub fn fold_addresses<F: Field>(&self, eq_address: &[F]) -> Vec<F> {
        self.addresses
            .iter()
            .map(|&address| {
                if address == NO_ACCESS {
                    F::zero()
                } else {
                    eq_address[address as usize]
                }
            })
            .collect()
    }

    /// The cycle-eq fold of the one-hot `ra` grid:
    /// `out[k] = Σ_j eq(r_cycle, j) · ra(k, j) = Σ_{j : addresses[j] = k} eq_cycle[j]`.
    /// Reproduces `views::cycle_fold` of the dense grid without
    /// materializing it.
    pub fn fold_cycles<F: Field>(&self, eq_cycle: &[F], ram_k: usize) -> Vec<F> {
        let mut out = vec![F::zero(); ram_k];
        for (&address, &eq) in self.addresses.iter().zip(eq_cycle) {
            if address != NO_ACCESS {
                out[address as usize] += eq;
            }
        }
        out
    }

    /// Reconstruct the initial RAM state from the trace and the final-state
    /// oracle: an accessed address's initial value is its first access's
    /// pre-value; a never-accessed address's value never changes, so its
    /// final value IS its initial value.
    ///
    /// WARNING: this is the honest-prover data path — it relies on the trace
    /// being consistent with the final memory image (exactly what the RAM
    /// val/output sumchecks prove). A dishonest witness diverges here and
    /// fails the engine's round checks loudly.
    pub fn reconstruct_val_init<F: Field>(&self, val_final: Vec<F>) -> Vec<F> {
        let mut val_init = val_final;
        let mut seen = vec![false; val_init.len()];
        for (&address, &pre_value) in self.addresses.iter().zip(&self.pre_values) {
            if address == NO_ACCESS {
                continue;
            }
            let address = address as usize;
            if !seen[address] {
                seen[address] = true;
                val_init[address] = F::from_u64(pre_value);
            }
        }
        val_init
    }
}
