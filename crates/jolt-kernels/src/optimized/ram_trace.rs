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
use jolt_witness::{stream_witnesses, JoltWitnessPlane, StreamConsumer, WitnessBundle};
#[cfg(feature = "parallel")]
use jolt_witness::{RandomAccessRows, WitnessError};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::{KernelError, ProofSession};

/// `addresses` sentinel for cycles with no (remappable) RAM access.
pub(crate) const NO_ACCESS: u32 = u32::MAX;

#[derive(Clone, Copy, Debug, WitnessBundle)]
struct RamAccessBundle {
    address: RemappedRamAddress,
    pre_value: RamReadValue,
    post_value: RamWriteValue,
}

#[derive(Clone, Copy)]
enum AddressEncodingError {
    TooLarge,
    SentinelCollision,
}

impl AddressEncodingError {
    fn into_kernel_error<F: Field>(self) -> KernelError<F> {
        let reason = match self {
            Self::TooLarge => "optimized RAM kernels require remapped addresses below 2^32 - 1",
            Self::SentinelCollision => {
                "optimized RAM kernels reserve u32::MAX as the no-access sentinel"
            }
        };
        KernelError::Unsupported { reason }
    }
}

fn encode_address(address: Option<u64>) -> Result<u32, AddressEncodingError> {
    let Some(address) = address else {
        return Ok(NO_ACCESS);
    };
    let address = u32::try_from(address).map_err(|_| AddressEncodingError::TooLarge)?;
    if address == NO_ACCESS {
        return Err(AddressEncodingError::SentinelCollision);
    }
    Ok(address)
}

struct CollectRamAccessColumns {
    addresses: Vec<u32>,
    pre_values: Vec<u64>,
    post_values: Vec<u64>,
    address_error: Option<AddressEncodingError>,
}

impl StreamConsumer for CollectRamAccessColumns {
    type Witness = RamAccessBundle;

    fn consume(&mut self, chunk: &[RamAccessBundle]) {
        for bundle in chunk {
            let address = encode_address(bundle.address.0).unwrap_or_else(|failure| {
                let _ = self.address_error.get_or_insert(failure);
                NO_ACCESS
            });
            self.addresses.push(address);
            self.pre_values.push(bundle.pre_value.0);
            self.post_values.push(bundle.post_value.0);
        }
    }
}

#[cfg(feature = "parallel")]
enum CollectFailure {
    Witness(WitnessError),
    Address(AddressEncodingError),
}

/// Column-major per-cycle RAM access data over the full padded cycle domain.
pub(crate) struct RamAccessColumns {
    /// Remapped word address per cycle; [`NO_ACCESS`] when the cycle makes no
    /// remappable RAM access (no-ops and address 0).
    pub addresses: Vec<u32>,
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for RamAccessColumns {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        visitor.visit_simple(
            allocative::Key::new("addresses"),
            crate::backend::vec_heap_bytes(&self.addresses),
        );
        visitor.exit();
    }
}

/// RAM values have one final consumer in stage 4, so they are parked
/// separately from the address column and consumed there.
pub(crate) struct RamAccessValues {
    /// Pre-access word value per cycle (a read's value, a write's pre-value);
    /// 0 on no-access cycles.
    pub pre_values: Vec<u64>,
    /// Post-access word value per cycle (equals the pre-value for reads).
    pub post_values: Vec<u64>,
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for RamAccessValues {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        visitor.visit_simple(
            allocative::Key::new("pre_values"),
            crate::backend::vec_heap_bytes(&self.pre_values),
        );
        visitor.visit_simple(
            allocative::Key::new("post_values"),
            crate::backend::vec_heap_bytes(&self.post_values),
        );
        visitor.exit();
    }
}

impl RamAccessColumns {
    fn collect<F: Field>(
        witness: &dyn JoltWitnessPlane<F>,
        log_t: usize,
    ) -> Result<(Self, RamAccessValues), KernelError<F>> {
        let cycles = 1usize << log_t;
        #[cfg(feature = "parallel")]
        if let Some(access) = witness.random_access() {
            if cycles <= access.cycles() {
                return Self::collect_par(&access, cycles);
            }
        }

        const COLLECT_CHUNK: usize = 1 << 16;
        let mut consumers = (CollectRamAccessColumns {
            addresses: Vec::with_capacity(cycles),
            pre_values: Vec::with_capacity(cycles),
            post_values: Vec::with_capacity(cycles),
            address_error: None,
        },);
        stream_witnesses(witness, 0..cycles, COLLECT_CHUNK, &mut consumers)?;
        let collected = consumers.0;
        if let Some(failure) = collected.address_error {
            return Err(failure.into_kernel_error());
        }
        Ok((
            Self {
                addresses: collected.addresses,
            },
            RamAccessValues {
                pre_values: collected.pre_values,
                post_values: collected.post_values,
            },
        ))
    }

    /// Slice-backed traces scatter directly into the three final columns,
    /// avoiding a full-width `RamAccessBundle` vector at the collection peak.
    #[cfg(feature = "parallel")]
    fn collect_par<F: Field>(
        access: &RandomAccessRows,
        cycles: usize,
    ) -> Result<(Self, RamAccessValues), KernelError<F>> {
        const CHUNK: usize = 1 << 12;
        let mut addresses = Vec::with_capacity(cycles);
        let mut pre_values = Vec::with_capacity(cycles);
        let mut post_values = Vec::with_capacity(cycles);
        let error = std::sync::Mutex::new(None);
        (
            addresses.spare_capacity_mut()[..cycles].par_chunks_mut(CHUNK),
            pre_values.spare_capacity_mut()[..cycles].par_chunks_mut(CHUNK),
            post_values.spare_capacity_mut()[..cycles].par_chunks_mut(CHUNK),
        )
            .into_par_iter()
            .enumerate()
            .for_each(|(chunk_index, (addresses, pre_values, post_values))| {
                let base = chunk_index * CHUNK;
                for offset in 0..addresses.len() {
                    let bundle = match access.window::<RamAccessBundle>(base + offset) {
                        Ok(bundle) => bundle,
                        Err(failure) => {
                            if let Ok(mut guard) = error.try_lock() {
                                let _ = guard.get_or_insert(CollectFailure::Witness(failure));
                            }
                            return;
                        }
                    };
                    let address = match encode_address(bundle.address.0) {
                        Ok(address) => address,
                        Err(failure) => {
                            if let Ok(mut guard) = error.try_lock() {
                                let _ = guard.get_or_insert(CollectFailure::Address(failure));
                            }
                            return;
                        }
                    };
                    let _ = addresses[offset].write(address);
                    let _ = pre_values[offset].write(bundle.pre_value.0);
                    let _ = post_values[offset].write(bundle.post_value.0);
                }
            });
        #[expect(clippy::unwrap_used, reason = "no lock user can panic")]
        if let Some(failure) = error.into_inner().unwrap() {
            return Err(match failure {
                CollectFailure::Witness(failure) => failure.into(),
                CollectFailure::Address(failure) => failure.into_kernel_error(),
            });
        }
        // SAFETY: with no latched error, every worker initialized its entire
        // disjoint span in all three vectors.
        unsafe {
            addresses.set_len(cycles);
            pre_values.set_len(cycles);
            post_values.set_len(cycles);
        }
        Ok((
            Self { addresses },
            RamAccessValues {
                pre_values,
                post_values,
            },
        ))
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
            let (columns, values) = Self::collect(witness, log_t)?;
            let columns = Arc::new(columns);
            session.park(columns);
            session.park(values);
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

    /// Reclaims the value columns at their final consumer while leaving the
    /// shared address column available to later stages.
    pub fn shared_with_values<F: Field>(
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        log_t: usize,
    ) -> Result<(Arc<Self>, RamAccessValues), KernelError<F>> {
        let columns = Self::shared(session, witness, log_t)?;
        let values = session
            .take::<RamAccessValues>()
            .ok_or(KernelError::InvariantViolation {
                reason: "RAM access value columns were already consumed",
            })?;
        Ok((columns, values))
    }

    /// Bounds-check every accessed address against the proof's `K`, matching
    /// the grid materializers' fail-loud contract.
    pub fn validate_addresses<F: Field>(&self, ram_k: usize) -> Result<(), KernelError<F>> {
        if self
            .addresses
            .iter()
            .any(|&address| address != NO_ACCESS && address as usize >= ram_k)
        {
            return Err(KernelError::InvariantViolation {
                reason: "RAM access address remapped beyond ram_K",
            });
        }
        Ok(())
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
    pub fn reconstruct_val_init<F: Field>(&self, pre_values: &[u64], val_final: Vec<F>) -> Vec<F> {
        debug_assert_eq!(self.addresses.len(), pre_values.len());
        let mut val_init = val_final;
        let mut seen = vec![false; val_init.len()];
        for (&address, &pre_value) in self.addresses.iter().zip(pre_values) {
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
