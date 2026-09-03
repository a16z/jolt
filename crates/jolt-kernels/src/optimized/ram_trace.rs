//! Sparse per-cycle RAM addresses and values shared across RAM kernels.
//! [`SharedRamAddresses`] survives stages 2–6b; [`RamAccessColumns`] owns the
//! stage-2-only values.

use core::marker::PhantomData;
use std::sync::Arc;

use jolt_field::JoltField;
use jolt_poly::EqPolynomial;
#[cfg(feature = "parallel")]
use jolt_utils::FirstErrorLatch;
use jolt_witness::witnesses::{RamReadValue, RamWriteValue, RemappedRamAddress};
#[cfg(feature = "parallel")]
use jolt_witness::RandomAccessRows;
use jolt_witness::{stream_witnesses, JoltWitnessPlane, StreamConsumer, WitnessBundle};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::{KernelError, ProofSession};

/// Streaming extraction window.
const SPLIT_CHUNK: usize = 1 << 16;

/// Parallel scatter grain.
#[cfg(feature = "parallel")]
const PAR_CHUNK: usize = 1 << 14;

/// `addresses` sentinel for cycles with no (remappable) RAM access.
pub(crate) const NO_ACCESS: u32 = u32::MAX;

#[derive(Clone, Copy, Debug, WitnessBundle)]
struct RamValueBundle {
    pre_value: RamReadValue,
    post_value: RamWriteValue,
}

#[derive(Clone, Copy, Debug, WitnessBundle)]
struct RamAddressBundle {
    address: RemappedRamAddress,
}

/// Pack a remapped address. The column is `u32` because it outlives every
/// other trace-sized RAM array (stages 2–6b); the sparse matrices index it
/// as `u32` as well.
fn encode_address<F: JoltField>(address: Option<u64>) -> Result<u32, KernelError<F>> {
    let Some(address) = address else {
        return Ok(NO_ACCESS);
    };
    match u32::try_from(address) {
        Ok(address) if address != NO_ACCESS => Ok(address),
        _ => Err(KernelError::Unsupported {
            reason: "optimized RAM kernels pack remapped addresses as u32 below the u32::MAX \
                     no-access sentinel",
        }),
    }
}

/// Session-shared remapped addresses; [`NO_ACCESS`] marks absent accesses.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
pub(crate) struct SharedRamAddresses(pub(crate) Arc<Vec<u32>>);

impl SharedRamAddresses {
    /// Collect once, then return shared references.
    #[expect(
        clippy::expect_used,
        reason = "the entry is parked by this function right above the read"
    )]
    pub fn shared<F: JoltField>(
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        log_t: usize,
    ) -> Result<Arc<Vec<u32>>, KernelError<F>> {
        if session.state::<Self>().is_none() {
            let cycles = 1usize << log_t;
            let [addresses] =
                collect_split_columns::<F, RamAddressBundle, u32, 1>(witness, cycles, |bundle| {
                    encode_address(bundle.address.0).map(|address| [address])
                })?;
            session.park(Self(Arc::new(addresses)));
        }
        let addresses = Arc::clone(
            &session
                .state::<Self>()
                .expect("RAM address column parked above")
                .0,
        );
        // Reuse by type must not cross trace domains.
        if addresses.len() != 1usize << log_t {
            return Err(KernelError::TableSizeMismatch {
                table: "session-shared RAM address column".to_owned(),
                expected: 1usize << log_t,
                got: addresses.len(),
            });
        }
        Ok(addresses)
    }
}

/// Collect SoA columns by parallel scatter or ordered streaming.
fn collect_split_columns<F: JoltField, B, T, const N: usize>(
    witness: &dyn JoltWitnessPlane<F>,
    cycles: usize,
    split: impl Fn(&B) -> Result<[T; N], KernelError<F>> + Send + Sync,
) -> Result<[Vec<T>; N], KernelError<F>>
where
    B: WitnessBundle + Copy + Send + Sync,
    T: Copy + Send + Sync,
{
    #[cfg(feature = "parallel")]
    if let Some(access) = witness.random_access() {
        if cycles <= access.cycles() {
            return collect_split_columns_par(&access, cycles, &split);
        }
    }
    struct ColumnSplitter<F: JoltField, B, S, T, const N: usize> {
        columns: [Vec<T>; N],
        split: S,
        error: Option<KernelError<F>>,
        _bundle: PhantomData<B>,
    }
    impl<F: JoltField, B, S, T, const N: usize> StreamConsumer for ColumnSplitter<F, B, S, T, N>
    where
        B: WitnessBundle + Copy + Send + Sync,
        S: Fn(&B) -> Result<[T; N], KernelError<F>> + Send + Sync,
        T: Copy + Send + Sync,
    {
        type Witness = B;

        fn consume(&mut self, chunk: &[B]) {
            for bundle in chunk {
                match (self.split)(bundle) {
                    Ok(values) => {
                        for (column, value) in self.columns.iter_mut().zip(values) {
                            column.push(value);
                        }
                    }
                    Err(failure) => {
                        if self.error.is_none() {
                            self.error = Some(failure);
                        }
                    }
                }
            }
        }
    }
    let mut consumers = (ColumnSplitter {
        columns: core::array::from_fn(|_| Vec::with_capacity(cycles)),
        split,
        error: None,
        _bundle: PhantomData::<B>,
    },);
    stream_witnesses(witness, 0..cycles, SPLIT_CHUNK, &mut consumers)?;
    let ColumnSplitter { columns, error, .. } = consumers.0;
    if let Some(failure) = error {
        return Err(failure);
    }
    Ok(columns)
}

/// Scatter rows directly into column spare capacity.
#[cfg(feature = "parallel")]
fn collect_split_columns_par<F: JoltField, B, T, const N: usize>(
    access: &RandomAccessRows,
    cycles: usize,
    split: &(impl Fn(&B) -> Result<[T; N], KernelError<F>> + Sync),
) -> Result<[Vec<T>; N], KernelError<F>>
where
    B: WitnessBundle + Copy + Send + Sync,
    T: Copy + Send + Sync,
{
    use core::mem::MaybeUninit;
    let mut columns: [Vec<T>; N] = core::array::from_fn(|_| Vec::with_capacity(cycles));
    let chunk_count = cycles.div_ceil(PAR_CHUNK).max(1);
    let error = FirstErrorLatch::new();
    {
        let mut chunk_views: Vec<[&mut [MaybeUninit<T>]; N]> = Vec::with_capacity(chunk_count);
        let mut rests: [&mut [MaybeUninit<T>]; N] = columns
            .each_mut()
            .map(|column| &mut column.spare_capacity_mut()[..cycles]);
        for chunk_index in 0..chunk_count {
            let take = PAR_CHUNK.min(cycles - chunk_index * PAR_CHUNK);
            let mut views: [&mut [MaybeUninit<T>]; N] =
                core::array::from_fn(|_| Default::default());
            for (view, rest) in views.iter_mut().zip(rests.iter_mut()) {
                let (head, tail) = core::mem::take(rest).split_at_mut(take);
                *view = head;
                *rest = tail;
            }
            chunk_views.push(views);
        }
        chunk_views
            .into_par_iter()
            .enumerate()
            .for_each(|(chunk_index, mut views)| {
                let base = chunk_index * PAR_CHUNK;
                let take = PAR_CHUNK.min(cycles - base);
                for offset in 0..take {
                    let values = access
                        .window::<B>(base + offset)
                        .map_err(KernelError::from)
                        .and_then(|bundle| split(&bundle));
                    match values {
                        Ok(values) => {
                            for (view, value) in views.iter_mut().zip(values) {
                                let _ = view[offset].write(value);
                            }
                        }
                        Err(failure) => {
                            error.record(base + offset, failure);
                            return;
                        }
                    }
                }
            });
    }
    if let Some(failure) = error.take() {
        return Err(failure);
    }
    // SAFETY: successful chunks initialized every slot in each disjoint span.
    unsafe {
        for column in &mut columns {
            column.set_len(cycles);
        }
    }
    Ok(columns)
}

/// Stage-2 address and value columns.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
pub(crate) struct RamAccessColumns {
    pub addresses: Arc<Vec<u32>>,
    /// Pre-access value, or zero without an access.
    pub pre_values: Vec<u64>,
    /// Post-access word value per cycle (equals the pre-value for reads).
    pub post_values: Vec<u64>,
}

impl RamAccessColumns {
    /// Reuse or create the shared address column, then collect the values.
    ///
    /// Two bundle passes instead of one fused pass: the `u32` address column
    /// outlives the `u64` values by four stages, so it is collected on its
    /// own. Stage 4 calls this again rather than parking the values from
    /// stage 2 — re-reading the trace is cheaper than keeping `16·T` bytes
    /// resident across stages 2–4.
    pub fn collect_full<F: JoltField>(
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        log_t: usize,
    ) -> Result<Self, KernelError<F>> {
        let cycles = 1usize << log_t;
        let addresses = SharedRamAddresses::shared(session, witness, log_t)?;
        let [pre_values, post_values] =
            collect_split_columns::<F, RamValueBundle, u64, 2>(witness, cycles, |bundle| {
                Ok([bundle.pre_value.0, bundle.post_value.0])
            })?;
        Ok(Self {
            addresses,
            pre_values,
            post_values,
        })
    }

    /// Recover initial values from each address's first access; untouched
    /// addresses retain their final value.
    ///
    /// WARNING: RAM sumchecks enforce the assumed trace/final-image agreement.
    pub fn reconstruct_val_init<F: JoltField>(&self, val_final: Vec<F>) -> Vec<F> {
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

/// Reject addresses outside the proof's RAM domain.
pub(crate) fn validate_addresses<F: JoltField>(
    addresses: &[u32],
    ram_k: usize,
) -> Result<(), KernelError<F>> {
    if addresses
        .iter()
        .any(|&address| address != NO_ACCESS && address as usize >= ram_k)
    {
        return Err(KernelError::InvariantViolation {
            reason: "RAM access address remapped beyond ram_K",
        });
    }
    Ok(())
}

/// Address fold: `out[j] = eq_address[addresses[j]]`, or zero without access.
pub(crate) fn fold_addresses<F: JoltField>(addresses: &[u32], eq_address: &[F]) -> Vec<F> {
    addresses
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

/// Eq rows staged per cycle-fold chunk.
const FOLD_CYCLES_CHUNK: usize = 1 << 20;

/// Cycle fold of the one-hot `ra` grid:
/// `out[k] = Σ_j eq(r_cycle, j) · ra(k, j) = Σ_{j : addresses[j] = k} eq_cycle[j]`.
/// Eq values are generated in chunks from `e_hi ⊗ e_lo`.
pub(crate) fn fold_cycles<F: JoltField>(addresses: &[u32], r_cycle: &[F], ram_k: usize) -> Vec<F> {
    let mid = r_cycle.len() / 2;
    let (r_hi, r_lo) = r_cycle.split_at(mid);
    let e_hi = EqPolynomial::<F>::evals(r_hi, None);
    let e_lo = EqPolynomial::<F>::evals(r_lo, None);
    let lo_bits = r_lo.len();
    let lo_mask = e_lo.len() - 1;

    let mut out = vec![F::zero(); ram_k];
    let mut staging: Vec<F> = vec![F::zero(); FOLD_CYCLES_CHUNK.min(addresses.len())];
    for (chunk_index, chunk) in addresses.chunks(FOLD_CYCLES_CHUNK).enumerate() {
        let base = chunk_index * FOLD_CYCLES_CHUNK;
        let staging = &mut staging[..chunk.len()];
        let fill = |(offset, slot): (usize, &mut F)| {
            let j = base + offset;
            *slot = e_hi[j >> lo_bits] * e_lo[j & lo_mask];
        };
        #[cfg(feature = "parallel")]
        staging
            .par_iter_mut()
            .enumerate()
            .with_min_len(1 << 10)
            .for_each(fill);
        #[cfg(not(feature = "parallel"))]
        staging.iter_mut().enumerate().for_each(fill);

        for (&address, &eq) in chunk.iter().zip(staging.iter()) {
            if address != NO_ACCESS {
                out[address as usize] += eq;
            }
        }
    }
    out
}

#[cfg(test)]
#[expect(clippy::panic, reason = "test module")]
mod tests {
    use jolt_field::Fr;
    use jolt_witness::testing::with_sample_backend;

    use super::*;

    #[test]
    fn rejects_session_carry_from_another_cycle_domain() {
        with_sample_backend(|witness| {
            let mut session = ProofSession::default();
            session.park(SharedRamAddresses(Arc::new(vec![NO_ACCESS; 2])));

            let error = match SharedRamAddresses::shared::<Fr>(&mut session, witness, 2) {
                Ok(_) => panic!("wrong-domain RAM addresses were accepted"),
                Err(error) => error,
            };
            assert!(matches!(
                error,
                KernelError::TableSizeMismatch {
                    expected: 4,
                    got: 2,
                    ..
                }
            ));
        });
    }
}
