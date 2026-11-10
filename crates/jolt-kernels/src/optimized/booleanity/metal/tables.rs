//! Bounded uploads of the CPU's canonical categorical and paired tables.
use std::mem::{size_of, MaybeUninit};

use super::*;
use crate::optimized::lazy_ra::PairedBranchTables;

// This bounds host plus device tables, their construction temporary, the
// existing branch tables, and caller-supplied dispatch buffers. There is no
// flattened host upload mirror and no new trace-row storage.
const MAX_TABLE_PEAK_BYTES: usize = 128 << 20;

pub(super) struct TableBudget {
    pub(super) auxiliary_bytes: usize,
    pub(super) policy_cycles: usize,
    pub(super) weighted_copies: usize,
}

pub(super) struct CachedTables {
    pub(super) values: DeviceBuffer<AkitaField>,
    pub(super) squares: Option<DeviceBuffer<AkitaField>>,
    pub(super) mode: u32,
    pub(super) states: u32,
    pub(super) square_stride: u32,
    pub(super) device_bytes: usize,
    pub(super) peak_bytes: usize,
}

#[derive(Clone, Copy)]
struct Shape {
    mode: u32,
    states: usize,
    square_stride: usize,
}

impl CachedTables {
    pub(super) fn prepare(
        device: &Device,
        tables: &[Vec<AkitaField>],
        width: usize,
        source: &BooleanityChunks,
        rho: &[AkitaField],
        budget: TableBudget,
    ) -> Result<Option<Self>, MetalError> {
        type Full<'a> = CategoricalProducts<'a, AkitaField, BooleanityChunks, false>;
        type Triangular<'a> = CategoricalProducts<'a, AkitaField, BooleanityChunks, true>;
        let Some(first) = tables.first() else {
            return Ok(None);
        };
        let Some(addresses) = first.len().checked_div(width) else {
            return Ok(None);
        };
        let active = rho.iter().take_while(|rho| !rho.is_zero()).count();
        if active == 0 || active > tables.len() {
            return Ok(None);
        }
        let shape = if Full::fits_budget(width, addresses, budget.policy_cycles) {
            let states = (addresses + 1).pow(width as u32);
            Shape {
                mode: 1,
                states,
                square_stride: Full::square_entries(states),
            }
        } else if Triangular::fits_budget(width, addresses, budget.policy_cycles) {
            let states = (addresses + 1).pow(width as u32);
            Shape {
                mode: 2,
                states,
                square_stride: Triangular::square_entries(states),
            }
        } else if PairedBranchTables::fits_budget(
            &tables[..active],
            width,
            source.constant_suffix_start(),
        ) {
            Shape {
                mode: 3,
                states: PairedBranchTables::<AkitaField>::entries(width),
                square_stride: 0,
            }
        } else {
            return Ok(None);
        };
        let values_len = active.checked_mul(shape.states);
        let squares_len = active.checked_mul(shape.square_stride);
        let Some((values_len, squares_len)) = values_len.zip(squares_len) else {
            return Ok(None);
        };
        if u32::try_from(values_len).is_err() || u32::try_from(squares_len).is_err() {
            return Ok(None);
        }
        let Some(device_bytes) = values_len
            .checked_add(squares_len)
            .and_then(|len| len.checked_mul(size_of::<AkitaField>()))
        else {
            return Ok(None);
        };
        // Current CPU owner allocates one values temporary at a time. Count
        // one per active column anyway, plus both Vec headers per column.
        let temporary = active
            .checked_mul(shape.states)
            .and_then(|n| n.checked_mul(size_of::<AkitaField>()));
        let headers = active.checked_mul(2 * size_of::<Vec<AkitaField>>());
        let branches = tables.iter().try_fold(0usize, |sum, table| {
            table
                .len()
                .checked_mul(size_of::<AkitaField>())
                .and_then(|bytes| bytes.checked_add(size_of::<Vec<AkitaField>>()))
                .and_then(|bytes| sum.checked_add(bytes))
        });
        let Some(copies) = 2usize.checked_add(budget.weighted_copies) else {
            return Ok(None);
        };
        let Some(peak_bytes) = device_bytes
            .checked_mul(copies)
            .and_then(|bytes| bytes.checked_add(temporary?))
            .and_then(|bytes| bytes.checked_add(headers?))
            .and_then(|bytes| bytes.checked_add(branches?))
            .and_then(|bytes| bytes.checked_add(budget.auxiliary_bytes))
        else {
            return Ok(None);
        };
        let Some(weighted_bytes) = device_bytes.checked_mul(budget.weighted_copies) else {
            return Ok(None);
        };
        if peak_bytes > MAX_TABLE_PEAK_BYTES
            || device_bytes > device.limits().max_buffer_length
            || weighted_bytes > device.limits().max_buffer_length
        {
            return Ok(None);
        }
        match device.check_additional_peak(peak_bytes) {
            Ok(()) => (),
            Err(MetalError::CapacityExceeded { .. }) => return Ok(None),
            Err(error) => return Err(error),
        }
        // No expected decline occurs after construction starts. Allocation,
        // initialization, and later command errors abort the computation.
        let (values, squares) = match shape.mode {
            1 => Self::categorical::<false>(
                device,
                tables,
                width,
                source,
                rho,
                values_len,
                squares_len,
            )?,
            2 => Self::categorical::<true>(
                device,
                tables,
                width,
                source,
                rho,
                values_len,
                squares_len,
            )?,
            _ => {
                let paired = PairedBranchTables::new(&tables[..active]);
                (upload(device, values_len, paired.column_tables())?, None)
            }
        };
        Ok(Some(Self {
            values,
            squares,
            mode: shape.mode,
            states: u32::try_from(shape.states).map_err(|_| cache_error("state count"))?,
            square_stride: u32::try_from(shape.square_stride)
                .map_err(|_| cache_error("square stride"))?,
            device_bytes,
            peak_bytes,
        }))
    }

    fn categorical<const TRIANGULAR: bool>(
        device: &Device,
        tables: &[Vec<AkitaField>],
        width: usize,
        source: &BooleanityChunks,
        rho: &[AkitaField],
        values_len: usize,
        squares_len: usize,
    ) -> Result<(DeviceBuffer<AkitaField>, Option<DeviceBuffer<AkitaField>>), MetalError> {
        let products = CategoricalProducts::<_, _, TRIANGULAR>::new(tables, width, source, rho);
        let values = upload(
            device,
            values_len,
            products.column_tables().map(|(values, _)| values),
        )?;
        let squares = upload(
            device,
            squares_len,
            products.column_tables().map(|(_, squares)| squares),
        )?;
        Ok((values, Some(squares)))
    }
}

fn cache_error(reason: &str) -> MetalError {
    MetalError::InvalidDispatch {
        pipeline: "lazy_booleanity_cached".into(),
        reason: reason.into(),
    }
}

/// Copy each canonical column directly into its final shared allocation.
fn upload<'a>(
    device: &Device,
    len: usize,
    columns: impl Iterator<Item = &'a [AkitaField]>,
) -> Result<DeviceBuffer<AkitaField>, MetalError> {
    let mut buffer = DeviceBuffer::<MaybeUninit<AkitaField>>::uninit(device, len)?;
    let mut output = buffer.write_uninit()?.iter_mut();
    for column in columns {
        for value in column {
            let slot = output
                .next()
                .ok_or_else(|| cache_error("canonical table exceeds extent"))?;
            let _ = slot.write(*value);
        }
    }
    if output.next().is_some() {
        return Err(cache_error("canonical table does not fill extent"));
    }
    // SAFETY: every logical slot was written above, with exact length checked.
    unsafe { buffer.assume_init() }
}

/// Canonical low/high equality factors for sixteen adjacent message pairs.
pub(super) struct EqualityBlock<'a> {
    pub(super) high: &'a [AkitaField],
    active: Vec<AkitaField>,
}

impl<'a> EqualityBlock<'a> {
    pub(super) fn prepare(
        eq: &'a GruenSplitEqPolynomial<AkitaField>,
        width: usize,
        radix: usize,
        columns: usize,
    ) -> Result<Option<Self>, MetalError> {
        if !matches!(width, 1 | 4 | 8)
            || radix != 16
            || !(1..=128).contains(&columns)
            || eq.e_in_current_len() < 16
        {
            return Ok(None);
        }
        let (outer, high) = eq.e_out_in_for_window(5);
        let active = eq.e_active_for_window(5);
        if high.len().checked_mul(16) != Some(eq.e_in_current_len())
            || active.len() != 16
            || outer != eq.e_out_current()
        {
            return Err(cache_error(
                "canonical window does not factor current inner equality",
            ));
        }
        Ok(Some(Self { high, active }))
    }
}

/// GPU-written weighted copies; no host field view is exposed before or after dispatch.
pub(super) struct WeightedTables {
    pub(super) values: DeviceBuffer<MaybeUninit<AkitaField>>,
    active: DeviceBuffer<AkitaField>,
    params: [u32; 3],
}

impl WeightedTables {
    pub(super) fn new(
        device: &Device,
        cache: &CachedTables,
        block: &EqualityBlock<'_>,
    ) -> Result<Self, MetalError> {
        let values = cache.values.len();
        let squares = cache.squares.as_ref().map_or(0, DeviceBuffer::len);
        let len = values
            .checked_add(squares)
            .and_then(|n| n.checked_mul(block.active.len()))
            .filter(|n| u32::try_from(*n).is_ok())
            .ok_or_else(|| cache_error("weighted table exceeds 32-bit indexing"))?;
        Ok(Self {
            values: DeviceBuffer::uninit(device, len)?,
            active: DeviceBuffer::from_slice(device, &block.active)?,
            params: [values as u32, squares as u32, block.active.len() as u32],
        })
    }

    pub(super) fn bytes(&self) -> usize {
        self.values.len() * size_of::<AkitaField>()
    }

    pub(super) fn encode<'a>(
        &'a self,
        cache: &'a CachedTables,
        batch: &mut Batch<'a>,
        pipeline: &Pipeline,
        group: usize,
    ) -> Result<(), MetalError> {
        // SAFETY: the rounded grid uniquely writes every weighted index; the
        // checked table lengths and sixteen canonical factors cover all reads.
        // The following message dispatch uses the same serial, tracked batch.
        unsafe {
            batch.dispatch_unchecked(
                pipeline,
                &[
                    Binding::buffer(&cache.values),
                    Binding::buffer(cache.squares.as_ref().unwrap_or(&cache.values)),
                    Binding::buffer(&self.active),
                    Binding::buffer(&self.values),
                    Binding::value(&self.params),
                ],
                Grid::linear(self.values.len().div_ceil(group) * group, group),
            )?;
        }
        Ok(())
    }
}
