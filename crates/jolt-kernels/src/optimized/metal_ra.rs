//! Device-owned row-major prefixes shared by lazy selector consumers.
use std::mem::MaybeUninit;

use jolt_akita::AkitaField as F;
use jolt_metal::runtime::{
    host_name, Batch, Binding, Device, DeviceBuffer, Grid, LibrarySpec, ShaderLibrary,
};
use jolt_metal::shaders::FIELD_HEADERS;
use jolt_metal::MetalError;

use super::lazy_ra::{ChunkIndexSource, LazyFoldedRa};
use super::metal_rows::UploadedRows;

pub(super) const CPU_TAIL_ROWS: usize = 1 << 12;

pub(super) struct MetalRaFold {
    device: Device,
    library: ShaderLibrary,
}

#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
pub(super) struct DeviceRa {
    pub(super) prefix: usize,
    pub(super) tails: Vec<F>,
    #[cfg_attr(feature = "allocative", allocative(skip))]
    pub(super) values: DeviceBuffer<F>,
}

impl DeviceRa {
    pub(super) fn restore<S: ChunkIndexSource>(mut self) -> Result<LazyFoldedRa<F, S>, MetalError> {
        Ok(LazyFoldedRa::Rows {
            values: self.values.read()?.to_vec(),
            tail: self.tails,
        })
    }
}

impl MetalRaFold {
    pub(super) fn new() -> Result<Self, MetalError> {
        let device = Device::system_default()?;
        let mut spec = LibrarySpec::new();
        for (name, source) in FIELD_HEADERS {
            spec = spec.source(name, source);
        }
        let spec = spec
            .source("ra_fold.metal", include_str!("ra_fold.metal"))
            .instantiate::<F>("ra_materialize")
            .instantiate::<F>("ra_fold");
        let library = ShaderLibrary::compile(&device, &spec)?;
        Ok(Self { device, library })
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "validated lazy selector view and next bind"
    )]
    pub(super) fn materialize(
        &self,
        rows: &UploadedRows,
        tables: &[Vec<F>],
        width: usize,
        selectors: &[[u32; 2]],
        prefix: usize,
        tails: &[F],
        challenge: F,
    ) -> Result<DeviceRa, MetalError> {
        let columns = tables.len();
        let radix = tables
            .first()
            .map_or(0, |table| table.len().checked_div(width).unwrap_or(0));
        if width == 0
            || columns == 0
            || selectors.len() != columns
            || tails.len() != columns
            || !matches!(radix, 16 | 256)
            || tables.iter().any(|table| table.len() != width * radix)
            || selectors
                .iter()
                .any(|selector| selector[0] > 3 || selector[1] + radix.ilog2() > 64)
        {
            return Err(Self::invalid("invalid lazy selector geometry"));
        }
        let _span = tracing::info_span!("metal_ra_materialize").entered();
        let len = prefix
            .checked_mul(columns)
            .ok_or_else(|| Self::invalid("prefix size overflows"))?;
        let output = DeviceBuffer::<MaybeUninit<F>>::uninit(&self.device, len)?;
        let flat: Vec<_> = tables.iter().flatten().copied().collect();
        let tables = DeviceBuffer::from_slice(&self.device, &flat)?;
        let selectors = DeviceBuffer::from_slice(&self.device, selectors)?;
        let params = Self::params([rows.prefix.len(), prefix, width, columns, radix])?;
        self.dispatch(
            "ra_materialize",
            &[
                Binding::buffer(&rows.prefix),
                Binding::buffer(&tables),
                Binding::buffer(&selectors),
                Binding::buffer(&output),
                Binding::value(&params),
                Binding::value(&rows.tail),
                Binding::value(&challenge),
            ],
            len,
        )?;
        // SAFETY: successful dispatch completion precedes this conversion.
        // ra_materialize writes one canonical, padding-free F at every index
        // in 0..len, including rows filled entirely from the constant tail.
        // If len is zero, dispatch does no work and there are no values to
        // initialize; the allocation's physical padding is never exposed.
        let output = unsafe { output.assume_init()? };
        Ok(DeviceRa {
            prefix,
            tails: tails.to_vec(),
            values: output,
        })
    }

    pub(super) fn bind(&self, previous: &DeviceRa, challenge: F) -> Result<DeviceRa, MetalError> {
        let _span = tracing::info_span!("metal_ra_dense_bind").entered();
        let prefix = previous.prefix.div_ceil(2);
        let len = prefix
            .checked_mul(previous.tails.len())
            .ok_or_else(|| Self::invalid("prefix size overflows"))?;
        let output = DeviceBuffer::<MaybeUninit<F>>::uninit(&self.device, len)?;
        let tails = DeviceBuffer::from_slice(&self.device, &previous.tails)?;
        let params = Self::params([previous.prefix, prefix, previous.tails.len()])?;
        self.dispatch(
            "ra_fold",
            &[
                Binding::buffer(&previous.values),
                Binding::buffer(&tails),
                Binding::buffer(&output),
                Binding::value(&params),
                Binding::value(&challenge),
            ],
            len,
        )?;
        // SAFETY: ra_fold has completed and writes every logical element,
        // including the odd last row paired with the per-column tail. Field
        // arithmetic returns canonical, padding-free F values. A zero-length
        // dispatch has no logical elements and needs no initialization.
        let output = unsafe { output.assume_init()? };
        Ok(DeviceRa {
            prefix,
            tails: previous.tails.clone(),
            values: output,
        })
    }

    fn params<const N: usize>(values: [usize; N]) -> Result<[u32; N], MetalError> {
        let mut result = [0; N];
        for (out, value) in result.iter_mut().zip(values) {
            *out =
                u32::try_from(value).map_err(|_| Self::invalid("shape exceeds 32-bit indexing"))?;
        }
        Ok(result)
    }

    fn dispatch(&self, name: &str, bindings: &[Binding<'_>], len: usize) -> Result<(), MetalError> {
        if len == 0 {
            return Ok(());
        }
        let _ =
            u32::try_from(len).map_err(|_| Self::invalid("dispatch exceeds 32-bit indexing"))?;
        let pipeline = self.library.pipeline(&host_name::<F>(name))?;
        let threads = pipeline.max_total_threads_per_threadgroup().min(256);
        if threads == 0 {
            return Err(Self::invalid("empty threadgroup"));
        }
        let mut batch = Batch::new(&self.device)?;
        // SAFETY: checked shapes cover all buffers; each thread owns one output
        // element and absent prefix rows select the constant tail.
        unsafe {
            batch.dispatch_unchecked(pipeline, bindings, Grid::linear(len, threads))?;
        }
        let _ = batch.commit_and_wait()?;
        Ok(())
    }

    fn invalid(reason: &str) -> MetalError {
        MetalError::InvalidDispatch {
            pipeline: "resident RA fold".into(),
            reason: reason.into(),
        }
    }
}
