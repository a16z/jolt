use super::message::MessageBuffers;
use super::*;

// Rows cover the explicit prefix, including empty rows. Values are compact:
// ascending set bits name the cells in offsets[row]..offsets[row + 1].
// Host masks alone advance the structural union; field values stay on device.
struct Topology {
    masks: Vec<u128>,
    offsets: Vec<u32>,
    device_masks: DeviceBuffer<u128>,
    device_offsets: DeviceBuffer<u32>,
}

impl Topology {
    fn new(gpu: &RegisterGpu, masks: Vec<u128>) -> Result<Self, MetalError> {
        let mut offsets = Vec::with_capacity(masks.len() + 1);
        let mut count = 0usize;
        for mask in &masks {
            offsets.push(u32_shape(count)?);
            count = count
                .checked_add(mask.count_ones() as usize)
                .ok_or_else(|| invalid("sparse count overflow"))?;
        }
        offsets.push(u32_shape(count)?);
        let _ = u32_shape(masks.len())?;
        let device_masks = DeviceBuffer::from_slice(&gpu.device, &masks)?;
        let device_offsets = DeviceBuffer::from_slice(&gpu.device, &offsets)?;
        Ok(Self {
            masks,
            offsets,
            device_masks,
            device_offsets,
        })
    }
    fn bound(&self, gpu: &RegisterGpu) -> Result<Self, MetalError> {
        let _span = tracing::info_span!("metal_registers_topology").entered();
        let source = self.masks.as_slice();
        let masks = super::super::super::support::map_indices(source.len().div_ceil(2), |row| {
            source[2 * row] | source.get(2 * row + 1).copied().unwrap_or(0)
        });
        Self::new(gpu, masks)
    }
    fn entries(&self) -> usize {
        self.offsets.last().copied().unwrap_or(0) as usize
    }
}

struct Seed {
    entries: DeviceBuffer<SeedEntry>,
    offsets: DeviceBuffer<u32>,
    raw_inc: Arc<Mutex<DeviceBuffer<i128>>>,
    rows: usize,
    topology: Topology,
    first: Option<F>,
}

enum Coefficients<V = F, C = u32> {
    Indexed(DeviceBuffer<C>),
    Direct {
        ra: DeviceBuffer<V>,
        wa: DeviceBuffer<V>,
        unused_codes: DeviceBuffer<u32>,
    },
}

struct Sparse<V = F, B = [u64; 2], C = u32> {
    topology: Topology,
    values: DeviceBuffer<V>,
    boundaries: DeviceBuffer<B>,
    coeffs: Coefficients<V, C>,
    inc: DeviceBuffer<V>,
}

/// Output storage stays uninitialized in the type until its full GPU writer
/// has completed. Inactive arguments remain ordinary initialized placeholders.
type PendingSparse = Sparse<MaybeUninit<F>, MaybeUninit<[u64; 2]>, MaybeUninit<u32>>;

impl PendingSparse {
    fn new(
        gpu: &RegisterGpu,
        topology: Topology,
        inc_len: usize,
        direct: bool,
    ) -> Result<Self, MetalError> {
        let _span = tracing::info_span!("metal_registers_sparse_allocate").entered();
        if inc_len > topology.masks.len() {
            return Err(invalid("increment output exceeds dispatched topology rows"));
        }
        let entries = topology.entries();
        let values = DeviceBuffer::uninit(&gpu.device, entries)?;
        let boundaries = DeviceBuffer::uninit(&gpu.device, entries)?;
        let coeffs = if direct {
            Coefficients::Direct {
                ra: DeviceBuffer::uninit(&gpu.device, entries)?,
                wa: DeviceBuffer::uninit(&gpu.device, entries)?,
                unused_codes: DeviceBuffer::zeroed(&gpu.device, 1)?,
            }
        } else {
            Coefficients::Indexed(DeviceBuffer::uninit(&gpu.device, entries)?)
        };
        Ok(Self {
            topology,
            values,
            boundaries,
            coeffs,
            inc: DeviceBuffer::uninit(&gpu.device, inc_len)?,
        })
    }
    /// The caller must have completed a seed bind, sparse bind or cooperative
    /// sparse bind/message over every topology row with matching extents.
    unsafe fn finish(self) -> Result<Sparse, MetalError> {
        // SAFETY: the caller supplies the completed full-write dispatch;
        // inactive placeholders are already initialized by new().
        unsafe {
            Ok(Sparse {
                topology: self.topology,
                values: self.values.assume_init()?,
                boundaries: self.boundaries.assume_init()?,
                inc: self.inc.assume_init()?,
                coeffs: match self.coeffs {
                    Coefficients::Indexed(codes) => Coefficients::Indexed(codes.assume_init()?),
                    Coefficients::Direct {
                        ra,
                        wa,
                        unused_codes,
                    } => Coefficients::Direct {
                        ra: ra.assume_init()?,
                        wa: wa.assume_init()?,
                        unused_codes,
                    },
                },
            })
        }
    }
}

impl<V, B, C> Sparse<V, B, C> {
    // Inactive field operands may alias values. The code operand still needs
    // a uint element: runtime reflection checks even unused argument types.
    fn codes(&self) -> Binding<'_> {
        match &self.coeffs {
            Coefficients::Indexed(codes) => Binding::buffer(codes),
            Coefficients::Direct { unused_codes, .. } => Binding::buffer(unused_codes),
        }
    }
    fn ra(&self) -> Binding<'_> {
        match &self.coeffs {
            Coefficients::Direct { ra, .. } => Binding::buffer(ra),
            Coefficients::Indexed(_) => Binding::buffer(&self.values),
        }
    }
    fn wa(&self) -> Binding<'_> {
        match &self.coeffs {
            Coefficients::Direct { wa, .. } => Binding::buffer(wa),
            Coefficients::Indexed(_) => Binding::buffer(&self.values),
        }
    }
    fn direct(&self) -> bool {
        matches!(self.coeffs, Coefficients::Direct { .. })
    }
}

enum Phase {
    Seed(Seed),
    Sparse(Sparse),
}

pub(super) struct PreparedDeviceCycle {
    pub cycle: DeviceCycle,
    pub increments: Arc<Mutex<DeviceBuffer<i128>>>,
}

pub(super) struct DeviceCycle {
    phase: Phase,
    ra_lut: CoeffLut<F>,
    wa_lut: CoeffLut<F>,
}

pub(super) struct BoundCycle {
    pub cycle: DeviceCycle,
    pub message: Option<[F; 3]>,
}

impl DeviceCycle {
    pub(super) fn prepare(
        gpu: &RegisterGpu,
        seed: SeedTransfer<F, DeviceBuffer<SeedEntry>>,
    ) -> Result<PreparedDeviceCycle, MetalError> {
        let _span = tracing::info_span!("metal_registers_upload").entered();
        let SeedTransfer {
            entries: mut device_entries,
            ra_lut,
            wa_lut,
            rd_inc: raw_inc,
        } = seed;
        let entries = device_entries.read()?;
        let _ = u32_shape(entries.len())?;
        let _ = u32_shape(raw_inc.len())?;
        let rows = entries.last().map_or(0, |entry| entry.row as usize + 1);
        if raw_inc.len() > rows {
            return Err(invalid("increment prefix exceeds represented seed rows"));
        }
        let groups = rows.div_ceil(4);
        let topology_span = tracing::info_span!("metal_registers_topology").entered();
        let mut offsets = vec![0u32; groups + 1];
        let mut masks = vec![0u128; groups];
        let fill = |(tile, (offsets, masks)): (usize, (&mut [u32], &mut [u128]))| {
            let first = tile * 1024;
            let mut cursor = entries.partition_point(|entry| (entry.row as usize / 4) < first);
            for (local, (offset, mask)) in offsets.iter_mut().zip(masks).enumerate() {
                *offset = cursor as u32;
                let group = first + local;
                while cursor < entries.len() && entries[cursor].row as usize / 4 == group {
                    *mask |= 1u128 << entries[cursor].col;
                    cursor += 1;
                }
            }
        };
        #[cfg(feature = "parallel")]
        offsets[..groups]
            .par_chunks_mut(1024)
            .zip(masks.par_chunks_mut(1024))
            .enumerate()
            .for_each(fill);
        #[cfg(not(feature = "parallel"))]
        offsets[..groups]
            .chunks_mut(1024)
            .zip(masks.chunks_mut(1024))
            .enumerate()
            .for_each(fill);
        offsets[groups] = entries.len() as u32;
        let topology = Topology::new(gpu, masks)?;
        drop(topology_span);
        let raw_inc = Arc::new(Mutex::new(DeviceBuffer::from_slice(&gpu.device, &raw_inc)?));
        let seed = Seed {
            entries: device_entries,
            offsets: DeviceBuffer::from_slice(&gpu.device, &offsets)?,
            raw_inc: Arc::clone(&raw_inc),
            rows,
            topology,
            first: None,
        };
        Ok(PreparedDeviceCycle {
            cycle: Self {
                phase: Phase::Seed(seed),
                ra_lut,
                wa_lut,
            },
            increments: raw_inc,
        })
    }

    pub(super) fn ready_for_cpu(&self, entries: usize) -> bool {
        matches!(&self.phase, Phase::Sparse(state) if state.direct() && state.topology.entries() <= entries)
    }

    pub(super) fn bind(
        mut self,
        gpu: &RegisterGpu,
        r: F,
        weights: Option<RoundWeights<'_>>,
        tail_entries: usize,
    ) -> Result<BoundCycle, MetalError> {
        let span = tracing::info_span!("metal_registers_bind", phase = "seed", fused = false);
        let _span = span.enter();
        if let Phase::Seed(seed) = &mut self.phase {
            if seed.first.is_none() {
                seed.first = Some(r);
                self.ra_lut.bind(r);
                self.wa_lut.bind(r);
                return Ok(BoundCycle {
                    cycle: self,
                    message: None,
                });
            }
        }
        let mut message = None;
        let mut lut_advanced = false;
        let next = match self.phase {
            Phase::Seed(seed) => {
                let raw_inc = seed
                    .raw_inc
                    .lock()
                    .map_err(|_| invalid("increment source lock poisoned"))?;
                // Structural topology is independent of either binding challenge.
                let params = [
                    u32_shape(seed.topology.masks.len())?,
                    u32_shape(raw_inc.len())?,
                ];
                let next =
                    PendingSparse::new(gpu, seed.topology, raw_inc.len().div_ceil(4), false)?;
                let first = seed
                    .first
                    .ok_or_else(|| invalid("second bind lacks first challenge"))?;
                gpu.dispatch(
                    "rw_seed_bind",
                    &[
                        Binding::buffer(&seed.entries),
                        Binding::buffer(&seed.offsets),
                        Binding::buffer(&*raw_inc),
                        Binding::buffer(&next.topology.device_masks),
                        Binding::buffer(&next.topology.device_offsets),
                        Binding::buffer(&next.values),
                        Binding::buffer(&next.boundaries),
                        next.codes(),
                        Binding::buffer(&next.inc),
                        Binding::value(&params),
                        Binding::value(&first),
                        Binding::value(&r),
                    ],
                    next.topology.masks.len(),
                )?;
                // SAFETY: the completed seed bind writes each union cell and
                // every ceil(raw_inc.len()/4) increment, including empty rows.
                unsafe { next.finish()? }
            }
            Phase::Sparse(old) => {
                let direct = old.direct() || self.ra_lut.saturated() || self.wa_lut.saturated();
                let phase = if old.direct() {
                    "direct"
                } else if direct {
                    "transition"
                } else {
                    "indexed"
                };
                let _ = span.record("phase", phase);
                let next = PendingSparse::new(
                    gpu,
                    old.topology.bound(gpu)?,
                    old.inc.len().div_ceil(2),
                    direct,
                )?;
                let ra_bits = self.ra_lut.bits();
                let wa_bits = self.wa_lut.bits();
                let fused_weights =
                    weights.filter(|_| !(direct && next.topology.entries() <= tail_entries));
                if fused_weights.is_some() && !direct {
                    // Indexed outputs use the bound codebook in their message;
                    // transition outputs instead resolve the old saturated LUT.
                    self.ra_lut.bind(r);
                    self.wa_lut.bind(r);
                    lut_advanced = true;
                }
                let ra_lut = DeviceBuffer::from_slice(&gpu.device, &self.ra_lut.values)?;
                let wa_lut = DeviceBuffer::from_slice(&gpu.device, &self.wa_lut.values)?;
                let lanes = if old.direct() { 8 } else { 4 };
                let mut buffers = fused_weights
                    .map(|weights| {
                        MessageBuffers::new(
                            gpu,
                            next.topology.masks.len(),
                            weights,
                            THREADS / lanes,
                        )
                    })
                    .transpose()?;
                let _ = span.record("fused", buffers.is_some());
                let params = [
                    u32_shape(old.topology.masks.len())?,
                    u32_shape(next.topology.masks.len())?,
                    u32_shape(old.inc.len())?,
                    u32::from(old.direct()),
                    u32::from(direct),
                    ra_bits,
                    wa_bits,
                ];
                let fused_params = if let (Some(weights), Some(buffers)) = (fused_weights, &buffers)
                {
                    [
                        params[0],
                        params[1],
                        params[2],
                        params[3],
                        params[4],
                        params[5],
                        params[6],
                        u32_shape(buffers.pairs)?,
                        u32_shape(weights.e_in.len())?,
                        u32::from(weights.need_one),
                    ]
                } else {
                    [0; 10]
                };
                let constants = if buffers.is_some() {
                    Binding::value(&fused_params)
                } else {
                    Binding::value(&params)
                };
                let mut bindings = vec![
                    Binding::buffer(&old.values),
                    Binding::buffer(&old.boundaries),
                    old.codes(),
                    old.ra(),
                    old.wa(),
                    Binding::buffer(&old.topology.device_masks),
                    Binding::buffer(&old.topology.device_offsets),
                    Binding::buffer(&old.inc),
                    Binding::buffer(&ra_lut),
                    Binding::buffer(&wa_lut),
                    Binding::buffer(&next.values),
                    Binding::buffer(&next.boundaries),
                    next.codes(),
                    next.ra(),
                    next.wa(),
                    Binding::buffer(&next.topology.device_masks),
                    Binding::buffer(&next.topology.device_offsets),
                    Binding::buffer(&next.inc),
                    constants,
                    Binding::value(&r),
                ];
                let (name, threads) = if let Some(buffers) = &buffers {
                    bindings.extend([
                        Binding::buffer(&buffers.e_in),
                        Binding::buffer(&buffers.e_out),
                        Binding::buffer(&buffers.partials),
                    ]);
                    (
                        "rw_sparse_bind_message_cooperative",
                        buffers.groups * THREADS,
                    )
                } else {
                    ("rw_sparse_bind", next.topology.masks.len())
                };
                gpu.dispatch(name, &bindings, threads)?;
                drop(bindings);
                if let Some(buffers) = &mut buffers {
                    message = Some(buffers.totals()?);
                }
                // SAFETY: the completed sparse bind writes each union cell,
                // both active coefficient planes and every bound increment.
                unsafe { next.finish()? }
            }
        };
        if !next.direct() && !lut_advanced {
            self.ra_lut.bind(r);
            self.wa_lut.bind(r);
        }
        if next.direct() {
            self.ra_lut = CoeffLut::new(vec![F::zero()]);
            self.wa_lut = CoeffLut::new(vec![F::zero()]);
        }
        self.phase = Phase::Sparse(next);
        Ok(BoundCycle {
            cycle: self,
            message,
        })
    }

    pub(super) fn message(
        &mut self,
        gpu: &RegisterGpu,
        weights: RoundWeights<'_>,
    ) -> Result<[F; 3], MetalError> {
        let rows = match &self.phase {
            Phase::Seed(seed) => {
                if seed.first.is_some() {
                    seed.rows.div_ceil(2)
                } else {
                    seed.rows
                }
            }
            Phase::Sparse(state) => state.topology.masks.len(),
        };
        let mut buffers = MessageBuffers::new(gpu, rows, weights, PAIRS_PER_GROUP)?;
        if buffers.groups == 0 {
            return Ok([F::zero(); 3]);
        }
        let ra_lut = DeviceBuffer::from_slice(&gpu.device, &self.ra_lut.values)?;
        let wa_lut = DeviceBuffer::from_slice(&gpu.device, &self.wa_lut.values)?;
        match &self.phase {
            Phase::Seed(seed) => {
                let raw_inc = seed
                    .raw_inc
                    .lock()
                    .map_err(|_| invalid("increment source lock poisoned"))?;
                let params = [
                    u32_shape(rows)?,
                    u32_shape(buffers.pairs)?,
                    u32_shape(weights.e_in.len())?,
                    u32_shape(raw_inc.len())?,
                    u32::from(seed.first.is_some()),
                    u32::from(weights.need_one),
                ];
                let first = seed.first.unwrap_or_else(F::zero);
                gpu.dispatch(
                    "rw_seed_message",
                    &[
                        Binding::buffer(&seed.entries),
                        Binding::buffer(&seed.offsets),
                        Binding::buffer(&*raw_inc),
                        Binding::buffer(&seed.topology.device_masks),
                        Binding::buffer(&ra_lut),
                        Binding::buffer(&wa_lut),
                        Binding::buffer(&buffers.e_in),
                        Binding::buffer(&buffers.e_out),
                        Binding::buffer(&buffers.partials),
                        Binding::value(&params),
                        Binding::value(&first),
                    ],
                    buffers.groups * THREADS,
                )?;
            }
            Phase::Sparse(state) => {
                let params = [
                    u32_shape(rows)?,
                    u32_shape(buffers.pairs)?,
                    u32_shape(weights.e_in.len())?,
                    u32_shape(state.inc.len())?,
                    u32::from(state.direct()),
                    u32::from(weights.need_one),
                ];
                gpu.dispatch(
                    "rw_sparse_message",
                    &[
                        Binding::buffer(&state.values),
                        Binding::buffer(&state.boundaries),
                        state.codes(),
                        state.ra(),
                        state.wa(),
                        Binding::buffer(&state.topology.device_masks),
                        Binding::buffer(&state.topology.device_offsets),
                        Binding::buffer(&state.inc),
                        Binding::buffer(&ra_lut),
                        Binding::buffer(&wa_lut),
                        Binding::buffer(&buffers.e_in),
                        Binding::buffer(&buffers.e_out),
                        Binding::buffer(&buffers.partials),
                        Binding::value(&params),
                    ],
                    buffers.groups * THREADS,
                )?;
            }
        }
        buffers.totals()
    }

    pub(super) fn restore(self) -> Result<CycleState<F>, MetalError> {
        let _span = tracing::info_span!("metal_registers_restore").entered();
        let Phase::Sparse(mut state) = self.phase else {
            return Err(invalid("CPU restoration before sparse materialization"));
        };
        let Coefficients::Direct { mut ra, mut wa, .. } = state.coeffs else {
            return Err(invalid("CPU restoration before LUT transition"));
        };
        let values = state.values.read()?;
        let boundaries = state.boundaries.read()?;
        let ra = ra.read()?;
        let wa = wa.read()?;
        let mut entries = Vec::with_capacity(values.len());
        for (row, &mask) in state.topology.masks.iter().enumerate() {
            let mut mask = mask;
            let mut index = state.topology.offsets[row] as usize;
            while mask != 0 {
                let col = mask.trailing_zeros() as u8;
                mask &= mask - 1;
                entries.push(SparseEntry {
                    val: values[index],
                    prev_val: boundaries[index][0],
                    next_val: boundaries[index][1],
                    row,
                    ra: ra[index],
                    wa: wa[index],
                    col,
                });
                index += 1;
            }
        }
        Ok(CycleState::from_direct(entries, state.inc.read()?.to_vec()))
    }
}

impl RegisterGpu {
    fn dispatch(
        &self,
        name: &str,
        bindings: &[Binding<'_>],
        threads: usize,
    ) -> Result<(), MetalError> {
        if threads == 0 {
            return Ok(());
        }
        let _ = u32_shape(threads)?;
        let pipeline = self.library.pipeline(&host_name::<F>(name))?;
        let width = pipeline.thread_execution_width();
        if width == 0
            || !THREADS.is_multiple_of(width)
            || THREADS / width > 32
            || THREADS / width > width
            || (name == "rw_sparse_bind_message_cooperative"
                && (width < 8 || !width.is_multiple_of(8)))
            || pipeline.max_total_threads_per_threadgroup() < THREADS
            || pipeline.static_threadgroup_memory_length()
                > self.device.limits().max_threadgroup_memory_length
        {
            return Err(invalid("unsupported SIMD reduction geometry"));
        }
        let span = tracing::info_span!(
            "metal_registers_dispatch",
            kernel = name,
            gpu_s = tracing::field::Empty
        );
        let _span = span.enter();
        let mut batch = Batch::new(&self.device)?;
        // SAFETY: callers check 32-bit extents, topology offsets describe exact
        // allocations, and shaders guard prefix loads and uniquely own outputs.
        unsafe {
            batch.dispatch_unchecked(pipeline, bindings, Grid::linear(threads, THREADS))?;
        }
        let time = batch.commit_and_wait()?;
        let _ = span.record("gpu_s", time.as_secs_f64());
        Ok(())
    }
}
