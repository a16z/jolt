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
        u32_shape(masks.len())?;
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
        let masks =
            super::super::super::support::map_indices(self.masks.len().div_ceil(2), |row| {
                self.masks[2 * row] | self.masks.get(2 * row + 1).copied().unwrap_or(0)
            });
        Self::new(gpu, masks)
    }
    fn entries(&self) -> usize {
        self.offsets.last().copied().unwrap_or(0) as usize
    }
}

struct Seed {
    entries: DeviceBuffer<[u64; 3]>,
    offsets: DeviceBuffer<u32>,
    raw_inc: DeviceBuffer<i128>,
    rows: usize,
    topology: Topology,
    first: Option<F>,
}

enum Coefficients {
    Indexed(DeviceBuffer<u32>),
    Direct {
        ra: DeviceBuffer<F>,
        wa: DeviceBuffer<F>,
    },
}

struct Sparse {
    topology: Topology,
    values: DeviceBuffer<F>,
    boundaries: DeviceBuffer<[u64; 2]>,
    coeffs: Coefficients,
    inc: DeviceBuffer<F>,
}

impl Sparse {
    fn new(
        gpu: &RegisterGpu,
        topology: Topology,
        inc_len: usize,
        direct: bool,
    ) -> Result<Self, MetalError> {
        let entries = topology.entries();
        let values = DeviceBuffer::zeroed(&gpu.device, entries)?;
        let boundaries = DeviceBuffer::zeroed(&gpu.device, entries)?;
        let coeffs = if direct {
            Coefficients::Direct {
                ra: DeviceBuffer::zeroed(&gpu.device, entries)?,
                wa: DeviceBuffer::zeroed(&gpu.device, entries)?,
            }
        } else {
            Coefficients::Indexed(DeviceBuffer::zeroed(&gpu.device, entries)?)
        };
        Ok(Self {
            topology,
            values,
            boundaries,
            coeffs,
            inc: DeviceBuffer::zeroed(&gpu.device, inc_len)?,
        })
    }
    // Inactive coefficient operands alias the value buffer; the shader's
    // indexed/direct branch never accesses those operands.
    fn codes(&self) -> Binding<'_> {
        match &self.coeffs {
            Coefficients::Indexed(codes) => Binding::buffer(codes),
            _ => Binding::buffer(&self.values),
        }
    }
    fn ra(&self) -> Binding<'_> {
        match &self.coeffs {
            Coefficients::Direct { ra, .. } => Binding::buffer(ra),
            _ => Binding::buffer(&self.values),
        }
    }
    fn wa(&self) -> Binding<'_> {
        match &self.coeffs {
            Coefficients::Direct { wa, .. } => Binding::buffer(wa),
            _ => Binding::buffer(&self.values),
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

pub(super) struct DeviceCycle {
    phase: Phase,
    ra_lut: CoeffLut<F>,
    wa_lut: CoeffLut<F>,
}

impl DeviceCycle {
    pub(super) fn new(gpu: &RegisterGpu, cycle: CycleState<F>) -> Result<Self, MetalError> {
        let _span = tracing::info_span!("metal_registers_upload").entered();
        let (entries, ra_lut, wa_lut, raw_inc) = cycle
            .into_seed()
            .ok_or_else(|| invalid("expected seed cycle owner"))?;
        u32_shape(entries.len())?;
        u32_shape(raw_inc.len())?;
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
        let packed = super::super::super::support::map_indices(entries.len(), |i| {
            let entry = entries[i];
            [
                entry.prev_val,
                entry.next_val,
                u64::from(entry.row)
                    | (u64::from(entry.ra) << 32)
                    | (u64::from(entry.wa) << 40)
                    | (u64::from(entry.col) << 48),
            ]
        });
        drop(entries);
        let seed = Seed {
            entries: DeviceBuffer::from_slice(&gpu.device, &packed)?,
            offsets: DeviceBuffer::from_slice(&gpu.device, &offsets)?,
            raw_inc: DeviceBuffer::from_slice(&gpu.device, &raw_inc)?,
            rows,
            topology,
            first: None,
        };
        Ok(Self {
            phase: Phase::Seed(seed),
            ra_lut,
            wa_lut,
        })
    }

    pub(super) fn ready_for_cpu(&self, entries: usize) -> bool {
        matches!(&self.phase, Phase::Sparse(state) if state.direct() && state.topology.entries() <= entries)
    }

    pub(super) fn bind(mut self, gpu: &RegisterGpu, r: F) -> Result<Self, MetalError> {
        let _span = tracing::info_span!("metal_registers_bind").entered();
        if let Phase::Seed(seed) = &mut self.phase {
            if seed.first.is_none() {
                seed.first = Some(r);
                self.ra_lut.bind(r);
                self.wa_lut.bind(r);
                return Ok(self);
            }
        }
        let ra_lut = DeviceBuffer::from_slice(&gpu.device, &self.ra_lut.values)?;
        let wa_lut = DeviceBuffer::from_slice(&gpu.device, &self.wa_lut.values)?;
        let next = match self.phase {
            Phase::Seed(seed) => {
                // Structural topology is independent of either binding challenge.
                let params = [
                    u32_shape(seed.topology.masks.len())?,
                    u32_shape(seed.raw_inc.len())?,
                ];
                let next = Sparse::new(gpu, seed.topology, seed.raw_inc.len().div_ceil(4), false)?;
                let challenges = [
                    seed.first
                        .ok_or_else(|| invalid("second bind lacks first challenge"))?,
                    r,
                ];
                gpu.dispatch(
                    "rw_seed_bind",
                    &[
                        Binding::buffer(&seed.entries),
                        Binding::buffer(&seed.offsets),
                        Binding::buffer(&seed.raw_inc),
                        Binding::buffer(&next.topology.device_masks),
                        Binding::buffer(&next.topology.device_offsets),
                        Binding::buffer(&next.values),
                        Binding::buffer(&next.boundaries),
                        next.codes(),
                        Binding::buffer(&next.inc),
                        Binding::value(&params),
                        Binding::value(&challenges),
                    ],
                    next.topology.masks.len(),
                )?;
                next
            }
            Phase::Sparse(old) => {
                let direct = old.direct() || self.ra_lut.saturated() || self.wa_lut.saturated();
                let next = Sparse::new(
                    gpu,
                    old.topology.bound(gpu)?,
                    old.inc.len().div_ceil(2),
                    direct,
                )?;
                let params = [
                    u32_shape(old.topology.masks.len())?,
                    u32_shape(next.topology.masks.len())?,
                    u32_shape(old.inc.len())?,
                    u32::from(old.direct()),
                    u32::from(direct),
                    self.ra_lut.bits(),
                    self.wa_lut.bits(),
                ];
                gpu.dispatch(
                    "rw_sparse_bind",
                    &[
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
                        Binding::value(&params),
                        Binding::value(&r),
                    ],
                    next.topology.masks.len(),
                )?;
                next
            }
        };
        if !next.direct() {
            self.ra_lut.bind(r);
            self.wa_lut.bind(r);
        }
        if next.direct() {
            self.ra_lut = CoeffLut::new(vec![F::zero()]);
            self.wa_lut = CoeffLut::new(vec![F::zero()]);
        }
        self.phase = Phase::Sparse(next);
        Ok(self)
    }

    pub(super) fn message(
        &mut self,
        gpu: &RegisterGpu,
        e_in: &[F],
        e_out: &[F],
        need_one: bool,
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
        let pairs = rows.div_ceil(2);
        if e_in.is_empty()
            || !e_in.len().is_power_of_two()
            || pairs.div_ceil(e_in.len()) > e_out.len()
        {
            return Err(invalid("equality tables do not cover sparse prefix"));
        }
        let groups = pairs.div_ceil(PAIRS_PER_GROUP);
        let mut partials = DeviceBuffer::<F>::zeroed(
            &gpu.device,
            groups
                .checked_mul(3)
                .ok_or_else(|| invalid("partial length overflow"))?,
        )?;
        if groups == 0 {
            return Ok([F::zero(); 3]);
        }
        let in_values = DeviceBuffer::from_slice(&gpu.device, e_in)?;
        let out_values = DeviceBuffer::from_slice(&gpu.device, e_out)?;
        let ra_lut = DeviceBuffer::from_slice(&gpu.device, &self.ra_lut.values)?;
        let wa_lut = DeviceBuffer::from_slice(&gpu.device, &self.wa_lut.values)?;
        match &self.phase {
            Phase::Seed(seed) => {
                let params = [
                    u32_shape(rows)?,
                    u32_shape(pairs)?,
                    u32_shape(e_in.len())?,
                    u32_shape(seed.raw_inc.len())?,
                    u32::from(seed.first.is_some()),
                    u32::from(need_one),
                ];
                let first = seed.first.unwrap_or_else(F::zero);
                gpu.dispatch(
                    "rw_seed_message",
                    &[
                        Binding::buffer(&seed.entries),
                        Binding::buffer(&seed.offsets),
                        Binding::buffer(&seed.raw_inc),
                        Binding::buffer(&seed.topology.device_masks),
                        Binding::buffer(&ra_lut),
                        Binding::buffer(&wa_lut),
                        Binding::buffer(&in_values),
                        Binding::buffer(&out_values),
                        Binding::buffer(&partials),
                        Binding::value(&params),
                        Binding::value(&first),
                    ],
                    groups * THREADS,
                )?;
            }
            Phase::Sparse(state) => {
                let params = [
                    u32_shape(rows)?,
                    u32_shape(pairs)?,
                    u32_shape(e_in.len())?,
                    u32_shape(state.inc.len())?,
                    u32::from(state.direct()),
                    u32::from(need_one),
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
                        Binding::buffer(&in_values),
                        Binding::buffer(&out_values),
                        Binding::buffer(&partials),
                        Binding::value(&params),
                    ],
                    groups * THREADS,
                )?;
            }
        }
        let mut sums = [F::zero(); 3];
        for part in partials.read()?.chunks_exact(3) {
            for (sum, value) in sums.iter_mut().zip(part) {
                *sum += value;
            }
        }
        Ok(sums)
    }

    pub(super) fn restore(self) -> Result<CycleState<F>, MetalError> {
        let _span = tracing::info_span!("metal_registers_restore").entered();
        let Phase::Sparse(mut state) = self.phase else {
            return Err(invalid("CPU restoration before sparse materialization"));
        };
        let Coefficients::Direct { mut ra, mut wa } = state.coeffs else {
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
        u32_shape(threads)?;
        let pipeline = self.library.pipeline(&host_name::<F>(name))?;
        let width = pipeline.thread_execution_width();
        if width == 0
            || THREADS % width != 0
            || THREADS / width > 32
            || THREADS / width > width
            || pipeline.max_total_threads_per_threadgroup() < THREADS
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
