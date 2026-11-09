//! Resident address weights and bounded class histograms. Protocol geometry
//! and the canonical suffix/table mapping stay in the parent CPU kernel.
use super::*;

pub(super) const KERNELS: [&str; 5] = [
    "address_condense",
    "address_cache",
    "address_histogram",
    "address_combine",
    "address_cache_finish",
];
const THREADS: usize = 256;
const JOB_ROWS: usize = 1 << 16;
const JOB_BATCH: usize = 256;
const LANES: usize = 6;
const NONE: u32 = u32::MAX;

/// Explicit adapter; never interpret a protocol enum discriminant in Metal.
fn suffix_opcode(suffix: Suffixes) -> u32 {
    match suffix {
        Suffixes::One => 0,
        Suffixes::And => 1,
        Suffixes::AndNot => 2,
        Suffixes::Xor => 3,
        Suffixes::Or => 4,
        Suffixes::RightOperand => 5,
        Suffixes::UpperWord => 6,
        Suffixes::LowerWord => 7,
        Suffixes::LowerHalfWord => 8,
        Suffixes::LessThan => 9,
        Suffixes::GreaterThan => 10,
        Suffixes::Eq => 11,
        Suffixes::LeftOperandIsZero => 12,
        Suffixes::RightOperandIsZero => 13,
        Suffixes::Lsb => 14,
        Suffixes::DivByZero => 15,
        Suffixes::Pow2 => 16,
        Suffixes::Pow2W => 17,
        Suffixes::RightOperandW => 18,
        Suffixes::OverflowBitsZero => 19,
        Suffixes::SignExtensionUpperHalf => 20,
        Suffixes::SignExtensionRightOperand => 21,
        Suffixes::SignExtension => 22,
        Suffixes::RightShiftPadding => 23,
        Suffixes::RightShift => 24,
        Suffixes::RightShiftHelper => 25,
        Suffixes::LeftShift => 26,
        Suffixes::RightShiftW => 27,
        Suffixes::RightShiftWHelper => 28,
        Suffixes::LeftShiftW => 29,
        Suffixes::LeftShiftWHelper => 30,
        Suffixes::SignExtensionW => 31,
        Suffixes::X31Y0 => 32,
        Suffixes::Pow2OffsetB => 33,
        Suffixes::Pow2OffsetH => 34,
        Suffixes::Pow2OffsetW => 35,
        Suffixes::ShiftDataB => 36,
        Suffixes::ShiftDataH => 37,
        Suffixes::ShiftDataW => 38,
        Suffixes::OffsetScaleB => 39,
        Suffixes::OffsetScaleH => 40,
        Suffixes::OffsetScaleW => 41,
        Suffixes::AddressOffsetB => 42,
        Suffixes::AddressOffsetH => 43,
        Suffixes::AddressOffsetW => 44,
        Suffixes::Pext => 45,
        Suffixes::PextHelper => 46,
        Suffixes::WindowSign => 47,
        Suffixes::WindowSignPow2 => 48,
        Suffixes::Rev8W => 49,
        Suffixes::XorRot16 => 50,
        Suffixes::XorRot24 => 51,
        Suffixes::XorRot32 => 52,
        Suffixes::XorRot63 => 53,
        Suffixes::XorRotW16 => 54,
        Suffixes::XorRotW12 => 55,
        Suffixes::XorRotW8 => 56,
        Suffixes::XorRotW7 => 57,
        Suffixes::XorRotW22 => 58,
        Suffixes::XorRotW19 => 59,
        Suffixes::XorRotW6 => 60,
        Suffixes::XorRotL1Pairs => 61,
        Suffixes::TopYBit => 62,
        Suffixes::BottomXBit => 63,
    }
}

#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
pub(super) struct DeviceAddressScan {
    #[cfg_attr(feature = "allocative", allocative(skip))]
    gpu: Arc<ReadRafGpu>,
    rows: Arc<ResidentInstructionRows>,
    #[cfg_attr(feature = "allocative", allocative(skip))]
    records: DeviceBuffer<[u32; 2]>,
    #[cfg_attr(feature = "allocative", allocative(skip))]
    weights: DeviceBuffer<AkitaField>,
    #[cfg_attr(feature = "allocative", allocative(skip))]
    cache_meta: DeviceBuffer<[u32; 4]>,
    #[cfg_attr(feature = "allocative", allocative(skip))]
    cache_values: DeviceBuffer<AkitaField>,
    #[cfg_attr(feature = "allocative", allocative(skip))]
    cache_valid: DeviceBuffer<u32>,
    #[cfg_attr(feature = "allocative", allocative(skip))]
    partials: DeviceBuffer<AkitaField>,
    #[cfg_attr(feature = "allocative", allocative(skip))]
    output: DeviceBuffer<AkitaField>,
    jobs: Vec<[u32; 14]>,
    table_lanes: Vec<Option<(usize, usize)>>,
    caches: usize,
    cache_outputs: [u32; 5],
    next_phase: usize,
}

fn u32_len(value: usize) -> Result<u32, MetalError> {
    u32::try_from(value).map_err(|_| invalid("address scan exceeds 32-bit indexing"))
}

fn dispatch<'a>(
    gpu: &'a ReadRafGpu,
    batch: &mut Batch<'a>,
    name: &str,
    bindings: &[Binding<'a>],
    threads: usize,
) -> Result<(), MetalError> {
    let _ = u32_len(threads)?;
    let pipeline = gpu.library.pipeline(&host_name::<AkitaField>(name))?;
    // SAFETY: records index the validated compact rows or explicit tail;
    // jobs and output sizes come from the bounded host layout below.
    unsafe { batch.dispatch_unchecked(pipeline, bindings, Grid::linear(threads, THREADS)) }
}

impl DeviceAddressScan {
    pub(super) fn try_new(
        gpu: Arc<ReadRafGpu>,
        session: &mut ProofSession,
        host: &HostAddressScan<AkitaField>,
        minimum: usize,
    ) -> Result<Option<Self>, MetalError> {
        if host.weights.len() < minimum || CHUNK_SIZE != 256 || RISCV_XLEN != 64 {
            return Ok(None);
        }
        let mut present = vec![false; LookupTableKind::<RISCV_XLEN>::COUNT];
        for block in &host.blocks.layout {
            for (class, endpoints) in block.starts.windows(2).enumerate() {
                if endpoints[0] != endpoints[1] {
                    if let Some(table) = (class / 2).checked_sub(1) {
                        present[table] = true;
                    }
                }
            }
        }
        let tables: Vec<_> = LookupTableKind::<RISCV_XLEN>::iter().collect();
        let mut table_lanes = vec![None; tables.len()];
        let mut opcodes = vec![Vec::new(); tables.len()];
        let mut lanes = LANES;
        for (index, table) in tables.iter().enumerate().filter(|(i, _)| present[*i]) {
            for &suffix in table.suffixes() {
                opcodes[index].push(suffix_opcode(suffix));
            }
            table_lanes[index] = Some((lanes, opcodes[index].len()));
            lanes += opcodes[index].len();
        }
        for name in KERNELS {
            let pipeline = gpu.library.pipeline(&host_name::<AkitaField>(name))?;
            if pipeline.thread_execution_width() != 32
                || pipeline.max_total_threads_per_threadgroup() < THREADS
                || pipeline.static_threadgroup_memory_length()
                    > gpu.device.limits().max_threadgroup_memory_length
            {
                return Ok(None);
            }
        }
        let _span = tracing::info_span!("metal_read_raf_address_prepare").entered();
        let _ = u32_len(host.weights.len())?;
        let mut records = Vec::new();
        let mut weights = Vec::new();
        let mut cache_meta = Vec::new();
        let mut jobs = Vec::new();
        let range_table = RangePrefixSums::<AkitaField>::table();
        let range_cacheable = range_table
            .suffixes()
            .iter()
            .all(|s| matches!(s, Suffixes::One | Suffixes::LowerWord));
        let pack_span = tracing::info_span!("metal_read_raf_address_pack").entered();
        for class in 0..SCAN_CLASSES {
            let class_start = records.len();
            for (block_index, block) in host.blocks.layout.iter().enumerate() {
                let start = usize::from(block.starts[class]);
                let end = usize::from(block.starts[class + 1]);
                if start == end {
                    continue;
                }
                let cached = range_cacheable && class == ScanBlock::range_class();
                let cache = if cached {
                    u32_len(cache_meta.len())?
                } else {
                    NONE
                };
                let begin = u32_len(records.len())?;
                for &offset in &block.offsets[start..end] {
                    let row = block_index * SCAN_BLOCK + usize::from(offset);
                    records.push([u32_len(row)?, cache]);
                    weights.push(host.weights[row]);
                }
                if cached {
                    cache_meta.push([
                        begin,
                        u32_len(records.len())?,
                        u32::from(block.range_key_bits),
                        0,
                    ]);
                }
            }
            let suffixes = (class / 2)
                .checked_sub(1)
                .map(|table| (table, &opcodes[table]));
            let suffix_count = suffixes.map_or(0, |(_, codes)| codes.len());
            for suffix_batch in (0..suffix_count.max(1)).step_by(3) {
                for start in (class_start..records.len()).step_by(JOB_ROWS) {
                    // start/end/flag/RAF-present/suffix-count, three opcodes,
                    // then six logical output lanes. Metal uses the same uint layout.
                    let mut job = [NONE; 14];
                    job[..5].copy_from_slice(&[
                        u32_len(start)?,
                        u32_len((start + JOB_ROWS).min(records.len()))?,
                        (class % 2) as u32,
                        u32::from(suffix_batch == 0),
                        u32_len(suffix_count.saturating_sub(suffix_batch).min(3))?,
                    ]);
                    if suffix_batch == 0 {
                        let raf = if class.is_multiple_of(2) {
                            [0, 1, 2]
                        } else {
                            [3, 4, 5]
                        };
                        job[8..11].copy_from_slice(&raf);
                    }
                    if let Some((table, codes)) = suffixes {
                        let first_lane = table_lanes[table]
                            .ok_or_else(|| invalid("missing suffix output layout"))?
                            .0;
                        for (local, &opcode) in codes.iter().skip(suffix_batch).take(3).enumerate()
                        {
                            job[5 + local] = opcode;
                            job[11 + local] = u32_len(first_lane + suffix_batch + local)?;
                        }
                    }
                    jobs.push(job);
                }
            }
        }
        drop(pack_span);
        let caches = cache_meta.len();
        let mut cache_outputs = [3, 4, 5, NONE, NONE];
        if let Some((start, _)) = table_lanes[range_table.index()] {
            for (slot, suffix) in range_table.suffixes().iter().enumerate() {
                match suffix {
                    Suffixes::One => cache_outputs[3] = u32_len(start + slot)?,
                    Suffixes::LowerWord => cache_outputs[4] = u32_len(start + slot)?,
                    _ => {}
                }
            }
        }
        let rows = ResidentInstructionRows::shared(session, &host.rows)
            .map_err(|error| invalid(&error.to_string()))?;
        // Upload the shared prefix once. Its explicit tail represents the final
        // aggregate, including the last physical row in a full-prefix domain.
        drop(rows.resident(&gpu.device)?);
        let device = &gpu.device;
        let records = DeviceBuffer::from_slice(device, &records)?;
        let weights = DeviceBuffer::from_slice(device, &weights)?;
        let cache_meta = DeviceBuffer::from_slice(
            device,
            if cache_meta.is_empty() {
                &[[0; 4]][..]
            } else {
                &cache_meta
            },
        )?;
        let cache_values = DeviceBuffer::zeroed(device, caches.max(1) * 3)?;
        let cache_valid = DeviceBuffer::zeroed(device, caches.max(1))?;
        let partials = DeviceBuffer::zeroed(
            device,
            jobs.len().min(JOB_BATCH).max(1) * LANES * CHUNK_SIZE,
        )?;
        let output = DeviceBuffer::zeroed(device, lanes * CHUNK_SIZE)?;
        tracing::info!(
            representatives = weights.len(),
            caches,
            jobs = jobs.len(),
            lanes,
            "prepared resident ReadRAF address scan"
        );
        Ok(Some(Self {
            gpu,
            rows,
            records,
            weights,
            cache_meta,
            cache_values,
            cache_valid,
            partials,
            output,
            jobs,
            table_lanes,
            caches,
            cache_outputs,
            next_phase: 0,
        }))
    }

    fn execute(
        &mut self,
        request: &PhaseScan<'_, AkitaField>,
    ) -> Result<PhaseSums<AkitaField>, MetalError> {
        let span = tracing::info_span!(
            "metal_read_raf_address_phase",
            suffix_len = request.suffix_len,
            gpu_s = tracing::field::Empty,
        );
        let _span = span.enter();
        if request.condense.is_some() != (self.next_phase != 0) {
            return Err(invalid("address phases are out of order"));
        }
        let previous = request.condense.map_or(&[][..], |(_, values)| values);
        if !previous.is_empty() && previous.len() != CHUNK_SIZE {
            return Err(invalid("wrong address equality width"));
        }
        let previous = DeviceBuffer::from_slice(
            &self.gpu.device,
            if previous.is_empty() {
                &[AkitaField::zero()][..]
            } else {
                previous
            },
        )?;
        let owner = self.rows.resident(&self.gpu.device)?;
        let rows = owner
            .as_ref()
            .ok_or_else(|| invalid("missing address row owner"))?;
        let params = [
            u32_len(self.records.len())?,
            u32_len(rows.prefix.len())?,
            u32_len(request.suffix_len)?,
            u32_len(request.upper_suffix_bits)?,
            u32::from(request.condense.is_some()),
            request.condense.map_or(0, |(shift, _)| shift as u32),
            u32_len(self.caches)?,
            u32_len(self.output.len() / CHUNK_SIZE)?,
            u32::from(CANONICAL_INSTRUCTION_ADDRESS),
        ];
        let mut batch = Batch::new(&self.gpu.device)?;
        dispatch(
            &self.gpu,
            &mut batch,
            "address_condense",
            &[
                Binding::buffer(&self.records),
                Binding::buffer(&self.weights),
                Binding::buffer(&rows.prefix),
                Binding::value(&rows.tail),
                Binding::buffer(&previous),
                Binding::value(&params),
            ],
            self.records.len(),
        )?;
        dispatch(
            &self.gpu,
            &mut batch,
            "address_cache",
            &[
                Binding::buffer(&self.records),
                Binding::buffer(&self.weights),
                Binding::buffer(&rows.prefix),
                Binding::value(&rows.tail),
                Binding::buffer(&previous),
                Binding::value(&params),
                Binding::buffer(&self.cache_meta),
                Binding::buffer(&self.cache_values),
                Binding::buffer(&self.cache_valid),
            ],
            self.caches * THREADS,
        )?;
        // Job descriptors and their output-lane lists remain alive until the
        // single ordered command buffer completes. Partial storage is reused.
        let device_jobs: Vec<_> = self
            .jobs
            .chunks(JOB_BATCH)
            .map(|jobs| DeviceBuffer::from_slice(&self.gpu.device, jobs))
            .collect::<Result<_, _>>()?;
        let windows: Vec<_> = self
            .jobs
            .chunks(JOB_BATCH)
            .enumerate()
            .map(|(index, jobs)| [jobs.len() as u32, u32::from(index == 0)])
            .collect();
        for (jobs, window) in device_jobs.iter().zip(&windows) {
            dispatch(
                &self.gpu,
                &mut batch,
                "address_histogram",
                &[
                    Binding::buffer(&self.records),
                    Binding::buffer(&self.weights),
                    Binding::buffer(&rows.prefix),
                    Binding::value(&rows.tail),
                    Binding::buffer(&self.cache_meta),
                    Binding::buffer(jobs),
                    Binding::buffer(&self.partials),
                    Binding::value(&params),
                ],
                jobs.len() * THREADS,
            )?;
            dispatch(
                &self.gpu,
                &mut batch,
                "address_combine",
                &[
                    Binding::buffer(jobs),
                    Binding::buffer(&self.partials),
                    Binding::buffer(&self.output),
                    Binding::value(window),
                    Binding::value(&params),
                ],
                self.output.len(),
            )?;
        }
        dispatch(
            &self.gpu,
            &mut batch,
            "address_cache_finish",
            &[
                Binding::buffer(&self.cache_meta),
                Binding::buffer(&self.cache_values),
                Binding::buffer(&self.output),
                Binding::value(&self.cache_outputs),
                Binding::value(&params),
            ],
            5 * THREADS,
        )?;
        let gpu_time = batch.commit_and_wait()?;
        let _ = span.record("gpu_s", gpu_time.as_secs_f64());
        drop(owner);
        let fields = self.output.read()?;
        let get = |lane| fields[lane * CHUNK_SIZE..(lane + 1) * CHUNK_SIZE].to_vec();
        let raf = RafSums {
            shift_half: get(0),
            left: get(1),
            right: get(2),
            shift_full: get(3),
            identity: get(4),
            upper_all_ones: get(5),
        };
        let suffixes = self
            .table_lanes
            .iter()
            .map(|entry| {
                entry.map(|(start, count)| {
                    fields[start * CHUNK_SIZE..(start + count) * CHUNK_SIZE].to_vec()
                })
            })
            .collect();
        self.next_phase += 1;
        Ok(PhaseSums { raf, suffixes })
    }
}

impl AddressPhaseScanner<AkitaField> for DeviceAddressScan {
    fn scan(
        &mut self,
        request: &PhaseScan<'_, AkitaField>,
    ) -> Result<PhaseSums<AkitaField>, SumcheckError<AkitaField>> {
        self.execute(request).map_err(compute_error)
    }
}

#[cfg(test)]
pub(super) fn check_suffixes(gpu: &ReadRafGpu) -> Result<(), MetalError> {
    let mut suffixes = Vec::new();
    for table in LookupTableKind::<RISCV_XLEN>::iter() {
        for &suffix in table.suffixes() {
            if !suffixes.contains(&suffix) {
                suffixes.push(suffix);
            }
        }
    }
    let mut input = Vec::new();
    let mut expected = Vec::new();
    for suffix in suffixes {
        for bits in (0..128).step_by(CHUNK_LEN) {
            for key in [0u128, 0x8721_1a05_98bf_a37d, u128::from(u64::MAX)] {
                let value = LookupBits::new(key, bits);
                // The legacy W helper is defined on valid masks with fewer
                // than 32 leading ones; Rust rejects wider shifts in debug.
                if suffix == Suffixes::LeftShiftWHelper
                    && value.uninterleave().1.leading_ones() >= 32
                {
                    continue;
                }
                input.push([
                    key as u64,
                    (key >> 64) as u64,
                    bits as u64,
                    u64::from(suffix_opcode(suffix)),
                ]);
                expected.push(suffix.suffix_mle(value));
            }
        }
    }
    let count = u32_len(input.len())?;
    let input = DeviceBuffer::from_slice(&gpu.device, &input)?;
    let mut output = DeviceBuffer::<u64>::zeroed(&gpu.device, expected.len())?;
    let mut batch = Batch::new(&gpu.device)?;
    dispatch(
        gpu,
        &mut batch,
        "address_suffix_oracle",
        &[
            Binding::buffer(&input),
            Binding::buffer(&output),
            Binding::value(&count),
        ],
        expected.len(),
    )?;
    let _ = batch.commit_and_wait()?;
    assert_eq!(
        output.read()?,
        expected.as_slice(),
        "current suffix scalar definitions"
    );
    Ok(())
}
