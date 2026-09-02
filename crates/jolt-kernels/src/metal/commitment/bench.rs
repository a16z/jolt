//! `bench-utils` fixture for the tier-1 `jk_g1_seg_sum` dispatch at
//! production shape (drives `jolt-eval`'s `st0-contention` g1 legs; see
//! `crates/jolt-kernels/BENCHES.md`).

use super::builder::{stage_one_hot_row, BucketLayout};
use super::*;

#[cfg(feature = "bench-utils")]
pub struct G1SegBenchFixture {
    bases: crate::metal::buffers::OwnedDeviceBuffer<u32>,
    rows: Vec<CommittedColumnsWitness>,
    one_hot: Vec<(usize, ColumnKind)>,
    row_width: usize,
    one_hot_k: usize,
    windows_total: usize,
}

#[cfg(feature = "bench-utils")]
pub struct G1SegBenchCase {
    indices: crate::metal::buffers::OwnedDeviceBuffer<u32>,
    bounds: crate::metal::buffers::OwnedDeviceBuffer<u32>,
    out: crate::metal::buffers::OwnedDeviceBuffer<u32>,
    destinations: Vec<u64>,
    additions: usize,
    useful_bytes: usize,
    montgomery_muls: usize,
}

#[cfg(feature = "bench-utils")]
#[derive(Clone, Copy, Debug)]
pub struct G1SegBenchSample {
    pub gpu_s: f64,
    pub wall_s: f64,
    pub segments: usize,
    pub additions: usize,
    pub useful_gbps: f64,
    pub gmul_s: f64,
}

#[cfg(feature = "bench-utils")]
impl G1SegBenchFixture {
    pub fn new(
        source: &dyn RowSource,
        ids: &[JoltCommittedPolynomial],
        grid: CommitmentGrid,
        setup: &DoryProverSetup,
    ) -> Result<Self, String> {
        let cycles = 1usize << grid.log_t;
        let row_width = grid.num_columns();
        let one_hot_k = 1usize << grid.log_k_chunk;
        let windows_total = cycles / row_width;
        let sample_windows = (SUPERCHUNK_CYCLES / row_width).clamp(1, windows_total);
        let sample_cycles = sample_windows * row_width;
        let kinds = column_kinds::<Fr>(ids, grid).map_err(|error| error.to_string())?;
        let one_hot = kinds
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, kind)| kind.is_one_hot())
            .collect();
        let rows = jolt_witness::collect_bundles::<CommittedColumnsWitness>(source, sample_cycles)
            .map_err(|error| error.to_string())?;
        let bases = DoryScheme::begin_one_hot_column_major_stream(setup, row_width);
        let context = MetalContext::global().map_err(|error| error.to_string())?;
        let bases = context
            .own_page_aligned(PageAlignedVec::from_slice(bases_as_u32s(&bases)))
            .map_err(|error| error.to_string())?;
        Ok(Self {
            bases,
            rows,
            one_hot,
            row_width,
            one_hot_k,
            windows_total,
        })
    }

    /// Base words + row width for the attribution rig.
    pub fn bases_words(&self) -> (&[u32], usize) {
        (self.bases.as_slice(), self.row_width)
    }

    pub fn build_case(&self, max_segment_len: usize) -> Result<G1SegBenchCase, MetalError> {
        assert!(max_segment_len.is_power_of_two());
        let context = MetalContext::global()?;
        // Stage the pre-collected bundles exactly as the fused extraction
        // pass would, then run the shared job layout.
        let n = self.rows.len();
        let mut scratch = DriverScratch::new(self.row_width);
        scratch.ensure_one_hot(n, self.one_hot.len(), self.one_hot_k);
        let sub = scratch.sub;
        let layout = BucketLayout::new(n / sub, self.row_width / sub, self.one_hot_k);
        let counts = scratch.oh_bases.as_mut_ptr();
        for (s, rows) in self.rows.chunks(sub).enumerate() {
            let hot = &mut scratch.hot[s * self.one_hot.len() * sub..];
            let subchunk_base = layout.subchunk_base(s);
            for (local, facts) in rows.iter().enumerate() {
                // SAFETY: serial fill — every subchunk's slots are ours.
                unsafe {
                    stage_one_hot_row(
                        facts,
                        &self.one_hot,
                        layout,
                        subchunk_base,
                        sub,
                        local,
                        hot,
                        counts,
                    );
                }
            }
        }
        let job = build_one_hot_job(
            &mut scratch,
            &mut SlabPool::detached(),
            n,
            self.one_hot.len(),
            self.one_hot_k,
            self.row_width,
            0,
            self.windows_total,
            max_segment_len,
        );
        let segments = job.segs.len();
        let additions: usize = job.seg_bounds[..3 * segments]
            .chunks_exact(3)
            .map(|bounds| (bounds[1] - bounds[0]) as usize)
            .sum();
        let useful_bytes =
            additions * (2 * FR_U32_LIMBS * 4 + 4) + segments * 12 + segments * JAC_U32S * 4;
        let mixed_adds = additions.saturating_sub(segments);
        let montgomery_muls = mixed_adds * 10 + segments * 4;
        let destinations = job
            .segs
            .iter()
            .map(|seg| (u64::from(seg.column) << 32) | u64::from(seg.row))
            .collect();
        Ok(G1SegBenchCase {
            indices: context.own_page_aligned(job.indices)?,
            bounds: context.own_page_aligned(job.seg_bounds)?,
            out: context.own_page_aligned(PageAlignedVec::from_elem(0u32, segments * JAC_U32S))?,
            destinations,
            additions,
            useful_bytes,
            montgomery_muls,
        })
    }

    pub fn assert_equivalent(&self, cases: &[&G1SegBenchCase]) -> Result<(), MetalError> {
        let mut expected = None;
        for case in cases {
            let _ = case.sample(self)?;
            let reduced = case.reduced_outputs();
            if let Some(expected) = &expected {
                assert_eq!(*expected, reduced, "segment cap changed tier-1 row sums");
            } else {
                expected = Some(reduced);
            }
        }
        Ok(())
    }
}

#[cfg(feature = "bench-utils")]
impl G1SegBenchCase {
    pub fn sample(&self, fixture: &G1SegBenchFixture) -> Result<G1SegBenchSample, MetalError> {
        let context = MetalContext::global()?;
        let bases = fixture.bases.device_buffer();
        let indices = self.indices.device_buffer();
        let bounds = self.bounds.device_buffer();
        let out = self.out.device_buffer();
        let mut pass = context.begin_pass()?;
        pass.dispatch_width(
            KernelId::G1SegSum,
            &[self.destinations.len() as u32],
            &[&bases, &indices, &bounds, &out],
            self.destinations.len(),
            SEG_SUM_WIDTH,
        );
        let wall_start = std::time::Instant::now();
        let gpu_s = pass.commit().wait_timed()?.as_secs_f64();
        let wall_s = wall_start.elapsed().as_secs_f64();
        Ok(G1SegBenchSample {
            gpu_s,
            wall_s,
            segments: self.destinations.len(),
            additions: self.additions,
            useful_gbps: self.useful_bytes as f64 / gpu_s / 1e9,
            gmul_s: self.montgomery_muls as f64 / gpu_s / 1e9,
        })
    }

    /// Raw case data for the attribution rig: production-shaped gather
    /// indices, length-sorted `[start, end, out_slot]` bounds triples, and
    /// the output slab.
    pub fn raw_parts(
        &self,
    ) -> (
        &[u32],
        &[u32],
        &crate::metal::buffers::OwnedDeviceBuffer<u32>,
    ) {
        (
            &self.indices.as_slice()[..self.additions],
            &self.bounds.as_slice()[..3 * self.destinations.len()],
            &self.out,
        )
    }

    fn reduced_outputs(&self) -> std::collections::BTreeMap<u64, G1Projective> {
        let mut rows = std::collections::BTreeMap::new();
        for (segment, &destination) in self.destinations.iter().enumerate() {
            let point = jac_from_device_limbs(
                &self.out.as_slice()[segment * JAC_U32S..(segment + 1) * JAC_U32S],
            );
            let _ = rows
                .entry(destination)
                .and_modify(|sum: &mut G1Projective| *sum += point)
                .or_insert(point);
        }
        rows
    }
}
