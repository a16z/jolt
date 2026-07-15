use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use cudarc::driver::{
    result as cuda_result, CudaContext, CudaFunction, CudaStream, LaunchConfig, PushKernelArg,
};
pub use cudarc::driver::CudaSlice;
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

use jolt_field::Fr;

const LIMBS: usize = 4;
const BLOCK: u32 = 256;

const KERNEL_SRC: &str = concat!(
    include_str!("cuda/prelude.cu"),
    include_str!("cuda/add.cu"),
    include_str!("cuda/sub.cu"),
    include_str!("cuda/mul.cu"),
    include_str!("cuda/fma.cu"),
    include_str!("cuda/bind.cu"),
    include_str!("cuda/eq_double.cu"),
    include_str!("cuda/lt_double.cu"),
    include_str!("cuda/rd_wa_gather.cu"),
    include_str!("cuda/add_scalar.cu"),
    include_str!("cuda/row_dots.cu"),
    include_str!("cuda/dense_outer_fused.cu"),
    include_str!("cuda/dense_outer.cu"),
    include_str!("cuda/cubic.cu"),
    include_str!("cuda/round_poly.cu"),
    include_str!("cuda/dense_product.cu"),
    include_str!("cuda/gruen_round_poly.cu"),
    include_str!("cuda/uniskip.cu"),
    include_str!("cuda/gather8.cu"),
    include_str!("cuda/core_booleanity_gather.cu"),
    include_str!("cuda/core_booleanity_sparse.cu"),
    include_str!("cuda/hamming.cu"),
    include_str!("cuda/hamming_booleanity.cu"),
    include_str!("cuda/core_booleanity_cycle.cu"),
    include_str!("cuda/core_booleanity_address.cu"),
    include_str!("cuda/sparse_register.cu"),
    include_str!("cuda/instruction_raf_cycle.cu"),
    include_str!("cuda/instruction_raf_cycle_sparse.cu"),
    include_str!("cuda/ra_virtual_d4.cu"),
    include_str!("cuda/ra_virtual_d4_sparse.cu"),
    include_str!("cuda/bytecode_cycle_sparse.cu"),
    include_str!("cuda/raf_q_scatter.cu"),
    include_str!("cuda/ram_derive.cu"),
    include_str!("cuda/raf_weight_phase_update.cu"),
    include_str!("cuda/suffix_mle.cu"),
    include_str!("cuda/prefix_combine.cu"),
    include_str!("cuda/read_suffix_scatter.cu"),
    include_str!("cuda/ram_rw_cycle.cu"),
    include_str!("cuda/ram_rw_address.cu"),
    include_str!("cuda/reduce.cu"),
);

#[derive(Debug)]
pub enum CudaError {
    Compile(cudarc::nvrtc::CompileError),
    Driver(cudarc::driver::DriverError),
    Pool,
    Unsupported,
}

impl std::fmt::Display for CudaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CudaError::Compile(e) => write!(f, "nvrtc compile error: {e:?}"),
            CudaError::Driver(e) => write!(f, "cuda driver error: {e:?}"),
            CudaError::Pool => write!(f, "pinned staging pool invariant violated"),
            CudaError::Unsupported => write!(f, "cuda kernel does not support these inputs"),
        }
    }
}

impl std::error::Error for CudaError {}

impl From<cudarc::nvrtc::CompileError> for CudaError {
    fn from(e: cudarc::nvrtc::CompileError) -> Self {
        CudaError::Compile(e)
    }
}

impl From<cudarc::driver::DriverError> for CudaError {
    fn from(e: cudarc::driver::DriverError) -> Self {
        CudaError::Driver(e)
    }
}

pub struct CudaKernelContext {
    stream: Arc<CudaStream>,
    add: CudaFunction,
    sub: CudaFunction,
    mul: CudaFunction,
    fma: CudaFunction,
    bind: CudaFunction,
    eq_double: CudaFunction,
    lt_double: CudaFunction,
    raf_q_scatter: CudaFunction,
    raf_q_scatter_reduce: CudaFunction,
    raf_weight_phase_update: CudaFunction,
    #[cfg(test)]
    suffix_mle_probe: CudaFunction,
    #[cfg(test)]
    prefix_combine_probe: CudaFunction,
    read_suffix_scatter: CudaFunction,
    ram_rw_cycle_round_pairs: CudaFunction,
    ram_rw_cycle_bind: CudaFunction,
    u64_to_mont: CudaFunction,
    ram_rw_address_round_pairs: CudaFunction,
    ram_rw_address_bind: CudaFunction,
    rd_wa_gather: CudaFunction,
    add_scalar: CudaFunction,
    dense_outer: CudaFunction,
    dense_outer_fused: CudaFunction,
    row_dots: CudaFunction,
    cubic_pairs: CudaFunction,
    cubic_tuple_reduce: CudaFunction,
    round_poly_pairs: CudaFunction,
    eq_round_poly_pairs: CudaFunction,
    round_poly_reduce: CudaFunction,
    dense_product_pairs: CudaFunction,
    gruen_round_poly_pairs: CudaFunction,
    uniskip_pairs: CudaFunction,
    gather8_materialize: CudaFunction,
    core_booleanity_gather: CudaFunction,
    core_booleanity_sparse_pairs: CudaFunction,
    core_booleanity_sparse_bind: CudaFunction,
    core_booleanity_sparse_collapse8: CudaFunction,
    hamming_pairs: CudaFunction,
    hamming_booleanity_pairs: CudaFunction,
    ra_virtual_d4_pairs: CudaFunction,
    ra_virtual_d4_sparse_pairs: CudaFunction,
    ra_virtual_d4_sparse_bind: CudaFunction,
    ra_virtual_d4_sparse_collapse: CudaFunction,
    bytecode_cycle_sparse_pairs: CudaFunction,
    core_booleanity_cycle_pairs: CudaFunction,
    core_booleanity_address_pairs: CudaFunction,
    sparse_register_round_pairs: CudaFunction,
    sparse_register_bind_kernel: CudaFunction,
    instruction_raf_cycle_pairs: CudaFunction,
    instruction_raf_cycle_sparse_pairs: CudaFunction,
    instruction_raf_cycle_sparse_collapse: CudaFunction,
    sum_reduce: CudaFunction,
    product_reduce: CudaFunction,
    one_dev: CudaSlice<u64>,
    staging: PinnedStaging,
    resident_witness: ResidentCache,
    resident_committed: CommittedCache,
}

pub struct DeviceFrVec {
    stream: Arc<CudaStream>,
    buf: CudaSlice<u64>,
    len: usize,
    staging: PinnedStaging,
}

impl DeviceFrVec {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn to_host(&self) -> Result<Vec<Fr>, CudaError> {
        if self.len == 0 {
            return Ok(Vec::new());
        }
        let n = self.len * LIMBS;
        xfer_stats::add_d2h(n * std::mem::size_of::<u64>());
        xfer_stats::timed(xfer_stats::Phase::D2h, || {
            let mut pool = lock_pool(&self.staging);
            let staging = pool.ensure(self.stream.context(), n)?;
            self.stream
                .memcpy_dtoh(&self.buf.slice(0..n), staging.as_mut_slice(n))?;
            self.stream.synchronize()?;
            Ok(unflatten(staging.as_slice(n)))
        })
    }

    pub fn try_clone(&self) -> Result<Self, CudaError> {
        Ok(Self {
            stream: self.stream.clone(),
            buf: self.stream.clone_dtod(&self.buf)?,
            len: self.len,
            staging: self.staging.clone(),
        })
    }

    pub fn first(&self) -> Result<Fr, CudaError> {
        xfer_stats::add_d2h(LIMBS * std::mem::size_of::<u64>());
        let raw = xfer_stats::timed(xfer_stats::Phase::D2h, || {
            self.stream.clone_dtoh(&self.buf.slice(0..LIMBS))
        })?;
        Ok(limbs_to_fr([raw[0], raw[1], raw[2], raw[3]]))
    }
}

#[inline]
fn fr_to_limbs(f: Fr) -> [u64; LIMBS] {
    f.inner_limbs().0
}

#[inline]
fn limbs_to_fr(limbs: [u64; LIMBS]) -> Fr {
    Fr::from_bigint_unchecked(jolt_field::Limbs(limbs))
}

pub(crate) struct RoundSchedule {
    pub(crate) even_idx: Vec<i32>,
    pub(crate) odd_idx: Vec<i32>,
    pub(crate) pair: Vec<u32>,
}

pub(crate) fn build_schedules<C: Copy + Ord>(
    rows: &[usize],
    cols: &[C],
    rounds: usize,
) -> (Vec<RoundSchedule>, Vec<C>) {
    let mut cur_rows: Vec<usize> = rows.to_vec();
    let mut cur_cols: Vec<C> = cols.to_vec();
    let mut schedules = Vec::with_capacity(rounds);

    for _ in 0..rounds {
        let mut even_idx = Vec::new();
        let mut odd_idx = Vec::new();
        let mut pair = Vec::new();
        let mut next_rows = Vec::new();
        let mut next_cols = Vec::new();

        let mut cursor = 0usize;
        while cursor < cur_rows.len() {
            let p = cur_rows[cursor] / 2;
            let even_row = 2 * p;
            let odd_row = even_row + 1;
            let even_start = cursor;
            while cursor < cur_rows.len() && cur_rows[cursor] == even_row {
                cursor += 1;
            }
            let even = even_start..cursor;
            let odd_start = cursor;
            while cursor < cur_rows.len() && cur_rows[cursor] == odd_row {
                cursor += 1;
            }
            let odd = odd_start..cursor;

            let mut i = even.start;
            let mut j = odd.start;
            while i < even.end || j < odd.end {
                let (ei, oi, col) = if j >= odd.end || (i < even.end && cur_cols[i] < cur_cols[j]) {
                    let out = (i as i32, -1i32, cur_cols[i]);
                    i += 1;
                    out
                } else if i >= even.end || cur_cols[j] < cur_cols[i] {
                    let out = (-1i32, j as i32, cur_cols[j]);
                    j += 1;
                    out
                } else {
                    let out = (i as i32, j as i32, cur_cols[i]);
                    i += 1;
                    j += 1;
                    out
                };
                even_idx.push(ei);
                odd_idx.push(oi);
                pair.push(p as u32);
                next_rows.push(p);
                next_cols.push(col);
            }
        }

        schedules.push(RoundSchedule {
            even_idx,
            odd_idx,
            pair,
        });
        cur_rows = next_rows;
        cur_cols = next_cols;
    }

    (schedules, cur_cols)
}

fn unflatten(raw: &[u64]) -> Vec<Fr> {
    raw.chunks_exact(LIMBS)
        .map(|c| limbs_to_fr([c[0], c[1], c[2], c[3]]))
        .collect()
}

pub(crate) fn shared_ctx() -> Option<&'static CudaKernelContext> {
    use std::sync::OnceLock;
    static CTX: OnceLock<Option<CudaKernelContext>> = OnceLock::new();
    CTX.get_or_init(|| CudaKernelContext::new(0).ok()).as_ref()
}

pub mod xfer_stats {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::OnceLock;

    #[derive(Default)]
    pub struct Counters {
        pub pack_d2d_bytes: AtomicU64,
        pub pack_d2d_calls: AtomicU64,
        pub h2d_bytes: AtomicU64,
        pub h2d_calls: AtomicU64,
        pub d2h_bytes: AtomicU64,
        pub d2h_calls: AtomicU64,
        pub h2d_small: AtomicU64,
        pub h2d_medium: AtomicU64,
        pub h2d_large: AtomicU64,
        pub h2d_large_bytes: AtomicU64,
        pub ns_materialize: AtomicU64,
        pub ns_upload: AtomicU64,
        pub ns_kernel: AtomicU64,
        pub ns_d2h: AtomicU64,
        pub ns_bind: AtomicU64,
        pub bind_calls: AtomicU64,
        pub h2d_raw_bytes: AtomicU64,
        pub h2d_raw_calls: AtomicU64,
        pub ns_h2d_raw: AtomicU64,
    }

    fn counters() -> &'static Counters {
        static C: OnceLock<Counters> = OnceLock::new();
        C.get_or_init(Counters::default)
    }

    pub fn enabled() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| std::env::var_os("JOLT_CUDA_XFER_STATS").is_some())
    }

    #[inline]
    #[expect(dead_code)]
    pub(crate) fn add_pack_d2d(bytes: usize) {
        if enabled() {
            let _ = counters().pack_d2d_bytes.fetch_add(bytes as u64, Ordering::Relaxed);
            let _ = counters().pack_d2d_calls.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[inline]
    pub(crate) fn add_h2d(bytes: usize) {
        if enabled() {
            let _ = counters().h2d_bytes.fetch_add(bytes as u64, Ordering::Relaxed);
            let _ = counters().h2d_calls.fetch_add(1, Ordering::Relaxed);
            if bytes < 64 * 1024 {
                let _ = counters().h2d_small.fetch_add(1, Ordering::Relaxed);
            } else if bytes < 1024 * 1024 {
                let _ = counters().h2d_medium.fetch_add(1, Ordering::Relaxed);
            } else {
                let _ = counters().h2d_large.fetch_add(1, Ordering::Relaxed);
                let _ = counters().h2d_large_bytes.fetch_add(bytes as u64, Ordering::Relaxed);
            }
        }
    }

    #[inline]
    pub(crate) fn add_h2d_raw(bytes: usize, ns: u64) {
        if enabled() {
            let c = counters();
            let _ = c.h2d_raw_bytes.fetch_add(bytes as u64, Ordering::Relaxed);
            let _ = c.h2d_raw_calls.fetch_add(1, Ordering::Relaxed);
            let _ = c.ns_h2d_raw.fetch_add(ns, Ordering::Relaxed);
        }
    }

    #[inline]
    pub(crate) fn add_d2h(bytes: usize) {
        if enabled() {
            let _ = counters().d2h_bytes.fetch_add(bytes as u64, Ordering::Relaxed);
            let _ = counters().d2h_calls.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub(crate) enum Phase {
        Materialize,
        Upload,
        #[expect(dead_code)]
        Kernel,
        D2h,
        Bind,
    }

    #[inline]
    pub(crate) fn timed<T>(phase: Phase, f: impl FnOnce() -> T) -> T {
        if !enabled() {
            return f();
        }
        let start = std::time::Instant::now();
        let out = f();
        let ns = start.elapsed().as_nanos() as u64;
        let c = counters();
        let bucket = match phase {
            Phase::Materialize => &c.ns_materialize,
            Phase::Upload => &c.ns_upload,
            Phase::Kernel => &c.ns_kernel,
            Phase::D2h => &c.ns_d2h,
            Phase::Bind => {
                let _ = c.bind_calls.fetch_add(1, Ordering::Relaxed);
                &c.ns_bind
            }
        };
        let _ = bucket.fetch_add(ns, Ordering::Relaxed);
        out
    }

    pub fn snapshot() -> [u64; 19] {
        let c = counters();
        [
            c.pack_d2d_bytes.load(Ordering::Relaxed),
            c.pack_d2d_calls.load(Ordering::Relaxed),
            c.h2d_bytes.load(Ordering::Relaxed),
            c.h2d_calls.load(Ordering::Relaxed),
            c.d2h_bytes.load(Ordering::Relaxed),
            c.d2h_calls.load(Ordering::Relaxed),
            c.h2d_small.load(Ordering::Relaxed),
            c.h2d_medium.load(Ordering::Relaxed),
            c.h2d_large.load(Ordering::Relaxed),
            c.h2d_large_bytes.load(Ordering::Relaxed),
            c.ns_materialize.load(Ordering::Relaxed),
            c.ns_upload.load(Ordering::Relaxed),
            c.ns_kernel.load(Ordering::Relaxed),
            c.ns_d2h.load(Ordering::Relaxed),
            c.ns_bind.load(Ordering::Relaxed),
            c.bind_calls.load(Ordering::Relaxed),
            c.h2d_raw_bytes.load(Ordering::Relaxed),
            c.h2d_raw_calls.load(Ordering::Relaxed),
            c.ns_h2d_raw.load(Ordering::Relaxed),
        ]
    }

    pub fn reset() {
        let c = counters();
        for a in [
            &c.pack_d2d_bytes,
            &c.pack_d2d_calls,
            &c.h2d_bytes,
            &c.h2d_calls,
            &c.d2h_bytes,
            &c.d2h_calls,
            &c.h2d_small,
            &c.h2d_medium,
            &c.h2d_large,
            &c.h2d_large_bytes,
            &c.ns_materialize,
            &c.ns_upload,
            &c.ns_kernel,
            &c.ns_d2h,
            &c.ns_bind,
            &c.bind_calls,
            &c.h2d_raw_bytes,
            &c.h2d_raw_calls,
            &c.ns_h2d_raw,
        ] {
            a.store(0, Ordering::Relaxed);
        }
    }
}

pub(crate) fn as_fr_slice<F: jolt_field::Field>(values: &[F]) -> Option<&[Fr]> {
    if std::any::TypeId::of::<F>() == std::any::TypeId::of::<Fr>() {
        // SAFETY: F and Fr are the same type (checked above), so &[F] and &[Fr]
        // have identical layout.
        Some(unsafe { &*(std::ptr::from_ref::<[F]>(values) as *const [Fr]) })
    } else {
        None
    }
}

pub(crate) fn into_fr<F: jolt_field::Field>(value: F) -> Option<Fr> {
    (Box::new(value) as Box<dyn std::any::Any>)
        .downcast::<Fr>()
        .ok()
        .map(|boxed| *boxed)
}

pub(crate) fn fr_into<F: jolt_field::Field>(value: Fr) -> Option<F> {
    (Box::new(value) as Box<dyn std::any::Any>)
        .downcast::<F>()
        .ok()
        .map(|boxed| *boxed)
}

struct PinnedBuf {
    ctx: Arc<CudaContext>,
    ptr: *mut u64,
    cap: usize,
}

impl PinnedBuf {
    fn with_capacity(ctx: &Arc<CudaContext>, cap: usize) -> Result<Self, CudaError> {
        ctx.bind_to_thread()?;
        // SAFETY: malloc_host returns uninitialized page-locked memory; callers
        // only read limbs they have written or that a DMA has filled.
        let ptr = unsafe { cuda_result::malloc_host(cap * std::mem::size_of::<u64>(), 0)? };
        Ok(Self {
            ctx: ctx.clone(),
            ptr: ptr.cast::<u64>(),
            cap,
        })
    }

    fn as_slice(&self, len: usize) -> &[u64] {
        debug_assert!(len <= self.cap);
        // SAFETY: `ptr` points at `cap >= len` u64s of page-locked memory owned by self.
        unsafe { std::slice::from_raw_parts(self.ptr, len) }
    }

    fn as_mut_slice(&mut self, len: usize) -> &mut [u64] {
        debug_assert!(len <= self.cap);
        // SAFETY: `ptr` points at `cap >= len` u64s of page-locked memory owned by
        // self, and `&mut self` guarantees exclusive access.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, len) }
    }
}

impl Drop for PinnedBuf {
    fn drop(&mut self) {
        let _ = self.ctx.bind_to_thread();
        // SAFETY: `ptr` was returned by malloc_host and is freed exactly once.
        self.ctx
            .record_err(unsafe { cuda_result::free_host(self.ptr.cast()) });
    }
}

// SAFETY: the allocation is owned solely by this `PinnedBuf` and only accessed
// behind a borrow (and, when shared, a Mutex), like cudarc's `PinnedHostSlice`.
unsafe impl Send for PinnedBuf {}
// SAFETY: see the `Send` impl above.
unsafe impl Sync for PinnedBuf {}

#[derive(Default)]
struct PinnedPool {
    buf: Option<PinnedBuf>,
}

impl PinnedPool {
    fn ensure(&mut self, ctx: &Arc<CudaContext>, len: usize) -> Result<&mut PinnedBuf, CudaError> {
        let grow = match &self.buf {
            Some(b) => b.cap < len,
            None => true,
        };
        if grow {
            self.buf = Some(PinnedBuf::with_capacity(ctx, len)?);
        }
        match self.buf.as_mut() {
            Some(b) => Ok(b),
            None => Err(CudaError::Pool),
        }
    }
}

type PinnedStaging = Arc<Mutex<PinnedPool>>;

#[derive(PartialEq, Eq, Clone, Copy)]
struct WitnessKey {
    ptr: usize,
    len: usize,
    fingerprint: u64,
}

impl WitnessKey {
    fn of(witness: &[Fr]) -> Self {
        let len = witness.len();
        let mut fingerprint = 0xcbf2_9ce4_8422_2325u64 ^ (len as u64);
        if len > 0 {
            let step = (len / 8).max(1);
            let mut i = 0;
            while i < len {
                for limb in fr_to_limbs(witness[i]) {
                    fingerprint = (fingerprint ^ limb).wrapping_mul(0x0000_0100_0000_01b3);
                }
                i += step;
            }
            for limb in fr_to_limbs(witness[len - 1]) {
                fingerprint = (fingerprint ^ limb).wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        Self {
            ptr: witness.as_ptr() as usize,
            len,
            fingerprint,
        }
    }
}

struct ResidentWitness {
    key: WitnessKey,
    buf: Arc<DeviceFrVec>,
}

type ResidentCache = Arc<Mutex<Option<ResidentWitness>>>;

fn lock_resident(cache: &ResidentCache) -> MutexGuard<'_, Option<ResidentWitness>> {
    cache.lock().unwrap_or_else(PoisonError::into_inner)
}

type CommittedCache = Arc<Mutex<Vec<ResidentWitness>>>;

const COMMITTED_CACHE_CAP: usize = 8;

fn lock_committed(cache: &CommittedCache) -> MutexGuard<'_, Vec<ResidentWitness>> {
    cache.lock().unwrap_or_else(PoisonError::into_inner)
}

fn lock_pool(pool: &PinnedStaging) -> MutexGuard<'_, PinnedPool> {
    pool.lock().unwrap_or_else(PoisonError::into_inner)
}

pub struct FusedOuterInputs<'a> {
    pub eq_evals: &'a [Fr],
    pub scale: Fr,
    pub witness: &'a [Fr],
    pub a_offsets: &'a [u32],
    pub a_vars: &'a [u32],
    pub a_coeffs: &'a [Fr],
    pub b_offsets: &'a [u32],
    pub b_vars: &'a [u32],
    pub b_coeffs: &'a [Fr],
    pub split: usize,
    pub num_vars_padded: usize,
}

pub struct RoundPolyTerms<'a> {
    pub factors: &'a [&'a DeviceFrVec],
    pub term_coeffs: &'a DeviceFrVec,
    pub term_factor_offsets: &'a [u32],
    pub term_factor_indices: &'a [u32],
    pub degree: usize,
}

pub struct RafQScatterInputs<'a> {
    pub weight: &'a DeviceFrVec,
    pub lookup_index_lo: &'a CudaSlice<u64>,
    pub lookup_index_hi: &'a CudaSlice<u64>,
    pub is_interleaved: &'a CudaSlice<u8>,
    pub trace_len: usize,
    pub suffix_len: usize,
    pub poly_len: usize,
}

pub struct ReadSuffixScatterInputs<'a> {
    pub weight: &'a DeviceFrVec,
    pub lookup_index_lo: &'a CudaSlice<u64>,
    pub lookup_index_hi: &'a CudaSlice<u64>,
    pub cycle_list: &'a CudaSlice<u32>,
    pub suffix_variants: &'a CudaSlice<u32>,
    pub m: usize,
    pub suffix_len: usize,
    pub poly_len: usize,
}

pub struct RamRwCycleRoundInputs<'a> {
    pub val_coeff: &'a DeviceFrVec,
    pub ra_coeff: &'a DeviceFrVec,
    pub prev_val: &'a DeviceFrVec,
    pub next_val: &'a DeviceFrVec,
    pub even_idx: &'a CudaSlice<i32>,
    pub odd_idx: &'a CudaSlice<i32>,
    pub pair: &'a CudaSlice<u32>,
    pub inc: &'a DeviceFrVec,
    pub e_in: &'a DeviceFrVec,
    pub e_out: &'a DeviceFrVec,
    pub gamma: Fr,
    pub in_pairs: u32,
    pub items: usize,
}

pub struct RamRwCycleBindInputs<'a> {
    pub val_coeff: &'a DeviceFrVec,
    pub ra_coeff: &'a DeviceFrVec,
    pub prev_val: &'a DeviceFrVec,
    pub next_val: &'a DeviceFrVec,
    pub even_idx: &'a CudaSlice<i32>,
    pub odd_idx: &'a CudaSlice<i32>,
    pub challenge: Fr,
    pub items: usize,
}

pub struct RamRwCycleEntries {
    pub val_coeff: DeviceFrVec,
    pub ra_coeff: DeviceFrVec,
    pub prev_val: DeviceFrVec,
    pub next_val: DeviceFrVec,
}

pub struct RamRwAddressRoundInputs<'a> {
    pub ra_coeff: &'a DeviceFrVec,
    pub val_coeff: &'a DeviceFrVec,
    pub val_init: &'a DeviceFrVec,
    pub even_idx: &'a CudaSlice<i32>,
    pub odd_idx: &'a CudaSlice<i32>,
    pub pair: &'a CudaSlice<u32>,
    pub eq: Fr,
    pub gamma: Fr,
    pub inc0: Fr,
    pub num_groups: usize,
}

pub struct RamRwAddressBindInputs<'a> {
    pub ra_coeff: &'a DeviceFrVec,
    pub val_coeff: &'a DeviceFrVec,
    pub prev_val: &'a DeviceFrVec,
    pub next_val: &'a DeviceFrVec,
    pub val_init: &'a DeviceFrVec,
    pub even_idx: &'a CudaSlice<i32>,
    pub odd_idx: &'a CudaSlice<i32>,
    pub pair: &'a CudaSlice<u32>,
    pub challenge: Fr,
    pub num_groups: usize,
}

pub struct RamRwAddressEntries {
    pub ra_coeff: DeviceFrVec,
    pub val_coeff: DeviceFrVec,
    pub prev_val: DeviceFrVec,
    pub next_val: DeviceFrVec,
}

pub struct GruenRoundPolyInputs<'a> {
    pub factors: &'a [&'a DeviceFrVec],
    pub term_coeffs: &'a DeviceFrVec,
    pub term_factor_offsets: &'a [u32],
    pub term_factor_indices: &'a [u32],
    pub e_in: &'a DeviceFrVec,
    pub e_out: &'a DeviceFrVec,
    pub degree: usize,
}

pub struct Gather8Inputs<'a> {
    pub table_groups: [&'a [Fr]; 8],
    pub indices: &'a [i16],
    pub num_chunks: usize,
    pub table_len: usize,
    pub new_len: usize,
}

pub struct CoreBooleanityGatherInputs<'a> {
    pub tables: &'a [Fr],
    pub present_mask: &'a [u64],
    pub values: &'a [u8],
    pub num_polys: usize,
    pub chunk_domain: usize,
    pub rows: usize,
    pub poly_stride: usize,
}

pub struct CoreBooleanitySparseInputs<'a> {
    pub tables: &'a CudaSlice<u64>,
    pub present_mask: &'a CudaSlice<u64>,
    pub values: &'a CudaSlice<u8>,
    pub source_rows: usize,
    pub rho: &'a DeviceFrVec,
    pub e_in: &'a DeviceFrVec,
    pub e_out: &'a DeviceFrVec,
    pub num_polys: usize,
    pub chunk_domain: usize,
    pub poly_stride: usize,
    pub round: u32,
}

pub struct HammingRoundPolyInputs<'a> {
    pub g: &'a [&'a DeviceFrVec],
    pub eq_virt: &'a [&'a DeviceFrVec],
    pub eq_bool: &'a DeviceFrVec,
    pub gamma_powers: &'a DeviceFrVec,
    pub scale: Fr,
}

pub struct HammingBooleanityInputs<'a> {
    pub hamming_weight: &'a DeviceFrVec,
    pub e_in: &'a DeviceFrVec,
    pub e_out: &'a DeviceFrVec,
}

pub struct RaVirtualD4Inputs<'a> {
    pub chunks: &'a [&'a DeviceFrVec],
    pub gamma_powers: &'a DeviceFrVec,
    pub e_in: &'a DeviceFrVec,
    pub e_out: &'a DeviceFrVec,
}

pub struct RaVirtualD4SparseInputs<'a> {
    pub tables: &'a CudaSlice<u64>,
    pub values: &'a CudaSlice<i16>,
    pub num_chunks: usize,
    pub chunk_domain: usize,
    pub source_rows: usize,
    pub gamma_powers: &'a DeviceFrVec,
    pub e_in: &'a DeviceFrVec,
    pub e_out: &'a DeviceFrVec,
    pub round: u32,
}

pub struct BytecodeCycleSparseInputs<'a> {
    pub tables: &'a CudaSlice<u64>,
    pub values: &'a CudaSlice<i16>,
    pub combined_eq: &'a DeviceFrVec,
    pub num_chunks: usize,
    pub chunk_domain: usize,
    pub source_rows: usize,
    pub degree: usize,
    pub round: u32,
}

pub struct InstructionRafCycleSparseInputs<'a> {
    pub tables: &'a CudaSlice<u64>,
    pub values: &'a CudaSlice<u16>,
    pub combined: &'a DeviceFrVec,
    pub num_chunks: usize,
    pub chunk_domain: usize,
    pub source_rows: usize,
    pub e_in: &'a DeviceFrVec,
    pub e_out: &'a DeviceFrVec,
    pub round: u32,
}

pub struct CoreBooleanityCycleInputs<'a> {
    pub h_polys: &'a [&'a DeviceFrVec],
    pub rho: &'a DeviceFrVec,
    pub e_in: &'a DeviceFrVec,
    pub e_out: &'a DeviceFrVec,
}

pub struct CoreBooleanityAddressInputs<'a> {
    pub g: &'a [&'a DeviceFrVec],
    pub f_values: &'a [Fr],
    pub gamma_squares: &'a DeviceFrVec,
    pub e_in: &'a DeviceFrVec,
    pub e_out: &'a DeviceFrVec,
    pub m: u32,
}

pub struct SparseRegisterRoundInputs<'a> {
    pub val: &'a DeviceFrVec,
    pub read_ra: &'a DeviceFrVec,
    pub rd_wa: &'a DeviceFrVec,
    pub prev_val: &'a DeviceFrVec,
    pub next_val: &'a DeviceFrVec,
    pub even_idx: &'a [i32],
    pub odd_idx: &'a [i32],
    pub pair: &'a [u32],
    pub rd_inc: &'a DeviceFrVec,
    pub e_in: &'a DeviceFrVec,
    pub e_out: &'a DeviceFrVec,
    pub in_pairs: u32,
}

pub struct SparseRegisterBindInputs<'a> {
    pub val: &'a DeviceFrVec,
    pub read_ra: &'a DeviceFrVec,
    pub rd_wa: &'a DeviceFrVec,
    pub prev_val: &'a DeviceFrVec,
    pub next_val: &'a DeviceFrVec,
    pub even_idx: &'a [i32],
    pub odd_idx: &'a [i32],
    pub challenge: Fr,
}

pub struct SparseRegisterEntries {
    pub val: DeviceFrVec,
    pub read_ra: DeviceFrVec,
    pub rd_wa: DeviceFrVec,
    pub prev_val: DeviceFrVec,
    pub next_val: DeviceFrVec,
}

pub struct InstructionRafCycleInputs<'a> {
    pub combined: &'a DeviceFrVec,
    pub chunks: &'a [&'a DeviceFrVec],
    pub e_in: &'a DeviceFrVec,
    pub e_out: &'a DeviceFrVec,
}

pub struct UniskipInputs<'a> {
    pub row_dots_a: &'a DeviceFrVec,
    pub row_dots_b: &'a DeviceFrVec,
    pub eq_evals: &'a DeviceFrVec,
    pub first_group_rows: &'a [u32],
    pub second_group_rows: &'a [u32],
    pub first_coeffs: &'a [Fr],
    pub second_coeffs: &'a [Fr],
    pub row_count: usize,
    pub degree: usize,
}

impl CudaKernelContext {
    pub fn new(ordinal: usize) -> Result<Self, CudaError> {
        let ctx = CudaContext::new(ordinal)?;
        let stream = ctx.default_stream();
        let opts = CompileOptions {
            options: vec!["--device-int128".to_string()],
            ..Default::default()
        };
        let module = ctx.load_module(compile_ptx_with_opts(KERNEL_SRC, opts)?)?;
        let one_dev = stream.clone_htod(&fr_to_limbs(<Fr as num_traits::One>::one()))?;
        Ok(Self {
            add: module.load_function("add_kernel")?,
            sub: module.load_function("sub_kernel")?,
            mul: module.load_function("mul_kernel")?,
            fma: module.load_function("fma_kernel")?,
            bind: module.load_function("bind_kernel")?,
            eq_double: module.load_function("eq_double")?,
            lt_double: module.load_function("lt_double")?,
            raf_q_scatter: module.load_function("raf_q_scatter")?,
            raf_q_scatter_reduce: module.load_function("raf_q_scatter_reduce")?,
            raf_weight_phase_update: module.load_function("raf_weight_phase_update")?,
            #[cfg(test)]
            suffix_mle_probe: module.load_function("suffix_mle_probe")?,
            #[cfg(test)]
            prefix_combine_probe: module.load_function("prefix_combine_probe")?,
            read_suffix_scatter: module.load_function("read_suffix_scatter")?,
            ram_rw_cycle_round_pairs: module.load_function("ram_rw_cycle_round_pairs")?,
            ram_rw_cycle_bind: module.load_function("ram_rw_cycle_bind")?,
            u64_to_mont: module.load_function("u64_to_mont")?,
            ram_rw_address_round_pairs: module.load_function("ram_rw_address_round_pairs")?,
            ram_rw_address_bind: module.load_function("ram_rw_address_bind")?,
            rd_wa_gather: module.load_function("rd_wa_gather")?,
            add_scalar: module.load_function("add_scalar")?,
            dense_outer: module.load_function("dense_outer_kernel")?,
            dense_outer_fused: module.load_function("dense_outer_fused_kernel")?,
            row_dots: module.load_function("row_dots_kernel")?,
            cubic_pairs: module.load_function("cubic_pairs")?,
            cubic_tuple_reduce: module.load_function("cubic_tuple_reduce")?,
            round_poly_pairs: module.load_function("round_poly_pairs")?,
            eq_round_poly_pairs: module.load_function("eq_round_poly_pairs")?,
            round_poly_reduce: module.load_function("round_poly_reduce")?,
            dense_product_pairs: module.load_function("dense_product_pairs")?,
            gruen_round_poly_pairs: module.load_function("gruen_round_poly_pairs")?,
            uniskip_pairs: module.load_function("uniskip_pairs")?,
            gather8_materialize: module.load_function("gather8_materialize")?,
            core_booleanity_gather: module.load_function("core_booleanity_gather")?,
            core_booleanity_sparse_pairs: module
                .load_function("core_booleanity_sparse_pairs")?,
            core_booleanity_sparse_bind: module
                .load_function("core_booleanity_sparse_bind")?,
            core_booleanity_sparse_collapse8: module
                .load_function("core_booleanity_sparse_collapse8")?,
            hamming_pairs: module.load_function("hamming_pairs")?,
            hamming_booleanity_pairs: module.load_function("hamming_booleanity_pairs")?,
            ra_virtual_d4_pairs: module.load_function("ra_virtual_d4_pairs")?,
            ra_virtual_d4_sparse_pairs: module.load_function("ra_virtual_d4_sparse_pairs")?,
            ra_virtual_d4_sparse_bind: module.load_function("ra_virtual_d4_sparse_bind")?,
            ra_virtual_d4_sparse_collapse: module
                .load_function("ra_virtual_d4_sparse_collapse")?,
            bytecode_cycle_sparse_pairs: module
                .load_function("bytecode_cycle_sparse_pairs")?,
            core_booleanity_cycle_pairs: module.load_function("core_booleanity_cycle_pairs")?,
            core_booleanity_address_pairs: module
                .load_function("core_booleanity_address_pairs")?,
            sparse_register_round_pairs: module
                .load_function("sparse_register_round_pairs")?,
            sparse_register_bind_kernel: module
                .load_function("sparse_register_bind")?,
            instruction_raf_cycle_pairs: module
                .load_function("instruction_raf_cycle_pairs")?,
            instruction_raf_cycle_sparse_pairs: module
                .load_function("instruction_raf_cycle_sparse_pairs")?,
            instruction_raf_cycle_sparse_collapse: module
                .load_function("instruction_raf_cycle_sparse_collapse")?,
            sum_reduce: module.load_function("sum_reduce")?,
            product_reduce: module.load_function("product_reduce")?,
            stream,
            one_dev,
            staging: Arc::new(Mutex::new(PinnedPool::default())),
            resident_witness: Arc::new(Mutex::new(None)),
            resident_committed: Arc::new(Mutex::new(Vec::new())),
        })
    }

    pub fn upload(&self, values: &[Fr]) -> Result<DeviceFrVec, CudaError> {
        let buf = if values.is_empty() {
            self.stream.alloc_zeros(0)?
        } else {
            let n = values.len() * LIMBS;
            xfer_stats::add_h2d(n * std::mem::size_of::<u64>());
            xfer_stats::timed(xfer_stats::Phase::Upload, || {
                let mut pool = lock_pool(&self.staging);
                let staging = pool.ensure(self.stream.context(), n)?;
                for (slot, &v) in staging.as_mut_slice(n).chunks_exact_mut(LIMBS).zip(values) {
                    slot.copy_from_slice(&fr_to_limbs(v));
                }
                let dev = self.stream.clone_htod(staging.as_slice(n))?;
                self.stream.synchronize()?;
                Ok::<_, CudaError>(dev)
            })?
        };
        Ok(DeviceFrVec {
            stream: self.stream.clone(),
            buf,
            len: values.len(),
            staging: self.staging.clone(),
        })
    }

    pub fn upload_many(&self, factors: &[&[Fr]]) -> Result<Vec<DeviceFrVec>, CudaError> {
        if factors.is_empty() {
            return Ok(Vec::new());
        }
        let lens: Vec<usize> = factors.iter().map(|f| f.len()).collect();
        let total = lens.iter().sum::<usize>();
        if total == 0 {
            return factors
                .iter()
                .map(|_| {
                    Ok(DeviceFrVec {
                        stream: self.stream.clone(),
                        buf: self.stream.alloc_zeros(0)?,
                        len: 0,
                        staging: self.staging.clone(),
                    })
                })
                .collect();
        }

        let n = total * LIMBS;
        xfer_stats::add_h2d(n * std::mem::size_of::<u64>());
        let packed = xfer_stats::timed(xfer_stats::Phase::Upload, || {
            let mut pool = lock_pool(&self.staging);
            let staging = pool.ensure(self.stream.context(), n)?;
            let slots = staging.as_mut_slice(n);
            let mut offset = 0;
            for factor in factors {
                for (slot, &v) in slots[offset..offset + factor.len() * LIMBS]
                    .chunks_exact_mut(LIMBS)
                    .zip(*factor)
                {
                    slot.copy_from_slice(&fr_to_limbs(v));
                }
                offset += factor.len() * LIMBS;
            }
            let dev = self.stream.clone_htod(staging.as_slice(n))?;
            self.stream.synchronize()?;
            Ok::<_, CudaError>(dev)
        })?;

        let mut out = Vec::with_capacity(factors.len());
        let mut offset = 0;
        for &len in &lens {
            if len == 0 {
                out.push(DeviceFrVec {
                    stream: self.stream.clone(),
                    buf: self.stream.alloc_zeros(0)?,
                    len: 0,
                    staging: self.staging.clone(),
                });
                continue;
            }
            let mut buf: CudaSlice<u64> = self.stream.alloc_zeros(len * LIMBS)?;
            self.stream
                .memcpy_dtod(&packed.slice(offset..offset + len * LIMBS), &mut buf)?;
            out.push(DeviceFrVec {
                stream: self.stream.clone(),
                buf,
                len,
                staging: self.staging.clone(),
            });
            offset += len * LIMBS;
        }
        self.stream.synchronize()?;
        Ok(out)
    }

    fn clone_htod_tracked<T: cudarc::driver::DeviceRepr>(
        &self,
        values: &[T],
    ) -> Result<CudaSlice<T>, CudaError> {
        if !xfer_stats::enabled() {
            return Ok(self.stream.clone_htod(values)?);
        }
        let bytes = std::mem::size_of_val(values);
        let start = std::time::Instant::now();
        let dev = self.stream.clone_htod(values)?;
        xfer_stats::add_h2d_raw(bytes, start.elapsed().as_nanos() as u64);
        Ok(dev)
    }

    pub fn upload_u64_slice(&self, values: &[u64]) -> Result<CudaSlice<u64>, CudaError> {
        self.clone_htod_tracked(values)
    }

    pub fn upload_u32_slice(&self, values: &[u32]) -> Result<CudaSlice<u32>, CudaError> {
        self.clone_htod_tracked(values)
    }

    pub fn upload_i32_slice(&self, values: &[i32]) -> Result<CudaSlice<i32>, CudaError> {
        self.clone_htod_tracked(values)
    }

    pub fn upload_u8_slice(&self, values: &[u8]) -> Result<CudaSlice<u8>, CudaError> {
        self.clone_htod_tracked(values)
    }

    pub fn upload_i16_slice(&self, values: &[i16]) -> Result<CudaSlice<i16>, CudaError> {
        self.clone_htod_tracked(values)
    }

    pub fn upload_u16_slice(&self, values: &[u16]) -> Result<CudaSlice<u16>, CudaError> {
        self.clone_htod_tracked(values)
    }

    pub fn download_u64(&self, buf: &CudaSlice<u64>) -> Result<Vec<u64>, CudaError> {
        let out = self.stream.clone_dtoh(buf)?;
        self.stream.synchronize()?;
        Ok(out)
    }

    fn factor_ptr_array(&self, factors: &[&DeviceFrVec]) -> Result<CudaSlice<u64>, CudaError> {
        use cudarc::driver::DevicePtr;
        let mut ptrs = Vec::with_capacity(factors.len());
        let mut guards = Vec::with_capacity(factors.len());
        for factor in factors {
            let (ptr, guard) = factor.buf.device_ptr(&self.stream);
            ptrs.push(ptr);
            guards.push(guard);
        }
        let array = self.clone_htod_tracked(&ptrs)?;
        self.stream.synchronize()?;
        drop(guards);
        Ok(array)
    }

    pub fn resident_witness(&self, witness: &[Fr]) -> Result<Arc<DeviceFrVec>, CudaError> {
        let key = WitnessKey::of(witness);
        let mut cache = lock_resident(&self.resident_witness);
        if let Some(resident) = cache.as_ref() {
            if resident.key == key {
                return Ok(resident.buf.clone());
            }
        }
        let buf = Arc::new(self.upload(witness)?);
        *cache = Some(ResidentWitness {
            key,
            buf: buf.clone(),
        });
        Ok(buf)
    }

    pub fn resident_committed_clone(&self, poly: &[Fr]) -> Result<DeviceFrVec, CudaError> {
        let key = WitnessKey::of(poly);
        let mut cache = lock_committed(&self.resident_committed);
        if let Some(resident) = cache.iter().find(|entry| entry.key == key) {
            return resident.buf.try_clone();
        }
        let buf = Arc::new(self.upload(poly)?);
        let clone = buf.try_clone()?;
        if cache.len() >= COMMITTED_CACHE_CAP {
            let _ = cache.remove(0);
        }
        cache.push(ResidentWitness { key, buf });
        Ok(clone)
    }

    #[expect(clippy::too_many_arguments)]
    pub fn dense_outer_construct(
        &self,
        eq_evals: &[Fr],
        scale: Fr,
        weights: &[Fr],
        row_dots_a: &[Fr],
        row_dots_b: &[Fr],
        row_count: usize,
        first_group_rows: &[u32],
        second_group_rows: &[u32],
    ) -> Result<(DeviceFrVec, DeviceFrVec, DeviceFrVec), CudaError> {
        let len = eq_evals.len();
        let cycles = len / 2;

        let mut eq: CudaSlice<u64> = self.stream.alloc_zeros(len * LIMBS)?;
        let mut az: CudaSlice<u64> = self.stream.alloc_zeros(len * LIMBS)?;
        let mut bz: CudaSlice<u64> = self.stream.alloc_zeros(len * LIMBS)?;

        if cycles > 0 {
            let eq_evals = self.upload(eq_evals)?;
            let scale = self.upload(&[scale])?;
            let weights = self.upload(weights)?;
            let row_dots_a = self.upload(row_dots_a)?;
            let row_dots_b = self.upload(row_dots_b)?;
            let first = self.clone_htod_tracked(first_group_rows)?;
            let second = self.clone_htod_tracked(second_group_rows)?;

            let first_len = first_group_rows.len() as u32;
            let second_len = second_group_rows.len() as u32;
            let row_count_arg = row_count as u64;
            let cycles_arg = cycles as u64;
            let cfg = LaunchConfig {
                grid_dim: ((cycles as u32).div_ceil(BLOCK), 1, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: 0,
            };
            let f = self.dense_outer.clone();
            let mut launch = self.stream.launch_builder(&f);
            let _ = launch
                .arg(&mut eq)
                .arg(&mut az)
                .arg(&mut bz)
                .arg(&eq_evals.buf)
                .arg(&scale.buf)
                .arg(&weights.buf)
                .arg(&row_dots_a.buf)
                .arg(&row_dots_b.buf)
                .arg(&first)
                .arg(&second)
                .arg(&first_len)
                .arg(&second_len)
                .arg(&row_count_arg)
                .arg(&cycles_arg);
            // SAFETY: one thread per cycle reads eq_evals[2c..2c+2], the scale, the
            // weights, and the per-cycle row_count dots from a/b, writing the two
            // output pairs at 2c/2c+1 in eq/az/bz. All device buffers are sized for
            // `len` field elements (cycles * row_count for the dot tables).
            let _ = unsafe { launch.launch(cfg) }?;
            self.stream.synchronize()?;
        }

        let make = |buf, len| DeviceFrVec {
            stream: self.stream.clone(),
            buf,
            len,
            staging: self.staging.clone(),
        };
        Ok((make(eq, len), make(az, len), make(bz, len)))
    }

    pub fn dense_outer_fused(
        &self,
        inputs: FusedOuterInputs<'_>,
    ) -> Result<(DeviceFrVec, DeviceFrVec, DeviceFrVec), CudaError> {
        let len = inputs.eq_evals.len();
        let cycles = len / 2;

        let mut eq: CudaSlice<u64> = self.stream.alloc_zeros(len * LIMBS)?;
        let mut az: CudaSlice<u64> = self.stream.alloc_zeros(len * LIMBS)?;
        let mut bz: CudaSlice<u64> = self.stream.alloc_zeros(len * LIMBS)?;

        if cycles > 0 {
            let eq_evals = self.upload(inputs.eq_evals)?;
            let scale = self.upload(&[inputs.scale])?;
            let witness = self.resident_witness(inputs.witness)?;
            let a_coeffs = self.upload(inputs.a_coeffs)?;
            let b_coeffs = self.upload(inputs.b_coeffs)?;
            let a_offsets = self.clone_htod_tracked(inputs.a_offsets)?;
            let a_vars = self.clone_htod_tracked(inputs.a_vars)?;
            let b_offsets = self.clone_htod_tracked(inputs.b_offsets)?;
            let b_vars = self.clone_htod_tracked(inputs.b_vars)?;

            let split = inputs.split as u32;
            let total_entries = (inputs.a_offsets.len() - 1) as u32;
            let num_vars_padded_arg = inputs.num_vars_padded as u64;
            let cycles_arg = cycles as u64;
            let cfg = LaunchConfig {
                grid_dim: ((cycles as u32).div_ceil(BLOCK), 1, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: 0,
            };
            let f = self.dense_outer_fused.clone();
            let mut launch = self.stream.launch_builder(&f);
            let _ = launch
                .arg(&mut eq)
                .arg(&mut az)
                .arg(&mut bz)
                .arg(&eq_evals.buf)
                .arg(&scale.buf)
                .arg(&witness.buf)
                .arg(&a_offsets)
                .arg(&a_vars)
                .arg(&a_coeffs.buf)
                .arg(&b_offsets)
                .arg(&b_vars)
                .arg(&b_coeffs.buf)
                .arg(&split)
                .arg(&total_entries)
                .arg(&num_vars_padded_arg)
                .arg(&cycles_arg);
            // SAFETY: one thread per cycle reads eq_evals[2c..2c+2], the scale, its
            // witness slice, and the two weighted-CSR groups (entries 0..split and
            // split..total), writing the output pairs at 2c/2c+1 in eq/az/bz.
            let _ = unsafe { launch.launch(cfg) }?;
            self.stream.synchronize()?;
        }

        let make = |buf, len| DeviceFrVec {
            stream: self.stream.clone(),
            buf,
            len,
            staging: self.staging.clone(),
        };
        Ok((make(eq, len), make(az, len), make(bz, len)))
    }

    #[expect(clippy::too_many_arguments)]
    pub fn compute_row_dots_device(
        &self,
        witness: &DeviceFrVec,
        a_offsets: &[u32],
        a_vars: &[u32],
        a_coeffs: &[Fr],
        b_offsets: &[u32],
        b_vars: &[u32],
        b_coeffs: &[Fr],
        row_count: usize,
        num_vars_padded: usize,
        num_cycles: usize,
    ) -> Result<(DeviceFrVec, DeviceFrVec), CudaError> {
        let total = num_cycles * row_count;
        let mut a_out: CudaSlice<u64> = self.stream.alloc_zeros(total * LIMBS)?;
        let mut b_out: CudaSlice<u64> = self.stream.alloc_zeros(total * LIMBS)?;

        if total > 0 {
            let a_coeffs_dev = self.upload(a_coeffs)?;
            let b_coeffs_dev = self.upload(b_coeffs)?;
            let a_offsets_dev = self.clone_htod_tracked(a_offsets)?;
            let a_vars_dev = self.clone_htod_tracked(a_vars)?;
            let b_offsets_dev = self.clone_htod_tracked(b_offsets)?;
            let b_vars_dev = self.clone_htod_tracked(b_vars)?;

            let row_count_arg = row_count as u64;
            let num_vars_padded_arg = num_vars_padded as u64;
            let total_arg = total as u64;
            let cfg = LaunchConfig {
                grid_dim: ((total as u32).div_ceil(BLOCK), 1, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: 0,
            };
            let f = self.row_dots.clone();
            let mut launch = self.stream.launch_builder(&f);
            let _ = launch
                .arg(&mut a_out)
                .arg(&mut b_out)
                .arg(&witness.buf)
                .arg(&a_offsets_dev)
                .arg(&a_vars_dev)
                .arg(&a_coeffs_dev.buf)
                .arg(&b_offsets_dev)
                .arg(&b_vars_dev)
                .arg(&b_coeffs_dev.buf)
                .arg(&row_count_arg)
                .arg(&num_vars_padded_arg)
                .arg(&total_arg);
            // SAFETY: one thread per (cycle, row) of `total` reads its CSR row's
            // nonzeros (bounded by the offsets) from the coeff/var arrays and the
            // matching witness slice, writing one element to a_out/b_out (total each).
            let _ = unsafe { launch.launch(cfg) }?;
            self.stream.synchronize()?;
        }

        let make = |buf, len| DeviceFrVec {
            stream: self.stream.clone(),
            buf,
            len,
            staging: self.staging.clone(),
        };
        Ok((make(a_out, total), make(b_out, total)))
    }

    #[expect(clippy::too_many_arguments)]
    pub fn compute_row_dots(
        &self,
        witness: &[Fr],
        a_offsets: &[u32],
        a_vars: &[u32],
        a_coeffs: &[Fr],
        b_offsets: &[u32],
        b_vars: &[u32],
        b_coeffs: &[Fr],
        row_count: usize,
        num_vars_padded: usize,
        num_cycles: usize,
    ) -> Result<(Vec<Fr>, Vec<Fr>), CudaError> {
        let total = num_cycles * row_count;
        if total == 0 {
            return Ok((Vec::new(), Vec::new()));
        }
        let witness_dev = self.upload(witness)?;
        let (a, b) = self.compute_row_dots_device(
            &witness_dev,
            a_offsets,
            a_vars,
            a_coeffs,
            b_offsets,
            b_vars,
            b_coeffs,
            row_count,
            num_vars_padded,
            num_cycles,
        )?;
        Ok((a.to_host()?, b.to_host()?))
    }

    pub fn ra_virtual_d4_round_poly(
        &self,
        inputs: RaVirtualD4Inputs<'_>,
    ) -> Result<[Fr; 4], CudaError> {
        use num_traits::Zero;
        let virtual_count = inputs.gamma_powers.len();
        assert!(virtual_count > 0, "ra-virtual d4 needs at least one virtual");
        assert_eq!(
            inputs.chunks.len(),
            virtual_count * 4,
            "ra-virtual d4 needs 4 chunks per virtual"
        );
        let half = inputs.chunks[0].len() / 2;
        for chunk in inputs.chunks {
            assert_eq!(chunk.len(), half * 2, "ra-virtual d4 chunks must have equal length");
        }
        assert!(inputs.e_in.len().is_power_of_two(), "e_in length must be a power of two");
        assert_eq!(
            inputs.e_in.len() * inputs.e_out.len(),
            half,
            "e_in * e_out must equal the chunk row count"
        );
        if half == 0 {
            return Ok([Fr::zero(); 4]);
        }
        let in_bits = inputs.e_in.len().trailing_zeros();

        let factor_ptrs = self.factor_ptr_array(inputs.chunks)?;

        const WIDTH: usize = 4;
        let tuple = WIDTH * LIMBS;
        let block = BLOCK;
        let shared = block * tuple as u32 * std::mem::size_of::<u64>() as u32;
        let blocks = (half as u32).div_ceil(block);
        let mut buf: CudaSlice<u64> = self.stream.alloc_zeros(blocks as usize * tuple)?;

        let pair_stride_arg = half as u64;
        let virtual_count_arg = virtual_count as u32;
        let half_arg = half as u64;
        let in_bits_arg = in_bits;
        let width_arg = WIDTH as u32;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: shared,
        };
        let f = self.ra_virtual_d4_pairs.clone();
        let mut launch = self.stream.launch_builder(&f);
        let _ = launch
            .arg(&mut buf)
            .arg(&factor_ptrs)
            .arg(&inputs.gamma_powers.buf)
            .arg(&inputs.e_in.buf)
            .arg(&inputs.e_out.buf)
            .arg(&pair_stride_arg)
            .arg(&virtual_count_arg)
            .arg(&half_arg)
            .arg(&in_bits_arg);
        // SAFETY: one thread per row reads its pair from each of the 4*virtual_count
        // packed chunks, the per-virtual gamma, and its e_in[row & mask] /
        // e_out[row >> in_bits] weights (in range since e_in*e_out == half), writing
        // one 4-lane tuple per block; shared memory holds `block` tuples.
        let _ = unsafe { launch.launch(cfg) }?;

        let mut len = blocks as usize;
        while len > 1 {
            let out_blocks = (len as u32).div_ceil(block);
            let mut out: CudaSlice<u64> = self.stream.alloc_zeros(out_blocks as usize * tuple)?;
            let len_arg = len as u64;
            let cfg = LaunchConfig {
                grid_dim: (out_blocks, 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: shared,
            };
            let f = self.round_poly_reduce.clone();
            let mut launch = self.stream.launch_builder(&f);
            let _ = launch.arg(&mut out).arg(&buf).arg(&width_arg).arg(&len_arg);
            // SAFETY: round_poly_reduce sums 4-lane tuples across up to `block`
            // blocks; shared memory holds `block` tuples.
            let _ = unsafe { launch.launch(cfg) }?;
            buf = out;
            len = out_blocks as usize;
        }

        let raw = self.stream.clone_dtoh(&buf.slice(0..tuple))?;
        self.stream.synchronize()?;
        Ok(core::array::from_fn(|e| {
            limbs_to_fr([raw[e * 4], raw[e * 4 + 1], raw[e * 4 + 2], raw[e * 4 + 3]])
        }))
    }

    pub fn ra_virtual_d4_sparse_round_poly(
        &self,
        inputs: RaVirtualD4SparseInputs<'_>,
    ) -> Result<[Fr; 4], CudaError> {
        use num_traits::Zero;
        let virtual_count = inputs.gamma_powers.len();
        let round = inputs.round;
        assert!((1..=3).contains(&round), "sparse round in 1..=3");
        assert!(virtual_count > 0, "ra-virtual d4 needs at least one virtual");
        assert_eq!(inputs.num_chunks, virtual_count * 4, "4 chunks per virtual");
        assert_eq!(
            inputs.values.len(),
            inputs.num_chunks * inputs.source_rows,
            "values shape (chunk-major)",
        );
        assert!(
            inputs.source_rows.is_multiple_of(1usize << round),
            "source rows split into 2^round groups per output",
        );
        let half = inputs.source_rows >> round;
        assert!(inputs.e_in.len().is_power_of_two(), "e_in length must be a power of two");
        assert_eq!(inputs.e_in.len() * inputs.e_out.len(), half, "e_in * e_out == half");
        if half == 0 {
            return Ok([Fr::zero(); 4]);
        }
        let in_bits = inputs.e_in.len().trailing_zeros();

        const WIDTH: usize = 4;
        let tuple = WIDTH * LIMBS;
        let block = BLOCK;
        let shared = block * tuple as u32 * std::mem::size_of::<u64>() as u32;
        let blocks = (half as u32).div_ceil(block);
        let mut buf: CudaSlice<u64> = self.stream.alloc_zeros(blocks as usize * tuple)?;

        let num_chunks_arg = inputs.num_chunks as u64;
        let chunk_domain_arg = inputs.chunk_domain as u64;
        let source_rows_arg = inputs.source_rows as u64;
        let virtual_count_arg = virtual_count as u32;
        let half_arg = half as u64;
        let in_bits_arg = in_bits;
        let round_arg = round;
        let width_arg = WIDTH as u32;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: shared,
        };
        let f = self.ra_virtual_d4_sparse_pairs.clone();
        let mut launch = self.stream.launch_builder(&f);
        let _ = launch
            .arg(&mut buf)
            .arg(inputs.tables)
            .arg(inputs.values)
            .arg(&inputs.gamma_powers.buf)
            .arg(&inputs.e_in.buf)
            .arg(&inputs.e_out.buf)
            .arg(&num_chunks_arg)
            .arg(&chunk_domain_arg)
            .arg(&source_rows_arg)
            .arg(&virtual_count_arg)
            .arg(&half_arg)
            .arg(&in_bits_arg)
            .arg(&round_arg);
        // SAFETY: one thread per output row (of `half`) gathers, for each of the 4 chunks
        // per virtual, a (lo,hi) pair by summing 2^(round-1) table-sets indexed via the
        // resident chunk-major `values` (i16; -1 = absent) at source base=row<<round, runs
        // the d4 product weighted by gamma/e_in/e_out, and writes one 4-lane tuple per block.
        let _ = unsafe { launch.launch(cfg) }?;

        let mut len = blocks as usize;
        while len > 1 {
            let out_blocks = (len as u32).div_ceil(block);
            let mut out: CudaSlice<u64> = self.stream.alloc_zeros(out_blocks as usize * tuple)?;
            let len_arg = len as u64;
            let cfg = LaunchConfig {
                grid_dim: (out_blocks, 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: shared,
            };
            let f = self.round_poly_reduce.clone();
            let mut launch = self.stream.launch_builder(&f);
            let _ = launch.arg(&mut out).arg(&buf).arg(&width_arg).arg(&len_arg);
            // SAFETY: round_poly_reduce sums 4-lane tuples across up to `block` blocks.
            let _ = unsafe { launch.launch(cfg) }?;
            buf = out;
            len = out_blocks as usize;
        }

        let raw = self.stream.clone_dtoh(&buf.slice(0..tuple))?;
        self.stream.synchronize()?;
        Ok(core::array::from_fn(|e| {
            limbs_to_fr([raw[e * 4], raw[e * 4 + 1], raw[e * 4 + 2], raw[e * 4 + 3]])
        }))
    }

    pub fn ra_virtual_d4_sparse_bind(
        &self,
        tables: &CudaSlice<u64>,
        num_sets: usize,
        set_elems: usize,
        challenge: Fr,
    ) -> Result<CudaSlice<u64>, CudaError> {
        let total = num_sets * set_elems;
        let mut out: CudaSlice<u64> = self.stream.alloc_zeros(2 * total.max(1) * LIMBS)?;
        if total == 0 {
            return Ok(out);
        }
        let challenge_dev = self.upload(&[challenge])?;
        let one_minus_dev = self.upload(&[<Fr as num_traits::One>::one() - challenge])?;
        let set_elems_arg = set_elems as u64;
        let num_sets_arg = num_sets as u64;
        let cfg = LaunchConfig {
            grid_dim: ((total as u32).div_ceil(BLOCK), 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let f = self.ra_virtual_d4_sparse_bind.clone();
        let mut launch = self.stream.launch_builder(&f);
        let _ = launch
            .arg(&mut out)
            .arg(tables)
            .arg(&challenge_dev.buf)
            .arg(&one_minus_dev.buf)
            .arg(&set_elems_arg)
            .arg(&num_sets_arg);
        // SAFETY: one thread per input element i writes out[i]=(1-c)*in[i] and
        // out[total+i]=c*in[i]; `out` sized for 2*total. No shared memory.
        let _ = unsafe { launch.launch(cfg) }?;
        self.stream.synchronize()?;
        Ok(out)
    }

    pub fn ra_virtual_d4_sparse_collapse(
        &self,
        tables: &CudaSlice<u64>,
        values: &CudaSlice<i16>,
        num_chunks: usize,
        chunk_domain: usize,
        source_rows: usize,
        out_len: usize,
    ) -> Result<Vec<DeviceFrVec>, CudaError> {
        let total = num_chunks * out_len;
        if total == 0 {
            return (0..num_chunks)
                .map(|_| {
                    Ok(DeviceFrVec {
                        stream: self.stream.clone(),
                        buf: self.stream.alloc_zeros(0)?,
                        len: 0,
                        staging: self.staging.clone(),
                    })
                })
                .collect();
        }
        let mut out: CudaSlice<u64> = self.stream.alloc_zeros(total * LIMBS)?;
        let num_chunks_arg = num_chunks as u64;
        let chunk_domain_arg = chunk_domain as u64;
        let source_rows_arg = source_rows as u64;
        let out_len_arg = out_len as u64;
        let cfg = LaunchConfig {
            grid_dim: ((total as u32).div_ceil(BLOCK), 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let f = self.ra_virtual_d4_sparse_collapse.clone();
        let mut launch = self.stream.launch_builder(&f);
        let _ = launch
            .arg(&mut out)
            .arg(tables)
            .arg(values)
            .arg(&num_chunks_arg)
            .arg(&chunk_domain_arg)
            .arg(&source_rows_arg)
            .arg(&out_len_arg);
        // SAFETY: one thread per (chunk, row) sums the 8 round-3 table-sets gathered via the
        // chunk-major `values` at source 8j+set; `out` sized for num_chunks*out_len. No shared.
        let _ = unsafe { launch.launch(cfg) }?;
        self.stream.synchronize()?;

        let mut chunks = Vec::with_capacity(num_chunks);
        for i in 0..num_chunks {
            let mut buf: CudaSlice<u64> = self.stream.alloc_zeros(out_len * LIMBS)?;
            self.stream.memcpy_dtod(
                &out.slice(i * out_len * LIMBS..(i + 1) * out_len * LIMBS),
                &mut buf,
            )?;
            chunks.push(DeviceFrVec {
                stream: self.stream.clone(),
                buf,
                len: out_len,
                staging: self.staging.clone(),
            });
        }
        self.stream.synchronize()?;
        Ok(chunks)
    }

    pub fn bytecode_cycle_sparse_round_poly(
        &self,
        inputs: BytecodeCycleSparseInputs<'_>,
    ) -> Result<Vec<Fr>, CudaError> {
        use num_traits::Zero;
        let round = inputs.round;
        assert!((1..=3).contains(&round), "sparse round in 1..=3");
        let degree = inputs.degree;
        assert!((2..=8).contains(&degree), "bytecode cycle degree in 2..=8");
        assert_eq!(
            inputs.values.len(),
            inputs.num_chunks * inputs.source_rows,
            "values shape (chunk-major)",
        );
        assert!(
            inputs.source_rows.is_multiple_of(1usize << round),
            "source rows split into 2^round groups per output",
        );
        let half = inputs.source_rows >> round;
        assert_eq!(inputs.combined_eq.len(), half * 2, "combined_eq length == 2*half");
        if half == 0 {
            return Ok(vec![Fr::zero(); degree]);
        }

        let points: Vec<Fr> = (0..degree)
            .map(|p| <Fr as jolt_field::Field>::from_u64(if p == 0 { 0 } else { (p + 1) as u64 }))
            .collect();
        let points_dev = self.upload(&points)?;

        let tuple = degree * LIMBS;
        let max_block = (48 * 1024 / (tuple * std::mem::size_of::<u64>())).max(1) as u32;
        let capped = BLOCK.min(max_block);
        let block = 1u32 << (u32::BITS - 1 - capped.leading_zeros());
        let shared = block * tuple as u32 * std::mem::size_of::<u64>() as u32;
        let blocks = (half as u32).div_ceil(block);
        let mut buf: CudaSlice<u64> = self.stream.alloc_zeros(blocks as usize * tuple)?;

        let num_chunks_arg = inputs.num_chunks as u64;
        let chunk_domain_arg = inputs.chunk_domain as u64;
        let source_rows_arg = inputs.source_rows as u64;
        let half_arg = half as u64;
        let degree_arg = degree as u32;
        let round_arg = round;
        let width_arg = degree as u32;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: shared,
        };
        let f = self.bytecode_cycle_sparse_pairs.clone();
        let mut launch = self.stream.launch_builder(&f);
        let _ = launch
            .arg(&mut buf)
            .arg(inputs.tables)
            .arg(inputs.values)
            .arg(&inputs.combined_eq.buf)
            .arg(&points_dev.buf)
            .arg(&num_chunks_arg)
            .arg(&chunk_domain_arg)
            .arg(&source_rows_arg)
            .arg(&half_arg)
            .arg(&degree_arg)
            .arg(&round_arg);
        // SAFETY: one thread per output row (of `half`) gathers each chunk's (lo,hi) pair by
        // summing 2^(round-1) table-sets via the chunk-major `values` at source base=row<<round,
        // forms ra_product = prod_chunk (lo + slope*x) times (combined_eq low+slope*x) at each of
        // `degree` points, writing one degree-lane tuple per block; shared holds `block` tuples.
        let _ = unsafe { launch.launch(cfg) }?;

        let mut len = blocks as usize;
        while len > 1 {
            let out_blocks = (len as u32).div_ceil(block);
            let mut out: CudaSlice<u64> = self.stream.alloc_zeros(out_blocks as usize * tuple)?;
            let len_arg = len as u64;
            let cfg = LaunchConfig {
                grid_dim: (out_blocks, 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: shared,
            };
            let f = self.round_poly_reduce.clone();
            let mut launch = self.stream.launch_builder(&f);
            let _ = launch.arg(&mut out).arg(&buf).arg(&width_arg).arg(&len_arg);
            // SAFETY: round_poly_reduce sums degree-lane tuples across up to `block` blocks.
            let _ = unsafe { launch.launch(cfg) }?;
            buf = out;
            len = out_blocks as usize;
        }

        let raw = self.stream.clone_dtoh(&buf.slice(0..tuple))?;
        self.stream.synchronize()?;
        Ok((0..degree)
            .map(|e| limbs_to_fr([raw[e * 4], raw[e * 4 + 1], raw[e * 4 + 2], raw[e * 4 + 3]]))
            .collect())
    }

    pub fn core_booleanity_cycle_round_poly(
        &self,
        inputs: CoreBooleanityCycleInputs<'_>,
    ) -> Result<[Fr; 2], CudaError> {
        use num_traits::Zero;
        let num_polys = inputs.h_polys.len();
        assert!(num_polys > 0, "core booleanity cycle needs at least one h poly");
        assert_eq!(inputs.rho.len(), num_polys);
        let half = inputs.h_polys[0].len() / 2;
        for h in inputs.h_polys {
            assert_eq!(h.len(), half * 2, "core booleanity h polys must have equal length");
        }
        assert!(inputs.e_in.len().is_power_of_two(), "e_in length must be a power of two");
        assert_eq!(
            inputs.e_in.len() * inputs.e_out.len(),
            half,
            "e_in * e_out must equal the h row count"
        );
        if half == 0 {
            return Ok([Fr::zero(); 2]);
        }
        let in_bits = inputs.e_in.len().trailing_zeros();

        let factor_ptrs = self.factor_ptr_array(inputs.h_polys)?;

        const WIDTH: usize = 2;
        let tuple = WIDTH * LIMBS;
        let block = BLOCK;
        let shared = block * tuple as u32 * std::mem::size_of::<u64>() as u32;
        let blocks = (half as u32).div_ceil(block);
        let mut buf: CudaSlice<u64> = self.stream.alloc_zeros(blocks as usize * tuple)?;

        let pair_stride_arg = half as u64;
        let num_polys_arg = num_polys as u32;
        let half_arg = half as u64;
        let in_bits_arg = in_bits;
        let width_arg = WIDTH as u32;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: shared,
        };
        let f = self.core_booleanity_cycle_pairs.clone();
        let mut launch = self.stream.launch_builder(&f);
        let _ = launch
            .arg(&mut buf)
            .arg(&factor_ptrs)
            .arg(&inputs.rho.buf)
            .arg(&inputs.e_in.buf)
            .arg(&inputs.e_out.buf)
            .arg(&pair_stride_arg)
            .arg(&num_polys_arg)
            .arg(&half_arg)
            .arg(&in_bits_arg);
        // SAFETY: one thread per row reads its pair from each of the num_polys packed
        // h polys, the per-poly rho, and its e_in[row & mask] / e_out[row >> in_bits]
        // weights (in range since e_in*e_out == half), writing one 2-lane tuple per
        // block; shared memory holds `block` tuples.
        let _ = unsafe { launch.launch(cfg) }?;

        let mut len = blocks as usize;
        while len > 1 {
            let out_blocks = (len as u32).div_ceil(block);
            let mut out: CudaSlice<u64> = self.stream.alloc_zeros(out_blocks as usize * tuple)?;
            let len_arg = len as u64;
            let cfg = LaunchConfig {
                grid_dim: (out_blocks, 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: shared,
            };
            let f = self.round_poly_reduce.clone();
            let mut launch = self.stream.launch_builder(&f);
            let _ = launch.arg(&mut out).arg(&buf).arg(&width_arg).arg(&len_arg);
            // SAFETY: round_poly_reduce sums 2-lane tuples across up to `block`
            // blocks; shared memory holds `block` tuples.
            let _ = unsafe { launch.launch(cfg) }?;
            buf = out;
            len = out_blocks as usize;
        }

        let raw = self.stream.clone_dtoh(&buf.slice(0..tuple))?;
        self.stream.synchronize()?;
        Ok([
            limbs_to_fr([raw[0], raw[1], raw[2], raw[3]]),
            limbs_to_fr([raw[4], raw[5], raw[6], raw[7]]),
        ])
    }

    pub fn core_booleanity_sparse_round_poly(
        &self,
        inputs: CoreBooleanitySparseInputs<'_>,
    ) -> Result<[Fr; 2], CudaError> {
        use num_traits::Zero;
        let num_polys = inputs.num_polys;
        let round = inputs.round;
        assert!((1..=3).contains(&round), "sparse round in 1..=3");
        assert!(num_polys > 0, "core booleanity sparse needs at least one poly");
        assert_eq!(inputs.rho.len(), num_polys);
        assert!(num_polys <= inputs.poly_stride, "num_polys within row stride");
        let source_rows = inputs.source_rows;
        assert!(
            source_rows.is_multiple_of(1usize << round),
            "source rows split into 2^round groups per output",
        );
        let half = source_rows >> round;
        assert!(inputs.e_in.len().is_power_of_two(), "e_in length must be a power of two");
        assert_eq!(
            inputs.e_in.len() * inputs.e_out.len(),
            half,
            "e_in * e_out must equal the output row count",
        );
        if half == 0 {
            return Ok([Fr::zero(); 2]);
        }
        let in_bits = inputs.e_in.len().trailing_zeros();


        const WIDTH: usize = 2;
        let tuple = WIDTH * LIMBS;
        let block = BLOCK;
        let shared = block * tuple as u32 * std::mem::size_of::<u64>() as u32;
        let blocks = (half as u32).div_ceil(block);
        let mut buf: CudaSlice<u64> = self.stream.alloc_zeros(blocks as usize * tuple)?;

        let num_polys_arg = num_polys as u64;
        let chunk_domain_arg = inputs.chunk_domain as u64;
        let poly_stride_arg = inputs.poly_stride as u64;
        let half_arg = half as u64;
        let in_bits_arg = in_bits;
        let round_arg = round;
        let width_arg = WIDTH as u32;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: shared,
        };
        let f = self.core_booleanity_sparse_pairs.clone();
        let mut launch = self.stream.launch_builder(&f);
        let _ = launch
            .arg(&mut buf)
            .arg(inputs.tables)
            .arg(inputs.present_mask)
            .arg(inputs.values)
            .arg(&inputs.rho.buf)
            .arg(&inputs.e_in.buf)
            .arg(&inputs.e_out.buf)
            .arg(&num_polys_arg)
            .arg(&chunk_domain_arg)
            .arg(&poly_stride_arg)
            .arg(&half_arg)
            .arg(&in_bits_arg)
            .arg(&round_arg);
        // SAFETY: one thread per output row (of `half`) gathers (h0,h1) for each of num_polys
        // polys by summing 2^(round-1) table-sets, indexed via the resident present_mask /
        // values at source offsets base=row<<round (all in-range for source_rows), applies the
        // booleanity body weighted by e_in[row & mask] / e_out[row >> in_bits], and writes one
        // 2-lane tuple per block; shared memory holds `block` tuples.
        let _ = unsafe { launch.launch(cfg) }?;

        let mut len = blocks as usize;
        while len > 1 {
            let out_blocks = (len as u32).div_ceil(block);
            let mut out: CudaSlice<u64> = self.stream.alloc_zeros(out_blocks as usize * tuple)?;
            let len_arg = len as u64;
            let cfg = LaunchConfig {
                grid_dim: (out_blocks, 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: shared,
            };
            let f = self.round_poly_reduce.clone();
            let mut launch = self.stream.launch_builder(&f);
            let _ = launch.arg(&mut out).arg(&buf).arg(&width_arg).arg(&len_arg);
            // SAFETY: round_poly_reduce sums 2-lane tuples across up to `block` blocks;
            // shared memory holds `block` tuples.
            let _ = unsafe { launch.launch(cfg) }?;
            buf = out;
            len = out_blocks as usize;
        }

        let raw = self.stream.clone_dtoh(&buf.slice(0..tuple))?;
        self.stream.synchronize()?;
        Ok([
            limbs_to_fr([raw[0], raw[1], raw[2], raw[3]]),
            limbs_to_fr([raw[4], raw[5], raw[6], raw[7]]),
        ])
    }

    pub fn core_booleanity_sparse_bind(
        &self,
        tables: &CudaSlice<u64>,
        num_sets: usize,
        set_elems: usize,
        challenge: Fr,
    ) -> Result<CudaSlice<u64>, CudaError> {
        let total = num_sets * set_elems;
        let mut out: CudaSlice<u64> = self.stream.alloc_zeros(2 * total.max(1) * LIMBS)?;
        if total == 0 {
            return Ok(out);
        }
        let challenge_dev = self.upload(&[challenge])?;
        let one_minus_dev = self.upload(&[<Fr as num_traits::One>::one() - challenge])?;

        let set_elems_arg = set_elems as u64;
        let num_sets_arg = num_sets as u64;
        let cfg = LaunchConfig {
            grid_dim: ((total as u32).div_ceil(BLOCK), 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let f = self.core_booleanity_sparse_bind.clone();
        let mut launch = self.stream.launch_builder(&f);
        let _ = launch
            .arg(&mut out)
            .arg(tables)
            .arg(&challenge_dev.buf)
            .arg(&one_minus_dev.buf)
            .arg(&set_elems_arg)
            .arg(&num_sets_arg);
        // SAFETY: one thread per element i of the `total` input elements writes
        // out[i]=(1-c)*in[i] and out[total+i]=c*in[i]; `out` is sized for 2*total elements
        // and `in`/challenge are read-only. No shared memory.
        let _ = unsafe { launch.launch(cfg) }?;
        self.stream.synchronize()?;
        Ok(out)
    }

    #[expect(clippy::too_many_arguments)]
    pub fn core_booleanity_sparse_collapse8(
        &self,
        tables: &CudaSlice<u64>,
        present_mask: &CudaSlice<u64>,
        values: &CudaSlice<u8>,
        num_polys: usize,
        chunk_domain: usize,
        poly_stride: usize,
        out_len: usize,
    ) -> Result<Vec<DeviceFrVec>, CudaError> {
        let total = num_polys * out_len;
        if total == 0 {
            return (0..num_polys)
                .map(|_| {
                    Ok(DeviceFrVec {
                        stream: self.stream.clone(),
                        buf: self.stream.alloc_zeros(0)?,
                        len: 0,
                        staging: self.staging.clone(),
                    })
                })
                .collect();
        }
        let mut out: CudaSlice<u64> = self.stream.alloc_zeros(total * LIMBS)?;

        let num_polys_arg = num_polys as u64;
        let chunk_domain_arg = chunk_domain as u64;
        let poly_stride_arg = poly_stride as u64;
        let out_len_arg = out_len as u64;
        let cfg = LaunchConfig {
            grid_dim: ((total as u32).div_ceil(BLOCK), 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let f = self.core_booleanity_sparse_collapse8.clone();
        let mut launch = self.stream.launch_builder(&f);
        let _ = launch
            .arg(&mut out)
            .arg(tables)
            .arg(present_mask)
            .arg(values)
            .arg(&num_polys_arg)
            .arg(&chunk_domain_arg)
            .arg(&poly_stride_arg)
            .arg(&out_len_arg);
        // SAFETY: one thread per (poly, row) of `total` sums the 8 round-3 table-sets gathered
        // via present_mask[8j+set]/values[..] (in-range for the round-3 source rows), writing
        // out[poly*out_len + j]; `out` sized for `total`. No shared memory.
        let _ = unsafe { launch.launch(cfg) }?;
        self.stream.synchronize()?;

        let mut polys = Vec::with_capacity(num_polys);
        for i in 0..num_polys {
            let mut buf: CudaSlice<u64> = self.stream.alloc_zeros(out_len * LIMBS)?;
            self.stream.memcpy_dtod(
                &out.slice(i * out_len * LIMBS..(i + 1) * out_len * LIMBS),
                &mut buf,
            )?;
            polys.push(DeviceFrVec {
                stream: self.stream.clone(),
                buf,
                len: out_len,
                staging: self.staging.clone(),
            });
        }
        self.stream.synchronize()?;
        Ok(polys)
    }

    pub fn core_booleanity_address_round_poly(
        &self,
        inputs: CoreBooleanityAddressInputs<'_>,
    ) -> Result<[Fr; 2], CudaError> {
        use num_traits::Zero;
        let num_polys = inputs.g.len();
        assert!(num_polys > 0, "core booleanity address needs at least one g poly");
        assert_eq!(inputs.gamma_squares.len(), num_polys);
        let m = inputs.m as usize;
        assert!(m >= 1, "core booleanity address block size exponent must be >= 1");
        let block_len = 1usize << m;
        let chunk_domain = inputs.g[0].len();
        for g in inputs.g {
            assert_eq!(g.len(), chunk_domain, "core booleanity g polys must have equal length");
        }
        assert_eq!(
            inputs.f_values.len(),
            1usize << (m - 1),
            "f_values length must be 2^(m-1)"
        );
        let groups = chunk_domain / block_len;
        assert!(inputs.e_in.len().is_power_of_two(), "e_in length must be a power of two");
        assert_eq!(
            inputs.e_in.len() * inputs.e_out.len(),
            groups,
            "e_in * e_out must equal the group count"
        );
        if groups == 0 {
            return Ok([Fr::zero(); 2]);
        }
        let in_bits = inputs.e_in.len().trailing_zeros();

        let factor_ptrs = self.factor_ptr_array(inputs.g)?;
        let f_dev = self.upload(inputs.f_values)?;

        const WIDTH: usize = 2;
        let tuple = WIDTH * LIMBS;
        let block = BLOCK;
        let shared = block * tuple as u32 * std::mem::size_of::<u64>() as u32;
        let blocks = (groups as u32).div_ceil(block);
        let mut buf: CudaSlice<u64> = self.stream.alloc_zeros(blocks as usize * tuple)?;

        let group_stride_arg = chunk_domain as u64;
        let num_polys_arg = num_polys as u32;
        let groups_arg = groups as u64;
        let in_bits_arg = in_bits;
        let m_arg = m as u32;
        let width_arg = WIDTH as u32;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: shared,
        };
        let f = self.core_booleanity_address_pairs.clone();
        let mut launch = self.stream.launch_builder(&f);
        let _ = launch
            .arg(&mut buf)
            .arg(&factor_ptrs)
            .arg(&f_dev.buf)
            .arg(&inputs.gamma_squares.buf)
            .arg(&inputs.e_in.buf)
            .arg(&inputs.e_out.buf)
            .arg(&group_stride_arg)
            .arg(&num_polys_arg)
            .arg(&groups_arg)
            .arg(&in_bits_arg)
            .arg(&m_arg);
        // SAFETY: one thread per group folds a 2^m block of each of the num_polys packed
        // g polys against the 2^(m-1) f_values, weighted by e_in[group & mask] /
        // e_out[group >> in_bits] and the per-poly gamma_square, writing one 2-lane tuple
        // per block; shared memory holds `block` tuples.
        let _ = unsafe { launch.launch(cfg) }?;

        let mut len = blocks as usize;
        while len > 1 {
            let out_blocks = (len as u32).div_ceil(block);
            let mut out: CudaSlice<u64> = self.stream.alloc_zeros(out_blocks as usize * tuple)?;
            let len_arg = len as u64;
            let cfg = LaunchConfig {
                grid_dim: (out_blocks, 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: shared,
            };
            let f = self.round_poly_reduce.clone();
            let mut launch = self.stream.launch_builder(&f);
            let _ = launch.arg(&mut out).arg(&buf).arg(&width_arg).arg(&len_arg);
            // SAFETY: round_poly_reduce sums 2-lane tuples across up to `block`
            // blocks; shared memory holds `block` tuples.
            let _ = unsafe { launch.launch(cfg) }?;
            buf = out;
            len = out_blocks as usize;
        }

        let raw = self.stream.clone_dtoh(&buf.slice(0..tuple))?;
        self.stream.synchronize()?;
        Ok([
            limbs_to_fr([raw[0], raw[1], raw[2], raw[3]]),
            limbs_to_fr([raw[4], raw[5], raw[6], raw[7]]),
        ])
    }

    pub fn sparse_register_round_poly(
        &self,
        inputs: SparseRegisterRoundInputs<'_>,
    ) -> Result<[Fr; 2], CudaError> {
        use num_traits::Zero;
        let items = inputs.even_idx.len();
        assert_eq!(inputs.odd_idx.len(), items, "even/odd work-item lists must match");
        assert_eq!(inputs.pair.len(), items, "pair list must match work-item count");
        if items == 0 {
            return Ok([Fr::zero(); 2]);
        }

        let even_dev = self.clone_htod_tracked(inputs.even_idx)?;
        let odd_dev = self.clone_htod_tracked(inputs.odd_idx)?;
        let pair_dev = self.clone_htod_tracked(inputs.pair)?;

        const WIDTH: usize = 2;
        let tuple = WIDTH * LIMBS;
        let block = BLOCK;
        let shared = block * tuple as u32 * std::mem::size_of::<u64>() as u32;
        let blocks = (items as u32).div_ceil(block);
        let mut buf: CudaSlice<u64> = self.stream.alloc_zeros(blocks as usize * tuple)?;

        let in_pairs_arg = inputs.in_pairs;
        let items_arg = items as u64;
        let width_arg = WIDTH as u32;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: shared,
        };
        let f = self.sparse_register_round_pairs.clone();
        let mut launch = self.stream.launch_builder(&f);
        let _ = launch
            .arg(&mut buf)
            .arg(&inputs.val.buf)
            .arg(&inputs.read_ra.buf)
            .arg(&inputs.rd_wa.buf)
            .arg(&inputs.prev_val.buf)
            .arg(&inputs.next_val.buf)
            .arg(&even_dev)
            .arg(&odd_dev)
            .arg(&pair_dev)
            .arg(&inputs.rd_inc.buf)
            .arg(&inputs.e_in.buf)
            .arg(&inputs.e_out.buf)
            .arg(&in_pairs_arg)
            .arg(&items_arg);
        // SAFETY: one thread per merged-column work-item reads its even/odd entry (or -1
        // for absent) from the entry SoA, the per-pair rd_inc[2*pair]/[2*pair+1] and the
        // e_in/e_out weight, writing one 2-lane tuple per block; shared memory holds
        // `block` tuples.
        let _ = unsafe { launch.launch(cfg) }?;

        let mut len = blocks as usize;
        while len > 1 {
            let out_blocks = (len as u32).div_ceil(block);
            let mut out: CudaSlice<u64> = self.stream.alloc_zeros(out_blocks as usize * tuple)?;
            let len_arg = len as u64;
            let cfg = LaunchConfig {
                grid_dim: (out_blocks, 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: shared,
            };
            let f = self.round_poly_reduce.clone();
            let mut launch = self.stream.launch_builder(&f);
            let _ = launch.arg(&mut out).arg(&buf).arg(&width_arg).arg(&len_arg);
            // SAFETY: round_poly_reduce sums 2-lane tuples across up to `block` blocks;
            // shared memory holds `block` tuples.
            let _ = unsafe { launch.launch(cfg) }?;
            buf = out;
            len = out_blocks as usize;
        }

        let raw = self.stream.clone_dtoh(&buf.slice(0..tuple))?;
        self.stream.synchronize()?;
        Ok([
            limbs_to_fr([raw[0], raw[1], raw[2], raw[3]]),
            limbs_to_fr([raw[4], raw[5], raw[6], raw[7]]),
        ])
    }

    pub fn sparse_register_bind(
        &self,
        inputs: SparseRegisterBindInputs<'_>,
    ) -> Result<SparseRegisterEntries, CudaError> {
        let items = inputs.even_idx.len();
        assert_eq!(inputs.odd_idx.len(), items, "even/odd work-item lists must match");

        let make = |len: usize| -> Result<DeviceFrVec, CudaError> {
            Ok(DeviceFrVec {
                stream: self.stream.clone(),
                buf: self.stream.alloc_zeros(len * LIMBS)?,
                len,
                staging: self.staging.clone(),
            })
        };
        let mut val = make(items)?;
        let mut read_ra = make(items)?;
        let mut rd_wa = make(items)?;
        let mut prev_val = make(items)?;
        let mut next_val = make(items)?;
        if items == 0 {
            return Ok(SparseRegisterEntries { val, read_ra, rd_wa, prev_val, next_val });
        }

        let even_dev = self.clone_htod_tracked(inputs.even_idx)?;
        let odd_dev = self.clone_htod_tracked(inputs.odd_idx)?;
        let challenge_dev = self.upload(&[inputs.challenge])?;

        let block = BLOCK;
        let blocks = (items as u32).div_ceil(block);
        let items_arg = items as u64;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let f = self.sparse_register_bind_kernel.clone();
        let mut launch = self.stream.launch_builder(&f);
        let _ = launch
            .arg(&mut val.buf)
            .arg(&mut read_ra.buf)
            .arg(&mut rd_wa.buf)
            .arg(&mut prev_val.buf)
            .arg(&mut next_val.buf)
            .arg(&inputs.val.buf)
            .arg(&inputs.read_ra.buf)
            .arg(&inputs.rd_wa.buf)
            .arg(&inputs.prev_val.buf)
            .arg(&inputs.next_val.buf)
            .arg(&even_dev)
            .arg(&odd_dev)
            .arg(&challenge_dev.buf)
            .arg(&items_arg);
        // SAFETY: one thread per output entry reads its even/odd source (or -1 for
        // absent) from the input SoA and writes the linear-eval at `challenge` plus the
        // carried prev_val/next_val into the freshly allocated output SoA of length
        // `items`; no shared memory.
        let _ = unsafe { launch.launch(cfg) }?;
        self.stream.synchronize()?;
        Ok(SparseRegisterEntries { val, read_ra, rd_wa, prev_val, next_val })
    }

    pub fn instruction_raf_cycle_round_poly(
        &self,
        inputs: InstructionRafCycleInputs<'_>,
    ) -> Result<[Fr; 9], CudaError> {
        use num_traits::Zero;
        assert_eq!(inputs.chunks.len(), 8, "instruction raf cycle needs 8 ra chunks");
        let half = inputs.combined.len() / 2;
        for chunk in inputs.chunks {
            assert_eq!(chunk.len(), half * 2, "instruction raf chunks must match combined length");
        }
        assert!(inputs.e_in.len().is_power_of_two(), "e_in length must be a power of two");
        assert_eq!(
            inputs.e_in.len() * inputs.e_out.len(),
            half,
            "e_in * e_out must equal the row count"
        );
        if half == 0 {
            return Ok([Fr::zero(); 9]);
        }
        let in_bits = inputs.e_in.len().trailing_zeros();

        let mut factors: Vec<&DeviceFrVec> = Vec::with_capacity(9);
        factors.push(inputs.combined);
        factors.extend_from_slice(inputs.chunks);
        let factor_ptrs = self.factor_ptr_array(&factors)?;

        const WIDTH: usize = 9;
        let tuple = WIDTH * LIMBS;
        let max_block = (48 * 1024 / (tuple * std::mem::size_of::<u64>())).max(1) as u32;
        let capped = BLOCK.min(max_block);
        let block = 1u32 << (u32::BITS - 1 - capped.leading_zeros());
        let shared = block * tuple as u32 * std::mem::size_of::<u64>() as u32;
        let blocks = (half as u32).div_ceil(block);
        let mut buf: CudaSlice<u64> = self.stream.alloc_zeros(blocks as usize * tuple)?;

        let pair_stride_arg = half as u64;
        let half_arg = half as u64;
        let in_bits_arg = in_bits;
        let width_arg = WIDTH as u32;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: shared,
        };
        let f = self.instruction_raf_cycle_pairs.clone();
        let mut launch = self.stream.launch_builder(&f);
        let _ = launch
            .arg(&mut buf)
            .arg(&factor_ptrs)
            .arg(&inputs.e_in.buf)
            .arg(&inputs.e_out.buf)
            .arg(&pair_stride_arg)
            .arg(&half_arg)
            .arg(&in_bits_arg);
        // SAFETY: one thread per row reads its pair from the 9 factors (combined + 8 ra
        // chunks) via the device pointer array and the e_in[row & mask] / e_out[row >>
        // in_bits] weights, building the 9 degree-product evals in a thread-local buffer;
        // shared memory holds `block` 9-lane tuples.
        let _ = unsafe { launch.launch(cfg) }?;

        let mut len = blocks as usize;
        while len > 1 {
            let out_blocks = (len as u32).div_ceil(block);
            let mut out: CudaSlice<u64> = self.stream.alloc_zeros(out_blocks as usize * tuple)?;
            let len_arg = len as u64;
            let cfg = LaunchConfig {
                grid_dim: (out_blocks, 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: shared,
            };
            let f = self.round_poly_reduce.clone();
            let mut launch = self.stream.launch_builder(&f);
            let _ = launch.arg(&mut out).arg(&buf).arg(&width_arg).arg(&len_arg);
            // SAFETY: round_poly_reduce sums 9-lane tuples across up to `block` blocks;
            // shared memory holds `block` tuples.
            let _ = unsafe { launch.launch(cfg) }?;
            buf = out;
            len = out_blocks as usize;
        }

        let raw = self.stream.clone_dtoh(&buf.slice(0..tuple))?;
        self.stream.synchronize()?;
        Ok(std::array::from_fn(|e| {
            limbs_to_fr([raw[e * 4], raw[e * 4 + 1], raw[e * 4 + 2], raw[e * 4 + 3]])
        }))
    }

    pub fn instruction_raf_cycle_sparse_round_poly(
        &self,
        inputs: InstructionRafCycleSparseInputs<'_>,
    ) -> Result<[Fr; 9], CudaError> {
        use num_traits::Zero;
        let round = inputs.round;
        assert!((1..=3).contains(&round), "sparse round in 1..=3");
        assert_eq!(inputs.num_chunks, 8, "instruction raf cycle sparse needs 8 ra chunks");
        assert_eq!(
            inputs.values.len(),
            inputs.num_chunks * inputs.source_rows,
            "values shape (chunk-major)",
        );
        assert!(
            inputs.source_rows.is_multiple_of(1usize << round),
            "source rows split into 2^round groups per output",
        );
        let half = inputs.source_rows >> round;
        assert_eq!(inputs.combined.len(), half * 2, "combined length == 2*half");
        assert!(inputs.e_in.len().is_power_of_two(), "e_in length must be a power of two");
        assert_eq!(
            inputs.e_in.len() * inputs.e_out.len(),
            half,
            "e_in * e_out must equal the row count"
        );
        if half == 0 {
            return Ok([Fr::zero(); 9]);
        }
        let in_bits = inputs.e_in.len().trailing_zeros();

        const WIDTH: usize = 9;
        let tuple = WIDTH * LIMBS;
        let max_block = (48 * 1024 / (tuple * std::mem::size_of::<u64>())).max(1) as u32;
        let capped = BLOCK.min(max_block);
        let block = 1u32 << (u32::BITS - 1 - capped.leading_zeros());
        let shared = block * tuple as u32 * std::mem::size_of::<u64>() as u32;
        let blocks = (half as u32).div_ceil(block);
        let mut buf: CudaSlice<u64> = self.stream.alloc_zeros(blocks as usize * tuple)?;

        let num_chunks_arg = inputs.num_chunks as u64;
        let chunk_domain_arg = inputs.chunk_domain as u64;
        let source_rows_arg = inputs.source_rows as u64;
        let half_arg = half as u64;
        let in_bits_arg = in_bits;
        let round_arg = round;
        let width_arg = WIDTH as u32;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: shared,
        };
        let f = self.instruction_raf_cycle_sparse_pairs.clone();
        let mut launch = self.stream.launch_builder(&f);
        let _ = launch
            .arg(&mut buf)
            .arg(inputs.tables)
            .arg(inputs.values)
            .arg(&inputs.combined.buf)
            .arg(&inputs.e_in.buf)
            .arg(&inputs.e_out.buf)
            .arg(&num_chunks_arg)
            .arg(&chunk_domain_arg)
            .arg(&source_rows_arg)
            .arg(&half_arg)
            .arg(&in_bits_arg)
            .arg(&round_arg);
        // SAFETY: one thread per output row (of `half`) forms 9 pairs: pair 0 = the dense
        // `combined` linear pair scaled by e_in[row & mask], pairs 1..8 = each ra chunk's (lo,hi)
        // gathered by summing 2^(round-1) table-sets via the chunk-major u16 `values` at source
        // base=row<<round; irc_ep9 builds the degree-9 evals, weighted by e_out[row >> in_bits];
        // shared holds `block` 9-lane tuples.
        let _ = unsafe { launch.launch(cfg) }?;

        let mut len = blocks as usize;
        while len > 1 {
            let out_blocks = (len as u32).div_ceil(block);
            let mut out: CudaSlice<u64> = self.stream.alloc_zeros(out_blocks as usize * tuple)?;
            let len_arg = len as u64;
            let cfg = LaunchConfig {
                grid_dim: (out_blocks, 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: shared,
            };
            let f = self.round_poly_reduce.clone();
            let mut launch = self.stream.launch_builder(&f);
            let _ = launch.arg(&mut out).arg(&buf).arg(&width_arg).arg(&len_arg);
            // SAFETY: round_poly_reduce sums 9-lane tuples across up to `block` blocks.
            let _ = unsafe { launch.launch(cfg) }?;
            buf = out;
            len = out_blocks as usize;
        }

        let raw = self.stream.clone_dtoh(&buf.slice(0..tuple))?;
        self.stream.synchronize()?;
        Ok(std::array::from_fn(|e| {
            limbs_to_fr([raw[e * 4], raw[e * 4 + 1], raw[e * 4 + 2], raw[e * 4 + 3]])
        }))
    }

    pub fn instruction_raf_cycle_sparse_collapse(
        &self,
        tables: &CudaSlice<u64>,
        values: &CudaSlice<u16>,
        num_chunks: usize,
        chunk_domain: usize,
        source_rows: usize,
        out_len: usize,
    ) -> Result<Vec<DeviceFrVec>, CudaError> {
        let total = num_chunks * out_len;
        if total == 0 {
            return (0..num_chunks)
                .map(|_| {
                    Ok(DeviceFrVec {
                        stream: self.stream.clone(),
                        buf: self.stream.alloc_zeros(0)?,
                        len: 0,
                        staging: self.staging.clone(),
                    })
                })
                .collect();
        }
        let mut out: CudaSlice<u64> = self.stream.alloc_zeros(total * LIMBS)?;
        let num_chunks_arg = num_chunks as u64;
        let chunk_domain_arg = chunk_domain as u64;
        let source_rows_arg = source_rows as u64;
        let out_len_arg = out_len as u64;
        let cfg = LaunchConfig {
            grid_dim: ((total as u32).div_ceil(BLOCK), 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let f = self.instruction_raf_cycle_sparse_collapse.clone();
        let mut launch = self.stream.launch_builder(&f);
        let _ = launch
            .arg(&mut out)
            .arg(tables)
            .arg(values)
            .arg(&num_chunks_arg)
            .arg(&chunk_domain_arg)
            .arg(&source_rows_arg)
            .arg(&out_len_arg);
        // SAFETY: one thread per (chunk, row) sums the 8 round-3 table-sets gathered via the
        // chunk-major u16 `values` at source 8j+set; `out` sized for num_chunks*out_len. No shared.
        let _ = unsafe { launch.launch(cfg) }?;
        self.stream.synchronize()?;

        let mut chunks = Vec::with_capacity(num_chunks);
        for i in 0..num_chunks {
            let mut buf: CudaSlice<u64> = self.stream.alloc_zeros(out_len * LIMBS)?;
            self.stream.memcpy_dtod(
                &out.slice(i * out_len * LIMBS..(i + 1) * out_len * LIMBS),
                &mut buf,
            )?;
            chunks.push(DeviceFrVec {
                stream: self.stream.clone(),
                buf,
                len: out_len,
                staging: self.staging.clone(),
            });
        }
        self.stream.synchronize()?;
        Ok(chunks)
    }

    pub fn hamming_booleanity_round_poly(
        &self,
        inputs: HammingBooleanityInputs<'_>,
    ) -> Result<[Fr; 2], CudaError> {
        use num_traits::Zero;
        let half = inputs.hamming_weight.len() / 2;
        assert!(inputs.e_in.len().is_power_of_two(), "e_in length must be a power of two");
        assert_eq!(
            inputs.e_in.len() * inputs.e_out.len(),
            half,
            "e_in * e_out must equal the hamming-weight row count"
        );
        if half == 0 {
            return Ok([Fr::zero(); 2]);
        }
        let in_bits = inputs.e_in.len().trailing_zeros();

        const WIDTH: usize = 2;
        let tuple = WIDTH * LIMBS;
        let block = BLOCK;
        let shared = block * tuple as u32 * std::mem::size_of::<u64>() as u32;
        let blocks = (half as u32).div_ceil(block);
        let mut buf: CudaSlice<u64> = self.stream.alloc_zeros(blocks as usize * tuple)?;

        let half_arg = half as u64;
        let in_bits_arg = in_bits;
        let width_arg = WIDTH as u32;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: shared,
        };
        let f = self.hamming_booleanity_pairs.clone();
        let mut launch = self.stream.launch_builder(&f);
        let _ = launch
            .arg(&mut buf)
            .arg(&inputs.hamming_weight.buf)
            .arg(&inputs.e_in.buf)
            .arg(&inputs.e_out.buf)
            .arg(&half_arg)
            .arg(&in_bits_arg);
        // SAFETY: one thread per row reads hamming_weight[2row], [2row+1] and its
        // e_in[row & mask] / e_out[row >> in_bits] weights (in range since
        // e_in*e_out == half), writing one 2-lane tuple per block; shared memory
        // holds `block` tuples.
        let _ = unsafe { launch.launch(cfg) }?;

        let mut len = blocks as usize;
        while len > 1 {
            let out_blocks = (len as u32).div_ceil(block);
            let mut out: CudaSlice<u64> = self.stream.alloc_zeros(out_blocks as usize * tuple)?;
            let len_arg = len as u64;
            let cfg = LaunchConfig {
                grid_dim: (out_blocks, 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: shared,
            };
            let f = self.round_poly_reduce.clone();
            let mut launch = self.stream.launch_builder(&f);
            let _ = launch.arg(&mut out).arg(&buf).arg(&width_arg).arg(&len_arg);
            // SAFETY: round_poly_reduce sums 2-lane tuples across up to `block`
            // blocks; shared memory holds `block` tuples.
            let _ = unsafe { launch.launch(cfg) }?;
            buf = out;
            len = out_blocks as usize;
        }

        let raw = self.stream.clone_dtoh(&buf.slice(0..tuple))?;
        self.stream.synchronize()?;
        Ok([
            limbs_to_fr([raw[0], raw[1], raw[2], raw[3]]),
            limbs_to_fr([raw[4], raw[5], raw[6], raw[7]]),
        ])
    }

    pub fn hamming_round_poly(
        &self,
        inputs: HammingRoundPolyInputs<'_>,
    ) -> Result<[Fr; 2], CudaError> {
        use jolt_field::Field;
        use num_traits::Zero;
        let num_ra = inputs.g.len();
        assert!(num_ra > 0, "hamming round poly needs at least one RA poly");
        assert_eq!(inputs.eq_virt.len(), num_ra);
        assert_eq!(inputs.gamma_powers.len(), 3 * num_ra);
        let len = inputs.g[0].len();
        let pair_stride = len / 2;
        for poly in inputs.g.iter().chain(inputs.eq_virt.iter()) {
            assert_eq!(poly.len(), len, "hamming polys must have equal length");
        }
        assert_eq!(inputs.eq_bool.len(), len, "eq_bool length mismatch");
        if pair_stride == 0 {
            return Ok([Fr::zero(); 2]);
        }

        let g_packed = self.factor_ptr_array(inputs.g)?;
        let eq_virt_packed = self.factor_ptr_array(inputs.eq_virt)?;
        let two_dev = self.stream.clone_htod(&fr_to_limbs(Fr::from_u64(2)))?;

        const WIDTH: usize = 2;
        let tuple = WIDTH * LIMBS;
        let block = BLOCK;
        let shared = block * tuple as u32 * std::mem::size_of::<u64>() as u32;
        let blocks = (pair_stride as u32).div_ceil(block);
        let mut buf: CudaSlice<u64> = self.stream.alloc_zeros(blocks as usize * tuple)?;

        let num_ra_arg = num_ra as u32;
        let pair_stride_arg = pair_stride as u64;
        let half_arg = pair_stride as u64;
        let width_arg = WIDTH as u32;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: shared,
        };
        let f = self.hamming_pairs.clone();
        let mut launch = self.stream.launch_builder(&f);
        let _ = launch
            .arg(&mut buf)
            .arg(&g_packed)
            .arg(&eq_virt_packed)
            .arg(&inputs.eq_bool.buf)
            .arg(&inputs.gamma_powers.buf)
            .arg(&two_dev)
            .arg(&num_ra_arg)
            .arg(&pair_stride_arg)
            .arg(&half_arg);
        // SAFETY: one thread per row reads its pair from each of the num_ra packed g
        // and eq_virt polys, the shared eq_bool pair, and the 3*num_ra gammas,
        // writing one 2-lane tuple per block; shared memory holds `block` tuples.
        let _ = unsafe { launch.launch(cfg) }?;

        let mut len_reduce = blocks as usize;
        while len_reduce > 1 {
            let out_blocks = (len_reduce as u32).div_ceil(block);
            let mut out: CudaSlice<u64> = self.stream.alloc_zeros(out_blocks as usize * tuple)?;
            let len_arg = len_reduce as u64;
            let cfg = LaunchConfig {
                grid_dim: (out_blocks, 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: shared,
            };
            let f = self.round_poly_reduce.clone();
            let mut launch = self.stream.launch_builder(&f);
            let _ = launch.arg(&mut out).arg(&buf).arg(&width_arg).arg(&len_arg);
            // SAFETY: round_poly_reduce sums 2-lane tuples across up to `block`
            // blocks; shared memory holds `block` tuples.
            let _ = unsafe { launch.launch(cfg) }?;
            buf = out;
            len_reduce = out_blocks as usize;
        }

        let raw = self.stream.clone_dtoh(&buf.slice(0..tuple))?;
        self.stream.synchronize()?;
        let eval0 = limbs_to_fr([raw[0], raw[1], raw[2], raw[3]]) * inputs.scale;
        let eval1 = limbs_to_fr([raw[4], raw[5], raw[6], raw[7]]) * inputs.scale;
        Ok([eval0, eval1])
    }

    pub fn gather8_materialize(
        &self,
        inputs: Gather8Inputs<'_>,
    ) -> Result<Vec<Vec<Fr>>, CudaError> {
        let Gather8Inputs {
            table_groups,
            indices,
            num_chunks,
            table_len,
            new_len,
        } = inputs;
        let total = num_chunks * new_len;
        if total == 0 {
            return Ok((0..num_chunks).map(|_| Vec::new()).collect());
        }
        assert_eq!(indices.len(), num_chunks * new_len * 8);

        let group_devs: Vec<DeviceFrVec> = table_groups
            .iter()
            .map(|group| self.upload(group))
            .collect::<Result<_, _>>()?;
        let indices_dev = self.clone_htod_tracked(indices)?;
        let mut out: CudaSlice<u64> = self.stream.alloc_zeros(total * LIMBS)?;

        let num_chunks_arg = num_chunks as u64;
        let table_len_arg = table_len as u64;
        let new_len_arg = new_len as u64;
        let cfg = LaunchConfig {
            grid_dim: ((total as u32).div_ceil(BLOCK), 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let f = self.gather8_materialize.clone();
        let mut launch = self.stream.launch_builder(&f);
        let _ = launch
            .arg(&mut out)
            .arg(&group_devs[0].buf)
            .arg(&group_devs[1].buf)
            .arg(&group_devs[2].buf)
            .arg(&group_devs[3].buf)
            .arg(&group_devs[4].buf)
            .arg(&group_devs[5].buf)
            .arg(&group_devs[6].buf)
            .arg(&group_devs[7].buf)
            .arg(&indices_dev)
            .arg(&num_chunks_arg)
            .arg(&table_len_arg)
            .arg(&new_len_arg);
        // SAFETY: one thread per (chunk, index) of `total` reads its 8 indices from
        // `indices` (num_chunks*new_len*8 entries) and gathers from the 8 table-group
        // buffers (each num_chunks*table_len; idx < table_len since idx is a valid
        // u8 table position), writing one element to `out` (total entries).
        let _ = unsafe { launch.launch(cfg) }?;
        self.stream.synchronize()?;

        let raw = self.stream.clone_dtoh(&out)?;
        Ok((0..num_chunks)
            .map(|chunk| unflatten(&raw[chunk * new_len * LIMBS..(chunk + 1) * new_len * LIMBS]))
            .collect())
    }

    pub fn core_booleanity_gather(
        &self,
        inputs: CoreBooleanityGatherInputs<'_>,
    ) -> Result<Vec<DeviceFrVec>, CudaError> {
        let CoreBooleanityGatherInputs {
            tables,
            present_mask,
            values,
            num_polys,
            chunk_domain,
            rows,
            poly_stride,
        } = inputs;
        assert_eq!(tables.len(), num_polys * chunk_domain, "tables shape");
        assert_eq!(present_mask.len(), rows, "present_mask length");
        assert_eq!(values.len(), rows * poly_stride, "values shape");
        assert!(num_polys <= poly_stride, "num_polys within row stride");

        let total = num_polys * rows;
        if total == 0 {
            return (0..num_polys)
                .map(|_| {
                    Ok(DeviceFrVec {
                        stream: self.stream.clone(),
                        buf: self.stream.alloc_zeros(0)?,
                        len: 0,
                        staging: self.staging.clone(),
                    })
                })
                .collect();
        }
        let mut out: CudaSlice<u64> = self.stream.alloc_zeros(total * LIMBS)?;

        let tables_dev = self.upload(tables)?;
        let mask_dev = self.clone_htod_tracked(present_mask)?;
        let values_dev = self.clone_htod_tracked(values)?;

        let num_polys_arg = num_polys as u64;
        let chunk_domain_arg = chunk_domain as u64;
        let rows_arg = rows as u64;
        let poly_stride_arg = poly_stride as u64;
        let cfg = LaunchConfig {
            grid_dim: ((total as u32).div_ceil(BLOCK), 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let f = self.core_booleanity_gather.clone();
        let mut launch = self.stream.launch_builder(&f);
        let _ = launch
            .arg(&mut out)
            .arg(&tables_dev.buf)
            .arg(&mask_dev)
            .arg(&values_dev)
            .arg(&num_polys_arg)
            .arg(&chunk_domain_arg)
            .arg(&rows_arg)
            .arg(&poly_stride_arg);
        // SAFETY: one thread per (poly, row) of `total` reads present_mask[row] and
        // values[row*poly_stride + poly] (both in-range by the asserts above), gathers
        // tables[poly*chunk_domain + value] when present (value is a u8 table position <
        // chunk_domain), and writes out[poly*rows + row]. No shared memory.
        let _ = unsafe { launch.launch(cfg) }?;
        self.stream.synchronize()?;

        let mut polys = Vec::with_capacity(num_polys);
        for i in 0..num_polys {
            let mut buf: CudaSlice<u64> = self.stream.alloc_zeros(rows * LIMBS)?;
            self.stream.memcpy_dtod(
                &out.slice(i * rows * LIMBS..(i + 1) * rows * LIMBS),
                &mut buf,
            )?;
            polys.push(DeviceFrVec {
                stream: self.stream.clone(),
                buf,
                len: rows,
                staging: self.staging.clone(),
            });
        }
        self.stream.synchronize()?;
        Ok(polys)
    }

    pub fn uniskip_extended_evals(&self, inputs: UniskipInputs<'_>) -> Result<Vec<Fr>, CudaError> {
        use num_traits::Zero;
        let degree = inputs.degree;
        assert!(degree >= 1, "uniskip degree must be >= 1");
        let first_len = inputs.first_group_rows.len();
        let second_len = inputs.second_group_rows.len();
        assert_eq!(inputs.first_coeffs.len(), degree * first_len);
        assert_eq!(inputs.second_coeffs.len(), degree * second_len);
        let cycles = inputs.eq_evals.len() / 2;
        if cycles == 0 {
            return Ok(vec![Fr::zero(); degree]);
        }
        assert_eq!(inputs.row_dots_a.len(), cycles * inputs.row_count);
        assert_eq!(inputs.row_dots_b.len(), cycles * inputs.row_count);

        let first_coeffs = self.upload(inputs.first_coeffs)?;
        let second_coeffs = self.upload(inputs.second_coeffs)?;
        let first_rows = self.clone_htod_tracked(inputs.first_group_rows)?;
        let second_rows = self.clone_htod_tracked(inputs.second_group_rows)?;

        let tuple = degree * LIMBS;
        let max_block = (48 * 1024 / (tuple * std::mem::size_of::<u64>())).max(1) as u32;
        let capped = BLOCK.min(max_block);
        let block = 1u32 << (u32::BITS - 1 - capped.leading_zeros());
        let shared = block * tuple as u32 * std::mem::size_of::<u64>() as u32;
        let blocks = (cycles as u32).div_ceil(block);
        let mut buf: CudaSlice<u64> = self.stream.alloc_zeros(blocks as usize * tuple)?;

        let first_len_arg = first_len as u32;
        let second_len_arg = second_len as u32;
        let row_count_arg = inputs.row_count as u64;
        let degree_arg = degree as u32;
        let cycles_arg = cycles as u64;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: shared,
        };
        let f = self.uniskip_pairs.clone();
        let mut launch = self.stream.launch_builder(&f);
        let _ = launch
            .arg(&mut buf)
            .arg(&inputs.row_dots_a.buf)
            .arg(&inputs.row_dots_b.buf)
            .arg(&inputs.eq_evals.buf)
            .arg(&first_coeffs.buf)
            .arg(&second_coeffs.buf)
            .arg(&first_rows)
            .arg(&second_rows)
            .arg(&first_len_arg)
            .arg(&second_len_arg)
            .arg(&row_count_arg)
            .arg(&degree_arg)
            .arg(&cycles_arg);
        // SAFETY: one thread per cycle reads its row_count-strided dots, the eq pair,
        // and the two groups' coeff/row tables (bounded by first_len/second_len),
        // writing one `degree`-tuple per block; shared memory holds `block` tuples.
        let _ = unsafe { launch.launch(cfg) }?;

        let mut len = blocks as usize;
        while len > 1 {
            let out_blocks = (len as u32).div_ceil(block);
            let mut out: CudaSlice<u64> = self.stream.alloc_zeros(out_blocks as usize * tuple)?;
            let len_arg = len as u64;
            let cfg = LaunchConfig {
                grid_dim: (out_blocks, 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: shared,
            };
            let f = self.round_poly_reduce.clone();
            let mut launch = self.stream.launch_builder(&f);
            let _ = launch.arg(&mut out).arg(&buf).arg(&degree_arg).arg(&len_arg);
            // SAFETY: each block reads up to `block` tuples from `buf` (len total) and
            // writes one tuple to `out`; shared memory holds `block` tuples.
            let _ = unsafe { launch.launch(cfg) }?;
            buf = out;
            len = out_blocks as usize;
        }

        let raw = self.stream.clone_dtoh(&buf.slice(0..tuple))?;
        self.stream.synchronize()?;
        Ok((0..degree)
            .map(|e| limbs_to_fr([raw[e * 4], raw[e * 4 + 1], raw[e * 4 + 2], raw[e * 4 + 3]]))
            .collect())
    }

    fn map(&self, func: &CudaFunction, a: &mut DeviceFrVec, b: &DeviceFrVec) -> Result<(), CudaError> {
        assert_eq!(a.len, b.len, "map operands must have equal length");
        let n = a.len;
        if n == 0 {
            return Ok(());
        }

        let cfg = LaunchConfig {
            grid_dim: ((n as u32).div_ceil(BLOCK), 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_arg = n as u64;
        let mut launch = self.stream.launch_builder(func);
        let _ = launch.arg(&mut a.buf).arg(&b.buf).arg(&n_arg);
        // SAFETY: each thread i reads a[i] and b[i] and writes a[i] in place; the
        // elementwise access pattern means the in/out alias on `a` is hazard-free.
        // Both buffers hold n * LIMBS u64s.
        let _ = unsafe { launch.launch(cfg) }?;

        Ok(())
    }

    pub fn add(&self, a: &mut DeviceFrVec, b: &DeviceFrVec) -> Result<(), CudaError> {
        let f = self.add.clone();
        self.map(&f, a, b)
    }

    pub fn sub(&self, a: &mut DeviceFrVec, b: &DeviceFrVec) -> Result<(), CudaError> {
        let f = self.sub.clone();
        self.map(&f, a, b)
    }

    pub fn mul(&self, a: &mut DeviceFrVec, b: &DeviceFrVec) -> Result<(), CudaError> {
        let f = self.mul.clone();
        self.map(&f, a, b)
    }

    pub fn fma(
        &self,
        a: &mut DeviceFrVec,
        b: &DeviceFrVec,
        c: &DeviceFrVec,
    ) -> Result<(), CudaError> {
        assert_eq!(a.len, b.len, "fma operands must have equal length");
        assert_eq!(a.len, c.len, "fma operands must have equal length");
        let n = a.len;
        if n == 0 {
            return Ok(());
        }

        let cfg = LaunchConfig {
            grid_dim: ((n as u32).div_ceil(BLOCK), 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_arg = n as u64;
        let f = self.fma.clone();
        let mut launch = self.stream.launch_builder(&f);
        let _ = launch.arg(&mut a.buf).arg(&b.buf).arg(&c.buf).arg(&n_arg);
        // SAFETY: elementwise kernel; the in/out alias on `a` is hazard-free, and
        // all three buffers hold n * LIMBS u64s.
        let _ = unsafe { launch.launch(cfg) }?;

        Ok(())
    }

    pub fn add_scalar(&self, a: &mut DeviceFrVec, scalar: Fr) -> Result<(), CudaError> {
        let n = a.len;
        if n == 0 {
            return Ok(());
        }
        let scalar_dev = self.stream.clone_htod(&fr_to_limbs(scalar))?;
        let cfg = LaunchConfig {
            grid_dim: ((n as u32).div_ceil(BLOCK), 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_arg = n as u64;
        let f = self.add_scalar.clone();
        let mut launch = self.stream.launch_builder(&f);
        let _ = launch.arg(&mut a.buf).arg(&scalar_dev).arg(&n_arg);
        // SAFETY: each thread i reads a[i] and the single scalar, writing a[i] in place;
        // a holds n * LIMBS u64s. No shared memory.
        let _ = unsafe { launch.launch(cfg) }?;

        Ok(())
    }

    pub fn u64_to_mont(&self, values: &[u64]) -> Result<DeviceFrVec, CudaError> {
        let n = values.len();
        let mut out = DeviceFrVec {
            stream: self.stream.clone(),
            buf: self.stream.alloc_zeros(n * LIMBS)?,
            len: n,
            staging: self.staging.clone(),
        };
        if n == 0 {
            return Ok(out);
        }
        let in_dev = self.upload_u64_slice(values)?;
        let n_arg = n as u64;
        let cfg = LaunchConfig {
            grid_dim: ((n as u32).div_ceil(BLOCK), 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let f = self.u64_to_mont.clone();
        let mut launch = self.stream.launch_builder(&f);
        let _ = launch.arg(&mut out.buf).arg(&in_dev).arg(&n_arg);
        // SAFETY: each thread i reads in[i] and writes the Montgomery form (raw limbs
        // [in[i],0,0,0] times R2) to out[i*LIMBS..]; out holds n * LIMBS u64s, in holds n.
        // No shared memory.
        let _ = unsafe { launch.launch(cfg) }?;
        self.stream.synchronize()?;
        Ok(out)
    }

    pub fn batched_bind(
        &self,
        values: &mut DeviceFrVec,
        scratch: &mut DeviceFrVec,
        num_buffers: usize,
        challenge: Fr,
    ) -> Result<(), CudaError> {
        assert!(num_buffers > 0, "num_buffers must be > 0");
        assert_eq!(
            values.len % num_buffers,
            0,
            "packed buffer length must split evenly across num_buffers"
        );
        let seg_len = values.len / num_buffers;
        assert_eq!(seg_len % 2, 0, "each buffer must have even length");

        // Each segment has even length, so pairing (0,1),(2,3),... over the whole
        // packed buffer never straddles a segment boundary: batched bind is a single
        // bind over all num_buffers*half pairs.
        let half = values.len / 2;
        if scratch.buf.len() < half * LIMBS {
            scratch.buf = self.stream.alloc_zeros(half * LIMBS)?;
        }
        scratch.len = half;
        if half == 0 {
            std::mem::swap(values, scratch);
            return Ok(());
        }

        let challenge_dev = self.stream.clone_htod(&fr_to_limbs(challenge))?;
        let cfg = LaunchConfig {
            grid_dim: ((half as u32).div_ceil(BLOCK), 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let half_arg = half as u64;
        let f = self.bind.clone();
        xfer_stats::timed(xfer_stats::Phase::Bind, || {
            let mut launch = self.stream.launch_builder(&f);
            let _ = launch
                .arg(&mut scratch.buf)
                .arg(&values.buf)
                .arg(&challenge_dev)
                .arg(&half_arg);
            // SAFETY: each thread i reads values[2i], values[2i+1] and the single
            // challenge, writing scratch[i]; scratch and values are distinct buffers
            // holding >= half and 2*half elements respectively.
            let _ = unsafe { launch.launch(cfg) }?;
            self.stream.synchronize()
        })?;

        std::mem::swap(values, scratch);
        Ok(())
    }

    pub fn bind(
        &self,
        values: &mut DeviceFrVec,
        scratch: &mut DeviceFrVec,
        challenge: Fr,
    ) -> Result<(), CudaError> {
        let half = values.len / 2;
        if scratch.buf.len() < half * LIMBS {
            scratch.buf = self.stream.alloc_zeros(half * LIMBS)?;
        }
        scratch.len = half;
        if half == 0 {
            std::mem::swap(values, scratch);
            return Ok(());
        }

        let challenge_dev = self.stream.clone_htod(&fr_to_limbs(challenge))?;
        let cfg = LaunchConfig {
            grid_dim: ((half as u32).div_ceil(BLOCK), 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let half_arg = half as u64;
        let f = self.bind.clone();
        xfer_stats::timed(xfer_stats::Phase::Bind, || {
            let mut launch = self.stream.launch_builder(&f);
            let _ = launch
                .arg(&mut scratch.buf)
                .arg(&values.buf)
                .arg(&challenge_dev)
                .arg(&half_arg);
            // SAFETY: each thread i reads values[2i], values[2i+1] and the single
            // challenge, writing scratch[i]; scratch and values are distinct buffers
            // holding >= half and 2*half elements respectively.
            let _ = unsafe { launch.launch(cfg) }?;
            self.stream.synchronize()
        })?;

        std::mem::swap(values, scratch);
        Ok(())
    }

    pub fn cubic_accumulate(
        &self,
        eq: &DeviceFrVec,
        az: &DeviceFrVec,
        bz: &DeviceFrVec,
    ) -> Result<[Fr; 4], CudaError> {
        use num_traits::Zero;
        assert_eq!(eq.len, az.len, "cubic operands must have equal length");
        assert_eq!(eq.len, bz.len, "cubic operands must have equal length");
        let pairs = eq.len / 2;
        if pairs == 0 {
            return Ok([Fr::zero(); 4]);
        }

        const TUPLE: usize = 4 * LIMBS;
        let shared = BLOCK * TUPLE as u32 * std::mem::size_of::<u64>() as u32;
        let blocks = (pairs as u32).div_ceil(BLOCK);
        let mut buf: CudaSlice<u64> = self.stream.alloc_zeros(blocks as usize * TUPLE)?;

        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: shared,
        };
        let pairs_arg = pairs as u64;
        let f = self.cubic_pairs.clone();
        let mut launch = self.stream.launch_builder(&f);
        let _ = launch
            .arg(&mut buf)
            .arg(&eq.buf)
            .arg(&az.buf)
            .arg(&bz.buf)
            .arg(&pairs_arg);
        // SAFETY: each block reads up to BLOCK pairs from eq/az/bz (2*pairs elements
        // each) and writes one 4-coefficient tuple to `buf` (blocks tuples). Shared
        // memory holds BLOCK tuples as configured above.
        let _ = unsafe { launch.launch(cfg) }?;

        let mut len = blocks as usize;
        while len > 1 {
            let out_blocks = (len as u32).div_ceil(BLOCK);
            let mut out: CudaSlice<u64> = self.stream.alloc_zeros(out_blocks as usize * TUPLE)?;
            let cfg = LaunchConfig {
                grid_dim: (out_blocks, 1, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: shared,
            };
            let len_arg = len as u64;
            let f = self.cubic_tuple_reduce.clone();
            let mut launch = self.stream.launch_builder(&f);
            let _ = launch.arg(&mut out).arg(&buf).arg(&len_arg);
            // SAFETY: each block reads up to BLOCK tuples from `buf` (len tuples) and
            // writes one tuple to `out` (out_blocks tuples). Shared memory holds
            // BLOCK tuples as configured above.
            let _ = unsafe { launch.launch(cfg) }?;
            buf = out;
            len = out_blocks as usize;
        }

        let raw = self.stream.clone_dtoh(&buf.slice(0..TUPLE))?;
        Ok(core::array::from_fn(|i| {
            limbs_to_fr([raw[i * 4], raw[i * 4 + 1], raw[i * 4 + 2], raw[i * 4 + 3]])
        }))
    }

    pub fn dense_product_round_poly(
        &self,
        terms: RoundPolyTerms<'_>,
    ) -> Result<Vec<Fr>, CudaError> {
        use num_traits::Zero;
        let degree = terms.degree;
        assert!(degree >= 1, "round poly degree must be >= 1");
        assert!(degree < 16, "dense product degree exceeds kernel bound");
        let width = degree + 1;
        let pair_stride = terms.factors[0].len() / 2;
        for factor in terms.factors {
            assert_eq!(
                factor.len(),
                pair_stride * 2,
                "round poly factors must have equal length"
            );
        }
        if pair_stride == 0 {
            return Ok(vec![Fr::zero(); width]);
        }

        let factor_ptrs = self.factor_ptr_array(terms.factors)?;

        let offsets_dev = self.clone_htod_tracked(terms.term_factor_offsets)?;
        let indices_dev = self.clone_htod_tracked(terms.term_factor_indices)?;

        let tuple = width * LIMBS;
        let max_block = (48 * 1024 / (tuple * std::mem::size_of::<u64>())).max(1) as u32;
        let capped = BLOCK.min(max_block);
        let block = 1u32 << (u32::BITS - 1 - capped.leading_zeros());
        let shared = block * tuple as u32 * std::mem::size_of::<u64>() as u32;
        let blocks = (pair_stride as u32).div_ceil(block);
        let mut buf: CudaSlice<u64> = self.stream.alloc_zeros(blocks as usize * tuple)?;

        let pair_stride_arg = pair_stride as u64;
        let num_terms_arg = terms.term_coeffs.len() as u32;
        let degree_arg = degree as u32;
        let width_arg = width as u32;
        let half_arg = pair_stride as u64;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: shared,
        };
        let f = self.dense_product_pairs.clone();
        let mut launch = self.stream.launch_builder(&f);
        let _ = launch
            .arg(&mut buf)
            .arg(&factor_ptrs)
            .arg(&terms.term_coeffs.buf)
            .arg(&offsets_dev)
            .arg(&indices_dev)
            .arg(&pair_stride_arg)
            .arg(&num_terms_arg)
            .arg(&degree_arg)
            .arg(&half_arg);
        // SAFETY: one thread per row reads its pair from each factor (via the device
        // pointer array) and the term tables (bounded by offsets), building the
        // width=degree+1 monomial coefficients in a thread-local buffer; shared memory
        // holds `block` tuples. factor_ptrs stays alive until the launch is scheduled.
        let _ = unsafe { launch.launch(cfg) }?;

        let mut len = blocks as usize;
        while len > 1 {
            let out_blocks = (len as u32).div_ceil(block);
            let mut out: CudaSlice<u64> = self.stream.alloc_zeros(out_blocks as usize * tuple)?;
            let len_arg = len as u64;
            let cfg = LaunchConfig {
                grid_dim: (out_blocks, 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: shared,
            };
            let f = self.round_poly_reduce.clone();
            let mut launch = self.stream.launch_builder(&f);
            let _ = launch.arg(&mut out).arg(&buf).arg(&width_arg).arg(&len_arg);
            // SAFETY: round_poly_reduce sums tuples of the lane count in its third
            // arg (here `width` = degree+1) across up to `block` blocks; shared
            // memory holds `block` tuples.
            let _ = unsafe { launch.launch(cfg) }?;
            buf = out;
            len = out_blocks as usize;
        }

        let raw = self.stream.clone_dtoh(&buf.slice(0..tuple))?;
        self.stream.synchronize()?;
        Ok((0..width)
            .map(|e| limbs_to_fr([raw[e * 4], raw[e * 4 + 1], raw[e * 4 + 2], raw[e * 4 + 3]]))
            .collect())
    }

    pub fn sum_of_products_round_poly(
        &self,
        terms: RoundPolyTerms<'_>,
    ) -> Result<Vec<Fr>, CudaError> {
        use jolt_field::Field;
        let degree = terms.degree;
        assert!(degree >= 1, "round poly degree must be >= 1");
        // Evaluation points {0, 2, 3, ..., degree}: index 0 is x=0, index e>=1 is x=e+1.
        let points: Vec<Fr> = (0..degree)
            .map(|e| Fr::from_u64(if e == 0 { 0 } else { (e + 1) as u64 }))
            .collect();
        self.sum_of_products_round_poly_at(terms, &points)
    }

    pub fn sum_of_products_round_poly_at(
        &self,
        terms: RoundPolyTerms<'_>,
        points: &[Fr],
    ) -> Result<Vec<Fr>, CudaError> {
        use num_traits::Zero;
        let num_evals = points.len();
        assert!(num_evals >= 1, "round poly needs at least one eval point");
        let pair_stride = terms.factors[0].len() / 2;
        for factor in terms.factors {
            assert_eq!(
                factor.len(),
                pair_stride * 2,
                "round poly factors must have equal length"
            );
        }
        if pair_stride == 0 {
            return Ok(vec![Fr::zero(); num_evals]);
        }

        let factor_ptrs = self.factor_ptr_array(terms.factors)?;

        let points_dev = self.upload(points)?;
        let offsets_dev = self.clone_htod_tracked(terms.term_factor_offsets)?;
        let indices_dev = self.clone_htod_tracked(terms.term_factor_indices)?;

        let tuple = num_evals * LIMBS;
        // Cap the block so the per-thread tuple accumulators fit in 48 KB shared memory;
        // the tree reduce needs a power-of-two block, so take the largest such <= cap.
        let max_block = (48 * 1024 / (tuple * std::mem::size_of::<u64>())).max(1) as u32;
        let capped = BLOCK.min(max_block);
        let block = 1u32 << (u32::BITS - 1 - capped.leading_zeros());
        let shared = block * tuple as u32 * std::mem::size_of::<u64>() as u32;
        let blocks = (pair_stride as u32).div_ceil(block);
        let mut buf: CudaSlice<u64> = self.stream.alloc_zeros(blocks as usize * tuple)?;

        let pair_stride_arg = pair_stride as u64;
        let num_terms_arg = terms.term_coeffs.len() as u32;
        let num_evals_arg = num_evals as u32;
        let half_arg = pair_stride as u64;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: shared,
        };
        let f = self.round_poly_pairs.clone();
        let mut launch = self.stream.launch_builder(&f);
        let _ = launch
            .arg(&mut buf)
            .arg(&factor_ptrs)
            .arg(&points_dev.buf)
            .arg(&terms.term_coeffs.buf)
            .arg(&offsets_dev)
            .arg(&indices_dev)
            .arg(&pair_stride_arg)
            .arg(&num_terms_arg)
            .arg(&num_evals_arg)
            .arg(&half_arg);
        // SAFETY: one thread per row reads its pair from each packed factor and the
        // term tables (bounded by offsets), writing one `num_evals`-tuple per block;
        // shared memory holds `block` tuples as configured above.
        let _ = unsafe { launch.launch(cfg) }?;

        let mut len = blocks as usize;
        while len > 1 {
            let out_blocks = (len as u32).div_ceil(block);
            let mut out: CudaSlice<u64> = self.stream.alloc_zeros(out_blocks as usize * tuple)?;
            let len_arg = len as u64;
            let cfg = LaunchConfig {
                grid_dim: (out_blocks, 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: shared,
            };
            let f = self.round_poly_reduce.clone();
            let mut launch = self.stream.launch_builder(&f);
            let _ = launch.arg(&mut out).arg(&buf).arg(&num_evals_arg).arg(&len_arg);
            // SAFETY: each block reads up to `block` tuples from `buf` (len total) and
            // writes one tuple to `out`; shared memory holds `block` tuples.
            let _ = unsafe { launch.launch(cfg) }?;
            buf = out;
            len = out_blocks as usize;
        }

        let raw = self.stream.clone_dtoh(&buf.slice(0..tuple))?;
        self.stream.synchronize()?;
        Ok((0..num_evals)
            .map(|e| limbs_to_fr([raw[e * 4], raw[e * 4 + 1], raw[e * 4 + 2], raw[e * 4 + 3]]))
            .collect())
    }

    pub fn gruen_round_poly(&self, inputs: GruenRoundPolyInputs<'_>) -> Result<Vec<Fr>, CudaError> {
        use num_traits::Zero;
        let _ = inputs.degree;
        let pair_stride = inputs.factors[0].len() / 2;
        for factor in inputs.factors {
            assert_eq!(
                factor.len(),
                pair_stride * 2,
                "gruen round poly factors must have equal length"
            );
        }
        assert!(inputs.e_in.len().is_power_of_two(), "e_in length must be a power of two");
        assert_eq!(
            inputs.e_in.len() * inputs.e_out.len(),
            pair_stride * 2,
            "e_in * e_out must equal the factor length"
        );
        if pair_stride == 0 {
            return Ok(vec![Fr::zero(); 2]);
        }
        let low_round = inputs.e_in.len() > 1;
        let in_pairs = inputs.e_in.len() / 2;

        let factor_ptrs = self.factor_ptr_array(inputs.factors)?;

        let offsets_dev = self.clone_htod_tracked(inputs.term_factor_offsets)?;
        let indices_dev = self.clone_htod_tracked(inputs.term_factor_indices)?;

        const WIDTH: usize = 2;
        let tuple = WIDTH * LIMBS;
        let block = BLOCK;
        let shared = block * tuple as u32 * std::mem::size_of::<u64>() as u32;
        let blocks = (pair_stride as u32).div_ceil(block);
        let mut buf: CudaSlice<u64> = self.stream.alloc_zeros(blocks as usize * tuple)?;

        let pair_stride_arg = pair_stride as u64;
        let num_terms_arg = inputs.term_coeffs.len() as u32;
        let half_arg = pair_stride as u64;
        let in_pairs_arg = in_pairs as u64;
        let low_round_arg = u32::from(low_round);
        let width_arg = WIDTH as u32;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: shared,
        };
        let f = self.gruen_round_poly_pairs.clone();
        let mut launch = self.stream.launch_builder(&f);
        let _ = launch
            .arg(&mut buf)
            .arg(&factor_ptrs)
            .arg(&inputs.term_coeffs.buf)
            .arg(&offsets_dev)
            .arg(&indices_dev)
            .arg(&inputs.e_in.buf)
            .arg(&inputs.e_out.buf)
            .arg(&pair_stride_arg)
            .arg(&num_terms_arg)
            .arg(&half_arg)
            .arg(&in_pairs_arg)
            .arg(&low_round_arg);
        // SAFETY: one thread per row reads its pair from each packed factor, the term
        // tables (bounded by offsets), and its pair-summed split-eq weight (in range
        // since e_in*e_out == 2*pair_stride), writing one [q_constant, q_top] tuple
        // per block; shared memory holds `block` 2-lane tuples.
        let _ = unsafe { launch.launch(cfg) }?;

        let mut len = blocks as usize;
        while len > 1 {
            let out_blocks = (len as u32).div_ceil(block);
            let mut out: CudaSlice<u64> = self.stream.alloc_zeros(out_blocks as usize * tuple)?;
            let len_arg = len as u64;
            let cfg = LaunchConfig {
                grid_dim: (out_blocks, 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: shared,
            };
            let f = self.round_poly_reduce.clone();
            let mut launch = self.stream.launch_builder(&f);
            let _ = launch.arg(&mut out).arg(&buf).arg(&width_arg).arg(&len_arg);
            // SAFETY: round_poly_reduce sums 2-lane tuples across up to `block`
            // blocks; shared memory holds `block` tuples.
            let _ = unsafe { launch.launch(cfg) }?;
            buf = out;
            len = out_blocks as usize;
        }

        let raw = self.stream.clone_dtoh(&buf.slice(0..tuple))?;
        self.stream.synchronize()?;
        Ok((0..WIDTH)
            .map(|e| limbs_to_fr([raw[e * 4], raw[e * 4 + 1], raw[e * 4 + 2], raw[e * 4 + 3]]))
            .collect())
    }

    pub fn eq_weighted_round_poly(
        &self,
        terms: RoundPolyTerms<'_>,
        e_in: &DeviceFrVec,
        e_out: &DeviceFrVec,
    ) -> Result<Vec<Fr>, CudaError> {
        use jolt_field::Field;
        use num_traits::Zero;
        let degree = terms.degree;
        assert!(degree >= 1, "round poly degree must be >= 1");
        let pair_stride = terms.factors[0].len() / 2;
        for factor in terms.factors {
            assert_eq!(
                factor.len(),
                pair_stride * 2,
                "round poly factors must have equal length"
            );
        }
        assert!(e_in.len().is_power_of_two(), "e_in length must be a power of two");
        assert_eq!(
            e_in.len() * e_out.len(),
            pair_stride,
            "e_in * e_out must equal the number of rows"
        );
        if pair_stride == 0 {
            return Ok(vec![Fr::zero(); degree]);
        }
        let in_bits = e_in.len().trailing_zeros();

        let factor_ptrs = self.factor_ptr_array(terms.factors)?;

        let points: Vec<Fr> = (0..degree)
            .map(|e| Fr::from_u64(if e == 0 { 0 } else { (e + 1) as u64 }))
            .collect();
        let points_dev = self.upload(&points)?;
        let offsets_dev = self.clone_htod_tracked(terms.term_factor_offsets)?;
        let indices_dev = self.clone_htod_tracked(terms.term_factor_indices)?;

        let tuple = degree * LIMBS;
        let max_block = (48 * 1024 / (tuple * std::mem::size_of::<u64>())).max(1) as u32;
        let capped = BLOCK.min(max_block);
        let block = 1u32 << (u32::BITS - 1 - capped.leading_zeros());
        let shared = block * tuple as u32 * std::mem::size_of::<u64>() as u32;
        let blocks = (pair_stride as u32).div_ceil(block);
        let mut buf: CudaSlice<u64> = self.stream.alloc_zeros(blocks as usize * tuple)?;

        let pair_stride_arg = pair_stride as u64;
        let num_terms_arg = terms.term_coeffs.len() as u32;
        let degree_arg = degree as u32;
        let half_arg = pair_stride as u64;
        let in_bits_arg = in_bits;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: shared,
        };
        let f = self.eq_round_poly_pairs.clone();
        let mut launch = self.stream.launch_builder(&f);
        let _ = launch
            .arg(&mut buf)
            .arg(&factor_ptrs)
            .arg(&points_dev.buf)
            .arg(&terms.term_coeffs.buf)
            .arg(&offsets_dev)
            .arg(&indices_dev)
            .arg(&e_in.buf)
            .arg(&e_out.buf)
            .arg(&pair_stride_arg)
            .arg(&num_terms_arg)
            .arg(&degree_arg)
            .arg(&half_arg)
            .arg(&in_bits_arg);
        // SAFETY: one thread per row reads its pair from each packed factor, the term
        // tables (bounded by offsets), and its e_in[row & mask] / e_out[row >> in_bits]
        // weights (in range since e_in*e_out == rows); shared memory holds `block`
        // tuples as configured above.
        let _ = unsafe { launch.launch(cfg) }?;

        let mut len = blocks as usize;
        while len > 1 {
            let out_blocks = (len as u32).div_ceil(block);
            let mut out: CudaSlice<u64> = self.stream.alloc_zeros(out_blocks as usize * tuple)?;
            let len_arg = len as u64;
            let cfg = LaunchConfig {
                grid_dim: (out_blocks, 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: shared,
            };
            let f = self.round_poly_reduce.clone();
            let mut launch = self.stream.launch_builder(&f);
            let _ = launch.arg(&mut out).arg(&buf).arg(&degree_arg).arg(&len_arg);
            // SAFETY: each block reads up to `block` tuples from `buf` (len total) and
            // writes one tuple to `out`; shared memory holds `block` tuples.
            let _ = unsafe { launch.launch(cfg) }?;
            buf = out;
            len = out_blocks as usize;
        }

        let raw = self.stream.clone_dtoh(&buf.slice(0..tuple))?;
        self.stream.synchronize()?;
        Ok((0..degree)
            .map(|e| limbs_to_fr([raw[e * 4], raw[e * 4 + 1], raw[e * 4 + 2], raw[e * 4 + 3]]))
            .collect())
    }

    pub fn eq_evals(&self, r: &[Fr], scaling_factor: Option<Fr>) -> Result<DeviceFrVec, CudaError> {
        use num_traits::One;
        let scaling = scaling_factor.unwrap_or_else(Fr::one);
        let total = 1usize << r.len();

        let mut cur: CudaSlice<u64> = self.stream.alloc_zeros(total * LIMBS)?;
        self.stream
            .memcpy_htod(&fr_to_limbs(scaling), &mut cur.slice_mut(0..LIMBS))?;
        let mut next: CudaSlice<u64> = self.stream.alloc_zeros(total * LIMBS)?;

        let mut size_in = 1usize;
        for &r_j in r {
            let challenge_dev = self.stream.clone_htod(&fr_to_limbs(r_j))?;
            let size_in_arg = size_in as u64;
            let cfg = LaunchConfig {
                grid_dim: ((size_in as u32).div_ceil(BLOCK), 1, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: 0,
            };
            let f = self.eq_double.clone();
            let mut launch = self.stream.launch_builder(&f);
            let _ = launch
                .arg(&mut next)
                .arg(&cur)
                .arg(&challenge_dev)
                .arg(&size_in_arg);
            // SAFETY: each thread i reads cur[i] (size_in elements) and the single
            // challenge, writing next[2i], next[2i+1]; cur and next are distinct
            // buffers holding >= 2*size_in elements after this round.
            let _ = unsafe { launch.launch(cfg) }?;
            std::mem::swap(&mut cur, &mut next);
            size_in *= 2;
        }
        self.stream.synchronize()?;

        Ok(DeviceFrVec {
            stream: self.stream.clone(),
            buf: cur,
            len: total,
            staging: self.staging.clone(),
        })
    }

    pub fn lt_evals(&self, point: &[Fr]) -> Result<DeviceFrVec, CudaError> {
        use num_traits::Zero;
        let total = 1usize << point.len();
        if point.is_empty() {
            let mut cur: CudaSlice<u64> = self.stream.alloc_zeros(LIMBS)?;
            self.stream
                .memcpy_htod(&fr_to_limbs(Fr::zero()), &mut cur.slice_mut(0..LIMBS))?;
            self.stream.synchronize()?;
            return Ok(DeviceFrVec {
                stream: self.stream.clone(),
                buf: cur,
                len: 1,
                staging: self.staging.clone(),
            });
        }

        let mut cur: CudaSlice<u64> = self.stream.alloc_zeros(total * LIMBS)?;
        let mut next: CudaSlice<u64> = self.stream.alloc_zeros(total * LIMBS)?;
        let mut size_in = 1usize;
        for &r_j in point.iter().rev() {
            let challenge_dev = self.stream.clone_htod(&fr_to_limbs(r_j))?;
            let size_in_arg = size_in as u64;
            let cfg = LaunchConfig {
                grid_dim: ((size_in as u32).div_ceil(BLOCK), 1, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: 0,
            };
            let f = self.lt_double.clone();
            let mut launch = self.stream.launch_builder(&f);
            let _ = launch
                .arg(&mut next)
                .arg(&cur)
                .arg(&challenge_dev)
                .arg(&size_in_arg);
            // SAFETY: each thread i reads cur[i] (size_in elements) and the single challenge,
            // writing next[i] and next[size_in + i]; cur and next are distinct buffers holding
            // >= 2*size_in elements after this round.
            let _ = unsafe { launch.launch(cfg) }?;
            std::mem::swap(&mut cur, &mut next);
            size_in *= 2;
        }
        self.stream.synchronize()?;

        Ok(DeviceFrVec {
            stream: self.stream.clone(),
            buf: cur,
            len: total,
            staging: self.staging.clone(),
        })
    }

    pub fn ram_rw_cycle_round_coefficients(
        &self,
        inputs: RamRwCycleRoundInputs<'_>,
    ) -> Result<(Fr, Fr), CudaError> {
        use num_traits::{One, Zero};
        let items = inputs.items;
        if items == 0 {
            return Ok((Fr::zero(), Fr::zero()));
        }

        let gamma_dev = self.stream.clone_htod(&fr_to_limbs(inputs.gamma))?;
        let one_plus_gamma_dev =
            self.stream.clone_htod(&fr_to_limbs(Fr::one() + inputs.gamma))?;

        const WIDTH: usize = 2;
        let tuple = WIDTH * LIMBS;
        let block = BLOCK;
        let shared = block * tuple as u32 * std::mem::size_of::<u64>() as u32;
        let blocks = (items as u32).div_ceil(block);
        let mut buf: CudaSlice<u64> = self.stream.alloc_zeros(blocks as usize * tuple)?;

        let in_pairs_arg = inputs.in_pairs;
        let items_arg = items as u64;
        let width_arg = WIDTH as u32;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: shared,
        };
        let f = self.ram_rw_cycle_round_pairs.clone();
        let mut launch = self.stream.launch_builder(&f);
        let _ = launch
            .arg(&mut buf)
            .arg(&inputs.val_coeff.buf)
            .arg(&inputs.ra_coeff.buf)
            .arg(&inputs.prev_val.buf)
            .arg(&inputs.next_val.buf)
            .arg(inputs.even_idx)
            .arg(inputs.odd_idx)
            .arg(inputs.pair)
            .arg(&inputs.inc.buf)
            .arg(&inputs.e_in.buf)
            .arg(&inputs.e_out.buf)
            .arg(&gamma_dev)
            .arg(&one_plus_gamma_dev)
            .arg(&in_pairs_arg)
            .arg(&items_arg);
        // SAFETY: one thread per merged-column work-item reads its even/odd entry (or -1
        // for absent) from the entry SoA, the per-pair inc[2*pair]/[2*pair+1] and the
        // e_in/e_out weight, writing one 2-lane tuple per block; shared memory holds
        // `block` tuples.
        let _ = unsafe { launch.launch(cfg) }?;

        let mut len = blocks as usize;
        while len > 1 {
            let out_blocks = (len as u32).div_ceil(block);
            let mut out: CudaSlice<u64> = self.stream.alloc_zeros(out_blocks as usize * tuple)?;
            let len_arg = len as u64;
            let cfg = LaunchConfig {
                grid_dim: (out_blocks, 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: shared,
            };
            let f = self.round_poly_reduce.clone();
            let mut launch = self.stream.launch_builder(&f);
            let _ = launch.arg(&mut out).arg(&buf).arg(&width_arg).arg(&len_arg);
            // SAFETY: round_poly_reduce sums 2-lane tuples across up to `block` blocks;
            // shared memory holds `block` tuples.
            let _ = unsafe { launch.launch(cfg) }?;
            buf = out;
            len = out_blocks as usize;
        }

        let raw = self.stream.clone_dtoh(&buf.slice(0..tuple))?;
        self.stream.synchronize()?;
        Ok((
            limbs_to_fr([raw[0], raw[1], raw[2], raw[3]]),
            limbs_to_fr([raw[4], raw[5], raw[6], raw[7]]),
        ))
    }

    pub fn ram_rw_cycle_bind(
        &self,
        inputs: RamRwCycleBindInputs<'_>,
    ) -> Result<RamRwCycleEntries, CudaError> {
        let items = inputs.items;
        let make = |len: usize| -> Result<DeviceFrVec, CudaError> {
            Ok(DeviceFrVec {
                stream: self.stream.clone(),
                buf: self.stream.alloc_zeros(len * LIMBS)?,
                len,
                staging: self.staging.clone(),
            })
        };
        let mut val_coeff = make(items)?;
        let mut ra_coeff = make(items)?;
        let mut prev_val = make(items)?;
        let mut next_val = make(items)?;
        if items == 0 {
            return Ok(RamRwCycleEntries { val_coeff, ra_coeff, prev_val, next_val });
        }

        let challenge_dev = self.stream.clone_htod(&fr_to_limbs(inputs.challenge))?;

        let block = BLOCK;
        let blocks = (items as u32).div_ceil(block);
        let items_arg = items as u64;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let f = self.ram_rw_cycle_bind.clone();
        let mut launch = self.stream.launch_builder(&f);
        let _ = launch
            .arg(&mut val_coeff.buf)
            .arg(&mut ra_coeff.buf)
            .arg(&mut prev_val.buf)
            .arg(&mut next_val.buf)
            .arg(&inputs.val_coeff.buf)
            .arg(&inputs.ra_coeff.buf)
            .arg(&inputs.prev_val.buf)
            .arg(&inputs.next_val.buf)
            .arg(inputs.even_idx)
            .arg(inputs.odd_idx)
            .arg(&challenge_dev)
            .arg(&items_arg);
        // SAFETY: one thread per output entry reads its even/odd source (or -1 for absent)
        // from the input SoA and writes the linear-eval at `challenge` for val_coeff/ra_coeff
        // plus the carried prev_val/next_val into the freshly allocated output SoA of length
        // `items`; no shared memory.
        let _ = unsafe { launch.launch(cfg) }?;
        self.stream.synchronize()?;
        Ok(RamRwCycleEntries { val_coeff, ra_coeff, prev_val, next_val })
    }

    pub fn ram_rw_address_round_coefficients(
        &self,
        inputs: RamRwAddressRoundInputs<'_>,
    ) -> Result<(Fr, Fr), CudaError> {
        use num_traits::{One, Zero};
        let num_groups = inputs.num_groups;
        if num_groups == 0 {
            return Ok((Fr::zero(), Fr::zero()));
        }

        let one_plus_gamma_dev =
            self.stream.clone_htod(&fr_to_limbs(Fr::one() + inputs.gamma))?;
        let gc_dev = self.stream.clone_htod(&fr_to_limbs(inputs.gamma * inputs.inc0))?;

        const WIDTH: usize = 2;
        let tuple = WIDTH * LIMBS;
        let block = BLOCK;
        let shared = block * tuple as u32 * std::mem::size_of::<u64>() as u32;
        let blocks = (num_groups as u32).div_ceil(block);
        let mut buf: CudaSlice<u64> = self.stream.alloc_zeros(blocks as usize * tuple)?;

        let num_groups_arg = num_groups as u64;
        let width_arg = WIDTH as u32;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: shared,
        };
        let f = self.ram_rw_address_round_pairs.clone();
        let mut launch = self.stream.launch_builder(&f);
        let _ = launch
            .arg(&mut buf)
            .arg(&inputs.ra_coeff.buf)
            .arg(&inputs.val_coeff.buf)
            .arg(&inputs.val_init.buf)
            .arg(inputs.even_idx)
            .arg(inputs.odd_idx)
            .arg(inputs.pair)
            .arg(&one_plus_gamma_dev)
            .arg(&gc_dev)
            .arg(&num_groups_arg);
        // SAFETY: one thread per col-pair group reads its even/odd entry (or -1 for absent,
        // filling the missing side from val_init[2p]/[2p+1]) from the entry SoA, builds the
        // two x in {0,2} evals of ra*((1+gamma)*val + gamma*inc0) in shared memory, and
        // width-2 block-reduces to one tuple per block.
        let _ = unsafe { launch.launch(cfg) }?;

        let mut len = blocks as usize;
        while len > 1 {
            let out_blocks = (len as u32).div_ceil(block);
            let mut out: CudaSlice<u64> = self.stream.alloc_zeros(out_blocks as usize * tuple)?;
            let len_arg = len as u64;
            let cfg = LaunchConfig {
                grid_dim: (out_blocks, 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: shared,
            };
            let f = self.round_poly_reduce.clone();
            let mut launch = self.stream.launch_builder(&f);
            let _ = launch.arg(&mut out).arg(&buf).arg(&width_arg).arg(&len_arg);
            // SAFETY: round_poly_reduce sums 2-lane tuples across up to `block` blocks;
            // shared memory holds `block` tuples.
            let _ = unsafe { launch.launch(cfg) }?;
            buf = out;
            len = out_blocks as usize;
        }

        let raw = self.stream.clone_dtoh(&buf.slice(0..tuple))?;
        self.stream.synchronize()?;
        let q0 = limbs_to_fr([raw[0], raw[1], raw[2], raw[3]]);
        let q2 = limbs_to_fr([raw[4], raw[5], raw[6], raw[7]]);
        Ok((inputs.eq * q0, inputs.eq * q2))
    }

    pub fn ram_rw_address_bind(
        &self,
        inputs: RamRwAddressBindInputs<'_>,
    ) -> Result<RamRwAddressEntries, CudaError> {
        let num_groups = inputs.num_groups;
        let make = |len: usize| -> Result<DeviceFrVec, CudaError> {
            Ok(DeviceFrVec {
                stream: self.stream.clone(),
                buf: self.stream.alloc_zeros(len * LIMBS)?,
                len,
                staging: self.staging.clone(),
            })
        };
        let mut ra_coeff = make(num_groups)?;
        let mut val_coeff = make(num_groups)?;
        let mut prev_val = make(num_groups)?;
        let mut next_val = make(num_groups)?;
        if num_groups == 0 {
            return Ok(RamRwAddressEntries { ra_coeff, val_coeff, prev_val, next_val });
        }

        let challenge_dev = self.stream.clone_htod(&fr_to_limbs(inputs.challenge))?;

        let block = BLOCK;
        let blocks = (num_groups as u32).div_ceil(block);
        let num_groups_arg = num_groups as u64;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let f = self.ram_rw_address_bind.clone();
        let mut launch = self.stream.launch_builder(&f);
        let _ = launch
            .arg(&mut ra_coeff.buf)
            .arg(&mut val_coeff.buf)
            .arg(&mut prev_val.buf)
            .arg(&mut next_val.buf)
            .arg(&inputs.ra_coeff.buf)
            .arg(&inputs.val_coeff.buf)
            .arg(&inputs.prev_val.buf)
            .arg(&inputs.next_val.buf)
            .arg(&inputs.val_init.buf)
            .arg(inputs.even_idx)
            .arg(inputs.odd_idx)
            .arg(inputs.pair)
            .arg(&challenge_dev)
            .arg(&num_groups_arg);
        // SAFETY: one thread per output col-pair group reads its even/odd source (or -1 for
        // absent, filling the missing side from val_init[2p]/[2p+1]) and writes the linear-eval
        // at `challenge` of ra_coeff/val_coeff/prev_val/next_val into the freshly allocated
        // output SoA of length `num_groups`; no shared memory.
        let _ = unsafe { launch.launch(cfg) }?;
        self.stream.synchronize()?;
        Ok(RamRwAddressEntries { ra_coeff, val_coeff, prev_val, next_val })
    }

    pub fn raf_q_scatter(
        &self,
        inputs: RafQScatterInputs<'_>,
    ) -> Result<Vec<DeviceFrVec>, CudaError> {
        let poly_len = inputs.poly_len;
        let trace_len = inputs.trace_len;
        if inputs.weight.len != trace_len
            || poly_len == 0
            || !poly_len.is_power_of_two()
        {
            return Err(CudaError::Unsupported);
        }
        let slots = 5 * poly_len;

        if trace_len == 0 {
            let banks = (0..5)
                .map(|_| self.upload(&vec![Fr::from(0u64); poly_len]))
                .collect::<Result<Vec<_>, _>>()?;
            return Ok(banks);
        }

        let num_workers = trace_len.min(4096);
        let mut worker_banks: CudaSlice<u64> =
            self.stream.alloc_zeros(num_workers * slots * LIMBS)?;
        let mut final_banks: CudaSlice<u64> = self.stream.alloc_zeros(slots * LIMBS)?;

        let scatter_cfg = LaunchConfig {
            grid_dim: ((num_workers as u32).div_ceil(BLOCK), 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let suffix_len_arg = inputs.suffix_len as u64;
        let poly_len_arg = poly_len as u64;
        let trace_len_arg = trace_len as u64;
        let num_workers_arg = num_workers as u64;
        let scatter = self.raf_q_scatter.clone();
        let mut launch = self.stream.launch_builder(&scatter);
        let _ = launch
            .arg(&mut worker_banks)
            .arg(&inputs.weight.buf)
            .arg(inputs.lookup_index_lo)
            .arg(inputs.lookup_index_hi)
            .arg(inputs.is_interleaved)
            .arg(&suffix_len_arg)
            .arg(&poly_len_arg)
            .arg(&trace_len_arg)
            .arg(&num_workers_arg);
        // SAFETY: worker `w` writes only its exclusive `worker_banks[w]` slice of
        // slots*LIMBS u64s; reads weight[c] (trace_len elems) and the resident
        // index/flag arrays (trace_len elems each) it is passed.
        let _ = unsafe { launch.launch(scatter_cfg) }?;

        let reduce_cfg = LaunchConfig {
            grid_dim: ((slots as u32).div_ceil(BLOCK), 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let slots_arg = slots as u64;
        let reduce = self.raf_q_scatter_reduce.clone();
        let mut launch = self.stream.launch_builder(&reduce);
        let _ = launch
            .arg(&mut final_banks)
            .arg(&worker_banks)
            .arg(&slots_arg)
            .arg(&num_workers_arg);
        // SAFETY: thread `slot` sums worker_banks[*][slot] across num_workers and
        // writes final_banks[slot]; slots threads, each buffer holds >= slots*LIMBS
        // (final) / num_workers*slots*LIMBS (worker) u64s.
        let _ = unsafe { launch.launch(reduce_cfg) }?;
        self.stream.synchronize()?;

        let mut banks = Vec::with_capacity(5);
        for i in 0..5 {
            let mut buf: CudaSlice<u64> = self.stream.alloc_zeros(poly_len * LIMBS)?;
            let start = i * poly_len * LIMBS;
            self.stream
                .memcpy_dtod(&final_banks.slice(start..start + poly_len * LIMBS), &mut buf)?;
            banks.push(DeviceFrVec {
                stream: self.stream.clone(),
                buf,
                len: poly_len,
                staging: self.staging.clone(),
            });
        }
        self.stream.synchronize()?;
        Ok(banks)
    }

    pub fn read_suffix_scatter(
        &self,
        inputs: ReadSuffixScatterInputs<'_>,
    ) -> Result<Vec<DeviceFrVec>, CudaError> {
        let poly_len = inputs.poly_len;
        let suffix_count = inputs.suffix_variants.len();
        let m = inputs.m;
        if poly_len == 0 || !poly_len.is_power_of_two() || suffix_count == 0 {
            return Err(CudaError::Unsupported);
        }
        let slots = suffix_count * poly_len;

        if m == 0 {
            let banks = (0..suffix_count)
                .map(|_| self.upload(&vec![Fr::from(0u64); poly_len]))
                .collect::<Result<Vec<_>, _>>()?;
            return Ok(banks);
        }

        let num_workers = m.min(4096);
        let mut worker_banks: CudaSlice<u64> =
            self.stream.alloc_zeros(num_workers * slots * LIMBS)?;
        let mut final_banks: CudaSlice<u64> = self.stream.alloc_zeros(slots * LIMBS)?;

        let scatter_cfg = LaunchConfig {
            grid_dim: ((num_workers as u32).div_ceil(BLOCK), 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let suffix_count_arg = suffix_count as u64;
        let suffix_len_arg = inputs.suffix_len as u64;
        let poly_len_arg = poly_len as u64;
        let m_arg = m as u64;
        let num_workers_arg = num_workers as u64;
        let scatter = self.read_suffix_scatter.clone();
        let mut launch = self.stream.launch_builder(&scatter);
        let _ = launch
            .arg(&mut worker_banks)
            .arg(&inputs.weight.buf)
            .arg(inputs.lookup_index_lo)
            .arg(inputs.lookup_index_hi)
            .arg(inputs.cycle_list)
            .arg(inputs.suffix_variants)
            .arg(&suffix_count_arg)
            .arg(&suffix_len_arg)
            .arg(&poly_len_arg)
            .arg(&m_arg)
            .arg(&num_workers_arg);
        // SAFETY: worker `w` writes only its exclusive `worker_banks[w]` slice of
        // slots*LIMBS u64s; reads weight[cycle_list[j]] (cycle_list values index the
        // resident weight), the resident index arrays, and cycle_list / suffix_variants
        // (m / suffix_count elems) it is passed.
        let _ = unsafe { launch.launch(scatter_cfg) }?;

        let reduce_cfg = LaunchConfig {
            grid_dim: ((slots as u32).div_ceil(BLOCK), 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let slots_arg = slots as u64;
        let reduce = self.raf_q_scatter_reduce.clone();
        let mut launch = self.stream.launch_builder(&reduce);
        let _ = launch
            .arg(&mut final_banks)
            .arg(&worker_banks)
            .arg(&slots_arg)
            .arg(&num_workers_arg);
        // SAFETY: thread `slot` sums worker_banks[*][slot] across num_workers and
        // writes final_banks[slot]; slots threads, each buffer holds >= slots*LIMBS
        // (final) / num_workers*slots*LIMBS (worker) u64s.
        let _ = unsafe { launch.launch(reduce_cfg) }?;
        self.stream.synchronize()?;

        let mut banks = Vec::with_capacity(suffix_count);
        for i in 0..suffix_count {
            let mut buf: CudaSlice<u64> = self.stream.alloc_zeros(poly_len * LIMBS)?;
            let start = i * poly_len * LIMBS;
            self.stream
                .memcpy_dtod(&final_banks.slice(start..start + poly_len * LIMBS), &mut buf)?;
            banks.push(DeviceFrVec {
                stream: self.stream.clone(),
                buf,
                len: poly_len,
                staging: self.staging.clone(),
            });
        }
        self.stream.synchronize()?;
        Ok(banks)
    }

    pub fn raf_weight_phase_update(
        &self,
        weight: &mut DeviceFrVec,
        eq_table: &[Fr],
        lookup_index_lo: &CudaSlice<u64>,
        lookup_index_hi: &CudaSlice<u64>,
        shift: usize,
        mask: usize,
    ) -> Result<(), CudaError> {
        let trace_len = weight.len;
        if eq_table.len() != mask + 1 || !(mask + 1).is_power_of_two() {
            return Err(CudaError::Unsupported);
        }
        if trace_len == 0 {
            return Ok(());
        }

        let eq_dev = self.upload(eq_table)?;
        let shift_arg = shift as u64;
        let mask_arg = mask as u64;
        let trace_len_arg = trace_len as u64;
        let cfg = LaunchConfig {
            grid_dim: ((trace_len as u32).div_ceil(BLOCK), 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let f = self.raf_weight_phase_update.clone();
        let mut launch = self.stream.launch_builder(&f);
        let _ = launch
            .arg(&mut weight.buf)
            .arg(&eq_dev.buf)
            .arg(lookup_index_lo)
            .arg(lookup_index_hi)
            .arg(&shift_arg)
            .arg(&mask_arg)
            .arg(&trace_len_arg);
        // SAFETY: thread c reads weight[c], lookup_index_{lo,hi}[c] (trace_len elems)
        // and eq_table[slot] (slot < mask+1 = eq_table.len()), writing weight[c].
        let _ = unsafe { launch.launch(cfg) }?;
        self.stream.synchronize()?;
        Ok(())
    }

    #[cfg(test)]
    pub fn suffix_mle_probe(
        &self,
        bits_lo: &[u64],
        bits_hi: &[u64],
        len: &[u32],
        variant: &[u32],
    ) -> Result<Vec<u64>, CudaError> {
        let n = bits_lo.len();
        if bits_hi.len() != n || len.len() != n || variant.len() != n {
            return Err(CudaError::Unsupported);
        }
        if n == 0 {
            return Ok(Vec::new());
        }
        let mut out: CudaSlice<u64> = self.stream.alloc_zeros(n)?;
        let lo_dev = self.clone_htod_tracked(bits_lo)?;
        let hi_dev = self.clone_htod_tracked(bits_hi)?;
        let len_dev = self.clone_htod_tracked(len)?;
        let variant_dev = self.clone_htod_tracked(variant)?;
        let n_arg = n as u64;
        let cfg = LaunchConfig {
            grid_dim: ((n as u32).div_ceil(BLOCK), 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let f = self.suffix_mle_probe.clone();
        let mut launch = self.stream.launch_builder(&f);
        let _ = launch
            .arg(&mut out)
            .arg(&lo_dev)
            .arg(&hi_dev)
            .arg(&len_dev)
            .arg(&variant_dev)
            .arg(&n_arg);
        // SAFETY: thread i reads bits_{lo,hi}[i], len[i], variant[i] (n elems each)
        // and writes out[i]; all buffers hold >= n elements.
        let _ = unsafe { launch.launch(cfg) }?;
        self.stream.synchronize()?;
        out.try_into().map_err(|_| CudaError::Pool)
    }

    #[cfg(test)]
    pub fn prefix_combine_probe(
        &self,
        prefixes: &[Fr],
        suffixes: &[Fr],
        suffix_count: &[u32],
        variant: &[u32],
    ) -> Result<Vec<Fr>, CudaError> {
        let n = variant.len();
        if suffix_count.len() != n || prefixes.len() != n * 46 || suffixes.len() != n * 4 {
            return Err(CudaError::Unsupported);
        }
        if n == 0 {
            return Ok(Vec::new());
        }
        let prefixes_dev = self.upload(prefixes)?;
        let suffixes_dev = self.upload(suffixes)?;
        let count_dev = self.clone_htod_tracked(suffix_count)?;
        let variant_dev = self.clone_htod_tracked(variant)?;
        let mut out = DeviceFrVec {
            stream: self.stream.clone(),
            buf: self.stream.alloc_zeros(n * LIMBS)?,
            len: n,
            staging: self.staging.clone(),
        };
        let n_arg = n as u64;
        let cfg = LaunchConfig {
            grid_dim: ((n as u32).div_ceil(BLOCK), 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let f = self.prefix_combine_probe.clone();
        let mut launch = self.stream.launch_builder(&f);
        let _ = launch
            .arg(&mut out.buf)
            .arg(&prefixes_dev.buf)
            .arg(&suffixes_dev.buf)
            .arg(&count_dev)
            .arg(&variant_dev)
            .arg(&n_arg);
        // SAFETY: thread i reads prefixes[i*46..], suffixes[i*4..], suffix_count[i],
        // variant[i] and writes out[i*LIMBS..]; all buffers hold >= n items.
        let _ = unsafe { launch.launch(cfg) }?;
        self.stream.synchronize()?;
        out.to_host()
    }

    pub fn rd_wa_gather(
        &self,
        address_eq: &[Fr],
        addresses: &[i16],
    ) -> Result<DeviceFrVec, CudaError> {
        let trace_len = addresses.len();
        let register_count = address_eq.len();
        let mut out: CudaSlice<u64> = self.stream.alloc_zeros(trace_len.max(1) * LIMBS)?;
        if trace_len == 0 {
            return Ok(DeviceFrVec {
                stream: self.stream.clone(),
                buf: out,
                len: 0,
                staging: self.staging.clone(),
            });
        }
        let address_eq_dev = self.upload(address_eq)?;
        let addresses_dev = self.upload_i16_slice(addresses)?;
        let trace_len_arg = trace_len as u64;
        let register_count_arg = register_count as u64;
        let cfg = LaunchConfig {
            grid_dim: ((trace_len as u32).div_ceil(BLOCK), 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let f = self.rd_wa_gather.clone();
        let mut launch = self.stream.launch_builder(&f);
        let _ = launch
            .arg(&mut out)
            .arg(&address_eq_dev.buf)
            .arg(&addresses_dev)
            .arg(&trace_len_arg)
            .arg(&register_count_arg);
        // SAFETY: one thread per cycle c reads addresses[c] and (if in range) address_eq[addr];
        // out holds trace_len Fr. No shared memory.
        let _ = unsafe { launch.launch(cfg) }?;
        self.stream.synchronize()?;

        Ok(DeviceFrVec {
            stream: self.stream.clone(),
            buf: out,
            len: trace_len,
            staging: self.staging.clone(),
        })
    }

    fn reduce_to_device(&self, sum: bool, values: &DeviceFrVec) -> Result<DeviceFrVec, CudaError> {
        use num_traits::{One, Zero};
        if values.len == 0 {
            return self.upload(&[if sum { Fr::zero() } else { Fr::one() }]);
        }
        if values.len == 1 {
            return values.try_clone();
        }

        let mut owned: Option<CudaSlice<u64>> = None;
        let mut len = values.len;

        while len > 1 {
            let input: &CudaSlice<u64> = owned.as_ref().unwrap_or(&values.buf);
            let blocks = (len as u32).div_ceil(BLOCK);
            let mut out_dev: CudaSlice<u64> =
                self.stream.alloc_zeros(blocks as usize * LIMBS)?;
            let cfg = LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: BLOCK * LIMBS as u32 * std::mem::size_of::<u64>() as u32,
            };
            let len_arg = len as u64;
            let func = if sum {
                self.sum_reduce.clone()
            } else {
                self.product_reduce.clone()
            };
            let mut launch = self.stream.launch_builder(&func);
            let _ = launch.arg(&mut out_dev).arg(input).arg(&len_arg);
            if !sum {
                let _ = launch.arg(&self.one_dev);
            }
            // SAFETY: each block reads up to BLOCK elements from `input` (len total)
            // and writes one element per block to `out_dev` (blocks total). Shared
            // memory holds BLOCK field elements as configured above.
            let _ = unsafe { launch.launch(cfg) }?;

            owned = Some(out_dev);
            len = blocks as usize;
        }

        match owned {
            Some(buf) => Ok(DeviceFrVec {
                stream: self.stream.clone(),
                buf,
                len: 1,
                staging: self.staging.clone(),
            }),
            None => Err(CudaError::Pool),
        }
    }

    pub fn sum(&self, values: &DeviceFrVec) -> Result<Fr, CudaError> {
        let host = self.reduce_to_device(true, values)?.to_host()?;
        Ok(host[0])
    }

    pub fn product(&self, values: &DeviceFrVec) -> Result<Fr, CudaError> {
        let host = self.reduce_to_device(false, values)?.to_host()?;
        Ok(host[0])
    }

    pub fn sum_device(&self, values: &DeviceFrVec) -> Result<DeviceFrVec, CudaError> {
        self.reduce_to_device(true, values)
    }

    pub fn product_device(&self, values: &DeviceFrVec) -> Result<DeviceFrVec, CudaError> {
        self.reduce_to_device(false, values)
    }
}

pub(crate) fn suffix_mle_variants() -> Vec<jolt_lookup_tables::tables::Suffixes> {
    use jolt_lookup_tables::tables::Suffixes as S;
    vec![
        S::One,
        S::And,
        S::AndNot,
        S::Or,
        S::Xor,
        S::RightOperand,
        S::RightOperandW,
        S::ChangeDivisor,
        S::ChangeDivisorW,
        S::UpperWord,
        S::LowerWord,
        S::LowerHalfWord,
        S::LessThan,
        S::GreaterThan,
        S::Eq,
        S::LeftOperandIsZero,
        S::RightOperandIsZero,
        S::Lsb,
        S::DivByZero,
        S::Pow2,
        S::Pow2W,
        S::Rev8W,
        S::RightShiftPadding,
        S::RightShift,
        S::RightShiftHelper,
        S::SignExtension,
        S::LeftShift,
        S::TwoLsb,
        S::SignExtensionUpperHalf,
        S::SignExtensionRightOperand,
        S::RightShiftW,
        S::RightShiftWHelper,
        S::LeftShiftWHelper,
        S::LeftShiftW,
        S::OverflowBitsZero,
        S::XorRot16,
        S::XorRot24,
        S::XorRot32,
        S::XorRot63,
        S::XorRotW7,
        S::XorRotW8,
        S::XorRotW12,
        S::XorRotW16,
    ]
}

pub(crate) fn suffix_variant_code(suffix: jolt_lookup_tables::tables::Suffixes) -> Option<u32> {
    suffix_mle_variants()
        .iter()
        .position(|&v| v == suffix)
        .map(|i| i as u32)
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use jolt_field::Field;
    use num_traits::{One, Zero};
    use proptest::prelude::*;

    fn ctx() -> CudaKernelContext {
        CudaKernelContext::new(0).unwrap()
    }

    fn fr_strategy() -> impl Strategy<Value = Fr> {
        any::<[u8; 32]>().prop_map(|bytes| Fr::from_bytes(&bytes))
    }

    fn fr_vec_strategy(max: usize) -> impl Strategy<Value = Vec<Fr>> {
        prop::collection::vec(fr_strategy(), 0..max)
    }

    #[test]
    fn ram_rw_cycle_round_coefficients_matches_cpu() {
        use crate::split_eq::SplitEqState;
        use crate::stage2::{cycle_low_round_coefficients, RamCycleEntry};

        let log_t = 6usize;
        let t = 1usize << log_t;
        let gamma = Fr::from_u64(9);
        let r_cycle: Vec<Fr> = (0..log_t).map(|i| Fr::from_u64((i + 3) as u64)).collect();

        let raw: [(usize, usize); 12] = [
            (0, 3),
            (1, 3),
            (2, 1),
            (3, 4),
            (4, 2),
            (4, 9),
            (5, 2),
            (6, 5),
            (7, 8),
            (10, 0),
            (10, 6),
            (11, 6),
        ];
        let entries: Vec<RamCycleEntry<Fr>> = raw
            .iter()
            .enumerate()
            .map(|(g, &(row, col))| RamCycleEntry {
                row,
                col,
                prev_val: (g % 5) as u64,
                next_val: (g % 7) as u64,
                val_coeff: Fr::from_u64((g + 1) as u64),
                ra_coeff: Fr::from_u64((g + 100) as u64),
            })
            .collect();
        let inc: Vec<Fr> = (0..t).map(|row| Fr::from_u64((row + 33) as u64)).collect();
        let cycle_eq = SplitEqState::new_low_to_high(&r_cycle, None);
        let expected =
            cycle_low_round_coefficients(&entries, &inc, cycle_eq.e_in(), cycle_eq.e_out(), gamma);

        let val_coeff: Vec<Fr> = entries.iter().map(|e| e.val_coeff).collect();
        let ra_coeff: Vec<Fr> = entries.iter().map(|e| e.ra_coeff).collect();
        let prev_val: Vec<Fr> = entries.iter().map(|e| Fr::from_u64(e.prev_val)).collect();
        let next_val: Vec<Fr> = entries.iter().map(|e| Fr::from_u64(e.next_val)).collect();
        let in_pairs = cycle_eq.e_in().len() / 2;

        let mut even_idx = Vec::new();
        let mut odd_idx = Vec::new();
        let mut pair = Vec::new();
        let mut start = 0;
        while start < entries.len() {
            let p = entries[start].row / 2;
            let mut end = start;
            while end < entries.len() && entries[end].row / 2 == p {
                end += 1;
            }
            let group = &entries[start..end];
            let odd_start = group.partition_point(|e| e.row % 2 == 0);
            let evens: Vec<usize> = (start..start + odd_start).collect();
            let odds: Vec<usize> = (start + odd_start..end).collect();
            let mut push = |ei: i32, oi: i32| {
                even_idx.push(ei);
                odd_idx.push(oi);
                pair.push(p as u32);
            };
            let (mut i, mut j) = (0, 0);
            while i < evens.len() && j < odds.len() {
                match entries[evens[i]].col.cmp(&entries[odds[j]].col) {
                    std::cmp::Ordering::Equal => {
                        push(evens[i] as i32, odds[j] as i32);
                        i += 1;
                        j += 1;
                    }
                    std::cmp::Ordering::Less => {
                        push(evens[i] as i32, -1);
                        i += 1;
                    }
                    std::cmp::Ordering::Greater => {
                        push(-1, odds[j] as i32);
                        j += 1;
                    }
                }
            }
            for &e in &evens[i..] {
                push(e as i32, -1);
            }
            for &o in &odds[j..] {
                push(-1, o as i32);
            }
            start = end;
        }
        let items = even_idx.len();

        let c = ctx();
        let got = c
            .ram_rw_cycle_round_coefficients(RamRwCycleRoundInputs {
                val_coeff: &c.upload(&val_coeff).unwrap(),
                ra_coeff: &c.upload(&ra_coeff).unwrap(),
                prev_val: &c.upload(&prev_val).unwrap(),
                next_val: &c.upload(&next_val).unwrap(),
                even_idx: &c.upload_i32_slice(&even_idx).unwrap(),
                odd_idx: &c.upload_i32_slice(&odd_idx).unwrap(),
                pair: &c.upload_u32_slice(&pair).unwrap(),
                inc: &c.upload(&inc).unwrap(),
                e_in: &c.upload(cycle_eq.e_in()).unwrap(),
                e_out: &c.upload(cycle_eq.e_out()).unwrap(),
                gamma,
                in_pairs: in_pairs as u32,
                items,
            })
            .unwrap();
        assert_eq!(got, expected);
    }

    #[test]
    fn ram_rw_cycle_bind_matches_cpu() {
        use crate::stage2::{bind_cycle_entries_parallel, RamCycleEntry};
        use jolt_field::Field;

        let challenge = Fr::from_u64(7);
        let raw: [(usize, usize); 12] = [
            (0, 3),
            (1, 3),
            (2, 1),
            (3, 4),
            (4, 2),
            (4, 9),
            (5, 2),
            (6, 5),
            (7, 8),
            (10, 0),
            (10, 6),
            (11, 6),
        ];
        let entries: Vec<RamCycleEntry<Fr>> = raw
            .iter()
            .enumerate()
            .map(|(g, &(row, col))| RamCycleEntry {
                row,
                col,
                prev_val: (g % 5) as u64,
                next_val: (g % 7) as u64,
                val_coeff: Fr::from_u64((g + 1) as u64),
                ra_coeff: Fr::from_u64((g + 100) as u64),
            })
            .collect();
        let expected = bind_cycle_entries_parallel(&entries, challenge);

        let val_coeff: Vec<Fr> = entries.iter().map(|e| e.val_coeff).collect();
        let ra_coeff: Vec<Fr> = entries.iter().map(|e| e.ra_coeff).collect();
        let prev_val: Vec<Fr> = entries.iter().map(|e| Fr::from_u64(e.prev_val)).collect();
        let next_val: Vec<Fr> = entries.iter().map(|e| Fr::from_u64(e.next_val)).collect();

        let mut even_idx = Vec::new();
        let mut odd_idx = Vec::new();
        let mut start = 0;
        while start < entries.len() {
            let p = entries[start].row / 2;
            let mut end = start;
            while end < entries.len() && entries[end].row / 2 == p {
                end += 1;
            }
            let group = &entries[start..end];
            let odd_start = group.partition_point(|e| e.row % 2 == 0);
            let evens: Vec<usize> = (start..start + odd_start).collect();
            let odds: Vec<usize> = (start + odd_start..end).collect();
            let mut push = |ei: i32, oi: i32| {
                even_idx.push(ei);
                odd_idx.push(oi);
            };
            let (mut i, mut j) = (0, 0);
            while i < evens.len() && j < odds.len() {
                match entries[evens[i]].col.cmp(&entries[odds[j]].col) {
                    std::cmp::Ordering::Equal => {
                        push(evens[i] as i32, odds[j] as i32);
                        i += 1;
                        j += 1;
                    }
                    std::cmp::Ordering::Less => {
                        push(evens[i] as i32, -1);
                        i += 1;
                    }
                    std::cmp::Ordering::Greater => {
                        push(-1, odds[j] as i32);
                        j += 1;
                    }
                }
            }
            for &e in &evens[i..] {
                push(e as i32, -1);
            }
            for &o in &odds[j..] {
                push(-1, o as i32);
            }
            start = end;
        }
        let items = even_idx.len();
        assert_eq!(items, expected.len());

        let c = ctx();
        let got = c
            .ram_rw_cycle_bind(RamRwCycleBindInputs {
                val_coeff: &c.upload(&val_coeff).unwrap(),
                ra_coeff: &c.upload(&ra_coeff).unwrap(),
                prev_val: &c.upload(&prev_val).unwrap(),
                next_val: &c.upload(&next_val).unwrap(),
                even_idx: &c.upload_i32_slice(&even_idx).unwrap(),
                odd_idx: &c.upload_i32_slice(&odd_idx).unwrap(),
                challenge,
                items,
            })
            .unwrap();

        let got_val = got.val_coeff.to_host().unwrap();
        let got_ra = got.ra_coeff.to_host().unwrap();
        let got_prev = got.prev_val.to_host().unwrap();
        let got_next = got.next_val.to_host().unwrap();
        for (slot, e) in expected.iter().enumerate() {
            assert_eq!(got_val[slot], e.val_coeff, "val_coeff slot {slot}");
            assert_eq!(got_ra[slot], e.ra_coeff, "ra_coeff slot {slot}");
            assert_eq!(got_prev[slot], Fr::from_u64(e.prev_val), "prev_val slot {slot}");
            assert_eq!(got_next[slot], Fr::from_u64(e.next_val), "next_val slot {slot}");
        }
    }

    #[test]
    fn ram_rw_address_round_coefficients_matches_cpu() {
        use crate::split_eq::SplitEqState;
        use crate::stage2::{RamAddressEntry, RamReadWriteState};
        use jolt_field::Field;
        use jolt_poly::UnivariatePoly;

        let gamma = Fr::from_u64(9);
        let inc0 = Fr::from_u64(17);
        let previous_claim = Fr::from_u64(5);
        let cols: [usize; 7] = [0, 1, 2, 5, 6, 7, 11];
        let entries: Vec<RamAddressEntry<Fr>> = cols
            .iter()
            .enumerate()
            .map(|(g, &col)| RamAddressEntry {
                row: 0,
                col,
                prev_val: Fr::from_u64((g + 1) as u64),
                next_val: Fr::from_u64((g + 20) as u64),
                val_coeff: Fr::from_u64((g + 40) as u64),
                ra_coeff: Fr::from_u64((g + 60) as u64),
            })
            .collect();
        let val_init: Vec<Fr> = (0..16).map(|i| Fr::from_u64((i + 3) as u64)).collect();

        let point: Vec<Fr> = (0..4).map(|i| Fr::from_u64((i + 2) as u64)).collect();
        let cycle_eq = SplitEqState::new_low_to_high(&point, None);
        let eq = cycle_eq.eval();

        let state = RamReadWriteState {
            gamma,
            log_t: point.len(),
            round: point.len(),
            cycle_eq,
            cycle_entries: Vec::new(),
            address_entries: entries.clone(),
            address_scratch: Vec::new(),
            inc: vec![inc0],
            inc_scratch: Vec::new(),
            val_init: val_init.clone(),
            val_init_scratch: Vec::new(),
            cuda: None,
        };
        let expected = state.address_round_poly(previous_claim);

        let ra_coeff: Vec<Fr> = entries.iter().map(|e| e.ra_coeff).collect();
        let val_coeff: Vec<Fr> = entries.iter().map(|e| e.val_coeff).collect();

        let mut even_idx = Vec::new();
        let mut odd_idx = Vec::new();
        let mut pair = Vec::new();
        let mut start = 0;
        while start < entries.len() {
            let p = entries[start].col / 2;
            let mut end = start;
            while end < entries.len() && entries[end].col / 2 == p {
                end += 1;
            }
            let group = &entries[start..end];
            let odd_start = group.partition_point(|e| e.col % 2 == 0);
            let ei = if odd_start > 0 { start as i32 } else { -1 };
            let oi = if odd_start < group.len() {
                (start + odd_start) as i32
            } else {
                -1
            };
            even_idx.push(ei);
            odd_idx.push(oi);
            pair.push(p as u32);
            start = end;
        }
        let num_groups = pair.len();

        let c = ctx();
        let (q0, q2) = c
            .ram_rw_address_round_coefficients(RamRwAddressRoundInputs {
                ra_coeff: &c.upload(&ra_coeff).unwrap(),
                val_coeff: &c.upload(&val_coeff).unwrap(),
                val_init: &c.upload(&val_init).unwrap(),
                even_idx: &c.upload_i32_slice(&even_idx).unwrap(),
                odd_idx: &c.upload_i32_slice(&odd_idx).unwrap(),
                pair: &c.upload_u32_slice(&pair).unwrap(),
                eq,
                gamma,
                inc0,
                num_groups,
            })
            .unwrap();
        let got = UnivariatePoly::from_evals_and_hint(previous_claim, &[q0, q2]);
        assert_eq!(got, expected);
    }

    #[test]
    fn ram_rw_address_bind_matches_cpu() {
        use crate::split_eq::SplitEqState;
        use crate::stage2::{RamAddressEntry, RamReadWriteState};
        use jolt_field::Field;

        let gamma = Fr::from_u64(9);
        let challenge = Fr::from_u64(7);
        let cols: [usize; 7] = [0, 1, 2, 5, 6, 7, 11];
        let entries: Vec<RamAddressEntry<Fr>> = cols
            .iter()
            .enumerate()
            .map(|(g, &col)| RamAddressEntry {
                row: 0,
                col,
                prev_val: Fr::from_u64((g + 1) as u64),
                next_val: Fr::from_u64((g + 20) as u64),
                val_coeff: Fr::from_u64((g + 40) as u64),
                ra_coeff: Fr::from_u64((g + 60) as u64),
            })
            .collect();
        let val_init: Vec<Fr> = (0..16).map(|i| Fr::from_u64((i + 3) as u64)).collect();

        let ra_coeff: Vec<Fr> = entries.iter().map(|e| e.ra_coeff).collect();
        let val_coeff: Vec<Fr> = entries.iter().map(|e| e.val_coeff).collect();
        let prev_val: Vec<Fr> = entries.iter().map(|e| e.prev_val).collect();
        let next_val: Vec<Fr> = entries.iter().map(|e| e.next_val).collect();

        let point: Vec<Fr> = (0..4).map(|i| Fr::from_u64((i + 2) as u64)).collect();
        let mut state = RamReadWriteState {
            gamma,
            log_t: point.len(),
            round: point.len(),
            cycle_eq: SplitEqState::new_low_to_high(&point, None),
            cycle_entries: Vec::new(),
            address_entries: entries.clone(),
            address_scratch: Vec::new(),
            inc: vec![Fr::from_u64(17)],
            inc_scratch: Vec::new(),
            val_init: val_init.clone(),
            val_init_scratch: Vec::new(),
            cuda: None,
        };
        state.bind_address(challenge);
        let expected = &state.address_entries;

        let mut even_idx = Vec::new();
        let mut odd_idx = Vec::new();
        let mut pair = Vec::new();
        let mut start = 0;
        while start < entries.len() {
            let p = entries[start].col / 2;
            let mut end = start;
            while end < entries.len() && entries[end].col / 2 == p {
                end += 1;
            }
            let group = &entries[start..end];
            let odd_start = group.partition_point(|e| e.col % 2 == 0);
            let ei = if odd_start > 0 { start as i32 } else { -1 };
            let oi = if odd_start < group.len() {
                (start + odd_start) as i32
            } else {
                -1
            };
            even_idx.push(ei);
            odd_idx.push(oi);
            pair.push(p as u32);
            start = end;
        }
        let num_groups = pair.len();
        assert_eq!(num_groups, expected.len());

        let c = ctx();
        let got = c
            .ram_rw_address_bind(RamRwAddressBindInputs {
                ra_coeff: &c.upload(&ra_coeff).unwrap(),
                val_coeff: &c.upload(&val_coeff).unwrap(),
                prev_val: &c.upload(&prev_val).unwrap(),
                next_val: &c.upload(&next_val).unwrap(),
                val_init: &c.upload(&val_init).unwrap(),
                even_idx: &c.upload_i32_slice(&even_idx).unwrap(),
                odd_idx: &c.upload_i32_slice(&odd_idx).unwrap(),
                pair: &c.upload_u32_slice(&pair).unwrap(),
                challenge,
                num_groups,
            })
            .unwrap();

        let got_ra = got.ra_coeff.to_host().unwrap();
        let got_val = got.val_coeff.to_host().unwrap();
        let got_prev = got.prev_val.to_host().unwrap();
        let got_next = got.next_val.to_host().unwrap();
        for (slot, e) in expected.iter().enumerate() {
            assert_eq!(got_ra[slot], e.ra_coeff, "ra_coeff slot {slot}");
            assert_eq!(got_val[slot], e.val_coeff, "val_coeff slot {slot}");
            assert_eq!(got_prev[slot], e.prev_val, "prev_val slot {slot}");
            assert_eq!(got_next[slot], e.next_val, "next_val slot {slot}");
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 16, .. ProptestConfig::default()
        })]

        #[test]
        fn add_matches_cpu(a in fr_vec_strategy(300)) {
            let b = a.iter().rev().copied().collect::<Vec<_>>();
            let expected: Vec<Fr> = a.iter().zip(&b).map(|(x, y)| *x + *y).collect();
            let c = ctx();
            let mut a_dev = c.upload(&a).unwrap();
            c.add(&mut a_dev, &c.upload(&b).unwrap()).unwrap();
            prop_assert_eq!(a_dev.to_host().unwrap(), expected);
        }

        #[test]
        fn sub_matches_cpu(a in fr_vec_strategy(300)) {
            let b = a.iter().rev().copied().collect::<Vec<_>>();
            let expected: Vec<Fr> = a.iter().zip(&b).map(|(x, y)| *x - *y).collect();
            let c = ctx();
            let mut a_dev = c.upload(&a).unwrap();
            c.sub(&mut a_dev, &c.upload(&b).unwrap()).unwrap();
            prop_assert_eq!(a_dev.to_host().unwrap(), expected);
        }

        #[test]
        fn mul_matches_cpu(a in fr_vec_strategy(300)) {
            let b = a.iter().rev().copied().collect::<Vec<_>>();
            let expected: Vec<Fr> = a.iter().zip(&b).map(|(x, y)| *x * *y).collect();
            let c = ctx();
            let mut a_dev = c.upload(&a).unwrap();
            c.mul(&mut a_dev, &c.upload(&b).unwrap()).unwrap();
            prop_assert_eq!(a_dev.to_host().unwrap(), expected);
        }

        #[test]
        fn fma_matches_cpu(a in fr_vec_strategy(300)) {
            let b = a.iter().rev().copied().collect::<Vec<_>>();
            let cc = a.iter().map(|x| *x + Fr::from_u64(1)).collect::<Vec<_>>();
            let expected: Vec<Fr> = a
                .iter()
                .zip(&b)
                .zip(&cc)
                .map(|((x, y), z)| *x * *y + *z)
                .collect();
            let c = ctx();
            let mut a_dev = c.upload(&a).unwrap();
            c.fma(&mut a_dev, &c.upload(&b).unwrap(), &c.upload(&cc).unwrap())
                .unwrap();
            prop_assert_eq!(a_dev.to_host().unwrap(), expected);
        }

        #[test]
        fn sum_matches_cpu(a in fr_vec_strategy(2000)) {
            let expected: Fr = a.iter().copied().sum();
            let c = ctx();
            let got = c.sum(&c.upload(&a).unwrap()).unwrap();
            prop_assert_eq!(got, expected);
        }

        #[test]
        fn product_matches_cpu(a in fr_vec_strategy(2000)) {
            let expected: Fr = a.iter().copied().product();
            let c = ctx();
            let got = c.product(&c.upload(&a).unwrap()).unwrap();
            prop_assert_eq!(got, expected);
        }

        #[test]
        fn sum_device_matches_cpu(a in fr_vec_strategy(2000)) {
            let expected: Fr = a.iter().copied().sum();
            let c = ctx();
            let got = c.sum_device(&c.upload(&a).unwrap()).unwrap();
            prop_assert_eq!(got.len(), 1);
            prop_assert_eq!(got.to_host().unwrap(), vec![expected]);
        }

        #[test]
        fn product_device_matches_cpu(a in fr_vec_strategy(2000)) {
            let expected: Fr = a.iter().copied().product();
            let c = ctx();
            let got = c.product_device(&c.upload(&a).unwrap()).unwrap();
            prop_assert_eq!(got.len(), 1);
            prop_assert_eq!(got.to_host().unwrap(), vec![expected]);
        }

        #[test]
        fn cubic_accumulate_matches_cpu(
            (eq, az, bz) in (1usize..600).prop_flat_map(|pairs| {
                let len = pairs * 2;
                (
                    prop::collection::vec(fr_strategy(), len),
                    prop::collection::vec(fr_strategy(), len),
                    prop::collection::vec(fr_strategy(), len),
                )
            })
        ) {
            let expected = cpu_cubic(&eq, &az, &bz);
            let c = ctx();
            let got = c
                .cubic_accumulate(
                    &c.upload(&eq).unwrap(),
                    &c.upload(&az).unwrap(),
                    &c.upload(&bz).unwrap(),
                )
                .unwrap();
            prop_assert_eq!(got, expected);
        }
    }

    fn cpu_cubic(eq: &[Fr], az: &[Fr], bz: &[Fr]) -> [Fr; 4] {
        let mut c = [Fr::zero(); 4];
        for ((eq_pair, az_pair), bz_pair) in eq
            .chunks_exact(2)
            .zip(az.chunks_exact(2))
            .zip(bz.chunks_exact(2))
        {
            let eq0 = eq_pair[0];
            let eqd = eq_pair[1] - eq_pair[0];
            let az0 = az_pair[0];
            let azd = az_pair[1] - az_pair[0];
            let bz0 = bz_pair[0];
            let bzd = bz_pair[1] - bz_pair[0];
            let az0bz0 = az0 * bz0;
            let azdbz0 = azd * bz0;
            let az0bzd = az0 * bzd;
            let azdbzd = azd * bzd;
            c[0] += eq0 * az0bz0;
            c[1] += eqd * az0bz0 + eq0 * azdbz0 + eq0 * az0bzd;
            c[2] += eqd * azdbz0 + eqd * az0bzd + eq0 * azdbzd;
            c[3] += eqd * azdbzd;
        }
        c
    }

    struct Csr {
        offsets: Vec<u32>,
        vars: Vec<u32>,
        coeffs: Vec<Fr>,
    }

    // `rows` lists each row's (var, coeff) nonzeros.
    fn build_csr(rows: &[Vec<(u32, Fr)>]) -> Csr {
        let mut offsets = vec![0u32];
        let mut vars = Vec::new();
        let mut coeffs = Vec::new();
        for row in rows {
            for &(var, coeff) in row {
                vars.push(var);
                coeffs.push(coeff);
            }
            offsets.push(vars.len() as u32);
        }
        Csr {
            offsets,
            vars,
            coeffs,
        }
    }

    fn cpu_row_dots(
        witness: &[Fr],
        csr: &Csr,
        row_count: usize,
        num_vars_padded: usize,
        num_cycles: usize,
    ) -> Vec<Fr> {
        let mut out = vec![Fr::zero(); num_cycles * row_count];
        for cycle in 0..num_cycles {
            let base = cycle * num_vars_padded;
            for row in 0..row_count {
                let start = csr.offsets[row] as usize;
                let end = csr.offsets[row + 1] as usize;
                let mut acc = Fr::zero();
                for k in start..end {
                    acc += csr.coeffs[k] * witness[base + csr.vars[k] as usize];
                }
                out[cycle * row_count + row] = acc;
            }
        }
        out
    }

    const ROW_DOTS_NUM_VARS: usize = 32;

    fn row_strategy() -> impl Strategy<Value = Vec<(u32, Fr)>> {
        prop::collection::vec(
            (0u32..ROW_DOTS_NUM_VARS as u32, fr_strategy()),
            0..6,
        )
    }

    proptest! {
        #[test]
        fn compute_row_dots_matches_cpu(
            log_cycles in 0usize..8,
            a_rows in prop::collection::vec(row_strategy(), 1..24),
            b_rows in prop::collection::vec(row_strategy(), 1..24),
            seed in fr_strategy(),
        ) {
            let row_count = a_rows.len().min(b_rows.len());
            let a = build_csr(&a_rows[..row_count]);
            let b = build_csr(&b_rows[..row_count]);

            let num_cycles = 1usize << log_cycles;
            let num_vars_padded = ROW_DOTS_NUM_VARS.next_power_of_two();
            let witness: Vec<Fr> = (0..num_cycles * num_vars_padded)
                .map(|i| seed + Fr::from_u64(i as u64))
                .collect();

            let expected_a = cpu_row_dots(&witness, &a, row_count, num_vars_padded, num_cycles);
            let expected_b = cpu_row_dots(&witness, &b, row_count, num_vars_padded, num_cycles);

            let c = ctx();
            let (got_a, got_b) = c
                .compute_row_dots(
                    &witness,
                    &a.offsets,
                    &a.vars,
                    &a.coeffs,
                    &b.offsets,
                    &b.vars,
                    &b.coeffs,
                    row_count,
                    num_vars_padded,
                    num_cycles,
                )
                .unwrap();
            prop_assert_eq!(got_a, expected_a);
            prop_assert_eq!(got_b, expected_b);
        }
    }

    fn flatten_terms(terms: &[(Fr, Vec<u32>)]) -> (Vec<Fr>, Vec<u32>, Vec<u32>) {
        let mut coeffs = Vec::new();
        let mut offsets = vec![0u32];
        let mut indices = Vec::new();
        for (coeff, factor_indices) in terms {
            coeffs.push(*coeff);
            indices.extend_from_slice(factor_indices);
            offsets.push(indices.len() as u32);
        }
        (coeffs, offsets, indices)
    }

    proptest! {
        #[test]
        fn sum_of_products_round_poly_matches_cpu(
            log_pairs in 0usize..10,
            num_factors in 1usize..6,
            terms_spec in prop::collection::vec(
                (fr_strategy(), prop::collection::vec(0u32..6, 1..4)),
                1..5,
            ),
            seed in fr_strategy(),
        ) {
            let half = 1usize << log_pairs;
            let len = half * 2;
            let terms_spec: Vec<(Fr, Vec<u32>)> = terms_spec
                .into_iter()
                .map(|(coeff, idxs)| {
                    (coeff, idxs.into_iter().map(|i| i % num_factors as u32).collect())
                })
                .collect();
            let degree = terms_spec.iter().map(|(_, idxs)| idxs.len()).max().unwrap();

            let factors: Vec<Vec<Fr>> = (0..num_factors)
                .map(|f| (0..len).map(|i| seed + Fr::from_u64((f * len + i) as u64)).collect())
                .collect();

            let cpu_terms: Vec<crate::stage6::DenseTerm<Fr>> = terms_spec
                .iter()
                .map(|(coeff, idxs)| crate::stage6::DenseTerm {
                    coefficient: *coeff,
                    factors: idxs.iter().map(|&i| i as usize).collect(),
                })
                .collect();
            let mut expected = vec![Fr::zero(); degree];
            let mut row_evals = vec![Fr::zero(); degree];
            for row in 0..half {
                for e in &mut row_evals {
                    *e = Fr::zero();
                }
                crate::stage6::accumulate_dense_row_evaluations(
                    &factors,
                    &cpu_terms,
                    row,
                    &mut row_evals,
                );
                for (acc, e) in expected.iter_mut().zip(&row_evals) {
                    *acc += *e;
                }
            }

            let (term_coeffs, term_factor_offsets, term_factor_indices) =
                flatten_terms(&terms_spec);
            let c = ctx();
            let factor_devs: Vec<DeviceFrVec> =
                factors.iter().map(|f| c.upload(f).unwrap()).collect();
            let factor_refs: Vec<&DeviceFrVec> = factor_devs.iter().collect();
            let got = c
                .sum_of_products_round_poly(RoundPolyTerms {
                    factors: &factor_refs,
                    term_coeffs: &c.upload(&term_coeffs).unwrap(),
                    term_factor_offsets: &term_factor_offsets,
                    term_factor_indices: &term_factor_indices,
                    degree,
                })
                .unwrap();
            prop_assert_eq!(got, expected);
        }

        #[test]
        fn dense_product_round_poly_matches_cpu(
            log_pairs in 0usize..10,
            degree in 1usize..4,
            seed in fr_strategy(),
        ) {
            let half = 1usize << log_pairs;
            let len = half * 2;
            let factors: Vec<Vec<Fr>> = (0..degree)
                .map(|f| (0..len).map(|i| seed + Fr::from_u64((f * len + i) as u64)).collect())
                .collect();

            let factor_slices: Vec<&[Fr]> = factors.iter().map(Vec::as_slice).collect();
            let expected = crate::stage2::round_poly_from_factor_slices(&factor_slices, degree)
                .into_coefficients();

            let single_term: Vec<(Fr, Vec<u32>)> =
                vec![(Fr::one(), (0..degree as u32).collect())];
            let (term_coeffs, term_factor_offsets, term_factor_indices) =
                flatten_terms(&single_term);
            let c = ctx();
            let factor_devs: Vec<DeviceFrVec> =
                factors.iter().map(|f| c.upload(f).unwrap()).collect();
            let factor_refs: Vec<&DeviceFrVec> = factor_devs.iter().collect();
            let got = c
                .dense_product_round_poly(RoundPolyTerms {
                    factors: &factor_refs,
                    term_coeffs: &c.upload(&term_coeffs).unwrap(),
                    term_factor_offsets: &term_factor_offsets,
                    term_factor_indices: &term_factor_indices,
                    degree,
                })
                .unwrap();
            prop_assert_eq!(got, expected);
        }

        #[test]
        fn gruen_round_poly_instruction_input_matches_cpu(
            num_vars in 1usize..9,
            gamma in fr_strategy(),
            seed in fr_strategy(),
        ) {
            use crate::split_eq::SplitEqState;
            use crate::stage3::instruction_input_split_round_coefficients;

            const NUM_FACTORS: usize = 8;
            let point: Vec<Fr> = (0..num_vars)
                .map(|i| seed + Fr::from_u64(i as u64))
                .collect();
            let mut split_eq = SplitEqState::<Fr>::new_low_to_high(&point, None);
            let mut factors: Vec<Vec<Fr>> = (0..NUM_FACTORS)
                .map(|f| {
                    (0..(1usize << num_vars))
                        .map(|i| seed + Fr::from_u64((f * 977 + i + 1) as u64))
                        .collect()
                })
                .collect();
            let mut scratch: Vec<Vec<Fr>> = vec![Vec::new(); NUM_FACTORS];

            let terms: Vec<(Fr, Vec<u32>)> = vec![
                (Fr::one(), vec![0, 1]),
                (Fr::one(), vec![2, 3]),
                (gamma, vec![4, 5]),
                (gamma, vec![6, 7]),
            ];
            let (term_coeffs, term_factor_offsets, term_factor_indices) = flatten_terms(&terms);
            let c = ctx();

            for _round in 0..num_vars {
                let (q_constant, q_quadratic) =
                    instruction_input_split_round_coefficients(&factors, &split_eq, gamma);

                let factor_devs: Vec<DeviceFrVec> =
                    factors.iter().map(|f| c.upload(f).unwrap()).collect();
                let factor_refs: Vec<&DeviceFrVec> = factor_devs.iter().collect();
                let e_in = c.upload(split_eq.e_in()).unwrap();
                let e_out = c.upload(split_eq.e_out()).unwrap();
                let term_coeffs_dev = c.upload(&term_coeffs).unwrap();
                let got = c
                    .gruen_round_poly(GruenRoundPolyInputs {
                        factors: &factor_refs,
                        term_coeffs: &term_coeffs_dev,
                        term_factor_offsets: &term_factor_offsets,
                        term_factor_indices: &term_factor_indices,
                        e_in: &e_in,
                        e_out: &e_out,
                        degree: 2,
                    })
                    .unwrap();
                prop_assert_eq!(got[0], q_constant);
                prop_assert_eq!(got[1], q_quadratic);

                let challenge = seed + Fr::from_u64(_round as u64 + 100);
                for (factor, scratch) in factors.iter_mut().zip(&mut scratch) {
                    crate::dense::bind_dense_evals_reuse(factor, scratch, challenge);
                }
                split_eq.bind(challenge);
            }
        }

        #[test]
        fn gruen_round_poly_registers_matches_cpu(
            num_vars in 1usize..9,
            gamma in fr_strategy(),
            gamma2 in fr_strategy(),
            seed in fr_strategy(),
        ) {
            use crate::split_eq::SplitEqState;
            use crate::stage3::registers_split_round_constant;

            const NUM_FACTORS: usize = 3;
            let point: Vec<Fr> = (0..num_vars)
                .map(|i| seed + Fr::from_u64(i as u64))
                .collect();
            let mut split_eq = SplitEqState::<Fr>::new_low_to_high(&point, None);
            let mut factors: Vec<Vec<Fr>> = (0..NUM_FACTORS)
                .map(|f| {
                    (0..(1usize << num_vars))
                        .map(|i| seed + Fr::from_u64((f * 977 + i + 1) as u64))
                        .collect()
                })
                .collect();
            let mut scratch: Vec<Vec<Fr>> = vec![Vec::new(); NUM_FACTORS];

            let terms: Vec<(Fr, Vec<u32>)> =
                vec![(Fr::one(), vec![0]), (gamma, vec![1]), (gamma2, vec![2])];
            let (term_coeffs, term_factor_offsets, term_factor_indices) = flatten_terms(&terms);
            let c = ctx();

            for _round in 0..num_vars {
                let q_constant =
                    registers_split_round_constant(&factors, &split_eq, gamma, gamma2);

                let factor_devs: Vec<DeviceFrVec> =
                    factors.iter().map(|f| c.upload(f).unwrap()).collect();
                let factor_refs: Vec<&DeviceFrVec> = factor_devs.iter().collect();
                let e_in = c.upload(split_eq.e_in()).unwrap();
                let e_out = c.upload(split_eq.e_out()).unwrap();
                let term_coeffs_dev = c.upload(&term_coeffs).unwrap();
                let got = c
                    .gruen_round_poly(GruenRoundPolyInputs {
                        factors: &factor_refs,
                        term_coeffs: &term_coeffs_dev,
                        term_factor_offsets: &term_factor_offsets,
                        term_factor_indices: &term_factor_indices,
                        e_in: &e_in,
                        e_out: &e_out,
                        degree: 1,
                    })
                    .unwrap();
                prop_assert_eq!(got[0], q_constant);

                let challenge = seed + Fr::from_u64(_round as u64 + 100);
                for (factor, scratch) in factors.iter_mut().zip(&mut scratch) {
                    crate::dense::bind_dense_evals_reuse(factor, scratch, challenge);
                }
                split_eq.bind(challenge);
            }
        }

        #[test]
        fn eq_evals_matches_cpu(
            r in prop::collection::vec(fr_strategy(), 0..12),
            scaling in prop::option::of(fr_strategy()),
        ) {
            let expected = jolt_poly::EqPolynomial::<Fr>::evals(&r, scaling);
            let c = ctx();
            let got = c.eq_evals(&r, scaling).unwrap().to_host().unwrap();
            prop_assert_eq!(got, expected);
        }

        #[test]
        fn read_suffix_scatter_matches_cpu(
            log_cycles in 1usize..11,
            suffix_len in 1usize..40,
            seed in fr_strategy(),
        ) {
            use jolt_lookup_tables::LookupBits;
            use std::panic::{catch_unwind, AssertUnwindSafe};

            let variant_codes: Vec<u32> = vec![0, 4, 12, 25, 22];
            let variants = super::suffix_mle_variants();
            let suffix_count = variant_codes.len();
            let poly_len = 1usize << 8;
            let m = 1usize << log_cycles;
            let suffix_mask: u128 = if suffix_len >= 128 {
                u128::MAX
            } else {
                (1u128 << suffix_len) - 1
            };
            let index_mask: u128 = if (suffix_len + 8) < 128 {
                (1u128 << (suffix_len + 8)) - 1
            } else {
                u128::MAX
            };

            let weight: Vec<Fr> = (0..m).map(|c| seed + Fr::from_u64((c + 1) as u64)).collect();
            let lookup_index: Vec<u128> = (0..m)
                .map(|c| ((c as u128).wrapping_mul(0x9e37_79b1) ^ (c as u128) << 19) & index_mask)
                .collect();
            let cycle_list: Vec<u32> = (0..m as u32).collect();

            let prev_hook = std::panic::take_hook();
            std::panic::set_hook(Box::new(|_| {}));
            let mut expected = vec![Fr::zero(); suffix_count * poly_len];
            let mut defined = true;
            'outer: for c in 0..m {
                let index = ((lookup_index[c] >> suffix_len) as usize) & (poly_len - 1);
                let suffix_bits = LookupBits::new(lookup_index[c] & suffix_mask, suffix_len);
                for (s, &code) in variant_codes.iter().enumerate() {
                    let Ok(mle) =
                        catch_unwind(AssertUnwindSafe(|| variants[code as usize].suffix_mle(suffix_bits)))
                    else {
                        defined = false;
                        break 'outer;
                    };
                    if mle != 0 {
                        expected[s * poly_len + index] += weight[c] * Fr::from_u64(mle);
                    }
                }
            }
            std::panic::set_hook(prev_hook);
            prop_assume!(defined);

            let c = ctx();
            let weight_dev = c.upload(&weight).unwrap();
            let lo: Vec<u64> = lookup_index.iter().map(|&v| v as u64).collect();
            let hi: Vec<u64> = lookup_index.iter().map(|&v| (v >> 64) as u64).collect();
            let lo_dev = c.upload_u64_slice(&lo).unwrap();
            let hi_dev = c.upload_u64_slice(&hi).unwrap();
            let cycle_dev = c.upload_u32_slice(&cycle_list).unwrap();
            let variant_dev = c.upload_u32_slice(&variant_codes).unwrap();
            let banks = c
                .read_suffix_scatter(ReadSuffixScatterInputs {
                    weight: &weight_dev,
                    lookup_index_lo: &lo_dev,
                    lookup_index_hi: &hi_dev,
                    cycle_list: &cycle_dev,
                    suffix_variants: &variant_dev,
                    m: cycle_list.len(),
                    suffix_len,
                    poly_len,
                })
                .unwrap();
            prop_assert_eq!(banks.len(), suffix_count);
            let mut got = Vec::with_capacity(suffix_count * poly_len);
            for bank in &banks {
                got.extend(bank.to_host().unwrap());
            }
            prop_assert_eq!(got, expected);
        }

        #[test]
        fn raf_q_scatter_matches_cpu(
            log_trace in 1usize..12,
            suffix_len in 1usize..40,
            seed in fr_strategy(),
        ) {
            use jolt_lookup_tables::uninterleave_bits;

            let trace_len = 1usize << log_trace;
            let poly_len = 1usize << 8;
            let suffix_mask: u128 = if suffix_len >= 128 {
                u128::MAX
            } else {
                (1u128 << suffix_len) - 1
            };
            let index_mask: u128 = if (suffix_len + 8) < 128 {
                (1u128 << (suffix_len + 8)) - 1
            } else {
                u128::MAX
            };

            let weight: Vec<Fr> = (0..trace_len)
                .map(|c| seed + Fr::from_u64((c + 1) as u64))
                .collect();
            let lookup_index: Vec<u128> = (0..trace_len)
                .map(|c| ((c as u128).wrapping_mul(0x9e37_79b1) ^ (c as u128) << 17) & index_mask)
                .collect();
            let is_interleaved: Vec<bool> = (0..trace_len).map(|c| c % 3 == 0).collect();

            let mut expected = vec![Fr::zero(); 5 * poly_len];
            let shift_half_off = 0;
            let left_off = poly_len;
            let right_off = 2 * poly_len;
            let shift_full_off = 3 * poly_len;
            let identity_off = 4 * poly_len;
            for c in 0..trace_len {
                let index = ((lookup_index[c] >> suffix_len) as usize) & (poly_len - 1);
                let suffix_bits = lookup_index[c] & suffix_mask;
                let w = weight[c];
                if is_interleaved[c] {
                    expected[shift_half_off + index] += w;
                    let (left_suffix, right_suffix) = uninterleave_bits(suffix_bits);
                    if left_suffix != 0 {
                        expected[left_off + index] += w * Fr::from_u64(left_suffix);
                    }
                    if right_suffix != 0 {
                        expected[right_off + index] += w * Fr::from_u64(right_suffix);
                    }
                } else {
                    expected[shift_full_off + index] += w;
                    if suffix_bits != 0 {
                        expected[identity_off + index] += w * Fr::from_u128(suffix_bits);
                    }
                }
            }

            let c = ctx();
            let weight_dev = c.upload(&weight).unwrap();
            let lo: Vec<u64> = lookup_index.iter().map(|&v| v as u64).collect();
            let hi: Vec<u64> = lookup_index.iter().map(|&v| (v >> 64) as u64).collect();
            let flags: Vec<u8> = is_interleaved.iter().map(|&b| u8::from(b)).collect();
            let lo_dev = c.upload_u64_slice(&lo).unwrap();
            let hi_dev = c.upload_u64_slice(&hi).unwrap();
            let flags_dev = c.upload_u8_slice(&flags).unwrap();
            let banks = c
                .raf_q_scatter(RafQScatterInputs {
                    weight: &weight_dev,
                    lookup_index_lo: &lo_dev,
                    lookup_index_hi: &hi_dev,
                    is_interleaved: &flags_dev,
                    trace_len,
                    suffix_len,
                    poly_len,
                })
                .unwrap();
            prop_assert_eq!(banks.len(), 5);
            let mut got = Vec::with_capacity(5 * poly_len);
            for bank in &banks {
                got.extend(bank.to_host().unwrap());
            }
            prop_assert_eq!(got, expected);
        }

        #[test]
        fn suffix_mle_probe_matches_cpu(
            raw in prop::collection::vec((any::<u128>(), 0usize..=128), 1..64),
        ) {
            use jolt_lookup_tables::LookupBits;
            use std::panic::{catch_unwind, AssertUnwindSafe};

            let prev_hook = std::panic::take_hook();
            std::panic::set_hook(Box::new(|_| {}));
            let variants = super::suffix_mle_variants();
            let mut bits_lo = Vec::new();
            let mut bits_hi = Vec::new();
            let mut lens = Vec::new();
            let mut codes = Vec::new();
            let mut expected = Vec::new();
            for (raw_bits, raw_len) in raw {
                for (code, suffix) in variants.iter().enumerate() {
                    let lb = LookupBits::new(raw_bits, raw_len);
                    let Ok(value) = catch_unwind(AssertUnwindSafe(|| suffix.suffix_mle(lb))) else {
                        continue;
                    };
                    let masked = u128::from(lb);
                    bits_lo.push(masked as u64);
                    bits_hi.push((masked >> 64) as u64);
                    lens.push(raw_len as u32);
                    codes.push(code as u32);
                    expected.push(value);
                }
            }
            std::panic::set_hook(prev_hook);

            let c = ctx();
            let got = c.suffix_mle_probe(&bits_lo, &bits_hi, &lens, &codes).unwrap();
            prop_assert_eq!(got, expected);
        }

        #[test]
        fn prefix_combine_probe_matches_cpu(seed in fr_strategy()) {
            use jolt_lookup_tables::tables::{LookupTableKind, PrefixEval};

            let tables = LookupTableKind::<64>::all();
            let mut prefixes_flat = Vec::new();
            let mut suffixes_flat = Vec::new();
            let mut counts = Vec::new();
            let mut variants = Vec::new();
            let mut expected = Vec::new();
            for (code, table) in tables.iter().enumerate() {
                let prefix_vals: Vec<Fr> =
                    (0..46).map(|p| seed + Fr::from_u64((code * 46 + p + 1) as u64)).collect();
                let count = table.suffixes().len();
                let suffix_vals: Vec<Fr> =
                    (0..count).map(|s| seed + Fr::from_u64((code * 4 + s + 500) as u64)).collect();

                let prefix_evals: Vec<PrefixEval<Fr>> =
                    prefix_vals.iter().map(|&v| PrefixEval::from(v)).collect();
                expected.push(table.combine(&prefix_evals, &suffix_vals));

                prefixes_flat.extend_from_slice(&prefix_vals);
                let mut padded = suffix_vals.clone();
                padded.resize(4, Fr::zero());
                suffixes_flat.extend_from_slice(&padded);
                counts.push(count as u32);
                variants.push(code as u32);
            }

            let c = ctx();
            let got = c
                .prefix_combine_probe(&prefixes_flat, &suffixes_flat, &counts, &variants)
                .unwrap();
            prop_assert_eq!(got, expected);
        }

        #[test]
        fn raf_weight_phase_update_matches_cpu(
            log_trace in 1usize..12,
            chunk_bits in 1usize..8,
            shift in 0usize..40,
            seed in fr_strategy(),
        ) {
            let trace_len = 1usize << log_trace;
            let poly_len = 1usize << chunk_bits;
            let mask = poly_len - 1;

            let weight: Vec<Fr> = (0..trace_len)
                .map(|c| seed + Fr::from_u64((c + 1) as u64))
                .collect();
            let eq_table: Vec<Fr> = (0..poly_len)
                .map(|i| seed + Fr::from_u64((i + 100) as u64))
                .collect();
            let lookup_index: Vec<u128> = (0..trace_len)
                .map(|c| (c as u128).wrapping_mul(0x9e37_79b1) ^ ((c as u128) << 20))
                .collect();

            let mut expected = weight.clone();
            for (c, w) in expected.iter_mut().enumerate() {
                let slot = ((lookup_index[c] >> shift) as usize) & mask;
                *w *= eq_table[slot];
            }

            let c = ctx();
            let mut weight_dev = c.upload(&weight).unwrap();
            let lo: Vec<u64> = lookup_index.iter().map(|&v| v as u64).collect();
            let hi: Vec<u64> = lookup_index.iter().map(|&v| (v >> 64) as u64).collect();
            let lo_dev = c.upload_u64_slice(&lo).unwrap();
            let hi_dev = c.upload_u64_slice(&hi).unwrap();
            c.raf_weight_phase_update(&mut weight_dev, &eq_table, &lo_dev, &hi_dev, shift, mask)
                .unwrap();
            let got = weight_dev.to_host().unwrap();
            prop_assert_eq!(got, expected);
        }

        #[test]
        fn lt_evals_matches_cpu(
            point in prop::collection::vec(fr_strategy(), 0..12),
        ) {
            let mut expected = vec![Fr::zero(); 1usize << point.len()];
            for (index, r) in point.iter().rev().enumerate() {
                let (left, right) = expected.split_at_mut(1usize << index);
                left.iter_mut().zip(right).for_each(|(left, right)| {
                    *right = *left * *r;
                    *left += *r - *right;
                });
            }
            let c = ctx();
            let got = c.lt_evals(&point).unwrap().to_host().unwrap();
            prop_assert_eq!(got, expected);
        }

        #[test]
        fn rd_wa_gather_matches_cpu(
            log_trace in 1usize..12,
            register_count in 1usize..40,
            seed in fr_strategy(),
        ) {
            let trace_len = 1usize << log_trace;
            let address_eq: Vec<Fr> = (0..register_count)
                .map(|i| seed + Fr::from_u64(i as u64 + 1))
                .collect();
            let addresses: Vec<i16> = (0..trace_len)
                .map(|c| {
                    if (c * 7 + 3) % 5 == 0 {
                        -1
                    } else {
                        ((c * 13) % register_count) as i16
                    }
                })
                .collect();
            let expected: Vec<Fr> = addresses
                .iter()
                .map(|&a| if a < 0 { Fr::zero() } else { address_eq[a as usize] })
                .collect();

            let c = ctx();
            let got = c.rd_wa_gather(&address_eq, &addresses).unwrap().to_host().unwrap();
            prop_assert_eq!(got, expected);
        }

        #[test]
        fn add_scalar_matches_cpu(
            log_n in 0usize..12,
            seed in fr_strategy(),
            scalar in fr_strategy(),
        ) {
            let n = 1usize << log_n;
            let base: Vec<Fr> = (0..n).map(|i| seed + Fr::from_u64(i as u64)).collect();
            let expected: Vec<Fr> = base.iter().map(|&x| x + scalar).collect();

            let c = ctx();
            let mut dev = c.upload(&base).unwrap();
            c.add_scalar(&mut dev, scalar).unwrap();
            prop_assert_eq!(dev.to_host().unwrap(), expected);
        }

        #[test]
        fn u64_to_mont_matches_cpu(mut values in prop::collection::vec(any::<u64>(), 0..300)) {
            values.push(0);
            values.push(u64::MAX);
            values.push(1);
            let expected: Vec<Fr> = values.iter().map(|&v| Fr::from_u64(v)).collect();

            let c = ctx();
            let got = c.u64_to_mont(&values).unwrap();
            prop_assert_eq!(got.to_host().unwrap(), expected);
        }

        #[test]
        fn eq_weighted_round_poly_matches_cpu(
            num_vars in 1usize..9,
            num_factors in 1usize..5,
            terms_spec in prop::collection::vec(
                (fr_strategy(), prop::collection::vec(0u32..5, 1..4)),
                1..4,
            ),
            point_seed in fr_strategy(),
            seed in fr_strategy(),
        ) {
            use jolt_poly::{BindingOrder, GruenSplitEqPolynomial};

            let terms_spec: Vec<(Fr, Vec<u32>)> = terms_spec
                .into_iter()
                .map(|(coeff, idxs)| {
                    (coeff, idxs.into_iter().map(|i| i % num_factors as u32).collect())
                })
                .collect();
            let degree = terms_spec.iter().map(|(_, idxs)| idxs.len()).max().unwrap();

            let point: Vec<Fr> = (0..num_vars)
                .map(|i| point_seed + Fr::from_u64(i as u64))
                .collect();
            let split_eq = GruenSplitEqPolynomial::<Fr>::new(&point, BindingOrder::LowToHigh);

            let num_groups = split_eq.e_in_current().len() * split_eq.e_out_current().len();
            let len = num_groups * 2;
            let factors: Vec<Vec<Fr>> = (0..num_factors)
                .map(|f| (0..len).map(|i| seed + Fr::from_u64((f * len + i) as u64)).collect())
                .collect();

            let cpu_terms: Vec<crate::stage6::DenseTerm<Fr>> = terms_spec
                .iter()
                .map(|(coeff, idxs)| crate::stage6::DenseTerm {
                    coefficient: *coeff,
                    factors: idxs.iter().map(|&i| i as usize).collect(),
                })
                .collect();
            let expected = split_eq.fold_out_in(
                || vec![Fr::zero(); degree],
                |inner: &mut Vec<Fr>, group, _x_in, e_in| {
                    let mut row = vec![Fr::zero(); degree];
                    crate::stage6::accumulate_dense_row_evaluations(
                        &factors, &cpu_terms, group, &mut row,
                    );
                    for (acc, e) in inner.iter_mut().zip(&row) {
                        *acc += e_in * *e;
                    }
                },
                |_x_out, e_out, inner: Vec<Fr>| {
                    inner.into_iter().map(|v| e_out * v).collect::<Vec<_>>()
                },
                |mut left: Vec<Fr>, right: Vec<Fr>| {
                    for (l, r) in left.iter_mut().zip(right) {
                        *l += r;
                    }
                    left
                },
            );

            let (term_coeffs, term_factor_offsets, term_factor_indices) =
                flatten_terms(&terms_spec);
            let c = ctx();
            let factor_devs: Vec<DeviceFrVec> =
                factors.iter().map(|f| c.upload(f).unwrap()).collect();
            let factor_refs: Vec<&DeviceFrVec> = factor_devs.iter().collect();
            let e_in = c.upload(split_eq.e_in_current()).unwrap();
            let e_out = c.upload(split_eq.e_out_current()).unwrap();
            let got = c
                .eq_weighted_round_poly(
                    RoundPolyTerms {
                        factors: &factor_refs,
                        term_coeffs: &c.upload(&term_coeffs).unwrap(),
                        term_factor_offsets: &term_factor_offsets,
                        term_factor_indices: &term_factor_indices,
                        degree,
                    },
                    &e_in,
                    &e_out,
                )
                .unwrap();
            prop_assert_eq!(got, expected);
        }

        #[test]
        fn batched_bind_matches_cpu(
            log_half in 0usize..10,
            num_buffers in 1usize..6,
            challenge in fr_strategy(),
            seed in fr_strategy(),
        ) {
            let half = 1usize << log_half;
            let buf_len = half * 2;
            let buffers: Vec<Vec<Fr>> = (0..num_buffers)
                .map(|b| {
                    (0..buf_len)
                        .map(|i| seed + Fr::from_u64((b * buf_len + i) as u64))
                        .collect()
                })
                .collect();

            let mut expected = Vec::with_capacity(num_buffers * half);
            for buffer in &buffers {
                let mut values = buffer.clone();
                let mut scratch = Vec::new();
                crate::dense::bind_dense_evals_reuse(&mut values, &mut scratch, challenge);
                expected.extend(values);
            }

            let packed: Vec<Fr> = buffers.into_iter().flatten().collect();
            let c = ctx();
            let mut values = c.upload(&packed).unwrap();
            let mut scratch = c.upload(&[]).unwrap();
            c.batched_bind(&mut values, &mut scratch, num_buffers, challenge).unwrap();
            prop_assert_eq!(values.to_host().unwrap(), expected);
        }

        #[test]
        fn uniskip_extended_evals_matches_cpu(
            log_cycles in 0usize..9,
            seed in fr_strategy(),
        ) {
            use crate::stage1::{
                Stage1OuterR1csData, OUTER_FIRST_GROUP_ROWS, OUTER_SECOND_GROUP_ROWS,
                OUTER_UNISKIP_DEGREE, OUTER_UNISKIP_TARGET_COEFFS,
            };
            use jolt_r1cs::R1csRowDotSlice;

            const ROW_COUNT: usize = 19;
            let num_cycles = 1usize << log_cycles;
            let row_dots_a: Vec<Fr> = (0..num_cycles * ROW_COUNT)
                .map(|i| seed + Fr::from_u64((i + 1) as u64))
                .collect();
            let row_dots_b: Vec<Fr> = (0..num_cycles * ROW_COUNT)
                .map(|i| seed + Fr::from_u64((i + 7) as u64))
                .collect();
            let eq_evals: Vec<Fr> = (0..num_cycles * 2)
                .map(|i| seed + Fr::from_u64((i + 13) as u64))
                .collect();

            let target_coeff_fields =
                OUTER_UNISKIP_TARGET_COEFFS.map(|coefficients| coefficients.map(Fr::from_i64));
            let mut expected = vec![Fr::zero(); OUTER_UNISKIP_DEGREE];
            for cycle in 0..num_cycles {
                let base = cycle * ROW_COUNT;
                let dots = R1csRowDotSlice {
                    a: &row_dots_a[base..base + ROW_COUNT],
                    b: &row_dots_b[base..base + ROW_COUNT],
                };
                let first = Stage1OuterR1csData::<Fr>::group_matvecs_all_uniskip_targets(
                    &OUTER_FIRST_GROUP_ROWS,
                    &target_coeff_fields,
                    dots,
                );
                let second = Stage1OuterR1csData::<Fr>::group_matvecs_all_uniskip_targets(
                    &OUTER_SECOND_GROUP_ROWS,
                    &target_coeff_fields,
                    dots,
                );
                for target in 0..OUTER_UNISKIP_DEGREE {
                    let (az_g0, bz_g0) = first[target];
                    let (az_g1, bz_g1) = second[target];
                    expected[target] += eq_evals[cycle * 2] * (az_g0 * bz_g0);
                    expected[target] += eq_evals[cycle * 2 + 1] * (az_g1 * bz_g1);
                }
            }

            let first_rows: Vec<u32> =
                OUTER_FIRST_GROUP_ROWS.iter().map(|&r| r as u32).collect();
            let second_rows: Vec<u32> =
                OUTER_SECOND_GROUP_ROWS.iter().map(|&r| r as u32).collect();
            let mut first_coeffs = Vec::new();
            let mut second_coeffs = Vec::new();
            for coeffs in &OUTER_UNISKIP_TARGET_COEFFS {
                for &coeff in &coeffs[..OUTER_FIRST_GROUP_ROWS.len()] {
                    first_coeffs.push(Fr::from_i64(coeff));
                }
                for &coeff in &coeffs[..OUTER_SECOND_GROUP_ROWS.len()] {
                    second_coeffs.push(Fr::from_i64(coeff));
                }
            }

            let c = ctx();
            let row_dots_a_dev = c.upload(&row_dots_a).unwrap();
            let row_dots_b_dev = c.upload(&row_dots_b).unwrap();
            let eq_evals_dev = c.upload(&eq_evals).unwrap();
            let got = c
                .uniskip_extended_evals(UniskipInputs {
                    row_dots_a: &row_dots_a_dev,
                    row_dots_b: &row_dots_b_dev,
                    eq_evals: &eq_evals_dev,
                    first_group_rows: &first_rows,
                    second_group_rows: &second_rows,
                    first_coeffs: &first_coeffs,
                    second_coeffs: &second_coeffs,
                    row_count: ROW_COUNT,
                    degree: OUTER_UNISKIP_DEGREE,
                })
                .unwrap();
            prop_assert_eq!(got, expected);
        }

        #[test]
        fn gather8_materialize_matches_cpu(
            num_chunks in 1usize..5,
            log_new_len in 0usize..8,
            table_len in 1usize..=8,
            seed in fr_strategy(),
        ) {
            let new_len = 1usize << log_new_len;
            let groups: Vec<Vec<Vec<Fr>>> = (0..8)
                .map(|g| {
                    (0..num_chunks)
                        .map(|chunk| {
                            (0..table_len)
                                .map(|e| seed + Fr::from_u64((g * 131 + chunk * 17 + e + 1) as u64))
                                .collect()
                        })
                        .collect()
                })
                .collect();

            let indices: Vec<Vec<Option<u8>>> = (0..num_chunks)
                .map(|chunk| {
                    (0..new_len * 8)
                        .map(|i| {
                            let h = (chunk * 7 + i * 13) % (table_len + 1);
                            if h == table_len {
                                None
                            } else {
                                Some(h as u8)
                            }
                        })
                        .collect()
                })
                .collect();

            let group_refs: [&Vec<Vec<Fr>>; 8] = std::array::from_fn(|g| &groups[g]);
            let expected = crate::stage6::materialize_gather8(&group_refs, &indices);

            let flat_groups: Vec<Vec<Fr>> = groups
                .iter()
                .map(|group| group.iter().flatten().copied().collect())
                .collect();
            let table_refs: [&[Fr]; 8] = std::array::from_fn(|g| flat_groups[g].as_slice());
            let flat_indices: Vec<i16> = indices
                .iter()
                .flat_map(|chunk| chunk.iter().map(|i| i.map_or(-1, i16::from)))
                .collect();

            let c = ctx();
            let got = c
                .gather8_materialize(Gather8Inputs {
                    table_groups: table_refs,
                    indices: &flat_indices,
                    num_chunks,
                    table_len,
                    new_len,
                })
                .unwrap();
            prop_assert_eq!(got, expected);
        }

        #[test]
        fn core_booleanity_gather_matches_cpu(
            num_polys in 1usize..12,
            log_rows in 0usize..8,
            log_chunk_domain in 1usize..6,
            seed in fr_strategy(),
        ) {
            const POLY_STRIDE: usize = 64;
            let rows = 1usize << log_rows;
            let chunk_domain = 1usize << log_chunk_domain;

            let tables: Vec<Vec<Fr>> = (0..num_polys)
                .map(|p| {
                    (0..chunk_domain)
                        .map(|e| seed + Fr::from_u64((p * chunk_domain + e + 1) as u64))
                        .collect()
                })
                .collect();

            let mut present_mask = vec![0u64; rows];
            let mut values = vec![0u8; rows * POLY_STRIDE];
            for j in 0..rows {
                for p in 0..num_polys {
                    if (j * 7 + p * 13) % 4 != 0 {
                        present_mask[j] |= 1u64 << p;
                        values[j * POLY_STRIDE + p] = ((j * 5 + p * 3) % chunk_domain) as u8;
                    }
                }
            }

            let expected: Vec<Vec<Fr>> = (0..num_polys)
                .map(|p| {
                    (0..rows)
                        .map(|j| {
                            if present_mask[j] & (1u64 << p) != 0 {
                                tables[p][usize::from(values[j * POLY_STRIDE + p])]
                            } else {
                                Fr::from_u64(0)
                            }
                        })
                        .collect()
                })
                .collect();

            let flat_tables: Vec<Fr> = tables.iter().flatten().copied().collect();

            let c = ctx();
            let got = c
                .core_booleanity_gather(CoreBooleanityGatherInputs {
                    tables: &flat_tables,
                    present_mask: &present_mask,
                    values: &values,
                    num_polys,
                    chunk_domain,
                    rows,
                    poly_stride: POLY_STRIDE,
                })
                .unwrap();
            let got_host: Vec<Vec<Fr>> =
                got.iter().map(|d| d.to_host().unwrap()).collect();
            prop_assert_eq!(got_host, expected);
        }

        #[test]
        fn core_booleanity_sparse_round_poly_matches_cpu(
            round in 1u32..=3,
            extra_vars in 1usize..6,
            num_polys in 1usize..12,
            log_chunk_domain in 1usize..6,
            seed in fr_strategy(),
        ) {
            use jolt_field::FieldAccumulator;
            use jolt_poly::{BindingOrder, GruenSplitEqPolynomial};

            const POLY_STRIDE: usize = 64;
            let num_sets = 1usize << (round - 1);
            let out_vars = extra_vars;
            let half = 1usize << out_vars;
            let source_rows = half << round;
            let chunk_domain = 1usize << log_chunk_domain;

            // num_sets table-sets, packed set-major: [set][poly][entry].
            let table_sets: Vec<Vec<Vec<Fr>>> = (0..num_sets)
                .map(|s| {
                    (0..num_polys)
                        .map(|p| {
                            (0..chunk_domain)
                                .map(|e| {
                                    seed + Fr::from_u64(
                                        (s * 977 + p * chunk_domain + e + 1) as u64,
                                    )
                                })
                                .collect()
                        })
                        .collect()
                })
                .collect();

            let mut present_mask = vec![0u64; source_rows];
            let mut values = vec![0u8; source_rows * POLY_STRIDE];
            for s in 0..source_rows {
                for p in 0..num_polys {
                    if (s * 7 + p * 13) % 4 != 0 {
                        present_mask[s] |= 1u64 << p;
                        values[s * POLY_STRIDE + p] = ((s * 5 + p * 3) % chunk_domain) as u8;
                    }
                }
            }

            let gather = |set: usize, s: usize, p: usize| -> Fr {
                if present_mask[s] & (1u64 << p) != 0 {
                    table_sets[set][p][usize::from(values[s * POLY_STRIDE + p])]
                } else {
                    Fr::from_u64(0)
                }
            };

            let rho: Vec<Fr> =
                (0..num_polys).map(|i| seed + Fr::from_u64((i + 101) as u64)).collect();
            let point: Vec<Fr> = (0..=out_vars)
                .map(|i| seed + Fr::from_u64((i + 1) as u64))
                .collect();
            let split_eq = GruenSplitEqPolynomial::<Fr>::new(&point, BindingOrder::LowToHigh);
            let e_in = split_eq.e_in_current();
            let e_out = split_eq.e_out_current();
            let in_bits = e_in.len().trailing_zeros() as usize;

            type Acc = <Fr as Field>::Accumulator;
            let mut outer = [Acc::default(); 2];
            for (x_out, &eo) in e_out.iter().enumerate() {
                let mut inner = [Acc::default(); 2];
                let base_x = x_out << in_bits;
                for (x_in, &ei) in e_in.iter().enumerate() {
                    let j = base_x | x_in;
                    let base = j << round;
                    let mut c = Acc::default();
                    let mut q = Acc::default();
                    for (p, &rho_p) in rho.iter().enumerate() {
                        let mut h0 = Fr::from_u64(0);
                        let mut h1 = Fr::from_u64(0);
                        for set in 0..num_sets {
                            h0 += gather(set, base + set, p);
                            h1 += gather(set, base + num_sets + set, p);
                        }
                        let delta = h1 - h0;
                        c.fmadd(h0, h0 - rho_p);
                        q.fmadd(delta, delta);
                    }
                    inner[0].fmadd(ei, c.reduce());
                    inner[1].fmadd(ei, q.reduce());
                }
                outer[0].fmadd(eo, inner[0].reduce());
                outer[1].fmadd(eo, inner[1].reduce());
            }
            let expected = [outer[0].reduce(), outer[1].reduce()];

            let c = ctx();
            let flat_tables: Vec<u64> = table_sets
                .iter()
                .flat_map(|set| set.iter().flatten().flat_map(|v| v.inner_limbs().0))
                .collect();
            let tables_dev = c.upload_u64_slice(&flat_tables).unwrap();
            let mask_dev = c.upload_u64_slice(&present_mask).unwrap();
            let values_dev = c.upload_u8_slice(&values).unwrap();
            let e_in_dev = c.upload(e_in).unwrap();
            let e_out_dev = c.upload(e_out).unwrap();
            let rho_dev = c.upload(&rho).unwrap();
            let got = c
                .core_booleanity_sparse_round_poly(CoreBooleanitySparseInputs {
                    tables: &tables_dev,
                    present_mask: &mask_dev,
                    values: &values_dev,
                    source_rows,
                    rho: &rho_dev,
                    e_in: &e_in_dev,
                    e_out: &e_out_dev,
                    num_polys,
                    chunk_domain,
                    poly_stride: POLY_STRIDE,
                    round,
                })
                .unwrap();
            prop_assert_eq!(got.to_vec(), expected.to_vec());
        }

        #[test]
        fn core_booleanity_sparse_bind_matches_cpu(
            log_num_sets in 0usize..3,
            set_elems in 1usize..40,
            challenge in fr_strategy(),
            seed in fr_strategy(),
        ) {
            let num_sets = 1usize << log_num_sets;
            let input: Vec<Fr> = (0..num_sets * set_elems)
                .map(|i| seed + Fr::from_u64((i + 1) as u64))
                .collect();

            let one_minus = Fr::from_u64(1) - challenge;
            let mut expected = Vec::with_capacity(2 * num_sets * set_elems);
            expected.extend(input.iter().map(|&v| one_minus * v));
            expected.extend(input.iter().map(|&v| challenge * v));

            let c = ctx();
            let in_limbs: Vec<u64> =
                input.iter().flat_map(|v| v.inner_limbs().0).collect();
            let in_dev = c.upload_u64_slice(&in_limbs).unwrap();
            let out = c
                .core_booleanity_sparse_bind(&in_dev, num_sets, set_elems, challenge)
                .unwrap();
            let raw = c.download_u64(&out).unwrap();
            let got: Vec<Fr> = raw
                .chunks_exact(4)
                .map(|c| Fr::from_bigint_unchecked(jolt_field::Limbs([c[0], c[1], c[2], c[3]])))
                .collect();
            prop_assert_eq!(got, expected);
        }

        #[test]
        fn core_booleanity_sparse_collapse8_matches_cpu(
            out_vars in 0usize..7,
            num_polys in 1usize..12,
            log_chunk_domain in 1usize..6,
            seed in fr_strategy(),
        ) {
            const POLY_STRIDE: usize = 64;
            let out_len = 1usize << out_vars;
            let source_rows = out_len * 8;
            let chunk_domain = 1usize << log_chunk_domain;

            let table_sets: Vec<Vec<Vec<Fr>>> = (0..8)
                .map(|s| {
                    (0..num_polys)
                        .map(|p| {
                            (0..chunk_domain)
                                .map(|e| {
                                    seed + Fr::from_u64((s * 977 + p * chunk_domain + e + 1) as u64)
                                })
                                .collect()
                        })
                        .collect()
                })
                .collect();

            let mut present_mask = vec![0u64; source_rows];
            let mut values = vec![0u8; source_rows * POLY_STRIDE];
            for s in 0..source_rows {
                for p in 0..num_polys {
                    if (s * 7 + p * 13) % 4 != 0 {
                        present_mask[s] |= 1u64 << p;
                        values[s * POLY_STRIDE + p] = ((s * 5 + p * 3) % chunk_domain) as u8;
                    }
                }
            }

            let expected: Vec<Vec<Fr>> = (0..num_polys)
                .map(|p| {
                    (0..out_len)
                        .map(|j| {
                            let mut acc = Fr::from_u64(0);
                            for (set, table_set) in table_sets.iter().enumerate() {
                                let s = 8 * j + set;
                                if present_mask[s] & (1u64 << p) != 0 {
                                    acc += table_set[p][usize::from(values[s * POLY_STRIDE + p])];
                                }
                            }
                            acc
                        })
                        .collect()
                })
                .collect();

            let c = ctx();
            let flat_tables: Vec<u64> = table_sets
                .iter()
                .flat_map(|set| set.iter().flatten().flat_map(|v| v.inner_limbs().0))
                .collect();
            let tables_dev = c.upload_u64_slice(&flat_tables).unwrap();
            let mask_dev = c.upload_u64_slice(&present_mask).unwrap();
            let values_dev = c.upload_u8_slice(&values).unwrap();
            let got = c
                .core_booleanity_sparse_collapse8(
                    &tables_dev,
                    &mask_dev,
                    &values_dev,
                    num_polys,
                    chunk_domain,
                    POLY_STRIDE,
                    out_len,
                )
                .unwrap();
            let got_host: Vec<Vec<Fr>> = got.iter().map(|d| d.to_host().unwrap()).collect();
            prop_assert_eq!(got_host, expected);
        }

        #[test]
        fn hamming_round_poly_matches_cpu(
            log_len in 1usize..10,
            num_ra in 1usize..5,
            previous_claim in fr_strategy(),
            scale in fr_strategy(),
            seed in fr_strategy(),
        ) {
            use crate::stage7::{HammingWeightClaimReductionState, Stage7Relation};
            use jolt_poly::UnivariatePoly;

            let len = 1usize << log_len;
            let g: Vec<Vec<Fr>> = (0..num_ra)
                .map(|i| (0..len).map(|j| seed + Fr::from_u64((i * len + j + 1) as u64)).collect())
                .collect();
            let eq_virt: Vec<Vec<Fr>> = (0..num_ra)
                .map(|i| (0..len).map(|j| seed + Fr::from_u64((i * len + j + 101) as u64)).collect())
                .collect();
            let eq_bool: Vec<Fr> =
                (0..len).map(|j| seed + Fr::from_u64((j + 1001) as u64)).collect();
            let gamma_powers: Vec<Fr> =
                (0..3 * num_ra).map(|i| seed + Fr::from_u64((i + 7) as u64)).collect();

            let mut state = HammingWeightClaimReductionState {
                g: g.clone(),
                eq_bool: eq_bool.clone(),
                eq_virt: eq_virt.clone(),
                gamma_powers: gamma_powers.clone(),
                outputs: Vec::new(),
                active_scale: scale,
                backend: "cpu",
                cuda: None,
            };
            let expected = state
                .round_poly(previous_claim, Stage7Relation::HammingWeightClaimReduction)
                .unwrap();

            let c = ctx();
            let g_devs: Vec<DeviceFrVec> = g.iter().map(|v| c.upload(v).unwrap()).collect();
            let eq_virt_devs: Vec<DeviceFrVec> =
                eq_virt.iter().map(|v| c.upload(v).unwrap()).collect();
            let eq_bool_dev = c.upload(&eq_bool).unwrap();
            let gamma_dev = c.upload(&gamma_powers).unwrap();
            let g_refs: Vec<&DeviceFrVec> = g_devs.iter().collect();
            let eq_virt_refs: Vec<&DeviceFrVec> = eq_virt_devs.iter().collect();

            let evals = c
                .hamming_round_poly(HammingRoundPolyInputs {
                    g: &g_refs,
                    eq_virt: &eq_virt_refs,
                    eq_bool: &eq_bool_dev,
                    gamma_powers: &gamma_dev,
                    scale,
                })
                .unwrap();
            let got = UnivariatePoly::from_evals_and_hint(previous_claim, &evals);
            prop_assert_eq!(got.coefficients(), expected.coefficients());
        }

        #[test]
        fn hamming_booleanity_round_poly_matches_cpu(
            num_vars in 1usize..10,
            previous_claim in fr_strategy(),
            point_seed in fr_strategy(),
            seed in fr_strategy(),
        ) {
            use crate::stage6::{FactorOutput, HammingBooleanityStage6State};
            use crate::stage6::Stage6Relation;

            // active_scale is 1 for hamming booleanity (it spans all rounds); the Gruen
            // poly is built so poly(0)+poly(1) == previous_claim, so any claim is valid.
            let scale = Fr::from_u64(1);
            let len = 1usize << num_vars;
            let point: Vec<Fr> = (0..num_vars)
                .map(|i| point_seed + Fr::from_u64(i as u64))
                .collect();
            let hamming_weight: Vec<Fr> =
                (0..len).map(|j| seed + Fr::from_u64((j + 1) as u64)).collect();

            let output = FactorOutput {
                name: "test",
                oracle: "test",
                factor: 0,
            };
            let state = HammingBooleanityStage6State::new_with_backend(
                &point,
                hamming_weight.clone(),
                output,
                scale,
                3,
                "cpu",
            )
            .unwrap();
            let expected = state
                .round_poly(previous_claim, Stage6Relation::HammingBooleanity)
                .unwrap();

            let c = ctx();
            let hamming_dev = c.upload(&hamming_weight).unwrap();
            let e_in = c.upload(state.eq.e_in_current()).unwrap();
            let e_out = c.upload(state.eq.e_out_current()).unwrap();
            let q = c
                .hamming_booleanity_round_poly(HammingBooleanityInputs {
                    hamming_weight: &hamming_dev,
                    e_in: &e_in,
                    e_out: &e_out,
                })
                .unwrap();
            let mut got = state.eq.gruen_poly_deg_3(q[0], q[1], previous_claim);
            got *= scale;
            prop_assert_eq!(got.coefficients(), expected.coefficients());
        }

        #[test]
        fn core_booleanity_cycle_round_poly_matches_cpu(
            num_vars in 1usize..9,
            num_polys in 1usize..6,
            seed in fr_strategy(),
        ) {
            use jolt_field::FieldAccumulator;
            use jolt_poly::{BindingOrder, GruenSplitEqPolynomial};

            let len = 1usize << num_vars;
            let point: Vec<Fr> = (0..num_vars)
                .map(|i| seed + Fr::from_u64((i + 1) as u64))
                .collect();
            let h_polys: Vec<Vec<Fr>> = (0..num_polys)
                .map(|p| (0..len).map(|j| seed + Fr::from_u64((p * len + j + 7) as u64)).collect())
                .collect();
            let rho: Vec<Fr> =
                (0..num_polys).map(|i| seed + Fr::from_u64((i + 101) as u64)).collect();

            let split_eq = GruenSplitEqPolynomial::<Fr>::new(&point, BindingOrder::LowToHigh);
            let e_in = split_eq.e_in_current();
            let e_out = split_eq.e_out_current();
            let in_bits = e_in.len().trailing_zeros() as usize;

            type Acc = <Fr as Field>::Accumulator;
            let mut outer = [Acc::default(); 2];
            for (x_out, &eo) in e_out.iter().enumerate() {
                let mut inner = [Acc::default(); 2];
                let base = x_out << in_bits;
                for (x_in, &ei) in e_in.iter().enumerate() {
                    let j = base | x_in;
                    let mut c = Acc::default();
                    let mut q = Acc::default();
                    for (i, h) in h_polys.iter().enumerate() {
                        let h0 = h[2 * j];
                        let h1 = h[2 * j + 1];
                        let delta = h1 - h0;
                        c.fmadd(h0, h0 - rho[i]);
                        q.fmadd(delta, delta);
                    }
                    inner[0].fmadd(ei, c.reduce());
                    inner[1].fmadd(ei, q.reduce());
                }
                outer[0].fmadd(eo, inner[0].reduce());
                outer[1].fmadd(eo, inner[1].reduce());
            }
            let expected = [outer[0].reduce(), outer[1].reduce()];

            let c = ctx();
            let h_devs: Vec<DeviceFrVec> = h_polys.iter().map(|v| c.upload(v).unwrap()).collect();
            let h_refs: Vec<&DeviceFrVec> = h_devs.iter().collect();
            let e_in_dev = c.upload(e_in).unwrap();
            let e_out_dev = c.upload(e_out).unwrap();
            let rho_dev = c.upload(&rho).unwrap();
            let got = c
                .core_booleanity_cycle_round_poly(CoreBooleanityCycleInputs {
                    h_polys: &h_refs,
                    rho: &rho_dev,
                    e_in: &e_in_dev,
                    e_out: &e_out_dev,
                })
                .unwrap();
            prop_assert_eq!(got.to_vec(), expected.to_vec());
        }

        #[test]
        fn core_booleanity_address_round_poly_matches_cpu(
            log_k_chunk in 2usize..8,
            m_minus_1 in 0usize..6,
            num_polys in 1usize..5,
            seed in fr_strategy(),
        ) {
            use jolt_field::FieldAccumulator;
            use jolt_poly::{BindingOrder, GruenSplitEqPolynomial};

            let m = m_minus_1 + 1;
            prop_assume!(m <= log_k_chunk);
            let chunk_domain = 1usize << log_k_chunk;
            let f_len = 1usize << (m - 1);

            let r_address: Vec<Fr> = (0..log_k_chunk)
                .map(|i| seed + Fr::from_u64((i + 1) as u64))
                .collect();
            let mut b = GruenSplitEqPolynomial::<Fr>::new(&r_address, BindingOrder::LowToHigh);
            for i in 0..(m - 1) {
                b.bind(seed + Fr::from_u64((i + 51) as u64));
            }
            let g: Vec<Vec<Fr>> = (0..num_polys)
                .map(|p| {
                    (0..chunk_domain)
                        .map(|k| seed + Fr::from_u64((p * chunk_domain + k + 7) as u64))
                        .collect()
                })
                .collect();
            let f_values: Vec<Fr> =
                (0..f_len).map(|k| seed + Fr::from_u64((k + 3) as u64)).collect();
            let gamma_squares: Vec<Fr> =
                (0..num_polys).map(|i| seed + Fr::from_u64((i + 101) as u64)).collect();

            type Acc = <Fr as Field>::Accumulator;
            let expected = b.fold_out_in(
                || [Acc::default(); 2],
                |inner, k_prime, _x_in, e_in| {
                    for (g_i, &gamma_square) in g.iter().zip(&gamma_squares) {
                        let mut eval_0 = Fr::zero();
                        let mut eval_infty = Fr::zero();
                        let block_start = k_prime << m;
                        for (k, &g_k) in g_i[block_start..block_start + (1 << m)].iter().enumerate() {
                            let k_m = k >> (m - 1);
                            let f_k = f_values[k & ((1 << (m - 1)) - 1)];
                            let g_times_f = g_k * f_k;
                            let eval_inf = g_times_f * f_k;
                            if k_m == 0 {
                                eval_0 += eval_inf - g_times_f;
                            }
                            eval_infty += eval_inf;
                        }
                        let weight = e_in * gamma_square;
                        inner[0].fmadd(weight, eval_0);
                        inner[1].fmadd(weight, eval_infty);
                    }
                },
                |_x_out, e_out, inner| {
                    let mut outer = [Acc::default(); 2];
                    outer[0].fmadd(e_out, inner[0].reduce());
                    outer[1].fmadd(e_out, inner[1].reduce());
                    outer
                },
                |mut left, right| {
                    left[0].merge(right[0]);
                    left[1].merge(right[1]);
                    left
                },
            );
            let expected = [expected[0].reduce(), expected[1].reduce()];

            let c = ctx();
            let g_devs: Vec<DeviceFrVec> = g.iter().map(|v| c.upload(v).unwrap()).collect();
            let g_refs: Vec<&DeviceFrVec> = g_devs.iter().collect();
            let e_in_dev = c.upload(b.e_in_current()).unwrap();
            let e_out_dev = c.upload(b.e_out_current()).unwrap();
            let gamma_dev = c.upload(&gamma_squares).unwrap();
            let got = c
                .core_booleanity_address_round_poly(CoreBooleanityAddressInputs {
                    g: &g_refs,
                    f_values: &f_values,
                    gamma_squares: &gamma_dev,
                    e_in: &e_in_dev,
                    e_out: &e_out_dev,
                    m: m as u32,
                })
                .unwrap();
            prop_assert_eq!(got.to_vec(), expected.to_vec());
        }

        #[test]
        fn sparse_register_round_poly_matches_cpu(
            log_pairs in 1usize..6,
            high_round in proptest::bool::ANY,
            s1 in 1usize..7,
            s2 in 1usize..7,
            seed in fr_strategy(),
        ) {
            use jolt_field::FieldAccumulator;

            const COLS: usize = 4;
            let num_pairs = 1usize << log_pairs;

            let (e_in, e_out, in_pairs): (Vec<Fr>, Vec<Fr>, u32) = if high_round {
                let e_in = vec![seed + Fr::from_u64(3)];
                let e_out: Vec<Fr> = (0..2 * num_pairs)
                    .map(|i| seed + Fr::from_u64((i + 11) as u64))
                    .collect();
                (e_in, e_out, 0)
            } else {
                let in_pairs = 1usize << (log_pairs / 2);
                let out_len = num_pairs / in_pairs;
                let e_in: Vec<Fr> = (0..2 * in_pairs)
                    .map(|i| seed + Fr::from_u64((i + 5) as u64))
                    .collect();
                let e_out: Vec<Fr> = (0..out_len)
                    .map(|i| seed + Fr::from_u64((i + 23) as u64))
                    .collect();
                (e_in, e_out, in_pairs as u32)
            };

            let mut val = Vec::new();
            let mut read_ra = Vec::new();
            let mut rd_wa = Vec::new();
            let mut prev_val = Vec::new();
            let mut next_val = Vec::new();
            let mut even_idx: Vec<i32> = Vec::new();
            let mut odd_idx: Vec<i32> = Vec::new();
            let mut pair_arr: Vec<u32> = Vec::new();

            for p in 0..num_pairs {
                let mut even_map = [-1i32; COLS];
                let mut odd_map = [-1i32; COLS];
                for c in 0..COLS {
                    if (p * 7 + c * 13 + s1) % 3 != 0 {
                        let g = val.len();
                        val.push(seed + Fr::from_u64((g + 1) as u64));
                        read_ra.push(seed + Fr::from_u64((g + 100) as u64));
                        rd_wa.push(seed + Fr::from_u64((g + 200) as u64));
                        prev_val.push(Fr::from_u64((g + 7) as u64));
                        next_val.push(Fr::from_u64((g + 9) as u64));
                        even_map[c] = g as i32;
                    }
                    if (p * 11 + c * 5 + s2) % 3 != 0 {
                        let g = val.len();
                        val.push(seed + Fr::from_u64((g + 1) as u64));
                        read_ra.push(seed + Fr::from_u64((g + 100) as u64));
                        rd_wa.push(seed + Fr::from_u64((g + 200) as u64));
                        prev_val.push(Fr::from_u64((g + 7) as u64));
                        next_val.push(Fr::from_u64((g + 9) as u64));
                        odd_map[c] = g as i32;
                    }
                }
                for c in 0..COLS {
                    if even_map[c] >= 0 || odd_map[c] >= 0 {
                        even_idx.push(even_map[c]);
                        odd_idx.push(odd_map[c]);
                        pair_arr.push(p as u32);
                    }
                }
            }

            let rd_inc: Vec<Fr> = (0..2 * num_pairs)
                .map(|i| seed + Fr::from_u64((i + 301) as u64))
                .collect();

            type Acc = <Fr as Field>::Accumulator;
            let mut acc = [Acc::default(); 2];
            for item in 0..even_idx.len() {
                let p = pair_arr[item] as usize;
                let ei = even_idx[item];
                let oi = odd_idx[item];
                let (v0, vd, r0, rd, w0, wd) = match (ei >= 0, oi >= 0) {
                    (true, true) => {
                        let e = ei as usize;
                        let o = oi as usize;
                        (
                            val[e],
                            val[o] - val[e],
                            read_ra[e],
                            read_ra[o] - read_ra[e],
                            rd_wa[e],
                            rd_wa[o] - rd_wa[e],
                        )
                    }
                    (true, false) => {
                        let e = ei as usize;
                        (
                            val[e],
                            next_val[e] - val[e],
                            read_ra[e],
                            -read_ra[e],
                            rd_wa[e],
                            -rd_wa[e],
                        )
                    }
                    (false, true) => {
                        let o = oi as usize;
                        (
                            prev_val[o],
                            val[o] - prev_val[o],
                            Fr::from_u64(0),
                            read_ra[o],
                            Fr::from_u64(0),
                            rd_wa[o],
                        )
                    }
                    (false, false) => (
                        Fr::from_u64(0),
                        Fr::from_u64(0),
                        Fr::from_u64(0),
                        Fr::from_u64(0),
                        Fr::from_u64(0),
                        Fr::from_u64(0),
                    ),
                };
                let inc0 = rd_inc[2 * p];
                let inc_delta = rd_inc[2 * p + 1] - rd_inc[2 * p];
                let body0 = w0 * (v0 + inc0) + r0 * v0;
                let body2 = wd * (vd + inc_delta) + rd * vd;
                let weight = if in_pairs == 0 {
                    e_in[0] * (e_out[2 * p] + e_out[2 * p + 1])
                } else {
                    let ip = in_pairs as usize;
                    let x_out = p / ip;
                    let x_in = p % ip;
                    e_out[x_out] * (e_in[2 * x_in] + e_in[2 * x_in + 1])
                };
                acc[0].fmadd(weight, body0);
                acc[1].fmadd(weight, body2);
            }
            let expected = [acc[0].reduce(), acc[1].reduce()];

            let c = ctx();
            let val_dev = c.upload(&val).unwrap();
            let read_ra_dev = c.upload(&read_ra).unwrap();
            let rd_wa_dev = c.upload(&rd_wa).unwrap();
            let prev_dev = c.upload(&prev_val).unwrap();
            let next_dev = c.upload(&next_val).unwrap();
            let rd_inc_dev = c.upload(&rd_inc).unwrap();
            let e_in_dev = c.upload(&e_in).unwrap();
            let e_out_dev = c.upload(&e_out).unwrap();
            let got = c
                .sparse_register_round_poly(SparseRegisterRoundInputs {
                    val: &val_dev,
                    read_ra: &read_ra_dev,
                    rd_wa: &rd_wa_dev,
                    prev_val: &prev_dev,
                    next_val: &next_dev,
                    even_idx: &even_idx,
                    odd_idx: &odd_idx,
                    pair: &pair_arr,
                    rd_inc: &rd_inc_dev,
                    e_in: &e_in_dev,
                    e_out: &e_out_dev,
                    in_pairs,
                })
                .unwrap();
            prop_assert_eq!(got.to_vec(), expected.to_vec());
        }

        #[test]
        fn sparse_register_bind_matches_cpu(
            num_entries in 1usize..40,
            s1 in 1usize..7,
            challenge in fr_strategy(),
            seed in fr_strategy(),
        ) {
            let mut val = Vec::new();
            let mut read_ra = Vec::new();
            let mut rd_wa = Vec::new();
            let mut prev_val = Vec::new();
            let mut next_val = Vec::new();
            let mut even_idx: Vec<i32> = Vec::new();
            let mut odd_idx: Vec<i32> = Vec::new();

            for item in 0..num_entries {
                let mode = (item * 7 + s1) % 3;
                let mut ei = -1i32;
                let mut oi = -1i32;
                if mode != 1 {
                    let g = val.len();
                    val.push(seed + Fr::from_u64((g + 1) as u64));
                    read_ra.push(seed + Fr::from_u64((g + 100) as u64));
                    rd_wa.push(seed + Fr::from_u64((g + 200) as u64));
                    prev_val.push(Fr::from_u64((g + 7) as u64));
                    next_val.push(Fr::from_u64((g + 9) as u64));
                    ei = g as i32;
                }
                if mode != 0 {
                    let g = val.len();
                    val.push(seed + Fr::from_u64((g + 1) as u64));
                    read_ra.push(seed + Fr::from_u64((g + 100) as u64));
                    rd_wa.push(seed + Fr::from_u64((g + 200) as u64));
                    prev_val.push(Fr::from_u64((g + 7) as u64));
                    next_val.push(Fr::from_u64((g + 9) as u64));
                    oi = g as i32;
                }
                even_idx.push(ei);
                odd_idx.push(oi);
            }

            let lin = |low: Fr, high: Fr| low + challenge * (high - low);
            let mut e_val = Vec::new();
            let mut e_read = Vec::new();
            let mut e_wa = Vec::new();
            let mut e_prev = Vec::new();
            let mut e_next = Vec::new();
            for item in 0..num_entries {
                let ei = even_idx[item];
                let oi = odd_idx[item];
                let (v, r, w, pv, nv) = match (ei >= 0, oi >= 0) {
                    (true, true) => {
                        let e = ei as usize;
                        let o = oi as usize;
                        (
                            lin(val[e], val[o]),
                            lin(read_ra[e], read_ra[o]),
                            lin(rd_wa[e], rd_wa[o]),
                            prev_val[e],
                            next_val[o],
                        )
                    }
                    (true, false) => {
                        let e = ei as usize;
                        (
                            lin(val[e], next_val[e]),
                            lin(read_ra[e], Fr::from_u64(0)),
                            lin(rd_wa[e], Fr::from_u64(0)),
                            prev_val[e],
                            next_val[e],
                        )
                    }
                    _ => {
                        let o = oi as usize;
                        (
                            lin(prev_val[o], val[o]),
                            lin(Fr::from_u64(0), read_ra[o]),
                            lin(Fr::from_u64(0), rd_wa[o]),
                            prev_val[o],
                            next_val[o],
                        )
                    }
                };
                e_val.push(v);
                e_read.push(r);
                e_wa.push(w);
                e_prev.push(pv);
                e_next.push(nv);
            }

            let c = ctx();
            let entries = c
                .sparse_register_bind(SparseRegisterBindInputs {
                    val: &c.upload(&val).unwrap(),
                    read_ra: &c.upload(&read_ra).unwrap(),
                    rd_wa: &c.upload(&rd_wa).unwrap(),
                    prev_val: &c.upload(&prev_val).unwrap(),
                    next_val: &c.upload(&next_val).unwrap(),
                    even_idx: &even_idx,
                    odd_idx: &odd_idx,
                    challenge,
                })
                .unwrap();
            prop_assert_eq!(entries.val.to_host().unwrap(), e_val);
            prop_assert_eq!(entries.read_ra.to_host().unwrap(), e_read);
            prop_assert_eq!(entries.rd_wa.to_host().unwrap(), e_wa);
            prop_assert_eq!(entries.prev_val.to_host().unwrap(), e_prev);
            prop_assert_eq!(entries.next_val.to_host().unwrap(), e_next);
        }

        #[test]
        fn instruction_raf_cycle_round_poly_matches_cpu(
            num_vars in 1usize..9,
            point_seed in fr_strategy(),
            seed in fr_strategy(),
        ) {
            use crate::stage5::eval_product_9;
            use jolt_field::FieldAccumulator;
            use jolt_poly::{BindingOrder, GruenSplitEqPolynomial};

            let len = 1usize << num_vars;
            let point: Vec<Fr> = (0..num_vars)
                .map(|i| point_seed + Fr::from_u64(i as u64))
                .collect();
            let combined: Vec<Fr> = (0..len)
                .map(|j| seed + Fr::from_u64((j + 1) as u64))
                .collect();
            let chunks: Vec<Vec<Fr>> = (0..8)
                .map(|c| {
                    (0..len)
                        .map(|j| seed + Fr::from_u64((c * len + j + 17) as u64))
                        .collect()
                })
                .collect();

            let split_eq = GruenSplitEqPolynomial::<Fr>::new(&point, BindingOrder::LowToHigh);
            let e_in = split_eq.e_in_current();
            let e_out = split_eq.e_out_current();
            let in_bits = e_in.len().trailing_zeros() as usize;

            type Acc = <Fr as Field>::Accumulator;
            let mut outer = [Acc::default(); 9];
            for (x_out, &eo) in e_out.iter().enumerate() {
                let mut inner = [Acc::default(); 9];
                let base = x_out << in_bits;
                for (x_in, &ei) in e_in.iter().enumerate() {
                    let group = base | x_in;
                    let pairs: [(Fr, Fr); 9] = std::array::from_fn(|index| {
                        if index == 0 {
                            (combined[2 * group] * ei, combined[2 * group + 1] * ei)
                        } else {
                            let c = &chunks[index - 1];
                            (c[2 * group], c[2 * group + 1])
                        }
                    });
                    let evals = eval_product_9(&pairs);
                    for (acc, eval) in inner.iter_mut().zip(evals) {
                        acc.acc_add(eval);
                    }
                }
                for (outer, inner) in outer.iter_mut().zip(inner) {
                    outer.fmadd(eo, inner.reduce());
                }
            }
            let expected: Vec<Fr> = outer.into_iter().map(|acc| acc.reduce()).collect();

            let c = ctx();
            let combined_dev = c.upload(&combined).unwrap();
            let chunk_devs: Vec<DeviceFrVec> =
                chunks.iter().map(|v| c.upload(v).unwrap()).collect();
            let chunk_refs: Vec<&DeviceFrVec> = chunk_devs.iter().collect();
            let e_in_dev = c.upload(e_in).unwrap();
            let e_out_dev = c.upload(e_out).unwrap();
            let got = c
                .instruction_raf_cycle_round_poly(InstructionRafCycleInputs {
                    combined: &combined_dev,
                    chunks: &chunk_refs,
                    e_in: &e_in_dev,
                    e_out: &e_out_dev,
                })
                .unwrap();
            prop_assert_eq!(got.to_vec(), expected);
        }

        #[test]
        fn ra_virtual_d4_round_poly_matches_cpu(
            num_vars in 1usize..9,
            num_virtuals in 1usize..4,
            point_seed in fr_strategy(),
            seed in fr_strategy(),
        ) {
            use crate::stage6::accumulate_instruction_ra_d4_products;
            use jolt_field::FieldAccumulator;
            use jolt_poly::{BindingOrder, GruenSplitEqPolynomial};

            let len = 1usize << num_vars;
            let point: Vec<Fr> = (0..num_vars)
                .map(|i| point_seed + Fr::from_u64(i as u64))
                .collect();
            let chunks: Vec<Vec<Fr>> = (0..4 * num_virtuals)
                .map(|c| {
                    (0..len)
                        .map(|j| seed + Fr::from_u64((c * len + j + 1) as u64))
                        .collect()
                })
                .collect();
            let gamma_powers: Vec<Fr> =
                (0..num_virtuals).map(|v| seed + Fr::from_u64((v + 3) as u64)).collect();

            let split_eq = GruenSplitEqPolynomial::<Fr>::new(&point, BindingOrder::LowToHigh);
            let e_in = split_eq.e_in_current();
            let e_out = split_eq.e_out_current();
            let in_bits = e_in.len().trailing_zeros() as usize;

            type Acc = <Fr as Field>::Accumulator;
            let mut expected_accs = [Acc::default(); 4];
            for (x_out, &eo) in e_out.iter().enumerate() {
                let mut inner = [Acc::default(); 4];
                let base = x_out << in_bits;
                for (x_in, &ei) in e_in.iter().enumerate() {
                    let row = base | x_in;
                    for (v, &gamma) in gamma_powers.iter().enumerate() {
                        let pair = |c: usize| {
                            let chunk = v * 4 + c;
                            (chunks[chunk][2 * row], chunks[chunk][2 * row + 1])
                        };
                        accumulate_instruction_ra_d4_products(
                            ei * gamma,
                            &mut inner,
                            pair(0),
                            pair(1),
                            pair(2),
                            pair(3),
                        );
                    }
                }
                for (acc, inner) in expected_accs.iter_mut().zip(inner) {
                    acc.fmadd(eo, inner.reduce());
                }
            }
            let expected: Vec<Fr> =
                expected_accs.into_iter().map(FieldAccumulator::reduce).collect();

            let c = ctx();
            let chunk_devs: Vec<DeviceFrVec> =
                chunks.iter().map(|v| c.upload(v).unwrap()).collect();
            let chunk_refs: Vec<&DeviceFrVec> = chunk_devs.iter().collect();
            let e_in_dev = c.upload(e_in).unwrap();
            let e_out_dev = c.upload(e_out).unwrap();
            let gamma_dev = c.upload(&gamma_powers).unwrap();
            let got = c
                .ra_virtual_d4_round_poly(RaVirtualD4Inputs {
                    chunks: &chunk_refs,
                    gamma_powers: &gamma_dev,
                    e_in: &e_in_dev,
                    e_out: &e_out_dev,
                })
                .unwrap();
            prop_assert_eq!(got.to_vec(), expected);
        }

        #[test]
        fn ra_virtual_d4_sparse_round_poly_matches_cpu(
            round in 1u32..=3,
            extra_vars in 1usize..6,
            num_virtuals in 1usize..4,
            log_chunk_domain in 1usize..6,
            seed in fr_strategy(),
        ) {
            use crate::stage6::accumulate_instruction_ra_d4_products;
            use jolt_field::FieldAccumulator;
            use jolt_poly::{BindingOrder, GruenSplitEqPolynomial};

            let num_chunks = 4 * num_virtuals;
            let num_sets = 1usize << (round - 1);
            let out_vars = extra_vars;
            let half = 1usize << out_vars;
            let source_rows = half << round;
            let chunk_domain = 1usize << log_chunk_domain;

            // num_sets table-sets, chunk-major within a set: [set][chunk][entry].
            let table_sets: Vec<Vec<Vec<Fr>>> = (0..num_sets)
                .map(|s| {
                    (0..num_chunks)
                        .map(|ch| {
                            (0..chunk_domain)
                                .map(|e| {
                                    seed + Fr::from_u64(
                                        (s * 977 + ch * chunk_domain + e + 1) as u64,
                                    )
                                })
                                .collect()
                        })
                        .collect()
                })
                .collect();

            // chunk-major values: values[chunk * source_rows + source], -1 = absent.
            let mut values = vec![-1i16; num_chunks * source_rows];
            for ch in 0..num_chunks {
                for s in 0..source_rows {
                    if (ch * 7 + s * 13) % 4 != 0 {
                        values[ch * source_rows + s] = ((ch * 5 + s * 3) % chunk_domain) as i16;
                    }
                }
            }

            let gather = |set: usize, ch: usize, s: usize| -> Fr {
                let v = values[ch * source_rows + s];
                if v >= 0 {
                    table_sets[set][ch][v as usize]
                } else {
                    Fr::from_u64(0)
                }
            };
            // Dense (lo,hi) for chunk `ch`, output row `row`: sum over sets, matching get_pair.
            let pair = |ch: usize, row: usize| -> (Fr, Fr) {
                let base = row << round;
                let mut lo = Fr::from_u64(0);
                let mut hi = Fr::from_u64(0);
                for set in 0..num_sets {
                    lo += gather(set, ch, base + set);
                    hi += gather(set, ch, base + num_sets + set);
                }
                (lo, hi)
            };

            let gamma_powers: Vec<Fr> =
                (0..num_virtuals).map(|v| seed + Fr::from_u64((v + 3) as u64)).collect();
            let point: Vec<Fr> = (0..=out_vars)
                .map(|i| seed + Fr::from_u64((i + 1) as u64))
                .collect();
            let split_eq = GruenSplitEqPolynomial::<Fr>::new(&point, BindingOrder::LowToHigh);
            let e_in = split_eq.e_in_current();
            let e_out = split_eq.e_out_current();
            let in_bits = e_in.len().trailing_zeros() as usize;

            type Acc = <Fr as Field>::Accumulator;
            let mut expected_accs = [Acc::default(); 4];
            for (x_out, &eo) in e_out.iter().enumerate() {
                let mut inner = [Acc::default(); 4];
                let base = x_out << in_bits;
                for (x_in, &ei) in e_in.iter().enumerate() {
                    let row = base | x_in;
                    for (v, &gamma) in gamma_powers.iter().enumerate() {
                        accumulate_instruction_ra_d4_products(
                            ei * gamma,
                            &mut inner,
                            pair(v * 4, row),
                            pair(v * 4 + 1, row),
                            pair(v * 4 + 2, row),
                            pair(v * 4 + 3, row),
                        );
                    }
                }
                for (acc, inner) in expected_accs.iter_mut().zip(inner) {
                    acc.fmadd(eo, inner.reduce());
                }
            }
            let expected: Vec<Fr> =
                expected_accs.into_iter().map(FieldAccumulator::reduce).collect();

            let c = ctx();
            let flat_tables: Vec<u64> = table_sets
                .iter()
                .flat_map(|set| set.iter().flatten().flat_map(|v| v.inner_limbs().0))
                .collect();
            let tables_dev = c.upload_u64_slice(&flat_tables).unwrap();
            let values_dev = c.upload_i16_slice(&values).unwrap();
            let e_in_dev = c.upload(e_in).unwrap();
            let e_out_dev = c.upload(e_out).unwrap();
            let gamma_dev = c.upload(&gamma_powers).unwrap();
            let got = c
                .ra_virtual_d4_sparse_round_poly(RaVirtualD4SparseInputs {
                    tables: &tables_dev,
                    values: &values_dev,
                    num_chunks,
                    chunk_domain,
                    source_rows,
                    gamma_powers: &gamma_dev,
                    e_in: &e_in_dev,
                    e_out: &e_out_dev,
                    round,
                })
                .unwrap();
            prop_assert_eq!(got.to_vec(), expected);
        }

        #[test]
        fn ra_virtual_d4_sparse_collapse_matches_cpu(
            out_vars in 0usize..7,
            num_chunks in 1usize..12,
            log_chunk_domain in 1usize..6,
            seed in fr_strategy(),
        ) {
            let out_len = 1usize << out_vars;
            let source_rows = out_len * 8;
            let chunk_domain = 1usize << log_chunk_domain;

            let table_sets: Vec<Vec<Vec<Fr>>> = (0..8)
                .map(|s| {
                    (0..num_chunks)
                        .map(|ch| {
                            (0..chunk_domain)
                                .map(|e| {
                                    seed + Fr::from_u64((s * 977 + ch * chunk_domain + e + 1) as u64)
                                })
                                .collect()
                        })
                        .collect()
                })
                .collect();

            let mut values = vec![-1i16; num_chunks * source_rows];
            for ch in 0..num_chunks {
                for s in 0..source_rows {
                    if (ch * 7 + s * 13) % 4 != 0 {
                        values[ch * source_rows + s] = ((ch * 5 + s * 3) % chunk_domain) as i16;
                    }
                }
            }

            let expected: Vec<Vec<Fr>> = (0..num_chunks)
                .map(|ch| {
                    (0..out_len)
                        .map(|j| {
                            let mut acc = Fr::from_u64(0);
                            for (set, table_set) in table_sets.iter().enumerate() {
                                let v = values[ch * source_rows + (8 * j + set)];
                                if v >= 0 {
                                    acc += table_set[ch][v as usize];
                                }
                            }
                            acc
                        })
                        .collect()
                })
                .collect();

            let c = ctx();
            let flat_tables: Vec<u64> = table_sets
                .iter()
                .flat_map(|set| set.iter().flatten().flat_map(|v| v.inner_limbs().0))
                .collect();
            let tables_dev = c.upload_u64_slice(&flat_tables).unwrap();
            let values_dev = c.upload_i16_slice(&values).unwrap();
            let got = c
                .ra_virtual_d4_sparse_collapse(
                    &tables_dev,
                    &values_dev,
                    num_chunks,
                    chunk_domain,
                    source_rows,
                    out_len,
                )
                .unwrap();
            let got_host: Vec<Vec<Fr>> = got.iter().map(|d| d.to_host().unwrap()).collect();
            prop_assert_eq!(got_host, expected);
        }

        #[test]
        fn bytecode_cycle_sparse_round_poly_matches_cpu(
            round in 1u32..=3,
            extra_vars in 1usize..6,
            num_chunks in 1usize..5,
            log_chunk_domain in 1usize..6,
            degree in 2usize..=6,
            seed in fr_strategy(),
        ) {
            use jolt_field::FieldAccumulator;

            let num_sets = 1usize << (round - 1);
            let half = 1usize << extra_vars;
            let source_rows = half << round;
            let chunk_domain = 1usize << log_chunk_domain;

            let table_sets: Vec<Vec<Vec<Fr>>> = (0..num_sets)
                .map(|s| {
                    (0..num_chunks)
                        .map(|ch| {
                            (0..chunk_domain)
                                .map(|e| {
                                    seed + Fr::from_u64((s * 977 + ch * chunk_domain + e + 1) as u64)
                                })
                                .collect()
                        })
                        .collect()
                })
                .collect();

            let mut values = vec![-1i16; num_chunks * source_rows];
            for ch in 0..num_chunks {
                for s in 0..source_rows {
                    if (ch * 7 + s * 13) % 4 != 0 {
                        values[ch * source_rows + s] = ((ch * 5 + s * 3) % chunk_domain) as i16;
                    }
                }
            }
            let combined_eq: Vec<Fr> = (0..half * 2)
                .map(|i| seed + Fr::from_u64((i + 51) as u64))
                .collect();

            let gather = |set: usize, ch: usize, s: usize| -> Fr {
                let v = values[ch * source_rows + s];
                if v >= 0 { table_sets[set][ch][v as usize] } else { Fr::from_u64(0) }
            };
            let pair = |ch: usize, row: usize| -> (Fr, Fr) {
                let base = row << round;
                let mut lo = Fr::from_u64(0);
                let mut hi = Fr::from_u64(0);
                for set in 0..num_sets {
                    lo += gather(set, ch, base + set);
                    hi += gather(set, ch, base + num_sets + set);
                }
                (lo, hi)
            };
            let point = |p: usize| -> Fr {
                if p == 0 { Fr::from_u64(0) } else { Fr::from_u64((p + 1) as u64) }
            };

            type Acc = <Fr as Field>::Accumulator;
            let mut evals = vec![Acc::default(); degree];
            for row in 0..half {
                for (p, eval) in evals.iter_mut().enumerate() {
                    let x = point(p);
                    let mut ra_product = Fr::from_u64(1);
                    for ch in 0..num_chunks {
                        let (lo, hi) = pair(ch, row);
                        ra_product *= lo + (hi - lo) * x;
                    }
                    let clo = combined_eq[2 * row];
                    let chi = combined_eq[2 * row + 1];
                    let weighted = clo + (chi - clo) * x;
                    eval.fmadd(ra_product, weighted);
                }
            }
            let expected: Vec<Fr> = evals.into_iter().map(FieldAccumulator::reduce).collect();

            let c = ctx();
            let flat_tables: Vec<u64> = table_sets
                .iter()
                .flat_map(|set| set.iter().flatten().flat_map(|v| v.inner_limbs().0))
                .collect();
            let tables_dev = c.upload_u64_slice(&flat_tables).unwrap();
            let values_dev = c.upload_i16_slice(&values).unwrap();
            let combined_dev = c.upload(&combined_eq).unwrap();
            let got = c
                .bytecode_cycle_sparse_round_poly(BytecodeCycleSparseInputs {
                    tables: &tables_dev,
                    values: &values_dev,
                    combined_eq: &combined_dev,
                    num_chunks,
                    chunk_domain,
                    source_rows,
                    degree,
                    round,
                })
                .unwrap();
            prop_assert_eq!(got, expected);
        }

        #[test]
        fn instruction_raf_cycle_sparse_round_poly_matches_cpu(
            round in 1u32..=3,
            extra_vars in 1usize..6,
            log_chunk_domain in 1usize..6,
            seed in fr_strategy(),
        ) {
            use crate::stage5::eval_product_9;
            use jolt_field::FieldAccumulator;
            use jolt_poly::{BindingOrder, GruenSplitEqPolynomial};

            let num_chunks = 8usize;
            let num_sets = 1usize << (round - 1);
            let half = 1usize << extra_vars;
            let source_rows = half << round;
            let chunk_domain = 1usize << log_chunk_domain;

            let table_sets: Vec<Vec<Vec<Fr>>> = (0..num_sets)
                .map(|s| {
                    (0..num_chunks)
                        .map(|ch| {
                            (0..chunk_domain)
                                .map(|e| {
                                    seed + Fr::from_u64((s * 977 + ch * chunk_domain + e + 1) as u64)
                                })
                                .collect()
                        })
                        .collect()
                })
                .collect();

            // Always-present u16 values (no sentinel): values[chunk * source_rows + source].
            let mut values = vec![0u16; num_chunks * source_rows];
            for ch in 0..num_chunks {
                for s in 0..source_rows {
                    values[ch * source_rows + s] = ((ch * 5 + s * 3) % chunk_domain) as u16;
                }
            }
            let combined: Vec<Fr> = (0..half * 2)
                .map(|i| seed + Fr::from_u64((i + 51) as u64))
                .collect();

            let gather = |set: usize, ch: usize, s: usize| -> Fr {
                table_sets[set][ch][values[ch * source_rows + s] as usize]
            };
            let pair = |ch: usize, row: usize| -> (Fr, Fr) {
                let base = row << round;
                let mut lo = Fr::from_u64(0);
                let mut hi = Fr::from_u64(0);
                for set in 0..num_sets {
                    lo += gather(set, ch, base + set);
                    hi += gather(set, ch, base + num_sets + set);
                }
                (lo, hi)
            };

            let point: Vec<Fr> = (0..=extra_vars)
                .map(|i| seed + Fr::from_u64((i + 1) as u64))
                .collect();
            let split_eq = GruenSplitEqPolynomial::<Fr>::new(&point, BindingOrder::LowToHigh);
            let e_in = split_eq.e_in_current();
            let e_out = split_eq.e_out_current();
            let in_bits = e_in.len().trailing_zeros() as usize;

            type Acc = <Fr as Field>::Accumulator;
            let mut outer = [Acc::default(); 9];
            for (x_out, &eo) in e_out.iter().enumerate() {
                let mut inner = [Acc::default(); 9];
                let base = x_out << in_bits;
                for (x_in, &ei) in e_in.iter().enumerate() {
                    let row = base | x_in;
                    let pairs: [(Fr, Fr); 9] = std::array::from_fn(|index| {
                        if index == 0 {
                            (combined[2 * row] * ei, combined[2 * row + 1] * ei)
                        } else {
                            pair(index - 1, row)
                        }
                    });
                    let evals = eval_product_9(&pairs);
                    for (acc, eval) in inner.iter_mut().zip(evals) {
                        acc.acc_add(eval);
                    }
                }
                for (outer, inner) in outer.iter_mut().zip(inner) {
                    outer.fmadd(eo, inner.reduce());
                }
            }
            let expected: Vec<Fr> = outer.into_iter().map(|acc| acc.reduce()).collect();

            let c = ctx();
            let flat_tables: Vec<u64> = table_sets
                .iter()
                .flat_map(|set| set.iter().flatten().flat_map(|v| v.inner_limbs().0))
                .collect();
            let tables_dev = c.upload_u64_slice(&flat_tables).unwrap();
            let values_dev = c.upload_u16_slice(&values).unwrap();
            let combined_dev = c.upload(&combined).unwrap();
            let e_in_dev = c.upload(e_in).unwrap();
            let e_out_dev = c.upload(e_out).unwrap();
            let got = c
                .instruction_raf_cycle_sparse_round_poly(InstructionRafCycleSparseInputs {
                    tables: &tables_dev,
                    values: &values_dev,
                    combined: &combined_dev,
                    num_chunks,
                    chunk_domain,
                    source_rows,
                    e_in: &e_in_dev,
                    e_out: &e_out_dev,
                    round,
                })
                .unwrap();
            prop_assert_eq!(got.to_vec(), expected);
        }

        #[test]
        fn instruction_raf_cycle_sparse_collapse_matches_cpu(
            out_vars in 0usize..7,
            log_chunk_domain in 1usize..6,
            seed in fr_strategy(),
        ) {
            let num_chunks = 8usize;
            let out_len = 1usize << out_vars;
            let source_rows = out_len * 8;
            let chunk_domain = 1usize << log_chunk_domain;

            let table_sets: Vec<Vec<Vec<Fr>>> = (0..8)
                .map(|s| {
                    (0..num_chunks)
                        .map(|ch| {
                            (0..chunk_domain)
                                .map(|e| {
                                    seed + Fr::from_u64((s * 977 + ch * chunk_domain + e + 1) as u64)
                                })
                                .collect()
                        })
                        .collect()
                })
                .collect();

            let mut values = vec![0u16; num_chunks * source_rows];
            for ch in 0..num_chunks {
                for s in 0..source_rows {
                    values[ch * source_rows + s] = ((ch * 5 + s * 3) % chunk_domain) as u16;
                }
            }

            let expected: Vec<Vec<Fr>> = (0..num_chunks)
                .map(|ch| {
                    (0..out_len)
                        .map(|j| {
                            let mut acc = Fr::from_u64(0);
                            for (set, table_set) in table_sets.iter().enumerate() {
                                let v = values[ch * source_rows + (8 * j + set)];
                                acc += table_set[ch][v as usize];
                            }
                            acc
                        })
                        .collect()
                })
                .collect();

            let c = ctx();
            let flat_tables: Vec<u64> = table_sets
                .iter()
                .flat_map(|set| set.iter().flatten().flat_map(|v| v.inner_limbs().0))
                .collect();
            let tables_dev = c.upload_u64_slice(&flat_tables).unwrap();
            let values_dev = c.upload_u16_slice(&values).unwrap();
            let got = c
                .instruction_raf_cycle_sparse_collapse(
                    &tables_dev,
                    &values_dev,
                    num_chunks,
                    chunk_domain,
                    source_rows,
                    out_len,
                )
                .unwrap();
            let got_host: Vec<Vec<Fr>> =
                got.iter().map(|chunk| chunk.to_host().unwrap()).collect();
            prop_assert_eq!(got_host, expected);
        }
    }

    #[test]
    fn edge_cases() {
        let c = ctx();
        let zero = Fr::zero();
        let one = Fr::one();

        let mut empty = c.upload(&[]).unwrap();
        c.add(&mut empty, &c.upload(&[]).unwrap()).unwrap();
        assert_eq!(empty.to_host().unwrap(), Vec::<Fr>::new());
        assert_eq!(c.sum(&empty).unwrap(), zero);
        assert_eq!(c.product(&empty).unwrap(), one);

        let single = c.upload(&[Fr::from_u64(42)]).unwrap();
        assert_eq!(c.sum(&single).unwrap(), Fr::from_u64(42));
        assert_eq!(c.product(&single).unwrap(), Fr::from_u64(42));

        let b = c.upload(&[one, one, Fr::from_u64(6)]).unwrap();

        let mut a = c.upload(&[zero, one, Fr::from_u64(7)]).unwrap();
        c.add(&mut a, &b).unwrap();
        assert_eq!(
            a.to_host().unwrap(),
            vec![one, Fr::from_u64(2), Fr::from_u64(13)]
        );

        let mut a = c.upload(&[zero, one, Fr::from_u64(7)]).unwrap();
        c.mul(&mut a, &b).unwrap();
        assert_eq!(
            a.to_host().unwrap(),
            vec![zero, one, Fr::from_u64(42)]
        );
    }

    #[test]
    fn resident_witness_caches_and_refreshes() {
        let c = ctx();
        let witness: Vec<Fr> = (0..512u64).map(Fr::from_u64).collect();

        let first = c.resident_witness(&witness).unwrap();
        let second = c.resident_witness(&witness).unwrap();
        assert!(Arc::ptr_eq(&first, &second), "same witness must hit the cache");
        assert_eq!(first.to_host().unwrap(), witness);

        let mut other = witness.clone();
        other[300] += Fr::one();
        let third = c.resident_witness(&other).unwrap();
        assert!(!Arc::ptr_eq(&first, &third), "changed witness must re-upload");
        assert_eq!(third.to_host().unwrap(), other);

        let again = c.resident_witness(&witness).unwrap();
        assert!(
            !Arc::ptr_eq(&first, &again),
            "cache holds one entry; a different witness evicts the prior one"
        );
        assert_eq!(again.to_host().unwrap(), witness);
    }

    #[test]
    fn resident_committed_clone_dedups_and_is_independent() {
        let c = ctx();
        let poly_a: Vec<Fr> = (0..512u64).map(Fr::from_u64).collect();
        let poly_b: Vec<Fr> = (0..512u64).map(|v| Fr::from_u64(v) + Fr::one()).collect();

        let first = c.resident_committed_clone(&poly_a).unwrap();
        let second = c.resident_committed_clone(&poly_a).unwrap();
        assert_eq!(first.to_host().unwrap(), poly_a);
        assert_eq!(second.to_host().unwrap(), poly_a);

        let mut scratch = c.upload(&[]).unwrap();
        let mut bound = first;
        c.bind(&mut bound, &mut scratch, Fr::from_u64(9)).unwrap();
        assert_eq!(bound.len(), poly_a.len() / 2);
        assert_eq!(second.to_host().unwrap(), poly_a, "sibling clone unaffected by bind");

        let other = c.resident_committed_clone(&poly_b).unwrap();
        assert_eq!(other.to_host().unwrap(), poly_b);

        let again = c.resident_committed_clone(&poly_a).unwrap();
        assert_eq!(again.to_host().unwrap(), poly_a);
    }

    #[test]
    fn upload_many_matches_individual_uploads() {
        let c = ctx();
        let f0: Vec<Fr> = (0..300u64).map(Fr::from_u64).collect();
        let f1: Vec<Fr> = (1000..1001u64).map(Fr::from_u64).collect();
        let f2: Vec<Fr> = (5..517u64).map(|v| Fr::from_u64(v) + Fr::one()).collect();
        let factors = [f0.as_slice(), f1.as_slice(), f2.as_slice()];

        let batched = c.upload_many(&factors).unwrap();
        assert_eq!(batched.len(), 3);
        for (dev, expected) in batched.iter().zip(&factors) {
            assert_eq!(dev.len(), expected.len());
            assert_eq!(&dev.to_host().unwrap(), *expected);
        }

        // Each batched factor must be independently bindable (i.e. an owned buffer, not a view).
        let mut scratch = c.upload(&[]).unwrap();
        let mut first = batched.into_iter().next().unwrap();
        c.bind(&mut first, &mut scratch, Fr::from_u64(7)).unwrap();
        assert_eq!(first.len(), f0.len() / 2);
    }

    #[test]
    fn upload_many_handles_empty_and_zero_len() {
        let c = ctx();
        assert!(c.upload_many(&[]).unwrap().is_empty());

        let nonempty: Vec<Fr> = (0..8u64).map(Fr::from_u64).collect();
        let empty: Vec<Fr> = Vec::new();
        let factors = [empty.as_slice(), nonempty.as_slice(), empty.as_slice()];
        let batched = c.upload_many(&factors).unwrap();
        assert_eq!(batched.len(), 3);
        assert_eq!(batched[0].len(), 0);
        assert_eq!(batched[1].to_host().unwrap(), nonempty);
        assert_eq!(batched[2].len(), 0);
    }

    #[test]
    fn try_clone_is_independent() {
        let c = ctx();
        let original = vec![Fr::from_u64(1), Fr::from_u64(2), Fr::from_u64(3)];
        let a = c.upload(&original).unwrap();
        let mut cloned = a.try_clone().unwrap();
        let ones = c.upload(&[Fr::one(); 3]).unwrap();
        c.add(&mut cloned, &ones).unwrap();

        assert_eq!(a.to_host().unwrap(), original);
        assert_eq!(
            cloned.to_host().unwrap(),
            vec![Fr::from_u64(2), Fr::from_u64(3), Fr::from_u64(4)]
        );
    }

    #[test]
    fn reduce_preserves_input() {
        let c = ctx();
        // Larger than BLOCK so the multi-pass loop runs.
        let original: Vec<Fr> = (1..=1000u64).map(Fr::from_u64).collect();
        let a = c.upload(&original).unwrap();

        let expected_sum: Fr = original.iter().copied().sum();
        assert_eq!(c.sum(&a).unwrap(), expected_sum);
        assert_eq!(a.to_host().unwrap(), original);

        let expected_product: Fr = original.iter().copied().product();
        assert_eq!(c.product(&a).unwrap(), expected_product);
        assert_eq!(a.to_host().unwrap(), original);
    }
}
