use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use cudarc::driver::{
    result as cuda_result, CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig,
    PushKernelArg,
};
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
    include_str!("cuda/row_dots.cu"),
    include_str!("cuda/dense_outer_fused.cu"),
    include_str!("cuda/dense_outer.cu"),
    include_str!("cuda/cubic.cu"),
    include_str!("cuda/round_poly.cu"),
    include_str!("cuda/dense_product.cu"),
    include_str!("cuda/gruen_round_poly.cu"),
    include_str!("cuda/uniskip.cu"),
    include_str!("cuda/gather8.cu"),
    include_str!("cuda/hamming.cu"),
    include_str!("cuda/hamming_booleanity.cu"),
    include_str!("cuda/reduce.cu"),
);

#[derive(Debug)]
pub enum CudaError {
    Compile(cudarc::nvrtc::CompileError),
    Driver(cudarc::driver::DriverError),
    Pool,
}

impl std::fmt::Display for CudaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CudaError::Compile(e) => write!(f, "nvrtc compile error: {e:?}"),
            CudaError::Driver(e) => write!(f, "cuda driver error: {e:?}"),
            CudaError::Pool => write!(f, "pinned staging pool invariant violated"),
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
    hamming_pairs: CudaFunction,
    hamming_booleanity_pairs: CudaFunction,
    sum_reduce: CudaFunction,
    product_reduce: CudaFunction,
    one_dev: CudaSlice<u64>,
    staging: PinnedStaging,
    resident_witness: ResidentCache,
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
        let mut pool = lock_pool(&self.staging);
        let staging = pool.ensure(self.stream.context(), n)?;
        self.stream
            .memcpy_dtoh(&self.buf.slice(0..n), staging.as_mut_slice(n))?;
        self.stream.synchronize()?;
        Ok(unflatten(staging.as_slice(n)))
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
        let raw = self.stream.clone_dtoh(&self.buf.slice(0..LIMBS))?;
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
    len: usize,
    hash: u64,
}

impl WitnessKey {
    fn of(witness: &[Fr]) -> Self {
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        for &value in witness {
            for limb in fr_to_limbs(value) {
                hash = (hash ^ limb).wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        Self {
            len: witness.len(),
            hash,
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
    pub term_coeffs: &'a [Fr],
    pub term_factor_offsets: &'a [u32],
    pub term_factor_indices: &'a [u32],
    pub degree: usize,
}

pub struct GruenRoundPolyInputs<'a> {
    pub factors: &'a [&'a DeviceFrVec],
    pub term_coeffs: &'a [Fr],
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

pub struct HammingRoundPolyInputs<'a> {
    pub g: &'a [&'a DeviceFrVec],
    pub eq_virt: &'a [&'a DeviceFrVec],
    pub eq_bool: &'a DeviceFrVec,
    pub gamma_powers: &'a [Fr],
    pub scale: Fr,
}

pub struct HammingBooleanityInputs<'a> {
    pub hamming_weight: &'a DeviceFrVec,
    pub e_in: &'a DeviceFrVec,
    pub e_out: &'a DeviceFrVec,
}

pub struct RaVirtualD4Inputs<'a> {
    pub chunks: [&'a DeviceFrVec; 4],
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
            hamming_pairs: module.load_function("hamming_pairs")?,
            hamming_booleanity_pairs: module.load_function("hamming_booleanity_pairs")?,
            sum_reduce: module.load_function("sum_reduce")?,
            product_reduce: module.load_function("product_reduce")?,
            stream,
            one_dev,
            staging: Arc::new(Mutex::new(PinnedPool::default())),
            resident_witness: Arc::new(Mutex::new(None)),
        })
    }

    pub fn upload(&self, values: &[Fr]) -> Result<DeviceFrVec, CudaError> {
        let buf = if values.is_empty() {
            self.stream.alloc_zeros(0)?
        } else {
            let n = values.len() * LIMBS;
            let mut pool = lock_pool(&self.staging);
            let staging = pool.ensure(self.stream.context(), n)?;
            for (slot, &v) in staging.as_mut_slice(n).chunks_exact_mut(LIMBS).zip(values) {
                slot.copy_from_slice(&fr_to_limbs(v));
            }
            let dev = self.stream.clone_htod(staging.as_slice(n))?;
            self.stream.synchronize()?;
            dev
        };
        Ok(DeviceFrVec {
            stream: self.stream.clone(),
            buf,
            len: values.len(),
            staging: self.staging.clone(),
        })
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
            let first = self.stream.clone_htod(first_group_rows)?;
            let second = self.stream.clone_htod(second_group_rows)?;

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
            let a_offsets = self.stream.clone_htod(inputs.a_offsets)?;
            let a_vars = self.stream.clone_htod(inputs.a_vars)?;
            let b_offsets = self.stream.clone_htod(inputs.b_offsets)?;
            let b_vars = self.stream.clone_htod(inputs.b_vars)?;

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
            let a_offsets_dev = self.stream.clone_htod(a_offsets)?;
            let a_vars_dev = self.stream.clone_htod(a_vars)?;
            let b_offsets_dev = self.stream.clone_htod(b_offsets)?;
            let b_vars_dev = self.stream.clone_htod(b_vars)?;

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

    #[expect(clippy::todo, unused_variables)]
    pub fn ra_virtual_d4_round_poly(
        &self,
        inputs: RaVirtualD4Inputs<'_>,
    ) -> Result<[Fr; 4], CudaError> {
        todo!()
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

        let pack = |polys: &[&DeviceFrVec]| -> Result<CudaSlice<u64>, CudaError> {
            let mut packed: CudaSlice<u64> = self.stream.alloc_zeros(num_ra * len * LIMBS)?;
            for (index, poly) in polys.iter().enumerate() {
                let offset = index * len * LIMBS;
                self.stream.memcpy_dtod(
                    &poly.buf.slice(0..len * LIMBS),
                    &mut packed.slice_mut(offset..offset + len * LIMBS),
                )?;
            }
            Ok(packed)
        };
        let g_packed = pack(inputs.g)?;
        let eq_virt_packed = pack(inputs.eq_virt)?;
        let gammas_dev = self.upload(inputs.gamma_powers)?;
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
            .arg(&gammas_dev.buf)
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
        let indices_dev = self.stream.clone_htod(indices)?;
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
        let first_rows = self.stream.clone_htod(inputs.first_group_rows)?;
        let second_rows = self.stream.clone_htod(inputs.second_group_rows)?;

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
        self.stream.synchronize()?;

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
        self.stream.synchronize()?;

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

        let mut packed: CudaSlice<u64> =
            self.stream.alloc_zeros(terms.factors.len() * pair_stride * 2 * LIMBS)?;
        for (index, factor) in terms.factors.iter().enumerate() {
            let offset = index * pair_stride * 2 * LIMBS;
            self.stream.memcpy_dtod(
                &factor.buf.slice(0..pair_stride * 2 * LIMBS),
                &mut packed.slice_mut(offset..offset + pair_stride * 2 * LIMBS),
            )?;
        }

        let coeffs_dev = self.upload(terms.term_coeffs)?;
        let offsets_dev = self.stream.clone_htod(terms.term_factor_offsets)?;
        let indices_dev = self.stream.clone_htod(terms.term_factor_indices)?;

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
            .arg(&packed)
            .arg(&coeffs_dev.buf)
            .arg(&offsets_dev)
            .arg(&indices_dev)
            .arg(&pair_stride_arg)
            .arg(&num_terms_arg)
            .arg(&degree_arg)
            .arg(&half_arg);
        // SAFETY: one thread per row reads its pair from each packed factor and the
        // term tables (bounded by offsets), building the width=degree+1 monomial
        // coefficients in a thread-local buffer; shared memory holds `block` tuples.
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

        let mut packed: CudaSlice<u64> =
            self.stream.alloc_zeros(terms.factors.len() * pair_stride * 2 * LIMBS)?;
        for (index, factor) in terms.factors.iter().enumerate() {
            let offset = index * pair_stride * 2 * LIMBS;
            self.stream.memcpy_dtod(
                &factor.buf.slice(0..pair_stride * 2 * LIMBS),
                &mut packed.slice_mut(offset..offset + pair_stride * 2 * LIMBS),
            )?;
        }

        let points_dev = self.upload(points)?;
        let coeffs_dev = self.upload(terms.term_coeffs)?;
        let offsets_dev = self.stream.clone_htod(terms.term_factor_offsets)?;
        let indices_dev = self.stream.clone_htod(terms.term_factor_indices)?;

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
            .arg(&packed)
            .arg(&points_dev.buf)
            .arg(&coeffs_dev.buf)
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

        let mut packed: CudaSlice<u64> =
            self.stream.alloc_zeros(inputs.factors.len() * pair_stride * 2 * LIMBS)?;
        for (index, factor) in inputs.factors.iter().enumerate() {
            let offset = index * pair_stride * 2 * LIMBS;
            self.stream.memcpy_dtod(
                &factor.buf.slice(0..pair_stride * 2 * LIMBS),
                &mut packed.slice_mut(offset..offset + pair_stride * 2 * LIMBS),
            )?;
        }

        let coeffs_dev = self.upload(inputs.term_coeffs)?;
        let offsets_dev = self.stream.clone_htod(inputs.term_factor_offsets)?;
        let indices_dev = self.stream.clone_htod(inputs.term_factor_indices)?;

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
            .arg(&packed)
            .arg(&coeffs_dev.buf)
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

        let mut packed: CudaSlice<u64> =
            self.stream.alloc_zeros(terms.factors.len() * pair_stride * 2 * LIMBS)?;
        for (index, factor) in terms.factors.iter().enumerate() {
            let offset = index * pair_stride * 2 * LIMBS;
            self.stream.memcpy_dtod(
                &factor.buf.slice(0..pair_stride * 2 * LIMBS),
                &mut packed.slice_mut(offset..offset + pair_stride * 2 * LIMBS),
            )?;
        }

        let points: Vec<Fr> = (0..degree)
            .map(|e| Fr::from_u64(if e == 0 { 0 } else { (e + 1) as u64 }))
            .collect();
        let points_dev = self.upload(&points)?;
        let coeffs_dev = self.upload(terms.term_coeffs)?;
        let offsets_dev = self.stream.clone_htod(terms.term_factor_offsets)?;
        let indices_dev = self.stream.clone_htod(terms.term_factor_indices)?;

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
            .arg(&packed)
            .arg(&points_dev.buf)
            .arg(&coeffs_dev.buf)
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
                    term_coeffs: &term_coeffs,
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
                    term_coeffs: &term_coeffs,
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
                let got = c
                    .gruen_round_poly(GruenRoundPolyInputs {
                        factors: &factor_refs,
                        term_coeffs: &term_coeffs,
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
                let got = c
                    .gruen_round_poly(GruenRoundPolyInputs {
                        factors: &factor_refs,
                        term_coeffs: &term_coeffs,
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
                        term_coeffs: &term_coeffs,
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
            };
            let expected = state
                .round_poly(previous_claim, Stage7Relation::HammingWeightClaimReduction)
                .unwrap();

            let c = ctx();
            let g_devs: Vec<DeviceFrVec> = g.iter().map(|v| c.upload(v).unwrap()).collect();
            let eq_virt_devs: Vec<DeviceFrVec> =
                eq_virt.iter().map(|v| c.upload(v).unwrap()).collect();
            let eq_bool_dev = c.upload(&eq_bool).unwrap();
            let g_refs: Vec<&DeviceFrVec> = g_devs.iter().collect();
            let eq_virt_refs: Vec<&DeviceFrVec> = eq_virt_devs.iter().collect();

            let evals = c
                .hamming_round_poly(HammingRoundPolyInputs {
                    g: &g_refs,
                    eq_virt: &eq_virt_refs,
                    eq_bool: &eq_bool_dev,
                    gamma_powers: &gamma_powers,
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
        #[ignore = "CudaKernelContext::ra_virtual_d4_round_poly is todo!()"]
        fn ra_virtual_d4_round_poly_matches_cpu(
            num_vars in 1usize..10,
            point_seed in fr_strategy(),
            seed in fr_strategy(),
        ) {
            use crate::stage6::accumulate_instruction_ra_d4_products;
            use jolt_field::FieldAccumulator;
            use jolt_poly::{BindingOrder, GruenSplitEqPolynomial};

            // RamRaVirtual config: 4 dense chunks, 1 virtual, weight = e_in (gamma
            // absorbed into a unit gamma^0). Mirror round_poly_sparse_d4's fold.
            let len = 1usize << num_vars;
            let point: Vec<Fr> = (0..num_vars)
                .map(|i| point_seed + Fr::from_u64(i as u64))
                .collect();
            let chunks: Vec<Vec<Fr>> = (0..4)
                .map(|c| {
                    (0..len)
                        .map(|j| seed + Fr::from_u64((c * len + j + 1) as u64))
                        .collect()
                })
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
                    let pair = |c: usize| (chunks[c][2 * row], chunks[c][2 * row + 1]);
                    accumulate_instruction_ra_d4_products(
                        ei,
                        &mut inner,
                        pair(0),
                        pair(1),
                        pair(2),
                        pair(3),
                    );
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
            let e_in_dev = c.upload(e_in).unwrap();
            let e_out_dev = c.upload(e_out).unwrap();
            let got = c
                .ra_virtual_d4_round_poly(RaVirtualD4Inputs {
                    chunks: [
                        &chunk_devs[0],
                        &chunk_devs[1],
                        &chunk_devs[2],
                        &chunk_devs[3],
                    ],
                    e_in: &e_in_dev,
                    e_out: &e_out_dev,
                })
                .unwrap();
            prop_assert_eq!(got.to_vec(), expected);
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
