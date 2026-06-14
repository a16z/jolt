use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use cudarc::driver::{
    result as cuda_result, CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig,
    PushKernelArg,
};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

use crate::Fr;

const LIMBS: usize = 4;
const BLOCK: u32 = 256;

const KERNEL_SRC: &str = r#"
typedef unsigned long long u64;
typedef unsigned __int128 u128;

__device__ __constant__ u64 MODULUS[4] = {
    0x43e1f593f0000001ULL, 0x2833e84879b97091ULL,
    0xb85045b68181585dULL, 0x30644e72e131a029ULL
};
__device__ __constant__ u64 INV = 0xc2e1f593efffffffULL;

__device__ __forceinline__ u64 mac(u64 a, u64 b, u64 c, u64 *carry) {
    u128 t = (u128)a + (u128)b * (u128)c + (u128)(*carry);
    *carry = (u64)(t >> 64);
    return (u64)t;
}

__device__ __forceinline__ u64 adc(u64 a, u64 b, u64 *carry) {
    u128 t = (u128)a + (u128)b + (u128)(*carry);
    *carry = (u64)(t >> 64);
    return (u64)t;
}

__device__ __forceinline__ u64 sbb(u64 a, u64 b, u64 *borrow) {
    u128 t = ((u128)1 << 64) + (u128)a - (u128)b - (u128)(*borrow);
    *borrow = (t >> 64) == 0 ? 1 : 0;
    return (u64)t;
}

__device__ __forceinline__ int geq_modulus(const u64 *a) {
    for (int i = 3; i >= 0; i--) {
        if (a[i] != MODULUS[i]) return a[i] > MODULUS[i];
    }
    return 1;
}

__device__ __forceinline__ void sub_modulus(u64 *a) {
    u64 borrow = 0;
    for (int i = 0; i < 4; i++) a[i] = sbb(a[i], MODULUS[i], &borrow);
}

__device__ void fr_add(const u64 *a, const u64 *b, u64 *out) {
    u64 carry = 0;
    for (int i = 0; i < 4; i++) out[i] = adc(a[i], b[i], &carry);
    if (carry != 0 || geq_modulus(out)) sub_modulus(out);
}

__device__ void fr_sub(const u64 *a, const u64 *b, u64 *out) {
    u64 borrow = 0;
    for (int i = 0; i < 4; i++) out[i] = sbb(a[i], b[i], &borrow);
    if (borrow != 0) {
        u64 carry = 0;
        for (int i = 0; i < 4; i++) out[i] = adc(out[i], MODULUS[i], &carry);
    }
}

__device__ void fr_mul(const u64 *a, const u64 *b, u64 *out) {
    u64 t[6];
    for (int i = 0; i < 6; i++) t[i] = 0;

    for (int i = 0; i < 4; i++) {
        u64 carry = 0;
        for (int j = 0; j < 4; j++) t[j] = mac(t[j], a[j], b[i], &carry);
        u64 c = 0;
        t[4] = adc(t[4], carry, &c);
        t[5] = adc(t[5], 0, &c);

        u64 m = t[0] * INV;
        u64 c2 = 0;
        mac(t[0], m, MODULUS[0], &c2);
        for (int j = 1; j < 4; j++) t[j - 1] = mac(t[j], m, MODULUS[j], &c2);
        u64 c3 = 0;
        t[3] = adc(t[4], c2, &c3);
        t[4] = adc(t[5], 0, &c3);
        t[5] = 0;
    }

    for (int i = 0; i < 4; i++) out[i] = t[i];
    if (t[4] != 0 || geq_modulus(out)) sub_modulus(out);
}

extern "C" __global__ void add_kernel(u64 *io, const u64 *b, unsigned long n) {
    unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) fr_add(io + i * 4, b + i * 4, io + i * 4);
}

extern "C" __global__ void sub_kernel(u64 *io, const u64 *b, unsigned long n) {
    unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) fr_sub(io + i * 4, b + i * 4, io + i * 4);
}

extern "C" __global__ void mul_kernel(u64 *io, const u64 *b, unsigned long n) {
    unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) fr_mul(io + i * 4, b + i * 4, io + i * 4);
}

extern "C" __global__ void fma_kernel(u64 *io, const u64 *b, const u64 *c, unsigned long n) {
    unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        u64 prod[4];
        fr_mul(io + i * 4, b + i * 4, prod);
        fr_add(prod, c + i * 4, io + i * 4);
    }
}

extern "C" __global__ void sum_reduce(u64 *out, const u64 *in, unsigned long n) {
    extern __shared__ u64 sdata[];
    u64 *acc = sdata + threadIdx.x * 4;

    unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        for (int k = 0; k < 4; k++) acc[k] = in[i * 4 + k];
    } else {
        for (int k = 0; k < 4; k++) acc[k] = 0;
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            u64 *other = sdata + (threadIdx.x + s) * 4;
            u64 tmp[4];
            fr_add(acc, other, tmp);
            for (int k = 0; k < 4; k++) acc[k] = tmp[k];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        for (int k = 0; k < 4; k++) out[blockIdx.x * 4 + k] = acc[k];
    }
}

extern "C" __global__ void product_reduce(u64 *out, const u64 *in, unsigned long n, const u64 *one) {
    extern __shared__ u64 sdata[];
    u64 *acc = sdata + threadIdx.x * 4;

    unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        for (int k = 0; k < 4; k++) acc[k] = in[i * 4 + k];
    } else {
        for (int k = 0; k < 4; k++) acc[k] = one[k];
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            u64 *other = sdata + (threadIdx.x + s) * 4;
            u64 tmp[4];
            fr_mul(acc, other, tmp);
            for (int k = 0; k < 4; k++) acc[k] = tmp[k];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        for (int k = 0; k < 4; k++) out[blockIdx.x * 4 + k] = acc[k];
    }
}
"#;

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

pub struct CudaFieldContext {
    stream: Arc<CudaStream>,
    add: CudaFunction,
    sub: CudaFunction,
    mul: CudaFunction,
    fma: CudaFunction,
    sum_reduce: CudaFunction,
    product_reduce: CudaFunction,
    one_dev: CudaSlice<u64>,
    staging: PinnedStaging,
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
        self.stream.memcpy_dtoh(&self.buf, staging.as_mut_slice(n))?;
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
}

#[inline]
fn fr_to_limbs(f: Fr) -> [u64; LIMBS] {
    f.inner_limbs().0
}

#[inline]
fn limbs_to_fr(limbs: [u64; LIMBS]) -> Fr {
    Fr::from_bigint_unchecked(crate::Limbs(limbs))
}

fn unflatten(raw: &[u64]) -> Vec<Fr> {
    raw.chunks_exact(LIMBS)
        .map(|c| limbs_to_fr([c[0], c[1], c[2], c[3]]))
        .collect()
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

fn lock_pool(pool: &PinnedStaging) -> MutexGuard<'_, PinnedPool> {
    pool.lock().unwrap_or_else(PoisonError::into_inner)
}

impl CudaFieldContext {
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
            sum_reduce: module.load_function("sum_reduce")?,
            product_reduce: module.load_function("product_reduce")?,
            stream,
            one_dev,
            staging: Arc::new(Mutex::new(PinnedPool::default())),
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

    fn reduce(&self, sum: bool, values: &DeviceFrVec) -> Result<Fr, CudaError> {
        use num_traits::{One, Zero};
        if values.len == 0 {
            return Ok(if sum { Fr::zero() } else { Fr::one() });
        }

        // The reduce kernels only read their input and write to a fresh `out_dev`,
        // so the first pass reads `values.buf` directly; later passes consume the
        // owned intermediate. No copy of the input is needed.
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

        let final_buf = owned.as_ref().unwrap_or(&values.buf);
        let raw = self.stream.clone_dtoh(final_buf)?;
        Ok(limbs_to_fr([raw[0], raw[1], raw[2], raw[3]]))
    }

    pub fn sum(&self, values: &DeviceFrVec) -> Result<Fr, CudaError> {
        self.reduce(true, values)
    }

    pub fn product(&self, values: &DeviceFrVec) -> Result<Fr, CudaError> {
        self.reduce(false, values)
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::Field;
    use num_traits::{One, Zero};
    use proptest::prelude::*;

    fn ctx() -> CudaFieldContext {
        CudaFieldContext::new(0).unwrap()
    }

    fn fr_strategy() -> impl Strategy<Value = Fr> {
        any::<[u8; 32]>().prop_map(|bytes| Fr::from_bytes(&bytes))
    }

    fn fr_vec_strategy(max: usize) -> impl Strategy<Value = Vec<Fr>> {
        prop::collection::vec(fr_strategy(), 0..max)
    }

    proptest! {
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
