use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
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

extern "C" __global__ void add_kernel(u64 *out, const u64 *a, const u64 *b, unsigned long n) {
    unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) fr_add(a + i * 4, b + i * 4, out + i * 4);
}

extern "C" __global__ void sub_kernel(u64 *out, const u64 *a, const u64 *b, unsigned long n) {
    unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) fr_sub(a + i * 4, b + i * 4, out + i * 4);
}

extern "C" __global__ void mul_kernel(u64 *out, const u64 *a, const u64 *b, unsigned long n) {
    unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) fr_mul(a + i * 4, b + i * 4, out + i * 4);
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
}

impl std::fmt::Display for CudaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CudaError::Compile(e) => write!(f, "nvrtc compile error: {e:?}"),
            CudaError::Driver(e) => write!(f, "cuda driver error: {e:?}"),
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
    sum_reduce: CudaFunction,
    product_reduce: CudaFunction,
}

#[inline]
fn fr_to_limbs(f: Fr) -> [u64; LIMBS] {
    f.inner_limbs().0
}

#[inline]
fn limbs_to_fr(limbs: [u64; LIMBS]) -> Fr {
    Fr::from_bigint_unchecked(crate::Limbs(limbs))
}

fn flatten(values: &[Fr]) -> Vec<u64> {
    let mut out = Vec::with_capacity(values.len() * LIMBS);
    for &v in values {
        out.extend_from_slice(&fr_to_limbs(v));
    }
    out
}

fn unflatten(raw: &[u64]) -> Vec<Fr> {
    raw.chunks_exact(LIMBS)
        .map(|c| limbs_to_fr([c[0], c[1], c[2], c[3]]))
        .collect()
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
        Ok(Self {
            add: module.load_function("add_kernel")?,
            sub: module.load_function("sub_kernel")?,
            mul: module.load_function("mul_kernel")?,
            sum_reduce: module.load_function("sum_reduce")?,
            product_reduce: module.load_function("product_reduce")?,
            stream,
        })
    }

    fn map(&self, func: &CudaFunction, a: &[Fr], b: &[Fr]) -> Result<Vec<Fr>, CudaError> {
        assert_eq!(a.len(), b.len(), "map operands must have equal length");
        let n = a.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        let a_dev = self.stream.clone_htod(&flatten(a))?;
        let b_dev = self.stream.clone_htod(&flatten(b))?;
        let mut out_dev: CudaSlice<u64> = self.stream.alloc_zeros(n * LIMBS)?;

        let cfg = LaunchConfig {
            grid_dim: ((n as u32).div_ceil(BLOCK), 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_arg = n as u64;
        let mut launch = self.stream.launch_builder(func);
        let _ = launch
            .arg(&mut out_dev)
            .arg(&a_dev)
            .arg(&b_dev)
            .arg(&n_arg);
        // SAFETY: kernel reads n elements from a/b and writes n elements to out,
        // all of which are allocated with n * LIMBS u64s.
        let _ = unsafe { launch.launch(cfg) }?;

        let raw = self.stream.clone_dtoh(&out_dev)?;
        Ok(unflatten(&raw))
    }

    pub fn add(&self, a: &[Fr], b: &[Fr]) -> Result<Vec<Fr>, CudaError> {
        let f = self.add.clone();
        self.map(&f, a, b)
    }

    pub fn sub(&self, a: &[Fr], b: &[Fr]) -> Result<Vec<Fr>, CudaError> {
        let f = self.sub.clone();
        self.map(&f, a, b)
    }

    pub fn mul(&self, a: &[Fr], b: &[Fr]) -> Result<Vec<Fr>, CudaError> {
        let f = self.mul.clone();
        self.map(&f, a, b)
    }

    fn reduce(&self, sum: bool, values: &[Fr]) -> Result<Fr, CudaError> {
        use num_traits::{One, Zero};
        let identity = if sum { Fr::zero() } else { Fr::one() };
        if values.is_empty() {
            return Ok(identity);
        }

        let one_limbs = fr_to_limbs(Fr::one());
        let one_dev = self.stream.clone_htod(&one_limbs)?;

        let mut current = self.stream.clone_htod(&flatten(values))?;
        let mut len = values.len();

        while len > 1 {
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
            let _ = launch.arg(&mut out_dev).arg(&current).arg(&len_arg);
            if !sum {
                let _ = launch.arg(&one_dev);
            }
            // SAFETY: each block reads up to BLOCK elements from `current` (len total)
            // and writes one element per block to `out_dev` (blocks total). Shared
            // memory holds BLOCK field elements as configured above.
            let _ = unsafe { launch.launch(cfg) }?;

            current = out_dev;
            len = blocks as usize;
        }

        let raw = self.stream.clone_dtoh(&current)?;
        Ok(limbs_to_fr([raw[0], raw[1], raw[2], raw[3]]))
    }

    pub fn sum(&self, values: &[Fr]) -> Result<Fr, CudaError> {
        self.reduce(true, values)
    }

    pub fn product(&self, values: &[Fr]) -> Result<Fr, CudaError> {
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
            let got = ctx().add(&a, &b).unwrap();
            prop_assert_eq!(got, expected);
        }

        #[test]
        fn sub_matches_cpu(a in fr_vec_strategy(300)) {
            let b = a.iter().rev().copied().collect::<Vec<_>>();
            let expected: Vec<Fr> = a.iter().zip(&b).map(|(x, y)| *x - *y).collect();
            let got = ctx().sub(&a, &b).unwrap();
            prop_assert_eq!(got, expected);
        }

        #[test]
        fn mul_matches_cpu(a in fr_vec_strategy(300)) {
            let b = a.iter().rev().copied().collect::<Vec<_>>();
            let expected: Vec<Fr> = a.iter().zip(&b).map(|(x, y)| *x * *y).collect();
            let got = ctx().mul(&a, &b).unwrap();
            prop_assert_eq!(got, expected);
        }

        #[test]
        fn sum_matches_cpu(a in fr_vec_strategy(2000)) {
            let expected: Fr = a.iter().copied().sum();
            let got = ctx().sum(&a).unwrap();
            prop_assert_eq!(got, expected);
        }

        #[test]
        fn product_matches_cpu(a in fr_vec_strategy(2000)) {
            let expected: Fr = a.iter().copied().product();
            let got = ctx().product(&a).unwrap();
            prop_assert_eq!(got, expected);
        }
    }

    #[test]
    fn edge_cases() {
        let c = ctx();
        let zero = Fr::zero();
        let one = Fr::one();

        assert_eq!(c.add(&[], &[]).unwrap(), Vec::<Fr>::new());
        assert_eq!(c.sum(&[]).unwrap(), zero);
        assert_eq!(c.product(&[]).unwrap(), one);
        assert_eq!(c.sum(&[Fr::from_u64(42)]).unwrap(), Fr::from_u64(42));
        assert_eq!(c.product(&[Fr::from_u64(42)]).unwrap(), Fr::from_u64(42));

        let a = vec![zero, one, Fr::from_u64(7)];
        let b = vec![one, one, Fr::from_u64(6)];
        assert_eq!(
            c.add(&a, &b).unwrap(),
            vec![one, Fr::from_u64(2), Fr::from_u64(13)]
        );
        assert_eq!(
            c.mul(&a, &b).unwrap(),
            vec![zero, one, Fr::from_u64(42)]
        );
    }
}
