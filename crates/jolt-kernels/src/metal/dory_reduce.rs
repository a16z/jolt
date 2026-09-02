//! Device-resident prefix of the transparent Dory reduce-round loop.

#![expect(
    clippy::expect_used,
    reason = "an engaged Metal loop cannot safely fall back after transcript absorption"
)]

use ark_bn254::{Fr, G1Projective, G2Projective};
use ark_ec::scalar_mul::glv::GLVConfig;
use ark_ff::{AdditiveGroup, BigInteger, PrimeField, Zero};
use dory::backends::arkworks::{ArkFr, ArkG1, ArkG2, BN254 as InnerBN254};
use dory::messages::{FirstReduceMessage, SecondReduceMessage};
use dory::primitives::arithmetic::{
    DoryRoutines, PairingCurve, ResidentRoundHooks, ResidentRoundStart, ResidentRoundState,
};

use jolt_dory::{FastTail, JoltG1Routines, JoltG2Routines};

use super::buffers::DeviceBuffer;
use super::field::FR_U32_LIMBS;
use super::g1::JAC_U32S;
use super::g2::G2_JAC_U32S;
use super::runtime::{DetachedPass, KernelId, MetalContext};
use super::testing;

pub const ENV_MIN_LOOP_TERMS: &str = "JOLT_METAL_DORY_LOOP_MIN_TERMS";
pub const ENV_HANDOFF_TERMS: &str = "JOLT_METAL_DORY_HANDOFF_TERMS";

const DEFAULT_MIN_LOOP_TERMS: usize = 1 << 12;
const DEFAULT_HANDOFF_TERMS: usize = 1 << 9;
const GLV_WINDOWS: usize = 33;
const FOLD_WINDOWS: usize = 64;
const MSM_WINDOWS: usize = 32;
const MSM_BUCKETS: usize = 128;
const MSM_BINS: usize = MSM_WINDOWS * MSM_BUCKETS;
const MSM_SORT_MIN: usize = 1 << 13;
const MSM_OWNER_WIDTH: usize = 128;
const MSM_WORK_PER_TERM_LOG2: usize = 3;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.trim().parse().ok())
        .unwrap_or(default)
}

fn plan_device_rounds(n: usize) -> usize {
    if std::env::var("JOLT_METAL_DISABLE").is_ok_and(|v| !v.is_empty() && v != "0")
        || !n.is_power_of_two()
        || n < env_usize(ENV_MIN_LOOP_TERMS, DEFAULT_MIN_LOOP_TERMS)
    {
        return 0;
    }
    let handoff = env_usize(ENV_HANDOFF_TERMS, DEFAULT_HANDOFF_TERMS);
    let mut live = n;
    let mut rounds = 0;
    while live > handoff.max(1) {
        live /= 2;
        rounds += 1;
    }
    rounds
}

fn plan(n: usize) -> usize {
    if plan_device_rounds(n) > 0 {
        n.trailing_zeros() as usize
    } else {
        0
    }
}

fn wrapper_words<T>(values: &[T], words_per_value: usize) -> &[u32] {
    assert_eq!(std::mem::size_of::<T>(), words_per_value * 4);
    // SAFETY: dory's ArkG1/ArkG2 are repr(transparent) over the matching
    // arkworks projective structs; every projective limb bit pattern is u32.
    unsafe {
        std::slice::from_raw_parts(
            values.as_ptr().cast::<u32>(),
            values.len() * words_per_value,
        )
    }
}

#[cfg(test)]
fn bigint_limbs(big: &ark_ff::BigInt<4>) -> [u32; FR_U32_LIMBS] {
    let mut limbs = [0u32; FR_U32_LIMBS];
    for (i, word) in big.0.iter().enumerate() {
        limbs[2 * i] = *word as u32;
        limbs[2 * i + 1] = (*word >> 32) as u32;
    }
    limbs
}

fn fq2_words(value: &ark_bn254::Fq2) -> [u32; 2 * FR_U32_LIMBS] {
    let mut out = [0u32; 2 * FR_U32_LIMBS];
    // SAFETY: Fq2 is two contiguous Fq values, each four Montgomery u64s.
    let words =
        unsafe { std::slice::from_raw_parts(std::ptr::from_ref(value).cast::<u32>(), out.len()) };
    out.copy_from_slice(words);
    out
}

fn fq_words(value: &ark_bn254::Fq) -> [u32; FR_U32_LIMBS] {
    let mut out = [0u32; FR_U32_LIMBS];
    // SAFETY: Fq is four contiguous Montgomery u64 limbs.
    let words =
        unsafe { std::slice::from_raw_parts(std::ptr::from_ref(value).cast::<u32>(), out.len()) };
    out.copy_from_slice(words);
    out
}

fn fold_digits(scalar: &Fr) -> [i8; FOLD_WINDOWS] {
    let bytes = scalar.into_bigint().to_bytes_le();
    let mut digits = [0i8; FOLD_WINDOWS];
    let mut carry = 0i16;
    for (window, digit) in digits.iter_mut().enumerate() {
        let byte = i16::from(*bytes.get(window / 2).unwrap_or(&0));
        let nibble = if window % 2 == 0 {
            byte & 0xf
        } else {
            byte >> 4
        };
        let raw = nibble + carry;
        if raw >= 8 {
            *digit = i8::try_from(raw - 16).expect("signed nibble fits i8");
            carry = 1;
        } else {
            *digit = i8::try_from(raw).expect("nibble fits i8");
            carry = 0;
        }
    }
    debug_assert_eq!(carry, 0, "canonical BN254 scalar cannot carry out");
    digits
}

fn glv_fold_digits<C>(scalar: &Fr) -> [i8; 2 * GLV_WINDOWS]
where
    C: GLVConfig<ScalarField = Fr>,
{
    let ((sign1, k1), (sign2, k2)) = C::scalar_decomposition(*scalar);
    let mut digits = [0i8; 2 * GLV_WINDOWS];
    for (offset, (sign, half)) in [(0, (sign1, k1)), (GLV_WINDOWS, (sign2, k2))] {
        let full = fold_digits(&half);
        debug_assert!(full[GLV_WINDOWS..].iter().all(|&digit| digit == 0));
        for (slot, &digit) in digits[offset..offset + GLV_WINDOWS]
            .iter_mut()
            .zip(&full[..GLV_WINDOWS])
        {
            *slot = if sign { digit } else { -digit };
        }
    }
    digits
}

fn msm_digits(scalar: &ArkFr) -> [i8; MSM_WINDOWS] {
    let bytes = scalar.0.into_bigint().to_bytes_le();
    let mut digits = [0i8; MSM_WINDOWS];
    let mut carry = 0i16;
    for (window, digit) in digits.iter_mut().enumerate() {
        let raw = i16::from(*bytes.get(window).unwrap_or(&0)) + carry;
        if raw >= 128 {
            *digit = i8::try_from(raw - 256).expect("signed byte fits i8");
            carry = 1;
        } else {
            *digit = i8::try_from(raw).expect("byte fits i8");
            carry = 0;
        }
    }
    debug_assert_eq!(carry, 0, "canonical BN254 scalar cannot carry out");
    digits
}

fn msm_digit_matrix(scalars: &[ArkFr]) -> Vec<i8> {
    let len = scalars.len();
    let mut matrix = vec![0i8; MSM_WINDOWS * len];
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        let scalar_major: Vec<_> = scalars.par_iter().map(msm_digits).collect();
        matrix
            .par_chunks_mut(len)
            .enumerate()
            .for_each(|(window, row)| {
                for (digit, scalar_digits) in row.iter_mut().zip(&scalar_major) {
                    *digit = scalar_digits[window];
                }
            });
    }
    #[cfg(not(feature = "parallel"))]
    for (index, scalar) in scalars.iter().enumerate() {
        for (window, digit) in msm_digits(scalar).into_iter().enumerate() {
            matrix[window * len + index] = digit;
        }
    }
    matrix
}

fn msm_parts(len: usize) -> usize {
    match len {
        0..=2048 => 2,
        2049..=8192 => 8,
        8193..=32768 => 16,
        _ => 32,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MsmCurve {
    G1,
    G2,
}

struct SortedMsmJob<'a> {
    curve: MsmCurve,
    bases: &'a DeviceBuffer<'a>,
    base_offset: usize,
    len: usize,
    parts: usize,
    digits: DeviceBuffer<'static>,
    hist: DeviceBuffer<'static>,
    offsets: DeviceBuffer<'static>,
    cursors: DeviceBuffer<'static>,
    order: DeviceBuffer<'static>,
    bucket_sums: DeviceBuffer<'static>,
    partials: DeviceBuffer<'static>,
}

impl<'a> SortedMsmJob<'a> {
    fn new(
        context: &MetalContext,
        curve: MsmCurve,
        bases: &'a DeviceBuffer<'a>,
        base_offset: usize,
        scalars: &[ArkFr],
    ) -> Result<Self, super::MetalError> {
        let point_words = match curve {
            MsmCurve::G1 => JAC_U32S,
            MsmCurve::G2 => G2_JAC_U32S,
        };
        let digits = msm_digit_matrix(scalars);
        // SAFETY: i8 and u8 have identical layout; Metal interprets the
        // copied payload as signed bytes.
        let digit_bytes =
            unsafe { std::slice::from_raw_parts(digits.as_ptr().cast::<u8>(), digits.len()) };
        Ok(Self {
            curve,
            bases,
            base_offset,
            len: scalars.len(),
            parts: msm_parts(scalars.len()),
            digits: context.copy_bytes(digit_bytes)?,
            hist: context.copy_u32s(&[0u32; MSM_BINS])?,
            offsets: context.alloc_u32s(MSM_BINS + 1)?,
            cursors: context.alloc_u32s(MSM_BINS + 1)?,
            order: context.alloc_u32s(MSM_WINDOWS * scalars.len())?,
            bucket_sums: context.alloc_u32s(MSM_BINS * point_words)?,
            partials: context.alloc_u32s(MSM_WINDOWS * point_words)?,
        })
    }

    fn encode<'b>(&'b self, pass: &mut super::runtime::ComputePass<'_, 'b>) {
        let items = MSM_WINDOWS * self.len;
        let hist_groups = items.div_ceil(256).min(2048);
        let len = u32::try_from(self.len).expect("Dory MSM length fits u32");
        pass.dispatch(
            KernelId::DoryMsmHist,
            &[len],
            &[&self.digits, &self.hist],
            hist_groups * 256,
        );
        pass.dispatch(
            KernelId::DoryMsmOffsets,
            &[],
            &[&self.hist, &self.offsets, &self.cursors],
            256,
        );
        pass.dispatch(
            KernelId::DoryMsmScatter,
            &[len],
            &[&self.digits, &self.cursors, &self.order],
            items,
        );
        let buckets_per_group = MSM_OWNER_WIDTH / self.parts;
        let owner_groups = MSM_BINS.div_ceil(buckets_per_group);
        let owner = match self.curve {
            MsmCurve::G1 => KernelId::G1DoryMsmOwner,
            MsmCurve::G2 => KernelId::G2DoryMsmOwner,
        };
        pass.dispatch_width(
            owner,
            &[
                u32::try_from(self.parts).expect("MSM parts fit u32"),
                u32::try_from(self.base_offset).expect("MSM base offset fits u32"),
            ],
            &[self.bases, &self.order, &self.offsets, &self.bucket_sums],
            owner_groups * MSM_OWNER_WIDTH,
            MSM_OWNER_WIDTH,
        );
        let fold = match self.curve {
            MsmCurve::G1 => KernelId::G1DoryMsmWindowFold,
            MsmCurve::G2 => KernelId::G2DoryMsmWindowFold,
        };
        pass.dispatch_width(
            fold,
            &[],
            &[&self.bucket_sums, &self.partials],
            MSM_WINDOWS * MSM_BUCKETS,
            MSM_BUCKETS,
        );
    }

    fn finish_g1(&self) -> ArkG1 {
        assert_eq!(self.curve, MsmCurve::G1);
        let windows = self.partials.typed_slice::<G1Projective>(MSM_WINDOWS);
        let mut acc = G1Projective::zero();
        for (window, partial) in windows.iter().enumerate().rev() {
            if window != MSM_WINDOWS - 1 {
                for _ in 0..8 {
                    let _ = acc.double_in_place();
                }
            }
            acc += partial;
        }
        ArkG1(acc)
    }

    fn finish_g2(&self) -> ArkG2 {
        assert_eq!(self.curve, MsmCurve::G2);
        let windows = self.partials.typed_slice::<G2Projective>(MSM_WINDOWS);
        let mut acc = G2Projective::zero();
        for (window, partial) in windows.iter().enumerate().rev() {
            if window != MSM_WINDOWS - 1 {
                for _ in 0..8 {
                    let _ = acc.double_in_place();
                }
            }
            acc += partial;
        }
        ArkG2(acc)
    }
}

struct PendingSortedMsms<'a> {
    jobs: Vec<SortedMsmJob<'a>>,
    pass: DetachedPass,
}

impl<'a> PendingSortedMsms<'a> {
    fn start(context: &MetalContext, jobs: Vec<SortedMsmJob<'a>>) -> Self {
        let mut pass = context.begin_pass().expect("resident MSM pass");
        for job in &jobs {
            job.encode(&mut pass);
        }
        // SAFETY: `self.jobs` owns every temporary buffer until `finish`
        // waits; each bases buffer is owned by the borrowed ResidentLoop.
        let pass = unsafe { pass.commit().detach() };
        Self { jobs, pass }
    }

    fn finish_beta(self) -> (ArkG1, ArkG2) {
        self.pass.wait().expect("resident beta MSM kernels");
        (self.jobs[0].finish_g1(), self.jobs[1].finish_g2())
    }

    fn finish_beta_d2(self) -> ((ArkG1, ArkG2), (ArkG1, ArkG1)) {
        self.pass.wait().expect("resident beta+D2 MSM kernels");
        (
            (self.jobs[0].finish_g1(), self.jobs[1].finish_g2()),
            (self.jobs[2].finish_g1(), self.jobs[3].finish_g1()),
        )
    }

    fn finish_cross(self) -> ((ArkG1, ArkG1), (ArkG2, ArkG2)) {
        self.pass.wait().expect("resident cross MSM kernels");
        (
            (self.jobs[0].finish_g1(), self.jobs[1].finish_g1()),
            (self.jobs[2].finish_g2(), self.jobs[3].finish_g2()),
        )
    }
}

/// One host-argument G1 MSM (`Σ scalars[i]·bases[i]`) as a sorted-MSM pass —
/// the VMV preamble's `t_vec·v`, `Γ₁-prefix·v`, and `e1` MSMs. `None`
/// (undersized, dead device, or failed) falls back to the CPU path; served
/// results are group-equal, and every consumer (pairings, the serialized
/// `e1` message) normalizes before use, so proof bytes are unchanged.
/// Kill switch: `JOLT_METAL_MIN_TERMS_DORY_HOST_MSM=1000000000000` (or
/// `JOLT_METAL_DISABLE=1`) restores the host MSM bit-exactly.
pub(super) fn host_msm_g1(bases: &[G1Projective], scalars: &[ArkFr]) -> Option<ArkG1> {
    // The default gate engages at the sorted-MSM floor (2^13; measured 2.7×
    // over the host MSM already at 2^14): one term ≈ 2^3 gate work items.
    if bases.len() < MSM_SORT_MIN
        || !super::metal_gate("dory_host_msm", bases.len() << MSM_WORK_PER_TERM_LOG2)
    {
        return None;
    }
    let context = MetalContext::global().ok()?;
    let run = || -> Result<ArkG1, super::MetalError> {
        let bases_buffer = context.wrap_slice(wrapper_words(bases, JAC_U32S))?;
        testing::note_copied_buffers(u64::from(bases_buffer.was_copied()));
        let job = SortedMsmJob::new(context, MsmCurve::G1, &bases_buffer, 0, scalars)?;
        let mut pass = context.begin_pass()?;
        job.encode(&mut pass);
        pass.run()?;
        testing::note_device_round();
        Ok(job.finish_g1())
    };
    match tracing::info_span!("dory_host_msm_device", len = bases.len()).in_scope(run) {
        Ok(out) => Some(out),
        Err(error) => {
            tracing::warn!(slot = "dory_host_msm", %error, "device MSM failed; CPU fallback");
            None
        }
    }
}

fn g1_params(n: usize, p_offset: usize, q_offset: usize, scalar: &Fr) -> Vec<u32> {
    let mut params = Vec::with_capacity(3 + 2 * GLV_WINDOWS + FR_U32_LIMBS);
    params.extend([
        u32::try_from(n).expect("G1 fold length fits u32"),
        u32::try_from(p_offset).expect("G1 P offset fits u32"),
        u32::try_from(q_offset).expect("G1 Q offset fits u32"),
    ]);
    params
        .extend(glv_fold_digits::<ark_bn254::g1::Config>(scalar).map(|digit| digit as i32 as u32));
    params.extend(fq_words(
        &<ark_bn254::g1::Config as GLVConfig>::ENDO_COEFFS[0],
    ));
    params
}

fn g2_params(n: usize, p_offset: usize, q_offset: usize, scalar: &Fr) -> Vec<u32> {
    let mut params = Vec::with_capacity(3 + 2 * GLV_WINDOWS + 2 * FR_U32_LIMBS);
    params.extend([
        u32::try_from(n).expect("G2 fold length fits u32"),
        u32::try_from(p_offset).expect("G2 P offset fits u32"),
        u32::try_from(q_offset).expect("G2 Q offset fits u32"),
    ]);
    params
        .extend(glv_fold_digits::<ark_bn254::g2::Config>(scalar).map(|digit| digit as i32 as u32));
    params.extend(fq2_words(
        &<ark_bn254::g2::Config as GLVConfig>::ENDO_COEFFS[0],
    ));
    params
}

struct ResidentLoop {
    context: &'static MetalContext,
    v1: [DeviceBuffer<'static>; 2],
    v2: [DeviceBuffer<'static>; 2],
    g1: DeviceBuffer<'static>,
    g2: DeviceBuffer<'static>,
    active: usize,
    n: usize,
    handoff: usize,
    tail: Option<FastTail>,
}

impl ResidentLoop {
    fn start(
        v1: &[ArkG1],
        v2: &[ArkG2],
        g1: &[ArkG1],
        g2: &[ArkG2],
        _rounds: usize,
    ) -> Result<Self, super::MetalError> {
        let context = MetalContext::global()?;
        let n = v1.len();
        assert_eq!(v2.len(), n);
        assert_eq!(g1.len(), n);
        assert_eq!(g2.len(), n);
        Ok(Self {
            context,
            v1: [
                context.copy_u32s(wrapper_words(v1, JAC_U32S))?,
                context.alloc_u32s(n * JAC_U32S)?,
            ],
            v2: [
                context.copy_u32s(wrapper_words(v2, G2_JAC_U32S))?,
                context.alloc_u32s(n * G2_JAC_U32S)?,
            ],
            g1: context.copy_u32s(wrapper_words(g1, JAC_U32S))?,
            g2: context.copy_u32s(wrapper_words(g2, G2_JAC_U32S))?,
            active: 0,
            n,
            handoff: env_usize(ENV_HANDOFF_TERMS, DEFAULT_HANDOFF_TERMS).max(1),
            tail: None,
        })
    }

    fn v1(&self) -> &[ArkG1] {
        if let Some(tail) = &self.tail {
            return tail.vectors().0;
        }
        self.v1[self.active].typed_slice(self.n)
    }

    fn v2(&self) -> &[ArkG2] {
        if let Some(tail) = &self.tail {
            return tail.vectors().1;
        }
        self.v2[self.active].typed_slice(self.n)
    }

    fn g1(&self) -> &[ArkG1] {
        self.g1.typed_slice(self.n)
    }

    fn g2(&self) -> &[ArkG2] {
        self.g2.typed_slice(self.n)
    }

    fn apply(&mut self, g1_scalar: &Fr, g2_scalar: &Fr, fold_halves: bool) {
        if let Some(tail) = &mut self.tail {
            let g1_scalar = ArkFr(*g1_scalar);
            let g2_scalar = ArkFr(*g2_scalar);
            if fold_halves {
                tail.apply_second_challenge(&g1_scalar, &g2_scalar);
                self.n /= 2;
            } else {
                tail.apply_first_challenge(&g1_scalar, &g2_scalar);
            }
            return;
        }
        let out = 1 - self.active;
        let live = if fold_halves { self.n / 2 } else { self.n };
        let q_offset = if fold_halves { live } else { 0 };
        let g1_p = if fold_halves {
            &self.v1[self.active]
        } else {
            &self.g1
        };
        let g2_p = if fold_halves {
            &self.v2[self.active]
        } else {
            &self.g2
        };
        let mut pass = self.context.begin_pass().expect("resident Dory pass");
        pass.dispatch(
            KernelId::G1ProjectiveMulAdd,
            &g1_params(live, 0, q_offset, g1_scalar),
            &[g1_p, &self.v1[self.active], &self.v1[out]],
            live,
        );
        pass.dispatch(
            KernelId::G2ProjectiveMulAdd,
            &g2_params(live, 0, q_offset, g2_scalar),
            &[g2_p, &self.v2[self.active], &self.v2[out]],
            live,
        );
        pass.run().expect("resident Dory fold kernels");
        testing::note_device_round();
        self.active = out;
        if fold_halves {
            self.n = live;
            if self.n > 1 && self.n <= self.handoff {
                self.tail = Some(FastTail::new(
                    self.v1[self.active].typed_slice(self.n).to_vec(),
                    self.v2[self.active].typed_slice(self.n).to_vec(),
                    self.g1.typed_slice(self.n).to_vec(),
                    self.g2.typed_slice(self.n).to_vec(),
                ));
            }
        }
    }

    fn start_beta_msms<'a>(&'a self, s1: &[ArkFr], s2: &[ArkFr]) -> PendingSortedMsms<'a> {
        let jobs = vec![
            SortedMsmJob::new(self.context, MsmCurve::G1, &self.g1, 0, s2)
                .expect("resident G1 beta MSM buffers"),
            SortedMsmJob::new(self.context, MsmCurve::G2, &self.g2, 0, s1)
                .expect("resident G2 beta MSM buffers"),
        ];
        PendingSortedMsms::start(self.context, jobs)
    }

    /// The beta MSMs plus the round-0 D₂ shortcut's two Γ₁'-prefix MSMs
    /// (`d2_left`/`d2_right` are the `v2_scalars` halves), one detached pass.
    fn start_beta_and_d2_msms<'a>(
        &'a self,
        s1: &[ArkFr],
        s2: &[ArkFr],
        d2_left: &[ArkFr],
        d2_right: &[ArkFr],
    ) -> PendingSortedMsms<'a> {
        let jobs = vec![
            SortedMsmJob::new(self.context, MsmCurve::G1, &self.g1, 0, s2)
                .expect("resident G1 beta MSM buffers"),
            SortedMsmJob::new(self.context, MsmCurve::G2, &self.g2, 0, s1)
                .expect("resident G2 beta MSM buffers"),
            SortedMsmJob::new(self.context, MsmCurve::G1, &self.g1, 0, d2_left)
                .expect("resident G1 D2-left MSM buffers"),
            SortedMsmJob::new(self.context, MsmCurve::G1, &self.g1, 0, d2_right)
                .expect("resident G1 D2-right MSM buffers"),
        ];
        PendingSortedMsms::start(self.context, jobs)
    }

    fn start_cross_msms<'a>(
        &'a self,
        s1_l: &[ArkFr],
        s1_r: &[ArkFr],
        s2_l: &[ArkFr],
        s2_r: &[ArkFr],
    ) -> PendingSortedMsms<'a> {
        let n2 = self.n / 2;
        let jobs = vec![
            SortedMsmJob::new(self.context, MsmCurve::G1, &self.v1[self.active], 0, s2_r)
                .expect("resident G1 plus MSM buffers"),
            SortedMsmJob::new(self.context, MsmCurve::G1, &self.v1[self.active], n2, s2_l)
                .expect("resident G1 minus MSM buffers"),
            SortedMsmJob::new(self.context, MsmCurve::G2, &self.v2[self.active], n2, s1_l)
                .expect("resident G2 plus MSM buffers"),
            SortedMsmJob::new(self.context, MsmCurve::G2, &self.v2[self.active], 0, s1_r)
                .expect("resident G2 minus MSM buffers"),
        ];
        PendingSortedMsms::start(self.context, jobs)
    }
}

#[cfg(feature = "parallel")]
fn join<A, B, RA, RB>(a: A, b: B) -> (RA, RB)
where
    A: FnOnce() -> RA + Send,
    B: FnOnce() -> RB + Send,
    RA: Send,
    RB: Send,
{
    rayon::join(a, b)
}

#[cfg(not(feature = "parallel"))]
fn join<A, B, RA, RB>(a: A, b: B) -> (RA, RB)
where
    A: FnOnce() -> RA,
    B: FnOnce() -> RB,
{
    (a(), b())
}

fn downcast(state: &mut ResidentRoundState) -> &mut ResidentLoop {
    state
        .downcast_mut::<ResidentLoop>()
        .expect("resident Dory state type")
}

/// Merged-dispatch toggle for the reduce rounds' multi-pairings
/// (`JOLT_MILLER_MERGE_DISPATCH=0` restores one hook call per pairing —
/// the W3 shape). Merging a message's calls into one device dispatch
/// (`multi_pair_device_batch`) exposes their thread SUM — the mid-ladder
/// singles (4096/2048 pairs) sit under the device's saturation knee — and
/// extends device service to rounds whose singles fall below the 2048-pair
/// gate. Read per message; the benches A/B it in-process.
fn merge_dispatch_enabled() -> bool {
    !std::env::var("JOLT_MILLER_MERGE_DISPATCH").is_ok_and(|value| value.trim() == "0")
}

/// Round-0 D₂ shortcut toggle (`JOLT_DORY_R0_D2_MSM=0` restores the
/// four-call pairing shape). With `v2 = Γ₂fin·scalars` (true only before the
/// first challenge), D₂ halves collapse from n/2-pair multi-pairings to one
/// Γ₁'-prefix MSM plus a single pairing each — the host arm's `compute_d2`
/// identity `Π e(Γ₁'ᵢ, sᵢ·Γ₂fin) = e(Σ sᵢ·Γ₁'ᵢ, Γ₂fin)`. The GT value is
/// equal by bilinearity and GT bytes are value-unique, so the transcript is
/// unchanged; round 0 carries the largest pair count of the reduce, and this
/// removes half of it.
fn d2_msm_enabled() -> bool {
    !std::env::var("JOLT_DORY_R0_D2_MSM").is_ok_and(|value| value.trim() == "0")
}

fn start(v1: &[ArkG1], v2: &[ArkG2], g1: &[ArkG1], g2: &[ArkG2]) -> Option<ResidentRoundStart> {
    let rounds = plan(v1.len());
    if rounds == 0 {
        return None;
    }
    match ResidentLoop::start(v1, v2, g1, g2, rounds) {
        Ok(state) => Some(ResidentRoundStart {
            state: Box::new(state),
            rounds,
        }),
        Err(error) => {
            tracing::warn!(%error, "resident Dory start failed; using host loop");
            None
        }
    }
}

fn first_message(
    state: &mut ResidentRoundState,
    s1: &[ArkFr],
    s2: &[ArkFr],
    v2_scalars: Option<&[ArkFr]>,
) -> FirstReduceMessage<ArkG1, ArkG2, dory::backends::arkworks::ArkGT> {
    let state = downcast(state);
    if let Some(tail) = &state.tail {
        return tail.compute_first_message(s1, s2);
    }
    let n = state.n;
    let n2 = n / 2;
    let (v1_l, v1_r) = state.v1().split_at(n2);
    let (v2_l, v2_r) = state.v2().split_at(n2);
    let g1 = &state.g1()[..n2];
    let g2 = &state.g2()[..n2];

    if let Some(scalars) = v2_scalars.filter(|_| d2_msm_enabled()) {
        assert_eq!(scalars.len(), n, "v2_scalars must match the round width");
        let g2_fin = state.g2()[0];
        let (s_l, s_r) = scalars.split_at(n2);
        let d1_pairings = || {
            if merge_dispatch_enabled() {
                if let Some(mut gts) =
                    super::miller::multi_pair_device_batch(&[(v1_l, g2), (v1_r, g2)])
                {
                    let d1_right = gts.pop().expect("two GTs");
                    let d1_left = gts.pop().expect("two GTs");
                    return (d1_left, d1_right);
                }
            }
            join(
                || InnerBN254::multi_pair_g2_setup(v1_l, g2),
                || InnerBN254::multi_pair_g2_setup(v1_r, g2),
            )
        };
        let ((d1_left, d1_right), ((e1_beta, e2_beta), (d2_sum_left, d2_sum_right))) =
            if n >= MSM_SORT_MIN {
                let pending = state.start_beta_and_d2_msms(s1, s2, s_l, s_r);
                let d1 = d1_pairings();
                (d1, pending.finish_beta_d2())
            } else {
                let g1_full = state.g1();
                let g2_full = state.g2();
                join(d1_pairings, || {
                    join(
                        || {
                            join(
                                || JoltG1Routines::msm(g1_full, s2),
                                || JoltG2Routines::msm(g2_full, s1),
                            )
                        },
                        || {
                            join(
                                || JoltG1Routines::msm(g1, s_l),
                                || JoltG1Routines::msm(g1, s_r),
                            )
                        },
                    )
                })
            };
        let (d2_left, d2_right) = join(
            || InnerBN254::pair(&d2_sum_left, &g2_fin),
            || InnerBN254::pair(&d2_sum_right, &g2_fin),
        );
        return FirstReduceMessage {
            d1_left,
            d1_right,
            d2_left,
            d2_right,
            e1_beta,
            e2_beta,
        };
    }

    let pairings = || {
        if merge_dispatch_enabled() {
            if let Some(mut gts) = super::miller::multi_pair_device_batch(&[
                (v1_l, g2),
                (v1_r, g2),
                (g1, v2_l),
                (g1, v2_r),
            ]) {
                let d2_right = gts.pop().expect("four GTs");
                let d2_left = gts.pop().expect("four GTs");
                let d1_right = gts.pop().expect("four GTs");
                let d1_left = gts.pop().expect("four GTs");
                return ((d1_left, d1_right), (d2_left, d2_right));
            }
        }
        join(
            || {
                join(
                    || InnerBN254::multi_pair_g2_setup(v1_l, g2),
                    || InnerBN254::multi_pair_g2_setup(v1_r, g2),
                )
            },
            || {
                join(
                    || InnerBN254::multi_pair_g1_setup(g1, v2_l),
                    || InnerBN254::multi_pair_g1_setup(g1, v2_r),
                )
            },
        )
    };
    let (pairing_messages, (e1_beta, e2_beta)) = if n >= MSM_SORT_MIN {
        let pending = state.start_beta_msms(s1, s2);
        let pairing_messages = pairings();
        (pairing_messages, pending.finish_beta())
    } else {
        let g1_full = state.g1();
        let g2_full = state.g2();
        join(pairings, || {
            join(
                || JoltG1Routines::msm(g1_full, s2),
                || JoltG2Routines::msm(g2_full, s1),
            )
        })
    };
    let ((d1_left, d1_right), (d2_left, d2_right)) = pairing_messages;
    FirstReduceMessage {
        d1_left,
        d1_right,
        d2_left,
        d2_right,
        e1_beta,
        e2_beta,
    }
}

fn apply_first(state: &mut ResidentRoundState, beta: &ArkFr, beta_inv: &ArkFr) {
    downcast(state).apply(&beta.0, &beta_inv.0, false);
}

fn second_message(
    state: &mut ResidentRoundState,
    s1: &[ArkFr],
    s2: &[ArkFr],
) -> SecondReduceMessage<ArkG1, ArkG2, dory::backends::arkworks::ArkGT> {
    let state = downcast(state);
    if let Some(tail) = &state.tail {
        return tail.compute_second_message(s1, s2);
    }
    let n2 = state.n / 2;
    let (v1_l, v1_r) = state.v1().split_at(n2);
    let (v2_l, v2_r) = state.v2().split_at(n2);
    let (s1_l, s1_r) = s1.split_at(n2);
    let (s2_l, s2_r) = s2.split_at(n2);
    let pairings = || {
        if merge_dispatch_enabled() {
            if let Some(mut gts) =
                super::miller::multi_pair_device_batch(&[(v1_l, v2_r), (v1_r, v2_l)])
            {
                let c_minus = gts.pop().expect("two GTs");
                let c_plus = gts.pop().expect("two GTs");
                return (c_plus, c_minus);
            }
        }
        join(
            || InnerBN254::multi_pair(v1_l, v2_r),
            || InnerBN254::multi_pair(v1_r, v2_l),
        )
    };
    let ((c_plus, c_minus), ((e1_plus, e1_minus), (e2_plus, e2_minus))) = if n2 >= MSM_SORT_MIN {
        let pending = state.start_cross_msms(s1_l, s1_r, s2_l, s2_r);
        let pairing_messages = pairings();
        (pairing_messages, pending.finish_cross())
    } else {
        join(pairings, || {
            join(
                || {
                    join(
                        || JoltG1Routines::msm(v1_l, s2_r),
                        || JoltG1Routines::msm(v1_r, s2_l),
                    )
                },
                || {
                    join(
                        || JoltG2Routines::msm(v2_r, s1_l),
                        || JoltG2Routines::msm(v2_l, s1_r),
                    )
                },
            )
        })
    };
    SecondReduceMessage {
        c_plus,
        c_minus,
        e1_plus,
        e1_minus,
        e2_plus,
        e2_minus,
    }
}

fn apply_second(state: &mut ResidentRoundState, alpha: &ArkFr, alpha_inv: &ArkFr) {
    downcast(state).apply(&alpha.0, &alpha_inv.0, true);
}

fn finish(mut state: ResidentRoundState) -> (Vec<ArkG1>, Vec<ArkG2>) {
    let state = state
        .downcast_mut::<ResidentLoop>()
        .expect("resident Dory state type");
    if let Some(tail) = state.tail.take() {
        return tail.into_vectors();
    }
    (state.v1().to_vec(), state.v2().to_vec())
}

pub(super) fn hooks() -> ResidentRoundHooks<InnerBN254> {
    ResidentRoundHooks {
        plan,
        start,
        first_message,
        apply_first,
        second_message,
        apply_second,
        finish,
    }
}

#[cfg(test)]
mod tests {
    use ark_bn254::{G1Projective, G2Projective};
    use ark_ff::{Field, UniformRand};
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    use super::super::testing::{device_probe_count, gpu_lock};
    use super::*;

    #[test]
    fn resident_projective_folds_match_arkworks() {
        let _lock = gpu_lock();
        let mut rng = ChaCha20Rng::seed_from_u64(0xd0_72);
        let n = 64;
        let g1: Vec<ArkG1> = (0..n)
            .map(|_| ArkG1(G1Projective::rand(&mut rng)))
            .collect();
        let g2: Vec<ArkG2> = (0..n)
            .map(|_| ArkG2(G2Projective::rand(&mut rng)))
            .collect();
        let beta = ArkFr(Fr::rand(&mut rng));
        let beta_inv = ArkFr(beta.0.inverse().expect("random scalar is nonzero"));
        let alpha = ArkFr(Fr::rand(&mut rng));
        let alpha_inv = ArkFr(alpha.0.inverse().expect("random scalar is nonzero"));
        let mut v1: Vec<ArkG1> = (0..n)
            .map(|_| ArkG1(G1Projective::rand(&mut rng)))
            .collect();
        let mut v2: Vec<ArkG2> = (0..n)
            .map(|_| ArkG2(G2Projective::rand(&mut rng)))
            .collect();

        // Force identity, inverse, and doubling branches in both challenge
        // additions; random projective representatives rarely hit them.
        v1[0] = ArkG1(-g1[0].0 * beta.0);
        v1[1] = ArkG1(g1[1].0 * beta.0);
        v2[0] = ArkG2(-g2[0].0 * beta_inv.0);
        v2[1] = ArkG2(g2[1].0 * beta_inv.0);
        let left_g1 = G1Projective::rand(&mut rng);
        v1[2] = ArkG1(left_g1 - g1[2].0 * beta.0);
        v1[n / 2 + 2] = ArkG1(-left_g1 * alpha.0 - g1[n / 2 + 2].0 * beta.0);
        let left_g2 = G2Projective::rand(&mut rng);
        v2[2] = ArkG2(left_g2 - g2[2].0 * beta_inv.0);
        v2[n / 2 + 2] = ArkG2(-left_g2 * alpha_inv.0 - g2[n / 2 + 2].0 * beta_inv.0);

        let mut state = ResidentLoop::start(&v1, &v2, &g1, &g2, 1).expect("resident loop starts");
        state.apply(&beta.0, &beta_inv.0, false);

        let expected_g1: Vec<ArkG1> = v1
            .iter()
            .zip(&g1)
            .map(|(v, g)| ArkG1(v.0 + g.0 * beta.0))
            .collect();
        let expected_g2: Vec<ArkG2> = v2
            .iter()
            .zip(&g2)
            .map(|(v, g)| ArkG2(v.0 + g.0 * beta_inv.0))
            .collect();
        assert_eq!(state.v1(), expected_g1);
        assert_eq!(state.v2(), expected_g2);

        state.apply(&alpha.0, &alpha_inv.0, true);
        let expected_g1: Vec<ArkG1> = expected_g1[..n / 2]
            .iter()
            .zip(&expected_g1[n / 2..])
            .map(|(left, right)| ArkG1(left.0 * alpha.0 + right.0))
            .collect();
        let expected_g2: Vec<ArkG2> = expected_g2[..n / 2]
            .iter()
            .zip(&expected_g2[n / 2..])
            .map(|(left, right)| ArkG2(left.0 * alpha_inv.0 + right.0))
            .collect();
        assert_eq!(state.v1(), expected_g1);
        assert_eq!(state.v2(), expected_g2);
    }

    /// Merged-dispatch reduce messages = per-call messages = the CPU trait
    /// path (no hook installed in unit tests, so the unmerged arm IS the
    /// CPU reference). Every GT and every MSM leg must be exact — these
    /// values are absorbed into the transcript.
    #[test]
    fn reduce_messages_merged_match_unmerged() {
        let _lock = gpu_lock();
        std::env::set_var("JOLT_METAL_MIN_TERMS_MILLER_FLY", "1");
        let mut rng = ChaCha20Rng::seed_from_u64(0xd0_73);
        let n = 2048usize;
        let mut v1: Vec<ArkG1> = (0..n)
            .map(|_| ArkG1(G1Projective::rand(&mut rng)))
            .collect();
        let mut v2: Vec<ArkG2> = (0..n)
            .map(|_| ArkG2(G2Projective::rand(&mut rng)))
            .collect();
        let g1: Vec<ArkG1> = (0..n)
            .map(|_| ArkG1(G1Projective::rand(&mut rng)))
            .collect();
        let g2: Vec<ArkG2> = (0..n)
            .map(|_| ArkG2(G2Projective::rand(&mut rng)))
            .collect();
        v1[3] = ArkG1(G1Projective::zero());
        v2[n / 2 + 5] = ArkG2(G2Projective::zero());
        let s1: Vec<ArkFr> = (0..n).map(|_| ArkFr(Fr::rand(&mut rng))).collect();
        let s2: Vec<ArkFr> = (0..n).map(|_| ArkFr(Fr::rand(&mut rng))).collect();

        let message_pair = |merge: &str| {
            std::env::set_var("JOLT_MILLER_MERGE_DISPATCH", merge);
            let mut state: ResidentRoundState =
                Box::new(ResidentLoop::start(&v1, &v2, &g1, &g2, 1).expect("resident loop starts"));
            let first = first_message(&mut state, &s1, &s2, None);
            let second = second_message(&mut state, &s1, &s2);
            (first, second)
        };
        let (first_merged, second_merged) = message_pair("1");
        let (first_single, second_single) = message_pair("0");
        std::env::remove_var("JOLT_MILLER_MERGE_DISPATCH");

        assert_eq!(first_merged.d1_left.0, first_single.d1_left.0);
        assert_eq!(first_merged.d1_right.0, first_single.d1_right.0);
        assert_eq!(first_merged.d2_left.0, first_single.d2_left.0);
        assert_eq!(first_merged.d2_right.0, first_single.d2_right.0);
        assert_eq!(first_merged.e1_beta.0, first_single.e1_beta.0);
        assert_eq!(first_merged.e2_beta.0, first_single.e2_beta.0);
        assert_eq!(second_merged.c_plus.0, second_single.c_plus.0);
        assert_eq!(second_merged.c_minus.0, second_single.c_minus.0);
        assert_eq!(second_merged.e1_plus.0, second_single.e1_plus.0);
        assert_eq!(second_merged.e1_minus.0, second_single.e1_minus.0);
        assert_eq!(second_merged.e2_plus.0, second_single.e2_plus.0);
        assert_eq!(second_merged.e2_minus.0, second_single.e2_minus.0);
    }

    /// Round-0 D₂-shortcut first message = the four-pairing first message =
    /// the CPU trait path, field by field, on a `v2 = Γ₂fin·scalars` round
    /// (the only round the shortcut serves). Identity-bearing scalars
    /// included: a zero v2-scalar makes an identity v2 element, and a zero
    /// MSM digit column exercises the sort-owner path.
    #[test]
    fn reduce_first_message_d2_shortcut_matches_pairing() {
        let _lock = gpu_lock();
        std::env::set_var("JOLT_METAL_MIN_TERMS_MILLER_FLY", "1");
        // 2048 exercises the CPU-MSM arm, MSM_SORT_MIN the device-MSM arm.
        for n in [2048usize, MSM_SORT_MIN] {
            reduce_first_message_d2_shortcut_case(n);
        }
        std::env::remove_var("JOLT_METAL_MIN_TERMS_MILLER_FLY");
    }

    fn reduce_first_message_d2_shortcut_case(n: usize) {
        let mut rng = ChaCha20Rng::seed_from_u64(0xd0_74);
        let v1: Vec<ArkG1> = (0..n)
            .map(|_| ArkG1(G1Projective::rand(&mut rng)))
            .collect();
        let g1: Vec<ArkG1> = (0..n)
            .map(|_| ArkG1(G1Projective::rand(&mut rng)))
            .collect();
        let g2: Vec<ArkG2> = (0..n)
            .map(|_| ArkG2(G2Projective::rand(&mut rng)))
            .collect();
        let mut scalars: Vec<ArkFr> = (0..n).map(|_| ArkFr(Fr::rand(&mut rng))).collect();
        scalars[7] = ArkFr(Fr::from(0u64));
        scalars[n / 2 + 3] = ArkFr(-Fr::from(1u64));
        let g2_fin = g2[0];
        let v2: Vec<ArkG2> = scalars.iter().map(|s| ArkG2(g2_fin.0 * s.0)).collect();
        let s1: Vec<ArkFr> = (0..n).map(|_| ArkFr(Fr::rand(&mut rng))).collect();
        let s2: Vec<ArkFr> = (0..n).map(|_| ArkFr(Fr::rand(&mut rng))).collect();

        let message = |v2_scalars: Option<&[ArkFr]>| {
            let mut state: ResidentRoundState =
                Box::new(ResidentLoop::start(&v1, &v2, &g1, &g2, 1).expect("resident loop starts"));
            first_message(&mut state, &s1, &s2, v2_scalars)
        };
        let shortcut = message(Some(&scalars));
        let paired = message(None);

        // CPU trait reference for the D2 halves (the transcript values).
        let n2 = n / 2;
        let reference_left = InnerBN254::multi_pair_g1_setup(&g1[..n2], &v2[..n2]);
        let reference_right = InnerBN254::multi_pair_g1_setup(&g1[..n2], &v2[n2..]);

        assert_eq!(shortcut.d1_left.0, paired.d1_left.0);
        assert_eq!(shortcut.d1_right.0, paired.d1_right.0);
        assert_eq!(shortcut.d2_left.0, paired.d2_left.0);
        assert_eq!(shortcut.d2_right.0, paired.d2_right.0);
        assert_eq!(shortcut.d2_left.0, reference_left.0);
        assert_eq!(shortcut.d2_right.0, reference_right.0);
        assert_eq!(shortcut.e1_beta.0, paired.e1_beta.0);
        assert_eq!(shortcut.e2_beta.0, paired.e2_beta.0);
    }

    #[test]
    fn fold_digits_use_canonical_scalar() {
        let scalar = Fr::from(0x1234_5678_9abc_def0u64);
        let limbs = bigint_limbs(&scalar.into_bigint());
        assert_eq!(limbs[0], 0x9abc_def0);
        assert_eq!(limbs[1], 0x1234_5678);

        let mut reconstructed = Fr::from(0u64);
        let mut weight = Fr::from(1u64);
        for digit in fold_digits(&scalar) {
            let term = weight * Fr::from(u64::from(digit.unsigned_abs()));
            reconstructed = if digit < 0 {
                reconstructed - term
            } else {
                reconstructed + term
            };
            weight *= Fr::from(16u64);
        }
        assert_eq!(reconstructed, scalar);
    }

    #[test]
    fn sort_owner_msms_match_arkworks_at_engagement_floor() {
        let _lock = gpu_lock();
        let mut rng = ChaCha20Rng::seed_from_u64(0x34_d0_72);
        let n = MSM_SORT_MIN;
        let g1: Vec<ArkG1> = (0..n)
            .map(|_| ArkG1(G1Projective::rand(&mut rng)))
            .collect();
        let g2: Vec<ArkG2> = (0..n)
            .map(|_| ArkG2(G2Projective::rand(&mut rng)))
            .collect();
        let mut scalars: Vec<ArkFr> = (0..n).map(|_| ArkFr(Fr::rand(&mut rng))).collect();
        scalars[0] = ArkFr(Fr::from(0u64));
        scalars[1] = ArkFr(-Fr::from(1u64));

        let context = MetalContext::global().expect("Metal context");
        let g1_buffer = context
            .copy_u32s(wrapper_words(&g1, JAC_U32S))
            .expect("G1 buffer");
        let g2_buffer = context
            .copy_u32s(wrapper_words(&g2, G2_JAC_U32S))
            .expect("G2 buffer");
        let g1_job =
            SortedMsmJob::new(context, MsmCurve::G1, &g1_buffer, 0, &scalars).expect("G1 job");
        let g2_job =
            SortedMsmJob::new(context, MsmCurve::G2, &g2_buffer, 0, &scalars).expect("G2 job");
        let mut pass = context.begin_pass().expect("MSM pass");
        g1_job.encode(&mut pass);
        g2_job.encode(&mut pass);
        pass.run().expect("MSM kernels");

        let (expected_g1, expected_g2) = join(
            || JoltG1Routines::msm(&g1, &scalars),
            || JoltG2Routines::msm(&g2, &scalars),
        );
        assert_eq!(g1_job.finish_g1(), expected_g1);
        assert_eq!(g2_job.finish_g2(), expected_g2);
    }

    /// `host_msm_g1` (the VMV preamble hook's device MSM over host slices)
    /// against the CPU MSM on identity-planted bases — the preamble's
    /// `padded_row_commitments` carry identity padding whenever nu < sigma —
    /// plus zero/minus-one scalars, and the undersized decline.
    #[test]
    fn host_msm_matches_cpu_with_identity_bases() {
        let _lock = gpu_lock();
        std::env::remove_var("JOLT_METAL_DISABLE");
        std::env::set_var("JOLT_METAL_MIN_TERMS", "0");
        let mut rng = ChaCha20Rng::seed_from_u64(19);

        let n = MSM_SORT_MIN;
        let mut bases: Vec<G1Projective> = (0..n).map(|_| G1Projective::rand(&mut rng)).collect();
        // The whole upper half identity — the nu < sigma padding shape.
        for base in bases.iter_mut().skip(n / 2) {
            *base = G1Projective::zero();
        }
        bases[3] = G1Projective::zero();
        let mut scalars: Vec<ArkFr> = (0..n).map(|_| ArkFr(Fr::rand(&mut rng))).collect();
        scalars[0] = ArkFr(Fr::from(0u64));
        scalars[1] = ArkFr(-Fr::from(1u64));

        let ark_bases: Vec<ArkG1> = bases.iter().map(|base| ArkG1(*base)).collect();
        let expected = JoltG1Routines::msm(&ark_bases, &scalars);

        let probes_before = device_probe_count();
        let served = host_msm_g1(&bases, &scalars).expect("device MSM at the engagement floor");
        assert_eq!(device_probe_count() - probes_before, 1, "one device pass");
        assert_eq!(served, expected);

        assert!(
            host_msm_g1(&bases[..MSM_SORT_MIN - 1], &scalars[..MSM_SORT_MIN - 1]).is_none(),
            "undersized MSMs decline to the CPU path"
        );
    }
}
