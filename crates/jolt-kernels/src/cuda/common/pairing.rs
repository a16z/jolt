use std::cell::RefCell;

use ark_bn254::{Fq, Fq2};
use ark_ec::bn::BnConfig;
use ark_ec::short_weierstrass::SWCurveConfig;
use ark_ff::{AdditiveGroup, Field, One};
use cudarc::driver::{CudaSlice, PushKernelArg};

use super::context::CudaKernelContext;
use super::error::CudaError;
use super::msm::FQ_LIMBS;

pub const FQ12_LIMBS: usize = 12 * FQ_LIMBS;

pub const FQ2_LIMBS: usize = 2 * FQ_LIMBS;

pub const FQ6_LIMBS: usize = 6 * FQ_LIMBS;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TowerProbe {
    Fq6Mul,
    Fq6MulBy01,
    Fq6MulByFq2,
    Fq6MulByNonresidue,
    Fq12Mul,
    Fq12Sqr,
    Fq12MulBy034,
    Ell,
}

impl TowerProbe {
    const fn widths(self) -> (usize, usize) {
        match self {
            Self::Fq6Mul => (2 * FQ6_LIMBS, FQ6_LIMBS),
            Self::Fq6MulBy01 => (FQ6_LIMBS + 2 * FQ2_LIMBS, FQ6_LIMBS),
            Self::Fq6MulByFq2 => (FQ6_LIMBS + FQ2_LIMBS, FQ6_LIMBS),
            Self::Fq6MulByNonresidue => (FQ6_LIMBS, FQ6_LIMBS),
            Self::Fq12Mul => (2 * FQ12_LIMBS, FQ12_LIMBS),
            Self::Fq12Sqr => (FQ12_LIMBS, FQ12_LIMBS),
            Self::Fq12MulBy034 => (FQ12_LIMBS + 3 * FQ2_LIMBS, FQ12_LIMBS),
            Self::Ell => (FQ12_LIMBS + 3 * FQ2_LIMBS + 2 * FQ_LIMBS, FQ12_LIMBS),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum G2StepProbe {
    DoubleStep,
    AddStep,
    MulByChar,
}

impl G2StepProbe {
    const fn widths(self) -> (usize, usize) {
        match self {
            Self::DoubleStep => (3 * FQ2_LIMBS, 12 * FQ2_LIMBS),
            Self::AddStep => (5 * FQ2_LIMBS, 12 * FQ2_LIMBS),
            Self::MulByChar => (2 * FQ2_LIMBS, 4 * FQ2_LIMBS),
        }
    }
}

const PC_WORDS: usize = 28;

struct Constants {
    values: CudaSlice<u64>,
}

thread_local! {
    static CONSTANTS: RefCell<Vec<Option<Constants>>> = const { RefCell::new(Vec::new()) };
}

fn fq_words(value: &Fq) -> [u64; FQ_LIMBS] {
    value.0 .0
}

fn push_fq2(out: &mut Vec<u64>, value: &Fq2) {
    out.extend_from_slice(&fq_words(&value.c0));
    out.extend_from_slice(&fq_words(&value.c1));
}

#[expect(
    clippy::expect_used,
    reason = "two is invertible in Fq; the alternative is a Result on an infallible constant"
)]
fn constant_words() -> Vec<u64> {
    let mut out = Vec::with_capacity(PC_WORDS);
    push_fq2(&mut out, &<ark_bn254::Config as BnConfig>::TWIST_MUL_BY_Q_X);
    push_fq2(&mut out, &<ark_bn254::Config as BnConfig>::TWIST_MUL_BY_Q_Y);
    push_fq2(&mut out, &<ark_bn254::g2::Config as SWCurveConfig>::COEFF_B);
    let two_inv = Fq::one().double().inverse().expect("two is invertible");
    out.extend_from_slice(&fq_words(&two_inv));
    out
}

impl CudaKernelContext {
    fn with_pairing_constants<R>(
        &self,
        act: impl FnOnce(&Constants) -> Result<R, CudaError>,
    ) -> Result<R, CudaError> {
        CONSTANTS.with_borrow_mut(|cached| {
            let ordinal = self.ordinal();
            if cached.len() <= ordinal {
                cached.resize_with(ordinal + 1, || None);
            }
            let slot = cached
                .get_mut(ordinal)
                .ok_or(CudaError::InvariantViolation {
                    reason: "the per-device pairing constant cache was just sized",
                })?;
            if slot.is_none() {
                let words = constant_words();
                if words.len() != PC_WORDS {
                    return Err(CudaError::LengthMismatch {
                        expected: PC_WORDS,
                        got: words.len(),
                    });
                }
                *slot = Some(Constants {
                    values: self.upload_raw_u64(&words)?,
                });
            }
            match slot.as_ref() {
                Some(constants) => act(constants),
                None => Err(CudaError::InvariantViolation {
                    reason: "the pairing constants did not initialise",
                }),
            }
        })
    }

    pub fn tower_probe(&self, op: TowerProbe, input: &[u64]) -> Result<Vec<u64>, CudaError> {
        let (in_width, out_width) = op.widths();
        if in_width == 0 || !input.len().is_multiple_of(in_width) {
            return Err(CudaError::LengthMismatch {
                expected: in_width,
                got: input.len(),
            });
        }
        let elements = input.len() / in_width;
        if elements == 0 {
            return Ok(Vec::new());
        }
        let count = Self::count_of(elements)?;
        let device_in = self.upload_u64_slice(input)?;
        let mut device_out = self.alloc_u64(elements * out_width)?;
        let function = match op {
            TowerProbe::Fq6Mul => &self.fq6_mul_probe,
            TowerProbe::Fq6MulBy01 => &self.fq6_mul_by_01_probe,
            TowerProbe::Fq6MulByFq2 => &self.fq6_mul_by_fq2_probe,
            TowerProbe::Fq6MulByNonresidue => &self.fq6_mul_by_nonresidue_probe,
            TowerProbe::Fq12Mul => &self.fq12_mul_probe,
            TowerProbe::Fq12Sqr => &self.fq12_sqr_probe,
            TowerProbe::Fq12MulBy034 => &self.fq12_mul_by_034_probe,
            TowerProbe::Ell => &self.ell_probe,
        };
        let mut builder = self.stream().launch_builder(function);
        let _ = builder.arg(&device_in);
        let _ = builder.arg(&mut device_out);
        let _ = builder.arg(&count);
        // SAFETY: thread `i < count` reads exactly the `in_width` u64s at
        // `input[i * in_width]` and writes exactly the `out_width` u64s at
        // `output[i * out_width]`. The upload holds `count * in_width` u64s and
        // the allocation `count * out_width`, both sized above from the same
        // `elements`, and the two are distinct allocations, so no thread aliases
        // another's write. Threads with `i >= count` return before any access.
        let _ = unsafe { builder.launch(Self::launch_config(count)) }?;
        self.download_u64(&device_out)
    }

    pub fn g2_step_probe(&self, op: G2StepProbe, input: &[u64]) -> Result<Vec<u64>, CudaError> {
        let (in_width, out_width) = op.widths();
        if in_width == 0 || !input.len().is_multiple_of(in_width) {
            return Err(CudaError::LengthMismatch {
                expected: in_width,
                got: input.len(),
            });
        }
        let elements = input.len() / in_width;
        if elements == 0 {
            return Ok(Vec::new());
        }
        let count = Self::count_of(elements)?;
        let device_in = self.upload_u64_slice(input)?;
        let mut device_out = self.alloc_u64(elements * out_width)?;
        let function = match op {
            G2StepProbe::DoubleStep => &self.g2_double_step_probe,
            G2StepProbe::AddStep => &self.g2_add_step_probe,
            G2StepProbe::MulByChar => &self.g2_mul_by_char_probe,
        };
        self.with_pairing_constants(|constants| {
            let mut builder = self.stream().launch_builder(function);
            let _ = builder.arg(&device_in);
            let _ = builder.arg(&constants.values);
            let _ = builder.arg(&mut device_out);
            let _ = builder.arg(&count);
            // SAFETY: thread `i < count` reads exactly the `in_width` u64s at
            // `input[i * in_width]` plus the `PC_WORDS` pairing constants whose
            // length is fixed at upload, and writes exactly the `out_width` u64s
            // at `output[i * out_width]`. The upload holds `count * in_width`
            // u64s and the allocation `count * out_width`, both sized above from
            // the same `elements`, and the two are distinct allocations, so no
            // thread aliases another's write. Threads with `i >= count` return
            // before any access.
            let _ = unsafe { builder.launch(Self::launch_config(count)) }?;
            Ok(())
        })?;
        self.download_u64(&device_out)
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test module: device operations fail loudly"
)]
mod tests {
    use ark_bn254::{Fq12, Fq6, G2Projective};
    use ark_ff::{BigInt, UniformRand};
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    use super::*;
    use crate::cuda::common::context::shared_context;

    fn fq(words: &[u64]) -> Fq {
        Fq::new_unchecked(BigInt([words[0], words[1], words[2], words[3]]))
    }

    fn fq2(words: &[u64]) -> Fq2 {
        Fq2::new(fq(&words[..FQ_LIMBS]), fq(&words[FQ_LIMBS..]))
    }

    fn fq12(words: &[u64]) -> Fq12 {
        let fq6 = |w: &[u64]| {
            Fq6::new(
                fq2(&w[..2 * FQ_LIMBS]),
                fq2(&w[2 * FQ_LIMBS..4 * FQ_LIMBS]),
                fq2(&w[4 * FQ_LIMBS..]),
            )
        };
        Fq12::new(fq6(&words[..6 * FQ_LIMBS]), fq6(&words[6 * FQ_LIMBS..]))
    }

    fn fq2_words(value: &Fq2) -> Vec<u64> {
        let mut out = Vec::with_capacity(FQ2_LIMBS);
        out.extend_from_slice(&value.c0.0 .0);
        out.extend_from_slice(&value.c1.0 .0);
        out
    }

    fn fq6_words(value: &Fq6) -> Vec<u64> {
        let mut out = Vec::with_capacity(FQ6_LIMBS);
        for part in [&value.c0, &value.c1, &value.c2] {
            out.extend_from_slice(&fq2_words(part));
        }
        out
    }

    fn fq12_words(value: &Fq12) -> Vec<u64> {
        let mut out = Vec::with_capacity(FQ12_LIMBS);
        out.extend_from_slice(&fq6_words(&value.c0));
        out.extend_from_slice(&fq6_words(&value.c1));
        out
    }

    fn fq6_of(words: &[u64]) -> Fq6 {
        Fq6::new(
            fq2(&words[..FQ2_LIMBS]),
            fq2(&words[FQ2_LIMBS..2 * FQ2_LIMBS]),
            fq2(&words[2 * FQ2_LIMBS..]),
        )
    }

    fn tower_fq2_samples(rng: &mut ChaCha20Rng) -> Vec<Fq2> {
        let mut samples = vec![
            Fq2::ZERO,
            Fq2::ONE,
            -Fq2::ONE,
            Fq2::new(Fq::ZERO, Fq::ONE),
            Fq2::new(-Fq::ONE, Fq::ZERO),
            Fq2::new(-Fq::ONE, -Fq::ONE),
        ];
        samples.extend((0..6).map(|_| Fq2::rand(rng)));
        samples
    }

    fn tower_fq6_samples(rng: &mut ChaCha20Rng) -> Vec<Fq6> {
        let parts = tower_fq2_samples(rng);
        (0..parts.len())
            .map(|index| {
                Fq6::new(
                    parts[index],
                    parts[(index + 1) % parts.len()],
                    parts[(index + 2) % parts.len()],
                )
            })
            .collect()
    }

    fn tower_fq12_samples(rng: &mut ChaCha20Rng) -> Vec<Fq12> {
        let parts = tower_fq6_samples(rng);
        (0..parts.len())
            .map(|index| Fq12::new(parts[index], parts[(index + 1) % parts.len()]))
            .collect()
    }

    fn probe_fq6(op: TowerProbe, input: &[u64]) -> Vec<Fq6> {
        let context = shared_context().expect("a CUDA device");
        let got = context.tower_probe(op, input).expect("a tower probe");
        got.chunks_exact(FQ6_LIMBS).map(fq6_of).collect()
    }

    fn probe_fq12(op: TowerProbe, input: &[u64]) -> Vec<Fq12> {
        let context = shared_context().expect("a CUDA device");
        let got = context.tower_probe(op, input).expect("a tower probe");
        got.chunks_exact(FQ12_LIMBS).map(fq12).collect()
    }

    #[test]
    fn fq6_mul_matches_arkworks() {
        if shared_context().is_none() {
            return;
        }
        let mut rng = ChaCha20Rng::seed_from_u64(7_100);
        let samples = tower_fq6_samples(&mut rng);
        let mut input = Vec::new();
        let mut expected = Vec::new();
        for (index, left) in samples.iter().enumerate() {
            let right = samples[(index + 3) % samples.len()];
            input.extend_from_slice(&fq6_words(left));
            input.extend_from_slice(&fq6_words(&right));
            expected.push(*left * right);
        }
        assert_eq!(
            probe_fq6(TowerProbe::Fq6Mul, &input),
            expected,
            "fq6_mul diverged from arkworks"
        );
    }

    #[test]
    fn fq6_mul_by_01_matches_arkworks() {
        if shared_context().is_none() {
            return;
        }
        let mut rng = ChaCha20Rng::seed_from_u64(7_200);
        let samples = tower_fq6_samples(&mut rng);
        let parts = tower_fq2_samples(&mut rng);
        let mut input = Vec::new();
        let mut expected = Vec::new();
        for (index, left) in samples.iter().enumerate() {
            let c0 = parts[index % parts.len()];
            let c1 = parts[(index + 4) % parts.len()];
            input.extend_from_slice(&fq6_words(left));
            input.extend_from_slice(&fq2_words(&c0));
            input.extend_from_slice(&fq2_words(&c1));
            expected.push(*left * Fq6::new(c0, c1, Fq2::ZERO));
        }
        assert_eq!(
            probe_fq6(TowerProbe::Fq6MulBy01, &input),
            expected,
            "fq6_mul_by_01 diverged from arkworks"
        );
    }

    #[test]
    fn fq6_mul_by_fq2_matches_arkworks() {
        if shared_context().is_none() {
            return;
        }
        let mut rng = ChaCha20Rng::seed_from_u64(7_300);
        let samples = tower_fq6_samples(&mut rng);
        let parts = tower_fq2_samples(&mut rng);
        let mut input = Vec::new();
        let mut expected = Vec::new();
        for (index, left) in samples.iter().enumerate() {
            let scale = parts[(index + 2) % parts.len()];
            input.extend_from_slice(&fq6_words(left));
            input.extend_from_slice(&fq2_words(&scale));
            expected.push(*left * Fq6::new(scale, Fq2::ZERO, Fq2::ZERO));
        }
        assert_eq!(
            probe_fq6(TowerProbe::Fq6MulByFq2, &input),
            expected,
            "fq6_mul_by_fq2 diverged from arkworks"
        );
    }

    #[test]
    fn fq6_mul_by_nonresidue_matches_arkworks() {
        if shared_context().is_none() {
            return;
        }
        let mut rng = ChaCha20Rng::seed_from_u64(7_400);
        let samples = tower_fq6_samples(&mut rng);
        let mut input = Vec::new();
        let mut expected = Vec::new();
        for value in &samples {
            input.extend_from_slice(&fq6_words(value));
            expected.push(*value * Fq6::new(Fq2::ZERO, Fq2::ONE, Fq2::ZERO));
        }
        assert_eq!(
            probe_fq6(TowerProbe::Fq6MulByNonresidue, &input),
            expected,
            "fq6_mul_by_nonresidue diverged from arkworks"
        );
    }

    #[test]
    fn fq12_mul_matches_arkworks() {
        if shared_context().is_none() {
            return;
        }
        let mut rng = ChaCha20Rng::seed_from_u64(7_500);
        let samples = tower_fq12_samples(&mut rng);
        let mut input = Vec::new();
        let mut expected = Vec::new();
        for (index, left) in samples.iter().enumerate() {
            let right = samples[(index + 5) % samples.len()];
            input.extend_from_slice(&fq12_words(left));
            input.extend_from_slice(&fq12_words(&right));
            expected.push(*left * right);
        }
        assert_eq!(
            probe_fq12(TowerProbe::Fq12Mul, &input),
            expected,
            "fq12_mul diverged from arkworks"
        );
    }

    #[test]
    fn fq12_sqr_matches_arkworks() {
        if shared_context().is_none() {
            return;
        }
        let mut rng = ChaCha20Rng::seed_from_u64(7_600);
        let samples = tower_fq12_samples(&mut rng);
        let mut input = Vec::new();
        let mut expected = Vec::new();
        for value in &samples {
            input.extend_from_slice(&fq12_words(value));
            expected.push(value.square());
        }
        assert_eq!(
            probe_fq12(TowerProbe::Fq12Sqr, &input),
            expected,
            "fq12_sqr diverged from arkworks"
        );
    }

    #[test]
    fn fq12_mul_by_034_matches_arkworks() {
        if shared_context().is_none() {
            return;
        }
        let mut rng = ChaCha20Rng::seed_from_u64(7_700);
        let samples = tower_fq12_samples(&mut rng);
        let parts = tower_fq2_samples(&mut rng);
        let mut input = Vec::new();
        let mut expected = Vec::new();
        for (index, value) in samples.iter().enumerate() {
            let c0 = parts[index % parts.len()];
            let c3 = parts[(index + 3) % parts.len()];
            let c4 = parts[(index + 5) % parts.len()];
            input.extend_from_slice(&fq12_words(value));
            input.extend_from_slice(&fq2_words(&c0));
            input.extend_from_slice(&fq2_words(&c3));
            input.extend_from_slice(&fq2_words(&c4));
            let sparse = Fq12::new(
                Fq6::new(c0, Fq2::ZERO, Fq2::ZERO),
                Fq6::new(c3, c4, Fq2::ZERO),
            );
            expected.push(*value * sparse);
        }
        assert_eq!(
            probe_fq12(TowerProbe::Fq12MulBy034, &input),
            expected,
            "fq12_mul_by_034 diverged from arkworks"
        );
    }

    fn fq2_scale(value: Fq2, scale: Fq) -> Fq2 {
        Fq2::new(value.c0 * scale, value.c1 * scale)
    }

    fn probe_g2_step_pair(op: G2StepProbe, input: &[u64]) -> (Vec<Fq2>, Vec<Fq2>) {
        let context = shared_context().expect("a CUDA device");
        let (_, out_width) = op.widths();
        let raw = context.g2_step_probe(op, input).expect("a g2 step probe");
        let half = out_width / 2;
        let mut plain = Vec::new();
        let mut warp = Vec::new();
        for chunk in raw.chunks_exact(out_width) {
            plain.extend(chunk[..half].chunks_exact(FQ2_LIMBS).map(fq2));
            warp.extend(chunk[half..].chunks_exact(FQ2_LIMBS).map(fq2));
        }
        (plain, warp)
    }

    fn g2_step_input(count: usize, parts: usize, rng: &mut ChaCha20Rng) -> Vec<u64> {
        let mut input = Vec::new();
        for _ in 0..count {
            let mut emitted = 0;
            while emitted < parts {
                let point = G2Projective::rand(rng);
                for value in [&point.x, &point.y, &point.z] {
                    if emitted < parts {
                        input.extend_from_slice(&fq2_words(value));
                        emitted += 1;
                    }
                }
            }
        }
        input
    }

    #[test]
    fn ell_matches_arkworks() {
        if shared_context().is_none() {
            return;
        }
        let mut rng = ChaCha20Rng::seed_from_u64(7_900);
        let samples = tower_fq12_samples(&mut rng);
        let parts = tower_fq2_samples(&mut rng);
        let mut input = Vec::new();
        let mut expected = Vec::new();
        for (index, value) in samples.iter().enumerate() {
            let coeff0 = parts[index % parts.len()];
            let coeff1 = parts[(index + 2) % parts.len()];
            let coeff2 = parts[(index + 4) % parts.len()];
            let px = Fq::rand(&mut rng);
            let py = Fq::rand(&mut rng);
            input.extend_from_slice(&fq12_words(value));
            input.extend_from_slice(&fq2_words(&coeff0));
            input.extend_from_slice(&fq2_words(&coeff1));
            input.extend_from_slice(&fq2_words(&coeff2));
            input.extend_from_slice(&px.0 .0);
            input.extend_from_slice(&py.0 .0);
            let sparse = Fq12::new(
                Fq6::new(fq2_scale(coeff0, py), Fq2::ZERO, Fq2::ZERO),
                Fq6::new(fq2_scale(coeff1, px), coeff2, Fq2::ZERO),
            );
            expected.push(*value * sparse);
        }
        assert_eq!(
            probe_fq12(TowerProbe::Ell, &input),
            expected,
            "ell diverged from arkworks"
        );
    }

    #[test]
    fn g2_double_step_matches_the_warp_implementation() {
        if shared_context().is_none() {
            return;
        }
        let mut rng = ChaCha20Rng::seed_from_u64(8_000);
        let input = g2_step_input(12, 3, &mut rng);
        let (plain, warp) = probe_g2_step_pair(G2StepProbe::DoubleStep, &input);
        assert_eq!(
            plain, warp,
            "g2_double_step diverged from the mw_ implementation it mirrors"
        );
    }

    #[test]
    fn g2_add_step_matches_the_warp_implementation() {
        if shared_context().is_none() {
            return;
        }
        let mut rng = ChaCha20Rng::seed_from_u64(8_100);
        let input = g2_step_input(12, 5, &mut rng);
        let (plain, warp) = probe_g2_step_pair(G2StepProbe::AddStep, &input);
        assert_eq!(
            plain, warp,
            "g2_add_step diverged from the mw_ implementation it mirrors"
        );
    }

    #[test]
    fn g2_mul_by_char_matches_the_warp_implementation() {
        if shared_context().is_none() {
            return;
        }
        let mut rng = ChaCha20Rng::seed_from_u64(8_200);
        let input = g2_step_input(12, 2, &mut rng);
        let (plain, warp) = probe_g2_step_pair(G2StepProbe::MulByChar, &input);
        assert_eq!(
            plain, warp,
            "g2_mul_by_char diverged from the mw_ implementation it mirrors"
        );
    }
}
