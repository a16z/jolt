use super::sumcheck::SumcheckInstanceProof;
use crate::{
    field::JoltField,
    jolt::instruction::{
        prefixes::{PrefixCheckpoint, PrefixEval, Prefixes},
        suffixes::{SuffixEval, Suffixes},
        JoltInstruction,
    },
    poly::{
        dense_mlpoly::DensePolynomial,
        eq_poly::EqPolynomial,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        unipoly::{CompressedUniPoly, UniPoly},
    },
    utils::{
        errors::ProofVerifyError,
        math::Math,
        thread::unsafe_allocate_zero_vec,
        transcript::{AppendToTranscript, Transcript},
        uninterleave_bits,
    },
};
use num::FromPrimitive;
use rayon::{prelude::*, slice::Iter};
use std::{fmt::Display, ops::Index};
use strum::{EnumCount, IntoEnumIterator};

#[derive(Clone)]
struct ExpandingTable<F: JoltField> {
    len: usize,
    values: Vec<F>,
    scratch_space: Vec<F>,
}

impl<F: JoltField> ExpandingTable<F> {
    #[tracing::instrument(skip_all, name = "ExpandingTable::new")]
    fn new(capacity: usize) -> Self {
        let (values, scratch_space) = rayon::join(
            || unsafe_allocate_zero_vec(capacity),
            || unsafe_allocate_zero_vec(capacity),
        );
        Self {
            len: 0,
            values,
            scratch_space,
        }
    }

    fn reset(&mut self, value: F) {
        self.values[0] = value;
        self.len = 1;
    }

    #[tracing::instrument(skip_all, name = "ExpandingTable::update")]
    fn update(&mut self, r_j: F) {
        self.values[..self.len]
            .par_iter()
            .zip(self.scratch_space.par_chunks_mut(2))
            .for_each(|(&v_i, dest)| {
                let eval_1 = r_j * v_i;
                dest[0] = v_i - eval_1;
                dest[1] = eval_1;
            });
        std::mem::swap(&mut self.values, &mut self.scratch_space);
        self.len *= 2;
    }
}

impl<F: JoltField> Index<usize> for ExpandingTable<F> {
    type Output = F;

    fn index(&self, index: usize) -> &F {
        assert!(index < self.len);
        &self.values[index]
    }
}

impl<'data, F: JoltField> IntoParallelIterator for &'data ExpandingTable<F> {
    type Item = &'data F;
    type Iter = Iter<'data, F>;

    fn into_par_iter(self) -> Self::Iter {
        self.values[..self.len].into_par_iter()
    }
}

impl<'data, F: JoltField> ParallelSlice<F> for &'data ExpandingTable<F> {
    fn as_parallel_slice(&self) -> &[F] {
        self.values[..self.len].as_parallel_slice()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct LookupBits {
    bits: u64,
    len: usize,
}

impl LookupBits {
    pub fn new(mut bits: u64, len: usize) -> Self {
        debug_assert!(len <= 64);
        if len < 64 {
            bits = bits % (1 << len);
        }
        Self { bits, len }
    }

    pub fn uninterleave(&self) -> (Self, Self) {
        let (x_bits, y_bits) = uninterleave_bits(self.bits);
        let x = Self::new(x_bits as u64, self.len / 2);
        let y = Self::new(y_bits as u64, self.len - x.len);
        (x, y)
    }

    pub fn split(&self, suffix_len: usize) -> (Self, Self) {
        let suffix_bits = self.bits % (1 << suffix_len);
        let suffix = Self::new(suffix_bits, suffix_len);
        let prefix_bits = self.bits >> suffix_len;
        let prefix = Self::new(prefix_bits, self.len - suffix_len);
        (prefix, suffix)
    }

    pub fn pop_msb(&mut self) -> u8 {
        let msb = (self.bits >> (self.len - 1)) & 1;
        self.bits = self.bits % (1 << (self.len - 1));
        self.len -= 1;
        msb as u8
    }

    pub fn get_bit(&self, index: usize) -> u8 {
        assert!(index < self.len);
        let bit = (self.bits >> (self.len - 1 - index)) & 1;
        bit as u8
    }

    pub fn len(&self) -> usize {
        self.len
    }
}

impl Display for LookupBits {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:0width$b}", self.bits, width = self.len)
    }
}

impl From<LookupBits> for u64 {
    fn from(value: LookupBits) -> u64 {
        value.bits
    }
}
impl From<LookupBits> for usize {
    fn from(value: LookupBits) -> usize {
        value.bits.try_into().unwrap()
    }
}
impl From<LookupBits> for u32 {
    fn from(value: LookupBits) -> u32 {
        value.bits.try_into().unwrap()
    }
}
impl From<&LookupBits> for u64 {
    fn from(value: &LookupBits) -> u64 {
        value.bits
    }
}
impl From<&LookupBits> for usize {
    fn from(value: &LookupBits) -> usize {
        value.bits.try_into().unwrap()
    }
}
impl From<&LookupBits> for u32 {
    fn from(value: &LookupBits) -> u32 {
        value.bits.try_into().unwrap()
    }
}
impl std::ops::Rem<usize> for &LookupBits {
    type Output = usize;

    fn rem(self, rhs: usize) -> Self::Output {
        usize::from(self) % rhs
    }
}
impl std::ops::Rem<usize> for LookupBits {
    type Output = usize;

    fn rem(self, rhs: usize) -> Self::Output {
        usize::from(self) % rhs
    }
}
impl PartialEq for LookupBits {
    fn eq(&self, other: &Self) -> bool {
        u64::from(self) == u64::from(other)
    }
}

pub fn current_suffix_len(log_K: usize, j: usize) -> usize {
    let phase_length = log_K / 4;
    let suffix_len = log_K - (j / phase_length + 1) * phase_length;
    suffix_len
}

pub trait PrefixSuffixDecomposition<const WORD_SIZE: usize, F: JoltField>:
    JoltInstruction + Default
{
    fn prefixes() -> Vec<Prefixes>;
    fn suffixes() -> Vec<Suffixes>;
    fn combine(prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F;

    fn compute_sumcheck_prover_message(
        prefix_checkpoints: &[PrefixCheckpoint<F>],
        suffix_polys: &[DensePolynomial<F>],
        r: &[F],
        j: usize,
    ) -> [F; 2] {
        let len = suffix_polys[0].len();
        let log_len = len.log_2();

        let r_x = if j % 2 == 1 { r.last().copied() } else { None };

        let (eval_0, eval_2_left, eval_2_right): (F, F, F) = (0..len / 2)
            .into_par_iter()
            .map(|b| {
                let b = LookupBits::new(b as u64, log_len - 1);
                // TODO(moodlezoup): Avoid allocations
                let prefixes_c0: Vec<_> = Prefixes::iter()
                    .map(|prefix| {
                        prefix.prefix_mle::<WORD_SIZE, F>(prefix_checkpoints, r_x, 0, b, j)
                    })
                    .collect();
                let prefixes_c2: Vec<_> = Prefixes::iter()
                    .map(|prefix| {
                        prefix.prefix_mle::<WORD_SIZE, F>(prefix_checkpoints, r_x, 2, b, j)
                    })
                    .collect();
                let suffixes_left: Vec<_> = suffix_polys
                    .iter()
                    .map(|suffix_poly| suffix_poly[b.into()].into())
                    .collect();
                let suffixes_right: Vec<_> = suffix_polys
                    .iter()
                    .map(|suffix_poly| suffix_poly[usize::from(b) + len / 2].into())
                    .collect();
                (
                    Self::combine(&prefixes_c0, &suffixes_left),
                    Self::combine(&prefixes_c2, &suffixes_left),
                    Self::combine(&prefixes_c2, &suffixes_right),
                )
            })
            .reduce(
                || (F::zero(), F::zero(), F::zero()),
                |running, new| (running.0 + new.0, running.1 + new.1, running.2 + new.2),
            );

        [eval_0, eval_2_right + eval_2_right - eval_2_left]
    }
}

pub fn prove_single_instruction<
    const WORD_SIZE: usize,
    F: JoltField,
    I: PrefixSuffixDecomposition<WORD_SIZE, F>,
    ProofTranscript: Transcript,
>(
    instructions: &[I],
    r_cycle: Vec<F>,
    transcript: &mut ProofTranscript,
) -> (SumcheckInstanceProof<F, ProofTranscript>, F, [F; 4]) {
    let log_K: usize = 2 * WORD_SIZE;
    let log_m = log_K / 4;
    let m = log_m.pow2();

    let T = instructions.len();
    let log_T = T.log_2();
    debug_assert_eq!(r_cycle.len(), log_T);

    let num_chunks = rayon::current_num_threads().next_power_of_two().min(T);
    let chunk_size = (m / num_chunks).max(1);

    let num_rounds = log_K + log_T;
    let mut r: Vec<F> = Vec::with_capacity(num_rounds);
    let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::with_capacity(num_rounds);

    let span = tracing::span!(tracing::Level::INFO, "compute lookup indices");
    let _guard = span.enter();
    let lookup_indices: Vec<_> = instructions
        .par_iter()
        .map(|instruction| LookupBits::new(instruction.to_lookup_index(), log_K))
        .collect();
    drop(_guard);
    drop(span);

    let (eq_r_prime, mut u_evals) = rayon::join(
        || EqPolynomial::evals(&r_cycle),
        || EqPolynomial::evals_with_r2(&r_cycle),
    );

    let mut prefix_checkpoints: Vec<PrefixCheckpoint<F>> = vec![None.into(); Prefixes::COUNT];
    let mut v = ExpandingTable::new(m);

    let span = tracing::span!(tracing::Level::INFO, "compute rv_claim");
    let _guard = span.enter();
    let rv_claim = lookup_indices
        .par_iter()
        .zip(u_evals.par_iter())
        .map(|(k, u)| u.mul_u64_unchecked(I::default().materialize_entry(k.into())))
        .sum();
    drop(_guard);
    drop(span);

    let mut previous_claim = rv_claim;

    #[cfg(test)]
    let mut val_test: MultilinearPolynomial<F> =
        MultilinearPolynomial::from(I::default().materialize());
    #[cfg(test)]
    let mut eq_ra_test: MultilinearPolynomial<F> = {
        let mut eq_ra: Vec<F> = unsafe_allocate_zero_vec(val_test.len());
        for (j, k) in lookup_indices.iter().enumerate() {
            eq_ra[usize::from(k)] += eq_r_prime[j];
        }
        MultilinearPolynomial::from(eq_ra)
    };

    let mut j: usize = 0;
    let mut ra: Vec<MultilinearPolynomial<F>> = Vec::with_capacity(4);

    let mut suffix_polys: Vec<DensePolynomial<F>> = (0..Suffixes::COUNT)
        .into_par_iter()
        .map(|_| DensePolynomial::new(unsafe_allocate_zero_vec(m)))
        .collect();

    for phase in 0..3 {
        let span = tracing::span!(tracing::Level::INFO, "sparse-dense phase");
        let _guard = span.enter();

        // Condensation
        if phase != 0 {
            let span = tracing::span!(tracing::Level::INFO, "Update u_evals");
            let _guard = span.enter();
            lookup_indices
                .par_iter()
                .zip(u_evals.par_iter_mut())
                .for_each(|(k, u)| {
                    let (prefix, _) = k.split((4 - phase) * log_m);
                    let k_bound: usize = prefix % m;
                    *u *= v[k_bound as usize];
                });
            drop(_guard);
            drop(span);
        }

        let suffix_len = (3 - phase) * log_m;

        // Initialize suffix poly for each suffix
        let span = tracing::span!(tracing::Level::INFO, "compute instruction_index_iters");
        let _guard = span.enter();
        let instruction_index_iters: Vec<_> = (0..num_chunks)
            .into_par_iter()
            .map(|i| {
                lookup_indices.iter().enumerate().filter_map(move |(j, k)| {
                    let (prefix, _) = k.split(suffix_len);
                    let group = (prefix % m) / chunk_size;
                    if group == i {
                        Some(j)
                    } else {
                        None
                    }
                })
            })
            .collect();
        drop(_guard);
        drop(span);

        let span = tracing::span!(tracing::Level::INFO, "Compute suffix polys");
        let _guard = span.enter();
        suffix_polys
            .par_iter_mut()
            .enumerate()
            .for_each(|(index, poly)| {
                if phase != 0 {
                    poly.len = m;
                    poly.num_vars = poly.len.log_2();
                    poly.Z.par_iter_mut().for_each(|eval| *eval = F::zero());
                }
                instruction_index_iters
                    .par_iter()
                    .zip(poly.Z.par_chunks_mut(chunk_size))
                    .for_each(|(j_iter, evals)| {
                        j_iter.clone().for_each(|j| {
                            let k = lookup_indices[j];
                            let u = u_evals[j];
                            let (prefix_bits, suffix_bits) = k.split(suffix_len);
                            let suffix: Suffixes = FromPrimitive::from_u8(index as u8).unwrap();
                            let t = suffix.suffix_mle::<WORD_SIZE>(suffix_bits);
                            let index = prefix_bits % chunk_size;
                            evals[index] += u.mul_u64_unchecked(t as u64);
                        });
                    });
            });

        drop(_guard);
        drop(span);

        v.reset(F::one());

        for _round in 0..log_m {
            let span = tracing::span!(tracing::Level::INFO, "sparse-dense sumcheck round");
            let _guard = span.enter();

            let univariate_poly_evals =
                I::compute_sumcheck_prover_message(&prefix_checkpoints, &suffix_polys, &r, j);

            #[cfg(test)]
            {
                let expected: [F; 2] = (0..val_test.len() / 2)
                    .into_par_iter()
                    .map(|i| {
                        let eq_ra_evals = eq_ra_test.sumcheck_evals(i, 2, BindingOrder::HighToLow);
                        let val_evals = val_test.sumcheck_evals(i, 2, BindingOrder::HighToLow);

                        [eq_ra_evals[0] * val_evals[0], eq_ra_evals[1] * val_evals[1]]
                    })
                    .reduce(
                        || [F::zero(); 2],
                        |running, new| [running[0] + new[0], running[1] + new[1]],
                    );
                assert_eq!(
                    expected, univariate_poly_evals,
                    "Sumcheck sanity check failed in phase {phase} round {_round}"
                );
            }

            let univariate_poly = UniPoly::from_evals(&[
                univariate_poly_evals[0],
                previous_claim - univariate_poly_evals[0],
                univariate_poly_evals[1],
            ]);

            let compressed_poly = univariate_poly.compress();
            compressed_poly.append_to_transcript(transcript);
            compressed_polys.push(compressed_poly);

            let r_j = transcript.challenge_scalar::<F>();
            r.push(r_j);

            previous_claim = univariate_poly.evaluate(&r_j);

            suffix_polys
                .par_iter_mut()
                .for_each(|poly| poly.bind_parallel(r_j, BindingOrder::HighToLow));
            v.update(r_j);

            {
                if r.len() % 2 == 0 {
                    let span = tracing::span!(tracing::Level::INFO, "Update prefix checkpoints");
                    let _guard = span.enter();

                    Prefixes::update_checkpoints::<WORD_SIZE, F>(
                        &mut prefix_checkpoints,
                        r[r.len() - 2],
                        r[r.len() - 1],
                        j,
                    );
                }
            }

            #[cfg(test)]
            {
                eq_ra_test.bind_parallel(r_j, BindingOrder::HighToLow);
                val_test.bind_parallel(r_j, BindingOrder::HighToLow);
            }

            j += 1;
        }

        let span = tracing::span!(tracing::Level::INFO, "cache ra_i");
        let _guard = span.enter();

        let ra_i: Vec<F> = lookup_indices
            .par_iter()
            .map(|k| {
                let (prefix, _) = k.split(suffix_len);
                let k_bound: usize = prefix % m;
                v[k_bound as usize]
            })
            .collect();
        ra.push(MultilinearPolynomial::from(ra_i));
    }

    // At this point we switch from sparse-dense sumcheck (see Section 7.1 of the Twist+Shout
    // paper) to "vanilla" Shout, i.e. Section 6.2 where d=4.
    // Note that we've already bound 3/4 of the address variables, so ra_1, ra_2, and ra_3
    // are fully bound when we start "vanilla" Shout.

    // Modified version of the C array described in Equation (47) of the Twist+Shout paper
    let span = tracing::span!(tracing::Level::INFO, "Materialize eq_ra");
    let _guard = span.enter();
    let instruction_index_iters: Vec<_> = (0..num_chunks)
        .into_par_iter()
        .map(|i| {
            lookup_indices.iter().enumerate().filter_map(move |(j, k)| {
                let group = (k % m) / chunk_size;
                if group == i {
                    Some(j)
                } else {
                    None
                }
            })
        })
        .collect();

    let mut eq_ra: Vec<F> = unsafe_allocate_zero_vec(m);
    instruction_index_iters
        .into_par_iter()
        .zip(eq_ra.par_chunks_mut(chunk_size))
        .for_each(|(j_iter, chunk)| {
            j_iter.for_each(|j| {
                let k = lookup_indices[j];
                chunk[k % chunk_size] +=
                    eq_r_prime[j] * ra[0].get_coeff(j) * ra[1].get_coeff(j) * ra[2].get_coeff(j);
            });
        });
    let mut eq_ra = MultilinearPolynomial::from(eq_ra);
    drop(_guard);
    drop(span);

    #[cfg(test)]
    {
        for i in 0..m {
            assert_eq!(eq_ra.get_bound_coeff(i), eq_ra_test.get_bound_coeff(i));
        }
    }

    let span = tracing::span!(tracing::Level::INFO, "Materialize val");
    let _guard = span.enter();
    let prefixes: Vec<_> = prefix_checkpoints
        .into_iter()
        .map(|checkpoint| checkpoint.unwrap())
        .collect();
    let val: Vec<F> = (0..m)
        .into_par_iter()
        .map(|k| {
            let suffixes: Vec<_> = Suffixes::iter()
                .map(|suffix| {
                    F::from_u32(suffix.suffix_mle::<WORD_SIZE>(LookupBits::new(k as u64, log_m)))
                        .into()
                })
                .collect();
            I::combine(&prefixes, &suffixes)
        })
        .collect();
    let mut val = MultilinearPolynomial::from(val);
    drop(_guard);
    drop(span);

    #[cfg(test)]
    {
        for i in 0..m {
            assert_eq!(val.get_bound_coeff(i), val_test.get_bound_coeff(i));
        }
    }

    v.reset(F::one());

    let span = tracing::span!(tracing::Level::INFO, "Next log(m) sumcheck rounds");
    let _guard = span.enter();

    for _round in 0..log_m {
        let span = tracing::span!(tracing::Level::INFO, "Compute univariate poly");
        let _guard = span.enter();

        let univariate_poly_evals: [F; 2] = (0..eq_ra.len() / 2)
            .into_par_iter()
            .map(|i| {
                let eq_ra_evals = eq_ra.sumcheck_evals(i, 2, BindingOrder::HighToLow);
                let val_evals = val.sumcheck_evals(i, 2, BindingOrder::HighToLow);

                [eq_ra_evals[0] * val_evals[0], eq_ra_evals[1] * val_evals[1]]
            })
            .reduce(
                || [F::zero(); 2],
                |running, new| [running[0] + new[0], running[1] + new[1]],
            );

        let univariate_poly = UniPoly::from_evals(&[
            univariate_poly_evals[0],
            previous_claim - univariate_poly_evals[0],
            univariate_poly_evals[1],
        ]);

        drop(_guard);
        drop(span);

        let compressed_poly = univariate_poly.compress();
        compressed_poly.append_to_transcript(transcript);
        compressed_polys.push(compressed_poly);

        let r_j = transcript.challenge_scalar::<F>();
        r.push(r_j);

        previous_claim = univariate_poly.evaluate(&r_j);

        let span = tracing::span!(tracing::Level::INFO, "Binding");
        let _guard = span.enter();

        // Bind polynomials
        rayon::join(
            || eq_ra.bind_parallel(r_j, BindingOrder::HighToLow),
            || val.bind_parallel(r_j, BindingOrder::HighToLow),
        );

        v.update(r_j);
        j += 1;
    }

    drop(_guard);
    drop(span);

    let span = tracing::span!(tracing::Level::INFO, "cache ra_i");
    let _guard = span.enter();

    let ra_i: Vec<F> = lookup_indices
        .par_iter()
        .map(|k| {
            let k_bound = k % m;
            v[k_bound]
        })
        .collect();
    ra.push(MultilinearPolynomial::from(ra_i));
    drop(_guard);
    drop(span);

    let mut eq_r_prime = MultilinearPolynomial::from(eq_r_prime);
    let val_eval = val.final_sumcheck_claim();

    let span = tracing::span!(tracing::Level::INFO, "last log(T) sumcheck rounds");
    let _guard = span.enter();

    for _round in 0..log_T {
        let span = tracing::span!(tracing::Level::INFO, "Compute univariate poly");
        let _guard = span.enter();

        let mut univariate_poly_evals: [F; 5] = (0..eq_r_prime.len() / 2)
            .into_par_iter()
            .map(|i| {
                let eq_evals = eq_r_prime.sumcheck_evals(i, 5, BindingOrder::HighToLow);
                let ra_0_evals = ra[0].sumcheck_evals(i, 5, BindingOrder::HighToLow);
                let ra_1_evals = ra[1].sumcheck_evals(i, 5, BindingOrder::HighToLow);
                let ra_2_evals = ra[2].sumcheck_evals(i, 5, BindingOrder::HighToLow);
                let ra_3_evals = ra[3].sumcheck_evals(i, 5, BindingOrder::HighToLow);

                [
                    eq_evals[0] * ra_0_evals[0] * ra_1_evals[0] * ra_2_evals[0] * ra_3_evals[0],
                    eq_evals[1] * ra_0_evals[1] * ra_1_evals[1] * ra_2_evals[1] * ra_3_evals[1],
                    eq_evals[2] * ra_0_evals[2] * ra_1_evals[2] * ra_2_evals[2] * ra_3_evals[2],
                    eq_evals[3] * ra_0_evals[3] * ra_1_evals[3] * ra_2_evals[3] * ra_3_evals[3],
                    eq_evals[4] * ra_0_evals[4] * ra_1_evals[4] * ra_2_evals[4] * ra_3_evals[4],
                ]
            })
            .reduce(
                || [F::zero(); 5],
                |running, new| {
                    [
                        running[0] + new[0],
                        running[1] + new[1],
                        running[2] + new[2],
                        running[3] + new[3],
                        running[4] + new[4],
                    ]
                },
            );
        univariate_poly_evals
            .iter_mut()
            .for_each(|eval| *eval *= val_eval);

        let univariate_poly = UniPoly::from_evals(&[
            univariate_poly_evals[0],
            previous_claim - univariate_poly_evals[0],
            univariate_poly_evals[1],
            univariate_poly_evals[2],
            univariate_poly_evals[3],
            univariate_poly_evals[4],
        ]);

        drop(_guard);
        drop(span);

        let compressed_poly = univariate_poly.compress();
        compressed_poly.append_to_transcript(transcript);
        compressed_polys.push(compressed_poly);

        let r_j = transcript.challenge_scalar::<F>();
        r.push(r_j);

        previous_claim = univariate_poly.evaluate(&r_j);

        let span = tracing::span!(tracing::Level::INFO, "Binding");
        let _guard = span.enter();

        ra.par_iter_mut()
            .chain([&mut eq_r_prime].into_par_iter())
            .for_each(|poly| poly.bind_parallel(r_j, BindingOrder::HighToLow));
    }

    (
        SumcheckInstanceProof::new(compressed_polys),
        rv_claim,
        [
            ra[0].final_sumcheck_claim(),
            ra[1].final_sumcheck_claim(),
            ra[2].final_sumcheck_claim(),
            ra[3].final_sumcheck_claim(),
        ],
    )
}

pub fn verify_single_instruction<
    F: JoltField,
    I: JoltInstruction + Default,
    ProofTranscript: Transcript,
>(
    proof: SumcheckInstanceProof<F, ProofTranscript>,
    log_K: usize,
    log_T: usize,
    r_cycle: Vec<F>,
    rv_claim: F,
    ra_claims: [F; 4],
    transcript: &mut ProofTranscript,
) -> Result<(), ProofVerifyError> {
    let (sumcheck_claim, r) = proof.verify(rv_claim, log_K + log_T, 5, transcript)?;
    let (r_address, r_cycle_prime) = r.split_at(log_K);

    let val_eval = I::default().evaluate_mle(r_address);
    let eq_eval_cycle = EqPolynomial::new(r_cycle).evaluate(r_cycle_prime);

    assert_eq!(
        eq_eval_cycle * ra_claims.iter().product::<F>() * val_eval,
        sumcheck_claim,
        "Read-checking sumcheck failed"
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        jolt::instruction::{
            add::ADDInstruction, and::ANDInstruction, beq::BEQInstruction, bge::BGEInstruction,
            bgeu::BGEUInstruction, bne::BNEInstruction, mul::MULInstruction,
            mulhu::MULHUInstruction, mulu::MULUInstruction, or::ORInstruction, sll::SLLInstruction,
            slt::SLTInstruction, sltu::SLTUInstruction, sub::SUBInstruction,
            virtual_advice::ADVICEInstruction,
            virtual_assert_halfword_alignment::AssertHalfwordAlignmentInstruction,
            virtual_assert_lte::ASSERTLTEInstruction,
            virtual_assert_valid_div0::AssertValidDiv0Instruction,
            virtual_assert_valid_signed_remainder::AssertValidSignedRemainderInstruction,
            virtual_assert_valid_unsigned_remainder::AssertValidUnsignedRemainderInstruction,
            virtual_movsign::MOVSIGNInstruction, xor::XORInstruction,
        },
        utils::transcript::KeccakTranscript,
    };
    use ark_bn254::Fr;
    use rand::{rngs::StdRng, SeedableRng};

    const WORD_SIZE: usize = 8;
    const LOG_K: usize = 16;
    const LOG_T: usize = 8;
    const T: usize = 1 << LOG_T;

    fn test_single_instruction<I: PrefixSuffixDecomposition<WORD_SIZE, Fr>>() {
        let mut rng = StdRng::seed_from_u64(12345);

        let instructions: Vec<_> = (0..T).map(|_| I::default().random(&mut rng)).collect();

        let mut prover_transcript = KeccakTranscript::new(b"test_transcript");
        let r_cycle: Vec<Fr> = prover_transcript.challenge_vector(LOG_T);

        let (proof, rv_claim, ra_claims) = prove_single_instruction::<WORD_SIZE, _, _, _>(
            &instructions,
            r_cycle,
            &mut prover_transcript,
        );

        let mut verifier_transcript = KeccakTranscript::new(b"test_transcript");
        verifier_transcript.compare_to(prover_transcript);
        let r_cycle: Vec<Fr> = verifier_transcript.challenge_vector(LOG_T);
        let verification_result = verify_single_instruction::<_, I, _>(
            proof,
            LOG_K,
            LOG_T,
            r_cycle,
            rv_claim,
            ra_claims,
            &mut verifier_transcript,
        );
        assert!(
            verification_result.is_ok(),
            "Verification failed with error: {:?}",
            verification_result.err()
        );
    }

    #[test]
    fn test_add() {
        test_single_instruction::<ADDInstruction<WORD_SIZE>>();
    }

    #[test]
    fn test_sub() {
        test_single_instruction::<SUBInstruction<WORD_SIZE>>();
    }

    #[test]
    fn test_and() {
        test_single_instruction::<ANDInstruction<WORD_SIZE>>();
    }

    #[test]
    fn test_or() {
        test_single_instruction::<ORInstruction<WORD_SIZE>>();
    }

    #[test]
    fn test_xor() {
        test_single_instruction::<XORInstruction<WORD_SIZE>>();
    }

    #[test]
    fn test_beq() {
        test_single_instruction::<BEQInstruction<WORD_SIZE>>();
    }

    #[test]
    fn test_bge() {
        test_single_instruction::<BGEInstruction<WORD_SIZE>>();
    }

    #[test]
    fn test_bgeu() {
        test_single_instruction::<BGEUInstruction<WORD_SIZE>>();
    }

    #[test]
    fn test_bne() {
        test_single_instruction::<BNEInstruction<WORD_SIZE>>();
    }

    #[test]
    fn test_slt() {
        test_single_instruction::<SLTInstruction<WORD_SIZE>>();
    }

    #[test]
    fn test_sltu() {
        test_single_instruction::<SLTUInstruction<WORD_SIZE>>();
    }

    #[test]
    fn test_movsign() {
        test_single_instruction::<MOVSIGNInstruction<WORD_SIZE>>();
    }

    #[test]
    fn test_mul() {
        test_single_instruction::<MULInstruction<WORD_SIZE>>();
    }

    #[test]
    fn test_mulu() {
        test_single_instruction::<MULUInstruction<WORD_SIZE>>();
    }

    #[test]
    fn test_mulhu() {
        test_single_instruction::<MULHUInstruction<WORD_SIZE>>();
    }

    #[test]
    fn test_advice() {
        test_single_instruction::<ADVICEInstruction<WORD_SIZE>>();
    }

    #[test]
    fn test_assert_lte() {
        test_single_instruction::<ASSERTLTEInstruction<WORD_SIZE>>();
    }

    #[test]
    fn test_assert_valid_signed_remainder() {
        test_single_instruction::<AssertValidSignedRemainderInstruction<WORD_SIZE>>();
    }

    #[test]
    fn test_assert_valid_unsigned_remainder() {
        test_single_instruction::<AssertValidUnsignedRemainderInstruction<WORD_SIZE>>();
    }

    #[test]
    fn test_assert_valid_div0() {
        test_single_instruction::<AssertValidDiv0Instruction<WORD_SIZE>>();
    }

    #[test]
    fn test_assert_halfword_alignment() {
        test_single_instruction::<AssertHalfwordAlignmentInstruction<WORD_SIZE>>();
    }

    #[ignore = "not working"]
    #[test]
    fn test_sll() {
        test_single_instruction::<SLLInstruction<WORD_SIZE>>();
    }
}
