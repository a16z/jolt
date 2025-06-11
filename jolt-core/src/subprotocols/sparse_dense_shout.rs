use super::sumcheck::SumcheckInstanceProof;
use crate::{
    field::JoltField,
    jolt::{
        instruction::{InstructionLookup, LookupQuery},
        lookup_table::{
            prefixes::{PrefixCheckpoint, PrefixEval, Prefixes},
            LookupTables,
        },
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
        thread::{drop_in_background_thread, unsafe_allocate_zero_vec, unsafe_zero_slice},
        transcript::{AppendToTranscript, Transcript},
        uninterleave_bits,
    },
};
use rayon::{prelude::*, slice::Iter};
use std::{fmt::Display, ops::Index};
use strum::{EnumCount, IntoEnumIterator};
use tracer::instruction::RV32IMCycle;

/// Table containing the evaluations `EQ(x_1, ..., x_j, r_1, ..., r_j)`,
/// built up incrementally as we receive random challenges `r_j` over the
/// course of sumcheck.
#[derive(Clone)]
struct ExpandingTable<F: JoltField> {
    len: usize,
    values: Vec<F>,
    scratch_space: Vec<F>,
}

impl<F: JoltField> ExpandingTable<F> {
    /// Initializes an `ExpandingTable` with the given `capacity`.
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

    /// Resets this table to be length 1, containing only the given `value`.
    fn reset(&mut self, value: F) {
        self.values[0] = value;
        self.len = 1;
    }

    /// Updates this table (expanding it by a factor of 2) to incorporate
    /// the new random challenge `r_j`.
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

impl<F: JoltField> ParallelSlice<F> for &ExpandingTable<F> {
    fn as_parallel_slice(&self) -> &[F] {
        self.values[..self.len].as_parallel_slice()
    }
}

/// A bitvector type used to represent a (substring of a) lookup index.
#[derive(Clone, Copy, Debug)]
pub struct LookupBits {
    bits: u64,
    len: usize,
}

impl LookupBits {
    pub fn new(mut bits: u64, len: usize) -> Self {
        debug_assert!(len <= 64);
        if len < 64 {
            bits %= 1 << len;
        }
        Self { bits, len }
    }

    pub fn uninterleave(&self) -> (Self, Self) {
        let (x_bits, y_bits) = uninterleave_bits(self.bits);
        let x = Self::new(x_bits as u64, self.len / 2);
        let y = Self::new(y_bits as u64, self.len - x.len);
        (x, y)
    }

    /// Splits `self` into a tuple (prefix, suffix) of `LookupBits`, where
    /// `suffix.len() == suffix_len`.
    pub fn split(&self, suffix_len: usize) -> (Self, Self) {
        let suffix_bits = self.bits % (1 << suffix_len);
        let suffix = Self::new(suffix_bits, suffix_len);
        let prefix_bits = self.bits >> suffix_len;
        let prefix = Self::new(prefix_bits, self.len - suffix_len);
        (prefix, suffix)
    }

    /// Pops the most significant bit from `self`, decrementing `len`.
    pub fn pop_msb(&mut self) -> u8 {
        let msb = (self.bits >> (self.len - 1)) & 1;
        self.bits %= 1 << (self.len - 1);
        self.len -= 1;
        msb as u8
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn trailing_zeros(&self) -> u32 {
        std::cmp::min(self.bits.trailing_zeros(), self.len as u32)
    }

    pub fn leading_ones(&self) -> u32 {
        self.bits.unbounded_shl(64 - self.len as u32).leading_ones()
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

/// Computes the bit-length of the suffix, for the current (`j`th) round
/// of sumcheck.
pub fn current_suffix_len(log_K: usize, j: usize) -> usize {
    // Number of sumcheck rounds per "phase" of sparse-dense sumcheck.
    let phase_length = log_K / 4;
    // The suffix length is 3/4 * log_K at the beginning and shrinks by
    // log_K / 4 after each phase.
    log_K - (j / phase_length + 1) * phase_length
}

/// Compute the sumcheck prover message in round `j` using the prefix-suffix
/// decomposition. In the first log(K) rounds of sumcheck, while we're
/// binding the "address" variables (and the "cycle" variables remain unbound),
/// the univariate polynomial computed each round is degree 2.
///
/// To see this, observe that:
///   eq(r', j) * ra_1(k_1, j) * ra_2(k_2, j) * ra_3(k_3, j) * ra_4(k_4, j)
/// is multilinear in k, since ra_1, ra_2, ra_3, and ra_4 are polynomials in
/// non-overlapping variables of k (and the eq term doesn't involve k at all).
/// Val(k) is clearly multilinear in k, so the whole summand
///   eq(r', j) (\prod_i ra_i(k_i, j)) * \sum_l flag_l * Val_l(k)
/// is degree 2 in each "address" variable k.
#[tracing::instrument(skip_all)]
fn compute_sumcheck_prover_message<const WORD_SIZE: usize, F: JoltField>(
    prefix_checkpoints: &[PrefixCheckpoint<F>],
    suffix_polys: &[Vec<DensePolynomial<F>>],
    r: &[F],
    j: usize,
) -> [F; 2] {
    let lookup_tables: Vec<_> = LookupTables::<WORD_SIZE>::iter().collect();

    let len = suffix_polys[0][0].len();
    let log_len = len.log_2();

    let r_x = if j % 2 == 1 { r.last().copied() } else { None };

    let (eval_0, eval_2_left, eval_2_right): (F, F, F) = (0..len / 2)
        .into_par_iter()
        .flat_map_iter(|b| {
            let b = LookupBits::new(b as u64, log_len - 1);
            // Evaluate all prefix MLEs with the current variable fixed to c=0
            let prefixes_c0: Vec<_> = Prefixes::iter()
                .map(|prefix| prefix.prefix_mle::<WORD_SIZE, F>(prefix_checkpoints, r_x, 0, b, j))
                .collect();
            // Evaluate all prefix MLEs with the current variable fixed to c=2
            let prefixes_c2: Vec<_> = Prefixes::iter()
                .map(|prefix| prefix.prefix_mle::<WORD_SIZE, F>(prefix_checkpoints, r_x, 2, b, j))
                .collect();

            lookup_tables
                .iter()
                .zip(suffix_polys.iter())
                .map(move |(table, suffixes)| {
                    let suffixes_left: Vec<_> =
                        suffixes.iter().map(|suffix| suffix[b.into()]).collect();
                    let suffixes_right: Vec<_> = suffixes
                        .iter()
                        .map(|suffix| suffix[usize::from(b) + len / 2])
                        .collect();
                    (
                        table.combine(&prefixes_c0, &suffixes_left),
                        table.combine(&prefixes_c2, &suffixes_left),
                        table.combine(&prefixes_c2, &suffixes_right),
                    )
                })
        })
        .reduce(
            || (F::zero(), F::zero(), F::zero()),
            |running, new| (running.0 + new.0, running.1 + new.1, running.2 + new.2),
        );

    [eval_0, eval_2_right + eval_2_right - eval_2_left]
}

#[allow(clippy::type_complexity)]
pub fn prove_sparse_dense_shout<
    const WORD_SIZE: usize,
    F: JoltField,
    ProofTranscript: Transcript,
>(
    trace: &[RV32IMCycle],
    r_cycle: Vec<F>,
    transcript: &mut ProofTranscript,
) -> (
    SumcheckInstanceProof<F, ProofTranscript>,
    F,
    [F; 4],
    Vec<F>,
    Vec<F>,
) {
    let log_K: usize = 2 * WORD_SIZE;
    let log_m = log_K / 4;
    let m = log_m.pow2();

    let T = trace.len();
    let log_T = T.log_2();
    debug_assert_eq!(r_cycle.len(), log_T);

    let num_rounds = log_K + log_T;
    let mut r: Vec<F> = Vec::with_capacity(num_rounds);
    let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::with_capacity(num_rounds);

    let span = tracing::span!(tracing::Level::INFO, "compute lookup indices");
    let _guard = span.enter();
    let lookup_indices: Vec<_> = trace
        .par_iter()
        .map(|cycle| LookupBits::new(LookupQuery::<WORD_SIZE>::to_lookup_index(cycle), log_K))
        .collect();
    drop(_guard);
    drop(span);

    let eq_r_prime_evals = EqPolynomial::evals(&r_cycle);
    let mut u_evals = eq_r_prime_evals.clone();

    let mut prefix_checkpoints: Vec<PrefixCheckpoint<F>> = vec![None.into(); Prefixes::COUNT];
    let mut v = ExpandingTable::new(m);

    let span = tracing::span!(tracing::Level::INFO, "compute rv_claim");
    let _guard = span.enter();
    let rv_claim = trace
        .par_iter()
        .zip(lookup_indices.par_iter())
        .zip(u_evals.par_iter())
        .map(|((cycle, k), u)| {
            let table: Option<LookupTables<WORD_SIZE>> = cycle.lookup_table();
            match table {
                Some(table) => u.mul_u64(table.materialize_entry(k.into())),
                None => F::zero(),
            }
        })
        .sum();
    drop(_guard);
    drop(span);

    let mut previous_claim = rv_claim;

    let mut j: usize = 0;
    let mut ra: Vec<MultilinearPolynomial<F>> = Vec::with_capacity(4);

    let lookup_tables: Vec<_> = LookupTables::<WORD_SIZE>::iter().collect();
    let mut suffix_polys: Vec<Vec<DensePolynomial<F>>> = lookup_tables
        .par_iter()
        .map(|table| {
            table
                .suffixes()
                .par_iter()
                .map(|_| DensePolynomial::new(unsafe_allocate_zero_vec(m)))
                .collect()
        })
        .collect();

    let span = tracing::span!(tracing::Level::INFO, "Compute lookup_indices_by_table");
    let _guard = span.enter();

    let lookup_indices_by_table: Vec<_> = lookup_tables
        .par_iter()
        .map(|table| {
            let table_lookups: Vec<_> = trace
                .iter()
                .zip(lookup_indices.iter())
                .enumerate()
                .filter_map(|(j, (cycle, k))| match cycle.lookup_table() {
                    Some(lookup) => {
                        if LookupTables::<WORD_SIZE>::enum_index(&lookup)
                            == LookupTables::enum_index(table)
                        {
                            Some((j, k))
                        } else {
                            None
                        }
                    }
                    None => None,
                })
                .collect();
            table_lookups
        })
        .collect();

    drop(_guard);
    drop(span);

    for phase in 0..4 {
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
                    *u *= v[k_bound];
                });
        }

        let suffix_len = (3 - phase) * log_m;

        // Initialize suffix poly for each suffix
        let suffix_poly_span = tracing::span!(tracing::Level::INFO, "Compute suffix polys");
        let _suffix_poly_guard = suffix_poly_span.enter();
        lookup_tables
            .par_iter()
            .zip(suffix_polys.par_iter_mut())
            .zip(lookup_indices_by_table.par_iter())
            .for_each(|((table, polys), lookup_indices)| {
                table
                    .suffixes()
                    .par_iter()
                    .zip(polys.par_iter_mut())
                    .for_each(|(suffix, poly)| {
                        if phase != 0 {
                            // Reset polynomial
                            poly.len = m;
                            poly.num_vars = poly.len.log_2();
                            unsafe_zero_slice(&mut poly.Z);
                        }

                        for (j, k) in lookup_indices.iter() {
                            let (prefix_bits, suffix_bits) = k.split(suffix_len);
                            let t = suffix.suffix_mle::<WORD_SIZE>(suffix_bits);
                            if t != 0 {
                                let u = u_evals[*j];
                                poly.Z[prefix_bits % m] += u.mul_u64(t as u64);
                            }
                        }
                    });
            });
        drop(_suffix_poly_guard);
        drop(suffix_poly_span);

        v.reset(F::one());

        for _round in 0..log_m {
            let span = tracing::span!(tracing::Level::INFO, "sparse-dense sumcheck round");
            let _guard = span.enter();

            let univariate_poly_evals = compute_sumcheck_prover_message::<WORD_SIZE, F>(
                &prefix_checkpoints,
                &suffix_polys,
                &r,
                j,
            );

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

            let binding_span = tracing::span!(tracing::Level::INFO, "binding");
            let _binding_guard = binding_span.enter();

            suffix_polys.par_iter_mut().for_each(|polys| {
                polys
                    .par_iter_mut()
                    .for_each(|poly| poly.bind_parallel(r_j, BindingOrder::HighToLow))
            });
            v.update(r_j);

            {
                if r.len() % 2 == 0 {
                    Prefixes::update_checkpoints::<WORD_SIZE, F>(
                        &mut prefix_checkpoints,
                        r[r.len() - 2],
                        r[r.len() - 1],
                        j,
                    );
                }
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
                v[k_bound]
            })
            .collect();
        ra.push(MultilinearPolynomial::from(ra_i));
    }

    drop_in_background_thread(suffix_polys);

    let mut eq_r_prime = MultilinearPolynomial::from(eq_r_prime_evals.clone());

    let span = tracing::span!(
        tracing::Level::INFO,
        "compute combined_instruction_val_poly"
    );
    let _guard = span.enter();

    let prefixes: Vec<PrefixEval<F>> = prefix_checkpoints
        .into_iter()
        .map(|checkpoint| checkpoint.unwrap())
        .collect();

    let mut combined_instruction_val_poly: Vec<F> = unsafe_allocate_zero_vec(T);
    combined_instruction_val_poly
        .par_iter_mut()
        .zip(trace.par_iter())
        .for_each(|(val, step)| {
            let table: Option<LookupTables<WORD_SIZE>> = step.lookup_table();
            if let Some(table) = table {
                let suffixes: Vec<_> = table
                    .suffixes()
                    .iter()
                    .map(|suffix| {
                        F::from_u32(suffix.suffix_mle::<WORD_SIZE>(LookupBits::new(0, 0)))
                    })
                    .collect();
                *val = table.combine(&prefixes, &suffixes);
            }
        });
    let mut combined_instruction_val_poly =
        MultilinearPolynomial::from(combined_instruction_val_poly);

    drop(_guard);
    drop(span);

    let span = tracing::span!(tracing::Level::INFO, "last log(T) sumcheck rounds");
    let _guard = span.enter();

    // TODO(moodlezoup): Implement optimization from Section 6.2.2 "An optimization leveraging small memory size"

    for _round in 0..log_T {
        let span = tracing::span!(tracing::Level::INFO, "Compute univariate poly");
        let _guard = span.enter();

        let univariate_poly_evals: [F; 6] = (0..eq_r_prime.len() / 2)
            .into_par_iter()
            .map(|i| {
                let eq_evals = eq_r_prime.sumcheck_evals(i, 6, BindingOrder::HighToLow);
                let ra_0_evals = ra[0].sumcheck_evals(i, 6, BindingOrder::HighToLow);
                let ra_1_evals = ra[1].sumcheck_evals(i, 6, BindingOrder::HighToLow);
                let ra_2_evals = ra[2].sumcheck_evals(i, 6, BindingOrder::HighToLow);
                let ra_3_evals = ra[3].sumcheck_evals(i, 6, BindingOrder::HighToLow);
                let val_evals =
                    combined_instruction_val_poly.sumcheck_evals(i, 6, BindingOrder::HighToLow);

                std::array::from_fn(|i| {
                    eq_evals[i]
                        * ra_0_evals[i]
                        * ra_1_evals[i]
                        * ra_2_evals[i]
                        * ra_3_evals[i]
                        * val_evals[i]
                })
            })
            .reduce(
                || [F::zero(); 6],
                |running, new| {
                    [
                        running[0] + new[0],
                        running[1] + new[1],
                        running[2] + new[2],
                        running[3] + new[3],
                        running[4] + new[4],
                        running[5] + new[5],
                    ]
                },
            );

        let univariate_poly = UniPoly::from_evals(&[
            univariate_poly_evals[0],
            previous_claim - univariate_poly_evals[0],
            univariate_poly_evals[1],
            univariate_poly_evals[2],
            univariate_poly_evals[3],
            univariate_poly_evals[4],
            univariate_poly_evals[5],
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
            .chain([&mut combined_instruction_val_poly].into_par_iter())
            .chain([&mut eq_r_prime].into_par_iter())
            .for_each(|poly| poly.bind_parallel(r_j, BindingOrder::HighToLow));
    }

    let span = tracing::span!(tracing::Level::INFO, "compute flag claims");
    let _guard = span.enter();

    let r_cycle_prime = &r[r.len() - log_T..];
    let eq_r_cycle_prime = EqPolynomial::evals(r_cycle_prime);

    // Evaluate each flag polynomial on `r_cycle_prime` by computing its
    // dot product with EQ(r_cycle_prime, j)
    let flag_claims: Vec<_> = lookup_indices_by_table
        .into_par_iter()
        .map(|table_lookups| {
            table_lookups
                .into_iter()
                .map(|(j, _)| eq_r_cycle_prime[j])
                .sum::<F>()
        })
        .collect();
    drop(_guard);
    drop(span);

    let ra_claims = [
        ra[0].final_sumcheck_claim(),
        ra[1].final_sumcheck_claim(),
        ra[2].final_sumcheck_claim(),
        ra[3].final_sumcheck_claim(),
    ];

    drop_in_background_thread((combined_instruction_val_poly, eq_r_prime, ra));

    (
        SumcheckInstanceProof::new(compressed_polys),
        rv_claim,
        ra_claims,
        flag_claims,
        eq_r_prime_evals,
    )
}

pub fn verify_sparse_dense_shout<
    const WORD_SIZE: usize,
    F: JoltField,
    ProofTranscript: Transcript,
>(
    proof: &SumcheckInstanceProof<F, ProofTranscript>,
    log_T: usize,
    r_cycle: Vec<F>,
    rv_claim: F,
    ra_claims: [F; 4],
    flag_claims: &[F],
    transcript: &mut ProofTranscript,
) -> Result<(), ProofVerifyError> {
    let log_K = 2 * WORD_SIZE;
    let first_log_K_rounds = SumcheckInstanceProof::new(proof.compressed_polys[..log_K].to_vec());
    let last_log_T_rounds = SumcheckInstanceProof::new(proof.compressed_polys[log_K..].to_vec());
    // The first log(K) rounds' univariate polynomials are degree 2
    let (sumcheck_claim, r_address) = first_log_K_rounds.verify(rv_claim, log_K, 2, transcript)?;
    // The last log(T) rounds' univariate polynomials are degree 6
    let (sumcheck_claim, r_cycle_prime) =
        last_log_T_rounds.verify(sumcheck_claim, log_T, 6, transcript)?;

    let val_evals: Vec<_> = LookupTables::<WORD_SIZE>::iter()
        .map(|table| table.evaluate_mle(&r_address))
        .collect();
    let eq_eval_cycle = EqPolynomial::new(r_cycle).evaluate(&r_cycle_prime);

    assert_eq!(
        eq_eval_cycle
            * ra_claims.iter().product::<F>()
            * flag_claims
                .iter()
                .zip(val_evals.iter())
                .map(|(flag, val)| *flag * val)
                .sum::<F>(),
        sumcheck_claim,
        "Read-checking sumcheck failed"
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::transcript::KeccakTranscript;
    use ark_bn254::Fr;
    use rand::{rngs::StdRng, RngCore, SeedableRng};

    const WORD_SIZE: usize = 8;
    const LOG_T: usize = 8;
    const T: usize = 1 << LOG_T;

    fn random_instruction(rng: &mut StdRng, instruction: &Option<RV32IMCycle>) -> RV32IMCycle {
        let instruction = instruction.unwrap_or_else(|| {
            let index = rng.next_u64() as usize % RV32IMCycle::COUNT;
            RV32IMCycle::iter()
                .enumerate()
                .filter(|(i, _)| *i == index)
                .map(|(_, x)| x)
                .next()
                .unwrap()
        });

        match instruction {
            RV32IMCycle::ADD(cycle) => cycle.random(rng).into(),
            RV32IMCycle::ADDI(cycle) => cycle.random(rng).into(),
            RV32IMCycle::AND(cycle) => cycle.random(rng).into(),
            RV32IMCycle::ANDI(cycle) => cycle.random(rng).into(),
            RV32IMCycle::AUIPC(cycle) => cycle.random(rng).into(),
            RV32IMCycle::BEQ(cycle) => cycle.random(rng).into(),
            RV32IMCycle::BGE(cycle) => cycle.random(rng).into(),
            RV32IMCycle::BGEU(cycle) => cycle.random(rng).into(),
            RV32IMCycle::BLT(cycle) => cycle.random(rng).into(),
            RV32IMCycle::BLTU(cycle) => cycle.random(rng).into(),
            RV32IMCycle::BNE(cycle) => cycle.random(rng).into(),
            RV32IMCycle::FENCE(cycle) => cycle.random(rng).into(),
            RV32IMCycle::JAL(cycle) => cycle.random(rng).into(),
            RV32IMCycle::JALR(cycle) => cycle.random(rng).into(),
            RV32IMCycle::LUI(cycle) => cycle.random(rng).into(),
            RV32IMCycle::LW(cycle) => cycle.random(rng).into(),
            RV32IMCycle::MUL(cycle) => cycle.random(rng).into(),
            RV32IMCycle::MULHU(cycle) => cycle.random(rng).into(),
            RV32IMCycle::OR(cycle) => cycle.random(rng).into(),
            RV32IMCycle::ORI(cycle) => cycle.random(rng).into(),
            RV32IMCycle::SLT(cycle) => cycle.random(rng).into(),
            RV32IMCycle::SLTI(cycle) => cycle.random(rng).into(),
            RV32IMCycle::SLTIU(cycle) => cycle.random(rng).into(),
            RV32IMCycle::SLTU(cycle) => cycle.random(rng).into(),
            RV32IMCycle::SUB(cycle) => cycle.random(rng).into(),
            RV32IMCycle::SW(cycle) => cycle.random(rng).into(),
            RV32IMCycle::XOR(cycle) => cycle.random(rng).into(),
            RV32IMCycle::XORI(cycle) => cycle.random(rng).into(),
            RV32IMCycle::VirtualAdvice(cycle) => cycle.random(rng).into(),
            RV32IMCycle::VirtualAssertEQ(cycle) => cycle.random(rng).into(),
            RV32IMCycle::VirtualAssertHalfwordAlignment(cycle) => cycle.random(rng).into(),
            RV32IMCycle::VirtualAssertLTE(cycle) => cycle.random(rng).into(),
            RV32IMCycle::VirtualAssertValidDiv0(cycle) => cycle.random(rng).into(),
            RV32IMCycle::VirtualAssertValidSignedRemainder(cycle) => cycle.random(rng).into(),
            RV32IMCycle::VirtualAssertValidUnsignedRemainder(cycle) => cycle.random(rng).into(),
            RV32IMCycle::VirtualMove(cycle) => cycle.random(rng).into(),
            RV32IMCycle::VirtualMovsign(cycle) => cycle.random(rng).into(),
            RV32IMCycle::VirtualMULI(cycle) => cycle.random(rng).into(),
            RV32IMCycle::VirtualPow2(cycle) => cycle.random(rng).into(),
            RV32IMCycle::VirtualPow2I(cycle) => cycle.random(rng).into(),
            RV32IMCycle::VirtualShiftRightBitmask(cycle) => cycle.random(rng).into(),
            RV32IMCycle::VirtualShiftRightBitmaskI(cycle) => cycle.random(rng).into(),
            RV32IMCycle::VirtualSRA(cycle) => cycle.random(rng).into(),
            RV32IMCycle::VirtualSRAI(cycle) => cycle.random(rng).into(),
            RV32IMCycle::VirtualSRL(cycle) => cycle.random(rng).into(),
            RV32IMCycle::VirtualSRLI(cycle) => cycle.random(rng).into(),
            _ => RV32IMCycle::NoOp(0),
        }
    }

    fn test_sparse_dense_shout(instruction: Option<RV32IMCycle>) {
        let mut rng = StdRng::seed_from_u64(12345);

        let trace: Vec<_> = (0..T)
            .map(|_| random_instruction(&mut rng, &instruction))
            .collect();

        let mut prover_transcript = KeccakTranscript::new(b"test_transcript");
        let r_cycle: Vec<Fr> = prover_transcript.challenge_vector(LOG_T);

        let (proof, rv_claim, ra_claims, flag_claims, _) =
            prove_sparse_dense_shout::<WORD_SIZE, _, _>(&trace, r_cycle, &mut prover_transcript);

        let mut verifier_transcript = KeccakTranscript::new(b"test_transcript");
        verifier_transcript.compare_to(prover_transcript);
        let r_cycle: Vec<Fr> = verifier_transcript.challenge_vector(LOG_T);
        let verification_result = verify_sparse_dense_shout::<WORD_SIZE, _, _>(
            &proof,
            LOG_T,
            r_cycle,
            rv_claim,
            ra_claims,
            &flag_claims,
            &mut verifier_transcript,
        );
        assert!(
            verification_result.is_ok(),
            "Verification failed with error: {:?}",
            verification_result.err()
        );
    }

    #[test]
    fn test_random_instructions() {
        test_sparse_dense_shout(None);
    }

    #[test]
    fn test_add() {
        test_sparse_dense_shout(Some(RV32IMCycle::ADD(Default::default())));
    }

    #[test]
    fn test_addi() {
        test_sparse_dense_shout(Some(RV32IMCycle::ADDI(Default::default())));
    }

    #[test]
    fn test_and() {
        test_sparse_dense_shout(Some(RV32IMCycle::AND(Default::default())));
    }

    #[test]
    fn test_andi() {
        test_sparse_dense_shout(Some(RV32IMCycle::ANDI(Default::default())));
    }

    #[test]
    fn test_auipc() {
        test_sparse_dense_shout(Some(RV32IMCycle::AUIPC(Default::default())));
    }

    #[test]
    fn test_beq() {
        test_sparse_dense_shout(Some(RV32IMCycle::BEQ(Default::default())));
    }

    #[test]
    fn test_bge() {
        test_sparse_dense_shout(Some(RV32IMCycle::BGE(Default::default())));
    }

    #[test]
    fn test_bgeu() {
        test_sparse_dense_shout(Some(RV32IMCycle::BGEU(Default::default())));
    }

    #[test]
    fn test_blt() {
        test_sparse_dense_shout(Some(RV32IMCycle::BLT(Default::default())));
    }

    #[test]
    fn test_bltu() {
        test_sparse_dense_shout(Some(RV32IMCycle::BLTU(Default::default())));
    }

    #[test]
    fn test_bne() {
        test_sparse_dense_shout(Some(RV32IMCycle::BNE(Default::default())));
    }

    #[test]
    fn test_fence() {
        test_sparse_dense_shout(Some(RV32IMCycle::FENCE(Default::default())));
    }

    #[test]
    fn test_jal() {
        test_sparse_dense_shout(Some(RV32IMCycle::JAL(Default::default())));
    }

    #[test]
    fn test_jalr() {
        test_sparse_dense_shout(Some(RV32IMCycle::JALR(Default::default())));
    }

    #[test]
    fn test_lui() {
        test_sparse_dense_shout(Some(RV32IMCycle::LUI(Default::default())));
    }

    #[test]
    fn test_lw() {
        test_sparse_dense_shout(Some(RV32IMCycle::LW(Default::default())));
    }

    #[test]
    fn test_mul() {
        test_sparse_dense_shout(Some(RV32IMCycle::MUL(Default::default())));
    }

    #[test]
    fn test_mulhu() {
        test_sparse_dense_shout(Some(RV32IMCycle::MULHU(Default::default())));
    }

    #[test]
    fn test_or() {
        test_sparse_dense_shout(Some(RV32IMCycle::OR(Default::default())));
    }

    #[test]
    fn test_ori() {
        test_sparse_dense_shout(Some(RV32IMCycle::ORI(Default::default())));
    }

    #[test]
    fn test_slt() {
        test_sparse_dense_shout(Some(RV32IMCycle::SLT(Default::default())));
    }

    #[test]
    fn test_slti() {
        test_sparse_dense_shout(Some(RV32IMCycle::SLTI(Default::default())));
    }

    #[test]
    fn test_sltiu() {
        test_sparse_dense_shout(Some(RV32IMCycle::SLTIU(Default::default())));
    }

    #[test]
    fn test_sltu() {
        test_sparse_dense_shout(Some(RV32IMCycle::SLTU(Default::default())));
    }

    #[test]
    fn test_sub() {
        test_sparse_dense_shout(Some(RV32IMCycle::SUB(Default::default())));
    }

    #[test]
    fn test_sw() {
        test_sparse_dense_shout(Some(RV32IMCycle::SW(Default::default())));
    }

    #[test]
    fn test_xor() {
        test_sparse_dense_shout(Some(RV32IMCycle::XOR(Default::default())));
    }

    #[test]
    fn test_xori() {
        test_sparse_dense_shout(Some(RV32IMCycle::XORI(Default::default())));
    }

    #[test]
    fn test_advice() {
        test_sparse_dense_shout(Some(RV32IMCycle::VirtualAdvice(Default::default())));
    }

    #[test]
    fn test_asserteq() {
        test_sparse_dense_shout(Some(RV32IMCycle::VirtualAssertEQ(Default::default())));
    }

    #[test]
    fn test_asserthalfwordalignment() {
        test_sparse_dense_shout(Some(RV32IMCycle::VirtualAssertHalfwordAlignment(
            Default::default(),
        )));
    }

    #[test]
    fn test_assertlte() {
        test_sparse_dense_shout(Some(RV32IMCycle::VirtualAssertLTE(Default::default())));
    }

    #[test]
    fn test_assertvaliddiv0() {
        test_sparse_dense_shout(Some(
            RV32IMCycle::VirtualAssertValidDiv0(Default::default()),
        ));
    }

    #[test]
    fn test_assertvalidsignedremainder() {
        test_sparse_dense_shout(Some(RV32IMCycle::VirtualAssertValidSignedRemainder(
            Default::default(),
        )));
    }

    #[test]
    fn test_assertvalidunsignedremainder() {
        test_sparse_dense_shout(Some(RV32IMCycle::VirtualAssertValidUnsignedRemainder(
            Default::default(),
        )));
    }

    #[test]
    fn test_move() {
        test_sparse_dense_shout(Some(RV32IMCycle::VirtualMove(Default::default())));
    }

    #[test]
    fn test_movsign() {
        test_sparse_dense_shout(Some(RV32IMCycle::VirtualMovsign(Default::default())));
    }

    #[test]
    fn test_muli() {
        test_sparse_dense_shout(Some(RV32IMCycle::VirtualMULI(Default::default())));
    }

    #[test]
    fn test_pow2() {
        test_sparse_dense_shout(Some(RV32IMCycle::VirtualPow2(Default::default())));
    }

    #[test]
    fn test_pow2i() {
        test_sparse_dense_shout(Some(RV32IMCycle::VirtualPow2I(Default::default())));
    }

    #[test]
    fn test_shiftrightbitmask() {
        test_sparse_dense_shout(Some(RV32IMCycle::VirtualShiftRightBitmask(
            Default::default(),
        )));
    }

    #[test]
    fn test_shiftrightbitmaski() {
        test_sparse_dense_shout(Some(RV32IMCycle::VirtualShiftRightBitmaskI(
            Default::default(),
        )));
    }

    #[test]
    fn test_virtualrotri() {
        test_sparse_dense_shout(Some(RV32IMCycle::VirtualROTRI(Default::default())));
    }

    #[test]
    fn test_virtualsra() {
        test_sparse_dense_shout(Some(RV32IMCycle::VirtualSRA(Default::default())));
    }

    #[test]
    fn test_virtualsrai() {
        test_sparse_dense_shout(Some(RV32IMCycle::VirtualSRAI(Default::default())));
    }

    #[test]
    fn test_virtualsrl() {
        test_sparse_dense_shout(Some(RV32IMCycle::VirtualSRL(Default::default())));
    }

    #[test]
    fn test_virtualsrli() {
        test_sparse_dense_shout(Some(RV32IMCycle::VirtualSRLI(Default::default())));
    }
}
