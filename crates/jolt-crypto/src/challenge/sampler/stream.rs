//! One transcript message per sparse fold group.
//!
//! A group's `count` challenges are read from one squeezed message of a fixed
//! whole number of `u64` words, so the squeeze stays word aligned. The message
//! is a little-endian bit stream over little-endian `u64` words: bit `i` is bit
//! `i % 64` of word `i / 64`, which is bit `i % 8` of byte `i / 8`. Candidate
//! `j` occupies bits `[j K, (j + 1) K)`, and the bits past the last candidate
//! are ignored.
//!
//! A candidate is a `K`-bit integer `R`. Its low `w = count_pm1 + count_pm2`
//! bits are the signs of the nonzero coefficients in output order (0 is `+`).
//! The remaining `K - w` bits are a position rank `r = r_a + C(d, a) r_b`:
//! `r_a` unranks to the ±1 positions and `r_b` to the ±2 positions among the
//! `d - a` remaining slots, both in colexicographic order. On `[0, |S|)` this
//! is a bijection onto the shell `S` of all such vectors.
//!
//! An unfiltered family uses `K = floor(log2 |S|)`, so every candidate is
//! accepted and the challenges are exactly uniform on the first `2^K` shell
//! elements. An operator-norm family uses `K = ceil(log2 |S|)` over the whole
//! shell: a candidate is accepted when its rank is in range and the certified
//! predicate accepts it, and the first `count` accepted candidates are the
//! challenges. Running out is an error, so the verifier rejects. Whether a
//! candidate is accepted does not depend on the accepted values, so the
//! challenges of a message that does not run out are exactly uniform on the
//! accepted set. The pool size only bounds how often an honest message runs
//! out: at most `2^-161` at the sizing rate below the measured predicate rate.
//! At the certified rate the bound depends on the count `W`: at most `2^-32`
//! for D64 with `W <= 64` and `2^-79` for D128 with `W <= 191`, and no useful
//! bound for D64 past about `W = 200`.

use super::ChallengePolicy;
use crate::challenge::{
    Challenges, OperatorNormRejection, SparseChallenge, SparseChallengeCoefficients,
    SparseChallengeConfig, SparseChallengePositions,
};
use crate::CryptoError;
#[cfg(feature = "parallel")]
use jolt_algebra::parallel::*;
use std::cell::RefCell;
use std::sync::{Arc, Mutex, PoisonError};

/// Largest ring dimension the decoder supports.
const MAX_RING_DIM: usize = 2048;
/// Pool failure target: running out has probability at most `2^-161`.
const POOL_FAILURE_BITS: u128 = 161;
/// `224 >= 2 * 161 * ln 2`, the Chernoff exponent constant for `2^-161`.
const POOL_CHERNOFF_SCALE: u128 = 224;
/// Largest pool the sizing search considers.
const MAX_POOL_CANDIDATES: u128 = 1 << 32;
/// Largest message, matching Jolt's per-message squeeze limit.
const MAX_MESSAGE_BYTES: usize = 1 << 24;
/// Bytes in one message word.
const WORD_BYTES: usize = 8;
/// Bits in one message word.
const WORD_BITS: usize = u64::BITS as usize;
/// Most binomial tables kept for reuse across decodes.
const MAX_CACHED_BINOMIAL_TABLES: usize = 8;
#[cfg(feature = "parallel")]
const PARALLEL_DECODE_WORK_THRESHOLD: usize = 1 << 14;

/// Lower bounds used to size an operator-norm pool.
///
/// `accept` is a rational `u / v` for the probability that one candidate is
/// accepted: the exact in-range rate times a sizing predicate rate about 3.7
/// (D64) or 3.8 (D128) standard errors below the rate measured on 20,000
/// draws. It is an estimate, not a proven bound; the certificate proves only
/// a lower combined rate. `fail_bits` is a rational lower bound on
/// `-log2(1 - u / v)`. A wrong rate changes only how often an honest message
/// runs out, never which challenges are accepted.
struct PoolRates {
    accept: (u128, u128),
    fail_bits: (u128, u128),
}

/// Pool rates and certified accepted-set exponent of one rejection policy.
fn operator_norm_rates(rejection: OperatorNormRejection) -> Result<(PoolRates, u32), CryptoError> {
    match rejection {
        // In-range rate 0.5557, sizing predicate rate 0.65 (measured 0.6623).
        // The certificate retains 2^128.062 accepted challenges, a combined
        // rate of at least 0.1305.
        OperatorNormRejection::D64_SELECTIVE_L2 => Ok((
            PoolRates {
                accept: (3611, 10_000),
                fail_bits: (64, 100),
            },
            128,
        )),
        // In-range rate 0.7692, sizing predicate rate 0.78 (measured 0.7908).
        // The certificate retains 2^128.563 accepted challenges, a combined
        // rate of at least 0.3694.
        OperatorNormRejection::D128_SELECTIVE_L2 => Ok((
            PoolRates {
                accept: (5999, 10_000),
                fail_bits: (132, 100),
            },
            128,
        )),
        _ => Err(CryptoError::InvalidSetup(
            "unsupported operator-norm rejection policy".into(),
        )),
    }
}

/// The count-independent layout of one sparse family: its shell, rank
/// width and guaranteed set size. A family validates it once and then lays
/// out a message for any challenge count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SparseChallengeShell {
    ring_d: usize,
    config: SparseChallengeConfig,
    rejection: Option<OperatorNormRejection>,
    candidate_bits: usize,
    rank_bits: u32,
    rank_space: u128,
    pm1_rank_space: u128,
    set_bits: u32,
}

impl SparseChallengeShell {
    /// Validate one family.
    ///
    /// # Errors
    ///
    /// Returns [`CryptoError::InvalidSetup`] when the family is invalid or
    /// outside the decoder's supported range, the rejection policy does not
    /// cover the family, or a size overflows.
    pub fn new(
        ring_d: usize,
        config: &SparseChallengeConfig,
        rejection: Option<OperatorNormRejection>,
    ) -> Result<Self, CryptoError> {
        // The planner prices millions of folds against a handful of families,
        // and validating one costs two u128 binomials. A shell depends only on
        // its inputs, so each thread keeps the shells it validated.
        thread_local! {
            static SHELLS: RefCell<Vec<SparseChallengeShell>> = const { RefCell::new(Vec::new()) };
        }
        const MAX_CACHED_SHELLS: usize = 32;
        let cached = SHELLS
            .try_with(|shells| {
                shells.try_borrow().ok().and_then(|shells| {
                    shells
                        .iter()
                        .find(|shell| {
                            shell.ring_d == ring_d
                                && shell.config == *config
                                && shell.rejection == rejection
                        })
                        .copied()
                })
            })
            .ok()
            .flatten();
        if let Some(shell) = cached {
            return Ok(shell);
        }
        let shell = Self::validated(ring_d, config, rejection)?;
        // A thread-local being torn down only skips caching.
        SHELLS
            .try_with(|shells| {
                if let Ok(mut shells) = shells.try_borrow_mut() {
                    if shells.len() < MAX_CACHED_SHELLS {
                        shells.push(shell);
                    }
                }
            })
            .unwrap_or(());
        Ok(shell)
    }

    fn validated(
        ring_d: usize,
        config: &SparseChallengeConfig,
        rejection: Option<OperatorNormRejection>,
    ) -> Result<Self, CryptoError> {
        if ring_d == 0 || ring_d > MAX_RING_DIM {
            return Err(CryptoError::InvalidSetup(format!(
                "ring dimension {ring_d} is outside the sparse decoder range 1..={MAX_RING_DIM}"
            )));
        }
        config
            .validate_dyn(ring_d)
            .map_err(|error| CryptoError::InvalidSetup(error.into()))?;
        let pm1_rank_space = binomial(ring_d, config.count_pm1)?;
        let pm2_rank_space = binomial(ring_d - config.count_pm1, config.count_pm2)?;
        let rank_space = pm1_rank_space.checked_mul(pm2_rank_space).ok_or_else(|| {
            CryptoError::InvalidSetup("sparse position rank space overflow".into())
        })?;
        let weight = u32::try_from(config.weight())
            .map_err(|_| CryptoError::InvalidSetup("sparse challenge weight exceeds u32".into()))?;
        let (rank_bits, set_bits) = match rejection {
            None => {
                let rank_bits = u128::BITS - 1 - rank_space.leading_zeros();
                let set_bits = rank_bits.checked_add(weight).ok_or_else(|| {
                    CryptoError::InvalidSetup("sparse challenge width overflow".into())
                })?;
                (rank_bits, set_bits)
            }
            Some(rejection) => {
                rejection
                    .validate(ring_d, config)
                    .map_err(|error| CryptoError::InvalidSetup(error.into()))?;
                let (_, certified_bits) = operator_norm_rates(rejection)?;
                (
                    u128::BITS - (rank_space - 1).leading_zeros(),
                    certified_bits,
                )
            }
        };
        let candidate_bits = usize::try_from(rank_bits)
            .ok()
            .and_then(|bits| bits.checked_add(config.weight()))
            .filter(|&bits| bits > 0)
            .ok_or_else(|| CryptoError::InvalidSetup("sparse challenge width is invalid".into()))?;
        Ok(Self {
            ring_d,
            config: *config,
            rejection,
            candidate_bits,
            rank_bits,
            rank_space,
            pm1_rank_space,
            set_bits,
        })
    }

    pub const fn ring_d(&self) -> usize {
        self.ring_d
    }

    pub const fn config(&self) -> SparseChallengeConfig {
        self.config
    }

    pub const fn rejection(&self) -> Option<OperatorNormRejection> {
        self.rejection
    }

    /// Guaranteed `log2` size of the set each challenge is uniform on: `K` for
    /// an unfiltered family, the certified exponent for an operator-norm one.
    /// It does not depend on the challenge count.
    pub const fn set_bits(&self) -> u32 {
        self.set_bits
    }

    /// Lay out `count` challenges of this family.
    ///
    /// # Errors
    ///
    /// Returns [`CryptoError::InvalidSetup`] when `count` is zero, the pool
    /// size overflows, or the message exceeds the message limit.
    pub fn stream(&self, count: usize) -> Result<SparseChallengeStream, CryptoError> {
        if count == 0 {
            return Err(CryptoError::InvalidSetup(
                "a sparse challenge message needs at least one challenge".into(),
            ));
        }
        let candidates = match self.rejection {
            None => count,
            Some(rejection) => pool_len(count, &operator_norm_rates(rejection)?.0)?,
        };
        let message_len = candidates
            .checked_mul(self.candidate_bits)
            .map(|bits| bits.div_ceil(WORD_BITS) * WORD_BYTES)
            .filter(|&bytes| bytes <= MAX_MESSAGE_BYTES)
            .ok_or_else(|| {
                CryptoError::InvalidSetup(format!(
                    "sparse challenge message exceeds {MAX_MESSAGE_BYTES} bytes"
                ))
            })?;
        Ok(SparseChallengeStream {
            shell: *self,
            count,
            candidates,
            message_len,
        })
    }
}

/// Fixed-length message layout and decoder for one group's sparse challenges.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SparseChallengeStream {
    shell: SparseChallengeShell,
    count: usize,
    candidates: usize,
    message_len: usize,
}

impl SparseChallengeStream {
    /// Exact byte length of the group's challenge message: the candidate bits
    /// rounded up to whole `u64` words.
    #[must_use]
    pub const fn message_len(&self) -> usize {
        self.message_len
    }

    /// Guaranteed `log2` size of the set each challenge is uniform on: `K` for
    /// an unfiltered family, the certified exponent for an operator-norm one.
    #[must_use]
    pub const fn set_bits(&self) -> u32 {
        self.shell.set_bits
    }

    /// Number of candidates in the message: `count` for an unfiltered family,
    /// the pool size for an operator-norm one.
    #[must_use]
    pub const fn candidates(&self) -> usize {
        self.candidates
    }

    /// Bits read per candidate.
    #[must_use]
    pub const fn candidate_bits(&self) -> usize {
        self.shell.candidate_bits
    }

    /// Decode a message into `num_live_blocks * num_claims` claim-major
    /// [`Challenges`] that record this stream's family, or `None` when an
    /// operator-norm pool runs out. This is the only public way to obtain
    /// family-tagged challenges, whose evaluation trusts the sampler's shape.
    ///
    /// # Errors
    ///
    /// Returns [`CryptoError::InvalidInput`] when the block and claim counts
    /// do not multiply to the stream's challenge count, and otherwise as
    /// [`Self::decode`].
    pub fn decode_challenges(
        &self,
        message: &[u8],
        num_live_blocks: usize,
        num_claims: usize,
    ) -> Result<Option<Challenges>, CryptoError> {
        if num_live_blocks.checked_mul(num_claims) != Some(self.count) {
            return Err(CryptoError::InvalidInput(format!(
                "{num_live_blocks} blocks x {num_claims} claims do not match the stream's {} challenges",
                self.count
            )));
        }
        self.decode(message)?
            .map(|challenges| {
                Challenges::from_sampled(
                    challenges,
                    num_live_blocks,
                    num_claims,
                    self.shell.ring_d,
                    self.shell.config,
                )
            })
            .transpose()
    }

    /// Decode a message into `count` challenges, or `None` when an
    /// operator-norm pool runs out.
    ///
    /// A prover treats `None` as a failed candidate and moves on; a verifier
    /// rejects.
    ///
    /// # Errors
    ///
    /// Returns [`CryptoError::InvalidInput`] when the message has the wrong
    /// length, and [`CryptoError::InvalidSetup`] when the predicate table is
    /// unavailable.
    pub fn decode(&self, message: &[u8]) -> Result<Option<Vec<SparseChallenge>>, CryptoError> {
        if message.len() != self.message_len() {
            return Err(CryptoError::InvalidInput(
                "sparse challenge message has the wrong length".into(),
            ));
        }
        let policy =
            ChallengePolicy::new(self.shell.ring_d, &self.shell.config, self.shell.rejection)?;
        let binomials = binomial_table(
            self.shell.ring_d,
            self.shell.config.count_pm1.max(self.shell.config.count_pm2),
        )?;
        let decoder = CandidateDecoder {
            stream: self,
            message,
            binomials: &binomials,
        };
        match policy {
            ChallengePolicy::Unrestricted => self.decode_all(&decoder).map(Some),
            ChallengePolicy::OperatorNorm { threshold, table } => {
                let accept = |index: usize| -> Result<Option<SparseChallenge>, CryptoError> {
                    let Some(challenge) = decoder.candidate(index)? else {
                        return Ok(None);
                    };
                    Ok(table
                        .accept_strict_parts(
                            &challenge.positions,
                            &challenge.coeffs,
                            u64::from(threshold),
                        )?
                        .then_some(challenge))
                };
                self.first_accepted(accept)
            }
        }
    }

    fn decode_all(
        &self,
        decoder: &CandidateDecoder<'_>,
    ) -> Result<Vec<SparseChallenge>, CryptoError> {
        let decode = |index: usize| {
            decoder.candidate(index)?.ok_or_else(|| {
                CryptoError::Internal("an unfiltered sparse candidate was out of range".into())
            })
        };
        #[cfg(feature = "parallel")]
        if self.parallel() {
            return (0..self.count).into_par_iter().map(decode).collect();
        }
        (0..self.count).map(decode).collect()
    }

    fn first_accepted(
        &self,
        accept: impl Fn(usize) -> Result<Option<SparseChallenge>, CryptoError> + Sync,
    ) -> Result<Option<Vec<SparseChallenge>>, CryptoError> {
        let mut accepted = Vec::new();
        accepted
            .try_reserve_exact(self.count)
            .map_err(|_| CryptoError::InvalidInput("sparse challenge allocation failed".into()))?;
        // In parallel, decode the pool in order in chunks of twice the
        // challenge count and stop after the chunk that fills the count.
        #[cfg(feature = "parallel")]
        if self.parallel() {
            let chunk = self.count.saturating_mul(2);
            let mut start = 0;
            while accepted.len() < self.count && start < self.candidates {
                let end = start.saturating_add(chunk).min(self.candidates);
                let decoded = (start..end)
                    .into_par_iter()
                    .map(&accept)
                    .collect::<Result<Vec<_>, _>>()?;
                let missing = self.count - accepted.len();
                accepted.extend(decoded.into_iter().flatten().take(missing));
                start = end;
            }
            return Ok(self.full(accepted));
        }
        for index in 0..self.candidates {
            if accepted.len() == self.count {
                break;
            }
            if let Some(challenge) = accept(index)? {
                accepted.push(challenge);
            }
        }
        Ok(self.full(accepted))
    }

    fn full(&self, accepted: Vec<SparseChallenge>) -> Option<Vec<SparseChallenge>> {
        (accepted.len() == self.count).then_some(accepted)
    }

    /// Whether to decode in parallel, by the challenge count and not the pool
    /// size: a pool stops after a few candidates per challenge however long it
    /// is, and every parallel region wakes the whole thread pool.
    #[cfg(feature = "parallel")]
    fn parallel(&self) -> bool {
        self.count.saturating_mul(self.shell.config.weight()) >= PARALLEL_DECODE_WORK_THRESHOLD
    }
}

/// Least pool size `B >= count` whose run-out probability is at most
/// `2^-161` by either the union bound
/// `(B - W + 1) r >= 161 + (W - 1) bitlen(B)` or the Chernoff bound
/// `(B u - W v)^2 >= 224 B u v` with `B u > W v`.
fn pool_len(count: usize, rates: &PoolRates) -> Result<usize, CryptoError> {
    let overflow = || CryptoError::InvalidSetup("operator-norm pool size overflow".into());
    let (u, v) = rates.accept;
    let (r_num, r_den) = rates.fail_bits;
    let w = u128::try_from(count).map_err(|_| overflow())?;
    let chernoff = |b: u128| -> Option<bool> {
        let bu = b.checked_mul(u)?;
        let wv = w.checked_mul(v)?;
        if bu <= wv {
            return Some(false);
        }
        let gap = bu - wv;
        Some(
            gap.checked_mul(gap)?
                >= POOL_CHERNOFF_SCALE
                    .checked_mul(b)?
                    .checked_mul(u)?
                    .checked_mul(v)?,
        )
    };
    // With `h = 224 / 2`, the Chernoff condition is
    // `u^2 B^2 - 2 u v (W + h) B + W^2 v^2 >= 0` on `B u > W v`, which holds
    // exactly from its larger root `v (W + h + sqrt(h (2 W + h))) / u` on.
    // Start from that root rounded down and step to the least `B >= W`.
    let half_scale = POOL_CHERNOFF_SCALE / 2;
    let root = w
        .checked_mul(2)
        .and_then(|twice| twice.checked_add(half_scale))
        .and_then(|sum| sum.checked_mul(half_scale))
        .map(u128::isqrt)
        .and_then(|sqrt| w.checked_add(half_scale)?.checked_add(sqrt))
        .and_then(|sum| sum.checked_mul(v))
        .and_then(|scaled| scaled.checked_div(u))
        .ok_or_else(overflow)?;
    let mut least = root.max(w);
    while least > w && chernoff(least - 1).ok_or_else(overflow)? {
        least -= 1;
    }
    while !chernoff(least).ok_or_else(overflow)? {
        least = least.checked_add(1).ok_or_else(overflow)?;
    }
    // The pool may not pass the cap reached by doubling from `W` to a pool
    // that meets the Chernoff bound.
    let mut doubled = w.max(1);
    while doubled < least {
        doubled = doubled
            .checked_mul(2)
            .filter(|&b| b <= MAX_POOL_CANDIDATES)
            .ok_or_else(overflow)?;
    }
    // The union bound's right side steps up with `bitlen(B)`, so it can hold
    // and then fail again. Within one bit length it holds exactly from
    // `B = W - 1 + ceil((161 + (W - 1) L) r_den / r_num)` on, so the least
    // `B` in `[W, least)` is the first class's bound that lands in its class.
    let pool = if w == 0 {
        least
    } else {
        let bit_len = |b: u128| u128::BITS - b.leading_zeros();
        let mut pool = least;
        for len in bit_len(w)..=bit_len(least - 1) {
            let class_low = w.max(1 << (len - 1));
            let class_high = ((1u128 << len) - 1).min(least - 1);
            let tail = (w - 1)
                .checked_mul(u128::from(len))
                .and_then(|bits| bits.checked_add(POOL_FAILURE_BITS))
                .and_then(|bits| bits.checked_mul(r_den))
                .and_then(|need| need.checked_add(r_num.checked_sub(1)?)?.checked_div(r_num))
                .ok_or_else(overflow)?;
            let first = (w - 1)
                .checked_add(tail)
                .ok_or_else(overflow)?
                .max(class_low);
            if first <= class_high {
                pool = first;
                break;
            }
        }
        pool
    };
    usize::try_from(pool).map_err(|_| overflow())
}

/// `C(n, k)`, or an error when it does not fit in `u128`.
fn binomial(n: usize, k: usize) -> Result<u128, CryptoError> {
    if k > n {
        return Ok(0);
    }
    let k = k.min(n - k);
    let mut value = 1u128;
    for i in 0..k {
        // `value * (n - i)` is divisible by `i + 1` at every step.
        value = value
            .checked_mul((n - i) as u128)
            .map(|product| product / (i as u128 + 1))
            .ok_or_else(|| CryptoError::InvalidSetup("sparse binomial overflow".into()))?;
    }
    Ok(value)
}

/// The binomial table for `n <= max_n`, `k <= max_k`, built once and then
/// shared by every decode of a family with that ring dimension and weight.
fn binomial_table(max_n: usize, max_k: usize) -> Result<Arc<BinomialTable>, CryptoError> {
    static TABLES: Mutex<Vec<Arc<BinomialTable>>> = Mutex::new(Vec::new());
    let mut tables = TABLES.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some(table) = tables
        .iter()
        .find(|table| table.max_n == max_n && table.max_k == max_k)
    {
        return Ok(Arc::clone(table));
    }
    let table = Arc::new(BinomialTable::new(max_n, max_k)?);
    if tables.len() < MAX_CACHED_BINOMIAL_TABLES {
        tables.push(Arc::clone(&table));
    }
    Ok(table)
}

/// Pascal's triangle `C(n, k)` for `n <= max_n`, `k <= max_k`, saturating at
/// `u128::MAX`, stored by column so that `C(0..=max_n, k)` is contiguous.
/// Every rank the decoder unranks is below `u128::MAX`, so a saturated entry
/// still compares correctly as "too large".
struct BinomialTable {
    max_n: usize,
    max_k: usize,
    entries: Vec<u128>,
}

impl BinomialTable {
    fn new(max_n: usize, max_k: usize) -> Result<Self, CryptoError> {
        let rows = max_n + 1;
        let len = rows
            .checked_mul(max_k + 1)
            .ok_or_else(|| CryptoError::InvalidSetup("sparse binomial table overflow".into()))?;
        let mut entries: Vec<u128> = Vec::new();
        entries.try_reserve_exact(len).map_err(|_| {
            CryptoError::InvalidInput("sparse binomial table allocation failed".into())
        })?;
        entries.resize(len, 0);
        let mut columns = entries.chunks_exact_mut(rows);
        if let Some(mut previous) = columns.next() {
            previous.fill(1);
            for column in columns {
                // `C(n, k) = C(n - 1, k) + C(n - 1, k - 1)`, with `C(0, k) = 0`.
                let mut above = 0u128;
                for (entry, &diagonal) in column.iter_mut().skip(1).zip(previous.iter()) {
                    above = above.saturating_add(diagonal);
                    *entry = above;
                }
                previous = column;
            }
        }
        Ok(Self {
            max_n,
            max_k,
            entries,
        })
    }

    /// `C(0..upper, k)`.
    fn column(&self, k: usize, upper: usize) -> Result<&[u128], CryptoError> {
        let rows = self.max_n + 1;
        (k <= self.max_k && upper <= rows)
            .then(|| self.entries.get(k * rows..k * rows + upper))
            .flatten()
            .ok_or_else(|| CryptoError::Internal("sparse binomial table is too small".into()))
    }

    /// Unrank `rank < C(n, k)` into a `k`-subset of `0..n`, colexicographic,
    /// written in increasing order into `out`.
    ///
    /// Element `i` (from 1) is the largest `c` below element `i + 1` with
    /// `C(c, i) <= rank`. The elements decrease, so one downward scan over all
    /// of `0..n` finds them all; it beats a binary search per element, whose
    /// comparisons the CPU cannot predict.
    fn unrank(
        &self,
        mut rank: u128,
        n: usize,
        k: usize,
        out: &mut [u32],
    ) -> Result<(), CryptoError> {
        let out = out
            .get_mut(..k)
            .ok_or_else(|| CryptoError::Internal("sparse unrank output is too short".into()))?;
        let mut upper = n;
        for (i, slot) in out.iter_mut().enumerate().rev() {
            let column = self.column(i + 1, upper)?;
            let (element, &value) = column
                .iter()
                .enumerate()
                .rfind(|&(_, &value)| value <= rank)
                .ok_or_else(|| CryptoError::Internal("sparse rank was out of range".into()))?;
            rank -= value;
            *slot = u32::try_from(element)
                .map_err(|_| CryptoError::Internal("sparse position exceeds u32".into()))?;
            upper = element;
        }
        if rank != 0 {
            return Err(CryptoError::Internal("sparse rank was out of range".into()));
        }
        Ok(())
    }
}

struct CandidateDecoder<'a> {
    stream: &'a SparseChallengeStream,
    message: &'a [u8],
    binomials: &'a BinomialTable,
}

impl CandidateDecoder<'_> {
    /// Candidate `index`, or `None` when its position rank is out of range.
    fn candidate(&self, index: usize) -> Result<Option<SparseChallenge>, CryptoError> {
        let stream = self.stream;
        let (a, b) = (stream.shell.config.count_pm1, stream.shell.config.count_pm2);
        let weight = a + b;
        let start = index
            .checked_mul(stream.shell.candidate_bits)
            .ok_or_else(|| CryptoError::Internal("sparse candidate offset overflow".into()))?;
        let rank_start = start + weight;
        let rank = read_bits(self.message, rank_start, stream.shell.rank_bits)?;
        if rank >= stream.shell.rank_space {
            return Ok(None);
        }
        // One `u128` division, skipped for a ±1-only family.
        let pm2_rank = if b == 0 {
            0
        } else {
            rank / stream.shell.pm1_rank_space
        };
        let pm1_rank = rank - pm2_rank * stream.shell.pm1_rank_space;
        let mut positions = SparseChallengePositions::from_elem(0, weight);
        let (pm1, pm2) = positions.split_at_mut(a);
        self.binomials
            .unrank(pm1_rank, stream.shell.ring_d, a, pm1)?;
        if b > 0 {
            self.binomials
                .unrank(pm2_rank, stream.shell.ring_d - a, b, pm2)?;
            map_to_complement(pm1, pm2, stream.shell.ring_d)?;
        }
        let mut coeffs = SparseChallengeCoefficients::with_capacity(weight);
        for offset in (0..weight).step_by(u64::BITS as usize) {
            let take = (weight - offset).min(u64::BITS as usize);
            let signs = read_bits(self.message, start + offset, take as u32)?;
            for bit in 0..take {
                // Branch-free: the sign bits are random, so a branch on them
                // mispredicts half the time.
                let magnitude = 1 + i8::from(offset + bit >= a);
                let negative = ((signs >> bit) & 1) as i8;
                coeffs.push(magnitude - 2 * negative * magnitude);
            }
        }
        Ok(Some(SparseChallenge { positions, coeffs }))
    }
}

/// Replace each increasing index in `slots` by the index-th position, in
/// increasing order, of `0..ring_d` that is not in the increasing set `taken`.
fn map_to_complement(taken: &[u32], slots: &mut [u32], ring_d: usize) -> Result<(), CryptoError> {
    // Positions of `taken` passed so far; the slots increase, so it only grows.
    let mut skipped = 0usize;
    for slot in slots.iter_mut() {
        let mut position = *slot as usize + skipped;
        while taken
            .get(skipped)
            .is_some_and(|&taken| taken as usize <= position)
        {
            skipped += 1;
            position += 1;
        }
        if position >= ring_d {
            return Err(CryptoError::Internal(
                "sparse complement slot out of range".into(),
            ));
        }
        *slot = u32::try_from(position)
            .map_err(|_| CryptoError::Internal("sparse position exceeds u32".into()))?;
    }
    Ok(())
}

/// Read `width <= 128` bits starting at bit `offset` of a bit stream over
/// little-endian `u64` words.
fn read_bits(message: &[u8], offset: usize, width: u32) -> Result<u128, CryptoError> {
    let past_end = || CryptoError::Internal("sparse message read past its end".into());
    let word = |index: usize| {
        message
            .get(index * WORD_BYTES..(index + 1) * WORD_BYTES)
            .and_then(|bytes| <[u8; WORD_BYTES]>::try_from(bytes).ok())
            .map(|bytes| u128::from(u64::from_le_bytes(bytes)))
            .ok_or_else(past_end)
    };
    if width > u128::BITS {
        return Err(past_end());
    }
    if width == 0 {
        return Ok(0);
    }
    let first = offset / WORD_BITS;
    let last = (offset + width as usize - 1) / WORD_BITS;
    let shift = offset % WORD_BITS;
    let mut value = word(first)? >> shift;
    // A read of up to 128 bits spans three words only when it starts mid-word,
    // so both shifts below are below 128.
    if last > first {
        value |= word(first + 1)? << (WORD_BITS - shift);
    }
    if last > first + 1 {
        value |= word(first + 2)? << (2 * WORD_BITS - shift);
    }
    if width < u128::BITS {
        value &= (1u128 << width) - 1;
    }
    Ok(value)
}

#[cfg(test)]
mod tests;
