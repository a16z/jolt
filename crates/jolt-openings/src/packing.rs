//! Prefix packing for multiple logical multilinear polynomials in one PCS
//! commitment, following Peshawaria, *Batching Evaluation Claims Without
//! Homomorphic Commitments*.
//!
//! # Why pack
//!
//! A multilinear PIOP ends with evaluation claims `f_i(γ_i) = v_i` on `c`
//! committed polynomials of varying sizes, all of which must be opened
//! against commitments. Committing to each polynomial separately pays
//! per-commitment cryptographic cost `c` times over; padding everything to a
//! common `n_max`-variable polynomial costs the prover `c · 2^n_max` instead
//! of `Σ_i 2^n_i`. Packing avoids both: every logical polynomial lives inside
//! one dense committed polynomial sized proportionally to the total data, and
//! all claims are settled with a single opening proof.
//!
//! # Packing via prefix codes
//!
//! Given `c` multilinears `f_1, …, f_c` with `f_i` on `n_i` variables, the
//! packed polynomial `f` has `n := ceil_log2(Σ_i 2^n_i)` variables — at most
//! a 2× blowup of the total coefficient count. Each `f_i` receives a Boolean
//! prefix `b_i ∈ {0,1}^(n - n_i)` such that `{b_1, …, b_c}` is prefix-free;
//! Kraft's inequality (`Σ_i 2^-(n - n_i) ≤ 1`, guaranteed by the choice of
//! `n`) says such a code exists. Prefix-freeness makes the subcubes
//! `{(b_i, x) | x ∈ {0,1}^(n_i)}` pairwise disjoint, so
//!
//! ```text
//! f(z) = f_i(x)  if z = b_i || x for some i,    f(z) = 0  otherwise
//! ```
//!
//! is well defined on the Boolean cube. The committed polynomial is its
//! multilinear extension, and `f_i(x) = f(b_i || x)` holds for every Boolean
//! `x`.
//!
//! Code assignment is deterministic and greedy: sort polynomials by
//! descending arity, ties by `Id: Ord`, and advance an integer cursor through
//! the packed domain. A polynomial with `n_i` variables takes the prefix with
//! value `cursor >> n_i` and length `n - n_i`, its evaluations land at packed
//! indices `(prefix_value << n_i) | local_index` (prefix bits high-to-low),
//! and the cursor advances by `2^n_i`. Descending sizes keep the cursor
//! power-of-two aligned; strictly increasing code values make the code
//! prefix-free.
//!
//! # Reducing `c` claims to one opening
//!
//! Each logical claim `f_i(γ_i) = v_i` — the points `γ_i` may all differ —
//! is the Boolean-cube identity
//!
//! ```text
//! v_i = Σ_{z ∈ {0,1}^n} eq(z, b_i || γ_i) · f(z),
//! ```
//!
//! because `eq(z, b_i || γ_i)` factors as `eq(z_prefix, b_i) ·
//! eq(z_suffix, γ_i)`: the prefix factor selects slot `i`'s subcube and the
//! suffix factor is the multilinear-evaluation kernel inside it. The `c`
//! sumchecks for these identities run in parallel with shared round
//! challenges, batched into a single sumcheck by a random linear combination:
//! after the statement (commitment, prefixes, points, values) is bound to the
//! transcript, the verifier samples powers `α` and one `n`-round degree-2
//! sumcheck proves
//!
//! ```text
//! Σ_i α_i · v_i = Σ_{z ∈ {0,1}^n} E(z) · f(z),    E(z) := Σ_i α_i · eq(z, b_i || γ_i).
//! ```
//!
//! After the rounds the verifier holds the claim `E(r) · f(r)` at the
//! challenge point `r`. It evaluates `E(r)` itself — `c` short eq products —
//! and checks the claimed `f(r)` with one native PCS opening. The opening
//! point consists entirely of fresh verifier randomness.
//!
//! Soundness is the standard argument: if some `v_i` is false, the batched
//! input claim differs from the true sum except with probability
//! `(c-1)/|F|` over `α`, a sumcheck on a false claim survives with
//! probability at most `2n/|F|`, and the final evaluation of `f` is bound by
//! the PCS opening.
//!
//! The prover runs the reduction densely — materializing `E` and `f` over
//! the packed Boolean cube — or, for unit-valued one-hot witnesses, sparsely
//! from the one positions and the factored per-slot structure of `E`,
//! producing field-identical round polynomials without any table sized
//! `2^n`.

use std::{
    collections::{btree_map::Iter, BTreeMap, BTreeSet},
    fmt::Debug,
    ops::Index,
};

use jolt_field::Field;
use jolt_poly::{
    boolean_bits_msb, eq_index_msb, math::Math, thread::unsafe_allocate_zero_vec, EqPolynomial,
    Polynomial,
};
use jolt_transcript::{AppendToTranscript, Label, LabelWithCount, Transcript, U64Word};
use serde::{Deserialize, Serialize};

use crate::{CommitmentScheme, EvaluationClaim, OpeningsError};

/// Logical polynomial declaration used to build a deterministic packing.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PackedPolynomial<Id> {
    pub id: Id,
    pub num_vars: usize,
}

impl<Id> From<(Id, usize)> for PackedPolynomial<Id> {
    fn from((id, num_vars): (Id, usize)) -> Self {
        Self { id, num_vars }
    }
}

/// Physical slot assigned to one logical polynomial.
///
/// The prefix is stored as high-to-low Boolean bits. For a slot with `n`
/// logical variables, local Boolean index `j` is placed at packed index
/// `(prefix_value << n) | j`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrefixSlot {
    pub prefix: Vec<bool>,
    pub num_vars: usize,
}

impl PrefixSlot {
    fn new(prefix_value: usize, prefix_len: usize, num_vars: usize) -> Self {
        Self {
            prefix: boolean_bits_msb(prefix_len, prefix_value),
            num_vars,
        }
    }

    /// Prefix bits as an integer (MSB-first).
    pub fn prefix_index(&self) -> usize {
        self.prefix
            .iter()
            .fold(0usize, |acc, bit| (acc << 1) | usize::from(*bit))
    }

    /// Packed evaluation index of this slot's cell `local` — where witness
    /// assembly places the cell in the packed table.
    pub fn packed_index(&self, local: usize) -> usize {
        debug_assert!(local < 1usize << self.num_vars);
        (self.prefix_index() << self.num_vars) | local
    }
}

/// Deterministic prefix-free assignment from logical ids to packed slots.
///
/// A logical claim `P_i(point_i) = v_i` on a slot with prefix `b_i` is
/// equivalent to `W(b_i || point_i) = v_i`, i.e. the Boolean-cube sum
/// `v_i = Σ_z eq(z, b_i || point_i) · W(z)`. Batch openings prove all such
/// sums with one claim-reduction sumcheck (see
/// [`PackedBatchProof`]), reducing every logical claim — at arbitrary,
/// mutually independent points — to a single opening of `W` at a point made
/// entirely of fresh verifier challenges.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(
    serialize = "Id: Serialize",
    deserialize = "Id: Ord + Deserialize<'de>"
))]
#[serde(deny_unknown_fields)]
pub struct PrefixPacking<Id> {
    pub packed_num_vars: usize,
    pub(crate) slots: BTreeMap<Id, PrefixSlot>,
}

impl<Id> PrefixPacking<Id> {
    pub fn iter(&self) -> Iter<'_, Id, PrefixSlot> {
        self.slots.iter()
    }
}

impl<Id> PrefixPacking<Id>
where
    Id: Clone + Ord,
{
    /// Builds the canonical prefix assignment for the given logical polynomials.
    pub fn new<P>(polynomials: impl IntoIterator<Item = P>) -> Result<Self, OpeningsError>
    where
        P: Into<PackedPolynomial<Id>>,
    {
        let mut polynomials = polynomials.into_iter().map(Into::into).collect::<Vec<_>>();
        if polynomials.is_empty() {
            return Err(OpeningsError::InvalidSetup(
                "prefix packing requires at least one polynomial".to_owned(),
            ));
        }

        let mut total_coefficients = 0usize;
        for polynomial in &polynomials {
            let coefficients = coefficient_count(polynomial.num_vars)?;
            total_coefficients = total_coefficients
                .checked_add(coefficients)
                .ok_or_else(|| {
                    OpeningsError::InvalidSetup("prefix packing domain size overflow".to_owned())
                })?;
        }
        let packed_num_vars = total_coefficients.log_2();
        let packed_coefficients = coefficient_count(packed_num_vars)?;

        polynomials.sort_by(|left, right| {
            right
                .num_vars
                .cmp(&left.num_vars)
                .then_with(|| left.id.cmp(&right.id))
        });

        let mut cursor = 0usize;
        let mut slots = BTreeMap::new();
        for polynomial in polynomials {
            let coefficients = coefficient_count(polynomial.num_vars)?;
            if !cursor.is_multiple_of(coefficients) {
                return Err(OpeningsError::InvalidSetup(
                    "prefix packing cursor is not aligned to polynomial domain".to_owned(),
                ));
            }
            let prefix_len = packed_num_vars
                .checked_sub(polynomial.num_vars)
                .ok_or_else(|| {
                    OpeningsError::InvalidSetup(
                        "logical polynomial has too many variables".to_owned(),
                    )
                })?;
            let prefix_value = cursor >> polynomial.num_vars;
            let slot = PrefixSlot::new(prefix_value, prefix_len, polynomial.num_vars);
            if slots.insert(polynomial.id, slot).is_some() {
                return Err(OpeningsError::InvalidSetup(
                    "duplicate packed polynomial id".to_owned(),
                ));
            }
            cursor += coefficients;
        }

        debug_assert!(cursor <= packed_coefficients);
        Ok(Self {
            packed_num_vars,
            slots,
        })
    }

    /// Returns the number of prefix challenges needed before a logical point
    /// with `logical_num_vars` variables.
    pub fn prefix_challenge_len(&self, logical_num_vars: usize) -> Result<usize, OpeningsError> {
        self.packed_num_vars
            .checked_sub(logical_num_vars)
            .ok_or_else(|| {
                OpeningsError::InvalidBatch(format!(
                    "logical point has {logical_num_vars} variables but packed witness has {}",
                    self.packed_num_vars
                ))
            })
    }

    /// Forms the physical point `prefix_point || logical_point`.
    pub fn pack_point<F: Field>(
        &self,
        prefix_point: &[F],
        logical_point: &[F],
    ) -> Result<Vec<F>, OpeningsError> {
        let prefix_len = self.prefix_challenge_len(logical_point.len())?;
        if prefix_point.len() != prefix_len {
            return Err(OpeningsError::InvalidBatch(format!(
                "prefix point has {} variables but logical arity requires {prefix_len}",
                prefix_point.len()
            )));
        }
        let mut point = Vec::with_capacity(self.packed_num_vars);
        point.extend_from_slice(prefix_point);
        point.extend_from_slice(logical_point);
        Ok(point)
    }

    /// Extracts the logical suffix point for `id` from a full packed point.
    pub fn logical_point<F: Field>(
        &self,
        id: &Id,
        packed_point: &[F],
    ) -> Result<Vec<F>, OpeningsError> {
        let slot = self.slots.get(id).ok_or_else(|| {
            OpeningsError::InvalidBatch("unknown packed polynomial id".to_owned())
        })?;
        if packed_point.len() != self.packed_num_vars {
            return Err(OpeningsError::InvalidBatch(format!(
                "packed point has {} variables but packing has {}",
                packed_point.len(),
                self.packed_num_vars
            )));
        }
        Ok(packed_point[slot.prefix.len()..].to_vec())
    }

    pub(crate) fn prepare_statement<'a, F, C>(
        &'a self,
        statement: &'a PrefixPackedStatement<F, Id, C>,
    ) -> Result<PreparedPrefixPackedStatement<'a, F, C>, OpeningsError>
    where
        F: Field,
        Id: Debug,
    {
        let claims = statement.claims.as_slice();
        if claims.is_empty() {
            return Err(OpeningsError::InvalidBatch(
                "packing opening requires at least one claim".to_owned(),
            ));
        }

        let mut seen = BTreeSet::new();
        let mut ordered_claims = Vec::with_capacity(claims.len());

        for (id, evaluation) in claims {
            let slot = self.slots.get(id).ok_or_else(|| {
                OpeningsError::InvalidBatch(format!("unknown packed polynomial id: {id:?}"))
            })?;
            if evaluation.point.len() != slot.num_vars {
                return Err(OpeningsError::InvalidBatch(format!(
                    "claim for packed polynomial id {id:?} has point arity {} but slot has {} variables",
                    evaluation.point.len(),
                    slot.num_vars
                )));
            }
            if !seen.insert(id.clone()) {
                return Err(OpeningsError::InvalidBatch(format!(
                    "duplicate claim for packed polynomial id {id:?}"
                )));
            }
            ordered_claims.push((id, evaluation, slot));
        }

        if let Some(missing) = self.slots.keys().find(|id| !seen.contains(*id)) {
            return Err(OpeningsError::InvalidBatch(format!(
                "missing claim for packed polynomial id {missing:?}"
            )));
        }

        ordered_claims.sort_by_key(|(id, _, _)| *id);

        Ok(PreparedPrefixPackedStatement {
            packed_num_vars: self.packed_num_vars,
            commitment: &statement.commitment,
            ordered_claims: ordered_claims
                .into_iter()
                .map(|(_, evaluation, slot)| (evaluation, slot))
                .collect(),
        })
    }
}

impl<'a, Id> IntoIterator for &'a PrefixPacking<Id> {
    type Item = (&'a Id, &'a PrefixSlot);
    type IntoIter = Iter<'a, Id, PrefixSlot>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<Id> Index<&Id> for PrefixPacking<Id>
where
    Id: Ord,
{
    type Output = PrefixSlot;

    fn index(&self, id: &Id) -> &Self::Output {
        &self.slots[id]
    }
}

/// A batch statement for opening one packed witness commitment: one logical
/// claim `(id, evaluation)` per packed polynomial.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrefixPackedStatement<F, Id, C> {
    pub commitment: C,
    pub claims: Vec<(Id, EvaluationClaim<F>)>,
}

impl<F, Id, C> PrefixPackedStatement<F, Id, C> {
    pub fn new(commitment: C, claims: impl Into<Vec<(Id, EvaluationClaim<F>)>>) -> Self {
        Self {
            commitment,
            claims: claims.into(),
        }
    }
}

pub(crate) struct PreparedPrefixPackedStatement<'a, F: Field, C> {
    packed_num_vars: usize,
    pub(crate) commitment: &'a C,
    ordered_claims: Vec<(&'a EvaluationClaim<F>, &'a PrefixSlot)>,
}

impl<F, C> PreparedPrefixPackedStatement<'_, F, C>
where
    F: Field,
{
    pub(crate) fn num_claims(&self) -> usize {
        self.ordered_claims.len()
    }

    /// Batched sumcheck input claim `Σ_i α_i · v_i`.
    pub(crate) fn batched_claim(&self, alpha: &[F]) -> F {
        debug_assert_eq!(self.ordered_claims.len(), alpha.len());
        self.ordered_claims
            .iter()
            .zip(alpha)
            .fold(F::zero(), |acc, ((evaluation, _), alpha_i)| {
                acc + *alpha_i * evaluation.value
            })
    }

    /// Boolean-cube table of the batched selector
    /// `E(z) = Σ_i α_i · eq(z, prefix_i || point_i)`.
    ///
    /// Slot subcubes are disjoint, so each claim writes `α_i`-scaled eq
    /// evaluations of its logical point into its own index range; every other
    /// entry is zero.
    pub(crate) fn selector_table(&self, alpha: &[F]) -> Vec<F> {
        debug_assert_eq!(self.ordered_claims.len(), alpha.len());
        let mut table = unsafe_allocate_zero_vec(1usize << self.packed_num_vars);
        for ((evaluation, slot), alpha_i) in self.ordered_claims.iter().zip(alpha) {
            let offset = slot.packed_index(0);
            let evals = EqPolynomial::evals(evaluation.point.as_slice(), Some(*alpha_i));
            table[offset..offset + evals.len()].copy_from_slice(&evals);
        }
        table
    }

    /// Evaluates the batched selector `E` at an arbitrary packed point.
    pub(crate) fn selector_eval(&self, alpha: &[F], packed_point: &[F]) -> F {
        debug_assert_eq!(self.ordered_claims.len(), alpha.len());
        self.ordered_claims.iter().zip(alpha).fold(
            F::zero(),
            |acc, ((evaluation, slot), alpha_i)| {
                let prefix_len = slot.prefix.len();
                acc + *alpha_i
                    * eq_index_msb(&packed_point[..prefix_len], slot.prefix_index() as u128)
                    * EqPolynomial::<F>::mle(
                        &packed_point[prefix_len..],
                        evaluation.point.as_slice(),
                    )
            },
        )
    }

    /// Sparse variant of [`prove_reduction_sumcheck`] for unit-valued one-hot
    /// witnesses: field-identical round polynomials, challenges, and opening
    /// evaluation, without materializing the `2^packed_num_vars` selector and
    /// witness tables.
    ///
    /// The witness is carried as `(packed index, weight)` pairs whose weight
    /// accumulates the bound-variable factor (`1 - r` or `r`) each round; the
    /// selector is evaluated per one-position from the factored per-slot
    /// state (see [`SparseSelectorSlot`]). Once the remaining domain is no
    /// larger than the number of one-positions, the (now small) tables are
    /// materialized and the remaining rounds run the dense prover.
    pub(crate) fn prove_reduction_sumcheck_sparse<T>(
        &self,
        alpha: &[F],
        one_positions: Vec<usize>,
        transcript: &mut T,
    ) -> (Vec<[F; 3]>, Vec<F>, F)
    where
        T: Transcript<Challenge = F>,
    {
        debug_assert_eq!(self.ordered_claims.len(), alpha.len());
        debug_assert!(one_positions
            .iter()
            .all(|index| index >> self.packed_num_vars == 0));

        let mut slots: Vec<SparseSelectorSlot<'_, F>> = self
            .ordered_claims
            .iter()
            .zip(alpha)
            .map(|((evaluation, slot), alpha_i)| {
                SparseSelectorSlot::new(*alpha_i, slot, evaluation.point.as_slice())
            })
            .collect();
        let mut positions: Vec<(usize, F)> = one_positions
            .into_iter()
            .map(|index| (index, F::one()))
            .collect();

        let num_rounds = self.packed_num_vars;
        let mut round_polynomials = Vec::with_capacity(num_rounds);
        let mut point = Vec::with_capacity(num_rounds);

        for bound in 0..num_rounds {
            let remaining_vars = num_rounds - bound;
            if (1usize << remaining_vars) <= positions.len() {
                let (selector, witness) =
                    materialize_remaining(&slots, &positions, bound, remaining_vars);
                let (tail_polynomials, tail_point, opening_eval) =
                    prove_reduction_sumcheck(selector, witness, transcript);
                round_polynomials.extend(tail_polynomials);
                point.extend(tail_point);
                return (round_polynomials, point, opening_eval);
            }

            let half = 1usize << (remaining_vars - 1);
            let round_slots: Vec<RoundSelectorSlot<F>> = slots
                .iter()
                .map(|slot| RoundSelectorSlot::new(slot, bound, remaining_vars))
                .collect();

            let mut eval_zero = F::zero();
            let mut eval_one = F::zero();
            let mut quadratic = F::zero();
            for &(index, weight) in &positions {
                let low_index = index & (half - 1);
                let selector_low = selector_value_at(&round_slots, low_index);
                let selector_high = selector_value_at(&round_slots, low_index | half);
                let cross = weight * (selector_high - selector_low);
                if index & half == 0 {
                    eval_zero += weight * selector_low;
                    quadratic -= cross;
                } else {
                    eval_one += weight * selector_high;
                    quadratic += cross;
                }
            }

            let coefficients = [eval_zero, eval_one - eval_zero - quadratic, quadratic];
            append_round_polynomial(&coefficients, transcript);
            let challenge: F = transcript.challenge_scalar();
            point.push(challenge);
            for slot in &mut slots {
                slot.bind(bound, challenge);
            }
            let one_minus_challenge = F::one() - challenge;
            for (index, weight) in &mut positions {
                *weight *= if *index & half == 0 {
                    one_minus_challenge
                } else {
                    challenge
                };
            }
            round_polynomials.push(coefficients);
        }

        let opening_eval = positions
            .iter()
            .fold(F::zero(), |acc, &(_, weight)| acc + weight);
        (round_polynomials, point, opening_eval)
    }
}

impl<F, C> AppendToTranscript for PreparedPrefixPackedStatement<'_, F, C>
where
    F: Field,
    C: AppendToTranscript,
{
    fn append_to_transcript<T: Transcript>(&self, transcript: &mut T) {
        transcript.append(&Label(b"prefix_pack_batch"));
        transcript.append(&U64Word(self.packed_num_vars as u64));
        self.commitment.append_to_transcript(transcript);
        transcript.append(&LabelWithCount(
            b"prefix_pack_claims",
            self.ordered_claims.len() as u64,
        ));
        for (evaluation, slot) in &self.ordered_claims {
            transcript.append(&U64Word(slot.num_vars as u64));
            transcript.append(&LabelWithCount(
                b"prefix_pack_prefix",
                slot.prefix.len() as u64,
            ));
            let prefix_bytes = slot
                .prefix
                .iter()
                .map(|bit| u8::from(*bit))
                .collect::<Vec<_>>();
            transcript.append_bytes(&prefix_bytes);
            transcript.append(&LabelWithCount(
                b"prefix_pack_point",
                evaluation.point.len() as u64,
            ));
            for value in evaluation.point.as_slice() {
                value.append_to_transcript(transcript);
            }
            evaluation.value.append_to_transcript(transcript);
        }
    }
}

/// Proof of a prefix-packed batch opening: the claim-reduction sumcheck plus
/// one native PCS opening at its challenge point (see the [module
/// docs](self) for the protocol).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: Serialize, P: Serialize",
    deserialize = "F: Deserialize<'de>, P: Deserialize<'de>"
))]
#[serde(deny_unknown_fields)]
pub struct PackedBatchProof<F, P> {
    /// Coefficients `[c_0, c_1, c_2]` of each round polynomial
    /// `g_j(X) = c_0 + c_1·X + c_2·X²`, one per packed variable.
    pub round_polynomials: Vec<[F; 3]>,
    /// Claimed evaluation of the packed polynomial at the sumcheck point.
    pub opening_eval: F,
    /// Native PCS opening proof for `opening_eval`.
    pub pcs_proof: P,
}

/// Runs the claim-reduction sumcheck over `Σ_z E(z)·W(z)`, binding the
/// most-significant packed variable each round.
///
/// Returns the round polynomials, the opening point (high-to-low), and
/// `W(point)`.
pub(crate) fn prove_reduction_sumcheck<F, T>(
    selector: Vec<F>,
    witness: Vec<F>,
    transcript: &mut T,
) -> (Vec<[F; 3]>, Vec<F>, F)
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    debug_assert_eq!(selector.len(), witness.len());
    debug_assert!(selector.len().is_power_of_two());
    let mut selector = Polynomial::new(selector);
    let mut witness = Polynomial::new(witness);
    let num_rounds = selector.num_vars();
    let mut round_polynomials = Vec::with_capacity(num_rounds);
    let mut point = Vec::with_capacity(num_rounds);

    for _ in 0..num_rounds {
        let half = selector.evaluations().len() / 2;
        let (selector_low, selector_high) = selector.evaluations().split_at(half);
        let (witness_low, witness_high) = witness.evaluations().split_at(half);

        let mut eval_zero = F::zero();
        let mut eval_one = F::zero();
        let mut quadratic = F::zero();
        for ((s_low, s_high), (w_low, w_high)) in selector_low
            .iter()
            .zip(selector_high)
            .zip(witness_low.iter().zip(witness_high))
        {
            eval_zero += *s_low * *w_low;
            eval_one += *s_high * *w_high;
            quadratic += (*s_high - *s_low) * (*w_high - *w_low);
        }
        let coefficients = [eval_zero, eval_one - eval_zero - quadratic, quadratic];
        append_round_polynomial(&coefficients, transcript);
        let challenge: F = transcript.challenge_scalar();
        point.push(challenge);
        selector.bind(challenge);
        witness.bind(challenge);
        round_polynomials.push(coefficients);
    }

    let opening_eval = witness.evaluations()[0];
    (round_polynomials, point, opening_eval)
}

/// Verifies the claim-reduction sumcheck rounds against `input_claim`.
///
/// Returns the opening point and the final claim, which the caller must check
/// against `E(point) · W(point)`.
pub(crate) fn verify_reduction_sumcheck<F, T>(
    round_polynomials: &[[F; 3]],
    num_rounds: usize,
    input_claim: F,
    transcript: &mut T,
) -> Result<(Vec<F>, F), OpeningsError>
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    if round_polynomials.len() != num_rounds {
        return Err(OpeningsError::InvalidBatch(format!(
            "packed reduction proof has {} round polynomials but packing has {num_rounds} variables",
            round_polynomials.len()
        )));
    }
    let mut claim = input_claim;
    let mut point = Vec::with_capacity(num_rounds);
    for coefficients in round_polynomials {
        let eval_zero = coefficients[0];
        let eval_one = coefficients[0] + coefficients[1] + coefficients[2];
        if eval_zero + eval_one != claim {
            return Err(OpeningsError::VerificationFailed);
        }
        append_round_polynomial(coefficients, transcript);
        let challenge: F = transcript.challenge_scalar();
        point.push(challenge);
        claim = coefficients[0] + challenge * (coefficients[1] + challenge * coefficients[2]);
    }
    Ok((point, claim))
}

fn append_round_polynomial<F, T>(coefficients: &[F; 3], transcript: &mut T)
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    transcript.append(&Label(b"packed_reduction_round"));
    for coefficient in coefficients {
        coefficient.append_to_transcript(transcript);
    }
}

/// Factored per-slot state of the batched selector during the sparse
/// reduction sumcheck.
///
/// Slot `i`'s selector segment is `α_i · eq(z, b_i || γ_i)`, a product of
/// per-variable factors, so binding the most-significant variable to `r`
/// leaves the same shape one variable shorter: a prefix bit multiplies the
/// scalar by `r` or `1 - r`, a logical variable by `(1-r)(1-γ) + r·γ`. The
/// eq factors over the unbound variables stay symbolic.
struct SparseSelectorSlot<'a, F> {
    scalar: F,
    prefix: &'a [bool],
    prefix_value: usize,
    point: &'a [F],
}

impl<'a, F: Field> SparseSelectorSlot<'a, F> {
    fn new(alpha: F, slot: &'a PrefixSlot, point: &'a [F]) -> Self {
        Self {
            scalar: alpha,
            prefix: &slot.prefix,
            prefix_value: slot.prefix_index(),
            point,
        }
    }

    /// Coverage after `bound` rounds: the slot's selector is
    /// `scalar · eq(local, point)` on remaining-domain indices `j` with
    /// `j >> point.len() == block`, and zero elsewhere. Once every prefix bit
    /// is bound the slot covers the whole remaining domain — slots that were
    /// disjoint may then overlap, so contributions must always be summed.
    fn remaining(&self, bound: usize, remaining_vars: usize) -> (usize, &'a [F]) {
        if bound < self.prefix.len() {
            let block = self.prefix_value & ((1usize << (self.prefix.len() - bound)) - 1);
            (block, self.point)
        } else {
            debug_assert_eq!(self.prefix.len() + self.point.len() - bound, remaining_vars);
            (0, &self.point[bound - self.prefix.len()..])
        }
    }

    fn bind(&mut self, bound: usize, r: F) {
        if bound < self.prefix.len() {
            self.scalar *= if self.prefix[bound] { r } else { F::one() - r };
        } else {
            let gamma = self.point[bound - self.prefix.len()];
            self.scalar *= (F::one() - r) * (F::one() - gamma) + r * gamma;
        }
    }
}

/// Split-eq lookup for one slot at one sumcheck round: `O(1)` evaluation of
/// the slot's selector at any remaining-domain index, from two half tables of
/// combined size `O(2^{L/2})` for a length-`L` remaining logical point.
struct RoundSelectorSlot<F> {
    block: usize,
    logical_vars: usize,
    low_vars: usize,
    high: Vec<F>,
    low: Vec<F>,
}

impl<F: Field> RoundSelectorSlot<F> {
    fn new(slot: &SparseSelectorSlot<'_, F>, bound: usize, remaining_vars: usize) -> Self {
        let (block, point) = slot.remaining(bound, remaining_vars);
        let low_vars = point.len() / 2;
        let split = point.len() - low_vars;
        Self {
            block,
            logical_vars: point.len(),
            low_vars,
            high: EqPolynomial::evals(&point[..split], Some(slot.scalar)),
            low: EqPolynomial::evals(&point[split..], None),
        }
    }

    #[inline]
    fn value_at(&self, index: usize) -> Option<F> {
        (index >> self.logical_vars == self.block).then(|| {
            let local = index & ((1usize << self.logical_vars) - 1);
            self.high[local >> self.low_vars] * self.low[local & ((1usize << self.low_vars) - 1)]
        })
    }
}

fn selector_value_at<F: Field>(round_slots: &[RoundSelectorSlot<F>], index: usize) -> F {
    round_slots
        .iter()
        .filter_map(|slot| slot.value_at(index))
        .fold(F::zero(), |acc, value| acc + value)
}

/// Dense remaining-domain selector and witness tables for the delegated tail
/// rounds of the sparse prover.
fn materialize_remaining<F: Field>(
    slots: &[SparseSelectorSlot<'_, F>],
    positions: &[(usize, F)],
    bound: usize,
    remaining_vars: usize,
) -> (Vec<F>, Vec<F>) {
    let size = 1usize << remaining_vars;
    let mut selector: Vec<F> = unsafe_allocate_zero_vec(size);
    for slot in slots {
        let (block, point) = slot.remaining(bound, remaining_vars);
        let offset = block << point.len();
        for (cell, value) in selector[offset..]
            .iter_mut()
            .zip(EqPolynomial::evals(point, Some(slot.scalar)))
        {
            *cell += value;
        }
    }
    let mut witness: Vec<F> = unsafe_allocate_zero_vec(size);
    for &(index, weight) in positions {
        witness[index & (size - 1)] += weight;
    }
    (selector, witness)
}

/// Prover setup for opening a prefix-packed witness with a concrete PCS.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrefixPackedProverSetup<PCS: CommitmentScheme, Id = u64> {
    pub pcs: PCS::ProverSetup,
    pub packing: PrefixPacking<Id>,
}

/// Verifier setup for opening a prefix-packed witness with a concrete PCS.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(
    serialize = "PCS::VerifierSetup: Serialize, Id: Serialize",
    deserialize = "PCS::VerifierSetup: Deserialize<'de>, Id: Ord + Deserialize<'de>"
))]
#[serde(deny_unknown_fields)]
pub struct PrefixPackedVerifierSetup<PCS: CommitmentScheme, Id = u64> {
    pub pcs: PCS::VerifierSetup,
    pub packing: PrefixPacking<Id>,
}

fn coefficient_count(num_vars: usize) -> Result<usize, OpeningsError> {
    if num_vars >= usize::BITS as usize {
        return Err(OpeningsError::InvalidSetup(format!(
            "polynomial with {num_vars} variables exceeds addressable domain"
        )));
    }
    Ok(1usize << num_vars)
}
