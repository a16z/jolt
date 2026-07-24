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
//! # Joint opening across commitment objects
//!
//! A statement may span several packed commitments ("objects") of different
//! sizes — e.g. one per trust domain — with no homomorphism available to
//! combine them. [`prove_packed_openings`]/[`verify_packed_openings`] settle
//! all of them with ONE reduction sumcheck plus one native PCS opening per
//! object. After each object's within-object batching above (its `α` powers),
//! the verifier draws one cross-object coefficient `β_k` per object and runs
//! a single `n_max`-round sumcheck for
//!
//! ```text
//! Σ_k β_k · 2^(n_max − n_k) · Σ_i α_{k,i} · v_{k,i}
//!     = Σ_{z ∈ {0,1}^(n_max)} Σ_k β_k · E_k(z_suffix) · f_k(z_suffix),
//! ```
//!
//! where object `k`'s integrand ignores the leading `n_max − n_k` padding
//! variables (equivalently, its polynomial is extended as a constant in
//! them — hence the `2^(n_max − n_k)` scale on its input claim). An object
//! is inert until the rounds reach its own variables, so its bound point is
//! the suffix of the shared challenge point, at which its claimed evaluation
//! is absorbed and opened natively against its own commitment and setup.

use std::{
    collections::{btree_map::Iter, BTreeMap, BTreeSet},
    fmt::Debug,
    ops::Index,
};

use jolt_field::{Field, FromPrimitiveInt, MulPow2};
use jolt_poly::{
    boolean_bits_msb, eq_index_msb, math::Math, thread::unsafe_allocate_zero_vec, EqPolynomial,
    MultilinearPoly, Polynomial,
};
use jolt_transcript::{AppendToTranscript, Label, LabelWithCount, Transcript, U64Word};
use rayon::prelude::*;
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
/// [`verify_packed_openings`]), reducing every logical claim — at arbitrary,
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

    pub fn prepare_statement<'a, F, C>(
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

pub struct PreparedPrefixPackedStatement<'a, F: Field, C> {
    packed_num_vars: usize,
    pub(crate) commitment: &'a C,
    ordered_claims: Vec<(&'a EvaluationClaim<F>, &'a PrefixSlot)>,
}

impl<F, C> PreparedPrefixPackedStatement<'_, F, C>
where
    F: Field,
{
    pub fn num_claims(&self) -> usize {
        self.ordered_claims.len()
    }

    /// Batched sumcheck input claim `Σ_i α_i · v_i`.
    pub fn batched_claim(&self, alpha: &[F]) -> F {
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
    pub fn selector_eval(&self, alpha: &[F], packed_point: &[F]) -> F {
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

const PARALLEL_MIN_CHUNK: usize = 1 << 12;

/// The dense round fold shared by the dense object path and the sparse
/// stepper's dense tail: `[Σ s_lo·w_lo, Σ s_hi·w_hi, Σ (s_hi−s_lo)(w_hi−w_lo)]`.
fn dense_round_evaluations<F: Field>(
    selector_low: &[F],
    selector_high: &[F],
    witness_low: &[F],
    witness_high: &[F],
) -> [F; 3] {
    selector_low
        .par_iter()
        .zip(selector_high.par_iter())
        .zip(witness_low.par_iter().zip(witness_high.par_iter()))
        .with_min_len(PARALLEL_MIN_CHUNK)
        .fold(
            || [F::zero(); 3],
            |mut acc, ((s_low, s_high), (w_low, w_high))| {
                acc[0] += *s_low * *w_low;
                acc[1] += *s_high * *w_high;
                acc[2] += (*s_high - *s_low) * (*w_high - *w_low);
                acc
            },
        )
        .reduce(
            || [F::zero(); 3],
            |left, right| [left[0] + right[0], left[1] + right[1], left[2] + right[2]],
        )
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
}

/// Direct-indexed slot resolution for one sumcheck round: slots grouped by
/// remaining logical length, each group a `block -> slot ids` table (size
/// `2^{rem - L}`), so a probe touches one bucket per distinct length instead
/// of scanning every slot. Buckets hold multiple ids once bound rounds
/// truncate distinct prefixes onto the same block (the dyadic-nesting case),
/// so contributions are still summed.
struct GroupedRoundSelector<'a, F> {
    groups: Vec<(usize, Vec<Vec<u32>>)>,
    slots: &'a [RoundSelectorSlot<F>],
}

impl<'a, F: Field> GroupedRoundSelector<'a, F> {
    fn new(slots: &'a [RoundSelectorSlot<F>], remaining_vars: usize) -> Self {
        let mut groups: Vec<(usize, Vec<Vec<u32>>)> = Vec::new();
        for (slot_index, slot) in slots.iter().enumerate() {
            let position = groups
                .iter()
                .position(|(logical_vars, _)| *logical_vars == slot.logical_vars)
                .unwrap_or_else(|| {
                    groups.push((
                        slot.logical_vars,
                        vec![Vec::new(); 1usize << (remaining_vars - slot.logical_vars)],
                    ));
                    groups.len() - 1
                });
            groups[position].1[slot.block].push(slot_index as u32);
        }
        Self { groups, slots }
    }

    #[inline]
    fn value_at(&self, index: usize) -> F {
        let mut acc = F::zero();
        for (logical_vars, table) in &self.groups {
            for &slot_index in &table[index >> logical_vars] {
                let slot = &self.slots[slot_index as usize];
                let local = index & ((1usize << slot.logical_vars) - 1);
                acc += slot.high[local >> slot.low_vars]
                    * slot.low[local & ((1usize << slot.low_vars) - 1)];
            }
        }
        acc
    }
}

/// Dense remaining-domain selector and witness tables for the delegated tail
/// rounds of the sparse stepper.
fn materialize_remaining<F: Field>(
    slots: &[SparseSelectorSlot<'_, F>],
    positions: &[usize],
    bound_weights: &[F],
    bound: usize,
    remaining_vars: usize,
) -> (Vec<F>, Vec<F>) {
    let size = 1usize << remaining_vars;
    let round_slots: Vec<RoundSelectorSlot<F>> = slots
        .iter()
        .map(|slot| RoundSelectorSlot::new(slot, bound, remaining_vars))
        .collect();
    let grouped = GroupedRoundSelector::new(&round_slots, remaining_vars);
    let mut selector: Vec<F> = unsafe_allocate_zero_vec(size);
    selector
        .par_iter_mut()
        .enumerate()
        .with_min_len(PARALLEL_MIN_CHUNK)
        .for_each(|(index, cell)| *cell = grouped.value_at(index));
    let mut witness: Vec<F> = unsafe_allocate_zero_vec(size);
    for &index in positions {
        witness[index & (size - 1)] += bound_weights[index >> remaining_vars];
    }
    (selector, witness)
}

/// One object's claim-reduction sumcheck as a round-stepped instance over a
/// unit-sparse witness's one-positions: `round_evaluations` / `bind` per
/// round, `final_eval` after the last round. Produces the same field values
/// as the dense selector/witness tables without materializing the `2^n`
/// domain.
///
/// Sparse-to-dense switchover: once the remaining domain is no larger than
/// the number of one-positions, the (now small) selector/witness tables are
/// materialized and later rounds run dense. The check happens at the start
/// of `round_evaluations`.
struct SparseReductionInstance<'a, F: Field> {
    slots: Vec<SparseSelectorSlot<'a, F>>,
    positions: Vec<usize>,
    /// Bound challenges (msb-first). A position's accumulated weight is
    /// `eq(bound_challenges, its top bits)` — looked up in `bound_weights`
    /// (a `2^bound` eq table) instead of stored per position, so binding
    /// never rewrites the position list.
    bound_challenges: Vec<F>,
    bound_weights: Vec<F>,
    num_vars: usize,
    bound: usize,
    dense: Option<(Polynomial<F>, Polynomial<F>)>,
}

impl<'a, F: Field> SparseReductionInstance<'a, F> {
    #[tracing::instrument(
        skip_all,
        name = "SparseReductionInstance::new",
        fields(
            num_vars = statement.packed_num_vars,
            slots = statement.ordered_claims.len(),
            positions = one_positions.len(),
        )
    )]
    fn new(
        statement: &'a PreparedPrefixPackedStatement<'a, F, impl Sized>,
        alpha: &[F],
        one_positions: Vec<usize>,
    ) -> Self {
        debug_assert_eq!(statement.ordered_claims.len(), alpha.len());
        debug_assert!(one_positions
            .iter()
            .all(|index| index >> statement.packed_num_vars == 0));
        let slots = statement
            .ordered_claims
            .iter()
            .zip(alpha)
            .map(|((evaluation, slot), alpha_i)| {
                SparseSelectorSlot::new(*alpha_i, slot, evaluation.point.as_slice())
            })
            .collect();
        Self {
            slots,
            positions: one_positions,
            bound_challenges: Vec::new(),
            bound_weights: vec![F::one()],
            num_vars: statement.packed_num_vars,
            bound: 0,
            dense: None,
        }
    }

    fn remaining_vars(&self) -> usize {
        self.num_vars - self.bound
    }

    /// The current round's `[Σ E·W at 0, Σ E·W at 1, Σ ΔE·ΔW]`. Does not
    /// bind; call [`bind`](Self::bind) with the drawn challenge afterwards.
    fn round_evaluations(&mut self) -> [F; 3] {
        let remaining_vars = self.remaining_vars();
        debug_assert!(remaining_vars > 0, "instance is fully bound");
        if self.dense.is_none() && (1usize << remaining_vars) <= self.positions.len() {
            let _span = tracing::info_span!("SparseReductionInstance::materialize_dense").entered();
            let (selector, witness) = materialize_remaining(
                &self.slots,
                &self.positions,
                &self.bound_weights,
                self.bound,
                remaining_vars,
            );
            self.dense = Some((Polynomial::new(selector), Polynomial::new(witness)));
        }

        if let Some((selector, witness)) = &self.dense {
            let half = selector.evaluations().len() / 2;
            let (selector_low, selector_high) = selector.evaluations().split_at(half);
            let (witness_low, witness_high) = witness.evaluations().split_at(half);
            return dense_round_evaluations(selector_low, selector_high, witness_low, witness_high);
        }

        let _span = tracing::info_span!("SparseReductionInstance::sparse_round").entered();
        let half = 1usize << (remaining_vars - 1);
        let round_slots: Vec<RoundSelectorSlot<F>> = self
            .slots
            .iter()
            .map(|slot| RoundSelectorSlot::new(slot, self.bound, remaining_vars))
            .collect();
        let selector = GroupedRoundSelector::new(&round_slots, remaining_vars);
        let weight_shift = remaining_vars;
        let bound_weights = &self.bound_weights;

        self.positions
            .par_iter()
            .with_min_len(PARALLEL_MIN_CHUNK)
            .fold(
                || [F::zero(); 3],
                |mut acc, &index| {
                    let weight = bound_weights[index >> weight_shift];
                    let low_index = index & (half - 1);
                    let selector_low = selector.value_at(low_index);
                    let selector_high = selector.value_at(low_index | half);
                    let cross = weight * (selector_high - selector_low);
                    if index & half == 0 {
                        acc[0] += weight * selector_low;
                        acc[2] -= cross;
                    } else {
                        acc[1] += weight * selector_high;
                        acc[2] += cross;
                    }
                    acc
                },
            )
            .reduce(
                || [F::zero(); 3],
                |left, right| [left[0] + right[0], left[1] + right[1], left[2] + right[2]],
            )
    }

    /// Bind the round's drawn challenge, consuming one variable.
    fn bind(&mut self, challenge: F) {
        if let Some((selector, witness)) = &mut self.dense {
            selector.bind(challenge);
            witness.bind(challenge);
            self.bound += 1;
            return;
        }
        for slot in &mut self.slots {
            slot.bind(self.bound, challenge);
        }
        self.bound_challenges.push(challenge);
        self.bound_weights = EqPolynomial::evals(&self.bound_challenges, None);
        self.bound += 1;
    }

    /// The packed witness's evaluation at the fully bound point.
    fn final_eval(&self) -> F {
        debug_assert_eq!(self.bound, self.num_vars, "instance is not fully bound");
        if let Some((_, witness)) = &self.dense {
            return witness.evaluations()[0];
        }
        self.positions.iter().fold(F::zero(), |acc, &index| {
            acc + self.bound_weights[index >> self.remaining_vars()]
        })
    }
}

/// Proof of a joint packed opening: the cross-object claim-reduction
/// sumcheck, then per commitment object its claimed packed-witness evaluation
/// and one native PCS opening (see the [module docs](self) for the protocol).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: Serialize, P: Serialize",
    deserialize = "F: Deserialize<'de>, P: Deserialize<'de>"
))]
#[serde(deny_unknown_fields)]
pub struct PackedOpeningProof<F, P> {
    /// Coefficients `[c_0, c_1, c_2]` of each reduction round polynomial
    /// `g_j(X) = c_0 + c_1·X + c_2·X²`, one per packed variable of the widest
    /// object.
    pub round_polynomials: Vec<[F; 3]>,
    /// Per-object claimed evaluation of its packed polynomial at its bound
    /// point (the suffix of the reduction point), in object order.
    pub evaluations: Vec<F>,
    /// Per-object native PCS opening proof for the claimed evaluation.
    pub openings: Vec<P>,
}

/// One commitment object of a joint packed opening, verifier side: the
/// packing, the public statement (commitment plus one claim per slot), and
/// the PCS setup the object's native opening verifies against.
pub struct PackedVerifierObject<'a, PCS: CommitmentScheme, Id> {
    pub packing: &'a PrefixPacking<Id>,
    pub statement: &'a PrefixPackedStatement<PCS::Field, Id, PCS::Output>,
    pub setup: &'a PCS::VerifierSetup,
}

/// One commitment object of a joint packed opening, prover side: the
/// verifier-side statement data plus the packed witness polynomial. The
/// commit-time hint its native opening reuses lives on the object's
/// [`PackedProverGroup`], which owns the opening.
pub struct PackedProverObject<'a, PCS: CommitmentScheme, Id> {
    pub packing: &'a PrefixPacking<Id>,
    pub statement: &'a PrefixPackedStatement<PCS::Field, Id, PCS::Output>,
    pub polynomial: &'a (dyn MultilinearPoly<PCS::Field> + 'a),
    pub setup: &'a PCS::ProverSetup,
}

/// A contiguous run of packed objects opened by one native proof: singleton
/// runs open via [`CommitmentScheme::open`], longer runs are the members of
/// one commitment group and open together via
/// [`CommitmentScheme::open_batch`] at their shared suffix point.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PackedObjectGroup {
    pub start: usize,
    pub len: usize,
}

impl PackedObjectGroup {
    pub fn singleton(start: usize) -> Self {
        Self { start, len: 1 }
    }
}

/// A [`PackedObjectGroup`] plus the commit-time hint its native opening
/// reuses: the group commit's hint for a member group, the object's own for
/// a singleton (`None` recommits).
pub struct PackedProverGroup<H> {
    pub start: usize,
    pub len: usize,
    pub hint: Option<H>,
}

impl<H> PackedProverGroup<H> {
    pub fn singleton(start: usize, hint: Option<H>) -> Self {
        Self {
            start,
            len: 1,
            hint,
        }
    }

    fn span(&self) -> PackedObjectGroup {
        PackedObjectGroup {
            start: self.start,
            len: self.len,
        }
    }
}

/// Checks that `groups` is an in-order contiguous partition of
/// `0..num_objects`.
fn validate_groups(groups: &[PackedObjectGroup], num_objects: usize) -> Result<(), OpeningsError> {
    let mut next = 0usize;
    for group in groups {
        if group.start != next || group.len == 0 {
            return Err(OpeningsError::InvalidBatch(format!(
                "packed object groups must contiguously partition {num_objects} objects, got {groups:?}"
            )));
        }
        next += group.len;
    }
    if next != num_objects {
        return Err(OpeningsError::InvalidBatch(format!(
            "packed object groups cover {next} of {num_objects} objects"
        )));
    }
    Ok(())
}

/// Absorbs every object's prepared statement, then draws each object's
/// within-object claim-batching `α` powers and its cross-object coefficient
/// `β`, all in object order.
fn packed_opening_challenges<F, C, T>(
    prepared: &[PreparedPrefixPackedStatement<'_, F, C>],
    transcript: &mut T,
) -> (Vec<Vec<F>>, Vec<F>)
where
    F: Field,
    C: AppendToTranscript,
    T: Transcript<Challenge = F>,
{
    for statement in prepared {
        statement.append_to_transcript(transcript);
    }
    let alphas = prepared
        .iter()
        .map(|statement| transcript.challenge_scalar_powers(statement.num_claims()))
        .collect();
    let coefficients = prepared
        .iter()
        .map(|_| transcript.challenge_scalar())
        .collect();
    (alphas, coefficients)
}

/// Proves a joint packed opening: one reduction sumcheck across all objects,
/// then one native PCS opening per object group.
pub fn prove_packed_openings<PCS, Id, T>(
    objects: Vec<PackedProverObject<'_, PCS, Id>>,
    groups: Vec<PackedProverGroup<PCS::OpeningHint>>,
    transcript: &mut T,
) -> Result<PackedOpeningProof<PCS::Field, PCS::Proof>, OpeningsError>
where
    PCS: CommitmentScheme,
    PCS::Output: AppendToTranscript,
    Id: Clone + Debug + Ord,
    T: Transcript<Challenge = PCS::Field>,
{
    if objects.is_empty() {
        return Err(OpeningsError::InvalidBatch(
            "packed opening requires at least one object".to_owned(),
        ));
    }
    let spans: Vec<PackedObjectGroup> = groups.iter().map(PackedProverGroup::span).collect();
    validate_groups(&spans, objects.len())?;
    let prepared = objects
        .iter()
        .map(|object| object.packing.prepare_statement(object.statement))
        .collect::<Result<Vec<_>, _>>()?;
    for object in &objects {
        if object.polynomial.num_vars() != object.packing.packed_num_vars {
            return Err(OpeningsError::InvalidBatch(format!(
                "packed polynomial has {} variables but its prefix packing has {}",
                object.polynomial.num_vars(),
                object.packing.packed_num_vars
            )));
        }
    }
    let (alphas, coefficients) = packed_opening_challenges(&prepared, transcript);
    let max_num_vars = objects
        .iter()
        .map(|object| object.packing.packed_num_vars)
        .max()
        .unwrap_or(0);

    // Per-object sumcheck state: dense β-and-α-scaled selector/witness tables
    // for arbitrary witnesses, or the sparse round stepper for unit-sparse
    // (one-hot) witnesses — same field values, no `2^n` materialization —
    // plus the rounds to wait before the object's variables bind and its
    // total `Σ_z E(z)·W(z)` (the constant while padded).
    enum ObjectState<'a, F: Field> {
        Dense {
            selector: Polynomial<F>,
            witness: Polynomial<F>,
        },
        Sparse(SparseReductionInstance<'a, F>),
    }
    struct ObjectProver<'a, F: Field> {
        state: ObjectState<'a, F>,
        padding_rounds: usize,
        total: F,
    }
    let construction_span = tracing::info_span!("prove_packed_openings::build_tables").entered();
    let mut tables = prepared
        .iter()
        .zip(&alphas)
        .zip(&coefficients)
        .zip(&objects)
        .map(|(((statement, alpha), coefficient), object)| {
            let scaled_alpha: Vec<PCS::Field> =
                alpha.iter().map(|alpha| *alpha * *coefficient).collect();
            // `Σ_z E(z)·W(z)` over honest tables is exactly the (β-scaled)
            // batched claim — O(claims), not an O(2^n) dot product. A false
            // claimed value still fails verification at the object's first
            // bound round.
            let total = statement.batched_claim(&scaled_alpha);
            let padding_rounds = max_num_vars - object.packing.packed_num_vars;
            let state = if object.polynomial.is_one_hot() {
                let mut positions =
                    Vec::with_capacity(object.polynomial.one_hot_indices().map_or(0, <[_]>::len));
                object
                    .polynomial
                    .for_each_one(&mut |position| positions.push(position));
                ObjectState::Sparse(SparseReductionInstance::new(
                    statement,
                    &scaled_alpha,
                    positions,
                ))
            } else {
                ObjectState::Dense {
                    selector: Polynomial::new(statement.selector_table(&scaled_alpha)),
                    witness: Polynomial::new(object.polynomial.to_dense().into_owned()),
                }
            };
            ObjectProver {
                state,
                padding_rounds,
                total,
            }
        })
        .collect::<Vec<_>>();

    drop(construction_span);

    let rounds_span = tracing::info_span!("prove_packed_openings::rounds").entered();
    let mut round_polynomials = Vec::with_capacity(max_num_vars);
    let mut point: Vec<PCS::Field> = Vec::with_capacity(max_num_vars);
    for round in 0..max_num_vars {
        let mut eval_zero = PCS::Field::from_u64(0);
        let mut eval_one = PCS::Field::from_u64(0);
        let mut quadratic = PCS::Field::from_u64(0);
        for table in &mut tables {
            if table.padding_rounds > round {
                // Still padded: constant in the bound variable, halving each
                // padding round.
                let constant = table.total.mul_pow_2(table.padding_rounds - round - 1);
                eval_zero += constant;
                eval_one += constant;
                continue;
            }
            let [object_zero, object_one, object_quadratic] = match &mut table.state {
                ObjectState::Dense { selector, witness } => {
                    let half = selector.evaluations().len() / 2;
                    let (selector_low, selector_high) = selector.evaluations().split_at(half);
                    let (witness_low, witness_high) = witness.evaluations().split_at(half);
                    dense_round_evaluations(selector_low, selector_high, witness_low, witness_high)
                }
                ObjectState::Sparse(instance) => instance.round_evaluations(),
            };
            eval_zero += object_zero;
            eval_one += object_one;
            quadratic += object_quadratic;
        }
        let coefficients = [eval_zero, eval_one - eval_zero - quadratic, quadratic];
        append_round_polynomial(&coefficients, transcript);
        let challenge: PCS::Field = transcript.challenge_scalar();
        point.push(challenge);
        for table in &mut tables {
            if table.padding_rounds <= round {
                match &mut table.state {
                    ObjectState::Dense { selector, witness } => {
                        selector.bind(challenge);
                        witness.bind(challenge);
                    }
                    ObjectState::Sparse(instance) => instance.bind(challenge),
                }
            }
        }
        round_polynomials.push(coefficients);
    }

    drop(rounds_span);

    let evaluations: Vec<PCS::Field> = tables
        .iter()
        .map(|table| match &table.state {
            ObjectState::Dense { witness, .. } => witness.evaluations()[0],
            ObjectState::Sparse(instance) => instance.final_eval(),
        })
        .collect();
    for (table, evaluation) in tables.iter().zip(&evaluations) {
        let suffix = &point[table.padding_rounds..];
        EvaluationClaim::new(suffix.to_vec(), *evaluation).append_to_transcript(transcript);
    }
    let mut openings = Vec::with_capacity(groups.len());
    for group in groups {
        let members = group.start..group.start + group.len;
        if group.len == 1 {
            let object = &objects[group.start];
            openings.push(PCS::open(
                object.polynomial,
                &point[tables[group.start].padding_rounds..],
                evaluations[group.start],
                object.setup,
                group.hint,
                transcript,
            )?);
            continue;
        }
        let padding = tables[group.start].padding_rounds;
        if members
            .clone()
            .any(|index| tables[index].padding_rounds != padding)
        {
            return Err(OpeningsError::InvalidBatch(
                "packed object group members must share one opening point arity".to_owned(),
            ));
        }
        let hint = group.hint.ok_or_else(|| {
            OpeningsError::InvalidBatch(
                "packed object group is missing its group commit hint".to_owned(),
            )
        })?;
        let polynomials: Vec<&dyn MultilinearPoly<PCS::Field>> = members
            .clone()
            .map(|index| objects[index].polynomial)
            .collect();
        openings.push(PCS::open_batch(
            &polynomials,
            &point[padding..],
            &evaluations[members.clone()],
            objects[group.start].setup,
            hint,
            transcript,
        )?);
    }

    Ok(PackedOpeningProof {
        round_polynomials,
        evaluations,
        openings,
    })
}

/// Verifies a joint packed opening against the objects' statements.
pub fn verify_packed_openings<PCS, Id, T>(
    objects: &[PackedVerifierObject<'_, PCS, Id>],
    groups: &[PackedObjectGroup],
    proof: &PackedOpeningProof<PCS::Field, PCS::Proof>,
    transcript: &mut T,
) -> Result<(), OpeningsError>
where
    PCS: CommitmentScheme,
    PCS::Output: AppendToTranscript,
    Id: Clone + Debug + Ord,
    T: Transcript<Challenge = PCS::Field>,
{
    if objects.is_empty() {
        return Err(OpeningsError::InvalidBatch(
            "packed opening requires at least one object".to_owned(),
        ));
    }
    validate_groups(groups, objects.len())?;
    if proof.evaluations.len() != objects.len() || proof.openings.len() != groups.len() {
        return Err(OpeningsError::InvalidBatch(format!(
            "packed opening proof carries {} evaluations and {} openings for {} objects in {} groups",
            proof.evaluations.len(),
            proof.openings.len(),
            objects.len(),
            groups.len()
        )));
    }
    let prepared = objects
        .iter()
        .map(|object| object.packing.prepare_statement(object.statement))
        .collect::<Result<Vec<_>, _>>()?;
    let (alphas, coefficients) = packed_opening_challenges(&prepared, transcript);
    let max_num_vars = objects
        .iter()
        .map(|object| object.packing.packed_num_vars)
        .max()
        .unwrap_or(0);

    // Each object's batched claim joins scaled by 2^(padding rounds): its
    // integrand is constant in the leading variables it does not use.
    let input_claim = prepared
        .iter()
        .zip(&alphas)
        .zip(&coefficients)
        .zip(objects)
        .fold(
            PCS::Field::from_u64(0),
            |acc, (((statement, alpha), coefficient), object)| {
                let padding = max_num_vars - object.packing.packed_num_vars;
                acc + *coefficient * statement.batched_claim(alpha).mul_pow_2(padding)
            },
        );
    let (point, final_claim) = verify_reduction_sumcheck(
        &proof.round_polynomials,
        max_num_vars,
        input_claim,
        transcript,
    )?;

    // Absorb each object's claimed evaluation at its suffix point, then check
    // the reduced claim `Σ_k β_k · E_k(suffix_k) · W_k(suffix_k)`.
    let mut expected_final_claim = PCS::Field::from_u64(0);
    for (((statement, alpha), coefficient), (object, evaluation)) in prepared
        .iter()
        .zip(&alphas)
        .zip(&coefficients)
        .zip(objects.iter().zip(&proof.evaluations))
    {
        let suffix = &point[max_num_vars - object.packing.packed_num_vars..];
        EvaluationClaim::new(suffix.to_vec(), *evaluation).append_to_transcript(transcript);
        expected_final_claim += *coefficient * statement.selector_eval(alpha, suffix) * *evaluation;
    }
    if final_claim != expected_final_claim {
        return Err(OpeningsError::VerificationFailed);
    }

    for (group, opening) in groups.iter().zip(&proof.openings) {
        let members = group.start..group.start + group.len;
        let first = &objects[group.start];
        let suffix = &point[max_num_vars - first.packing.packed_num_vars..];
        if group.len == 1 {
            PCS::verify(
                &first.statement.commitment,
                suffix,
                proof.evaluations[group.start],
                opening,
                first.setup,
                transcript,
            )?;
            continue;
        }
        if members
            .clone()
            .any(|index| objects[index].packing.packed_num_vars != first.packing.packed_num_vars)
        {
            return Err(OpeningsError::InvalidBatch(
                "packed object group members must share one opening point arity".to_owned(),
            ));
        }
        PCS::verify_batch(
            &first.statement.commitment,
            suffix,
            &proof.evaluations[members],
            opening,
            first.setup,
            transcript,
        )?;
    }
    Ok(())
}

/// Verifies the claim-reduction sumcheck rounds against `input_claim`.
///
/// Returns the opening point and the final claim, which the caller must check
/// against the batched selector-times-witness evaluation.
fn verify_reduction_sumcheck<F, T>(
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
            "packed reduction proof has {} round polynomials but the widest packing has {num_rounds} variables",
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

fn coefficient_count(num_vars: usize) -> Result<usize, OpeningsError> {
    if num_vars >= usize::BITS as usize {
        return Err(OpeningsError::InvalidSetup(format!(
            "polynomial with {num_vars} variables exceeds addressable domain"
        )));
    }
    Ok(1usize << num_vars)
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use jolt_field::{Fr, FromPrimitiveInt};

    fn field(value: u64) -> Fr {
        Fr::from_u64(value)
    }

    type OracleClaims = Vec<(u64, EvaluationClaim<Fr>)>;

    /// A statement with mixed-arity slots, its dense selector/witness tables,
    /// and the one-positions of the (unit-sparse) witness.
    fn oracle_fixture() -> (PrefixPacking<u64>, OracleClaims, Vec<usize>) {
        // Three logical polynomials: two 16-cell columns and one 4-cell
        // column packed into 2^6 = 64 cells (mirrors the mixed-arity shape
        // of the real packings).
        let packing = PrefixPacking::<u64>::new([
            PackedPolynomial::from((0u64, 4usize)),
            (1u64, 4usize).into(),
            (2u64, 2usize).into(),
        ])
        .unwrap();
        let positions: Vec<usize> = vec![
            packing[&0].packed_index(3),
            packing[&0].packed_index(9),
            packing[&1].packed_index(0),
            packing[&1].packed_index(7),
            packing[&1].packed_index(15),
            packing[&2].packed_index(2),
        ];
        let mut witness = vec![field(0); 1 << packing.packed_num_vars];
        for &position in &positions {
            witness[position] = field(1);
        }
        let claims: OracleClaims = [0u64, 1, 2]
            .into_iter()
            .map(|id| {
                let slot = &packing[&id];
                let point: Vec<Fr> = (0..slot.num_vars)
                    .map(|i| field(3 + 7 * (id + 1) * (i as u64 + 1)))
                    .collect();
                let mut packed_point: Vec<Fr> = slot
                    .prefix
                    .iter()
                    .map(|bit| field(u64::from(*bit)))
                    .collect();
                packed_point.extend(point.iter().copied());
                let value = Polynomial::new(witness.clone()).evaluate(&packed_point);
                (id, EvaluationClaim::new(point, value))
            })
            .collect();
        (packing, claims, positions)
    }

    /// The sparse round stepper must produce the same field values as the
    /// dense selector/witness tables at every round — including across its
    /// sparse-to-dense switchover — and the same final witness evaluation.
    #[test]
    fn sparse_stepper_matches_the_dense_tables() {
        let (packing, claims, positions) = oracle_fixture();
        let statement = PrefixPackedStatement::new((), claims);
        let prepared = packing.prepare_statement(&statement).unwrap();
        let alpha: Vec<Fr> = (0..prepared.num_claims())
            .map(|i| field(11 + 5 * i as u64))
            .collect();

        let mut dense_selector = Polynomial::new(prepared.selector_table(&alpha));
        let mut dense_witness = {
            let mut table = vec![field(0); 1 << packing.packed_num_vars];
            for &position in &positions {
                table[position] = field(1);
            }
            Polynomial::new(table)
        };
        let mut instance = SparseReductionInstance::new(&prepared, &alpha, positions);

        for round in 0..packing.packed_num_vars {
            let half = dense_selector.evaluations().len() / 2;
            let (selector_low, selector_high) = dense_selector.evaluations().split_at(half);
            let (witness_low, witness_high) = dense_witness.evaluations().split_at(half);
            let dense =
                dense_round_evaluations(selector_low, selector_high, witness_low, witness_high);
            let sparse = instance.round_evaluations();
            assert_eq!(sparse, dense, "round {round} evaluations diverge");

            let challenge = field(101 + 13 * round as u64);
            dense_selector.bind(challenge);
            dense_witness.bind(challenge);
            instance.bind(challenge);
        }
        assert_eq!(
            instance.final_eval(),
            dense_witness.evaluations()[0],
            "final witness evaluations diverge"
        );
    }
}
