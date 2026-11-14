//! The read/write kernel's streamed cycle state: rounds read the witness's rows in
//! task-partitioned passes ([`pass`]) that regenerate their level's entries
//! and count the next level's. The first round from 1 on whose counted
//! entries and increments fit the cap, at the latest the last cycle round,
//! fills exactly those field-coefficient entries and increments in the pass
//! that yields its message, and the default kernel's direct phase
//! ([`CycleState`]) binds them from there. Passes and increments cover the
//! rows that can hold a register operand ([`StreamedRows`]), as the default
//! kernel's collect does.

use jolt_field::JoltField;
use jolt_sumcheck::SumcheckError;

use super::layout::SparseEntry;
use super::{
    lut_len, lut_saturated, CoeffLut, CyclePhase, CycleState, READ_SEED_VALUES, WRITE_SEED_VALUES,
};
use crate::optimized::prefix_product::PrefixColumn;
use crate::optimized::registers_read_write::rows::StreamedRows;

mod pass;

use pass::PassError;

/// The deepest level whose coefficients stay [`LutIndex`](super::LutIndex)es:
/// level `ℓ`'s tables are the seeds squared `ℓ` times, and level 3's read
/// table fills the `u16` index domain. A merge into level 4 dereferences both
/// coefficients with level 3's tables.
const LUT_LEVELS: usize = 3;

const _: () = assert!(
    !lut_saturated(lut_len(READ_SEED_VALUES, LUT_LEVELS as u32 - 1))
        && !lut_saturated(lut_len(WRITE_SEED_VALUES, LUT_LEVELS as u32 - 1))
        && lut_saturated(lut_len(READ_SEED_VALUES, LUT_LEVELS as u32))
);

/// Whether a level of `entries` field-coefficient entries and `rows`
/// increments fits `cap` bytes; a size past `usize` does not.
fn level_fits<F: JoltField>(entries: usize, rows: usize, cap: usize) -> bool {
    entries
        .checked_mul(size_of::<SparseEntry<F, F>>())
        .zip(rows.checked_mul(size_of::<F>()))
        .and_then(|(entries, increments)| entries.checked_add(increments))
        .is_some_and(|bytes| bytes <= cap)
}

/// Cycle rounds before the fill: each pass regenerates its round's level
/// from the rows.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
struct Frontier<F: JoltField> {
    /// Level `ℓ`'s `[ra, wa]` coefficient tables, `ℓ ≤ min(bound, LUT_LEVELS)`.
    luts: Vec<[CoeffLut<F>; 2]>,
    /// Each Gruen group's entries at the next round's level, counted by the
    /// last pass: one per column of either row of each of its pairs, whatever
    /// the coefficients. `None` before round 0's pass.
    counts: Option<Vec<usize>>,
    /// Bytes the filled entries and increments may take.
    cap: usize,
}

impl<F: JoltField> Frontier<F> {
    /// The last pass's counts when round `depth`'s level of `rows` rows,
    /// `increments` of them with a prefix cycle, is the one to fill: from
    /// level 1 on, the first that fits the cap ([`level_fits`]), or the last
    /// cycle round's two rows, which hold at most `2K` entries.
    fn admitted(&self, depth: usize, rows: usize, increments: usize) -> Option<&[usize]> {
        let counts = self.counts.as_deref()?;
        let entries = counts
            .iter()
            .try_fold(0usize, |sum, &count| sum.checked_add(count));
        let fits = entries.is_some_and(|entries| level_fits::<F>(entries, increments, self.cap));
        (depth >= 1 && (rows == 2 || fits)).then_some(counts)
    }
}

#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
enum Phase<F: JoltField> {
    Streamed(Frontier<F>),
    Filled(CycleState<F>),
}

#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
pub(in crate::optimized::registers_read_write) struct StreamedCycleState<F: JoltField>(Phase<F>);

/// The bound cycle row as dense `(ra, wa, val)` address arrays and its `inc`.
type DenseCycleRow<F> = (Vec<F>, Vec<F>, Vec<F>, F);

fn pass_failed<F: JoltField>(error: PassError) -> SumcheckError<F> {
    SumcheckError::MissingEvaluationSource {
        kind: match error {
            PassError::Rows => "register read/write trace rows",
            PassError::Diverged => "register frontier fill (entry counts disagree)",
        },
    }
}

impl<F: JoltField> StreamedCycleState<F> {
    /// `gamma` builds the level-0 coefficient tables; `cap` bounds the bytes
    /// of the filled entries and increments.
    pub(in crate::optimized::registers_read_write) fn new(gamma: F, cap: usize) -> Self {
        Self(Phase::Streamed(Frontier {
            luts: vec![CycleState::seed_luts(gamma)],
            counts: None,
            cap,
        }))
    }

    /// Round `r.len()`'s `[q(0), leading coefficient]`, `r` holding the bound
    /// cycle challenges. A streamed round whose level is admitted fills it;
    /// any other streamed round counts the next level.
    pub(in crate::optimized::registers_read_write) fn quadratic(
        &mut self,
        rows: &StreamedRows,
        e_in: &[F],
        e_out: &[F],
        r: &[F],
    ) -> Result<[F; 2], SumcheckError<F>> {
        let frontier = match &mut self.0 {
            Phase::Filled(cycle) => return Ok(cycle.quadratic(e_in, e_out)),
            Phase::Streamed(frontier) => frontier,
        };
        let admitted = frontier.admitted(
            r.len(),
            2 * e_out.len() * e_in.len(),
            rows.prefix_len.div_ceil(1 << r.len()),
        );
        let Some(counts) = admitted else {
            let (quadratic, counts) =
                pass::count(rows, e_in, e_out, r, &frontier.luts).map_err(pass_failed)?;
            frontier.counts = Some(counts);
            return Ok(quadratic);
        };
        let (quadratic, entries, inc) =
            pass::fill(rows, e_in, e_out, r, &frontier.luts, counts).map_err(pass_failed)?;
        // The increments end with the prefix; no later row writes a register,
        // so their tail is zero.
        self.0 = Phase::Filled(CycleState(CyclePhase::Direct {
            entries,
            inc: PrefixColumn {
                values: inc,
                tail: [F::zero(); 2],
            },
        }));
        Ok(quadratic)
    }

    pub(in crate::optimized::registers_read_write) fn q_at_one(
        &self,
        rows: &StreamedRows,
        e_in: &[F],
        e_out: &[F],
        r: &[F],
    ) -> Result<F, SumcheckError<F>> {
        match &self.0 {
            Phase::Streamed(frontier) => {
                pass::at_one(rows, e_in, e_out, r, &frontier.luts).map_err(pass_failed)
            }
            Phase::Filled(cycle) => Ok(cycle.q_at_one(e_in, e_out)),
        }
    }

    /// Binds a cycle challenge; returns whether a full entry generation was
    /// replaced. A streamed bind holds no entries: it adds the next level's
    /// coefficient tables, through level [`LUT_LEVELS`].
    pub(in crate::optimized::registers_read_write) fn bind(&mut self, r: F) -> bool {
        match &mut self.0 {
            Phase::Streamed(frontier) => {
                if frontier.luts.len() <= LUT_LEVELS {
                    let [ra, wa] = &frontier.luts[frontier.luts.len() - 1];
                    let mut next = [
                        CoeffLut::new(ra.values.clone()),
                        CoeffLut::new(wa.values.clone()),
                    ];
                    for lut in &mut next {
                        lut.bind(r);
                    }
                    frontier.luts.push(next);
                }
                false
            }
            Phase::Filled(cycle) => cycle.bind(r),
        }
    }

    /// The fully bound cycle row as dense `(ra, wa, val)` address arrays and
    /// the bound increment, releasing the cycle state. Every domain of two or
    /// more cycle rounds fills at its last round at the latest.
    pub(in crate::optimized::registers_read_write) fn take_dense(
        &mut self,
        k: usize,
    ) -> Result<DenseCycleRow<F>, SumcheckError<F>> {
        match &mut self.0 {
            Phase::Filled(cycle) => Ok(cycle.take_dense(k)),
            Phase::Streamed(_) => Err(SumcheckError::MissingEvaluationSource {
                kind: "register frontier (cycle rounds ended before the fill)",
            }),
        }
    }
}

#[cfg(test)]
impl<F: JoltField> StreamedCycleState<F> {
    /// The bytes of a level of `entries` entries and `rows` increments.
    pub(in crate::optimized::registers_read_write) fn level_bytes(
        entries: usize,
        rows: usize,
    ) -> usize {
        entries * size_of::<SparseEntry<F, F>>() + rows * size_of::<F>()
    }

    /// The next level's entries the last pass counted, while streamed.
    pub(in crate::optimized::registers_read_write) fn counted(&self) -> Option<usize> {
        match &self.0 {
            Phase::Streamed(frontier) => frontier.counts.as_ref().map(|counts| counts.iter().sum()),
            Phase::Filled(_) => None,
        }
    }

    /// The filled level's entries, their capacity and its increments.
    pub(in crate::optimized::registers_read_write) fn filled(
        &self,
    ) -> Option<(usize, usize, usize)> {
        match &self.0 {
            Phase::Filled(CycleState(CyclePhase::Direct { entries, inc })) => {
                Some((entries.len(), entries.capacity(), inc.values.len()))
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use jolt_field::Fr;

    use super::*;

    /// A level is admitted exactly at the cap, priced with the increments
    /// its prefix holds, never at level 0 or before a pass counted it, always
    /// at the last cycle round's two rows, and not when its bytes or its
    /// counts' sum pass `usize`.
    #[test]
    fn level_admission_is_exact_at_the_cap_and_past_usize_bytes() {
        let entry = size_of::<SparseEntry<Fr, Fr>>();
        let field = size_of::<Fr>();
        assert!(level_fits::<Fr>(6, 4, 6 * entry + 4 * field));
        assert!(!level_fits::<Fr>(6, 4, 6 * entry + 4 * field - 1));
        assert!(!level_fits::<Fr>(usize::MAX / entry + 1, 0, usize::MAX));
        assert!(!level_fits::<Fr>(0, usize::MAX / field + 1, usize::MAX));
        assert!(!level_fits::<Fr>(
            usize::MAX / entry,
            usize::MAX / field,
            usize::MAX
        ));

        let frontier = |counts: Option<Vec<usize>>, cap| Frontier::<Fr> {
            luts: Vec::new(),
            counts,
            cap,
        };
        let cap = 3 * entry + 8 * field;
        let counts = || Some(vec![1, 2]);
        assert_eq!(frontier(counts(), cap).admitted(1, 8, 8), Some(&[1, 2][..]));
        assert_eq!(frontier(counts(), cap - 1).admitted(1, 8, 8), None);
        assert_eq!(
            frontier(counts(), cap - 3 * field).admitted(1, 8, 5),
            Some(&[1, 2][..])
        );
        assert_eq!(
            frontier(counts(), cap - 3 * field - 1).admitted(1, 8, 5),
            None
        );
        assert_eq!(frontier(counts(), usize::MAX).admitted(0, 8, 8), None);
        assert_eq!(frontier(None, usize::MAX).admitted(1, 8, 8), None);
        assert_eq!(frontier(counts(), 0).admitted(3, 2, 2), Some(&[1, 2][..]));
        assert_eq!(
            frontier(Some(vec![usize::MAX, 2]), usize::MAX).admitted(1, 8, 8),
            None
        );
    }
}
