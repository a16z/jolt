//! The wiring zero-check as a stage-A batch member (degree 3, head-aligned
//! with the row relation):
//! `Σ_row eq(τ, row) · [Σ_s γ_s w_s(row) + pins(row)] − Σ_t P_t(row) · V_t(row)`,
//! where each read side `t` pairs a public kernel table `P_t` (the `eq(τ, ·)`
//! weights of every reader of the row, scaled by the slot coefficient) with
//! the linear form `V_t` of the source word's bits — bound as separate
//! multilinears so the final claim is `Σ_t P_t(r) · V_t(r)`.

use jolt_field::{Fr, Ring, Zero};
use jolt_poly::{EqPolynomial, UnivariatePoly};
use jolt_sumcheck::prover::ProveRounds;
use jolt_sumcheck::SumcheckError;
use rayon::prelude::*;

use super::layout::{
    WiredWord, WordColumn, CANON, CANON_BITS, COMMITTED, WIRED_BITS, WIRED_WORDS, WORD_BITS,
};
use super::wiring::{
    canonicality, source, word, PublicInputs, Source, VkColumns, Weights, WiringStatement,
    WordSlot, CELL_ROWS, LOG_CELL, MODULUS_HI,
};

/// A read side: the value form `Σ_j value_weight(slot, coefficient(j)) ·
/// bit_j(group)` shared by every reader of that shape.
#[derive(Clone, Copy, PartialEq, Eq)]
struct ReadKey {
    /// `Din` / `Bin` keep their per-bit coefficients; every word slot shares
    /// the `2^k` form and carries its slot coefficient in the kernel.
    slot: WordSlot,
    group: WordColumn,
    weights: Weights,
}

impl ReadKey {
    fn new(slot: WordSlot, group: WordColumn, weights: Weights) -> Self {
        let slot = match slot {
            WordSlot::Din | WordSlot::Bin => slot,
            WordSlot::Word(_) => WordSlot::Word(WiredWord::AIn),
        };
        Self {
            slot,
            group,
            weights,
        }
    }
}

/// One reader of a position: slot `slot` of position `position` copies this
/// row's word from the cell `cell_delta` cells away from the reader.
#[derive(Clone, Copy)]
struct Reader {
    position: usize,
    slot: WordSlot,
    key: usize,
    cell_delta: i8,
}

pub struct WiringProver {
    tau: Vec<Fr>,
    round: usize,
    claim: Fr,
    /// `eq(τ, row)`, the wired-side linear part and the `(selector, value)`
    /// pin products: the two message half-word pins and the two wire
    /// canonicality constraints.
    eq: Vec<Fr>,
    linear: Vec<Fr>,
    pins: Vec<(Vec<Fr>, Vec<Fr>)>,
    /// `(P_t, V_t)` per read key.
    reads: Vec<(Vec<Fr>, Vec<Fr>)>,
}

impl WiringProver {
    /// # Panics
    ///
    /// Panics unless every column has `2^tau.len()` rows.
    pub fn new(
        statement: &WiringStatement<'_>,
        bits: &[Vec<u8>],
        wired_bits: &[Vec<u8>],
        wired_words: &[Vec<u32>],
        vk: &VkColumns,
        public: &PublicInputs,
        tau: Vec<Fr>,
    ) -> Self {
        let rows = 1usize << tau.len();
        assert_eq!(bits[0].len(), rows, "column rows");
        let gammas = statement.gammas;
        let gamma_lo = gammas[WIRED_BITS + WIRED_WORDS];
        let gamma_hi = gammas[WIRED_BITS + WIRED_WORDS + 1];
        let gamma_canon = [
            gammas[WIRED_BITS + WIRED_WORDS + 2],
            gammas[WIRED_BITS + WIRED_WORDS + 3],
        ];
        let bound = Fr::from_u64(MODULUS_HI - 1);
        let canon_forms = [canonicality(false), canonicality(true)];
        let eq = EqPolynomial::<Fr>::evals(&tau, None);
        let (tau_hi, tau_lo) = tau.split_at(tau.len() - LOG_CELL);
        let eq_hi = EqPolynomial::<Fr>::evals(tau_hi, None);
        let eq_lo = EqPolynomial::<Fr>::evals(tau_lo, None);

        // Readers of every position, inverted from the wiring table, and the
        // distinct read keys.
        let mut keys: Vec<ReadKey> = Vec::new();
        let mut readers: Vec<Vec<Reader>> = vec![Vec::new(); CELL_ROWS];
        for p in 0..CELL_ROWS {
            for slot in WordSlot::all() {
                let (from, group, weights, cell_delta) = match source(p, slot) {
                    Source::Cell {
                        group,
                        weights,
                        delta,
                    } => (
                        (p as isize - isize::from(delta)) as usize,
                        group,
                        weights,
                        0,
                    ),
                    Source::Previous {
                        group,
                        weights,
                        position,
                    } => (usize::from(position), group, weights, 1),
                    Source::Next {
                        group,
                        weights,
                        position,
                    } => (usize::from(position), group, weights, -1),
                    Source::Zero | Source::Const(_) => continue,
                };
                let key = ReadKey::new(slot, group, weights);
                let index = keys.iter().position(|k| *k == key).unwrap_or_else(|| {
                    keys.push(key);
                    keys.len() - 1
                });
                readers[from].push(Reader {
                    position: p,
                    slot,
                    key: index,
                    cell_delta,
                });
            }
        }
        // Kernel scale of a reader: the slot coefficient for words, 1 for the
        // per-bit slots (their coefficients live in the value form).
        let scale = |slot: WordSlot| match slot {
            WordSlot::Din | WordSlot::Bin => Fr::from_u64(1),
            WordSlot::Word(_) => gammas[slot.gamma_index()],
        };
        let value_weights: Vec<[Fr; WORD_BITS]> = keys
            .iter()
            .map(|key| {
                let mut weights = [Fr::zero(); WORD_BITS];
                for (j, w) in weights.iter_mut().enumerate() {
                    if let Some(k) = key.weights.coefficient(j) {
                        *w = match key.slot {
                            WordSlot::Din | WordSlot::Bin => gammas[key.slot.gamma_index() + k],
                            WordSlot::Word(_) => Fr::from_u64(1).mul_pow_2(k),
                        };
                    }
                }
                weights
            })
            .collect();
        let word_gammas: Vec<Fr> = WiredWord::ALL
            .iter()
            .map(|w| gammas[WordSlot::Word(*w).gamma_index()])
            .collect();
        let cells = rows >> LOG_CELL;
        let tail: Vec<(usize, bool, u16)> = public.tail_halves().collect();

        let per_row = |row: usize| -> (Fr, [Fr; 8], Vec<Fr>, Vec<Fr>) {
            let (cell, p) = (row >> LOG_CELL, row & (CELL_ROWS - 1));
            let mut linear = Fr::zero();
            for (k, column) in wired_bits.iter().enumerate() {
                if column[row] == 1 {
                    linear += gammas[k];
                }
            }
            for (column, gamma) in wired_words.iter().zip(&word_gammas) {
                linear += *gamma * Fr::from_u32(column[row]);
            }
            let mut sel_lo = gamma_lo * Fr::from_u64(u64::from(vk.lo_is_const[row]));
            let mut sel_hi = gamma_hi * Fr::from_u64(u64::from(vk.hi_is_const[row]));
            let mut const_lo = gamma_lo * Fr::from_u64(u64::from(vk.lo_const[row]));
            let mut const_hi = gamma_hi * Fr::from_u64(u64::from(vk.hi_const[row]));
            if cell == 0 {
                for &(w, high, value) in &tail {
                    if w == p {
                        let (sel, constant, gamma) = if high {
                            (&mut sel_hi, &mut const_hi, gamma_hi)
                        } else {
                            (&mut sel_lo, &mut const_lo, gamma_lo)
                        };
                        *sel += gamma;
                        *constant += gamma * Fr::from_u64(u64::from(value));
                    }
                }
            }
            linear -= const_lo + const_hi;
            let m = word(bits, WordColumn::Message, row);
            let m_lo = Fr::from_u32(m & 0xffff);
            let m_hi = Fr::from_u32(m >> 16);
            let mut kernels = vec![Fr::zero(); keys.len()];
            for reader in &readers[p] {
                let reader_cell = cell as isize + isize::from(reader.cell_delta);
                if reader_cell < 0 || reader_cell >= cells as isize {
                    continue;
                }
                kernels[reader.key] +=
                    scale(reader.slot) * eq_hi[reader_cell as usize] * eq_lo[reader.position];
            }
            let values = keys
                .iter()
                .zip(&value_weights)
                .map(|(key, weights)| {
                    let value = word(bits, key.group, row);
                    let mut acc = Fr::zero();
                    for (j, w) in weights.iter().enumerate() {
                        if (value >> j) & 1 == 1 {
                            acc += *w;
                        }
                    }
                    acc
                })
                .collect();
            // Canonicality: sel · (Σ 2^k canon_k + w_hi) − (r_hi − 1) · sel.
            let canon = (0..CANON_BITS).fold(Fr::zero(), |acc, k| {
                acc + Fr::from_u64(u64::from(bits[CANON + k][row])).mul_pow_2(k)
            });
            let wire_sel = [vk.wire_aligned[row], vk.wire_shifted[row]];
            let mut canon_pins = [Fr::zero(); 4];
            for shifted in 0..2 {
                let sel = gamma_canon[shifted] * Fr::from_u64(u64::from(wire_sel[shifted]));
                linear -= sel * bound;
                let word_hi =
                    canon_forms[shifted]
                        .iter()
                        .fold(Fr::zero(), |acc, (column, weight)| {
                            let value = if *column < COMMITTED {
                                u64::from(bits[*column][row])
                            } else {
                                u64::from(wired_words[*column - COMMITTED][row])
                            };
                            acc + *weight * Fr::from_u64(value)
                        });
                canon_pins[2 * shifted] = sel;
                canon_pins[2 * shifted + 1] = canon + word_hi;
            }
            (
                linear,
                [
                    sel_lo,
                    m_lo,
                    sel_hi,
                    m_hi,
                    canon_pins[0],
                    canon_pins[1],
                    canon_pins[2],
                    canon_pins[3],
                ],
                kernels,
                values,
            )
        };
        let columns: Vec<_> = (0..rows).into_par_iter().map(per_row).collect();
        let reads = (0..keys.len())
            .map(|t| {
                (
                    columns.iter().map(|c| c.2[t]).collect(),
                    columns.iter().map(|c| c.3[t]).collect(),
                )
            })
            .collect();
        let pins = (0..4)
            .map(|p| {
                (
                    columns.iter().map(|c| c.1[2 * p]).collect(),
                    columns.iter().map(|c| c.1[2 * p + 1]).collect(),
                )
            })
            .collect();
        let mut this = Self {
            tau,
            round: 0,
            claim: Fr::zero(),
            eq,
            linear: columns.iter().map(|c| c.0).collect(),
            pins,
            reads,
        };
        drop(columns);
        this.claim = (0..rows).into_par_iter().map(|row| this.value(row)).sum();
        this
    }

    /// The batched sum `Σ_row H(row)`; equals `WiringStatement::input_claim`
    /// for a correctly wired table.
    pub fn input_claim(&self) -> Fr {
        self.claim
    }

    /// `H(row)` on the current (bound) tables.
    fn value(&self, row: usize) -> Fr {
        let pins: Fr = self.pins.iter().map(|(s, v)| s[row] * v[row]).sum();
        self.eq[row] * (self.linear[row] + pins)
            - self.reads.iter().map(|(p, v)| p[row] * v[row]).sum::<Fr>()
    }

    fn tables_mut(&mut self) -> Vec<&mut Vec<Fr>> {
        let mut tables = vec![&mut self.eq, &mut self.linear];
        for (p, v) in self.pins.iter_mut().chain(&mut self.reads) {
            tables.push(p);
            tables.push(v);
        }
        tables
    }

    /// Bind the top row variable (the stream's high-to-low order).
    fn bind(&mut self, r: Fr) {
        self.tables_mut().into_par_iter().for_each(|table| {
            let half = table.len() / 2;
            let (lo, hi) = table.split_at_mut(half);
            for (lo, hi) in lo.iter_mut().zip(hi.iter()) {
                *lo += r * (*hi - *lo);
            }
            table.truncate(half);
        });
        self.round += 1;
    }

    /// Round polynomial of degree 3 from its values at `X = 0, 1, 2, 3`.
    fn round_polynomial(&self) -> UnivariatePoly<Fr> {
        let half = self.eq.len() / 2;
        let evals = (0..half)
            .into_par_iter()
            .fold(
                || [Fr::zero(); 4],
                |mut acc, i| {
                    let at = |table: &[Fr], x: u64| {
                        let (lo, hi) = (table[i], table[i + half]);
                        lo + Fr::from_u64(x) * (hi - lo)
                    };
                    for (x, slot) in acc.iter_mut().enumerate() {
                        let x = x as u64;
                        let reads: Fr = self.reads.iter().map(|(p, v)| at(p, x) * at(v, x)).sum();
                        let pins: Fr = self.pins.iter().map(|(s, v)| at(s, x) * at(v, x)).sum();
                        *slot += at(&self.eq, x) * (at(&self.linear, x) + pins) - reads;
                    }
                    acc
                },
            )
            .reduce(
                || [Fr::zero(); 4],
                |a, b| [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]],
            );
        UnivariatePoly::from_evals(&evals)
    }
}

impl ProveRounds<Fr> for WiringProver {
    fn num_rounds(&self) -> usize {
        self.tau.len()
    }

    fn prove_round(
        &mut self,
        bind: Option<Fr>,
        round: usize,
        _previous_claim: Fr,
    ) -> Result<UnivariatePoly<Fr>, SumcheckError<Fr>> {
        if let Some(r) = bind {
            self.bind(r);
        }
        if round != self.round {
            return Err(SumcheckError::RoundCheckFailed {
                round,
                expected: Fr::from_u64(self.round as u64),
                actual: Fr::from_u64(round as u64),
            });
        }
        Ok(self.round_polynomial())
    }

    fn finish_rounds(&mut self, bind: Fr) -> Result<(), SumcheckError<Fr>> {
        self.bind(bind);
        Ok(())
    }

    fn append_bound_values(&self, _values: &mut Vec<Fr>) -> Result<(), SumcheckError<Fr>> {
        // Wiring uses columns already supplied by the hash-row member.
        Ok(())
    }
}
