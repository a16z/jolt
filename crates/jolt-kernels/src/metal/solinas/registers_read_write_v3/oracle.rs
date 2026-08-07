use jolt_field::Field;

use super::abi::{RegisterGeometry, REGISTER_CSR_BLOCK_CYCLES, REGISTER_CSR_COLUMNS};
use super::owner::{
    CertifiedRegisterOwner, RegisterCsr256, RegisterRead, RegisterRow, RegisterWrite,
};
use super::RegistersRwV3Error;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct DenseCell<F> {
    val: F,
    ra: F,
    wa: F,
}

impl<F: Field> DenseCell<F> {
    pub(crate) const fn val(self) -> F {
        self.val
    }

    pub(crate) const fn ra(self) -> F {
        self.ra
    }

    pub(crate) const fn wa(self) -> F {
        self.wa
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct CycleRoundReference<F> {
    q_zero: F,
    q_infinity: F,
    evaluations: [F; 4],
    round_sum: F,
}

impl<F: Field> CycleRoundReference<F> {
    pub(crate) const fn q_zero(self) -> F {
        self.q_zero
    }

    /// The coefficient of `t^2` in the quadratic inner factor.
    pub(crate) const fn q_infinity(self) -> F {
        self.q_infinity
    }

    pub(crate) const fn evaluations(self) -> [F; 4] {
        self.evaluations
    }

    pub(crate) const fn round_sum(self) -> F {
        self.round_sum
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Round8Junction<F> {
    round: CycleRoundReference<F>,
    rows: usize,
    cells: Vec<DenseCell<F>>,
    rd_inc: Vec<F>,
}

impl<F: Field> Round8Junction<F> {
    pub(crate) const fn round(&self) -> CycleRoundReference<F> {
        self.round
    }

    pub(crate) const fn rows(&self) -> usize {
        self.rows
    }

    pub(crate) fn cells(&self) -> &[DenseCell<F>] {
        &self.cells
    }

    pub(crate) fn rd_inc(&self) -> &[F] {
        &self.rd_inc
    }
}

/// Dense, cycle-row oracle built directly from raw register rows.
///
/// This path does not inspect CSR offsets or use sparse merge code. It is the
/// byte-parity control for the CSR-native evaluator and future MSL kernels.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct DenseRegisterRelation<F> {
    geometry: RegisterGeometry,
    rows_remaining: usize,
    rounds_bound: usize,
    val: Vec<F>,
    ra: Vec<F>,
    wa: Vec<F>,
    eq: Vec<F>,
    rd_inc: Vec<F>,
}

impl<F: Field> DenseRegisterRelation<F> {
    pub(crate) fn build(
        rows: &[RegisterRow],
        initial_values: &[u64; REGISTER_CSR_COLUMNS],
        r_cycle: &[F],
        gamma: F,
        rd_inc: &[F],
    ) -> Result<Self, RegistersRwV3Error> {
        let geometry = RegisterGeometry::new(rows.len())?;
        require_input_length("cycle equality point", geometry.log_t(), r_cycle.len())?;
        require_input_length("rd increment", geometry.cycles(), rd_inc.len())?;
        let cells = geometry
            .cycles()
            .checked_mul(REGISTER_CSR_COLUMNS)
            .ok_or(RegistersRwV3Error::SizeOverflow("dense relation cells"))?;
        let mut val = vec![F::zero(); cells];
        let mut ra = vec![F::zero(); cells];
        let mut wa = vec![F::zero(); cells];
        let mut state = *initial_values;
        let gamma_sq = gamma * gamma;

        for (cycle, row) in rows.iter().copied().enumerate() {
            let base = cycle
                .checked_mul(REGISTER_CSR_COLUMNS)
                .ok_or(RegistersRwV3Error::SizeOverflow("dense row offset"))?;
            for (register, value) in state.iter().copied().enumerate() {
                val[base + register] = F::from_u64(value);
            }
            if let Some(read) = row.rs1() {
                let register = validate_dense_read(cycle, "rs1", read, &state)?;
                ra[base + register] += gamma;
            }
            if let Some(read) = row.rs2() {
                let register = validate_dense_read(cycle, "rs2", read, &state)?;
                ra[base + register] += gamma_sq;
            }
            let expected_inc = if let Some(write) = row.rd() {
                let register = validate_dense_write(cycle, write, &state)?;
                wa[base + register] = F::one();
                state[register] = write.post_value();
                F::from_i128(i128::from(write.post_value()) - i128::from(write.pre_value()))
            } else {
                F::zero()
            };
            if rd_inc[cycle] != expected_inc {
                return Err(RegistersRwV3Error::IncrementMismatch { cycle });
            }
        }

        Ok(Self {
            geometry,
            rows_remaining: geometry.cycles(),
            rounds_bound: 0,
            val,
            ra,
            wa,
            eq: eq_evaluations(r_cycle),
            rd_inc: rd_inc.to_vec(),
        })
    }

    pub(crate) const fn rounds_bound(&self) -> usize {
        self.rounds_bound
    }

    pub(crate) const fn rows_remaining(&self) -> usize {
        self.rows_remaining
    }

    pub(crate) fn cycle_round(&self) -> Result<CycleRoundReference<F>, RegistersRwV3Error> {
        if self.rows_remaining < 2 {
            return Err(RegistersRwV3Error::CycleRoundUnavailable {
                remaining_rows: self.rows_remaining,
            });
        }
        let parents = self.rows_remaining / 2;
        let mut q_zero = F::zero();
        let mut q_infinity = F::zero();
        let mut evaluations = [F::zero(); 4];

        for parent in 0..parents {
            let even_row = 2 * parent;
            let odd_row = even_row + 1;
            let eq_0 = self.eq[even_row];
            let eq_m = self.eq[odd_row] - eq_0;
            let head = eq_0 + self.eq[odd_row];
            let inc_0 = self.rd_inc[even_row];
            let inc_m = self.rd_inc[odd_row] - inc_0;
            let mut parent_q_zero = F::zero();
            let mut parent_q_infinity = F::zero();
            let mut parent_evaluations = [F::zero(); 4];

            for register in 0..REGISTER_CSR_COLUMNS {
                let even = even_row * REGISTER_CSR_COLUMNS + register;
                let odd = odd_row * REGISTER_CSR_COLUMNS + register;
                let val_0 = self.val[even];
                let val_m = self.val[odd] - val_0;
                let ra_0 = self.ra[even];
                let ra_m = self.ra[odd] - ra_0;
                let wa_0 = self.wa[even];
                let wa_m = self.wa[odd] - wa_0;

                parent_q_zero += ra_0 * val_0 + wa_0 * (val_0 + inc_0);
                parent_q_infinity += ra_m * val_m + wa_m * (val_m + inc_m);
                for (sample, evaluation) in parent_evaluations.iter_mut().enumerate() {
                    let t = F::from_u64(sample as u64);
                    let val_t = val_0 + t * val_m;
                    let ra_t = ra_0 + t * ra_m;
                    let wa_t = wa_0 + t * wa_m;
                    let inc_t = inc_0 + t * inc_m;
                    *evaluation += ra_t * val_t + wa_t * (val_t + inc_t);
                }
            }

            q_zero += head * parent_q_zero;
            q_infinity += head * parent_q_infinity;
            for (sample, evaluation) in evaluations.iter_mut().enumerate() {
                let t = F::from_u64(sample as u64);
                *evaluation += (eq_0 + t * eq_m) * parent_evaluations[sample];
            }
        }

        Ok(CycleRoundReference {
            q_zero,
            q_infinity,
            round_sum: evaluations[0] + evaluations[1],
            evaluations,
        })
    }

    pub(crate) fn bind(&mut self, challenge: F) -> Result<(), RegistersRwV3Error> {
        if self.rows_remaining < 2 {
            return Err(RegistersRwV3Error::CycleRoundUnavailable {
                remaining_rows: self.rows_remaining,
            });
        }
        bind_dense_cells(&mut self.val, self.rows_remaining, challenge);
        bind_dense_cells(&mut self.ra, self.rows_remaining, challenge);
        bind_dense_cells(&mut self.wa, self.rows_remaining, challenge);
        bind_scalar_table(&mut self.eq, challenge);
        bind_scalar_table(&mut self.rd_inc, challenge);
        self.rows_remaining /= 2;
        self.rounds_bound += 1;
        Ok(())
    }

    pub(crate) fn round8_junction(&self) -> Result<Round8Junction<F>, RegistersRwV3Error> {
        if self.rounds_bound != 8 {
            return Err(RegistersRwV3Error::JunctionRoundMismatch {
                rounds_bound: self.rounds_bound,
            });
        }
        let expected_rows = self.geometry.blocks();
        if self.rows_remaining != expected_rows {
            return Err(RegistersRwV3Error::InputLength {
                name: "round-8 dense rows",
                expected: expected_rows,
                got: self.rows_remaining,
            });
        }
        let cells = self
            .val
            .iter()
            .copied()
            .zip(self.ra.iter().copied())
            .zip(self.wa.iter().copied())
            .map(|((val, ra), wa)| DenseCell { val, ra, wa })
            .collect();
        Ok(Round8Junction {
            round: self.cycle_round()?,
            rows: self.rows_remaining,
            cells,
            rd_inc: self.rd_inc.clone(),
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SparseEntry<F> {
    row: usize,
    register: u8,
    val: F,
    ra: F,
    wa: F,
    prev_value: u64,
    next_value: u64,
}

/// CSR-native reference for raw rounds 0 through 8.
///
/// It reconstructs touched cells only, carries constant values across absent
/// children, and emits the round-8 dense source before challenge `c_8` binds.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct SparseRegisterRelation<'a, F> {
    csr: &'a RegisterCsr256,
    rows_remaining: usize,
    rounds_bound: usize,
    entries: Vec<SparseEntry<F>>,
    eq: Vec<F>,
    rd_inc: Vec<F>,
}

impl<'a, F: Field> SparseRegisterRelation<'a, F> {
    pub(crate) fn build(
        owner: &'a CertifiedRegisterOwner,
        r_cycle: &[F],
        gamma: F,
        rd_inc: &[F],
    ) -> Result<Self, RegistersRwV3Error> {
        let csr = owner.csr();
        let geometry = csr.geometry();
        require_input_length("cycle equality point", geometry.log_t(), r_cycle.len())?;
        require_input_length("rd increment", geometry.cycles(), rd_inc.len())?;
        let (mut entries, expected_inc) = csr_sparse_entries(csr, gamma)?;
        for (cycle, (&expected, &got)) in expected_inc.iter().zip(rd_inc).enumerate() {
            if expected != got {
                return Err(RegistersRwV3Error::IncrementMismatch { cycle });
            }
        }
        entries.sort_unstable_by_key(|entry| (entry.row, entry.register));
        Ok(Self {
            csr,
            rows_remaining: geometry.cycles(),
            rounds_bound: 0,
            entries,
            eq: eq_evaluations(r_cycle),
            rd_inc: rd_inc.to_vec(),
        })
    }

    pub(crate) const fn rounds_bound(&self) -> usize {
        self.rounds_bound
    }

    pub(crate) const fn rows_remaining(&self) -> usize {
        self.rows_remaining
    }

    pub(crate) fn cycle_round(&self) -> Result<CycleRoundReference<F>, RegistersRwV3Error> {
        if self.rows_remaining < 2 {
            return Err(RegistersRwV3Error::CycleRoundUnavailable {
                remaining_rows: self.rows_remaining,
            });
        }
        let parents = self.rows_remaining / 2;
        let mut q_zero = F::zero();
        let mut q_infinity = F::zero();
        let mut evaluations = [F::zero(); 4];

        for parent in 0..parents {
            let even_row = 2 * parent;
            let odd_row = even_row + 1;
            let even = row_entries(&self.entries, even_row);
            let odd = row_entries(&self.entries, odd_row);
            let eq_0 = self.eq[even_row];
            let eq_m = self.eq[odd_row] - eq_0;
            let head = eq_0 + self.eq[odd_row];
            let inc_0 = self.rd_inc[even_row];
            let inc_m = self.rd_inc[odd_row] - inc_0;
            let mut parent_q_zero = F::zero();
            let mut parent_q_infinity = F::zero();
            let mut parent_evaluations = [F::zero(); 4];
            for_each_sparse_pair(even, odd, |even, odd| {
                if let Some(lines) = pair_linears(even, odd) {
                    parent_q_zero += lines.ra_0 * lines.val_0 + lines.wa_0 * (lines.val_0 + inc_0);
                    parent_q_infinity +=
                        lines.ra_m * lines.val_m + lines.wa_m * (lines.val_m + inc_m);
                    for (sample, evaluation) in parent_evaluations.iter_mut().enumerate() {
                        let t = F::from_u64(sample as u64);
                        let val_t = lines.val_0 + t * lines.val_m;
                        let ra_t = lines.ra_0 + t * lines.ra_m;
                        let wa_t = lines.wa_0 + t * lines.wa_m;
                        let inc_t = inc_0 + t * inc_m;
                        *evaluation += ra_t * val_t + wa_t * (val_t + inc_t);
                    }
                }
            });
            q_zero += head * parent_q_zero;
            q_infinity += head * parent_q_infinity;
            for (sample, evaluation) in evaluations.iter_mut().enumerate() {
                let t = F::from_u64(sample as u64);
                *evaluation += (eq_0 + t * eq_m) * parent_evaluations[sample];
            }
        }

        Ok(CycleRoundReference {
            q_zero,
            q_infinity,
            round_sum: evaluations[0] + evaluations[1],
            evaluations,
        })
    }

    pub(crate) fn bind(&mut self, challenge: F) -> Result<(), RegistersRwV3Error> {
        if self.rows_remaining < 2 {
            return Err(RegistersRwV3Error::CycleRoundUnavailable {
                remaining_rows: self.rows_remaining,
            });
        }
        let parents = self.rows_remaining / 2;
        let mut bound = Vec::with_capacity(self.entries.len());
        for parent in 0..parents {
            let even = row_entries(&self.entries, 2 * parent);
            let odd = row_entries(&self.entries, 2 * parent + 1);
            for_each_sparse_pair(even, odd, |even, odd| {
                if let Some(entry) = bind_pair(parent, even, odd, challenge) {
                    bound.push(entry);
                }
            });
        }
        bind_scalar_table(&mut self.eq, challenge);
        bind_scalar_table(&mut self.rd_inc, challenge);
        self.entries = bound;
        self.rows_remaining = parents;
        self.rounds_bound += 1;
        Ok(())
    }

    pub(crate) fn round8_junction(&self) -> Result<Round8Junction<F>, RegistersRwV3Error> {
        if self.rounds_bound != 8 {
            return Err(RegistersRwV3Error::JunctionRoundMismatch {
                rounds_bound: self.rounds_bound,
            });
        }
        let expected_rows = self.csr.geometry().blocks();
        if self.rows_remaining != expected_rows {
            return Err(RegistersRwV3Error::InputLength {
                name: "round-8 sparse rows",
                expected: expected_rows,
                got: self.rows_remaining,
            });
        }
        let cell_count = self
            .rows_remaining
            .checked_mul(REGISTER_CSR_COLUMNS)
            .ok_or(RegistersRwV3Error::SizeOverflow("junction cells"))?;
        let mut cells = Vec::with_capacity(cell_count);
        for block in 0..self.rows_remaining {
            for register in 0..REGISTER_CSR_COLUMNS {
                let start_value = self.csr.column(block, register)?.start_value();
                cells.push(DenseCell {
                    val: F::from_u64(start_value),
                    ra: F::zero(),
                    wa: F::zero(),
                });
            }
        }
        for entry in &self.entries {
            let index = entry
                .row
                .checked_mul(REGISTER_CSR_COLUMNS)
                .and_then(|base| base.checked_add(usize::from(entry.register)))
                .ok_or(RegistersRwV3Error::SizeOverflow("junction cell index"))?;
            let length = cells.len();
            let cell = cells
                .get_mut(index)
                .ok_or(RegistersRwV3Error::IndexOutOfRange {
                    name: "junction cells",
                    index,
                    length,
                })?;
            *cell = DenseCell {
                val: entry.val,
                ra: entry.ra,
                wa: entry.wa,
            };
        }
        Ok(Round8Junction {
            round: self.cycle_round()?,
            rows: self.rows_remaining,
            cells,
            rd_inc: self.rd_inc.clone(),
        })
    }
}

#[derive(Clone, Copy)]
struct PairLinears<F> {
    val_0: F,
    val_m: F,
    ra_0: F,
    ra_m: F,
    wa_0: F,
    wa_m: F,
}

fn pair_linears<F: Field>(
    even: Option<&SparseEntry<F>>,
    odd: Option<&SparseEntry<F>>,
) -> Option<PairLinears<F>> {
    let (val_0, val_1, ra_0, ra_1, wa_0, wa_1) = match (even, odd) {
        (Some(even), Some(odd)) => (even.val, odd.val, even.ra, odd.ra, even.wa, odd.wa),
        (Some(even), None) => (
            even.val,
            F::from_u64(even.next_value),
            even.ra,
            F::zero(),
            even.wa,
            F::zero(),
        ),
        (None, Some(odd)) => (
            F::from_u64(odd.prev_value),
            odd.val,
            F::zero(),
            odd.ra,
            F::zero(),
            odd.wa,
        ),
        (None, None) => return None,
    };
    Some(PairLinears {
        val_0,
        val_m: val_1 - val_0,
        ra_0,
        ra_m: ra_1 - ra_0,
        wa_0,
        wa_m: wa_1 - wa_0,
    })
}

fn bind_pair<F: Field>(
    row: usize,
    even: Option<&SparseEntry<F>>,
    odd: Option<&SparseEntry<F>>,
    challenge: F,
) -> Option<SparseEntry<F>> {
    let lines = pair_linears(even, odd)?;
    let source = even.or(odd)?;
    let (prev_value, next_value) = match (even, odd) {
        (Some(even), Some(odd)) => (even.prev_value, odd.next_value),
        (Some(even), None) => (even.prev_value, even.next_value),
        (None, Some(odd)) => (odd.prev_value, odd.next_value),
        (None, None) => return None,
    };
    Some(SparseEntry {
        row,
        register: source.register,
        val: lines.val_0 + challenge * lines.val_m,
        ra: lines.ra_0 + challenge * lines.ra_m,
        wa: lines.wa_0 + challenge * lines.wa_m,
        prev_value,
        next_value,
    })
}

fn for_each_sparse_pair<F>(
    even: &[SparseEntry<F>],
    odd: &[SparseEntry<F>],
    mut visit: impl FnMut(Option<&SparseEntry<F>>, Option<&SparseEntry<F>>),
) {
    let mut even_index = 0usize;
    let mut odd_index = 0usize;
    while even_index < even.len() && odd_index < odd.len() {
        match even[even_index].register.cmp(&odd[odd_index].register) {
            core::cmp::Ordering::Equal => {
                visit(Some(&even[even_index]), Some(&odd[odd_index]));
                even_index += 1;
                odd_index += 1;
            }
            core::cmp::Ordering::Less => {
                visit(Some(&even[even_index]), None);
                even_index += 1;
            }
            core::cmp::Ordering::Greater => {
                visit(None, Some(&odd[odd_index]));
                odd_index += 1;
            }
        }
    }
    for entry in &even[even_index..] {
        visit(Some(entry), None);
    }
    for entry in &odd[odd_index..] {
        visit(None, Some(entry));
    }
}

fn row_entries<F>(entries: &[SparseEntry<F>], row: usize) -> &[SparseEntry<F>] {
    let start = entries.partition_point(|entry| entry.row < row);
    let end = entries[start..].partition_point(|entry| entry.row == row) + start;
    &entries[start..end]
}

fn csr_sparse_entries<F: Field>(
    csr: &RegisterCsr256,
    gamma: F,
) -> Result<(Vec<SparseEntry<F>>, Vec<F>), RegistersRwV3Error> {
    let geometry = csr.geometry();
    let mut entries = Vec::with_capacity(csr.event_counts().checked_total()?);
    let mut rd_inc = vec![F::zero(); geometry.cycles()];
    let gamma_sq = gamma * gamma;

    for block in 0..geometry.blocks() {
        let block_len = csr.block_len(block)?;
        let block_start = block
            .checked_mul(REGISTER_CSR_BLOCK_CYCLES)
            .ok_or(RegistersRwV3Error::SizeOverflow("CSR oracle block"))?;
        for register in 0..REGISTER_CSR_COLUMNS {
            let column = csr.column(block, register)?;
            let mut value = column.start_value();
            let mut rs1_index = 0usize;
            let mut rs2_index = 0usize;
            let mut rd_index = 0usize;
            for position in 0..block_len {
                let local = position as u8;
                let has_rs1 = column.rs1_positions().get(rs1_index).copied() == Some(local);
                let has_rs2 = column.rs2_positions().get(rs2_index).copied() == Some(local);
                let has_rd = column.rd_positions().get(rd_index).copied() == Some(local);
                if !has_rs1 && !has_rs2 && !has_rd {
                    continue;
                }
                let previous = value;
                let mut ra = F::zero();
                if has_rs1 {
                    ra += gamma;
                    rs1_index += 1;
                }
                if has_rs2 {
                    ra += gamma_sq;
                    rs2_index += 1;
                }
                let wa = if has_rd { F::one() } else { F::zero() };
                if has_rd {
                    let post = column.rd_post_values().get(rd_index).copied().ok_or(
                        RegistersRwV3Error::IndexOutOfRange {
                            name: "rd post values",
                            index: rd_index,
                            length: column.rd_post_values().len(),
                        },
                    )?;
                    let cycle = block_start + position;
                    rd_inc[cycle] = F::from_i128(i128::from(post) - i128::from(previous));
                    value = post;
                    rd_index += 1;
                }
                entries.push(SparseEntry {
                    row: block_start + position,
                    register: register as u8,
                    val: F::from_u64(previous),
                    ra,
                    wa,
                    prev_value: previous,
                    next_value: value,
                });
            }
        }
    }
    Ok((entries, rd_inc))
}

fn validate_dense_read(
    cycle: usize,
    access: &'static str,
    read: RegisterRead,
    state: &[u64; REGISTER_CSR_COLUMNS],
) -> Result<usize, RegistersRwV3Error> {
    let register = usize::from(read.register());
    if register >= REGISTER_CSR_COLUMNS {
        return Err(RegistersRwV3Error::InvalidRegister {
            cycle,
            access,
            register: read.register(),
        });
    }
    let expected = state[register];
    if read.value() != expected {
        return Err(RegistersRwV3Error::ReadValueMismatch {
            cycle,
            access,
            register: read.register(),
            expected,
            got: read.value(),
        });
    }
    Ok(register)
}

fn validate_dense_write(
    cycle: usize,
    write: RegisterWrite,
    state: &[u64; REGISTER_CSR_COLUMNS],
) -> Result<usize, RegistersRwV3Error> {
    let register = usize::from(write.register());
    if register >= REGISTER_CSR_COLUMNS {
        return Err(RegistersRwV3Error::InvalidRegister {
            cycle,
            access: "rd",
            register: write.register(),
        });
    }
    let expected = state[register];
    if write.pre_value() != expected {
        return Err(RegistersRwV3Error::WritePreValueMismatch {
            cycle,
            register: write.register(),
            expected,
            got: write.pre_value(),
        });
    }
    Ok(register)
}

fn eq_evaluations<F: Field>(point: &[F]) -> Vec<F> {
    let mut evaluations = vec![F::one()];
    for &coordinate in point {
        let mut next = Vec::with_capacity(evaluations.len() * 2);
        for &prefix in &evaluations {
            next.push(prefix * (F::one() - coordinate));
            next.push(prefix * coordinate);
        }
        evaluations = next;
    }
    evaluations
}

fn bind_dense_cells<F: Field>(table: &mut Vec<F>, rows: usize, challenge: F) {
    let parents = rows / 2;
    for parent in 0..parents {
        for register in 0..REGISTER_CSR_COLUMNS {
            let destination = parent * REGISTER_CSR_COLUMNS + register;
            let even = 2 * parent * REGISTER_CSR_COLUMNS + register;
            let odd = even + REGISTER_CSR_COLUMNS;
            let value = table[even] + challenge * (table[odd] - table[even]);
            table[destination] = value;
        }
    }
    table.truncate(parents * REGISTER_CSR_COLUMNS);
}

fn bind_scalar_table<F: Field>(table: &mut Vec<F>, challenge: F) {
    let parents = table.len() / 2;
    for parent in 0..parents {
        let even = table[2 * parent];
        table[parent] = even + challenge * (table[2 * parent + 1] - even);
    }
    table.truncate(parents);
}

fn require_input_length(
    name: &'static str,
    expected: usize,
    got: usize,
) -> Result<(), RegistersRwV3Error> {
    if expected == got {
        Ok(())
    } else {
        Err(RegistersRwV3Error::InputLength {
            name,
            expected,
            got,
        })
    }
}
