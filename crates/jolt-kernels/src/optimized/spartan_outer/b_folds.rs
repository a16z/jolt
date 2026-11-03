//! Fold B's constraint coefficients once, before applying them to each native row.
use super::{DerivedWeights, SpartanOuterRow};
use jolt_claims::protocols::composed::r1cs::{rv64, spartan_outer_opening_columns};
use jolt_field::{signed::S256, Accumulator as _, JoltField, WithAccumulator};

const FIRST_COLUMNS: [usize; 11] = [
    rv64::V_RAM_ADDRESS,
    rv64::V_RAM_READ_VALUE,
    rv64::V_RAM_WRITE_VALUE,
    rv64::V_RD_WRITE_VALUE,
    rv64::V_RS2_VALUE,
    rv64::V_LEFT_LOOKUP_OPERAND,
    rv64::V_LEFT_INSTRUCTION_INPUT,
    rv64::V_LOOKUP_OUTPUT,
    rv64::V_NEXT_UNEXPANDED_PC,
    rv64::V_NEXT_PC,
    rv64::V_PC,
];
const SECOND_COLUMNS: [usize; 11] = [
    rv64::V_RAM_ADDRESS,
    rv64::V_RS1_VALUE,
    rv64::V_LOOKUP_OUTPUT,
    rv64::V_RIGHT_LOOKUP_OPERAND,
    rv64::V_LEFT_INSTRUCTION_INPUT,
    rv64::V_RIGHT_INSTRUCTION_INPUT,
    rv64::V_PRODUCT,
    rv64::V_RD_WRITE_VALUE,
    rv64::V_UNEXPANDED_PC,
    rv64::V_NEXT_UNEXPANDED_PC,
    rv64::V_IMM,
];

pub(super) struct PreparedBFolds<F: JoltField> {
    first: [F; 11],
    second: [F; 11],
    negative: [F; 11],
    first_constant: [F; 2],
    second_constant: [F; 4],
}
impl<F: JoltField> PreparedBFolds<F> {
    pub(super) fn new(derived: &DerivedWeights<F>) -> Option<Self> {
        // Reuse the canonical matrix's already-combined coefficients. No second
        // list of constraint formulas or assumptions about satisfying rows.
        let columns = spartan_outer_opening_columns();
        if derived
            .bz_weights
            .iter()
            .any(|weights| weights.len() != columns.len())
        {
            return None;
        }
        for (stream, selected) in [&FIRST_COLUMNS, &SECOND_COLUMNS].into_iter().enumerate() {
            for (&column, &weight) in columns.iter().zip(&derived.bz_weights[stream]) {
                let supported = selected.contains(&column)
                    || column == rv64::V_FLAG_DO_NOT_UPDATE_UNEXPANDED_PC
                    || (stream == 1 && column == rv64::V_FLAG_IS_COMPRESSED);
                if !supported && weight != F::zero() {
                    return None;
                }
            }
        }
        let weight = |stream: usize, column: usize| {
            columns
                .iter()
                .position(|&candidate| candidate == column)
                .map(|index| derived.bz_weights[stream][index])
        };
        let mut first = [F::zero(); 11];
        let mut second = [F::zero(); 11];
        for (slot, &column) in first.iter_mut().zip(&FIRST_COLUMNS) {
            *slot = weight(0, column)?;
        }
        for (slot, &column) in second.iter_mut().zip(&SECOND_COLUMNS) {
            *slot = weight(1, column)?;
        }
        let first_d = weight(0, rv64::V_FLAG_DO_NOT_UPDATE_UNEXPANDED_PC)?;
        let second_d = weight(1, rv64::V_FLAG_DO_NOT_UPDATE_UNEXPANDED_PC)?;
        let second_c = weight(1, rv64::V_FLAG_IS_COMPRESSED)?;
        Some(Self {
            first,
            negative: second.map(|value| -value),
            second,
            first_constant: [derived.bz_constant[0], derived.bz_constant[0] + first_d],
            second_constant: std::array::from_fn(|index| {
                derived.bz_constant[1]
                    + if index & 1 != 0 { second_d } else { F::zero() }
                    + if index & 2 != 0 { second_c } else { F::zero() }
            }),
        })
    }
    #[inline(never)]
    pub(super) fn fold(&self, row: &SpartanOuterRow) -> [F; 2] {
        let d = usize::from(row.do_not_update_unexpanded_pc.0);
        let c = usize::from(row.is_compressed.0);
        let first_values = [
            row.ram_address.0,
            row.ram_read_value.0,
            row.ram_write_value.0,
            row.rd_write_value.0,
            row.rs2_value.0,
            row.left_lookup_operand.0,
            row.left_instruction_input.0,
            row.lookup_output.0,
            row.next_unexpanded_pc.0,
            row.next_pc.0,
            row.pc.0,
        ];
        // Defer the u64 scalar products on both Fr and Fp128. Fr's general
        // accumulator would reduce each product separately.
        let mut first = <F as WithAccumulator>::SignedProductAccumulator::default();
        first.add(self.first_constant[d]);
        for (&w, &v) in self.first.iter().zip(&first_values) {
            if v != 0 {
                first.fmadd_u64(w, v);
            }
        }
        let mut second = <F as WithAccumulator>::SignedProductAccumulator::default();
        second.add(self.second_constant[d + 2 * c]);
        for (i, v) in [
            (0, row.ram_address.0),
            (1, row.rs1_value.0),
            (2, row.lookup_output.0),
            (4, row.left_instruction_input.0),
            (7, row.rd_write_value.0),
            (8, row.unexpanded_pc.0),
            (9, row.next_unexpanded_pc.0),
        ] {
            if v != 0 {
                second.fmadd_u64(self.second[i], v);
            }
        }
        let [lo, hi] = row.product.0.magnitude_limbs();
        for (i, v, negative) in [
            (3, row.right_lookup_operand.0, false),
            (
                5,
                row.right_instruction_input.0.unsigned_abs(),
                row.right_instruction_input.0 < 0,
            ),
            (
                6,
                u128::from(lo) + (u128::from(hi) << 64),
                !row.product.0.is_positive,
            ),
            (10, row.imm.0.unsigned_abs(), row.imm.0 < 0),
        ] {
            if v == 0 {
                continue;
            }
            let w = if negative {
                self.negative[i]
            } else {
                self.second[i]
            };
            if v <= u128::from(u64::MAX) {
                second.fmadd_u64(w, v as u64);
            } else {
                second.fmadd_s256(w, &S256::new([v as u64, (v >> 64) as u64, 0, 0], true));
            }
        }
        [first.reduce(), second.reduce()]
    }
}
