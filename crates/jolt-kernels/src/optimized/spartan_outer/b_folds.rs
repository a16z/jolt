//! Fold B's constraint coefficients once, before applying them to each native row.
use super::{DerivedWeights, SpartanOuterRow};
use crate::KernelError;
use jolt_claims::protocols::composed::r1cs::{
    rv64::{
        V_FLAG_DO_NOT_UPDATE_UNEXPANDED_PC, V_FLAG_IS_COMPRESSED, V_IMM, V_LEFT_INSTRUCTION_INPUT,
        V_LEFT_LOOKUP_OPERAND, V_LOOKUP_OUTPUT, V_NEXT_PC, V_NEXT_UNEXPANDED_PC, V_PC, V_PRODUCT,
        V_RAM_ADDRESS, V_RAM_READ_VALUE, V_RAM_WRITE_VALUE, V_RD_WRITE_VALUE,
        V_RIGHT_INSTRUCTION_INPUT, V_RIGHT_LOOKUP_OPERAND, V_RS1_VALUE, V_RS2_VALUE,
        V_UNEXPANDED_PC,
    },
    spartan_outer_opening_columns,
};
use jolt_field::{signed::S256, Accumulator as _, JoltField, WithAccumulator};
use jolt_riscv::CircuitFlags;

const FIRST_COLUMNS: [usize; 11] = [
    V_RAM_ADDRESS,
    V_RAM_READ_VALUE,
    V_RAM_WRITE_VALUE,
    V_RD_WRITE_VALUE,
    V_RS2_VALUE,
    V_LEFT_LOOKUP_OPERAND,
    V_LEFT_INSTRUCTION_INPUT,
    V_LOOKUP_OUTPUT,
    V_NEXT_UNEXPANDED_PC,
    V_NEXT_PC,
    V_PC,
];
// Second-group columns whose native values fit in a `u64`.
const SECOND_WORD_COLUMNS: [usize; 7] = [
    V_RAM_ADDRESS,
    V_RS1_VALUE,
    V_LOOKUP_OUTPUT,
    V_LEFT_INSTRUCTION_INPUT,
    V_RD_WRITE_VALUE,
    V_UNEXPANDED_PC,
    V_NEXT_UNEXPANDED_PC,
];
// Second-group columns with signed or 128-bit native values.
const SECOND_WIDE_COLUMNS: [usize; 4] = [
    V_RIGHT_LOOKUP_OPERAND,
    V_RIGHT_INSTRUCTION_INPUT,
    V_PRODUCT,
    V_IMM,
];
const UNSUPPORTED_COLUMN: &str =
    "Spartan outer B matrix uses a column the native fold does not read";

pub(super) struct PreparedBFolds<F: JoltField> {
    first: [F; 11],
    second_word: [F; 7],
    second_wide: [F; 4],
    negative_wide: [F; 4],
    first_constant: [F; 2],
    second_constant: [F; 4],
}
impl<F: JoltField> PreparedBFolds<F> {
    pub(super) fn new(derived: &DerivedWeights<F>) -> Result<Self, KernelError<F>> {
        // Reuse the canonical matrix's already-combined coefficients. No second
        // list of constraint formulas or assumptions about satisfying rows.
        let unsupported = || KernelError::InvariantViolation {
            reason: UNSUPPORTED_COLUMN,
        };
        let columns = spartan_outer_opening_columns();
        if derived
            .bz_weights
            .iter()
            .any(|weights| weights.len() != columns.len())
        {
            return Err(unsupported());
        }
        for (stream, weights) in derived.bz_weights.iter().enumerate() {
            for (&column, &weight) in columns.iter().zip(weights) {
                let supported = column == V_FLAG_DO_NOT_UPDATE_UNEXPANDED_PC
                    || if stream == 0 {
                        FIRST_COLUMNS.contains(&column)
                    } else {
                        SECOND_WORD_COLUMNS.contains(&column)
                            || SECOND_WIDE_COLUMNS.contains(&column)
                            || column == V_FLAG_IS_COMPRESSED
                    };
                if !supported && weight != F::zero() {
                    return Err(unsupported());
                }
            }
        }
        let weight = |stream: usize, column: usize| {
            columns
                .iter()
                .position(|&candidate| candidate == column)
                .map(|index| derived.bz_weights[stream][index])
                .ok_or_else(unsupported)
        };
        let mut first = [F::zero(); 11];
        let mut second_word = [F::zero(); 7];
        let mut second_wide = [F::zero(); 4];
        for (slot, &column) in first.iter_mut().zip(&FIRST_COLUMNS) {
            *slot = weight(0, column)?;
        }
        for (slot, &column) in second_word.iter_mut().zip(&SECOND_WORD_COLUMNS) {
            *slot = weight(1, column)?;
        }
        for (slot, &column) in second_wide.iter_mut().zip(&SECOND_WIDE_COLUMNS) {
            *slot = weight(1, column)?;
        }
        let first_d = weight(0, V_FLAG_DO_NOT_UPDATE_UNEXPANDED_PC)?;
        let second_d = weight(1, V_FLAG_DO_NOT_UPDATE_UNEXPANDED_PC)?;
        let second_c = weight(1, V_FLAG_IS_COMPRESSED)?;
        Ok(Self {
            first,
            second_word,
            negative_wide: second_wide.map(|value| -value),
            second_wide,
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
        let d = usize::from(row.circuit_flags[CircuitFlags::DoNotUpdateUnexpandedPC]);
        let c = usize::from(row.circuit_flags[CircuitFlags::IsCompressed]);
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
        let second_word_values = [
            row.ram_address.0,
            row.rs1_value.0,
            row.lookup_output.0,
            row.left_instruction_input.0,
            row.rd_write_value.0,
            row.unexpanded_pc.0,
            row.next_unexpanded_pc.0,
        ];
        for (&w, &v) in self.second_word.iter().zip(&second_word_values) {
            if v != 0 {
                second.fmadd_u64(w, v);
            }
        }
        let [lo, hi] = row.product.0.magnitude_limbs();
        let second_wide_values = [
            (row.right_lookup_operand.0, false),
            (
                row.right_instruction_input.0.unsigned_abs(),
                row.right_instruction_input.0 < 0,
            ),
            (
                u128::from(lo) + (u128::from(hi) << 64),
                !row.product.0.is_positive,
            ),
            (row.imm.0.unsigned_abs(), row.imm.0 < 0),
        ];
        for ((w_positive, w_negative), (v, negative)) in self
            .second_wide
            .iter()
            .zip(&self.negative_wide)
            .zip(second_wide_values)
        {
            if v == 0 {
                continue;
            }
            let w = if negative { *w_negative } else { *w_positive };
            if v <= u128::from(u64::MAX) {
                second.fmadd_u64(w, v as u64);
            } else {
                second.fmadd_s256(w, &S256::new([v as u64, (v >> 64) as u64, 0, 0], true));
            }
        }
        [first.reduce(), second.reduce()]
    }
}
