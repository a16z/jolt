mod cpu_constraints;
mod virtual_sequences;

#[macro_export]
macro_rules! test_instruction {
    ($instr:ident, $test_fn:ident, $operands:path $(, $field:ident : $value:expr )* $(,)?) => {
        paste::paste! {
            #[test]
            #[allow(nonstandard_style)]
            fn [<test_ $instr>]() {
                let instr = Instruction::$instr($instr {
                    operands: test_instruction!(@ $operands),
                    $($field: $value,)*
                    // unused by solver
                    address: 8,
                    is_compressed: false,
                    is_first_in_sequence: false,
                    virtual_sequence_remaining: None,
                });
                $test_fn(stringify!(instr), &instr);
            }
        }
    };
    // Expand operands
    (@ FormatR) => {
        FormatR { rd: 1, rs1: 2, rs2: 3 }
    };
    (@ FormatI) => {
        FormatI { rd: 1, rs1: 2, imm: 1234 }
    };
    (@ FormatU) => {
        FormatU { rd: 1, imm: 1234 }
    };
    (@ FormatB) => {
        FormatB { rs1: 2, rs2: 3, imm: 1234 }
    };
    (@ FormatJ) => {
        FormatJ { rd: 1, imm: 1234 }
    };
    (@ FormatLoad) => {
        FormatLoad { rd: 1, rs1: 2, imm: 1234 }
    };
    (@ FormatS) => {
        FormatS { rs1: 2, rs2: 3, imm: 1234 }
    };
    (@ AssertAlignFormat) => {
        AssertAlignFormat { rs1: 2, imm: 1234 }
    };
    (@ FormatVirtualRightShiftI) => {
        FormatVirtualRightShiftI { rd: 1, rs1: 2, imm: 1234 }
    };
    (@ FormatVirtualRightShiftR) => {
        FormatVirtualRightShiftR { rd: 1, rs1: 2, rs2: 3 }
    };
}
