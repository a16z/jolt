mod cpu_constraints;
mod virtual_sequences;

#[macro_export]
macro_rules! template_format {
    (FormatR) => {
        FormatR {
            rd: 1,
            rs1: 2,
            rs2: 3,
        }
    };
    (FormatI) => {
        FormatI {
            rd: 1,
            rs1: 2,
            imm: 1234,
        }
    };
    (FormatU) => {
        FormatU { rd: 1, imm: 1234 }
    };
    (FormatB) => {
        FormatB {
            rs1: 2,
            rs2: 3,
            imm: 1234,
        }
    };
    (FormatJ) => {
        FormatJ { rd: 1, imm: 1234 }
    };
    (FormatLoad) => {
        FormatLoad {
            rd: 1,
            rs1: 2,
            imm: 1234,
        }
    };
    (FormatS) => {
        FormatS {
            rs1: 2,
            rs2: 3,
            imm: 1234,
        }
    };
    (AssertAlignFormat) => {
        AssertAlignFormat { rs1: 2, imm: 1234 }
    };
    (FormatVirtualRightShiftI) => {
        FormatVirtualRightShiftI {
            rd: 1,
            rs1: 2,
            imm: 1234,
        }
    };
    (FormatVirtualRightShiftR) => {
        FormatVirtualRightShiftR {
            rd: 1,
            rs1: 2,
            rs2: 3,
        }
    };
    (FormatFence) => {
        FormatFence {}
    };
}
