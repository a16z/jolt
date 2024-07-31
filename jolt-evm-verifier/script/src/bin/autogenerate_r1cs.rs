use alloy_primitives::hex;
use jolt_core::field::JoltField;
use jolt_core::r1cs::key::SparseConstraints;
use jolt_core::r1cs::key::SparseEqualityItem;

// This code is provided as an example of how to turn the sparse constraints into solidity, and was the base
// for the code in R1CSMatrix, however because the key is not expected to change often full auto-generation is not
// provided and some tweaking is needed to produce sol from this.
#[allow(dead_code)]
fn partial_autogenerate<F: JoltField>(
    constraints: &SparseConstraints<F>,
    non_uni: Option<&SparseEqualityItem<F>>,
) {
    let bytes = |from: F| -> Vec<u8> {
        let mut buf = vec![];
        from.serialize_uncompressed(&mut buf).unwrap();
        buf.into_iter().rev().collect()
    };

    print!("Fr running = Fr.wrap(0);\nFr rv = Fr.wrap(0);\n");

    for (row, col, coeff) in constraints.vars.iter() {
        println!(
            "rv += Fr.wrap(0x{})*row[{:?}]*col[{:?}];",
            hex::encode(bytes(*coeff)),
            row,
            col
        );
    }

    print!("running += rv*eq_rx_ry_step;\n// then we do constant col\nFr rc = Fr.wrap(0);");

    for (row, coeff) in constraints.consts.iter() {
        println!(
            "rc += Fr.wrap(0x{})*row[{:?}];",
            hex::encode(bytes(*coeff)),
            row
        );
    }
    println!("running += rc*col_eq_constant;");

    if non_uni.is_some() {
        println!("Fr rnu = Fr.wrap(0);");

        for (col, offset, coeff) in non_uni.unwrap().offset_vars.iter() {
            if *offset {
                println!(
                    "rnu += Fr.wrap(0x{})*col[{:?}]*eq_step_offset_1;",
                    hex::encode(bytes(*coeff)),
                    col
                );
            } else {
                println!(
                    "rnu += Fr.wrap(0x{})*col[{:?}]*eq_rx_ry_step;",
                    hex::encode(bytes(*coeff)),
                    col
                );
            }
        }
        println!(
            "rnu += col_eq_const*Fr.wrap(0x{});",
            hex::encode(bytes(non_uni.unwrap().constant))
        );
        println!("running += rnu * row_constr_eq_non_uni;");
    }
    println!("return(running);");
}

fn main() {
    // TODO - Have example code here rather than in the verifier
}
