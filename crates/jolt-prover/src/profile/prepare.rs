use std::fs;
use std::path::PathBuf;

use clap::Args;
use jolt_host::Program;

use super::Workload;

/// Build artifacts only; this command never traces, sets up a PCS, or proves.
#[derive(Debug, Args)]
pub struct PrepareArgs {
    #[arg(long, value_enum)]
    pub name: Workload,
    #[arg(long, value_delimiter = ',', required = true)]
    pub operations: Vec<u32>,
    /// A private directory inside this checkout's target directory.
    #[arg(long)]
    pub guest_target_dir: PathBuf,
    #[arg(long)]
    pub out: PathBuf,
}

/// Write guest.elf and one input-N.bin per requested operation count.
pub fn prepare(args: &PrepareArgs) {
    assert!(args.operations.iter().all(|&count| count > 0));
    fs::create_dir_all(&args.out).expect("create prepared guest directory");
    let mut program = Program::new(&format!("{}-guest", args.name.as_str()));
    #[cfg(feature = "field-inline")]
    if args.name.uses_field_inline() {
        program.enable_field_inline();
    }
    program.build(
        args.guest_target_dir
            .to_str()
            .expect("guest target path must be UTF-8"),
    );
    let elf = program.elf.expect("guest build produces an ELF");
    fs::write(args.out.join("guest.elf"), elf.bytes()).expect("save guest ELF");
    for &count in &args.operations {
        fs::write(
            args.out.join(format!("input-{count}.bin")),
            args.name.input_for_operations(count),
        )
        .expect("save fixed guest input");
    }
}

pub(super) fn secp256k1_inputs(count: u32) -> Vec<u8> {
    // Valid signature from examples/secp256k1-ecdsa-verify/src/main.rs.
    const Z: [u64; 4] = [
        0x9088_f7ac_e2ef_cde9,
        0xc484_efe3_7a53_80ee,
        0xa52e_52d7_da7d_abfa,
        0xb94d_27b9_934d_3e08,
    ];
    const R: [u64; 4] = [
        0xb8fc_413b_4b96_7ed8,
        0x248d_4b0b_2829_ab00,
        0x587f_6929_6af3_cd88,
        0x3a5d_6a38_6e6c_f7c0,
    ];
    const S: [u64; 4] = [
        0x66a8_2f27_4e3d_cafc,
        0x299a_0248_6be4_0321,
        0x6212_d714_118f_617e,
        0x9d45_2f63_cf91_018d,
    ];
    const Q: [u64; 8] = [
        0x0012_563f_32ed_0216,
        0xee00_716a_f6a7_3670,
        0x91fc_70e3_4e00_e6c8,
        0xeeb6_be8b_9e68_868b,
        0x4780_de3d_5fda_972d,
        0xcb1b_42d7_2491_e47f,
        0xdc7f_3126_2e4b_a2b7,
        0xdc7b_004d_3bb2_800d,
    ];
    postcard::to_stdvec(&(count, Z, R, S, Q)).expect("serialize ECDSA verification input")
}

#[cfg(test)]
mod tests {
    use super::Workload;

    #[test]
    fn fixed_inputs_use_the_guests_postcard_wire_format() {
        assert_eq!(Workload::Fibonacci.input_for_operations(300), [0xac, 2]);
        assert_eq!(Workload::BTreeMap.input_for_operations(300), [0xac, 2]);
        let mut hash_input = vec![5; 32];
        hash_input.extend([0xac, 2]);
        assert_eq!(Workload::Sha2Chain.input_for_operations(300), hash_input);
        assert_eq!(Workload::Sha3Chain.input_for_operations(300), hash_input);
    }
}
