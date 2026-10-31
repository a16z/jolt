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
        0x9088f7ace2efcde9,
        0xc484efe37a5380ee,
        0xa52e52d7da7dabfa,
        0xb94d27b9934d3e08,
    ];
    const R: [u64; 4] = [
        0xb8fc413b4b967ed8,
        0x248d4b0b2829ab00,
        0x587f69296af3cd88,
        0x3a5d6a386e6cf7c0,
    ];
    const S: [u64; 4] = [
        0x66a82f274e3dcafc,
        0x299a02486be40321,
        0x6212d714118f617e,
        0x9d452f63cf91018d,
    ];
    const Q: [u64; 8] = [
        0x0012563f32ed0216,
        0xee00716af6a73670,
        0x91fc70e34e00e6c8,
        0xeeb6be8b9e68868b,
        0x4780de3d5fda972d,
        0xcb1b42d72491e47f,
        0xdc7f31262e4ba2b7,
        0xdc7b004d3bb2800d,
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
