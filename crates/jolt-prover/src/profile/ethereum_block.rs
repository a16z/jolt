//! The `ethereum-block` workload: the guest of jeth (`github.com/a16z/jeth`),
//! which statelessly validates one Ethereum mainnet block.
//!
//! jeth is not part of this workspace. Cargo pins it through the
//! `jeth-inlines-*` dependencies, which register the host side of jeth's own
//! inline instructions with the tracer, and `prepare` builds the guest from
//! Cargo's checkout of that revision against this checkout's Jolt crates.
//! `scripts/bench/ethereum-block/README.md` describes the inputs and the bump.

use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use jolt_host::Program;
use serde_json::Value;

use super::{PrepareArgs, Workload};

/// The sizes in the `#[jolt::provable]` attribute of `validate_block`, in jeth's
/// `crates/guest/src/lib.rs` at the pinned revision. The guest computes its
/// I/O addresses from the attribute at compile time. From these constants,
/// `jolt build` receives the heap and stack sizes and the tracer's memory
/// layout takes all four; no check compares the two sets. A different input
/// or output size moves the I/O region, and a smaller heap or stack can run
/// out, so `prepare` executing every block exposes a difference that matters.
const MAX_INPUT_SIZE: u64 = 16_777_216;
const MAX_OUTPUT_SIZE: u64 = 4096;
const HEAP_SIZE: u64 = 1_610_612_736;
const STACK_SIZE: u64 = 33_554_432;

/// The Git source from which jeth's manifests take their Jolt packages.
const JOLT_GIT: &str = "https://github.com/a16z/jolt";

/// The packages jeth's manifests take from [`JOLT_GIT`]. `prepare` fails after
/// the build if the guest's lockfile still has a package from that source.
const JOLT_PACKAGES: [&str; 10] = [
    "common",
    "jolt-inlines-bigint",
    "jolt-inlines-blake2",
    "jolt-inlines-keccak256",
    "jolt-inlines-p256",
    "jolt-inlines-sdk",
    "jolt-inlines-secp256k1",
    "jolt-inlines-sha2",
    "jolt-sdk",
    "tracer",
];

/// Checked-in inputs, relative to the workspace root: `<block>/input.bin` and
/// `<block>/advice.bin`.
const INPUTS: &str = "scripts/bench/ethereum-block";

/// The blocks with checked-in inputs and their mainnet hashes.
const BLOCKS: [(u64, &str); 2] = [
    (
        26_227_673,
        "4986ed4390cb3902fac0ee40e60b2409efbae03c4e306015a73729de65cc382e",
    ),
    (
        26_433_846,
        "09993aae466d59802abd7b4c7dff92ff1517c328f59a8ef8ff4b5ea6f1c2ef6d",
    ),
];

pub(super) fn program() -> Program {
    let mut program = Program::new("jeth-guest");
    program.set_func("validate_block");
    program.set_max_input_size(MAX_INPUT_SIZE);
    program.set_max_output_size(MAX_OUTPUT_SIZE);
    program.set_heap_size(HEAP_SIZE);
    program.set_stack_size(STACK_SIZE);
    program
}

/// Cargo's resolution of the profile harness: this workspace's packages and
/// their dependencies, among them the `jeth-inlines-*` crates.
struct Packages(Value);

impl Packages {
    fn resolve(workspace: &Path) -> Self {
        let output = Command::new("cargo")
            .args(["metadata", "--format-version", "1", "--locked"])
            .args(["--features", "profiling", "--manifest-path"])
            .arg(workspace.join("crates/jolt-prover/Cargo.toml"))
            .current_dir(workspace)
            .output()
            .expect("run cargo metadata");
        assert!(
            output.status.success(),
            "cargo metadata failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        Self(serde_json::from_slice(&output.stdout).expect("parse cargo metadata"))
    }

    fn dir(&self, name: &str) -> &Path {
        self.0["packages"]
            .as_array()
            .into_iter()
            .flatten()
            .find(|package| package["name"] == name)
            .and_then(|package| package["manifest_path"].as_str())
            .and_then(|manifest| Path::new(manifest).parent())
            .unwrap_or_else(|| panic!("cargo metadata lists no package {name}"))
    }

    fn stage_jeth(&self, guest_target_dir: &Path) -> PathBuf {
        let checkout = self
            .dir("jeth-inlines-blake2f")
            .ancestors()
            .nth(3)
            .expect("jeth's repository root");
        let staged = guest_target_dir.join("jeth");
        if staged.exists() {
            fs::remove_dir_all(&staged).expect("remove the previous jeth copy");
        }
        // Cargo's checkout stays as fetched. The patch table below re-sources
        // the Jolt packages in `crates/guest/Cargo.lock`, so the build rewrites
        // that lockfile, and Cargo's only stable way to keep it elsewhere is
        // to build a copy (`--locked` refuses the rewrite; `--lockfile-path`
        // is unstable). The copy holds what the guest build reads and leaves
        // out jeth's `rust-toolchain.toml`: under this checkout's target
        // directory the guest then builds with this checkout's toolchain,
        // like the other benchmark guests.
        let mut pending = ["Cargo.toml", "crates", "library"]
            .map(PathBuf::from)
            .to_vec();
        while let Some(relative) = pending.pop() {
            let from = checkout.join(&relative);
            if from.is_dir() {
                fs::create_dir_all(staged.join(&relative)).expect("create the jeth copy");
                for entry in fs::read_dir(&from).expect("read jeth's checkout") {
                    pending.push(relative.join(entry.expect("read jeth's checkout").file_name()));
                }
            } else {
                let _ = fs::copy(&from, staged.join(&relative)).expect("copy jeth's sources");
            }
        }
        let mut config = format!("[patch.\"{JOLT_GIT}\"]\n");
        for name in JOLT_PACKAGES {
            let path = self.dir(name).to_str().expect("package path must be UTF-8");
            writeln!(config, "{name} = {{ path = {} }}", Value::from(path))
                .expect("write to a string");
        }
        fs::create_dir_all(staged.join(".cargo")).expect("create the jeth copy");
        fs::write(staged.join(".cargo/config.toml"), config).expect("write the Jolt patch table");
        staged
    }
}

impl PrepareArgs {
    /// Write guest.elf and, per requested block, input-N.bin and advice-N.bin.
    ///
    /// Each block's checked-in input and advice tape are first executed in the
    /// built guest. A panic or another block's hash stops the command before
    /// it writes anything, so a file that no longer fits the pinned jeth fails
    /// here and not in a measurement.
    pub(super) fn ethereum_block(&self) {
        assert!(
            matches!(self.name, Workload::EthereumBlock),
            "--blocks selects inputs of ethereum-block; {} takes --operations",
            self.name.as_str()
        );
        let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .nth(2)
            .and_then(|root| root.canonicalize().ok())
            .expect("the workspace root");
        fs::create_dir_all(&self.guest_target_dir).expect("create the guest target directory");
        let guest_target_dir = self
            .guest_target_dir
            .canonicalize()
            .expect("resolve the guest target directory");
        assert!(
            guest_target_dir.starts_with(&workspace),
            "--guest-target-dir must be inside {}: the guest builds with that checkout's \
             toolchain, and `prepare` replaces the directory's `jeth` entry",
            workspace.display()
        );
        let source = Packages::resolve(&workspace).stage_jeth(&guest_target_dir);
        let mut program = program();
        program.set_workspace_dir(source.join("crates/guest"));
        program.build(
            guest_target_dir
                .to_str()
                .expect("guest target path must be UTF-8"),
        );
        let elf = program.elf.clone().expect("guest build produces an ELF");
        let lock = fs::read_to_string(source.join("crates/guest/Cargo.lock"))
            .expect("read the guest's lockfile");
        let from_git = format!("source = \"git+{JOLT_GIT}");
        assert!(
            !lock.lines().any(|line| line
                .strip_prefix(&from_git)
                .is_some_and(|rest| rest.starts_with(['?', '#']))),
            "jeth's guest takes a Jolt package from Git that JOLT_PACKAGES does not name"
        );

        let mut validated = Vec::new();
        for &block in &self.blocks {
            let (_, hash) = BLOCKS
                .into_iter()
                .find(|&(number, _)| number == block)
                .unwrap_or_else(|| panic!("{INPUTS} has no input for block {block}"));
            let files = workspace.join(INPUTS).join(block.to_string());
            let input = fs::read(files.join("input.bin")).expect("read the checked-in input");
            let input = postcard::to_stdvec(input.as_slice()).expect("serialize input");
            let advice = fs::read(files.join("advice.bin")).expect("read the checked-in advice");

            // A failed advice check in the guest aborts this process inside
            // the tracer, which names neither the block nor the file.
            eprintln!(
                "executing block {block} from {}; a failure below means its input.bin or \
                 advice.bin no longer fits the pinned jeth ({INPUTS}/README.md)",
                files.display()
            );
            let (rows, device) =
                program.execute_with_output(&input, &[], &[], Some(advice.clone()));
            assert!(!device.panic, "jeth's guest panicked on block {block}");
            let mut output = String::new();
            for byte in device.outputs.iter().take(32) {
                write!(output, "{byte:02x}").expect("write to a string");
            }
            assert_eq!(
                output, hash,
                "the checked-in input of block {block} holds another block"
            );
            eprintln!("block {block} traces {rows} rows");
            validated.push((block, input, advice));
        }

        fs::write(self.out.join("guest.elf"), elf.bytes()).expect("save guest ELF");
        for (block, input, advice) in validated {
            fs::write(self.out.join(format!("input-{block}.bin")), input)
                .expect("save fixed guest input");
            fs::write(self.out.join(format!("advice-{block}.bin")), advice)
                .expect("save fixed advice tape");
        }
    }
}
