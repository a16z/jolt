use std::path::{Path, PathBuf};
use std::process::Command;

use arbitrary::{Arbitrary, Unstructured};

use common::constants::{DEFAULT_MAX_TRUSTED_ADVICE_SIZE, DEFAULT_MAX_UNTRUSTED_ADVICE_SIZE};
use common::jolt_device::MemoryConfig;

use super::{Invariant, InvariantViolation};
use crate::TestCase;

/// Input for the soundness invariant.
///
/// The red-team agent produces a `patch` (unified diff) to apply to
/// the `guest-sandbox/` template project, along with the program input
/// and a dishonest claim about the output and panic flag.
///
/// The invariant proves the patched guest honestly, then checks that
/// the verifier rejects the dishonest claim.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct SoundnessInput {
    /// Unified diff to apply to `guest-sandbox/`.
    /// Only hunks touching files within the sandbox are applied.
    pub patch: String,
    /// Input bytes fed to the guest program.
    pub program_input: Vec<u8>,
    /// The output the malicious prover claims.
    pub claimed_output: Vec<u8>,
    /// The panic flag the malicious prover claims.
    pub claimed_panic: bool,
}

impl<'a> Arbitrary<'a> for SoundnessInput {
    fn arbitrary(_u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        // Soundness is RedTeam-only; Arbitrary is not meaningful.
        Err(arbitrary::Error::IncorrectFormat)
    }
}

/// Cached paths resolved once during setup.
pub struct SoundnessSetup {
    sandbox_dir: PathBuf,
}

#[jolt_eval_macros::invariant(RedTeam)]
#[derive(Default)]
pub struct SoundnessInvariant;

impl Invariant for SoundnessInvariant {
    type Setup = SoundnessSetup;
    type Input = SoundnessInput;

    fn name(&self) -> &str {
        "soundness"
    }

    fn description(&self) -> String {
        "For any deterministic guest program (no advice) and fixed input, \
         there is only one (output, panic) pair that the verifier accepts. \
         A counterexample is a guest patch + input + dishonest (output, panic) \
         claim that the verifier incorrectly accepts."
            .to_string()
    }

    fn setup(&self) -> SoundnessSetup {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        SoundnessSetup {
            sandbox_dir: manifest_dir.join("guest-sandbox"),
        }
    }

    fn check(
        &self,
        setup: &SoundnessSetup,
        input: SoundnessInput,
    ) -> Result<(), InvariantViolation> {
        // 1. Apply patch to sandbox in-place, revert on exit
        let _guard = apply_patch(&setup.sandbox_dir, &input.patch)?;

        // 2. Compile the patched guest
        let elf_bytes = compile_guest(&setup.sandbox_dir)?;

        // _guard drops here (or on early return), reverting the patch

        // 3. Build a TestCase and prove
        let memory_config = MemoryConfig {
            max_input_size: 4096,
            max_output_size: 4096,
            max_untrusted_advice_size: DEFAULT_MAX_UNTRUSTED_ADVICE_SIZE,
            max_trusted_advice_size: DEFAULT_MAX_TRUSTED_ADVICE_SIZE,
            stack_size: 65536,
            heap_size: 32768,
            program_size: None,
        };
        let test_case = TestCase {
            elf_contents: elf_bytes,
            memory_config,
            max_trace_length: 1048576,
        };
        let prover_pp = test_case.prover_preprocessing();
        let verifier_pp = TestCase::verifier_preprocessing(&prover_pp);
        let (proof, honest_device) = test_case.prove(&prover_pp, &input.program_input);

        // 4. Skip no-op claims (the claim matches the honest execution)
        if input.claimed_output == honest_device.outputs
            && input.claimed_panic == honest_device.panic
        {
            return Ok(());
        }

        // 5. Verify with the dishonest claim — this SHOULD fail
        match TestCase::verify_with_claims(
            &verifier_pp,
            proof,
            &honest_device.inputs,
            &input.claimed_output,
            input.claimed_panic,
        ) {
            Ok(()) => Err(InvariantViolation::with_details(
                "Verifier accepted dishonest claim",
                format!(
                    "honest_output={} bytes (panic={}), claimed_output={} bytes (panic={})",
                    honest_device.outputs.len(),
                    honest_device.panic,
                    input.claimed_output.len(),
                    input.claimed_panic,
                ),
            )),
            Err(_) => Ok(()),
        }
    }

    fn seed_corpus(&self) -> Vec<SoundnessInput> {
        vec![SoundnessInput {
            patch: String::new(),
            program_input: vec![1, 2, 3],
            claimed_output: vec![0xFF],
            claimed_panic: false,
        }]
    }
}

/// RAII guard that reverts a patch on drop via `git checkout`.
struct PatchGuard {
    dir: PathBuf,
    applied: bool,
}

impl Drop for PatchGuard {
    fn drop(&mut self) {
        if self.applied {
            let _ = Command::new("git")
                .current_dir(&self.dir)
                .args(["checkout", "."])
                .status();
        }
    }
}

/// Apply a filtered patch to `sandbox_dir` in-place. Returns a guard
/// that reverts the changes on drop (even on panic).
fn apply_patch(sandbox_dir: &Path, patch: &str) -> Result<PatchGuard, InvariantViolation> {
    let guard = PatchGuard {
        dir: sandbox_dir.to_path_buf(),
        applied: false,
    };

    if patch.trim().is_empty() {
        return Ok(guard);
    }

    let safe_patch = filter_patch(patch);
    if safe_patch.trim().is_empty() {
        return Ok(guard);
    }

    let mut child = Command::new("git")
        .current_dir(sandbox_dir)
        .args(["apply", "--allow-empty", "-"])
        .stdin(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| InvariantViolation::new(format!("git apply spawn: {e}")))?;

    if let Some(stdin) = child.stdin.as_mut() {
        use std::io::Write;
        let _ = stdin.write_all(safe_patch.as_bytes());
    }

    let output = child
        .wait_with_output()
        .map_err(|e| InvariantViolation::new(format!("git apply wait: {e}")))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(InvariantViolation::with_details(
            "Patch failed to apply",
            stderr.to_string(),
        ));
    }

    Ok(PatchGuard {
        dir: sandbox_dir.to_path_buf(),
        applied: true,
    })
}

/// Remove diff hunks that reference paths containing `..` to prevent
/// escaping the sandbox.
pub fn filter_patch(patch: &str) -> String {
    let mut result = String::new();
    let mut include_hunk = true;

    for line in patch.lines() {
        if line.starts_with("diff --git") || line.starts_with("--- ") || line.starts_with("+++ ")
        {
            include_hunk = !line.contains("..");
        }
        if include_hunk {
            result.push_str(line);
            result.push('\n');
        }
    }

    result
}

/// Compile the guest and return the ELF bytes.
fn compile_guest(sandbox_dir: &Path) -> Result<Vec<u8>, InvariantViolation> {
    let jolt_cmd = std::env::var("JOLT_PATH").unwrap_or_else(|_| "jolt".to_string());
    let target_dir = sandbox_dir.join("target");

    let output = Command::new(&jolt_cmd)
        .args([
            "build",
            "-p",
            "sandbox-guest",
            "--",
            "--release",
            "--target-dir",
        ])
        .arg(target_dir.as_os_str())
        .arg("--features")
        .arg("guest")
        .current_dir(sandbox_dir)
        .output()
        .map_err(|e| {
            InvariantViolation::new(format!(
                "jolt build: {e}. Make sure `jolt` is installed (cargo install --path .)"
            ))
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(InvariantViolation::with_details(
            "Guest compilation failed",
            stderr.to_string(),
        ));
    }

    let elf_path = target_dir
        .join("riscv64imac-unknown-none-elf")
        .join("release")
        .join("sandbox-guest");

    std::fs::read(&elf_path).map_err(|e| {
        InvariantViolation::with_details(
            "ELF not found after compilation",
            format!("{}: {e}", elf_path.display()),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::filter_patch;

    #[test]
    fn keeps_safe_hunks() {
        let patch = "\
diff --git a/src/lib.rs b/src/lib.rs
--- a/src/lib.rs
+++ b/src/lib.rs
@@ -1,3 +1,3 @@
-fn foo() {}
+fn bar() {}
";
        let filtered = filter_patch(patch);
        assert!(filtered.contains("+fn bar() {}"));
    }

    #[test]
    fn drops_hunks_with_path_traversal() {
        let patch = "\
diff --git a/../../jolt-core/src/lib.rs b/../../jolt-core/src/lib.rs
--- a/../../jolt-core/src/lib.rs
+++ b/../../jolt-core/src/lib.rs
@@ -1 +1 @@
-safe
+malicious
";
        let filtered = filter_patch(patch);
        assert!(!filtered.contains("malicious"));
    }

    #[test]
    fn mixed_safe_and_unsafe_hunks() {
        let patch = "\
diff --git a/src/lib.rs b/src/lib.rs
--- a/src/lib.rs
+++ b/src/lib.rs
@@ -1 +1 @@
-old
+new
diff --git a/../../../etc/passwd b/../../../etc/passwd
--- a/../../../etc/passwd
+++ b/../../../etc/passwd
@@ -1 +1 @@
-root
+hacked
diff --git a/Cargo.toml b/Cargo.toml
--- a/Cargo.toml
+++ b/Cargo.toml
@@ -1 +1 @@
-v1
+v2
";
        let filtered = filter_patch(patch);
        assert!(filtered.contains("+new"));
        assert!(!filtered.contains("hacked"));
        assert!(filtered.contains("+v2"));
    }

    #[test]
    fn empty_patch_stays_empty() {
        assert!(filter_patch("").is_empty());
        assert!(filter_patch("   \n  ").trim().is_empty());
    }
}
