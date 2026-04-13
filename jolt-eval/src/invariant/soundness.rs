use std::path::{Path, PathBuf};
use std::process::Command;

use arbitrary::{Arbitrary, Unstructured};

use common::constants::{DEFAULT_MAX_TRUSTED_ADVICE_SIZE, DEFAULT_MAX_UNTRUSTED_ADVICE_SIZE};
use common::jolt_device::MemoryConfig;
use jolt_core::host::Program;

use super::{CheckError, Invariant, InvariantViolation};
use crate::guests;

/// Guest memory layout parameters.
///
/// Serializable mirror of `common::jolt_device::MemoryConfig` for use
/// in JSON-based counterexamples.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct GuestMemoryConfig {
    pub max_input_size: u64,
    pub max_output_size: u64,
    pub stack_size: u64,
    pub heap_size: u64,
}

impl Default for GuestMemoryConfig {
    fn default() -> Self {
        Self {
            max_input_size: 4096,
            max_output_size: 4096,
            stack_size: 65536,
            heap_size: 32768,
        }
    }
}

/// Maximum allowed values for memory config parameters.
const MAX_INPUT_SIZE: u64 = 1 << 16;
const MAX_OUTPUT_SIZE: u64 = 1 << 16;
const MAX_STACK_SIZE: u64 = 1 << 16;
const MAX_HEAP_SIZE: u64 = 1 << 20;
const MAX_TRACE_LENGTH: usize = 1 << 20;

impl GuestMemoryConfig {
    pub fn validate(&self) -> Result<(), CheckError> {
        if self.max_input_size > MAX_INPUT_SIZE
            || self.max_output_size > MAX_OUTPUT_SIZE
            || self.stack_size > MAX_STACK_SIZE
            || self.heap_size > MAX_HEAP_SIZE
        {
            return Err(CheckError::InvalidInput(format!(
                "memory config exceeds limits: \
                 input={}, output={}, stack={}, heap={}; \
                 limits: input<={MAX_INPUT_SIZE}, output<={MAX_OUTPUT_SIZE}, stack<={MAX_STACK_SIZE}, heap<={MAX_HEAP_SIZE}",
                self.max_input_size, self.max_output_size, self.stack_size, self.heap_size,
            )));
        }
        Ok(())
    }

    fn to_memory_config(&self) -> MemoryConfig {
        MemoryConfig {
            max_input_size: self.max_input_size,
            max_output_size: self.max_output_size,
            max_untrusted_advice_size: DEFAULT_MAX_UNTRUSTED_ADVICE_SIZE,
            max_trusted_advice_size: DEFAULT_MAX_TRUSTED_ADVICE_SIZE,
            stack_size: self.stack_size,
            heap_size: self.heap_size,
            program_size: None,
        }
    }
}

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
    /// Guest memory layout. Defaults are reasonable for most programs.
    #[serde(default)]
    pub memory: GuestMemoryConfig,
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
        format!(
            "For any deterministic guest program (no advice) and fixed input, \
             there is only one (output, panic) pair that the verifier accepts. \
             A counterexample is a guest patch + input + dishonest (output, panic) \
             claim that the verifier incorrectly accepts. \
             For full context, read the invariant file: jolt-eval/src/invariant/soundness.rs \n\n\
             ## Guest sandbox\n\n\
             The guest template is at `jolt-eval/guest-sandbox/`. It contains:\n\
             - `Cargo.toml` — depends on `jolt-sdk`\n\
             - `src/lib.rs` — the `#[jolt::provable]` function (main patch target)\n\
             - `src/main.rs` — no_main entry point (rarely needs patching)\n\n\
             ## Producing a patch\n\n\
             Simply edit the files inside `jolt-eval/guest-sandbox/` directly. \
             The harness automatically captures your changes as a `git diff` \
             from the worktree before cleanup and uses it as the patch. \
             You do NOT need to put the patch in the JSON counterexample — \
             leave the `patch` field empty and the harness fills it in.\n\n\
             Alternatively, you can provide a patch explicitly in the JSON \
             `patch` field. If non-empty, it takes precedence over the \
             worktree diff. Hunks referencing paths with `..` are filtered out.\n\n\
             ## Limits\n\n\
             Memory config: max_input_size <= {MAX_INPUT_SIZE}, \
             max_output_size <= {MAX_OUTPUT_SIZE}, \
             stack_size <= {MAX_STACK_SIZE}, heap_size <= {MAX_HEAP_SIZE}. \
             The program's execution trace must not exceed {MAX_TRACE_LENGTH} cycles."
        )
    }

    fn setup(&self) -> SoundnessSetup {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        SoundnessSetup {
            sandbox_dir: manifest_dir.join("guest-sandbox"),
        }
    }

    fn check(&self, setup: &SoundnessSetup, input: SoundnessInput) -> Result<(), CheckError> {
        // 1. Validate memory config
        input.memory.validate()?;
        let mut memory_config = input.memory.to_memory_config();

        // 2. Apply patch to sandbox in-place, revert on exit
        let _guard = apply_patch(&setup.sandbox_dir, &input.patch)?;

        // 3. Compile the patched guest
        let elf_bytes = compile_guest(&setup.sandbox_dir, &memory_config)?;

        // _guard drops here (or on early return), reverting the patch

        // 4. Decode to get program_size, then trace to get actual length
        let (_bytecode, _memory_init, program_size, _e_entry) =
            jolt_core::guest::program::decode(&elf_bytes);
        memory_config.program_size = Some(program_size);

        let program = guests::GuestProgram::new(&elf_bytes, &memory_config);
        let (_lazy_trace, trace, _memory, _io) = program.trace(&input.program_input, &[], &[]);
        let max_trace_length = (trace.len() + 1).next_power_of_two();
        drop(trace);

        if max_trace_length > MAX_TRACE_LENGTH {
            return Err(CheckError::InvalidInput(format!(
                "trace length {max_trace_length} exceeds limit {MAX_TRACE_LENGTH}"
            )));
        }

        // 5. Prove and verify
        let prover_pp = guests::prover_preprocessing(&program, max_trace_length);
        let verifier_pp = guests::verifier_preprocessing(&prover_pp);
        let (proof, honest_device) = guests::prove(&program, &prover_pp, &input.program_input);

        // 6. Skip no-op claims (the claim matches the honest execution)
        if input.claimed_output == honest_device.outputs
            && input.claimed_panic == honest_device.panic
        {
            return Err(CheckError::InvalidInput(
                "claimed output/panic matches honest execution".into(),
            ));
        }

        // 7. Verify with the dishonest claim — this SHOULD fail
        match guests::verify_with_claims(
            &verifier_pp,
            proof,
            &honest_device.inputs,
            &input.claimed_output,
            input.claimed_panic,
        ) {
            Ok(()) => Err(CheckError::Violation(InvariantViolation::with_details(
                "Verifier accepted dishonest claim",
                format!(
                    "honest_output={} bytes (panic={}), claimed_output={} bytes (panic={})",
                    honest_device.outputs.len(),
                    honest_device.panic,
                    input.claimed_output.len(),
                    input.claimed_panic,
                ),
            ))),
            Err(_) => Ok(()),
        }
    }

    fn seed_corpus(&self) -> Vec<SoundnessInput> {
        vec![SoundnessInput {
            patch: String::new(),
            memory: GuestMemoryConfig::default(),
            program_input: postcard::to_stdvec::<[u8]>(&[1, 2, 3]).unwrap(),
            claimed_output: vec![0xFF],
            claimed_panic: false,
        }]
    }

    /// If the agent modified `guest-sandbox/` in its worktree, use that
    /// diff as the patch (unless the agent already provided one in JSON).
    fn enrich_input(&self, mut input: SoundnessInput, diff: Option<&str>) -> SoundnessInput {
        if input.patch.trim().is_empty() {
            if let Some(diff) = diff {
                input.patch = diff.to_string();
            }
        }
        input
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
            let _ = Command::new("git")
                .current_dir(&self.dir)
                .args(["clean", "-fd"])
                .status();
        }
    }
}

/// Apply a filtered patch to `sandbox_dir` in-place. Returns a guard
/// that reverts the changes on drop (even on panic).
fn apply_patch(sandbox_dir: &Path, patch: &str) -> Result<PatchGuard, CheckError> {
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
        .map_err(|e| CheckError::InvalidInput(format!("git apply spawn: {e}")))?;

    if let Some(stdin) = child.stdin.as_mut() {
        use std::io::Write;
        let _ = stdin.write_all(safe_patch.as_bytes());
    }

    let output = child
        .wait_with_output()
        .map_err(|e| CheckError::InvalidInput(format!("git apply wait: {e}")))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(CheckError::InvalidInput(format!(
            "patch failed to apply: {stderr}"
        )));
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
        if line.starts_with("diff --git") || line.starts_with("--- ") || line.starts_with("+++ ") {
            include_hunk = !line.contains("..");
        }
        if include_hunk {
            result.push_str(line);
            result.push('\n');
        }
    }

    result
}

/// Compile the sandbox guest and return the ELF bytes.
///
/// `Program::build` panics on compilation failure, so we catch it.
fn compile_guest(sandbox_dir: &Path, memory_config: &MemoryConfig) -> Result<Vec<u8>, CheckError> {
    let target_dir = sandbox_dir.join("target").to_string_lossy().to_string();
    let mc = *memory_config;
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut program = Program::new("sandbox-guest");
        program.set_memory_config(mc);
        program.build(&target_dir);
        program.get_elf_contents()
    }));
    match result {
        Ok(Some(elf)) => Ok(elf),
        Ok(None) => Err(CheckError::InvalidInput(
            "guest ELF not found after build".into(),
        )),
        Err(_) => Err(CheckError::InvalidInput(
            "guest compilation panicked".into(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Invariant;

    fn default_input() -> SoundnessInput {
        SoundnessInput {
            patch: String::new(),
            memory: GuestMemoryConfig::default(),
            program_input: postcard::to_stdvec::<[u8]>(&[1, 2, 3]).unwrap(),
            claimed_output: vec![0xFF],
            claimed_panic: false,
        }
    }

    // ── filter_patch ────────────────────────────────────────────────

    #[test]
    fn filter_keeps_safe_hunks() {
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
    fn filter_drops_path_traversal() {
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
    fn filter_mixed_safe_and_unsafe() {
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
    fn filter_empty_patch() {
        assert!(filter_patch("").is_empty());
        assert!(filter_patch("   \n  ").trim().is_empty());
    }

    // ── memory config validation ────────────────────────────────────

    #[test]
    fn validate_accepts_defaults() {
        assert!(GuestMemoryConfig::default().validate().is_ok());
    }

    #[test]
    fn validate_rejects_oversized_input() {
        let c = GuestMemoryConfig {
            max_input_size: u64::MAX,
            ..Default::default()
        };
        assert!(matches!(c.validate(), Err(CheckError::InvalidInput(_))));
    }

    #[test]
    fn validate_rejects_oversized_output() {
        let c = GuestMemoryConfig {
            max_output_size: u64::MAX,
            ..Default::default()
        };
        assert!(matches!(c.validate(), Err(CheckError::InvalidInput(_))));
    }

    #[test]
    fn validate_rejects_oversized_stack() {
        let c = GuestMemoryConfig {
            stack_size: u64::MAX,
            ..Default::default()
        };
        assert!(matches!(c.validate(), Err(CheckError::InvalidInput(_))));
    }

    #[test]
    fn validate_rejects_oversized_heap() {
        let c = GuestMemoryConfig {
            heap_size: u64::MAX,
            ..Default::default()
        };
        assert!(matches!(c.validate(), Err(CheckError::InvalidInput(_))));
    }

    #[test]
    fn check_rejects_oversized_memory_before_compilation() {
        let inv = SoundnessInvariant;
        let setup = inv.setup();
        let input = SoundnessInput {
            memory: GuestMemoryConfig {
                heap_size: u64::MAX,
                ..Default::default()
            },
            ..default_input()
        };
        assert!(matches!(
            inv.check(&setup, input),
            Err(CheckError::InvalidInput(_))
        ));
    }

    // ── patching ────────────────────────────────────────────────────

    #[test]
    fn check_garbage_patch_is_noop() {
        let inv = SoundnessInvariant;
        let setup = inv.setup();
        let input = SoundnessInput {
            patch: "this is not a valid unified diff\n+garbage".into(),
            ..default_input()
        };
        // Garbage with no diff headers passes filter_patch unchanged.
        // git apply --allow-empty treats it as a no-op (no hunks),
        // so the unpatched sandbox compiles and the check proceeds normally.
        assert!(inv.check(&setup, input).is_ok());
    }

    // ── compilation + prove/verify (slow) ───────────────────────────

    #[test]
    fn check_path_traversal_filtered_then_compiles() {
        let inv = SoundnessInvariant;
        let setup = inv.setup();
        let input = SoundnessInput {
            patch: "\
diff --git a/../../etc/passwd b/../../etc/passwd
--- a/../../etc/passwd
+++ b/../../etc/passwd
@@ -1 +1 @@
-root
+hacked
"
            .into(),
            ..default_input()
        };
        // Traversal hunks are filtered out → empty patch → compiles
        // unpatched sandbox → proves → verifier rejects dishonest claim.
        assert!(inv.check(&setup, input).is_ok());
    }

    #[test]
    fn check_unpatched_sandbox_rejects_dishonest_output() {
        let inv = SoundnessInvariant;
        let setup = inv.setup();
        // claimed_output=[0xFF] doesn't match the identity function's
        // honest output for input [1,2,3]. Verifier should reject.
        assert!(inv.check(&setup, default_input()).is_ok());
    }

    #[test]
    fn check_noop_claim_returns_invalid_input() {
        let inv = SoundnessInvariant;
        let setup = inv.setup();
        // The sandbox computes h = wrapping hash of input bytes.
        // For input [1,2,3]: h = ((0*31+1)*31+2)*31+3 = 1026
        let honest_output = postcard::to_stdvec(&1026u32).unwrap();
        let input = SoundnessInput {
            claimed_output: honest_output,
            claimed_panic: false,
            ..default_input()
        };
        assert!(matches!(
            inv.check(&setup, input),
            Err(CheckError::InvalidInput(_))
        ));
    }
}
