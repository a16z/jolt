//! Execution backends must agree.
//!
//! `specs/x86-tracer-backend.md` invariants 1-2: a non-reference backend's
//! output must be indistinguishable from the reference interpreter's for the
//! same program and inputs.
//!
//! Scope today is the **fast pass**, which is what the AOT backend
//! implements: total row count, `JoltDevice` (outputs and panic flag), final
//! memory, and the captured advice tape. Full `TraceRow`-stream equality
//! joins when the backend grows record mode (spec slice 3's record half);
//! the row *count* already catches control-flow divergence, and final memory
//! catches value divergence that reaches RAM.
//!
//! Two comparisons run, so the invariant is meaningful on every platform:
//! - the reference backend's eager trace vs its own chunked fast pass
//!   (backend-generic; the fast pass must not observe a different execution
//!   from the recording one), and
//! - on x86-64 Linux, additionally the AOT backend's fast pass against the
//!   same reference. Elsewhere `NativeBackend` *is* the interpreter, so that
//!   second comparison would be vacuous and is skipped rather than faked.

use common::constants::RAM_START_ADDRESS;
use common::jolt_device::MemoryConfig;
use jolt_program::execution::{
    ChunkedExecutionBackend, ExecutionBackend, JoltProgram, MemoryImage, TraceInputs,
};
use jolt_prover_legacy::host::Program;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tracer::TracerBackend;

use crate::invariant::{CheckError, Invariant, InvariantViolation};

/// Guests in the equivalence corpus. Each is a distinct execution shape:
/// a tight arithmetic loop, an inline-heavy hash chain, allocator and
/// pointer-chasing traffic, and the division/remainder advice groups.
const CORPUS: &[Guest] = &[
    Guest {
        package: "fibonacci-guest",
        func: "fib",
        kind: GuestInput::U32 { min: 1, max: 2000 },
    },
    Guest {
        package: "sha2-chain-guest",
        func: "sha2_chain",
        kind: GuestInput::Chain { min: 1, max: 16 },
    },
    Guest {
        package: "sha3-chain-guest",
        func: "sha3_chain",
        kind: GuestInput::Chain { min: 1, max: 8 },
    },
    Guest {
        package: "btreemap-guest",
        func: "btreemap",
        kind: GuestInput::U32 { min: 1, max: 64 },
    },
    Guest {
        package: "muldiv-guest",
        func: "muldiv",
        kind: GuestInput::MulDiv,
    },
];

struct Guest {
    package: &'static str,
    func: &'static str,
    kind: GuestInput,
}

/// How a fuzzed parameter becomes this guest's serialized input. Inputs stay
/// well-formed by construction: a guest that fails to deserialize its input
/// panics identically under both backends and would test nothing.
enum GuestInput {
    U32 {
        min: u32,
        max: u32,
    },
    Chain {
        min: u32,
        max: u32,
    },
    /// `muldiv(a, b, c)` — three u32s; `c` is forced nonzero so the DIV
    /// group actually executes instead of the guest panicking on a
    /// divide-by-zero (both backends agree on the panic, but it ends the
    /// run before the interesting rows).
    MulDiv,
}

impl Guest {
    fn encode(&self, parameter: u32) -> Vec<u8> {
        match self.kind {
            GuestInput::U32 { min, max } => {
                let n = min + parameter % (max - min + 1);
                postcard::to_stdvec(&n).unwrap_or_default()
            }
            GuestInput::Chain { min, max } => {
                let iterations = min + parameter % (max - min + 1);
                let mut bytes = postcard::to_stdvec(&[5u8; 32]).unwrap_or_default();
                bytes.extend(postcard::to_stdvec(&iterations).unwrap_or_default());
                bytes
            }
            GuestInput::MulDiv => {
                let a = parameter | 1;
                let b = parameter.rotate_left(7) | 1;
                let c = parameter.rotate_left(17) | 1;
                postcard::to_stdvec(&(a, b, c)).unwrap_or_default()
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, arbitrary::Arbitrary)]
pub struct TracerBackendEquivalenceInput {
    /// Selects a corpus guest (taken modulo the corpus size).
    pub guest: u8,
    /// Guest parameter (loop count, chain length, operand seed).
    pub parameter: u32,
}

/// What a fast pass observes, for equality comparison.
#[derive(PartialEq, Eq)]
struct Observation {
    rows: usize,
    outputs: Vec<u8>,
    panic: bool,
    final_memory: Option<MemoryImage>,
    advice_tape: Option<Vec<u8>>,
}

impl Observation {
    fn diff(&self, other: &Self) -> Option<String> {
        if self.rows != other.rows {
            return Some(format!("row count: {} vs {}", self.rows, other.rows));
        }
        if self.outputs != other.outputs {
            return Some(format!(
                "device outputs: {} bytes vs {} bytes",
                self.outputs.len(),
                other.outputs.len()
            ));
        }
        if self.panic != other.panic {
            return Some(format!("panic flag: {} vs {}", self.panic, other.panic));
        }
        if self.final_memory != other.final_memory {
            return Some("final memory differs".to_string());
        }
        if self.advice_tape != other.advice_tape {
            return Some("advice tape differs".to_string());
        }
        None
    }
}

pub struct CorpusProgram {
    program: JoltProgram,
    memory_config: MemoryConfig,
}

#[jolt_eval_macros::invariant(Test, Fuzz, RedTeam)]
#[derive(Default)]
pub struct TracerBackendEquivalenceInvariant;

impl Invariant for TracerBackendEquivalenceInvariant {
    type Setup = Vec<CorpusProgram>;
    type Input = TracerBackendEquivalenceInput;

    fn name(&self) -> &str {
        "tracer_backend_equivalence"
    }

    fn description(&self) -> String {
        "Execution backends must be indistinguishable: for the same program \
         and inputs, a fast pass must report the same row count, JoltDevice \
         outputs and panic flag, final memory, and advice tape as the \
         reference interpreter's eager trace. Checked for the reference \
         backend's own chunked fast pass everywhere, and additionally for \
         the AOT x86-64 backend on x86_64 Linux."
            .to_string()
    }

    fn setup(&self) -> Vec<CorpusProgram> {
        CORPUS
            .iter()
            .map(|guest| {
                let mut host = Program::new(guest.package);
                host.set_func(guest.func);
                let program = host
                    .jolt_program()
                    .expect("failed to build corpus guest program");
                let memory_config = MemoryConfig {
                    program_size: Some(program.program_end - RAM_START_ADDRESS),
                    ..Default::default()
                };
                CorpusProgram {
                    program,
                    memory_config,
                }
            })
            .collect()
    }

    fn check(
        &self,
        setup: &Vec<CorpusProgram>,
        input: TracerBackendEquivalenceInput,
    ) -> Result<(), CheckError> {
        if setup.is_empty() {
            return Err(CheckError::InvalidInput("empty corpus".into()));
        }
        let index = usize::from(input.guest) % setup.len();
        let guest = &CORPUS[index];
        let entry = &setup[index];
        let inputs = TraceInputs::new(
            guest.encode(input.parameter),
            Vec::new(),
            Vec::new(),
            entry.memory_config,
        );

        // The reference eager trace is the oracle.
        let mut reference = TracerBackend::new();
        let eager = reference
            .trace(&entry.program, inputs.clone())
            .map_err(|e| CheckError::InvalidInput(format!("reference trace failed: {e:?}")))?;
        let oracle = Observation {
            rows: eager.trace.rows().len(),
            outputs: eager.device.outputs.clone(),
            panic: eager.device.panic,
            final_memory: eager.final_memory.clone(),
            advice_tape: eager.advice_tape.clone(),
        };

        // The reference backend's own fast pass must observe the same
        // execution as its recording pass.
        let summary = reference
            .execute(&entry.program, inputs.clone(), 1 << 18)
            .map_err(|e| CheckError::InvalidInput(format!("reference fast pass failed: {e:?}")))?;
        let reference_fast = Observation {
            rows: summary.trace_len,
            outputs: summary.device.outputs.clone(),
            panic: summary.device.panic,
            final_memory: summary.final_memory.clone(),
            advice_tape: summary.advice_tape.clone(),
        };
        if let Some(diff) = oracle.diff(&reference_fast) {
            return Err(CheckError::Violation(InvariantViolation::with_details(
                "reference fast pass diverged from the eager trace",
                format!(
                    "guest={}, parameter={}: {diff}",
                    guest.package, input.parameter
                ),
            )));
        }

        // Where the AOT backend exists, it faces the same oracle.
        #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
        {
            let mut native = jolt_tracer_x86::X86TracerBackend::new();
            let fast = native
                .fast_run(&entry.program, inputs)
                .map_err(|e| CheckError::InvalidInput(format!("x86 fast run failed: {e:?}")))?;
            let native_fast = Observation {
                rows: fast.trace_len,
                outputs: fast.device.outputs.clone(),
                panic: fast.device.panic,
                final_memory: Some(fast.final_memory.clone()),
                advice_tape: Some(fast.advice_tape.clone()),
            };
            if let Some(diff) = oracle.diff(&native_fast) {
                return Err(CheckError::Violation(InvariantViolation::with_details(
                    "x86 backend diverged from the reference",
                    format!(
                        "guest={}, parameter={}: {diff}",
                        guest.package, input.parameter
                    ),
                )));
            }
        }

        Ok(())
    }

    fn seed_corpus(&self) -> Vec<TracerBackendEquivalenceInput> {
        (0..CORPUS.len() as u8)
            .flat_map(|guest| {
                [1u32, 7, 64]
                    .into_iter()
                    .map(move |parameter| TracerBackendEquivalenceInput { guest, parameter })
            })
            .collect()
    }
}
