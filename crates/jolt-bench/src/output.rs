use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StackLabel {
    Core,
    Modular,
}

impl StackLabel {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Core => "core",
            Self::Modular => "modular",
        }
    }
}

/// One row of the JSON output: median metrics for a single stack on a single program.
///
/// Encoded as a flat object. When the stack is unavailable for the program,
/// `unsupported: true` is set and the numeric metrics are omitted.
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct Run {
    pub stack: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub prove_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verify_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peak_rss_mb: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub proof_bytes: Option<u64>,

    /// Encoding used to compute `proof_bytes`. `ark-compressed` for jolt-core,
    /// `bincode-serde` for the modular stack.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub proof_encoding: Option<String>,

    #[serde(skip_serializing_if = "std::ops::Not::not", default)]
    pub unsupported: bool,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,

    /// Optional note about the verify path (e.g. `modular-prove + core-verify`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verify_note: Option<String>,
}

/// Top-level JSON report: one program, N runs (one per stack).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BenchReport {
    pub program: String,
    pub iters: usize,
    pub warmup: usize,
    pub runs: Vec<Run>,
}
