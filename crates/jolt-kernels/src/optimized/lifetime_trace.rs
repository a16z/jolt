//! Drop-site tracing for the trace-record family (diagnostic only).
//!
//! `JOLT_LIFETIME_TRACE=1` logs birth/death of every record-family allocation
//! (the TraceRecord lanes, register lanes, RAM access columns, shared
//! instruction rows, PC rows, opening increments) with elapsed time and a
//! jolt-frames-only backtrace at the death site. The death fires at the LAST
//! `Arc` drop — exactly the point where the backing memory returns to the
//! allocator — so the log discriminates "who held the final reference and in
//! which stage" from kernel-side reclaim timing (which the RSS/footprint
//! ledger tracks instead). Off by default; a disabled run costs one relaxed
//! atomic load per tagged object drop.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

fn units_gib(bytes: usize) -> f64 {
    bytes as f64 / (1u64 << 30) as f64
}

static ENABLED: OnceLock<bool> = OnceLock::new();
static EPOCH: OnceLock<Instant> = OnceLock::new();
/// Set while the process is inside `TraceRecord::release` — lets the log
/// distinguish the session's own drop from a last-consumer drop elsewhere.
static IN_RELEASE: AtomicBool = AtomicBool::new(false);

pub(crate) fn enabled() -> bool {
    *ENABLED.get_or_init(|| {
        std::env::var("JOLT_LIFETIME_TRACE").is_ok_and(|v| !v.is_empty() && v != "0")
    })
}

fn elapsed_s() -> f64 {
    EPOCH.get_or_init(Instant::now).elapsed().as_secs_f64()
}

pub(crate) fn mark_release_scope(active: bool) {
    if enabled() {
        IN_RELEASE.store(active, Ordering::Relaxed);
    }
}

/// Log a free-form lifetime event (session parks/takes, ref counts).
#[expect(clippy::print_stderr, reason = "env-gated audit trace")]
pub(crate) fn note(message: &std::fmt::Arguments<'_>) {
    if enabled() {
        eprintln!("[lifetime] t=+{:.3}s {message}", elapsed_s());
    }
}

macro_rules! lifetime_note {
    ($($arg:tt)*) => {
        crate::optimized::lifetime_trace::note(&format_args!($($arg)*))
    };
}
pub(crate) use lifetime_note;

/// Embedded birth/death marker. Construct with the owner's payload size; the
/// `Drop` impl fires when the owning struct (i.e. its last `Arc`) drops.
pub(crate) struct LifetimeTag {
    name: &'static str,
    bytes: usize,
}

impl LifetimeTag {
    #[expect(clippy::print_stderr, reason = "env-gated audit trace")]
    pub(crate) fn new(name: &'static str, bytes: usize) -> Self {
        if enabled() {
            eprintln!(
                "[lifetime] t=+{:.3}s born {name} ({:.2} GiB)",
                elapsed_s(),
                units_gib(bytes),
            );
        }
        Self { name, bytes }
    }
}

impl Drop for LifetimeTag {
    #[expect(clippy::print_stderr, reason = "env-gated audit trace")]
    fn drop(&mut self) {
        if !enabled() {
            return;
        }
        let in_release = IN_RELEASE.load(Ordering::Relaxed);
        // Also emit a tracing event so chrome-format runs place the death on
        // the span timeline.
        tracing::info!(
            target: "lifetime",
            object = self.name,
            gib = units_gib(self.bytes),
            in_release,
            "record family drop"
        );
        let backtrace = std::backtrace::Backtrace::force_capture().to_string();
        let holder_frames: Vec<&str> = backtrace
            .lines()
            .filter(|line| {
                let line = line.trim_start();
                line.contains("jolt") || line.contains("drop_in_place")
            })
            .take(16)
            .collect();
        eprintln!(
            "[lifetime] t=+{:.3}s drop {} ({:.2} GiB){}\n{}",
            elapsed_s(),
            self.name,
            units_gib(self.bytes),
            if in_release {
                " [in TraceRecord::release]"
            } else {
                ""
            },
            holder_frames.join("\n"),
        );
    }
}
