//! Tracing subscriber configuration for Perfetto and console output.
//!
//! Call [`setup_tracing`] once at binary startup. The returned [`TracingGuards`]
//! must be held alive for the duration of the program — dropping them flushes
//! and closes trace files.

use std::any::Any;
use std::sync::OnceLock;

use tracing_chrome::ChromeLayerBuilder;
use tracing_subscriber::{fmt::format::FmtSpan, prelude::*, EnvFilter};

/// Thread-safe storage for the pprof output prefix.
///
/// Initialized once during [`setup_tracing`] and read by [`PprofGuard`](crate::PprofGuard)
/// on drop. Avoids `std::env::set_var` which is unsound in multi-threaded contexts.
pub(crate) static PPROF_PREFIX: OnceLock<String> = OnceLock::new();

/// Output format for tracing subscribers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TracingFormat {
    /// Console output with span close events and compact formatting.
    Default,
    /// Chrome/Perfetto JSON trace file. View at <https://ui.perfetto.dev/>.
    Chrome,
}

/// Opaque container for tracing flush guards.
///
/// Must be held alive for the duration of profiling. Dropping this flushes
/// all pending trace data and stops background monitors.
#[must_use = "guards must be held alive for the duration of profiling"]
pub struct TracingGuards(#[expect(dead_code)] Vec<Box<dyn Any>>);

/// Initializes the global tracing subscriber with the requested output formats.
///
/// Always installs a minimal log layer that respects `RUST_LOG`. Additional
/// layers are added based on the `formats` slice.
///
/// Returns a [`TracingGuards`] value that **must be kept alive** until the
/// program exits. Dropping the guards flushes pending trace data.
///
/// # Chrome format
///
/// Writes to `benchmark-runs/perfetto_traces/{trace_name}.json`.
/// Open in [Perfetto UI](https://ui.perfetto.dev/) for timeline visualization.
///
/// # Panics
///
/// Panics if called more than once (the global subscriber can only be set once).
pub fn setup_tracing(formats: &[TracingFormat], trace_name: &str) -> TracingGuards {
    // Legacy pprof default; the run-dir layout below never sees it because
    // the OnceLock is initialized here first.
    let _ = PPROF_PREFIX.get_or_init(|| {
        std::env::var("PPROF_PREFIX")
            .unwrap_or_else(|_| format!("benchmark-runs/pprof/{trace_name}_"))
    });
    setup_tracing_with_trace_path(
        formats,
        std::path::Path::new(&format!("benchmark-runs/perfetto_traces/{trace_name}.json")),
    )
}

/// [`setup_tracing`] with an explicit chrome-trace output path — the modular
/// profile harness groups every artifact into a per-run directory and puts
/// the trace there. pprof output (if that lane is on) defaults into the same
/// directory as `pprof_{label}.pb`.
pub fn setup_tracing_with_trace_path(
    formats: &[TracingFormat],
    trace_path: &std::path::Path,
) -> TracingGuards {
    let _ = PPROF_PREFIX.get_or_init(|| {
        std::env::var("PPROF_PREFIX")
            .unwrap_or_else(|_| trace_path.with_file_name("pprof_").display().to_string())
    });

    let mut layers = Vec::new();

    let log_layer = tracing_subscriber::fmt::layer()
        .compact()
        .with_target(false)
        .with_file(false)
        .with_line_number(false)
        .with_thread_ids(false)
        .with_thread_names(false)
        .with_filter(EnvFilter::from_default_env())
        .boxed();
    layers.push(log_layer);

    let mut guards: Vec<Box<dyn Any>> = vec![];

    if formats.contains(&TracingFormat::Default) {
        let collector_layer = tracing_subscriber::fmt::layer()
            .with_span_events(FmtSpan::CLOSE)
            .compact()
            .with_target(false)
            .with_file(false)
            .with_line_number(false)
            .with_thread_ids(false)
            .with_thread_names(false)
            .boxed();
        layers.push(collector_layer);
    }
    if formats.contains(&TracingFormat::Chrome) {
        if let Some(parent) = trace_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let (chrome_layer, guard) = ChromeLayerBuilder::new()
            .include_args(true)
            .file(trace_path)
            .build();
        layers.push(chrome_layer.boxed());
        guards.push(Box::new(guard));
        tracing::info!("Chrome tracing enabled. Output: {}", trace_path.display());
    }

    // Boundary RSS sampling for the prover-stage spans; inert for all others.
    #[cfg(not(target_arch = "wasm32"))]
    layers.push(crate::stage_memory::StageMemoryLayer.boxed());

    tracing_subscriber::registry().with(layers).init();

    #[cfg(all(not(target_arch = "wasm32"), feature = "monitor"))]
    guards.push(Box::new({
        tracing::info!(
            "Starting MetricsMonitor — counter events are rewritten into Perfetto \
             counter tracks at flush by summary::finalize_trace"
        );
        crate::monitor::MetricsMonitor::start(
            std::env::var("MONITOR_INTERVAL")
                .unwrap_or_else(|_| "0.1".to_string())
                .parse::<f64>()
                .unwrap_or(0.1),
        )
    }));

    TracingGuards(guards)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tracing_format_is_copy() {
        let fmt = TracingFormat::Chrome;
        let fmt2 = fmt;
        assert_eq!(fmt, fmt2);
    }

    #[test]
    fn tracing_format_debug() {
        let fmt = TracingFormat::Default;
        let s = format!("{fmt:?}");
        assert_eq!(s, "Default");
    }

    #[test]
    fn tracing_format_eq() {
        assert_eq!(TracingFormat::Chrome, TracingFormat::Chrome);
        assert_ne!(TracingFormat::Chrome, TracingFormat::Default);
    }
}
