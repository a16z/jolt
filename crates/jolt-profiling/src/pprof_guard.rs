//! Scoped CPU profiler guard for `pprof` integration.
//!
//! Use the [`pprof_scope!`] macro to create a guard that starts a CPU profiler
//! on creation and writes a `.pb` flamegraph file on drop.
//!
//! Requires the `pprof` feature. Without it, the macro expands to `None::<PprofGuard>`.
//!
//! ```no_run
//! use jolt_profiling::pprof_scope;
//!
//! let _guard = pprof_scope!("my_function");
//! // ... profiled code ...
//! // guard drops here, writing benchmark-runs/pprof/my_function.pb
//! ```
//!
//! View with: `go tool pprof -http=:8080 benchmark-runs/pprof/my_function.pb`

/// Guard that holds a running pprof profiler and writes output on drop.
#[cfg(feature = "pprof")]
pub struct PprofGuard {
    guard: pprof::ProfilerGuard<'static>,
    label: &'static str,
}

#[cfg(feature = "pprof")]
impl PprofGuard {
    /// Creates a new profiler guard with the given label and sampling frequency.
    ///
    /// The label determines the output filename: `{PPROF_PREFIX}{label}.pb`.
    /// Typically called via the [`pprof_scope!`] macro rather than directly.
    pub fn new(label: &'static str, frequency: i32) -> Option<Self> {
        match pprof::ProfilerGuardBuilder::default()
            .frequency(frequency)
            .blocklist(&["libc", "libgcc", "pthread", "vdso"])
            .build()
        {
            Ok(guard) => Some(Self { guard, label }),
            Err(e) => {
                tracing::warn!(label = label, error = %e, "failed to initialize profiler");
                None
            }
        }
    }
}

/// Stub type when `pprof` feature is not enabled.
#[cfg(not(feature = "pprof"))]
pub struct PprofGuard;

#[cfg(feature = "pprof")]
impl Drop for PprofGuard {
    fn drop(&mut self) {
        use std::io::Write;

        let Ok(report) = self.guard.report().build() else {
            tracing::warn!(label = self.label, "failed to build pprof report");
            return;
        };

        let prefix = crate::setup::PPROF_PREFIX
            .get()
            .map(String::as_str)
            .unwrap_or("benchmark-runs/pprof/");
        let filename = format!("{prefix}{}.pb", self.label);

        if let Some(dir) = std::path::Path::new(&filename).parent() {
            let _ = std::fs::create_dir_all(dir);
        }

        let Ok(mut f) = std::fs::File::create(&filename) else {
            tracing::warn!(path = %filename, "failed to create pprof output file");
            return;
        };

        if let Ok(p) = report.pprof() {
            use pprof::protos::Message;
            let mut buf = Vec::new();
            if p.encode(&mut buf).is_ok() {
                if f.write_all(&buf).is_ok() {
                    tracing::info!(path = %filename, "wrote pprof profile");
                } else {
                    tracing::warn!(path = %filename, "failed to write pprof data");
                }
            }
        }
    }
}

/// Creates a scoped CPU profiler guard.
///
/// With the `pprof` feature enabled, returns `Some(PprofGuard)` that writes a
/// `.pb` file on drop. Without the feature, returns `None::<PprofGuard>`.
///
/// When called without arguments, uses `"default"` as the label.
///
/// Configure via environment variables:
/// - `PPROF_PREFIX` — output directory prefix (default: `"benchmark-runs/pprof/"`)
/// - `PPROF_FREQ` — sampling frequency in Hz (default: 100)
#[macro_export]
macro_rules! pprof_scope {
    ($label:expr) => {{
        #[cfg(feature = "pprof")]
        {
            $crate::PprofGuard::new(
                $label,
                std::env::var("PPROF_FREQ")
                    .unwrap_or_else(|_| "100".to_string())
                    .parse::<i32>()
                    .unwrap_or(100),
            )
        }
        #[cfg(not(feature = "pprof"))]
        None::<$crate::PprofGuard>
    }};
    () => {
        $crate::pprof_scope!("default")
    };
}

#[cfg(test)]
mod tests {
    #[test]
    fn pprof_scope_without_feature_returns_none() {
        let guard = pprof_scope!("test_label");
        #[cfg(not(feature = "pprof"))]
        assert!(guard.is_none());
        #[cfg(feature = "pprof")]
        assert!(guard.is_some());
    }

    #[test]
    fn pprof_scope_no_arg_variant() {
        let guard = pprof_scope!();
        #[cfg(not(feature = "pprof"))]
        assert!(guard.is_none());
        #[cfg(feature = "pprof")]
        assert!(guard.is_some());
    }

    #[test]
    fn pprof_guard_stub_exists() {
        #[cfg(not(feature = "pprof"))]
        {
            let _guard = super::PprofGuard;
        }
    }
}
