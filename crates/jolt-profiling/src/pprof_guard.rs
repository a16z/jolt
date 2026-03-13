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
    /// The active profiler guard.
    pub guard: pprof::ProfilerGuard<'static>,
    /// Label used for the output filename.
    pub label: &'static str,
}

/// Stub type when `pprof` feature is not enabled.
#[cfg(not(feature = "pprof"))]
pub struct PprofGuard;

#[cfg(feature = "pprof")]
impl Drop for PprofGuard {
    fn drop(&mut self) {
        if let Ok(report) = self.guard.report().build() {
            let prefix = std::env::var("PPROF_PREFIX")
                .unwrap_or_else(|_| String::from("benchmark-runs/pprof/"));
            let filename = format!("{}{}.pb", prefix, self.label);
            if let Some(dir) = std::path::Path::new(&filename).parent() {
                let _ = std::fs::create_dir_all(dir);
            }
            if let Ok(mut f) = std::fs::File::create(&filename) {
                use pprof::protos::Message;
                if let Ok(p) = report.pprof() {
                    let mut buf = Vec::new();
                    if p.encode(&mut buf).is_ok() {
                        let _ = std::io::Write::write_all(&mut f, &buf);
                        tracing::info!("Wrote pprof profile to {}", filename);
                    }
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
/// Configure via environment variables:
/// - `PPROF_PREFIX` — output directory prefix (default: `"benchmark-runs/pprof/"`)
/// - `PPROF_FREQ` — sampling frequency in Hz (default: 100)
#[macro_export]
macro_rules! pprof_scope {
    ($label:expr) => {{
        #[cfg(feature = "pprof")]
        {
            Some($crate::PprofGuard {
                guard: pprof::ProfilerGuardBuilder::default()
                    .frequency(
                        std::env::var("PPROF_FREQ")
                            .unwrap_or_else(|_| "100".to_string())
                            .parse::<i32>()
                            .unwrap_or(100),
                    )
                    .blocklist(&["libc", "libgcc", "pthread", "vdso"])
                    .build()
                    .expect("Failed to initialize profiler"),
                label: $label,
            })
        }
        #[cfg(not(feature = "pprof"))]
        None::<$crate::PprofGuard>
    }};
    () => {
        $crate::pprof_scope!("default")
    };
}
