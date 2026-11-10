use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

static H2D_BYTES: AtomicU64 = AtomicU64::new(0);
static H2D_CALLS: AtomicU64 = AtomicU64::new(0);
static D2H_BYTES: AtomicU64 = AtomicU64::new(0);
static D2H_CALLS: AtomicU64 = AtomicU64::new(0);
static D2D_BYTES: AtomicU64 = AtomicU64::new(0);
static D2D_CALLS: AtomicU64 = AtomicU64::new(0);
static H2D_NANOS: AtomicU64 = AtomicU64::new(0);
static D2H_NANOS: AtomicU64 = AtomicU64::new(0);
static D2D_NANOS: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    H2d,
    D2h,
    D2d,
}

pub fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("JOLT_CUDA_XFER_STATS").is_ok_and(|value| value != "0" && !value.is_empty())
    })
}

fn counters(phase: Phase) -> (&'static AtomicU64, &'static AtomicU64, &'static AtomicU64) {
    match phase {
        Phase::H2d => (&H2D_BYTES, &H2D_CALLS, &H2D_NANOS),
        Phase::D2h => (&D2H_BYTES, &D2H_CALLS, &D2H_NANOS),
        Phase::D2d => (&D2D_BYTES, &D2D_CALLS, &D2D_NANOS),
    }
}

pub fn timed<T>(phase: Phase, bytes: usize, body: impl FnOnce() -> T) -> T {
    if !enabled() {
        return body();
    }
    let (byte_counter, call_counter, nano_counter) = counters(phase);
    let _ = byte_counter.fetch_add(bytes as u64, Ordering::Relaxed);
    let _ = call_counter.fetch_add(1, Ordering::Relaxed);
    let start = Instant::now();
    let value = body();
    let _ = nano_counter.fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
    value
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PhaseStats {
    pub bytes: u64,
    pub calls: u64,
    pub nanos: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Snapshot {
    pub h2d: PhaseStats,
    pub d2h: PhaseStats,
    pub d2d: PhaseStats,
}

impl Snapshot {
    pub fn report(&self) -> String {
        let line = |label: &str, stats: PhaseStats| {
            format!(
                "  {label}: {:.1} MB over {} calls in {:.1} ms",
                stats.bytes as f64 / (1024.0 * 1024.0),
                stats.calls,
                stats.nanos as f64 / 1.0e6,
            )
        };
        format!(
            "CUDA transfer stats:\n{}\n{}\n{}",
            line("H2D", self.h2d),
            line("D2H", self.d2h),
            line("D2D", self.d2d),
        )
    }
}

fn read(phase: Phase) -> PhaseStats {
    let (bytes, calls, nanos) = counters(phase);
    PhaseStats {
        bytes: bytes.load(Ordering::Relaxed),
        calls: calls.load(Ordering::Relaxed),
        nanos: nanos.load(Ordering::Relaxed),
    }
}

pub fn snapshot() -> Snapshot {
    Snapshot {
        h2d: read(Phase::H2d),
        d2h: read(Phase::D2h),
        d2d: read(Phase::D2d),
    }
}

pub fn reset() {
    for phase in [Phase::H2d, Phase::D2h, Phase::D2d] {
        let (bytes, calls, nanos) = counters(phase);
        bytes.store(0, Ordering::Relaxed);
        calls.store(0, Ordering::Relaxed);
        nanos.store(0, Ordering::Relaxed);
    }
}
