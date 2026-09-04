//! Process-wide GPU hang watchdog for the Metal backend.
//!
//! macOS puts no time limit on compute command buffers. A kernel that never
//! terminates wedges the GPU: `wait_until_completed` blocks forever, WindowServer
//! stops receiving frames, and watchdogd panics the machine after 120 s. Four
//! kernel panics on 2026-09-04 came from one zero-stride loop in a candidate
//! Booleanity kernel running under `cargo nextest`.
//!
//! The watchdog thread dispatches a one-thread probe kernel on its own queue
//! every [`PROBE_INTERVAL`]. An empty command buffer would not do: the firmware
//! completes those without occupying a core, so they finish even while every
//! core spins. A healthy GPU runs the probe within milliseconds even under a
//! saturating proof because the firmware time-slices queues. If a probe has not
//! completed within the deadline the process aborts, which tears down every GPU
//! context it owns and lets the driver reset the GPU well before the system
//! watchdog acts.
//!
//! Starts with the first [`super::SolinasMetal`] in the process, so proofs,
//! benches and tests are all covered. `JOLT_METAL_HANG_WATCHDOG=0` disables it
//! and `JOLT_METAL_HANG_WATCHDOG_SECS` overrides the deadline (default 30).

use std::{
    sync::OnceLock,
    thread::{self, sleep},
    time::{Duration, Instant},
};

use metal::{
    objc::rc::autoreleasepool, CompileOptions, ComputePipelineState, Device,
    MTLCommandBufferStatus, MTLResourceOptions, MTLSize,
};

const DEFAULT_DEADLINE: Duration = Duration::from_secs(30);
const PROBE_INTERVAL: Duration = Duration::from_secs(5);
const POLL_INTERVAL: Duration = Duration::from_millis(250);
const PROBE_SOURCE: &str = "\
#include <metal_stdlib>
kernel void jolt_hang_probe(device uint* out [[buffer(0)]]) { out[0] += 1u; }
";

static STARTED: OnceLock<()> = OnceLock::new();

/// Starts the watchdog thread once per process; later calls do nothing.
pub(super) fn start(device: &Device) {
    if STARTED.set(()).is_err() {
        return;
    }
    let Some(deadline) = deadline_from_env() else {
        tracing::info!("metal hang watchdog disabled by JOLT_METAL_HANG_WATCHDOG=0");
        return;
    };
    let pipeline = match probe_pipeline(device) {
        Ok(pipeline) => pipeline,
        Err(err) => {
            tracing::warn!(err, "metal hang watchdog probe kernel did not compile");
            return;
        }
    };
    let device = device.clone();
    let spawned = thread::Builder::new()
        .name("jolt-metal-hang-watchdog".into())
        .spawn(move || probe_forever(&device, &pipeline, deadline));
    if let Err(err) = spawned {
        tracing::warn!(%err, "metal hang watchdog thread did not start");
    }
}

fn probe_pipeline(device: &Device) -> Result<ComputePipelineState, String> {
    let library = device.new_library_with_source(PROBE_SOURCE, &CompileOptions::new())?;
    let function = library.get_function("jolt_hang_probe", None)?;
    device.new_compute_pipeline_state_with_function(&function)
}

fn deadline_from_env() -> Option<Duration> {
    if std::env::var("JOLT_METAL_HANG_WATCHDOG").is_ok_and(|value| value == "0") {
        return None;
    }
    let deadline = std::env::var("JOLT_METAL_HANG_WATCHDOG_SECS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|&secs| secs > 0)
        .map_or(DEFAULT_DEADLINE, Duration::from_secs);
    Some(deadline)
}

fn probe_forever(device: &Device, pipeline: &ComputePipelineState, deadline: Duration) {
    let queue = device.new_command_queue();
    let counter = device.new_buffer(4, MTLResourceOptions::StorageModeShared);
    loop {
        sleep(PROBE_INTERVAL);
        // one pool per probe: new_command_buffer returns an autoreleased object
        // and this thread never returns to a pool of its own
        let latency = autoreleasepool(|| {
            let probe = queue.new_command_buffer();
            let encoder = probe.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_buffer(0, Some(&counter), 0);
            encoder.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
            encoder.end_encoding();
            probe.commit();
            let start = Instant::now();
            loop {
                match probe.status() {
                    MTLCommandBufferStatus::Completed | MTLCommandBufferStatus::Error => {
                        return Some(start.elapsed());
                    }
                    MTLCommandBufferStatus::NotEnqueued
                    | MTLCommandBufferStatus::Enqueued
                    | MTLCommandBufferStatus::Committed
                    | MTLCommandBufferStatus::Scheduled => {}
                }
                if start.elapsed() >= deadline {
                    return None;
                }
                sleep(POLL_INTERVAL);
            }
        });
        match latency {
            Some(latency) if latency > deadline / 4 => {
                tracing::warn!(?latency, ?deadline, "metal probe command buffer was slow");
            }
            Some(latency) => tracing::debug!(?latency, "metal probe command buffer completed"),
            None => abort_for_hung_gpu(deadline),
        }
    }
}

#[expect(
    clippy::print_stderr,
    reason = "last words before abort; nextest processes run without a tracing subscriber"
)]
fn abort_for_hung_gpu(deadline: Duration) -> ! {
    let message = format!(
        "jolt metal hang watchdog: the one-thread probe kernel did not complete within \
         {deadline:?}, so the GPU is hung; aborting to release the GPU contexts before \
         watchdogd panics the machine"
    );
    tracing::error!("{message}");
    eprintln!("{message}");
    std::process::abort()
}
