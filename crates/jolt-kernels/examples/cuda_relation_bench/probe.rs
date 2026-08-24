use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use jolt_kernels::cuda::CudaKernelContext;

const MEM_POLL: Duration = Duration::from_millis(20);

const SMI_WARMUP: Duration = Duration::from_millis(300);

#[derive(Clone, Copy, Default)]
pub struct DeviceProbe {
    pub baseline: usize,
    pub peak: usize,
    pub util: Option<f64>,
    pub mem_util: Option<f64>,
    pub watts: Option<f64>,
}

impl DeviceProbe {
    pub const fn own(&self) -> usize {
        self.peak.saturating_sub(self.baseline)
    }
}

#[derive(Clone, Default)]
pub struct GpuProbe {
    pub devices: Vec<DeviceProbe>,
    pub iteration: Duration,
    pub polled_iteration: Duration,
    pub smi_samples: usize,
}

fn per_iteration(elapsed: Duration, iterations: usize) -> Duration {
    elapsed
        .checked_div(u32::try_from(iterations).unwrap_or(u32::MAX))
        .unwrap_or_default()
}

struct SmiTotals {
    util: Vec<f64>,
    mem_util: Vec<f64>,
    watts: Vec<f64>,
    count: usize,
}

impl SmiTotals {
    fn new(devices: usize) -> Self {
        Self {
            util: vec![0.0; devices],
            mem_util: vec![0.0; devices],
            watts: vec![0.0; devices],
            count: 0,
        }
    }
}

fn smi_query(devices: usize) -> Option<Vec<(f64, f64, f64)>> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=utilization.gpu,utilization.memory,power.draw",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8(output.stdout).ok()?;
    let mut rows = Vec::with_capacity(devices);
    for line in text.lines().take(devices) {
        let mut fields = line.split(',').map(str::trim);
        let util = fields.next()?.parse().ok()?;
        let mem_util = fields.next()?.parse().ok()?;
        let watts = fields.next()?.parse().ok()?;
        rows.push((util, mem_util, watts));
    }
    (rows.len() == devices).then_some(rows)
}

fn sample_memory(context: &'static CudaKernelContext, stop: &AtomicBool) -> usize {
    let mut peak = 0;
    while !stop.load(Ordering::Relaxed) {
        if let Ok((used, _)) = context.memory_used() {
            peak = peak.max(used);
        }
        thread::sleep(MEM_POLL);
    }
    peak
}

fn sample_smi(devices: usize, poll: Duration, stop: &AtomicBool) -> SmiTotals {
    let mut totals = SmiTotals::new(devices);
    thread::sleep(SMI_WARMUP);
    while !stop.load(Ordering::Relaxed) {
        let started = Instant::now();
        if let Some(rows) = smi_query(devices) {
            for (index, (util, mem_util, watts)) in rows.into_iter().enumerate() {
                if let (Some(u), Some(m), Some(w)) = (
                    totals.util.get_mut(index),
                    totals.mem_util.get_mut(index),
                    totals.watts.get_mut(index),
                ) {
                    *u += util;
                    *m += mem_util;
                    *w += watts;
                }
            }
            totals.count += 1;
        }
        thread::sleep(poll.saturating_sub(started.elapsed()));
    }
    totals
}

fn hold_for<R>(hold: Duration, run: &mut impl FnMut() -> R) -> Duration {
    let started = Instant::now();
    let mut iterations = 0;
    while started.elapsed() < hold {
        let _ = run();
        iterations += 1;
    }
    per_iteration(started.elapsed(), iterations)
}

fn peaks_over<R>(
    devices: usize,
    hold: Duration,
    run: &mut impl FnMut() -> R,
) -> (Vec<usize>, Duration) {
    let stop = Arc::new(AtomicBool::new(false));
    let samplers: Vec<_> = (0..devices)
        .filter_map(jolt_kernels::cuda::context_for)
        .map(|context| {
            let stop = Arc::clone(&stop);
            thread::spawn(move || sample_memory(context, &stop))
        })
        .collect();
    let iteration = hold_for(hold, run);
    stop.store(true, Ordering::Relaxed);
    let peaks = samplers
        .into_iter()
        .map(|handle| handle.join().unwrap_or(0))
        .collect();
    (peaks, iteration)
}

fn smi_over<R>(
    devices: usize,
    hold: Duration,
    poll: Duration,
    run: &mut impl FnMut() -> R,
) -> (SmiTotals, Duration) {
    if poll.is_zero() {
        return (SmiTotals::new(devices), hold_for(hold, run));
    }
    let stop = Arc::new(AtomicBool::new(false));
    let sampler = {
        let stop = Arc::clone(&stop);
        thread::spawn(move || sample_smi(devices, poll, &stop))
    };
    let iteration = hold_for(hold, run);
    stop.store(true, Ordering::Relaxed);
    (
        sampler.join().unwrap_or_else(|_| SmiTotals::new(devices)),
        iteration,
    )
}

pub fn probe<R>(
    devices: usize,
    hold: Duration,
    smi_poll: Duration,
    mut run: impl FnMut() -> R,
) -> GpuProbe {
    let (totals, iteration) = smi_over(devices, hold, smi_poll, &mut run);
    let baseline = jolt_kernels::cuda::device_memory_used();
    let (peaks, polled_iteration) = peaks_over(devices, hold, &mut run);
    let mean = |total: Option<&f64>| {
        (totals.count > 0).then(|| total.copied().unwrap_or(0.0) / totals.count as f64)
    };
    GpuProbe {
        devices: (0..devices)
            .map(|index| DeviceProbe {
                baseline: baseline.get(index).copied().unwrap_or(0),
                peak: peaks.get(index).copied().unwrap_or(0),
                util: mean(totals.util.get(index)),
                mem_util: mean(totals.mem_util.get(index)),
                watts: mean(totals.watts.get(index)),
            })
            .collect(),
        iteration,
        polled_iteration,
        smi_samples: totals.count,
    }
}
