//! The Metal link against [`reference`]: the device `W` and every message and opening equal the
//! host's at 2^16 and 2^20. The bench is ignored and documented in the module docs.
#![expect(clippy::unwrap_used, reason = "test oracle")]

use std::{
    fmt::Debug,
    str::FromStr,
    time::{Duration, Instant},
};

use jolt_transcript::{Blake2bTranscript, Transcript};
use jolt_verifier::stages::byte_link::ByteLinkCompression;
use libc::{rusage_info_v4, RUSAGE_INFO_V4};
use metal::{Buffer, MTLResourceOptions};
use rayon::prelude::*;

use super::{
    gpu::{field, view, Phase},
    sort::W_CELLS,
    ByteLinkProver, ByteLinkSource, F, TRIPLE_BITS,
};
use crate::byte_link::fixtures::{scaled_active, SyntheticTrace};
use crate::byte_link::reference::{self, Histograms};
use crate::byte_link::{ByteLinkDraw, ByteLinkMessage, ByteLinkTranscript};
use crate::metal::solinas::{Fp128, SolinasMetal};

type T = Blake2bTranscript<F>;

/// A transcript stand-in: logs every message and derives every challenge from the log so far.
struct Recorder {
    transcript: T,
    log: Vec<(String, Vec<F>)>,
}

impl Recorder {
    /// A fresh recorder and the compression challenges it draws first.
    fn new() -> (Self, ByteLinkCompression<F>) {
        let mut transcript = T::new(b"byte-link-test");
        let gamma = std::array::from_fn(|_| std::array::from_fn(|_| transcript.challenge_scalar()));
        let compression = ByteLinkCompression {
            gamma,
            beta: transcript.challenge_scalar(),
        };
        (
            Self {
                transcript,
                log: Vec::new(),
            },
            compression,
        )
    }
}

fn flatten(message: ByteLinkMessage<'_, F>) -> (String, Vec<F>) {
    let pairs = |pairs: &[(F, F)]| pairs.iter().flat_map(|&(p, b)| [p, b]).collect::<Vec<_>>();
    match message {
        ByteLinkMessage::Roots(roots) => ("Roots".into(), pairs(roots)),
        ByteLinkMessage::LayerClaims {
            batch,
            layer,
            point,
            claims,
        } => (
            format!("LayerClaims({batch:?}, {layer})"),
            [point, &pairs(claims)].concat(),
        ),
        ByteLinkMessage::LayerRound {
            batch,
            layer,
            round,
            poly,
        } => (
            format!("LayerRound({batch:?}, {layer}, {round})"),
            poly.coefficients().to_vec(),
        ),
        ByteLinkMessage::Children {
            batch,
            layer,
            children,
        } => (format!("Children({batch:?}, {layer})"), children.concat()),
        ByteLinkMessage::QueryValues { group, values } => {
            (format!("QueryValues({group:?})"), values.to_vec())
        }
        ByteLinkMessage::QueryRound { group, round, poly } => (
            format!("QueryRound({group:?}, {round})"),
            poly.coefficients().to_vec(),
        ),
        ByteLinkMessage::QueryFinals { group, values } => {
            (format!("QueryFinals({group:?})"), values.to_vec())
        }
        ByteLinkMessage::SourceClaims {
            point,
            denominators,
        } => ("SourceClaims".into(), [point, denominators].concat()),
        ByteLinkMessage::SourceRound { round, poly } => (
            format!("SourceRound({round})"),
            poly.coefficients().to_vec(),
        ),
        ByteLinkMessage::SourceFinals(values) => ("SourceFinals".into(), values.to_vec()),
    }
}

impl ByteLinkTranscript<F> for Recorder {
    fn append(&mut self, message: ByteLinkMessage<'_, F>) {
        let (kind, values) = flatten(message);
        self.transcript.append_bytes(kind.as_bytes());
        self.transcript.append_values(b"values", &values);
        self.log.push((kind, values));
    }

    fn challenges(&mut self, draw: ByteLinkDraw, count: usize) -> Vec<F> {
        self.transcript.append_bytes(format!("{draw:?}").as_bytes());
        (0..count)
            .map(|_| self.transcript.challenge_scalar())
            .collect()
    }
}

fn device_source(metal: &SolinasMetal, bytes: &[i8]) -> Buffer {
    metal.device.new_buffer_with_data(
        bytes.as_ptr().cast(),
        bytes.len() as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

fn first_wrong_cell(device: &Buffer, host: &Histograms<F>) -> Option<(usize, usize)> {
    let cells = view::<Fp128>(device, 0, W_CELLS);
    host.tables.iter().enumerate().find_map(|(pack, table)| {
        let first = pack << TRIPLE_BITS;
        table
            .par_iter()
            .enumerate()
            .find_first(|&(h, &w)| field(cells[first + h]) != w)
            .map(|(h, _)| (pack, h))
    })
}

fn first_divergence(left: &[(String, Vec<F>)], right: &[(String, Vec<F>)]) -> Option<String> {
    left.iter()
        .zip(right)
        .position(|(left, right)| left != right)
        .map(|entry| format!("entry {entry}: {}", left[entry].0))
        .or_else(|| (left.len() != right.len()).then(|| "log lengths".to_owned()))
}

/// Proves `trace` on the device and on the host under the same challenges; panics where the
/// two disagree. Returns the device prover's log.
fn compare(
    prover: &mut ByteLinkProver,
    metal: &SolinasMetal,
    trace: &SyntheticTrace,
    active: usize,
) -> Vec<(String, Vec<F>)> {
    let inputs = trace.inputs::<F>(11);
    let bytes = device_source(metal, &trace.bytes);
    let source = ByteLinkSource {
        bytes: &bytes,
        log_rows: trace.plan.packing().logical_num_vars() as u32,
        active_rows: active,
    };
    let histograms = prover.histograms(&source, &inputs).unwrap();
    let host = reference::histograms(&trace.trace(), &inputs).unwrap();
    assert_eq!(first_wrong_cell(histograms.buffer(), &host), None, "W");
    let (mut device_log, compression) = Recorder::new();
    let device = prover
        .prove(&source, &histograms, &inputs, &compression, &mut device_log)
        .unwrap();
    let (mut host_log, _) = Recorder::new();
    let expected =
        reference::prove(&trace.trace(), &host, &inputs, &compression, &mut host_log).unwrap();
    assert_eq!(first_divergence(&device_log.log, &host_log.log), None);
    assert_eq!(device, expected);
    device_log.log
}

/// At 2^16 and 2^20 with the zero tail on uniform bytes with the edge cycles, and at 2^16 without
/// it on 256 hot tuples per pack, the Metal prover sends the reference prover's messages, and a
/// second proof repeats them.
#[test]
fn metal_link_sends_the_reference_messages() {
    let Ok(metal) = SolinasMetal::for_akita() else {
        return;
    };
    for (log_n, active, hot) in [
        (16, scaled_active(16), false),
        (16, 1 << 16, true),
        (20, scaled_active(20), false),
    ] {
        let trace = SyntheticTrace::new(log_n, active, 5, hot);
        let mut prover = ByteLinkProver::new(&metal).unwrap();
        let log = compare(&mut prover, &metal, &trace, active);
        let inputs = trace.inputs::<F>(11);
        let bytes = device_source(&metal, &trace.bytes);
        let source = ByteLinkSource {
            bytes: &bytes,
            log_rows: log_n as u32,
            active_rows: active,
        };
        let histograms = prover.histograms(&source, &inputs).unwrap();
        let (mut again, compression) = Recorder::new();
        let _ = prover
            .prove(&source, &histograms, &inputs, &compression, &mut again)
            .unwrap();
        assert!(
            again.log == log,
            "2^{log_n} hot={hot}: the second proof differs"
        );
    }
}

fn env_or<V: FromStr<Err: Debug>>(name: &str, default: V) -> V {
    std::env::var(name).map_or(default, |value| value.parse().unwrap())
}

extern "C" {
    /// libsystem_kernel (private libproc header): restarts `ri_interval_max_phys_footprint`.
    fn proc_reset_footprint_interval(pid: libc::c_int) -> libc::c_int;
}

/// `(ri_phys_footprint, ri_interval_max_phys_footprint)` of this process.
fn footprint() -> (u64, u64) {
    // SAFETY: proc_pid_rusage writes one complete rusage_info_v4 for RUSAGE_INFO_V4.
    unsafe {
        let mut info: rusage_info_v4 = std::mem::zeroed();
        let status = libc::proc_pid_rusage(libc::getpid(), RUSAGE_INFO_V4, (&raw mut info).cast());
        assert_eq!(status, 0);
        (info.ri_phys_footprint, info.ri_interval_max_phys_footprint)
    }
}

/// The Metal link at `2^LINK_LOG` rows (default 26): `LINK_REPS` proofs (default 3), each after
/// `LINK_COOL_SECONDS` idle (default 0.2), on U-scaled uniform bytes, or `LINK_FULL=1` without
/// the zero tail, `LINK_HOT=1` with 256 tuples per pack. Prints a `phase` row per phase and proof
/// (wall, GPU seconds, command buffers) and a `total` row (wall and GPU seconds of every phase,
/// the key sort included, transcript digest, footprint at the proof start and its peak during
/// the proof, arena bytes); with `LINK_CHECK=1` the first proof is also compared with the
/// reference prover's.
#[test]
#[ignore = "GPU bench; run through gpu-window.sh"]
#[expect(clippy::print_stdout, reason = "bench report")]
fn bench_link() {
    let log_n: u32 = env_or("LINK_LOG", 26);
    let active = if env_or("LINK_FULL", 0) == 1 {
        1 << log_n
    } else {
        scaled_active(log_n as usize)
    };
    let reps: usize = env_or("LINK_REPS", 3);
    let cool: f64 = env_or("LINK_COOL_SECONDS", 0.2);
    let check = env_or("LINK_CHECK", 0) == 1;
    let build = Instant::now();
    let trace = SyntheticTrace::new(
        log_n as usize,
        active,
        0x6c69_6e6b,
        env_or("LINK_HOT", 0) == 1,
    );
    let inputs = trace.inputs::<F>(7);
    let metal = SolinasMetal::for_akita().unwrap();
    let bytes = device_source(&metal, &trace.bytes);
    println!(
        "shape\tlog_n={log_n}\tactive={active}\tsetup_s={:.1}",
        build.elapsed().as_secs_f64()
    );
    let source = ByteLinkSource {
        bytes: &bytes,
        log_rows: log_n,
        active_rows: active,
    };
    let mut prover = ByteLinkProver::new(&metal).unwrap();
    for rep in 0..reps {
        std::thread::sleep(Duration::from_secs_f64(cool));
        let start = footprint().0;
        // SAFETY: takes only a pid and writes no caller memory.
        let _ = unsafe { proc_reset_footprint_interval(libc::getpid()) };
        let histograms = prover.histograms(&source, &inputs).unwrap();
        let (mut recorder, compression) = Recorder::new();
        let _ = prover
            .prove(&source, &histograms, &inputs, &compression, &mut recorder)
            .unwrap();
        let peak = footprint().1;
        let phases = prover.gpu.take_phases();
        let (mut wall, mut gpu) = (0.0, 0.0);
        for Phase {
            name,
            wall: w,
            gpu: g,
            commands,
        } in &phases
        {
            println!("phase\t{rep}\t{name}\t{w:.6}\t{g:.6}\t{commands}");
            wall += w;
            gpu += g;
        }
        let mut digest = T::new(b"byte-link-bench-digest");
        for (kind, values) in &recorder.log {
            digest.append_bytes(kind.as_bytes());
            digest.append_values(b"values", values);
        }
        let state = digest.state();
        println!(
            "total\t{rep}\t{wall:.6}\t{gpu:.6}\t{:016x}\t{start}\t{peak}\t{}",
            u64::from_be_bytes(state[..8].try_into().unwrap()),
            prover.gpu.arena_bytes()
        );
        if rep == 0 && check {
            drop(histograms);
            let _ = compare(&mut prover, &metal, &trace, active);
            println!("checked");
        }
    }
}
