#[cfg(not(feature = "zk"))]
fn main() {}

#[cfg(feature = "zk")]
mod profile {
    #![expect(clippy::print_stdout, reason = "benchmark reports metrics to stdout")]

    use std::error::Error;
    use std::process::Command;
    use std::sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc,
    };
    use std::thread::JoinHandle;
    use std::time::{Duration, Instant};

    use jolt_crypto::{Bn254, JoltGroup};
    use jolt_field::{Fr, FromPrimitiveInt, RandomSampling};
    use jolt_hyperkzg::{
        HyperKZGCommitment, HyperKZGProof, HyperKZGProverSetup, HyperKZGScheme,
        HyperKZGVerifierSetup,
    };
    use jolt_openings::{CommitmentScheme, ZkOpeningScheme};
    use jolt_poly::Polynomial;
    use jolt_transcript::{Blake2bTranscript, Transcript};
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    const NUM_VARS: usize = 18;
    const NUM_EVALS: usize = 1 << NUM_VARS;
    const DEFAULT_SAMPLES: usize = 3;
    const MEMORY_SAMPLE_INTERVAL: Duration = Duration::from_millis(1);
    const CHILD_MODE_ENV: &str = "JOLT_HYPERKZG_PROFILE_CHILD_MODE";
    const SAMPLES_ENV: &str = "JOLT_HYPERKZG_PROFILE_SAMPLES";
    const RESULT_PREFIX: &str = "RESULT";

    type BenchResult<T> = Result<T, Box<dyn Error>>;
    type Scheme = HyperKZGScheme<Bn254>;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Mode {
        Transparent,
        Zk,
    }

    impl Mode {
        const ALL: [Self; 2] = [Self::Transparent, Self::Zk];

        const fn as_str(self) -> &'static str {
            match self {
                Self::Transparent => "transparent",
                Self::Zk => "zk",
            }
        }

        fn parse(value: &str) -> BenchResult<Self> {
            match value {
                "transparent" => Ok(Self::Transparent),
                "zk" => Ok(Self::Zk),
                _ => Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!("unknown HyperKZG profile mode {value:?}"),
                )
                .into()),
            }
        }
    }

    pub fn main() -> BenchResult<()> {
        let samples = sample_count()?;
        if let Some(mode) = child_mode()? {
            let metrics = run_mode(mode, samples)?;
            print_machine_result(mode, &metrics);
            return Ok(());
        }

        let mut rows = Vec::with_capacity(Mode::ALL.len());
        for mode in Mode::ALL {
            rows.push(run_child(mode, samples)?);
        }
        print_report(samples, &rows);
        Ok(())
    }

    fn run_child(mode: Mode, samples: usize) -> BenchResult<(Mode, ModeMetrics)> {
        let output = Command::new(std::env::current_exe()?)
            .env(CHILD_MODE_ENV, mode.as_str())
            .env(SAMPLES_ENV, samples.to_string())
            .output()?;

        if !output.status.success() {
            return Err(std::io::Error::other(format!(
                "HyperKZG profile child {:?} failed with status {:?}\nstdout:\n{}\nstderr:\n{}",
                mode,
                output.status.code(),
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            ))
            .into());
        }

        parse_machine_result(&output.stdout)
    }

    fn run_mode(mode: Mode, samples: usize) -> BenchResult<ModeMetrics> {
        let mut metrics = ModeMetrics::default();
        match mode {
            Mode::Transparent => {
                let pk = make_clear_setup();
                let vk = Scheme::verifier_setup(&pk);
                for sample_index in 0..samples {
                    let (poly, point, eval) = sample_input(sample_index);
                    metrics.record(profile_clear_sample(&pk, &vk, &poly, &point, eval)?);
                }
            }
            Mode::Zk => {
                let pk = make_zk_setup();
                let vk = Scheme::verifier_setup(&pk);
                for sample_index in 0..samples {
                    let (poly, point, eval) = sample_input(sample_index);
                    metrics.record(profile_zk_sample(&pk, &vk, &poly, &point, eval)?);
                }
            }
        }
        Ok(metrics)
    }

    fn make_clear_setup() -> HyperKZGProverSetup<Bn254> {
        let beta = Fr::from_u64(12345);
        let g1 = Bn254::g1_generator();
        let g2 = Bn254::g2_generator();
        Scheme::setup_from_secret(beta, NUM_EVALS, g1, g2)
    }

    fn make_zk_setup() -> HyperKZGProverSetup<Bn254> {
        let beta = Fr::from_u64(12345);
        let g1 = Bn254::g1_generator();
        let g2 = Bn254::g2_generator();
        let hiding_g1 = g1.scalar_mul(&Fr::from_u64(17));
        Scheme::setup_zk_from_secret(beta, NUM_EVALS, g1, hiding_g1, g2)
    }

    fn sample_input(sample_index: usize) -> (Polynomial<Fr>, Vec<Fr>, Fr) {
        let mut rng = ChaCha20Rng::seed_from_u64(0x5eed_0000 + sample_index as u64);
        let poly = Polynomial::<Fr>::random(NUM_VARS, &mut rng);
        let point: Vec<Fr> = (0..NUM_VARS).map(|_| Fr::random(&mut rng)).collect();
        let eval = poly.evaluate(&point);
        (poly, point, eval)
    }

    fn profile_clear_sample(
        pk: &HyperKZGProverSetup<Bn254>,
        vk: &HyperKZGVerifierSetup<Bn254>,
        poly: &Polynomial<Fr>,
        point: &[Fr],
        eval: Fr,
    ) -> BenchResult<SampleMetrics> {
        let ((commitment, proof), prover_time, peak) = measure_prover_peak(|| {
            let (commitment, hint) = <Scheme as CommitmentScheme>::commit(poly.evaluations(), pk);
            let mut transcript = Blake2bTranscript::new(b"hyperkzg-profile-clear");
            let proof = <Scheme as CommitmentScheme>::open(
                poly,
                point,
                eval,
                pk,
                Some(hint),
                &mut transcript,
            );
            (commitment, proof)
        });

        let proof_bytes = serialized_proof_size(&proof)?;
        let verifier_time = time_verify_clear(&commitment, point, eval, &proof, vk)?;

        Ok(SampleMetrics {
            proof_bytes,
            prover_time,
            verifier_time,
            peak_rss_bytes: peak.peak_bytes,
            peak_delta_bytes: peak.delta_bytes,
        })
    }

    fn profile_zk_sample(
        pk: &HyperKZGProverSetup<Bn254>,
        vk: &HyperKZGVerifierSetup<Bn254>,
        poly: &Polynomial<Fr>,
        point: &[Fr],
        eval: Fr,
    ) -> BenchResult<SampleMetrics> {
        let ((commitment, proof, y_out), prover_time, peak) = measure_prover_peak(|| {
            let (commitment, hint) = <Scheme as ZkOpeningScheme>::commit_zk(poly.evaluations(), pk);
            let mut transcript = Blake2bTranscript::new(b"hyperkzg-profile-zk");
            let (proof, y_out, _blind) =
                Scheme::open_zk(poly, point, eval, pk, hint, &mut transcript);
            (commitment, proof, y_out)
        });

        let proof_bytes = serialized_proof_size(&proof)?;
        let verifier_time = time_verify_zk(&commitment, point, &proof, vk, &y_out)?;

        Ok(SampleMetrics {
            proof_bytes,
            prover_time,
            verifier_time,
            peak_rss_bytes: peak.peak_bytes,
            peak_delta_bytes: peak.delta_bytes,
        })
    }

    fn time_verify_clear(
        commitment: &HyperKZGCommitment<Bn254>,
        point: &[Fr],
        eval: Fr,
        proof: &HyperKZGProof<Bn254>,
        vk: &HyperKZGVerifierSetup<Bn254>,
    ) -> BenchResult<Duration> {
        let start = Instant::now();
        let mut transcript = Blake2bTranscript::new(b"hyperkzg-profile-clear");
        <Scheme as CommitmentScheme>::verify(commitment, point, eval, proof, vk, &mut transcript)?;
        Ok(start.elapsed())
    }

    fn time_verify_zk(
        commitment: &HyperKZGCommitment<Bn254>,
        point: &[Fr],
        proof: &HyperKZGProof<Bn254>,
        vk: &HyperKZGVerifierSetup<Bn254>,
        expected_y_out: &<Scheme as ZkOpeningScheme>::HidingCommitment,
    ) -> BenchResult<Duration> {
        let start = Instant::now();
        let mut transcript = Blake2bTranscript::new(b"hyperkzg-profile-zk");
        let verified_y_out = Scheme::verify_zk(commitment, point, proof, vk, &mut transcript)?;
        if &verified_y_out != expected_y_out {
            return Err(std::io::Error::other(
                "ZK verifier returned the wrong output hiding commitment",
            )
            .into());
        }
        Ok(start.elapsed())
    }

    fn serialized_proof_size(proof: &HyperKZGProof<Bn254>) -> BenchResult<usize> {
        let bytes = bincode::serde::encode_to_vec(proof, bincode::config::standard())?;
        Ok(bytes.len())
    }

    fn measure_prover_peak<T>(f: impl FnOnce() -> T) -> (T, Duration, PeakMeasurement) {
        let sampler = PeakSampler::start();
        let start = Instant::now();
        let result = f();
        let elapsed = start.elapsed();
        let peak = sampler.stop();
        (result, elapsed, peak)
    }

    #[derive(Clone, Copy)]
    struct PeakMeasurement {
        peak_bytes: Option<usize>,
        delta_bytes: Option<usize>,
    }

    struct PeakSampler {
        baseline: Option<usize>,
        peak: Arc<AtomicUsize>,
        stop: Arc<AtomicBool>,
        handle: Option<JoinHandle<()>>,
    }

    impl PeakSampler {
        fn start() -> Self {
            let baseline = current_rss_bytes();
            let peak = Arc::new(AtomicUsize::new(baseline.unwrap_or_default()));
            let stop = Arc::new(AtomicBool::new(false));
            let thread_peak = Arc::clone(&peak);
            let thread_stop = Arc::clone(&stop);

            let handle = std::thread::spawn(move || {
                while !thread_stop.load(Ordering::Relaxed) {
                    if let Some(bytes) = current_rss_bytes() {
                        let _ = thread_peak.fetch_max(bytes, Ordering::Relaxed);
                    }
                    std::thread::sleep(MEMORY_SAMPLE_INTERVAL);
                }
                if let Some(bytes) = current_rss_bytes() {
                    let _ = thread_peak.fetch_max(bytes, Ordering::Relaxed);
                }
            });

            Self {
                baseline,
                peak,
                stop,
                handle: Some(handle),
            }
        }

        fn stop(mut self) -> PeakMeasurement {
            self.stop.store(true, Ordering::Relaxed);
            if let Some(handle) = self.handle.take() {
                let _ = handle.join();
            }
            let Some(baseline) = self.baseline else {
                return PeakMeasurement {
                    peak_bytes: None,
                    delta_bytes: None,
                };
            };
            let peak = self.peak.load(Ordering::Relaxed);
            PeakMeasurement {
                peak_bytes: Some(peak),
                delta_bytes: Some(peak.saturating_sub(baseline)),
            }
        }
    }

    fn current_rss_bytes() -> Option<usize> {
        memory_stats::memory_stats().map(|stats| stats.physical_mem)
    }

    #[derive(Clone, Copy)]
    struct SampleMetrics {
        proof_bytes: usize,
        prover_time: Duration,
        verifier_time: Duration,
        peak_rss_bytes: Option<usize>,
        peak_delta_bytes: Option<usize>,
    }

    #[derive(Clone, Default)]
    struct ModeMetrics {
        samples: usize,
        proof_bytes_total: usize,
        prover_time_total: Duration,
        verifier_time_total: Duration,
        max_peak_rss_bytes: Option<usize>,
        max_peak_delta_bytes: Option<usize>,
    }

    impl ModeMetrics {
        fn record(&mut self, sample: SampleMetrics) {
            self.samples += 1;
            self.proof_bytes_total += sample.proof_bytes;
            self.prover_time_total += sample.prover_time;
            self.verifier_time_total += sample.verifier_time;
            self.max_peak_rss_bytes = max_optional(self.max_peak_rss_bytes, sample.peak_rss_bytes);
            self.max_peak_delta_bytes =
                max_optional(self.max_peak_delta_bytes, sample.peak_delta_bytes);
        }

        fn average_proof_bytes(&self) -> usize {
            self.proof_bytes_total / self.samples
        }

        fn average_prover_time(&self) -> Duration {
            self.prover_time_total / self.samples as u32
        }

        fn average_verifier_time(&self) -> Duration {
            self.verifier_time_total / self.samples as u32
        }
    }

    fn max_optional(lhs: Option<usize>, rhs: Option<usize>) -> Option<usize> {
        match (lhs, rhs) {
            (Some(lhs), Some(rhs)) => Some(lhs.max(rhs)),
            (None, Some(rhs)) => Some(rhs),
            (lhs, None) => lhs,
        }
    }

    fn print_report(samples: usize, rows: &[(Mode, ModeMetrics)]) {
        println!("HyperKZG 2^{NUM_VARS} profile ({samples} samples)");
        println!("prover = commit + open; setup and input generation excluded from timing");
        println!("memory = child-process max RSS observed during prover work");
        println!();
        println!(
            "{:<13} {:>12} {:>16} {:>16} {:>16} {:>16}",
            "mode", "proof bytes", "prover avg", "verifier avg", "peak RSS", "RSS delta"
        );
        for (mode, metrics) in rows {
            print_row(*mode, metrics);
        }
    }

    fn print_row(mode: Mode, metrics: &ModeMetrics) {
        println!(
            "{:<13} {:>12} {:>16} {:>16} {:>16} {:>16}",
            mode.as_str(),
            metrics.average_proof_bytes(),
            format_duration(metrics.average_prover_time()),
            format_duration(metrics.average_verifier_time()),
            format_memory(metrics.max_peak_rss_bytes),
            format_memory(metrics.max_peak_delta_bytes)
        );
    }

    fn print_machine_result(mode: Mode, metrics: &ModeMetrics) {
        println!(
            "{RESULT_PREFIX}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            mode.as_str(),
            metrics.samples,
            metrics.proof_bytes_total,
            metrics.prover_time_total.as_nanos(),
            metrics.verifier_time_total.as_nanos(),
            machine_optional_usize(metrics.max_peak_rss_bytes),
            machine_optional_usize(metrics.max_peak_delta_bytes)
        );
    }

    fn parse_machine_result(stdout: &[u8]) -> BenchResult<(Mode, ModeMetrics)> {
        let stdout = String::from_utf8(stdout.to_vec())?;
        let Some(line) = stdout.lines().find(|line| line.starts_with(RESULT_PREFIX)) else {
            return Err(std::io::Error::other(format!(
                "HyperKZG profile child did not emit a {RESULT_PREFIX} line:\n{stdout}"
            ))
            .into());
        };

        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() != 8 {
            return Err(std::io::Error::other(format!(
                "malformed HyperKZG profile result line: {line}"
            ))
            .into());
        }

        let mode = Mode::parse(fields[1])?;
        let samples = fields[2].parse::<usize>()?;
        let proof_bytes_total = fields[3].parse::<usize>()?;
        let prover_time_total = duration_from_nanos(fields[4].parse::<u128>()?)?;
        let verifier_time_total = duration_from_nanos(fields[5].parse::<u128>()?)?;
        let max_peak_rss_bytes = parse_optional_usize(fields[6])?;
        let max_peak_delta_bytes = parse_optional_usize(fields[7])?;

        Ok((
            mode,
            ModeMetrics {
                samples,
                proof_bytes_total,
                prover_time_total,
                verifier_time_total,
                max_peak_rss_bytes,
                max_peak_delta_bytes,
            },
        ))
    }

    fn duration_from_nanos(nanos: u128) -> BenchResult<Duration> {
        let nanos = u64::try_from(nanos)?;
        Ok(Duration::from_nanos(nanos))
    }

    fn machine_optional_usize(value: Option<usize>) -> String {
        value.map_or_else(|| "-".to_string(), |value| value.to_string())
    }

    fn parse_optional_usize(value: &str) -> BenchResult<Option<usize>> {
        if value == "-" {
            return Ok(None);
        }
        Ok(Some(value.parse::<usize>()?))
    }

    fn format_duration(duration: Duration) -> String {
        format!("{:.2} ms", duration.as_secs_f64() * 1_000.0)
    }

    fn format_memory(bytes: Option<usize>) -> String {
        bytes.map_or_else(
            || "unavailable".to_string(),
            |bytes| format!("{:.2} MiB", bytes as f64 / (1024.0 * 1024.0)),
        )
    }

    fn child_mode() -> BenchResult<Option<Mode>> {
        let Some(raw) = std::env::var_os(CHILD_MODE_ENV) else {
            return Ok(None);
        };
        let raw = raw.into_string().map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("{CHILD_MODE_ENV} must be valid UTF-8"),
            )
        })?;
        Ok(Some(Mode::parse(&raw)?))
    }

    fn sample_count() -> BenchResult<usize> {
        let Some(raw) = std::env::var_os(SAMPLES_ENV) else {
            return Ok(DEFAULT_SAMPLES);
        };
        let raw = raw.into_string().map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("{SAMPLES_ENV} must be valid UTF-8"),
            )
        })?;
        let samples = raw.parse::<usize>()?;
        if samples == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("{SAMPLES_ENV} must be nonzero"),
            )
            .into());
        }
        Ok(samples)
    }
}

#[cfg(feature = "zk")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    profile::main()
}
