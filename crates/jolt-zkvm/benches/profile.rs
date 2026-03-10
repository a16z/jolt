//! Profiling harness for the jolt-zkvm proving pipeline.
//!
//! Exercises the full prove → verify cycle at configurable scale with
//! tracing instrumentation. Produces Perfetto-compatible JSON traces
//! viewable at <https://ui.perfetto.dev/>.
//!
//! ```bash
//! # Default: 2^10 cycles, chrome tracing
//! cargo bench -p jolt-zkvm --bench profile
//!
//! # Custom scale with console output
//! cargo bench -p jolt-zkvm --bench profile -- --scale 14 --format default
//!
//! # With system metrics (CPU/memory counters in Perfetto)
//! cargo bench -p jolt-zkvm --bench profile --features jolt-profiling/monitor -- --scale 12
//! ```
#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::time::Instant;

use jolt_dory::DoryScheme;
use jolt_field::{Field, Fr};
use jolt_openings::AdditivelyHomomorphic;
use jolt_profiling::{setup_tracing, TracingFormat};
use jolt_transcript::{Blake2bTranscript, Transcript};
use jolt_zkvm::preprocessing::{preprocess, JoltConfig};
use jolt_zkvm::prover::prove;
use jolt_zkvm::r1cs;
use jolt_zkvm::stage::ProverStage;
use jolt_zkvm::stages::s3_claim_reductions::ClaimReductionStage;
use jolt_zkvm::stages::s4_ram_rw::RamRwCheckingStage;
use jolt_zkvm::stages::s4_rw_checking::RwCheckingStage;
use jolt_zkvm::stages::s5_ram_checking::RamCheckingStage;
use jolt_zkvm::stages::s6_booleanity::HammingBooleanityStage;
use jolt_zkvm::stages::s7_hamming_reduction::HammingReductionStage;
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

fn challenge_fn(c: u128) -> Fr {
    Fr::from_u128(c)
}

fn nop_cycle_witness(unexpanded_pc: u64, pc: u64) -> Vec<Fr> {
    let mut w = vec![Fr::from_u64(0); r1cs::NUM_VARS_PER_CYCLE];
    w[r1cs::V_CONST] = Fr::from_u64(1);
    w[r1cs::V_UNEXPANDED_PC] = Fr::from_u64(unexpanded_pc);
    w[r1cs::V_NEXT_UNEXPANDED_PC] = Fr::from_u64(unexpanded_pc + 4);
    w[r1cs::V_PC] = Fr::from_u64(pc);
    w[r1cs::V_NEXT_PC] = Fr::from_u64(pc + 1);
    w
}

fn add_cycle_witness(unexpanded_pc: u64, pc: u64) -> Vec<Fr> {
    let mut w = vec![Fr::from_u64(0); r1cs::NUM_VARS_PER_CYCLE];
    w[r1cs::V_CONST] = Fr::from_u64(1);
    w[r1cs::V_FLAG_ADD_OPERANDS] = Fr::from_u64(1);
    w[r1cs::V_FLAG_WRITE_LOOKUP_OUTPUT_TO_RD] = Fr::from_u64(1);
    w[r1cs::V_IS_RD_NOT_ZERO] = Fr::from_u64(1);
    w[r1cs::V_LEFT_INSTRUCTION_INPUT] = Fr::from_u64(7);
    w[r1cs::V_RIGHT_INSTRUCTION_INPUT] = Fr::from_u64(3);
    w[r1cs::V_PRODUCT] = Fr::from_u64(21);
    w[r1cs::V_LEFT_LOOKUP_OPERAND] = Fr::from_u64(0);
    w[r1cs::V_RIGHT_LOOKUP_OPERAND] = Fr::from_u64(10);
    w[r1cs::V_LOOKUP_OUTPUT] = Fr::from_u64(10);
    w[r1cs::V_WRITE_LOOKUP_OUTPUT_TO_RD] = Fr::from_u64(1);
    w[r1cs::V_WRITE_PC_TO_RD] = Fr::from_u64(0);
    w[r1cs::V_RD_WRITE_VALUE] = Fr::from_u64(10);
    w[r1cs::V_UNEXPANDED_PC] = Fr::from_u64(unexpanded_pc);
    w[r1cs::V_NEXT_UNEXPANDED_PC] = Fr::from_u64(unexpanded_pc + 4);
    w[r1cs::V_PC] = Fr::from_u64(pc);
    w[r1cs::V_NEXT_PC] = Fr::from_u64(pc + 1);
    w
}

/// Synthetic polynomial data for all sumcheck stages (S3–S7).
#[derive(Clone)]
struct SyntheticStageData {
    // S3: claim reduction
    poly_a: Vec<Fr>,
    poly_b: Vec<Fr>,
    s3_c0: Fr,
    s3_c1: Fr,
    // S4: register read-write checking (5 polys + challenges)
    reg_val: Vec<Fr>,
    rs1_ra: Vec<Fr>,
    rs2_ra: Vec<Fr>,
    rd_wa: Vec<Fr>,
    rd_inc: Vec<Fr>,
    reg_gamma: Fr,
    // S4b: RAM read-write checking (3 polys + challenges)
    ram_rw_ra: Vec<Fr>,
    ram_rw_val: Vec<Fr>,
    ram_rw_inc: Vec<Fr>,
    ram_rw_c0: Fr,
    ram_rw_c1: Fr,
    // S5: RAM checking (2 polys + challenges)
    val_final: Vec<Fr>,
    ram_ra: Vec<Fr>,
    s5_output_c0: Fr,
    s5_output_c1: Fr,
    s5_raf_c0: Fr,
    // S6: booleanity (boolean polynomial)
    h_evals: Vec<Fr>,
    // S7: hamming reduction (3 polys + coefficients)
    hamming_polys: Vec<Vec<Fr>>,
    hamming_coeffs: Vec<Fr>,
    num_vars: usize,
}

impl SyntheticStageData {
    fn generate(n: usize, num_vars: usize, rng: &mut ChaCha20Rng) -> Self {
        let rand_table =
            |rng: &mut ChaCha20Rng| -> Vec<Fr> { (0..n).map(|_| Fr::random(rng)).collect() };

        // Boolean polynomial for booleanity check — must be 0/1 valued.
        let h_evals: Vec<Fr> = (0..n).map(|i| Fr::from_u64((i % 2) as u64)).collect();

        Self {
            poly_a: rand_table(rng),
            poly_b: rand_table(rng),
            s3_c0: Fr::random(rng),
            s3_c1: Fr::random(rng),
            reg_val: rand_table(rng),
            rs1_ra: rand_table(rng),
            rs2_ra: rand_table(rng),
            rd_wa: rand_table(rng),
            rd_inc: rand_table(rng),
            reg_gamma: Fr::random(rng),
            ram_rw_ra: rand_table(rng),
            ram_rw_val: rand_table(rng),
            ram_rw_inc: rand_table(rng),
            ram_rw_c0: Fr::random(rng),
            ram_rw_c1: Fr::random(rng),
            val_final: rand_table(rng),
            ram_ra: rand_table(rng),
            s5_output_c0: Fr::random(rng),
            s5_output_c1: Fr::random(rng),
            s5_raf_c0: Fr::random(rng),
            h_evals,
            hamming_polys: (0..3).map(|_| rand_table(rng)).collect(),
            hamming_coeffs: vec![Fr::from_u64(1), Fr::random(rng), Fr::random(rng)],
            num_vars,
        }
    }

    fn build_stages(&self, r_y: &[Fr]) -> Vec<Box<dyn ProverStage<Fr, Blake2bTranscript>>> {
        let eq_point = r_y.to_vec();
        let nv = self.num_vars;
        let gamma = self.reg_gamma;
        let eq_eval = Fr::from_u64(1);

        let mut stages: Vec<Box<dyn ProverStage<Fr, Blake2bTranscript>>> = Vec::new();

        // S3: Claim reduction
        stages.push(Box::new(ClaimReductionStage::increment(
            self.poly_a.clone(),
            self.poly_b.clone(),
            eq_point.clone(),
            self.s3_c0,
            self.s3_c1,
        )));

        // S4: Register read-write checking
        let reg_eq: Vec<Fr> = (0..nv).map(|i| eq_point[i % eq_point.len()]).collect();
        stages.push(Box::new(RwCheckingStage::new(
            (
                self.reg_val.clone(),
                self.rs1_ra.clone(),
                self.rs2_ra.clone(),
                self.rd_wa.clone(),
                self.rd_inc.clone(),
            ),
            reg_eq.clone(),
            vec![eq_eval, eq_eval * gamma, eq_eval * gamma * gamma],
            (self.ram_rw_inc.clone(), self.rd_wa.clone()),
            reg_eq,
            Fr::random(&mut ChaCha20Rng::seed_from_u64(0)),
        )));

        // S4b: RAM read-write checking
        stages.push(Box::new(RamRwCheckingStage::new(
            self.ram_rw_ra.clone(),
            self.ram_rw_val.clone(),
            self.ram_rw_inc.clone(),
            eq_point.clone(),
            [self.ram_rw_c0, self.ram_rw_c1],
        )));

        // S5: RAM checking
        stages.push(Box::new(RamCheckingStage::new(
            self.val_final.clone(),
            eq_point.clone(),
            [self.s5_output_c0, self.s5_output_c1],
            self.ram_ra.clone(),
            eq_point.clone(),
            self.s5_raf_c0,
        )));

        // S6: Booleanity
        stages.push(Box::new(HammingBooleanityStage::new(
            self.h_evals.clone(),
            eq_point.clone(),
        )));

        // S7: Hamming reduction
        stages.push(Box::new(HammingReductionStage::new(
            self.hamming_polys.clone(),
            self.hamming_coeffs.clone(),
            eq_point,
        )));

        stages
    }
}

/// Generate cycle witnesses: alternating NOP/ADD for realistic diversity.
fn generate_cycle_witnesses(num_cycles: usize) -> Vec<Vec<Fr>> {
    (0..num_cycles)
        .map(|i| {
            let upc = (i as u64) * 4;
            let pc = i as u64;
            if i % 2 == 0 {
                nop_cycle_witness(upc, pc)
            } else {
                add_cycle_witness(upc, pc)
            }
        })
        .collect()
}

fn run_profile<PCS: AdditivelyHomomorphic<Field = Fr>>(
    num_cycles: usize,
    prover_setup: &PCS::ProverSetup,
    verifier_setup: &PCS::VerifierSetup,
) {
    let mut rng = ChaCha20Rng::seed_from_u64(42);

    let config = JoltConfig { num_cycles };
    let key = preprocess::<Fr, PCS>(&config, |_| (prover_setup.clone(), verifier_setup.clone()));

    let cycle_witnesses = {
        let _span = tracing::info_span!("generate_witnesses").entered();
        generate_cycle_witnesses(num_cycles)
    };

    let num_col_vars = key.spartan_key.num_col_vars();
    let n = 1usize << num_col_vars;

    // Synthetic stage data — random polynomials at production scale.
    let stage_data = {
        let _span = tracing::info_span!("generate_reduction_data").entered();
        SyntheticStageData::generate(n, num_col_vars, &mut rng)
    };

    let (com_a, _) = PCS::commit(&stage_data.poly_a, prover_setup);
    let (com_b, _) = PCS::commit(&stage_data.poly_b, prover_setup);
    let poly_commitments: Vec<PCS::Output> = vec![com_a, com_b];

    let mut pt = Blake2bTranscript::new(b"jolt-profile");

    let now = Instant::now();
    let proof = {
        let sd = stage_data.clone();
        prove::<PCS, Blake2bTranscript>(
            &key,
            &cycle_witnesses,
            poly_commitments,
            |_r_x, r_y| sd.build_stages(r_y),
            &mut pt,
            challenge_fn,
        )
        .expect("proving should succeed")
    };
    let prove_duration = now.elapsed();

    let num_stages = proof.stage_proofs.len();
    let num_openings = proof.opening_proofs.len();

    println!(
        "Proved {num_cycles} cycles in {:.3}s ({:.1} kHz)",
        prove_duration.as_secs_f64(),
        num_cycles as f64 / prove_duration.as_secs_f64() / 1000.0,
    );
    println!("  stages: {num_stages}, opening proofs: {num_openings}");

    // Full verification is only meaningful at small scales with real witnesses.
    // With synthetic multi-stage data, we skip the verifier round-trip and focus
    // profiling on the prover pipeline (which dominates wall-clock time).
    println!("(verifier skipped — synthetic multi-stage data)");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut scale: usize = 10;
    let mut format = TracingFormat::Chrome;
    let mut console = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--scale" => {
                i += 1;
                scale = args[i].parse().expect("--scale requires a number");
            }
            "--format" => {
                i += 1;
                match args[i].to_lowercase().as_str() {
                    "chrome" => format = TracingFormat::Chrome,
                    "default" => {
                        format = TracingFormat::Default;
                        console = true;
                    }
                    "both" => console = true,
                    _ => eprintln!("Unknown format: {}. Using chrome.", args[i]),
                }
            }
            _ => {}
        }
        i += 1;
    }

    let num_cycles = 1usize << scale;

    let mut formats = vec![format];
    if console && format == TracingFormat::Chrome {
        formats.push(TracingFormat::Default);
    }

    let trace_name = format!("jolt_zkvm_profile_{scale}");

    // cargo bench sets CWD to the crate directory; write traces to the
    // workspace-level benchmark-runs/ directory for consistency with jolt-core.
    let workspace_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root");
    let _ = std::env::set_current_dir(workspace_root);

    let _guards = setup_tracing(&formats, &trace_name);

    println!("jolt-zkvm profiling: 2^{scale} = {num_cycles} cycles");
    println!("Tracing output: benchmark-runs/perfetto_traces/{trace_name}.json");
    println!();

    // Compute max polynomial size from R1CS dimensions.
    let max_vars = {
        let num_vars_padded = r1cs::NUM_VARS_PER_CYCLE.next_power_of_two();
        let total_cols = num_cycles * num_vars_padded;
        total_cols.next_power_of_two().trailing_zeros() as usize
    };

    println!("Setting up Dory SRS for {max_vars} variables...");
    let (prover_setup, verifier_setup) = {
        let _span = tracing::info_span!("dory_setup", max_vars).entered();
        let ps = DoryScheme::setup_prover(max_vars);
        let vs = DoryScheme::setup_verifier(max_vars);
        (ps, vs)
    };

    let span = tracing::info_span!("E2E_profile", scale, num_cycles);
    span.in_scope(|| {
        run_profile::<DoryScheme>(num_cycles, &prover_setup, &verifier_setup);
    });
}
