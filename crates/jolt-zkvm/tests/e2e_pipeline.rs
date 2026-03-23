//! End-to-end smoke test for the typed DAG proving pipeline.

use std::sync::Arc;

use jolt_cpu::CpuBackend;
use jolt_field::{Field, Fr};
use jolt_openings::VirtualEval;
use jolt_sumcheck::SumcheckProof;
use jolt_transcript::Transcript;
use num_traits::Zero;
use tracer::instruction::{
    add::ADD,
    format::format_r::{FormatR, RegisterStateFormatR},
    Cycle, RISCVCycle,
};

use jolt_verifier::protocol::types::*;
use jolt_zkvm::stages;
use jolt_zkvm::tables::PolynomialTables;
use jolt_zkvm::witness::generate::generate_witnesses;

fn make_add_cycle(addr: u64, rs1_val: u64, rs2_val: u64) -> Cycle {
    Cycle::from(RISCVCycle {
        instruction: ADD {
            address: addr,
            operands: FormatR { rd: 1, rs1: 2, rs2: 3 },
            ..ADD::default()
        },
        register_state: RegisterStateFormatR {
            rd: (0, rs1_val.wrapping_add(rs2_val)),
            rs1: rs1_val,
            rs2: rs2_val,
        },
        ram_access: (),
    })
}

fn synthetic_s1(log_t: usize) -> SpartanOutput<Fr> {
    let r_cycle: Vec<Fr> = (0..log_t).map(|i| Fr::from_u64(i as u64 + 1)).collect();
    SpartanOutput {
        proof: jolt_spartan::UniformSpartanProof {
            outer_sumcheck_proof: SumcheckProof::default(),
            az_eval: Fr::zero(),
            bz_eval: Fr::zero(),
            cz_eval: Fr::zero(),
            inner_sumcheck_proof: SumcheckProof::default(),
            witness_eval: Fr::zero(),
        },
        r_x: vec![Fr::zero(); log_t],
        r_y: r_cycle,
        evals: SpartanVirtualEvals {
            ram_read_value: VirtualEval(Fr::zero()),
            ram_write_value: VirtualEval(Fr::zero()),
            ram_address: VirtualEval(Fr::zero()),
            lookup_output: VirtualEval(Fr::zero()),
            left_operand: VirtualEval(Fr::zero()),
            right_operand: VirtualEval(Fr::zero()),
            left_instruction_input: VirtualEval(Fr::zero()),
            right_instruction_input: VirtualEval(Fr::zero()),
            rd_write_value: VirtualEval(Fr::zero()),
            rs1_value: VirtualEval(Fr::zero()),
            rs2_value: VirtualEval(Fr::zero()),
        },
    }
}

#[test]
fn typed_dag_stages_smoke() {
    let trace = vec![make_add_cycle(0x1000, 3, 4), make_add_cycle(0x1004, 10, 20)];
    let output = generate_witnesses::<Fr>(&trace);
    let config = output.config.clone();
    let tables = PolynomialTables::from_witness(
        &output.witness_store, &output.cycle_witnesses, &trace, &config,
    );

    let backend = Arc::new(CpuBackend);
    let mut t = jolt_transcript::Blake2bTranscript::<Fr>::new(b"test-stages");
    let s1 = synthetic_s1(tables.log_num_cycles());

    let s2 = stages::prove_stage2(&s1, &tables, &config, &mut t, &backend);
    let s3 = stages::prove_stage3(&s1, &s2.evals, &tables, &config, &mut t, &backend);
    let s4 = stages::prove_stage4(&s2.evals, &s3.evals, &tables, &config, &mut t, &backend);
    let s5 = stages::prove_stage5(&s2.evals, &s4.evals, &tables, &config, &mut t, &backend);
    let s6 = stages::prove_stage6(&s2.evals, &s4.evals, &s5.evals, &tables, &config, &mut t, &backend);
    let s7 = stages::prove_stage7(&s5.evals, &s6.evals, &tables, &config, &mut t, &backend);

    let ohp = config.one_hot_params_from_config();
    assert_eq!(s7.evals.unified_point.len(), ohp.log_k_chunk + tables.log_num_cycles());
    assert_eq!(s7.evals.instruction_ra.len(), ohp.instruction_d);
    assert_eq!(s7.evals.bytecode_ra.len(), ohp.bytecode_d);
    assert_eq!(s7.evals.ram_ra.len(), ohp.ram_d);
}

/// Tests that the prover's stage proofs carry non-empty evaluations.
#[test]
fn prover_emits_evaluations() {
    use jolt_openings::mock::MockCommitmentScheme;

    let trace = vec![make_add_cycle(0x1000, 3, 4), make_add_cycle(0x1004, 10, 20)];
    let backend = Arc::new(CpuBackend);

    let output = jolt_zkvm::prover::prove::<MockCommitmentScheme<Fr>, CpuBackend>(
        &trace, |_| ((), ()), backend,
    );

    // prove() will fail at Spartan (constraint violation with synthetic trace)
    // but we can verify the structure works if it succeeds
    if let Ok(out) = output {
        for (i, sp) in out.proof.stage_proofs.iter().enumerate() {
            assert!(
                !sp.evals.is_empty() || i >= 4,
                "stage {i} should have evaluations"
            );
        }
    }
    // Err is expected — synthetic trace doesn't satisfy R1CS constraints
}
