//! Poseidon2 BN254 t=3 cycle-count benchmark — measures the v2 BN254 Fr
//! coprocessor SDK against a popular public Poseidon2 library.
//!
//! Two guests (identical HorizenLabs RC3 parameters: d=5, R_F=8, R_P=56,
//! t=3, MAT_DIAG_M_1=[1,1,2]):
//!
//!   * `bn254-fr-poseidon2-sdk-guest`: uses `jolt-inlines-bn254-fr::Fr`,
//!     which dispatches Fr add/sub/mul/inv to the v2 FR coprocessor's
//!     `FieldOp` instruction (1 cycle per op + 7-cycle Horner loads
//!     + 12-cycle advice-extract per call). Traced via the two-pass
//!     advice flow (compute_advice ELF populates tape, normal ELF
//!     reads it).
//!   * `bn254-fr-poseidon2-external-guest`: vendored `taceo-poseidon2`
//!     v0.2.1, the de-facto public Poseidon2 reference (MIT/Apache-2.0)
//!     running on stock RISC-V with `ark_bn254::Fr`.
//!
//! Both guests must produce identical permutation outputs; only the
//! underlying Fr arithmetic differs.
//!
//! Run with `--nocapture`:
//!   cargo nextest run -p jolt-equivalence --test poseidon2_cycle_count \
//!     --no-capture
#![allow(non_snake_case, clippy::print_stderr)]

use jolt_host::Program;
use tracer::instruction::Cycle;

fn count_real_cycles(trace: &[Cycle]) -> usize {
    trace.iter().filter(|c| !matches!(c, Cycle::NoOp)).count()
}

#[test]
fn poseidon2_cycle_count_sdk_vs_taceo() {
    // Input state (1, 2, 3) — each Fr fits in a single u64 limb.
    let s0: [u64; 4] = [1, 0, 0, 0];
    let s1: [u64; 4] = [2, 0, 0, 0];
    let s2: [u64; 4] = [3, 0, 0, 0];
    let inputs =
        postcard::to_stdvec(&(s0, s1, s2)).expect("postcard encode poseidon2 inputs");

    // SDK guest — two-pass advice flow:
    //   Pass 1 (compute_advice ELF): SDK Fr ops compute via ark_bn254 and
    //   write 4 limbs to the advice tape per op via VirtualHostIO.
    //   Pass 2 (normal ELF): SDK Fr ops emit 7-cyc Horner load + 1-cyc
    //   FieldOp + 12-cyc advice extract + FieldAssertEq per op.
    let mut sdk_program = Program::new("bn254-fr-poseidon2-sdk-guest");
    let _ = sdk_program
        .set_stack_size(1 << 20)
        .set_heap_size(1 << 20)
        .set_max_input_size(8192);
    let (_, sdk_trace, _, sdk_device) =
        sdk_program.trace_two_pass_advice(&inputs, &[], &[]);
    let sdk_real = count_real_cycles(&sdk_trace);
    let sdk_field_ops = sdk_trace
        .iter()
        .filter(|c| {
            matches!(
                c,
                Cycle::FieldOp(_)
                    | Cycle::FieldMov(_)
                    | Cycle::FieldSLL64(_)
                    | Cycle::FieldSLL128(_)
                    | Cycle::FieldSLL192(_)
                    | Cycle::FieldAssertEq(_)
            )
        })
        .count();

    // External reference — vendored taceo-poseidon2 (single-pass; no advice).
    let mut ext_program = Program::new("bn254-fr-poseidon2-external-guest");
    let _ = ext_program
        .set_stack_size(1 << 20)
        .set_heap_size(1 << 20)
        .set_max_input_size(8192);
    let (_, ext_trace, _, ext_device) = ext_program.trace(&inputs, &[], &[]);
    let ext_real = count_real_cycles(&ext_trace);

    // Cross-check correctness: both guests must produce the SAME
    // permutation output (HorizenLabs RC3 parameters).
    let sdk_out: [[u64; 4]; 3] =
        postcard::from_bytes(&sdk_device.outputs).expect("decode SDK poseidon2 output");
    let ext_out: [[u64; 4]; 3] = postcard::from_bytes(&ext_device.outputs)
        .expect("decode external poseidon2 output");
    assert_eq!(
        sdk_out, ext_out,
        "SDK guest and taceo-poseidon2 must produce identical output \
         under HorizenLabs RC3 parameters"
    );

    let n_fr_ops_approx: f64 = 600.0;

    eprintln!("\n=== Poseidon2 BN254 t=3 cycle count ===");
    eprintln!("  Params: d=5, R_F=8, R_P=56 (HorizenLabs RC3)");
    eprintln!("  Approx Fr-op count per permutation: {n_fr_ops_approx}");
    eprintln!();
    eprintln!(
        "  bn254-fr SDK (v2 FieldOp coprocessor) : {sdk_real:>9} cycles  ({:.1} cyc/op, {sdk_field_ops} FR-cycle subset)",
        sdk_real as f64 / n_fr_ops_approx
    );
    eprintln!(
        "  taceo-poseidon2 (v0.2.1, pure RV)     : {ext_real:>9} cycles  ({:.1} cyc/op)",
        ext_real as f64 / n_fr_ops_approx
    );
    eprintln!();
    if sdk_real > 0 {
        eprintln!(
            "  Speedup vs taceo-poseidon2: {:.2}x",
            ext_real as f64 / sdk_real as f64
        );
    }
    eprintln!();

    // Sanity: SDK must be faster than the pure-software taceo reference.
    assert!(
        sdk_real < ext_real,
        "SDK should be faster than taceo-poseidon2 — got SDK={sdk_real} \
         vs taceo={ext_real}"
    );
}
