//! Generate Poseidon test vectors and constants for Go implementation
//!
//! Run with: cargo run --release --bin poseidon_vectors
//! For constants: cargo run --release --bin poseidon_vectors -- --extract-constants

use ark_bn254::Fr;
use ark_ff::PrimeField;
use light_poseidon::{Poseidon, PoseidonHasher};

fn print_fr(label: &str, val: Fr) {
    let bigint = val.into_bigint();
    println!("{label} = {bigint}");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() > 1 && args[1] == "--extract-constants" {
        extract_constants();
        return;
    }

    if args.len() > 1 && args[1] == "--debug" {
        debug_poseidon();
        return;
    }

    println!("=== Rust Poseidon Test Vectors (light-poseidon circom) ===\n");

    // Width-3 tests
    println!("--- Width-3 ---");
    let mut hasher3 = Poseidon::<Fr>::new_circom(3).unwrap();
    let inputs = [Fr::from(0u64), Fr::from(0u64), Fr::from(0u64)];
    let result = hasher3.hash(&inputs).unwrap();
    print_fr("W3 hash([0, 0, 0])", result);

    let mut hasher3b = Poseidon::<Fr>::new_circom(3).unwrap();
    let inputs2 = [Fr::from(1u64), Fr::from(2u64), Fr::from(3u64)];
    let result2 = hasher3b.hash(&inputs2).unwrap();
    print_fr("W3 hash([1, 2, 3])", result2);

    // Width-4 tests: new_circom(n) takes n INPUTS, not width
    // According to light-poseidon docs, new_circom(n) creates a hasher that takes n inputs
    // The internal width is n+1 (domain_tag prepended)
    println!("\n--- new_circom(3) = 3 inputs, width 4 ---");
    let mut hasher_c3 = Poseidon::<Fr>::new_circom(3).unwrap();
    let inputs_c3 = [Fr::from(0u64), Fr::from(0u64), Fr::from(0u64)];
    let result_c3 = hasher_c3.hash(&inputs_c3).unwrap();
    print_fr("circom(3) hash([0, 0, 0])", result_c3);

    println!("\n--- new_circom(4) = 4 inputs, width 5 ---");
    let mut hasher_c4 = Poseidon::<Fr>::new_circom(4).unwrap();
    let inputs_c4 = [Fr::from(0u64), Fr::from(0u64), Fr::from(0u64), Fr::from(0u64)];
    let result_c4 = hasher_c4.hash(&inputs_c4).unwrap();
    print_fr("circom(4) hash([0, 0, 0, 0])", result_c4);

    let mut hasher4b = Poseidon::<Fr>::new_circom(4).unwrap();
    let inputs4b = [Fr::from(1u64), Fr::from(2u64), Fr::from(3u64), Fr::from(4u64)];
    let result4b = hasher4b.hash(&inputs4b).unwrap();
    print_fr("W4 hash([1, 2, 3, 4])", result4b);

    // Width-5 tests
    println!("\n--- Width-5 ---");
    let mut hasher5 = Poseidon::<Fr>::new_circom(5).unwrap();
    let inputs5 = [Fr::from(0u64), Fr::from(0u64), Fr::from(0u64), Fr::from(0u64), Fr::from(0u64)];
    let result5 = hasher5.hash(&inputs5).unwrap();
    print_fr("W5 hash([0, 0, 0, 0, 0])", result5);

    let mut hasher5b = Poseidon::<Fr>::new_circom(5).unwrap();
    let inputs5b = [Fr::from(1u64), Fr::from(2u64), Fr::from(3u64), Fr::from(4u64), Fr::from(5u64)];
    let result5b = hasher5b.hash(&inputs5b).unwrap();
    print_fr("W5 hash([1, 2, 3, 4, 5])", result5b);

    // Go Hash style comparison
    println!("\n--- Go Hash style (width-4, [0, in1, in2, in3]) ---");
    let mut hasher_go_style = Poseidon::<Fr>::new_circom(4).unwrap();
    let go_style_inputs = [Fr::from(0u64), Fr::from(1u64), Fr::from(2u64), Fr::from(3u64)];
    let go_style_result = hasher_go_style.hash(&go_style_inputs).unwrap();
    print_fr("W4 hash([0, 1, 2, 3]) (Go Hash style)", go_style_result);

    println!("\nRun with --extract-constants to dump constants for Go");

    // Test with "Jolt" label - this is the initial transcript state
    println!("\n--- Transcript initialization test ---");
    let mut label_padded = [0u8; 32];
    label_padded[..4].copy_from_slice(b"Jolt");
    let label_f = Fr::from_le_bytes_mod_order(&label_padded);
    println!("'Jolt' as Fr (LE) = {}", label_f.into_bigint());
    println!("Expected: 1953263434 (0x746C6F4A)");

    // Hash [label, 0, 0, 0] with new_circom(4) - this is how transcript.new() initializes
    let mut hasher_init = Poseidon::<Fr>::new_circom(4).unwrap();
    let init_inputs = [label_f, Fr::from(0u64), Fr::from(0u64), Fr::from(0u64)];
    let init_result = hasher_init.hash(&init_inputs).unwrap();
    println!("Initial state = hash([Jolt, 0, 0, 0]) = {}", init_result.into_bigint());
}

/// Extract Poseidon constants from light-poseidon for Go implementation
fn extract_constants() {
    use ark_bn254::Fr;
    use light_poseidon::parameters::bn254_x5::get_poseidon_parameters;

    let params = get_poseidon_parameters::<Fr>(4).unwrap();

    println!("package poseidon");
    println!();
    println!("import \"math/big\"");
    println!();
    println!("// Poseidon constants extracted from light-poseidon (circom-compatible)");
    println!("// For UNOPTIMIZED algorithm: full MDS in all rounds");
    println!("// Width: 4, Full rounds: {}, Partial rounds: {}", params.full_rounds, params.partial_rounds);
    println!();

    // ARK constants
    println!("var arkConstants []*big.Int");
    println!();
    println!("// MDS matrix (4x4)");
    println!("var mdsMatrix [][]*big.Int");
    println!();
    println!("func init() {{");
    println!("\tarkConstants = make([]*big.Int, {})", params.ark.len());
    println!();

    for (i, c) in params.ark.iter().enumerate() {
        println!("\tarkConstants[{}], _ = new(big.Int).SetString(\"{}\", 10)", i, c.into_bigint());
    }

    println!();
    println!("\tmdsMatrix = make([][]*big.Int, 4)");
    println!("\tfor i := 0; i < 4; i++ {{");
    println!("\t\tmdsMatrix[i] = make([]*big.Int, 4)");
    println!("\t}}");
    println!();

    for i in 0..4 {
        for j in 0..4 {
            println!("\tmdsMatrix[{}][{}], _ = new(big.Int).SetString(\"{}\", 10)", i, j, params.mds[i][j].into_bigint());
        }
    }

    println!("}}");
}

/// Debug Poseidon step by step
fn debug_poseidon() {
    use ark_bn254::Fr;
    use ark_ff::Field;
    use light_poseidon::parameters::bn254_x5::get_poseidon_parameters;

    let params = get_poseidon_parameters::<Fr>(4).unwrap();

    println!("=== Rust Width-4 Poseidon Debug ===\n");
    println!("Width: {}", params.width);
    println!("Full rounds: {}", params.full_rounds);
    println!("Partial rounds: {}", params.partial_rounds);
    println!("Alpha: {}", params.alpha);
    println!("ARK length: {}", params.ark.len());
    println!("MDS shape: {}x{}", params.mds.len(), params.mds[0].len());

    println!("\nFirst 4 ARK constants:");
    for i in 0..4 {
        println!("  ark[{}] = {}", i, params.ark[i].into_bigint());
    }

    println!("\nMDS[0][0] = {}", params.mds[0][0].into_bigint());

    // Run full Poseidon manually and compare
    let mut state = vec![Fr::from(0u64), Fr::from(0u64), Fr::from(0u64), Fr::from(0u64)];
    let half_rounds = params.full_rounds / 2;
    let all_rounds = params.full_rounds + params.partial_rounds;

    println!("\nRunning full Poseidon manually:");

    // First half full rounds
    for round in 0..half_rounds {
        // ARK
        for (i, s) in state.iter_mut().enumerate() {
            *s += params.ark[round * 4 + i];
        }
        // S-box full
        for s in &mut state {
            *s = s.pow([5]);
        }
        // MDS
        let old_state = state.clone();
        for (i, s) in state.iter_mut().enumerate() {
            *s = Fr::from(0u64);
            for (j, &old_s) in old_state.iter().enumerate() {
                *s += old_s * params.mds[i][j];
            }
        }
        if round < 2 {
            println!("After full round {round}: {} ...", &state[0].into_bigint().to_string()[..20]);
        }
    }
    println!("After first half ({} full rounds): {} ...", half_rounds, &state[0].into_bigint().to_string()[..20]);

    // Partial rounds
    for round in half_rounds..(half_rounds + params.partial_rounds) {
        // ARK
        for (i, s) in state.iter_mut().enumerate() {
            *s += params.ark[round * 4 + i];
        }
        // S-box partial (only first element)
        state[0] = state[0].pow([5]);
        // MDS
        let old_state = state.clone();
        for (i, s) in state.iter_mut().enumerate() {
            *s = Fr::from(0u64);
            for (j, &old_s) in old_state.iter().enumerate() {
                *s += old_s * params.mds[i][j];
            }
        }
    }
    println!("After partial rounds: {} ...", &state[0].into_bigint().to_string()[..20]);

    // Second half full rounds
    for round in (half_rounds + params.partial_rounds)..all_rounds {
        // ARK
        for (i, s) in state.iter_mut().enumerate() {
            *s += params.ark[round * 4 + i];
        }
        // S-box full
        for s in &mut state {
            *s = s.pow([5]);
        }
        // MDS
        let old_state = state.clone();
        for (i, s) in state.iter_mut().enumerate() {
            *s = Fr::from(0u64);
            for (j, &old_s) in old_state.iter().enumerate() {
                *s += old_s * params.mds[i][j];
            }
        }
    }
    println!("Final result: {}", state[0].into_bigint());

    // Verify against library
    let mut hasher4 = Poseidon::<Fr>::new_circom(4).unwrap();
    let inputs4 = [Fr::from(0u64), Fr::from(0u64), Fr::from(0u64), Fr::from(0u64)];
    let result4 = hasher4.hash(&inputs4).unwrap();
    println!("Library result: {}", result4.into_bigint());
}
