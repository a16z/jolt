//! Small native proof fixtures and incomplete-checkpoint controls for Solidity.
use ark_bn254::{Fq, Fq2, G2Affine};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use jolt_crypto::{Bn254, JoltGroup};
use jolt_field::{CanonicalBytes, CanonicalEncoding, Fr, One, Ring, Zero};
use jolt_hyperkzg::{
    HyperKZGProverSetup, HyperKZGScheme, HyperKZGSetupParams, HyperKZGVerifierSetup,
};
use jolt_openings::CommitmentScheme;
use jolt_poly::CompressedPoly;
use jolt_r1cs::ConstraintMatrices;
use jolt_spartan_prover::preprocessed::PreprocessedMatrices;
use jolt_spartan_verifier::preprocessed::sparse::{Network, ProductReduction, SparseQuery};
use jolt_spartan_verifier::{
    preprocessed::{ComputationKey, MatrixApplicationIds, PreprocessedProof, PublicColumnQuery},
    SpartanKey,
};
use jolt_sumcheck::{BooleanHypercube, SumcheckClaim, SUMCHECK_ROUND_TRANSCRIPT_LABEL};
use jolt_transcript::{Bn254WideBlake2bTranscript, Transcript};
use std::{error::Error, fs::DirEntry, path::PathBuf};
fn main() -> Result<(), Box<dyn Error>> {
    if std::env::args().nth(1).as_deref() == Some("--check-corpus") {
        return check_corpus(PathBuf::from(
            std::env::args_os().nth(2).ok_or("expected corpus path")?,
        ));
    }
    let output = PathBuf::from(
        std::env::args_os()
            .nth(1)
            .ok_or("expected fresh fixture directory")?,
    );
    std::fs::create_dir(&output)?;
    let (pk, vk) = fixture_setup()?;
    for name in ["toy", "empty", "empty-public", "zero-products"] {
        let empty = name != "toy";
        let o = Fr::one();
        let matrices = if empty {
            ConstraintMatrices::new(4, 8, vec![vec![]; 4], vec![vec![]; 4], vec![vec![]; 4])
        } else {
            ConstraintMatrices::new(
                4,
                8,
                vec![
                    vec![(3, o), (0, o), (3, o + o), (3, -(o + o)), (2, Fr::zero())],
                    vec![(5, o)],
                    vec![(2, o + o)],
                    vec![(7, o)],
                ],
                vec![vec![(4, o)], vec![(0, o)], vec![(6, o)], vec![(1, o)]],
                vec![vec![]; 4],
            )
        };
        let direct = SpartanKey::new(matrices, 2, [19; 32])?;
        let tables = PreprocessedMatrices::new(
            &direct,
            MatrixApplicationIds {
                circuit: [1; 32],
                profile: [2; 32],
                public_schema: [3; 32],
                table: [4; 32],
            },
            &pk,
        )?;
        let inputs = if name == "empty-public" {
            [23, 5].map(Fr::from_u64)
        } else if empty {
            [Fr::zero(); 2]
        } else {
            [5, 0].map(Fr::from_u64)
        };
        let private = if empty {
            [Fr::zero(); 5]
        } else {
            [3, 0, 0, 11, 0].map(Fr::from_u64)
        };
        let mut proof = tables.prove(&inputs, &private, &pk)?;
        let key = tables.key();
        key.verify(&key.id(), &inputs, &proof, &vk)?;
        if empty {
            let mut bad_pairing = proof.clone();
            bad_pairing.public.opening.w[0] = Bn254::g1_generator();
            if key.verify(&key.id(), &inputs, &bad_pairing, &vk).is_ok() {
                return Err("invalid public pairing was accepted by native verifier".into());
            }
        }
        if name == "zero-products" {
            proof.private_values = [Fr::zero(); 3];
            proof.sparse.roots = [Fr::zero(); 16];
            proof.sparse.dot_halves = [Fr::zero(); 6];
            for network in [&mut proof.sparse.operations, &mut proof.sparse.memory] {
                for layer in &mut network.layers {
                    for round in &mut layer.sumcheck.round_polynomials {
                        *round = CompressedPoly::new(vec![Fr::zero()]);
                    }
                    layer.ends.fill([Fr::zero(); 2]);
                    layer.dot_ends.fill([Fr::zero(); 3]);
                }
            }
            if key.verify(&key.id(), &inputs, &proof, &vk).is_ok() {
                return Err("zero-product partial fixture unexpectedly verified completely".into());
            }
        }
        let wire = key.encode_proof(&proof)?;
        if !empty && wire != include_bytes!("../tests/fixtures/preprocessed-wire-v1-toy.bin") {
            return Err("toy differs from accepted wire golden".into());
        }
        let mut transcript = Bn254WideBlake2bTranscript::new(b"spartan-preprocessed-clear-v2");
        let tau = key.begin(
            &key.id(),
            &vk,
            &inputs,
            &proof.witness_commitment,
            &mut transcript,
        )?;
        let path = output.join(name);
        std::fs::create_dir(&path)?;
        std::fs::write(path.join("key.bin"), key.canonical_bytes())?;
        std::fs::write(path.join("setup.bin"), vk.binding()?.canonical_bytes)?;
        let mut g2_affine = Vec::with_capacity(256);
        for bytes in vk.binding()?.canonical_bytes[96..].chunks_exact(64) {
            let point = G2Affine::deserialize_compressed(bytes)?;
            g2_affine.extend(g2_evm_bytes(point)?);
        }
        std::fs::write(path.join("g2-affine.bin"), g2_affine)?;
        if name == "empty" {
            let outsider = (0u64..100)
                .filter_map(|x| {
                    G2Affine::get_point_from_x_unchecked(
                        Fq2::new(Fq::from(x), Fq::from(0u64)),
                        false,
                    )
                })
                .find(|point| !point.is_in_correct_subgroup_assuming_on_curve())
                .ok_or("no deterministic non-subgroup fixture")?;
            let mut compressed = Vec::new();
            outsider.serialize_compressed(&mut compressed)?;
            if G2Affine::deserialize_compressed(compressed.as_slice()).is_ok() {
                return Err("native accepted non-subgroup fixture".into());
            }
            std::fs::write(path.join("non-subgroup-compressed.bin"), compressed)?;
            std::fs::write(
                path.join("non-subgroup-affine.bin"),
                g2_evm_bytes(outsider)?,
            )?;
        }
        std::fs::write(
            path.join("inputs.bin"),
            inputs
                .iter()
                .flat_map(CanonicalBytes::to_bytes_le_vec)
                .collect::<Vec<_>>(),
        )?;
        std::fs::write(path.join("proof.bin"), wire)?;
        let hex = |bytes: &[u8]| {
            bytes
                .iter()
                .flat_map(|x| {
                    let digits = b"0123456789abcdef";
                    [
                        char::from(digits[usize::from(x >> 4)]),
                        char::from(digits[usize::from(x & 15)]),
                    ]
                })
                .collect::<String>()
        };
        let tau = tau
            .iter()
            .map(|x| format!("\"{}\"", hex(&x.to_bytes_le_vec())))
            .collect::<Vec<_>>()
            .join(",");
        std::fs::write(
            path.join("prefix.json"),
            format!(
                "{{\"key_id\":\"{}\",\"state\":\"{}\",\"tau_le\":[{}]}}\n",
                hex(&key.id()),
                hex(&transcript.state()),
                tau
            ),
        )?;
        let outer = proof.outer.verify(
            &SumcheckClaim {
                num_vars: 2,
                degree: 3,
                claimed_sum: Fr::zero(),
            },
            BooleanHypercube,
            SUMCHECK_ROUND_TRANSCRIPT_LABEL,
            &mut transcript,
        )?;
        let weights = ComputationKey::outer_weights(&proof.outer_evaluations, &mut transcript);
        let mut query_transcript =
            Bn254WideBlake2bTranscript::new(b"spartan-preprocessed-clear-v2");
        let _ = key.begin(
            &key.id(),
            &vk,
            &inputs,
            &proof.witness_commitment,
            &mut query_transcript,
        )?;
        let _ = proof.outer.verify(
            &SumcheckClaim {
                num_vars: 2,
                degree: 3,
                claimed_sum: Fr::zero(),
            },
            BooleanHypercube,
            SUMCHECK_ROUND_TRANSCRIPT_LABEL,
            &mut query_transcript,
        )?;
        let _ = ComputationKey::outer_weights(&proof.outer_evaluations, &mut query_transcript);
        let (public_point, public_value) = key.public_opening_query(
            outer.point.as_slice(),
            &proof.public.evaluations,
            &mut query_transcript,
        )?;
        let public = key.verify_public(
            &key.id(),
            &vk,
            PublicColumnQuery {
                inputs: &inputs,
                point: outer.point.as_slice(),
                matrix_weights: weights,
            },
            &proof.public,
            &mut transcript,
        )?;
        let public_state = transcript.state();
        let claim =
            ComputationKey::inner_claim(proof.outer_evaluations, weights, public, &mut transcript);
        let inner = proof.inner.verify(
            &SumcheckClaim {
                num_vars: 3,
                degree: 2,
                claimed_sum: claim,
            },
            BooleanHypercube,
            SUMCHECK_ROUND_TRANSCRIPT_LABEL,
            &mut transcript,
        )?;
        let checkpoints = outer
            .point
            .as_slice()
            .iter()
            .copied()
            .chain([outer.value])
            .chain(weights)
            .chain(public_point)
            .chain([public_value, public, claim])
            .chain(inner.point.as_slice().iter().copied())
            .chain([inner.value])
            .map(|x| format!("\"{}\"", hex(&x.to_bytes_le_vec())))
            .collect::<Vec<_>>()
            .join(",");
        std::fs::write(
            path.join("algebra.json"),
            format!(
                "{{\"values_le\":[{}],\"post_public_state\":\"{}\",\"post_inner_state\":\"{}\"}}\n",
                checkpoints,
                hex(&public_state),
                hex(&transcript.state()),
            ),
        )?;
        let query = SparseQuery {
            rows: outer.point.as_slice(),
            columns: inner.point.as_slice(),
            values: proof.private_values,
        };
        let (alpha, beta) = key.begin_sparse(
            &query,
            &proof.sparse.dereference_commitment,
            &mut transcript,
        )?;
        let (ops, mem) = key.bind_roots(
            proof.private_values,
            &proof.sparse.roots,
            &proof.sparse.dot_halves,
            &mut transcript,
        )?;
        let operations = ProductReduction::new(
            Network::Operations,
            key.shape().operations.trailing_zeros() as usize,
            ops,
            Some(proof.sparse.dot_halves),
            &mut transcript,
        )?
        .verify(&proof.sparse.operations, &mut transcript)?;
        let memory = ProductReduction::new(
            Network::Memory,
            key.shape().memory.trailing_zeros() as usize,
            mem,
            None,
            &mut transcript,
        )?
        .verify(&proof.sparse.memory, &mut transcript)?;
        let values = [alpha, beta]
            .into_iter()
            .chain(operations.point)
            .chain(operations.tree_evaluations)
            .chain(
                operations
                    .dot_evaluations
                    .ok_or("missing native dot evaluations")?
                    .into_iter()
                    .flatten(),
            )
            .chain(memory.point)
            .chain(memory.tree_evaluations)
            .map(|x| format!("\"{}\"", hex(&x.to_bytes_le_vec())))
            .collect::<Vec<_>>()
            .join(",");
        std::fs::write(
            path.join("sparse.json"),
            format!(
                "{{\"values_le\":[{}],\"state\":\"{}\",\"full_native_acceptance\":{}}}\n",
                values,
                hex(&transcript.state()),
                name != "zero-products",
            ),
        )?;
        if name != "zero-products" {
            let mut bad_value = proof.clone();
            bad_value.witness_evaluation += Fr::one();
            let mut bad_opening = proof.clone();
            bad_opening.witness_opening.w[0] = Bn254::g1_generator();
            if key.verify(&key.id(), &inputs, &bad_value, &vk).is_ok()
                || key.verify(&key.id(), &inputs, &bad_opening, &vk).is_ok()
            {
                return Err("native accepted pending witness mutation".into());
            }
        }
        let leaves = leaf_checkpoint(key, &vk, &inputs, &proof);
        std::fs::write(
            path.join("complete.json"),
            match &leaves {
                Ok((_, state)) => format!("{{\"accepted\":true,\"state\":\"{}\"}}\n", hex(state)),
                Err(_) => "{\"accepted\":false}\n".to_owned(),
            },
        )?;
        if leaves.is_ok() != (name != "zero-products") {
            return Err("unexpected native sparse leaf acceptance".into());
        }
        std::fs::write(
            path.join("leaves.json"),
            match leaves {
                Ok((state, _)) => format!(
                    "{{\"accepted\":true,\"state\":\"{}\",\"witness_mutations_rejected\":true}}\n",
                    hex(&state)
                ),
                Err(_) => "{\"accepted\":false}\n".to_owned(),
            },
        )?;
    }
    Ok(())
}

// Uses the full native sparse verifier, including all three actual PCS checks and leaves.
fn leaf_checkpoint(
    key: &ComputationKey,
    setup: &HyperKZGVerifierSetup,
    inputs: &[Fr],
    proof: &PreprocessedProof,
) -> Result<([u8; 32], [u8; 32]), Box<dyn Error>> {
    let mut transcript = Bn254WideBlake2bTranscript::new(b"spartan-preprocessed-clear-v2");
    let _ = key.begin(
        &key.id(),
        setup,
        inputs,
        &proof.witness_commitment,
        &mut transcript,
    )?;
    let outer = proof.outer.verify(
        &SumcheckClaim {
            num_vars: key.shape().padded_rows.trailing_zeros() as usize,
            degree: 3,
            claimed_sum: Fr::zero(),
        },
        BooleanHypercube,
        SUMCHECK_ROUND_TRANSCRIPT_LABEL,
        &mut transcript,
    )?;
    let weights = ComputationKey::outer_weights(&proof.outer_evaluations, &mut transcript);
    let public = key.verify_public(
        &key.id(),
        setup,
        PublicColumnQuery {
            inputs,
            point: outer.point.as_slice(),
            matrix_weights: weights,
        },
        &proof.public,
        &mut transcript,
    )?;
    let claim =
        ComputationKey::inner_claim(proof.outer_evaluations, weights, public, &mut transcript);
    let inner = proof.inner.verify(
        &SumcheckClaim {
            num_vars: key.shape().padded_private.trailing_zeros() as usize,
            degree: 2,
            claimed_sum: claim,
        },
        BooleanHypercube,
        SUMCHECK_ROUND_TRANSCRIPT_LABEL,
        &mut transcript,
    )?;
    key.verify_sparse(
        SparseQuery {
            rows: outer.point.as_slice(),
            columns: inner.point.as_slice(),
            values: proof.private_values,
        },
        &proof.sparse,
        setup,
        &mut transcript,
    )?;
    let sparse_state = transcript.state();
    // Full native acceptance above already pins this product; repeat its owner transcript/opening here for the final checkpoint.
    SpartanKey::append_witness_evaluation(proof.witness_evaluation, &mut transcript);
    HyperKZGScheme::verify(
        &proof.witness_commitment,
        inner.point.as_slice(),
        proof.witness_evaluation,
        &proof.witness_opening,
        setup,
        &mut transcript,
    )?;
    Ok((sparse_state, transcript.state()))
}

fn fixture_setup() -> Result<(HyperKZGProverSetup, HyperKZGVerifierSetup), Box<dyn Error>> {
    let beta = Fr::from_u64(7);
    Ok(HyperKZGScheme::setup(HyperKZGSetupParams {
        g1_powers: std::iter::successors(Some(Fr::one()), |x| Some(*x * beta))
            .take(64)
            .map(|x| Bn254::g1_generator().scalar_mul(&x))
            .collect(),
        setup_id: [9; 32],
        max_public_degree: 63,
        g2: Bn254::g2_generator(),
        beta_g2: Bn254::g2_generator().scalar_mul(&beta),
    })?)
}

#[expect(
    clippy::print_stdout,
    reason = "intentional native/EVM boundary differential artifact"
)]
fn check_corpus(directory: PathBuf) -> Result<(), Box<dyn Error>> {
    let (_, setup) = fixture_setup()?;
    let setup_bytes = setup.binding()?.canonical_bytes;
    let mut entries = std::fs::read_dir(directory)?.collect::<Result<Vec<_>, _>>()?;
    entries.sort_by_key(DirEntry::file_name);
    let mut records = Vec::new();
    for entry in entries {
        let path = entry.path();
        let name = entry
            .file_name()
            .into_string()
            .map_err(|_| "nonutf8 case name")?;
        if !name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-') {
            return Err("invalid case name".into());
        }
        let decoded = (|| -> Result<(), Box<dyn Error>> {
            let key_bytes = std::fs::read(path.join("key.bin"))?;
            let key_id: [u8; 32] = std::fs::read(path.join("key-id.bin"))?
                .try_into()
                .map_err(|_| "key identity length")?;
            let supplied_setup = std::fs::read(path.join("setup.bin"))?;
            let setup_id: [u8; 32] = std::fs::read(path.join("setup-id.bin"))?
                .try_into()
                .map_err(|_| "setup identity length")?;
            if supplied_setup != setup_bytes || ComputationKey::digest(&supplied_setup) != setup_id
            {
                return Err("setup policy mismatch".into());
            }
            let key = ComputationKey::decode_wire(&key_bytes, &key_id, &setup)?;
            let inputs = std::fs::read(path.join("inputs.bin"))?;
            if inputs.len() != 32 * (key.shape().public_columns - 1) {
                return Err("input length".into());
            }
            for bytes in inputs.chunks_exact(32) {
                let _ = Fr::from_bytes_le_checked(bytes).ok_or("noncanonical input")?;
            }
            let _ = key.decode_proof(&std::fs::read(path.join("proof.bin"))?)?;
            Ok(())
        })()
        .is_ok();
        let expected = std::fs::read_to_string(path.join("expected.txt"))? == "decode";
        if decoded != expected {
            return Err(format!("native differential mismatch: {name}").into());
        }
        records.push(format!("{{\"name\":\"{name}\",\"decoded\":{decoded}}}"));
    }
    println!("[{}]", records.join(","));
    Ok(())
}

fn g2_evm_bytes(point: G2Affine) -> Result<Vec<u8>, Box<dyn Error>> {
    let mut output = Vec::with_capacity(128);
    for coordinate in [point.x.c1, point.x.c0, point.y.c1, point.y.c0] {
        let mut bytes = Vec::with_capacity(32);
        coordinate.serialize_compressed(&mut bytes)?;
        bytes.reverse();
        output.extend(bytes);
    }
    Ok(output)
}
