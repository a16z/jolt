use ark_bn254::Fr;
use ark_std::test_rng;
use criterion::Criterion;
use jolt_core::{
    field::JoltField,
    poly::multilinear_polynomial::MultilinearPolynomial,
    subprotocols::{
        large_degree_sumcheck::{
            compute_initial_eval_claim, AppendixCSumCheckProof, LargeDMulSumCheckProof,
            NaiveSumCheckProof,
        },
        toom::FieldMulSmall,
    },
    transcripts::{KeccakTranscript, Transcript},
    utils::{math::Math, thread::unsafe_allocate_zero_vec},
};
use rand_core::RngCore;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

fn test_func_data<F: JoltField>(d: usize, t: usize) -> Vec<MultilinearPolynomial<F>> {
    let mut rng = test_rng();
    let mut val_vec: Vec<Vec<F>> = vec![unsafe_allocate_zero_vec(t); d];

    for j in 0..t {
        for i in 0..d {
            val_vec[i][j] = F::from_u32(rng.next_u32());
        }
    }

    let val_mle = val_vec
        .into_par_iter()
        .map(|val| MultilinearPolynomial::from(val))
        .collect::<Vec<_>>();

    val_mle
}

fn benchmark_appendix_c_sumcheck<const D1: usize>(c: &mut Criterion, d: usize, t: usize) {
    let ra = test_func_data(d, t);

    let mut transcript = KeccakTranscript::new(b"test_transcript");
    let r_cycle: Vec<Fr> = transcript.challenge_vector(t.log_2());
    let previous_claim =
        compute_initial_eval_claim(&ra.iter().map(|x| &*x).collect::<Vec<_>>(), &r_cycle);

    c.bench_function(
        &format!("appendix_c_sumcheck_{}_{}", ra.len(), r_cycle.len().pow2()),
        |b| {
            b.iter_with_setup(
                || (ra.clone(), transcript.clone(), previous_claim.clone()),
                |(mut ra, mut transcript, mut previous_claim)| {
                    criterion::black_box(
                        AppendixCSumCheckProof::<Fr, KeccakTranscript>::prove::<D1>(
                            &mut ra.iter_mut().collect::<Vec<_>>(),
                            &r_cycle,
                            &mut previous_claim,
                            &mut transcript,
                        ),
                    );
                },
            );
        },
    );
}

fn benchmark_karatsuba_sumcheck<F: FieldMulSmall>(c: &mut Criterion, d: usize, t: usize) {
    let ra = test_func_data(d, t);

    let mut transcript = KeccakTranscript::new(b"test_transcript");
    let r_cycle: Vec<F> = transcript.challenge_vector(t.log_2());
    let previous_claim =
        compute_initial_eval_claim(&ra.iter().map(|x| &*x).collect::<Vec<_>>(), &r_cycle);

    c.bench_function(
        &format!("karatsuba_sumcheck_{}_{}", ra.len(), r_cycle.len().pow2()),
        |b| {
            b.iter_with_setup(
                || (ra.clone(), transcript.clone(), previous_claim.clone()),
                |(mut ra, mut transcript, mut previous_claim)| {
                    criterion::black_box(LargeDMulSumCheckProof::prove(
                        &mut ra,
                        &r_cycle,
                        &mut previous_claim,
                        &mut transcript,
                    ));
                },
            );
        },
    );
}

fn benchmark_naive_sumcheck<F: JoltField>(c: &mut Criterion, d: usize, t: usize) {
    let ra = test_func_data(d, t);

    let mut transcript = KeccakTranscript::new(b"test_transcript");
    let r_cycle: Vec<Fr> = transcript.challenge_vector(t.log_2());
    let previous_claim =
        compute_initial_eval_claim(&ra.iter().map(|x| &*x).collect::<Vec<_>>(), &r_cycle);

    c.bench_function(
        &format!("naive_sumcheck_{}_{}", ra.len(), r_cycle.len().pow2()),
        |b| {
            b.iter_with_setup(
                || (ra.clone(), transcript.clone(), previous_claim.clone()),
                |(mut ra, mut transcript, mut previous_claim)| {
                    criterion::black_box(NaiveSumCheckProof::prove(
                        &mut ra.iter_mut().collect::<Vec<_>>(),
                        &r_cycle,
                        &mut previous_claim,
                        &mut transcript,
                    ));
                },
            );
        },
    );
}

fn main() {
    let mut criterion = Criterion::default()
        .configure_from_args()
        .warm_up_time(std::time::Duration::from_secs(10));

    let t = 1 << 20;

    benchmark_karatsuba_sumcheck::<Fr>(&mut criterion, 16, t);
    benchmark_naive_sumcheck::<Fr>(&mut criterion, 16, t);

    criterion.final_summary();
}
