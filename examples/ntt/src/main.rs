use jolt_inlines_ntt as _;

fn main() {
    tracing_subscriber::fmt::init();
    const P: i64 = 998_244_353;
    let pow = |mut a: i64, mut n: usize| {
        let mut r = 1;
        while n != 0 {
            if n & 1 != 0 {
                r = r * a % P;
            }
            a = a * a % P;
            n >>= 1;
        }
        r
    };
    let root = pow(3, (P as usize - 1) / 128);
    let mont = (1i64 << 32) % P;
    let psi: [i32; 64] = core::array::from_fn(|i| (pow(root, i) * mont % P) as i32);
    let mut twiddles = [0i32; 64];
    for s in 0..6 {
        let len = 1 << s;
        for j in 0..len {
            twiddles[len - 1 + j] = (pow(root, 64 / len * j) * mont % P) as i32;
        }
    }
    let mut pinv = 1i32;
    for _ in 0..5 {
        pinv = pinv.wrapping_mul(2i32.wrapping_sub((P as i32).wrapping_mul(pinv)));
    }
    let input: [i32; 64] = core::array::from_fn(|i| ((i as i64 * 97 + 11) * mont % P) as i32);
    let expected = (0usize..64)
        .map(|i| {
            let point = pow(root, 2 * (i.reverse_bits() >> (usize::BITS - 6)) + 1);
            let value = input.iter().enumerate().fold(0, |sum, (j, a)| {
                (sum + i64::from(*a) * pow(point, j)).rem_euclid(P)
            });
            (i as u64 + 1) * value as u64
        })
        .sum::<u64>();
    let mut program = guest::compile_ntt("/tmp/jolt-guest-targets");
    let shared = guest::preprocess_shared_ntt(&mut program).unwrap();
    let prover = guest::preprocess_prover_ntt(shared.clone());
    let verifier =
        guest::preprocess_verifier_ntt(shared, prover.generators.to_verifier_setup(), None);
    let prove = guest::build_prover_ntt(program, prover);
    let verify = guest::build_verifier_ntt(verifier);
    let (output, proof, io) = prove(
        input.to_vec(),
        psi.to_vec(),
        twiddles.to_vec(),
        P as i32,
        pinv,
    );
    assert_eq!(output, expected);
    assert!(verify(
        input.to_vec(),
        psi.to_vec(),
        twiddles.to_vec(),
        P as i32,
        pinv,
        output,
        io.panic,
        proof
    ));
    println!("NTT proof verified; output matches direct DFT");
}
