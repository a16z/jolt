use ark_bn254::Fq;
use ark_ff::{AdditiveGroup, Field};
use rayon::vec;
use std::ops::Add;
use std::ops::Mul;

fn combine_limbs(l0: Fq, l1: Fq, l2: Fq) -> Fq {
    let two = Fq::from(2);
    l0 + (l1 * two.pow([125])) + (l2 * two.pow([250]))
}

fn evals(r: Vec<Fq>) -> Vec<Fq> {
    let ell = r.len();
    let pow_2 = 1 << ell;

    let mut evals: Vec<Fq> = vec![Fq::ONE; pow_2];
    let mut size = 1;
    for j in 0..ell {
        // in each iteration, we double the size of chis
        size *= 2;
        for i in (0..size).rev().step_by(2) {
            // copy each element from the prior iteration twice
            let scalar = evals[i / 2];
            evals[i] = scalar * r[j];
            evals[i - 1] = scalar - evals[i];
        }
    }
    evals
}

fn inner_product(a: Vec<Fq>, b: Vec<Fq>) -> Fq {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn verify_postponed_eval(input: Vec<Fq>, vec_to_eval_len: usize, l: usize) {
    let postponed_eval = &input[2..3 * l + 2];
    let vec_to_eval = &input[3 * l + 2..];

    let compressed_postponed_eval: Vec<Fq> = postponed_eval
        .chunks(3)
        .map(|chunk| combine_limbs(chunk[0], chunk[1], chunk[2]))
        .collect();
    let (pt, eval) = (
        compressed_postponed_eval[..compressed_postponed_eval.len() - 1].to_vec(),
        compressed_postponed_eval[compressed_postponed_eval.len() - 1],
    );

    // let (vec_to_eval1, vec_to_eval2) = (
    //     vec_to_eval[..vec_to_eval.len() - 60].to_vec(),
    //     vec_to_eval[vec_to_eval.len() - 60..].to_vec(),
    // );

    // let comms: Vec<Fq> = vec_to_eval2
    //     .chunks(3)
    //     .map(|chunk| combine_limbs(chunk[0], chunk[1], chunk[2]))
    //     .collect();

    // let mut pub_io = [
    //     [Fq::ONE].to_vec(),
    //     [Fq::ONE + Fq::ONE].to_vec(),
    //     vec_to_eval1,
    //     comms,
    // ]
    // .concat();

    let mut pub_io = [
        [Fq::ONE, Fq::ONE + Fq::ONE].to_vec(),
        vec_to_eval.to_vec(),
    ]
    .concat();

    let pad_length = pub_io.len().next_power_of_two();
    let log_pad_length = pad_length.ilog2() as usize;

    pub_io.resize(pad_length, Fq::ZERO);

    let required_pt = pt[pt.len() - log_pad_length..].to_vec();
    let evals = evals(required_pt);

    let mut computed_eval = inner_product(pub_io, evals);
    computed_eval *= pt[0..pt.len() - log_pad_length]
        .iter()
        .map(|r| Fq::ONE - r)
        .product::<Fq>();

    println!("difference = {}", computed_eval - eval);
    assert_eq!(eval, computed_eval);
}

#[cfg(test)]
mod tests {
    use std::{fs::File, io::Read};

    use ark_bn254::Fq;
    use num_bigint::BigUint;
    use serde_json::Value;

    use super::verify_postponed_eval;

    #[test]
    fn test_postponed_eval() {
        let witness_file_path = "src/parse/requirements/spartan_hyrax_witness.json";
        let mut witness_file = File::open(witness_file_path).expect("Failed to open witness.json");

        let mut witness_contents = String::new();
        witness_file
            .read_to_string(&mut witness_contents)
            .expect("Failed to read witness.json");

        let witness_json: Value =
            serde_json::from_str(&witness_contents).expect("Failed to parse witness.json");

        if let Some(witness_array) = witness_json.as_array() {
            let result: Vec<Fq> = witness_array
                .iter()
                .take(1791)
                .map(|entry| entry.as_str().map(|s| s.to_string()).unwrap())
                .map(|entry| Fq::from(BigUint::parse_bytes(entry.as_bytes(), 10).unwrap()))
                .collect();

            println!("The witness array is: {}", witness_array[1793].to_string());
            verify_postponed_eval(result, 0, 24);
        } else {
            eprintln!("The JSON is not an array or 'witness' field is missing");
        }
    }
}
