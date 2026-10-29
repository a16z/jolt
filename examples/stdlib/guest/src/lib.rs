use rayon::iter::{IntoParallelIterator, ParallelIterator};

#[jolt::provable(max_trace_length = 65536, stack_size = 1048576)]
fn int_to_string(n: i32) -> i32 {
    println!("Hello, from int_to_string! n = {n}");
    n
}

#[jolt::provable(max_trace_length = 131072, stack_size = 1048576)]
fn string_concat(n: i32) -> String {
    println!("Hello, world!");
    let mut res = String::new();
    for i in 0..n {
        res += &i.to_string();
    }

    res
}

#[jolt::provable(max_trace_length = 1048576, stack_size = 1048576)]
fn parallel_sum_of_squares(n: u32) -> u64 {
    println!("Computing parallel sum of squares from 1 to {n}");

    let pool = rayon::ThreadPoolBuilder::new()
        .build()
        .expect("Failed to create thread pool");

    let result = pool.install(|| {
        (1..=n)
            .into_par_iter()
            .map(|x| (x as u64) * (x as u64))
            .sum()
    });

    println!("Result: {result}");
    result
}

#[jolt::provable(max_trace_length = 262144, stack_size = 1048576)]
fn memory_copy(n: u32) -> Vec<u8> {
    assert!(n <= 2048);
    let n = n as usize;
    let mut data: Vec<u8> = (0..n + 8).map(|i| (i * 197 + i / 7) as u8).collect();
    data.copy_within(..n, 1);
    data.copy_within(2..n + 2, 0);
    let mut result = vec![0; n + 8];
    result[1..n + 1].copy_from_slice(&data[..n]);
    result
}
