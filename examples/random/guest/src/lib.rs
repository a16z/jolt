#![cfg_attr(feature = "guest", no_std)]

fn rand_v08(a: u32, b: u32) -> u32 {
    use rand_v08::Rng;
    use rand_v08::SeedableRng;
    // std_rng feature is needed for StdRng
    // getrandom feature is needed for from_entropy
    let mut rng = rand_v08::rngs::StdRng::from_entropy();
    rng.gen_range(a..b)
}

fn rand_v09(a: u32, b: u32) -> u32 {
    use rand_v09::Rng;
    use rand_v09::SeedableRng;
    // std_rng feature is needed for StdRng
    // os_rng feature is needed for from_os_rng
    let mut rng = rand_v09::rngs::StdRng::from_os_rng();
    rng.random_range(a..b)
}

#[jolt::provable(memory_size = 10240, max_trace_length = 65536)]
fn rand(a: u32, b: u32) -> u32 {
    use jolt::{end_cycle_tracking, start_cycle_tracking};

    start_cycle_tracking("rand");
    let result = rand_v08(a, b) + rand_v09(a, b);
    end_cycle_tracking("rand");
    result
}
