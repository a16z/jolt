use super::GuestConfig;

/// Fibonacci guest: computes fib(n).
pub struct Fibonacci(pub u32);

impl Default for Fibonacci {
    fn default() -> Self {
        Self(100)
    }
}

impl GuestConfig for Fibonacci {
    fn package(&self) -> &str {
        "fibonacci-guest"
    }
    fn input(&self) -> Vec<u8> {
        postcard::to_stdvec(&self.0).unwrap()
    }
    fn bench_name(&self) -> String {
        format!("prover_time_fibonacci_{}", self.0)
    }
}
