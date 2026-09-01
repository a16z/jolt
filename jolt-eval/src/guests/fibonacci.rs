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
    fn label(&self) -> String {
        format!("fibonacci_{}", self.0)
    }
    fn input(&self) -> Vec<u8> {
        postcard::to_stdvec(&self.0).unwrap()
    }
}
