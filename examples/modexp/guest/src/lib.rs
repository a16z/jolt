#![cfg_attr(feature = "guest", no_std)]

#[jolt::provable(memory_size = 10240, max_trace_length = 65536)]
fn modexp_chain(base: u128, exp: u128, modulus: u128, num_iters: u32) -> u128 {
    // Multiply modulo m without overflowing using shift-add multiplication
    fn mul_mod(mut a: u128, mut b: u128, m: u128) -> u128 {
        let mut res: u128 = 0;
        a %= m;

        while b > 0 {
            if (b & 1) == 1 {
                res = (res + a) % m;
            }
            a = (a << 1) % m;
            b >>= 1;
        }

        res
    }

    // Compute modular exponentiation: base^exp mod m
    fn mod_pow(mut base: u128, mut exp: u128, m: u128) -> u128 {
        let mut result: u128 = 1 % m;
        base %= m;

        while exp > 0 {
            if (exp & 1) == 1 {
                result = mul_mod(result, base, m);
            }
            base = mul_mod(base, base, m);
            exp >>= 1;
        }

        result
    }

    // Proper chained computation: output of each iteration feeds into the next
    let mut out: u128 = base;

    for _ in 0..num_iters {
        out = mod_pow(out, exp, modulus);
    }

    out
}
