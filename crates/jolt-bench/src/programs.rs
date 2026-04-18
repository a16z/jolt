use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum Program {
    #[value(name = "sha3-ex")]
    Sha3Ex,
    #[value(name = "sha2-ex")]
    Sha2Ex,
    #[value(name = "sha2-chain")]
    Sha2Chain,
    #[value(name = "sha3-chain")]
    Sha3Chain,
    Fibonacci,
    Muldiv,
    Btreemap,
}

impl Program {
    pub fn guest_name(self) -> &'static str {
        match self {
            Self::Sha3Ex => "sha3-guest",
            Self::Sha2Ex => "sha2-guest",
            Self::Sha2Chain => "sha2-chain-guest",
            Self::Sha3Chain => "sha3-chain-guest",
            Self::Fibonacci => "fib-guest",
            Self::Muldiv => "muldiv-guest",
            Self::Btreemap => "btreemap-guest",
        }
    }

    pub fn cli_name(self) -> &'static str {
        match self {
            Self::Sha3Ex => "sha3-ex",
            Self::Sha2Ex => "sha2-ex",
            Self::Sha2Chain => "sha2-chain",
            Self::Sha3Chain => "sha3-chain",
            Self::Fibonacci => "fibonacci",
            Self::Muldiv => "muldiv",
            Self::Btreemap => "btreemap",
        }
    }

    /// Canonical stdin bytes for this program, matching the postcard encoding
    /// the `#[jolt::provable]` macro emits for the example's `main.rs` inputs.
    /// Each public function arg is independently serialized via
    /// `postcard::to_stdvec(&arg)` and concatenated.
    ///
    /// `num_iters_override` applies to chain programs only (`sha2-chain`,
    /// `sha3-chain`). When `None`, the canonical default (1000) is used.
    pub fn canonical_inputs_with(self, num_iters_override: Option<u32>) -> Vec<u8> {
        match self {
            Self::Sha3Ex => {
                // sha3(input: &[u8])  →  example main uses b"Hello, world!"
                postcard::to_stdvec(&b"Hello, world!"[..]).expect("encode sha3 input")
            }
            Self::Sha2Ex => {
                // sha2(input: &[u8])  →  example main uses [5u8; 32]
                postcard::to_stdvec(&[5u8; 32][..]).expect("encode sha2 input")
            }
            Self::Fibonacci => {
                // fib(n: u32)  →  example main uses 50
                postcard::to_stdvec(&50u32).expect("encode fib input")
            }
            Self::Muldiv => {
                // muldiv(a: u32, b: u32, c: u32) — example main uses (12031293, 17, 92).
                // Each arg is independently postcard-encoded then concatenated
                // (matches `#[jolt::provable]` macro behaviour).
                let mut bytes = Vec::new();
                bytes.extend(postcard::to_stdvec(&12_031_293u32).expect("encode a"));
                bytes.extend(postcard::to_stdvec(&17u32).expect("encode b"));
                bytes.extend(postcard::to_stdvec(&92u32).expect("encode c"));
                bytes
            }
            Self::Btreemap => {
                // btreemap(n: u32)  →  example main uses 50
                postcard::to_stdvec(&50u32).expect("encode btreemap input")
            }
            Self::Sha2Chain => {
                // sha2_chain(input: [u8; 32], num_iters: u32). Default 1000;
                // override via --num-iters for bench scaling.
                let iters = num_iters_override.unwrap_or(1000);
                let mut bytes = Vec::new();
                bytes.extend(postcard::to_stdvec(&[5u8; 32]).expect("encode sha2-chain input"));
                bytes.extend(postcard::to_stdvec(&iters).expect("encode sha2-chain iters"));
                bytes
            }
            Self::Sha3Chain => {
                let iters = num_iters_override.unwrap_or(1000);
                let mut bytes = Vec::new();
                bytes.extend(postcard::to_stdvec(&[5u8; 32]).expect("encode sha3-chain input"));
                bytes.extend(postcard::to_stdvec(&iters).expect("encode sha3-chain iters"));
                bytes
            }
        }
    }
}

impl fmt::Display for Program {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.cli_name())
    }
}
