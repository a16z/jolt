//! Algebraic field law tests for `SymbolicField` constant folding.
//!
//! Verifies that the constant-folding path satisfies all field axioms. These
//! catch bugs in `scalar_ops` that would silently produce wrong constraints.

use jolt_field::Field;
use jolt_wrapper::arena::ArenaSession;
use jolt_wrapper::scalar_ops;
use jolt_wrapper::symbolic::SymbolicField;
use num_traits::{One, Zero};

use rand::rngs::StdRng;
use rand::SeedableRng;

fn random_const(rng: &mut StdRng) -> SymbolicField {
    let limbs = random_limbs(rng);
    SymbolicField::constant(limbs)
}

fn random_limbs(rng: &mut StdRng) -> [u64; 4] {
    let fr = jolt_field::Fr::random(rng);
    let bytes = fr.to_bytes();
    scalar_ops::from_bytes_le(&bytes)
}

const N: usize = 100;

#[test]
fn add_commutativity() {
    let _s = ArenaSession::new();
    let mut rng = StdRng::seed_from_u64(1);
    for _ in 0..N {
        let a = random_const(&mut rng);
        let b = random_const(&mut rng);
        assert_eq!((a + b).as_constant(), (b + a).as_constant());
    }
}

#[test]
fn add_associativity() {
    let _s = ArenaSession::new();
    let mut rng = StdRng::seed_from_u64(2);
    for _ in 0..N {
        let a = random_const(&mut rng);
        let b = random_const(&mut rng);
        let c = random_const(&mut rng);
        assert_eq!(((a + b) + c).as_constant(), (a + (b + c)).as_constant());
    }
}

#[test]
fn add_identity() {
    let _s = ArenaSession::new();
    let mut rng = StdRng::seed_from_u64(3);
    let zero = SymbolicField::zero();
    for _ in 0..N {
        let a = random_const(&mut rng);
        assert_eq!((a + zero).as_constant(), a.as_constant());
        assert_eq!((zero + a).as_constant(), a.as_constant());
    }
}

#[test]
fn add_inverse() {
    let _s = ArenaSession::new();
    let mut rng = StdRng::seed_from_u64(4);
    for _ in 0..N {
        let a = random_const(&mut rng);
        let neg_a = -a;
        let sum = a + neg_a;
        assert!(sum.is_zero(), "a + (-a) should be zero");
    }
}

#[test]
fn mul_commutativity() {
    let _s = ArenaSession::new();
    let mut rng = StdRng::seed_from_u64(5);
    for _ in 0..N {
        let a = random_const(&mut rng);
        let b = random_const(&mut rng);
        assert_eq!((a * b).as_constant(), (b * a).as_constant());
    }
}

#[test]
fn mul_associativity() {
    let _s = ArenaSession::new();
    let mut rng = StdRng::seed_from_u64(6);
    for _ in 0..N {
        let a = random_const(&mut rng);
        let b = random_const(&mut rng);
        let c = random_const(&mut rng);
        assert_eq!(((a * b) * c).as_constant(), (a * (b * c)).as_constant());
    }
}

#[test]
fn mul_identity() {
    let _s = ArenaSession::new();
    let mut rng = StdRng::seed_from_u64(7);
    let one = SymbolicField::one();
    for _ in 0..N {
        let a = random_const(&mut rng);
        assert_eq!((a * one).as_constant(), a.as_constant());
        assert_eq!((one * a).as_constant(), a.as_constant());
    }
}

#[test]
fn mul_inverse() {
    let _s = ArenaSession::new();
    let mut rng = StdRng::seed_from_u64(8);
    for _ in 0..N {
        let a = random_const(&mut rng);
        if a.is_zero() {
            continue;
        }
        let a_inv = a.inverse().unwrap();
        let product = a * a_inv;
        assert!(product.is_one(), "a * a^-1 should be one");
    }
}

#[test]
fn mul_zero_annihilates() {
    let _s = ArenaSession::new();
    let mut rng = StdRng::seed_from_u64(9);
    let zero = SymbolicField::zero();
    for _ in 0..N {
        let a = random_const(&mut rng);
        assert!((a * zero).is_zero());
        assert!((zero * a).is_zero());
    }
}

#[test]
fn distributivity() {
    let _s = ArenaSession::new();
    let mut rng = StdRng::seed_from_u64(10);
    for _ in 0..N {
        let a = random_const(&mut rng);
        let b = random_const(&mut rng);
        let c = random_const(&mut rng);
        // a * (b + c) == a*b + a*c
        assert_eq!(
            (a * (b + c)).as_constant(),
            (a * b + a * c).as_constant(),
            "left distributivity failed"
        );
        // (a + b) * c == a*c + b*c
        assert_eq!(
            ((a + b) * c).as_constant(),
            (a * c + b * c).as_constant(),
            "right distributivity failed"
        );
    }
}

#[test]
fn sub_is_add_neg() {
    let _s = ArenaSession::new();
    let mut rng = StdRng::seed_from_u64(11);
    for _ in 0..N {
        let a = random_const(&mut rng);
        let b = random_const(&mut rng);
        assert_eq!(
            (a - b).as_constant(),
            (a + (-b)).as_constant(),
            "a - b should equal a + (-b)"
        );
    }
}

#[test]
fn div_is_mul_inv() {
    let _s = ArenaSession::new();
    let mut rng = StdRng::seed_from_u64(12);
    for _ in 0..N {
        let a = random_const(&mut rng);
        let b = random_const(&mut rng);
        if b.is_zero() {
            continue;
        }
        let b_inv = b.inverse().unwrap();
        assert_eq!(
            (a / b).as_constant(),
            (a * b_inv).as_constant(),
            "a / b should equal a * b^-1"
        );
    }
}

#[test]
fn double_negation() {
    let _s = ArenaSession::new();
    let mut rng = StdRng::seed_from_u64(13);
    for _ in 0..N {
        let a = random_const(&mut rng);
        assert_eq!((-(-a)).as_constant(), a.as_constant());
    }
}

#[test]
fn square_is_self_mul() {
    let _s = ArenaSession::new();
    let mut rng = StdRng::seed_from_u64(14);
    for _ in 0..N {
        let a = random_const(&mut rng);
        assert_eq!(a.square().as_constant(), (a * a).as_constant());
    }
}

#[test]
fn from_to_u64_roundtrip() {
    let _s = ArenaSession::new();
    let values: Vec<u64> = vec![0, 1, 42, 1000, u32::MAX as u64];
    for v in values {
        let f = SymbolicField::from_u64(v);
        assert_eq!(f.to_u64(), Some(v), "roundtrip failed for {v}");
    }
}

#[test]
fn from_i64_consistency() {
    let _s = ArenaSession::new();
    // from_i64(n) + from_i64(-n) == 0 for all n != i64::MIN
    let values: Vec<i64> = vec![1, 42, 1000, i64::MAX];
    for v in values {
        let pos = SymbolicField::from_i64(v);
        let neg = SymbolicField::from_i64(-v);
        assert!(
            (pos + neg).is_zero(),
            "from_i64({v}) + from_i64(-{v}) should be 0"
        );
    }
}

#[test]
fn assign_operators() {
    let _s = ArenaSession::new();
    let mut rng = StdRng::seed_from_u64(15);
    for _ in 0..N {
        let a = random_const(&mut rng);
        let b = random_const(&mut rng);

        let mut add = a;
        add += b;
        assert_eq!(add.as_constant(), (a + b).as_constant());

        let mut sub = a;
        sub -= b;
        assert_eq!(sub.as_constant(), (a - b).as_constant());

        let mut mul = a;
        mul *= b;
        assert_eq!(mul.as_constant(), (a * b).as_constant());
    }
}

#[test]
fn zero_inverse_is_none() {
    let _s = ArenaSession::new();
    assert!(SymbolicField::zero().inverse().is_none());
}

#[test]
fn one_inverse_is_one() {
    let _s = ArenaSession::new();
    let one = SymbolicField::one();
    let inv = one.inverse().unwrap();
    assert!(inv.is_one());
}

#[test]
fn neg_zero_is_zero() {
    let _s = ArenaSession::new();
    assert!((-SymbolicField::zero()).is_zero());
}

#[test]
fn from_bool() {
    let _s = ArenaSession::new();
    assert!(SymbolicField::from_bool(false).is_zero());
    assert!(SymbolicField::from_bool(true).is_one());
}

#[test]
fn mul_pow_2() {
    let _s = ArenaSession::new();
    let a = SymbolicField::from_u64(1);
    let result = a.mul_pow_2(10);
    assert_eq!(result.to_u64(), Some(1024));
}
