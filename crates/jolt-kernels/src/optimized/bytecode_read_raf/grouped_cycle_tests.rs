use super::*;
use jolt_field::Fr;
#[cfg(feature = "akita")]
use jolt_field::Prime128Offset6A23801F;
use jolt_poly::BindingOrder;
#[cfg(feature = "akita")]
use jolt_witness::witnesses::FusedInc;
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

fn check<F: JoltField>(num_ra: usize, long_period: bool) {
    let log_n = if long_period { 17 } else { 16 };
    let n = 1 << log_n;
    let period = if long_period { 2064 } else { 31 };
    let mut rng = ChaCha20Rng::seed_from_u64(2187);
    let pcs: Vec<usize> = (0..n)
        .map(|j| if j < n - 17 { (j % period) * 7 } else { 0 })
        .collect();
    let rows = Arc::new(ConstantSuffix::from(
        pcs.iter()
            .map(|pc| {
                InstructionCycleRow::new(
                    0,
                    None,
                    false,
                    *pc,
                    None,
                    #[cfg(feature = "akita")]
                    FusedInc::default(),
                )
            })
            .collect::<Vec<_>>(),
    ));
    let tables: Vec<Vec<F>> = (0..num_ra)
        .map(|_| (0..16).map(|_| F::random(&mut rng)).collect())
        .collect();
    // The oracle constructs the original dense tables directly from PC bits.
    let mut dense: Vec<Polynomial<F>> = (0..num_ra)
        .map(|column| {
            Polynomial::new(
                pcs.iter()
                    .map(|pc| tables[column][(pc >> (4 * (num_ra - 1 - column))) & 15])
                    .collect(),
            )
        })
        .collect();
    let source = BytecodePcChunks {
        prefix_len: rows.prefix_len().max(1),
        rows,
        selectors: (0..num_ra)
            .map(|i| RaChunkSelector::new(i, num_ra, 4).unwrap())
            .collect(),
    };
    let plan = source.grouping_plan(n / 2, 1).unwrap();
    let mut cache = RepeatCache::new(plan.entries);
    let mut collision = false;
    for y in 0..period {
        let entry = cache.entry(source.group_hash(y, 1));
        collision |= entry.is_some_and(|old| !source.same_group(old, y, 1));
        *entry = Some(y);
    }
    assert!(
        collision,
        "fixture must exercise a full-key collision check"
    );
    let mut combined = Polynomial::new((0..n).map(|_| F::random(&mut rng)).collect());
    #[cfg(feature = "akita")]
    let mut fused_inc = Polynomial::new((0..n).map(|_| F::random(&mut rng)).collect());
    #[cfg(feature = "akita")]
    let mut fused_combined = Polynomial::new((0..n).map(|_| F::random(&mut rng)).collect());
    let degree = num_ra + if cfg!(feature = "akita") { 2 } else { 1 };
    let mut kernel = CycleKernel {
        progress: RoundProgress::new(log_n),
        degree,
        ra: CycleRa::Lazy(LazyFoldedRa::new(tables, source)),
        combined: EqPrefix::from_evals(combined.evals().to_vec()),
        #[cfg(feature = "akita")]
        fused_inc: LazyFusedInc::Prefix {
            values: fused_inc.evals().to_vec(),
            tail: F::zero(),
        },
        #[cfg(feature = "akita")]
        fused_combined: EqPrefix::from_evals(fused_combined.evals().to_vec()),
        output_openings: Vec::new(),
    };
    for round in 0..=4 {
        let groups = combined.len() / 2;
        let grouped = kernel.grouped_evals(groups);
        assert_eq!(grouped.is_some(), round < if long_period { 3 } else { 4 });
        let product = kernel.four_ra_evals(groups);
        assert_eq!(product.is_some(), num_ra == 4);
        let at = |poly: &Polynomial<F>, y: usize, t: F| {
            let (lo, hi) = pair(poly, y);
            lo + t * (hi - lo)
        };
        for slot in 0..degree {
            let t = F::from_u64(if slot == 0 { 0 } else { slot as u64 + 1 });
            let expected: F = (0..groups)
                .map(|y| {
                    let coefficient = at(&combined, y, t);
                    #[cfg(feature = "akita")]
                    let coefficient =
                        coefficient + at(&fused_inc, y, t) * at(&fused_combined, y, t);
                    dense
                        .iter()
                        .fold(coefficient, |acc, poly| acc * at(poly, y, t))
                })
                .sum();
            if let Some(actual) = &grouped {
                assert_eq!(actual[slot], expected, "grouped round {round}, slot {slot}");
            }
            if let Some(actual) = &product {
                assert_eq!(actual[slot], expected, "product round {round}, slot {slot}");
            }
        }
        let challenge = match round {
            0 => F::zero(),
            1 => F::one(),
            _ => F::random(&mut rng),
        };
        kernel.bind(challenge).unwrap();
        combined.bind_with_order(challenge, BindingOrder::LowToHigh);
        for poly in &mut dense {
            poly.bind_with_order(challenge, BindingOrder::LowToHigh);
        }
        #[cfg(feature = "akita")]
        {
            fused_inc.bind_with_order(challenge, BindingOrder::LowToHigh);
            fused_combined.bind_with_order(challenge, BindingOrder::LowToHigh);
        }
    }
}

#[test]
fn long_period_pc_groups_preserve_independent_coefficients() {
    check::<Fr>(4, true);
    #[cfg(feature = "akita")]
    check::<Prime128Offset6A23801F>(4, true);
}

#[test]
fn repeated_pc_groups_preserve_independent_coefficients() {
    for num_ra in [2, 4] {
        check::<Fr>(num_ra, false);
        #[cfg(feature = "akita")]
        check::<Prime128Offset6A23801F>(num_ra, false);
    }
}
