#![cfg_attr(feature = "guest", no_std)]

extern crate alloc;

fn wyhash64(mut x: u64) -> u64 {
    x ^= x >> 32;
    x = x.wrapping_mul(0xd6e8feb86659fd93);
    x ^= x >> 32;
    x = x.wrapping_mul(0xd6e8feb86659fd93);
    x
}

#[jolt::provable(stack_size = 10000, heap_size = 10000000)]
pub fn btreemap(n: u32) -> u128 {
    use alloc::collections::BTreeMap;

    let mut map = BTreeMap::new();
    let mut inserted_keys = alloc::vec::Vec::with_capacity(n as usize);

    for i in 0..n {
        let key = wyhash64(i as u64);
        inserted_keys.push(key);
        map.insert(key, i as u64);
    }

    let delete_count = n / 4;
    for i in 0..delete_count {
        let key = inserted_keys[i as usize];
        map.remove(&key);
    }

    for i in 0..(n / 2) {
        let key = wyhash64((i + n * 2) as u64);
        map.insert(key, (i + n) as u64);
    }

    let mut range_sum = 0u64;
    if let Some((&min_key, _)) = map.first_key_value() {
        if let Some((&max_key, _)) = map.last_key_value() {
            let range_size = (max_key - min_key) / 4;
            let start = min_key + range_size;
            let end = start + range_size;

            for (_, value) in map.range(start..end) {
                range_sum = range_sum.wrapping_add(*value);
            }
        }
    }

    (map.len() as u128).wrapping_add(range_sum as u128)
}
