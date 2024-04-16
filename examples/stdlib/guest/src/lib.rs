#![no_main]

#[jolt::provable]
fn fun_with_strings(n: u32) -> u32 {
    let mut v = Vec::<String>::new();
    for i in 0..100 {
        v.push(i.to_string());
    }

    v[n as usize].parse().unwrap()
}
