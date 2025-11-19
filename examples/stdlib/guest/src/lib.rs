use jolt::{jolt_print, jolt_println};

#[jolt::provable(memory_size = 32768, max_trace_length = 65536)]
fn int_to_string(n: i32) -> String {
    jolt_print!("Hello, from int_to_string!");
    n.to_string()
}

#[jolt::provable(memory_size = 32768, max_trace_length = 65536)]
fn string_concat(n: i32) -> String {
    jolt_println!("Hello, world!");
    let mut res = String::new();
    for i in 0..n {
        res += &i.to_string();
    }

    res
}
