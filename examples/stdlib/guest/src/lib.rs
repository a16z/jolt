use jolt::{jolt_print, jolt_println};

#[jolt::provable(max_trace_length = 65536, stack_size = 1048576)]
fn int_to_string(n: i32) -> i32 {
    jolt_println!("Hello, from int_to_string! n = {}", n);
    n
}

#[jolt::provable(max_trace_length = 65536, stack_size = 1048576)]
fn string_concat(n: i32) -> String {
    jolt_println!("Hello, world!");
    let mut res = String::new();
    for i in 0..n {
        res += &i.to_string();
    }

    res
}
