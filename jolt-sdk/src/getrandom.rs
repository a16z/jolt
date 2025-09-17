mod syscalls {
    static mut SEED: u64 = 0x123456789ABCDEF0;
    #[no_mangle]
    pub unsafe fn sys_rand(dest: *mut u8, len: usize) {
        for i in 0..len {
            SEED = SEED.wrapping_mul(1103515245).wrapping_add(12345);
            *dest.add(i) = (SEED >> 24) as u8;
        }
    }
}

pub fn _getrandom_v02(s: &mut [u8]) -> Result<(), getrandom_v02::Error> {
    unsafe {
        syscalls::sys_rand(s.as_mut_ptr(), s.len());
    }

    Ok(())
}

getrandom_v02::register_custom_getrandom!(_getrandom_v02);

#[no_mangle]
unsafe extern "Rust" fn __getrandom_v03_custom(
    dest: *mut u8,
    len: usize,
) -> Result<(), getrandom_v03::Error> {
    unsafe {
        syscalls::sys_rand(dest, len);
    }

    Ok(())
}
