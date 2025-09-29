use crate::field::JoltField;

pub fn drop_in_background_thread<T>(data: T)
where
    T: Send + 'static,
{
    // h/t https://abrams.cc/rust-dropping-things-in-another-thread
    rayon::spawn(move || drop(data));
}

pub fn unsafe_allocate_zero_vec<F: JoltField + Sized>(size: usize) -> Vec<F> {
    // https://stackoverflow.com/questions/59314686/how-to-efficiently-create-a-large-vector-of-items-initialized-to-the-same-value

    #[cfg(test)]
    {
        // Check for safety of 0 allocation
        unsafe {
            let value = &F::zero();
            let ptr = value as *const F as *const u8;
            let bytes = std::slice::from_raw_parts(ptr, std::mem::size_of::<F>());
            assert!(bytes.iter().all(|&byte| byte == 0));
        }
    }

    // Bulk allocate zeros, unsafely
    let result: Vec<F>;
    unsafe {
        let layout = std::alloc::Layout::array::<F>(size).unwrap();
        let ptr = std::alloc::alloc_zeroed(layout) as *mut F;

        if ptr.is_null() {
            panic!("Zero vec allocation failed");
        }

        result = Vec::from_raw_parts(ptr, size, size);
    }
    result
}
