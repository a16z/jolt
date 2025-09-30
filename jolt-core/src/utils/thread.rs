use num_traits::Zero;

pub fn drop_in_background_thread<T>(data: T)
where
    T: Send + 'static,
{
    // h/t https://abrams.cc/rust-dropping-things-in-another-thread
    rayon::spawn(move || drop(data));
}

pub fn unsafe_allocate_zero_vec<T: Sized + Zero>(size: usize) -> Vec<T> {
    // https://stackoverflow.com/questions/59314686/how-to-efficiently-create-a-large-vector-of-items-initialized-to-the-same-value

    #[cfg(test)]
    {
        // Check for safety of 0 allocation
        unsafe {
            let value = &T::zero();
            let ptr = value as *const T as *const u8;
            let bytes = std::slice::from_raw_parts(ptr, std::mem::size_of::<T>());
            assert!(bytes.iter().all(|&byte| byte == 0));
        }
    }

    // Bulk allocate zeros, unsafely
    let result: Vec<T>;
    unsafe {
        let layout = std::alloc::Layout::array::<T>(size).unwrap();
        let ptr = std::alloc::alloc_zeroed(layout) as *mut T;

        if ptr.is_null() {
            panic!("Zero vec allocation failed");
        }

        result = Vec::from_raw_parts(ptr, size, size);
    }
    result
}
