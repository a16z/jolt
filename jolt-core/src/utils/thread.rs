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

#[tracing::instrument(skip_all)]
pub fn unsafe_zero_slice<F: JoltField + Sized>(slice: &mut [F]) {
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

    // Zero out existing slice memory
    unsafe {
        std::ptr::write_bytes(slice.as_mut_ptr(), 0, slice.len());
    }
}

/// Fast allocation of zero vector of Unreduced field elements
#[inline]
pub fn unsafe_allocate_zero_vec_unreduced<F: crate::field::JoltField, const N: usize>(
    size: usize,
) -> Vec<F::Unreduced<N>> {
    // https://stackoverflow.com/questions/59314686/how-to-efficiently-create-a-large-vector-of-items-initialized-to-the-same-value

    #[cfg(test)]
    {
        // Check for safety of 0 allocation
        unsafe {
            let value = &F::Unreduced::<N>::default();
            let ptr = value as *const F::Unreduced<N> as *const u8;
            let bytes = std::slice::from_raw_parts(ptr, std::mem::size_of::<F::Unreduced<N>>());
            assert!(bytes.iter().all(|&byte| byte == 0));
        }
    }

    let mut vector = Vec::with_capacity(size);
    unsafe {
        let ptr = vector.as_mut_ptr();
        std::ptr::write_bytes(ptr, 0, size);
        vector.set_len(size);
    }
    vector
}

/// Fast allocation of zero vector of BigInt
/// TODO: refactor so that both this and unsafe_allocate_zero_vec are one function
#[inline]
pub fn unsafe_allocate_zero_vec_bigint<const N: usize>(size: usize) -> Vec<ark_ff::BigInt<N>> {
    // https://stackoverflow.com/questions/59314686/how-to-efficiently-create-a-large-vector-of-items-initialized-to-the-same-value

    #[cfg(test)]
    {
        // Check for safety of 0 allocation
        unsafe {
            let value = &ark_ff::BigInt::zero();
            let ptr = value as *const ark_ff::BigInt<N> as *const u8;
            let bytes = std::slice::from_raw_parts(ptr, std::mem::size_of::<ark_ff::BigInt<N>>());
            assert!(bytes.iter().all(|&byte| byte == 0));
        }
    }

    // Bulk allocate zeros, unsafely
    let result: Vec<ark_ff::BigInt<N>>;
    unsafe {
        let layout = std::alloc::Layout::array::<ark_ff::BigInt<N>>(size).unwrap();
        let ptr = std::alloc::alloc_zeroed(layout) as *mut ark_ff::BigInt<N>;

        if ptr.is_null() {
            panic!("Zero vec allocation failed");
        }

        result = Vec::from_raw_parts(ptr, size, size);
    }
    result
}
